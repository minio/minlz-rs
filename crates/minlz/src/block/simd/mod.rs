// Copyright 2026 MinIO Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! SIMD blocks (block type 1), described in `SPEC_SIMD.md` of the Go repository.
//!
//! Port of the shared part of Go's `decode-simd.go`. Go's `simd` name prefix is dropped
//! inside this module: `simdMaxLen` is [`MAX_LEN`], `simdLayouts` is [`LAYOUTS`].

mod decode;
#[cfg(test)]
mod decode_tests;
mod exec;
#[cfg(all(target_arch = "x86_64", not(miri)))]
mod exec_avx2;
#[cfg(all(target_arch = "x86_64", not(miri)))]
mod exec_avx512;
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
mod exec_neon;

pub(crate) use decode::{decode_simd, decode_simd_hash};

use super::format::get_uvarint;
use crate::Error;

mod encode;
mod parse_best;
mod parse_better;
mod parse_block;
mod parse_fast;

pub(crate) use encode::encode_block_simd;

/// Appends the size bytes that block, a SIMD block from its flags byte, decodes to, to dst.
///
/// The vector tiers decode into the spare capacity of dst: zeroing it first cost 2-15% of a
/// decode at 1-8 MB blocks. That relies on two properties of the decoder. On success it has written
/// all size bytes: the operations write the output contiguously from 0, and the literals after
/// them fill the rest. On error dst keeps its length, so nothing unwritten becomes visible. The
/// fast loops load bytes past the output position before they write them (matches with offsets
/// below their copy width, see the [memory rules](exec)); those bytes only reach output that later
/// operations overwrite. The vector loops do that through raw pointers; the Go loops through
/// slices, so the Go tier, the only one under Miri, gets a zeroed buffer.
///
/// Those loads, and the slice over unwritten bytes, are still undefined behavior in Rust's
/// abstract machine, which Miri can't check on the vector tiers: decoder code outside the vector
/// loops must only read output it has written. The decode runs on the tier checked here: picked
/// again, a [`cpu::disable`](crate::cpu::disable) on another thread could hand the unwritten
/// bytes to the Go loops.
// Inlined, it made the classic decoder in block::append_decoded 3-4% slower at 4 MB blocks.
#[inline(never)]
pub(crate) fn append_decoded(dst: &mut Vec<u8>, size: usize, block: &[u8]) -> Result<(), Error> {
    let start = dst.len();
    let tier = exec::Tier::current();
    if tier == exec::Tier::Go || !tier.usable() {
        dst.resize(start + size, 0);
        return decode_simd(&mut dst[start..], block).inspect_err(|_| dst.truncate(start));
    }
    dst.reserve(size);
    debug_assert!(dst.capacity() - start >= size);
    // SAFETY: reserve made size bytes of capacity from start, the decode on a vector tier writes
    // all of them before it returns Ok, and an error skips set_len. The slice over uninitialized
    // bytes breaks from_raw_parts_mut's contract (see above).
    unsafe {
        let out = core::slice::from_raw_parts_mut(dst.as_mut_ptr().add(start), size);
        decode::decode_tier(out, block, false, tier)?;
        dst.set_len(start + size);
    }
    Ok(())
}

pub(crate) const BLOCK_TYPE: u64 = 1;
pub(crate) const MIN_SAVING: usize = 64;
pub(crate) const MAX_CHUNK: usize = 1024 << 5;
/// The most chunks of a block.
pub(crate) const MAX_CHUNKS: usize = 128;
pub(crate) const MAX_LEN: usize = 32;
/// How far the fast loops write past the end of an operation.
pub(crate) const SLACK: usize = 32;
/// The number of readable bytes after decoded streams and literals.
pub(crate) const PAD: usize = 64;

// Streams, in chunk order, then the literals. DELTA is only a class layout, used by literals.
pub(crate) const TOK: usize = 0;
pub(crate) const LL: usize = 1;
pub(crate) const OFFC: usize = 2;
pub(crate) const ESC: usize = 3;
pub(crate) const LITS: usize = 4;
pub(crate) const DELTA: usize = 5;

// Entry modes.
pub(crate) const STORED: usize = 0;
pub(crate) const CURRENT: usize = 1;
pub(crate) const CODE_LENS: usize = 2;
pub(crate) const CLASS_LENS: usize = 3;
pub(crate) const DELTA_LENS: usize = 4;

/// The size of the code lengths table of each stream.
pub(crate) const CODE_BYTES: [usize; 5] = [68, 17, 88, 128, 128];

/// The base offset of each offset symbol, and the number of raw bits in the top byte.
/// Symbol 0 and symbols above 175 decode as offset 1.
pub(crate) static OFF_TAB: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut s = 0;
    while s < 256 {
        t[s] = if s == 0 || s > 175 {
            1
        } else if s < 16 {
            s as u32
        } else {
            let (b, m) = ((s + 16) >> 3, (s + 16) & 7);
            ((8 + m as u32) << (b - 3)) | ((b as u32 - 3) << 24)
        };
        s += 1;
    }
    t
};

/// A class layout. Values outside the stream's range have class 0xff.
pub(crate) struct Layout {
    pub(crate) class: [u8; 256],
    /// The in-class code length of each value.
    pub(crate) inlen: [u8; 256],
    /// The number of classes.
    pub(crate) n: usize,
}

/// The class layouts, indexed by stream. For tokens, the classes are the match length code groups.
pub(crate) static LAYOUTS: [Layout; 6] = [
    new_layout(TOK, 0, 33),
    new_layout(LL, 1, 33),
    new_layout(OFFC, 1, 175),
    new_layout(ESC, 0, 255),
    new_layout(LITS, 0, 255),
    new_layout(DELTA, 0, 255),
];

const fn bits_len(x: usize) -> usize {
    (usize::BITS - x.leading_zeros()) as usize
}

/// Returns the class of a value, given the last value of each class.
const fn class_bounds(last: &[usize], v: usize) -> usize {
    let mut c = 0;
    while last[c] < v {
        c += 1;
    }
    c
}

const fn lit_class(b: u8) -> usize {
    if b == 0 {
        0
    } else if b == b'\t' || b == b'\n' || b == b'\r' {
        1
    } else if b < b' ' || b == 0x7f {
        2
    } else if b == b' ' {
        3
    } else if b.is_ascii_digit() {
        4
    } else if matches!(b, b'a' | b'e' | b'i' | b'n' | b'o' | b'r' | b's' | b't') {
        5
    } else if b.is_ascii_lowercase() {
        6
    } else if b.is_ascii_uppercase() {
        7
    } else if matches!(b, b'"' | b',' | b'-' | b'.' | b'/' | b':' | b'_') {
        8
    } else if b < 0x7f {
        9
    } else if b < 0xc0 {
        10
    } else if b >= 0xc2 && b <= 0xf4 {
        11
    } else {
        12
    }
}

const fn class_of(k: usize, v: usize) -> usize {
    match k {
        TOK => class_bounds(&[0, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 31, 32, 33], v),
        LL => class_bounds(&[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 33], v),
        OFFC if v < 16 => bits_len(v) - 1,
        OFFC => (v + 16) >> 3,
        ESC => class_bounds(
            &[
                0, 1, 2, 4, 6, 10, 14, 22, 30, 46, 62, 94, 126, 190, 254, 255,
            ],
            v,
        ),
        LITS => lit_class(v as u8),
        _ if v == 128 => 8,
        _ => bits_len(if v < 256 - v { v } else { 256 - v }),
    }
}

/// Returns the layout of values lo to hi of stream k. Inside a class, values are in ascending
/// order and use a complete code of k or k+1 bits.
const fn new_layout(k: usize, lo: usize, hi: usize) -> Layout {
    let mut l = Layout {
        class: [0xff; 256],
        inlen: [0; 256],
        n: 0,
    };
    let (mut cnt, mut idx) = ([0usize; 24], [0usize; 24]);
    let mut v = lo;
    while v <= hi {
        let c = class_of(k, v);
        l.class[v] = c as u8;
        cnt[c] += 1;
        if c + 1 > l.n {
            l.n = c + 1;
        }
        v += 1;
    }
    v = lo;
    while v <= hi {
        let c = l.class[v] as usize;
        let k = bits_len(cnt[c]) - 1;
        l.inlen[v] = k as u8;
        if idx[c] >= (1 << (k + 1)) - cnt[c] {
            l.inlen[v] += 1;
        }
        idx[c] += 1;
        v += 1;
    }
    l
}

/// Reads the reverse varint that ends at the end of `b`.
/// Returns the value and its size, or a size of 0 if it is invalid.
#[inline]
pub(crate) fn rvarint(b: &[u8]) -> (u64, usize) {
    const MAX_VARINT_LEN64: usize = 10;
    let (mut x, mut s) = (0u64, 0u32);
    for i in 0..b.len().min(MAX_VARINT_LEN64) {
        let c = b[b.len() - 1 - i];
        if c < 0x80 {
            if i == MAX_VARINT_LEN64 - 1 && c > 1 {
                return (0, 0);
            }
            return (x | (c as u64) << s, i + 1);
        }
        x |= ((c & 0x7f) as u64) << s;
        s += 7;
    }
    (0, 0)
}

/// Parses the header of a SIMD block without its indicator byte.
/// Returns the decoded size and the block from its flags byte.
pub(crate) fn simd_header(src: &[u8]) -> Result<(usize, &[u8]), Error> {
    let (v, n) = match get_uvarint(src) {
        Some((v, n)) if n < src.len() && v <= u32::MAX as u64 => (v, n),
        _ => return Err(Error::Corrupt("bad SIMD block header")),
    };
    if v >> 24 != BLOCK_TYPE {
        return Err(Error::Unsupported);
    }
    let size = (v & ((1 << 24) - 1)) as usize;
    if size > super::MAX_BLOCK_SIZE {
        return Err(Error::TooLarge);
    }
    let f = src[n];
    if src.len() + MIN_SAVING > size || f & 0xe0 != 0 || f & 7 > 5 {
        return Err(Error::Corrupt("bad SIMD block flags or size"));
    }
    Ok((size, &src[n..]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_sizes() {
        // Class counts behind SPEC_SIMD.md 4.2's class table sizes: 14 token groups (+4+4 flags), 11, 24, 16, 13, 9.
        let n: Vec<usize> = LAYOUTS.iter().map(|l| l.n).collect();
        assert_eq!(n, [14, 11, 24, 16, 13, 9]);
        assert_eq!(OFF_TAB[16], 16 | 1 << 24);
        assert_eq!(OFF_TAB[24], 32 | 2 << 24);
        assert_eq!(OFF_TAB[175] & 0xffffff, 15_728_640);
    }

    #[test]
    fn rvarint_spec_examples() {
        assert_eq!(rvarint(&[0x05]), (5, 1));
        assert_eq!(rvarint(&[0x02, 0xac]), (300, 2));
        assert_eq!(rvarint(&[0x06, 0x8d, 0xa0]), (100000, 3));
        assert_eq!(rvarint(&[0x80]), (0, 0));
    }

    /// append_decoded writes every byte of the spare capacity it decodes into, on every tier,
    /// and leaves dst as it was on error.
    #[test]
    #[cfg_attr(miri, ignore = "encodes and decodes 200 KiB at 7 levels")]
    fn append_decoded_fills_spare_capacity() {
        use crate::block::{Level, encode};
        // Short repeats, overlapping matches and delta-friendly counters, over and under 64 KiB.
        let mut src = Vec::new();
        let mut x = 1u32;
        while src.len() < 200_000 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = (x >> 24) as usize % 48 + 1;
            match x % 4 {
                0 if src.len() > 64 => {
                    let o = (x >> 8) as usize % 40 + 1;
                    (0..n).for_each(|_| src.push(src[src.len() - o]));
                }
                1 => src.extend((0..n as u32).flat_map(|i| (i * 7 + x).to_le_bytes())),
                _ => src.extend((0..n).map(|i| b"etaoin shrdlu"[(i + n) % 13])),
            }
        }
        for level in [
            Level::SuperFastSimd,
            Level::FastestSimd,
            Level::BalancedSimd,
            Level::SmallestSimd,
            Level::SuperFastSimdDelta,
            Level::FastestSimdDelta,
            Level::BalancedSimdDelta,
        ] {
            for src in [&src[..], &src[..30_000]] {
                let mut block = Vec::new();
                encode(&mut block, src, level).unwrap();
                let typ = get_uvarint(&block[1..]).unwrap().0 >> 24;
                assert_eq!(typ, BLOCK_TYPE, "{level:?}: not a SIMD block");
                let mut bad = block.clone();
                bad.remove(bad.len() / 2);
                exec::simd_tiers(|tier| {
                    let mut dst = vec![0x5a; 6 + src.len() + 64];
                    dst.truncate(6);
                    crate::block::append_decoded(&mut dst, &block).unwrap();
                    assert!(dst[6..] == *src, "{level:?} {tier:?}");
                    let mut dst = vec![0x5a; 6 + src.len()];
                    dst.truncate(6);
                    assert!(crate::block::append_decoded(&mut dst, &bad).is_err());
                    assert_eq!(dst, [0x5a; 6], "{level:?} {tier:?}");
                });
            }
        }
    }
}
