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

//! MinLZ block codec — public API.
//!
//! Block format reference: [`SPEC.md`] §1–§2 and [`SPEC_SIMD.md`] in the upstream Go
//! repository. [`decode`](fn@decode) and [`append_decoded`] read classic and SIMD blocks.
//!
//! # SIMD blocks
//!
//! SIMD blocks are a block type added in v1.1 of the format. They store literals and operations
//! in separate Huffman-coded streams, laid out so they can be decoded with SIMD. The `*Simd`
//! [`Level`]s write them:
//!
//! ```
//! use minlz::{Level, decode, encode};
//!
//! let src = b"SIMD blocks decode with AVX-512, AVX2 or NEON. ".repeat(1000);
//! let mut block = Vec::new();
//! encode(&mut block, &src, Level::FastestSimd)?;
//! let mut out = Vec::new();
//! decode(&mut out, &block)?;
//! assert_eq!(out, src);
//! # Ok::<(), minlz::Error>(())
//! ```
//!
//! * Each block carries its own Huffman tables, so the SIMD levels pay off for blocks of 64 KiB
//!   and more, such as stream blocks at the default 2 MiB. On small blocks the classic levels
//!   compress faster, and usually decompress faster too.
//! * The delta levels ([`Level::SuperFastSimdDelta`], [`Level::FastestSimdDelta`],
//!   [`Level::BalancedSimdDelta`]) store literals as the difference to the bytes at the last match
//!   offset. That mainly helps binary data, like tables of numbers, but makes text larger and
//!   costs decompression speed. [`Level::SmallestSimd`] decides this itself.
//! * Decoders for format v1.0 cannot read SIMD blocks. Use the classic levels for data that every
//!   MinLZ decoder must read.
//! * The decoder picks its tier once: AVX-512 on x86-64 CPUs with AVX-512 (F, BW, VL, VBMI,
//!   VBMI2), POPCNT, AVX2 and BMI2; AVX2 on CPUs with AVX2 and BMI2; NEON on little-endian
//!   aarch64. Other targets, and Miri, run portable code. The encoders' output does not depend on
//!   the tier, and is the same as the Go encoders'.
//!
//! [`SPEC.md`]: https://github.com/minio/minlz/blob/main/SPEC.md
//! [`SPEC_SIMD.md`]: https://github.com/minio/minlz/blob/main/SPEC_SIMD.md

mod decode;
mod emit;
mod encode_l0;
mod encode_l1;
mod encode_l2;
mod encode_l3;
mod format;
mod hash;
mod load_store;
mod match_len;
pub(crate) mod simd;
mod table_pool;

pub use format::{MAX_BLOCK_SIZE, max_encoded_len};

use crate::Error;

/// Compression level for [`encode`].
///
/// The levels map to the Go API's int levels (`LevelFastest = 1`, …) with [`i32::from`] and
/// [`Level::try_from`].
///
/// The SIMD levels write SIMD blocks, which need a decoder for MinLZ format specification v1.1
/// or later; streams store them as SIMD chunks (`SPEC.md` 4.5B). SIMD blocks typically compress
/// better than classic ones, and decompress faster on platforms with SIMD when blocks are 64 KiB
/// or more. See [SIMD blocks](crate::block#simd-blocks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// The fastest compression level. It takes significant shortcuts and usually compresses
    /// much worse; use it only if [`Level::Fastest`] is confirmed to be too slow.
    /// Maps to Go's `LevelSuperFast` (-1) / `encodeFastBlockGo`.
    SuperFast,
    /// Single-pass with a 15-bit (or 13-bit for ≤64 KiB inputs) hash table.
    /// Maps to Go's `LevelFastest` / `encodeBlockGo`.
    Fastest,
    /// Two-table encoder.  Maps to Go's `LevelBalanced` / `encodeBlockBetterGo`.
    Balanced,
    /// Multi-candidate, size-scored encoder.  Maps to Go's
    /// `LevelSmallest` / `encodeBlockBest`.
    Smallest,
    /// The fastest SIMD compression level.
    /// It takes shortcuts that cost compression; use it only if [`Level::FastestSimd`] is
    /// confirmed to be too slow.
    /// Go's `LevelSuperFastSIMD` (4).
    SuperFastSimd,
    /// Fast SIMD compression with a good ratio, the SIMD counterpart of [`Level::Fastest`].
    /// Go's `LevelFastestSIMD` (5).
    FastestSimd,
    /// A balance between compression and speed, the SIMD counterpart of [`Level::Balanced`].
    /// This is targeted to be approximately half the speed of [`Level::FastestSimd`].
    /// Go's `LevelBalancedSIMD` (6).
    BalancedSimd,
    /// Will attempt the best possible compression, the SIMD counterpart of [`Level::Smallest`].
    /// There is no speed target for this level. It always tries both plain and delta literals.
    /// Go's `LevelSmallestSIMD` (7).
    SmallestSimd,
    /// [`Level::SuperFastSimd`] with delta literals, which are mainly useful for binary encoded
    /// data and have a small decode penalty. Go's `LevelSuperFastSIMD | LevelSIMDDelta` (260).
    SuperFastSimdDelta,
    /// [`Level::FastestSimd`] with delta literals. Go's `LevelFastestSIMD | LevelSIMDDelta` (261).
    FastestSimdDelta,
    /// [`Level::BalancedSimd`] with delta literals. Go's `LevelBalancedSIMD | LevelSIMDDelta` (262).
    BalancedSimdDelta,
}

impl TryFrom<i32> for Level {
    type Error = Error;

    /// Convert from Go's int levels (`-1`, `1`-`7`, `260`-`262`); any other value
    /// is [`Error::InvalidLevel`].
    fn try_from(v: i32) -> Result<Self, Error> {
        match v {
            -1 => Ok(Level::SuperFast),
            1 => Ok(Level::Fastest),
            2 => Ok(Level::Balanced),
            3 => Ok(Level::Smallest),
            4 => Ok(Level::SuperFastSimd),
            5 => Ok(Level::FastestSimd),
            6 => Ok(Level::BalancedSimd),
            7 => Ok(Level::SmallestSimd),
            260 => Ok(Level::SuperFastSimdDelta),
            261 => Ok(Level::FastestSimdDelta),
            262 => Ok(Level::BalancedSimdDelta),
            _ => Err(Error::InvalidLevel),
        }
    }
}

impl From<Level> for i32 {
    fn from(l: Level) -> i32 {
        match l {
            Level::SuperFast => -1,
            Level::Fastest => 1,
            Level::Balanced => 2,
            Level::Smallest => 3,
            Level::SuperFastSimd => 4,
            Level::FastestSimd => 5,
            Level::BalancedSimd => 6,
            Level::SmallestSimd => 7,
            Level::SuperFastSimdDelta => 4 | 256,
            Level::FastestSimdDelta => 5 | 256,
            Level::BalancedSimdDelta => 6 | 256,
        }
    }
}

/// Encode `src` as a MinLZ block, *replacing* the contents of `dst`.
///
/// Equivalent to Go's `Encode(dst, src, level)`.  `dst` is cleared and then
/// filled with the encoded block.
pub fn encode(dst: &mut Vec<u8>, src: &[u8], level: Level) -> Result<(), Error> {
    dst.clear();
    append_encoded(dst, src, level)
}

/// Encode `src` as a MinLZ block, *appending* to `dst`.
///
/// Equivalent to Go's `AppendEncoded(dst, src, level)`.
pub fn append_encoded(dst: &mut Vec<u8>, src: &[u8], level: Level) -> Result<(), Error> {
    let max = max_encoded_len(src.len()).ok_or(Error::TooLarge)?;
    dst.reserve(max);
    let start = dst.len();
    // Worst case: a single literal block (header + raw bytes).
    if src.len() < format::MIN_NON_LITERAL_BLOCK_SIZE {
        encode_uncompressed(dst, src);
        return Ok(());
    }

    // Layout: [tag=0][varint(len)][compressed body].
    dst.push(0);
    format::put_uvarint(dst, src.len() as u64);
    let mut body_start = dst.len();
    // Reserve worst-case body length so the encoder can write via index.
    dst.resize(start + max, 0);

    let n = match level {
        Level::SuperFast => encode_l0::encode_block(&mut dst[body_start..], src),
        Level::Fastest => encode_l1::encode_block(&mut dst[body_start..], src),
        Level::Balanced => encode_l2::encode_block(&mut dst[body_start..], src),
        Level::Smallest => encode_l3::encode_block(&mut dst[body_start..], src),
        _ => {
            // A SIMD block has its own header after the indicator byte.
            body_start = start + 1;
            simd::encode_block_simd(&mut dst[body_start..], src, level.into())
        }
    };

    if n > 0 {
        dst.truncate(body_start + n);
        Ok(())
    } else {
        // Not compressible: fall back to an uncompressed block.
        dst.truncate(start);
        encode_uncompressed(dst, src);
        Ok(())
    }
}

/// Try to encode `src` at the given [`Level`]; on success, append the block to
/// `dst` and return `true`. Returns `false` (leaving `dst` unchanged) if the
/// input is too small or does not compress to fewer bytes than the input — the
/// give-up path, parity with Go `TryEncode` returning `nil`.
pub fn try_encode(dst: &mut Vec<u8>, src: &[u8], level: Level) -> Result<bool, Error> {
    let max = max_encoded_len(src.len()).ok_or(Error::TooLarge)?;
    if src.len() < format::MIN_NON_LITERAL_BLOCK_SIZE {
        return Ok(false);
    }
    dst.reserve(max);
    let start = dst.len();
    dst.push(0);
    format::put_uvarint(dst, src.len() as u64);
    let mut body_start = dst.len();
    dst.resize(start + max, 0);

    let n = match level {
        Level::SuperFast => encode_l0::encode_block(&mut dst[body_start..], src),
        Level::Fastest => encode_l1::encode_block(&mut dst[body_start..], src),
        Level::Balanced => encode_l2::encode_block(&mut dst[body_start..], src),
        Level::Smallest => encode_l3::encode_block(&mut dst[body_start..], src),
        _ => {
            body_start = start + 1;
            simd::encode_block_simd(&mut dst[body_start..], src, level.into())
        }
    };

    // Go's TryEncode also rejects "compressed >= src.len()" (see encode.go).
    if n > 0 && (body_start - start) + n < src.len() {
        dst.truncate(body_start + n);
        Ok(true)
    } else {
        dst.truncate(start);
        Ok(false)
    }
}

/// Decode a MinLZ block into `dst`.
///
/// On success `dst` holds the decoded bytes; on error it is empty.
/// Equivalent to Go's `Decode(dst, src)` (without the Snappy/S2 fallback).
///
/// As Go's `Decode` reuses `dst`, a SIMD block decodes over the bytes `dst` holds and zero-fills
/// only the rest of its output first, so clearing a reused `Vec` costs a zero-fill.
pub fn decode(dst: &mut Vec<u8>, src: &[u8]) -> Result<(), Error> {
    decode_at(dst, 0, src)
}

/// Append the decoded form of `src` to `dst`. On error `dst` is unchanged.
///
/// Equivalent to Go's `AppendDecoded(dst, src)`. A SIMD block zero-fills the spare capacity it
/// decodes into.
pub fn append_decoded(dst: &mut Vec<u8>, src: &[u8]) -> Result<(), Error> {
    decode_at(dst, dst.len(), src)
}

/// Decodes `src` into `dst` from `start`, keeping `dst[..start]`, all `dst` keeps on error.
fn decode_at(dst: &mut Vec<u8>, start: usize, src: &[u8]) -> Result<(), Error> {
    let (literals, simd, body, dlen) = parse_header(src).inspect_err(|_| dst.truncate(start))?;
    if simd {
        return simd::decode_at(dst, start, dlen, body);
    }
    dst.truncate(start);
    if literals {
        dst.extend_from_slice(body);
        return Ok(());
    }
    // The decoder uses LZ4-style 16-byte overshoot stores in its hot path;
    // give it `OVERSHOOT_PAD` bytes of writable padding past `dlen`.
    dst.reserve(dlen + decode::OVERSHOOT_PAD);
    // SAFETY: we just reserved enough room; `dst.as_mut_ptr().add(start)`
    // points to at least `dlen + OVERSHOOT_PAD` writable bytes.  The
    // decoder writes `dlen` valid bytes on success; the padding region's
    // contents are undefined and will not be exposed via the returned Vec.
    unsafe {
        let dst_ptr = dst.as_mut_ptr().add(start);
        decode::minlz_decode(dst_ptr, dlen, body)?;
        dst.set_len(start + dlen);
    }
    Ok(())
}

/// Returns the uncompressed length of a MinLZ block.
///
/// Equivalent to Go's `DecodedLen(src)`.
pub fn decoded_len(src: &[u8]) -> Result<usize, Error> {
    let (_, _, _, dlen) = parse_header(src)?;
    Ok(dlen)
}

/// Returns `Some(uncompressed_size)` if `src` looks like a MinLZ block (leading
/// 0 byte), or `None` if it does not (e.g. a Snappy/S2 stream, which is out of
/// scope). `Err` only on a malformed MinLZ header.
///
/// Replaces a bare `(bool, usize)`: the size is meaningful exactly when the
/// answer is "yes", which `Option` expresses directly. Full validation still
/// happens inside [`decode`].
pub fn is_minlz(src: &[u8]) -> Result<Option<usize>, Error> {
    if src.is_empty() {
        return Err(Error::Corrupt("empty input"));
    }
    if src[0] != 0 {
        // Snappy/S2 fallback is out of scope.
        return Ok(None);
    }
    let (_, _, _, dlen) = parse_header(src)?;
    Ok(Some(dlen))
}

/// Parse the block header.  Returns `(is_literal_only, is_simd, body_slice, dlen)`.
/// SIMD blocks are returned from their flags byte.
fn parse_header(src: &[u8]) -> Result<(bool, bool, &[u8], usize), Error> {
    if src.is_empty() {
        return Err(Error::Corrupt("empty input"));
    }
    if src.len() == 1 {
        // Single byte: must be 0 (0-byte block).
        if src[0] == 0 {
            return Ok((true, false, &src[1..], 0));
        }
        return Err(Error::Corrupt("invalid single-byte block"));
    }
    if src[0] != 0 {
        // Snappy/S2 fallback path — refused here.
        return Err(Error::Corrupt("snappy/s2 stream not supported"));
    }
    let after_magic = &src[1..];
    if let Some((v, _)) = format::get_uvarint(after_magic)
        && v >= 1 << 24
    {
        let (size, block) = simd::simd_header(after_magic)?;
        return Ok((false, true, block, size));
    }
    let (v, n) =
        format::get_uvarint(after_magic).ok_or(Error::Corrupt("bad decoded-length varint"))?;
    // Match Go decodedLen: anything that wouldn't fit in uint32 is corrupt …
    if v > 0xffff_ffff {
        return Err(Error::Corrupt("decoded length exceeds u32"));
    }
    // … and anything between uint32 and MaxBlockSize is "too large".
    if v > MAX_BLOCK_SIZE as u64 {
        return Err(Error::TooLarge);
    }
    let dlen = v as usize;
    let body = &after_magic[n..];
    // Order matches Go's isMinLZ: body-empty before the v==0 short-circuit.
    if body.is_empty() {
        return Err(Error::Corrupt("empty block body"));
    }
    if dlen == 0 {
        return Ok((true, false, body, body.len()));
    }
    if dlen < body.len() {
        // A compressed block may not be larger than the decompressed block.
        return Err(Error::Corrupt("compressed larger than decoded size"));
    }
    Ok((false, false, body, dlen))
}

fn encode_uncompressed(dst: &mut Vec<u8>, src: &[u8]) {
    if src.is_empty() {
        dst.push(0);
        return;
    }
    dst.push(0);
    dst.push(0);
    dst.extend_from_slice(src);
}

// ----- Stream-codec helpers ----------------------------------------------
//
// The standalone block format produced by [`encode`] / [`append_encoded`]
// has a leading `0` marker byte that distinguishes it from a Snappy or S2
// block.  When a MinLZ block is embedded in a stream chunk (types 0x02 or
// 0x03), the chunk type itself identifies it, so the leading marker is
// omitted.  The three helpers below speak that "chunk body" form.

/// Encode `src` as a MinLZ stream-chunk body to `dst[start..]` and return its
/// size: `varint(dlen) + body`, no leading `0` marker, for a 0x02 chunk, or a
/// SIMD block, whose header holds its type, for a 0x10 chunk at the SIMD
/// levels.  Returns 0 if compression isn't worthwhile (caller should emit a
/// 0x01 chunk).
///
/// `dst` grows to [`max_encoded_len`] bytes past `start` and keeps its length:
/// the bytes after the body are left from earlier blocks.  The encoders only
/// write their output, so a stream reuses `dst` for every block without
/// zeroing it again (Go's `Writer.encodeBlock` on its `obuf`).
pub(crate) fn encode_chunk_body(
    dst: &mut Vec<u8>,
    start: usize,
    src: &[u8],
    level: Level,
) -> Result<usize, Error> {
    let max = max_encoded_len(src.len()).ok_or(Error::TooLarge)?;
    if dst.len() < start + max {
        dst.resize(start + max, 0);
    }
    let dst = &mut dst[start..start + max];
    let encode_block = match level {
        Level::SuperFast => encode_l0::encode_block,
        Level::Fastest => encode_l1::encode_block,
        Level::Balanced => encode_l2::encode_block,
        Level::Smallest => encode_l3::encode_block,
        _ => return Ok(simd::encode_block_simd(dst, src, level.into())),
    };
    if src.len() < format::MIN_NON_LITERAL_BLOCK_SIZE {
        return Ok(0);
    }
    let h = format::put_uvarint_at(dst, src.len() as u64);
    let n = encode_block(&mut dst[h..], src);
    Ok(if n > 0 && h + n < src.len() { h + n } else { 0 })
}

/// Decode a MinLZ stream-chunk body (`varint(dlen) + body`).  `dst` is
/// appended with the decoded bytes.
pub(crate) fn append_decoded_chunk_body(dst: &mut Vec<u8>, body: &[u8]) -> Result<(), Error> {
    let (dlen, compressed) = split_chunk_body(body)?;
    let start = dst.len();
    dst.reserve(dlen + decode::OVERSHOOT_PAD);
    // SAFETY: same invariants as `decode_at` — we just reserved
    // `dlen + OVERSHOOT_PAD` writable bytes past `start`.
    unsafe {
        let dst_ptr = dst.as_mut_ptr().add(start);
        decode::minlz_decode(dst_ptr, dlen, compressed)?;
        dst.set_len(start + dlen);
    }
    Ok(())
}

/// Return the uncompressed length of a MinLZ stream-chunk body.
pub(crate) fn decoded_len_chunk_body(body: &[u8]) -> Result<usize, Error> {
    split_chunk_body(body).map(|(dlen, _)| dlen)
}

/// Split a chunk body into `dlen` and the compressed bytes.  Unlike a
/// standalone block, a chunk has no `dlen == 0` literal form (incompressible
/// data goes in a 0x01 chunk), and its compressed bytes never exceed `dlen`
/// (SPEC §4.4/§4.5; Go's reader rejects both).
fn split_chunk_body(body: &[u8]) -> Result<(usize, &[u8]), Error> {
    let (v, n) = format::get_uvarint(body).ok_or(Error::Corrupt("bad chunk length varint"))?;
    if v >> 24 != 0 {
        return Err(Error::Corrupt("chunk decoded length exceeds 24 bits"));
    }
    if v > MAX_BLOCK_SIZE as u64 {
        return Err(Error::TooLarge);
    }
    let (dlen, compressed) = (v as usize, &body[n..]);
    if dlen == 0 {
        return Err(Error::Corrupt("zero decoded length in chunk"));
    }
    if dlen < compressed.len() {
        return Err(Error::Corrupt("compressed larger than decoded size"));
    }
    Ok((dlen, compressed))
}

#[cfg(test)]
mod tests;
