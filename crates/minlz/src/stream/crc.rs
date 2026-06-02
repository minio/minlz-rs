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

//! CRC32C (Castagnoli) with the MinLZ/S2/Snappy mask.
//!
//! Stage C bundles a small table-based CRC32C and, when the target CPU
//! supports it at runtime, dispatches to the hardware `CRC32C` instruction
//! (SSE 4.2 on x86_64, the `crc` ISA extension on aarch64).  The fallback
//! is identical to the original table-driven implementation.
//!
//! The `mask_crc32c` wrapper is the bit that differs from upstream Snappy: it
//! rotates the raw CRC right by 15 and adds the Snappy mask constant.  Go's
//! `minlz.go:crc` expresses this with operator-precedence-dependent code
//! (`c>>15 | c<<17 + 0xa282ead8`) that, taken literally, parses *differently*
//! in Rust — see the inline comment on [`mask_crc32c`].
//!
//! Test vectors come from `minlz-rs/_helpers/crc_vectors/main.go`.

use std::sync::OnceLock;

const POLY: u32 = 0x82F6_3B78;

const TABLE: [u32; 256] = build_table();

const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut j = 0;
        while j < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            j += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

type CrcFn = fn(u32, &[u8]) -> u32;

static CRC_IMPL: OnceLock<CrcFn> = OnceLock::new();

#[inline]
fn dispatch() -> CrcFn {
    *CRC_IMPL.get_or_init(select_impl)
}

fn select_impl() -> CrcFn {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse4.2") {
            return crc32c_hw_x86_64_safe;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("crc") {
            return crc32c_hw_aarch64_safe;
        }
    }
    crc32c_update_table
}

/// Compute CRC32C of `data` starting from a CRC of 0.
#[inline]
pub(crate) fn crc32c(data: &[u8]) -> u32 {
    crc32c_update(0, data)
}

/// Continue an in-progress CRC32C with more bytes.
#[inline]
pub(crate) fn crc32c_update(crc: u32, data: &[u8]) -> u32 {
    dispatch()(crc, data)
}

/// Portable scalar CRC32C.  Slice-by-8 inner loop with byte-at-a-time
/// tail.  Selected when no hardware CRC32C is available (legacy x86
/// without SSE 4.2, ARMv7, MIPS, RISC-V, big-endian, …).
fn crc32c_update_table(crc: u32, data: &[u8]) -> u32 {
    let mut c = !crc;
    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        // `chunks_exact(8)` yields 8-byte slices, so the 4-byte sub-slices
        // are always exactly `[u8; 4]` — `try_into` is infallible here.
        let v0 = u32::from_le_bytes(chunk[0..4].try_into().unwrap()) ^ c;
        let v1 = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
        // Slice-by-8: process 8 bytes per iteration via 8 parallel tables.
        c = SLICE_TABLE[7][(v0 & 0xff) as usize]
            ^ SLICE_TABLE[6][((v0 >> 8) & 0xff) as usize]
            ^ SLICE_TABLE[5][((v0 >> 16) & 0xff) as usize]
            ^ SLICE_TABLE[4][((v0 >> 24) & 0xff) as usize]
            ^ SLICE_TABLE[3][(v1 & 0xff) as usize]
            ^ SLICE_TABLE[2][((v1 >> 8) & 0xff) as usize]
            ^ SLICE_TABLE[1][((v1 >> 16) & 0xff) as usize]
            ^ SLICE_TABLE[0][((v1 >> 24) & 0xff) as usize];
    }
    for &b in chunks.remainder() {
        c = TABLE[((c ^ u32::from(b)) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

/// Slice-by-8 tables (the 0th column is the same as [`TABLE`]).  Each
/// `SLICE_TABLE[k]` is the byte-level CRC32C lookup shifted by `k` bytes.
const SLICE_TABLE: [[u32; 256]; 8] = build_slice_table();

const fn build_slice_table() -> [[u32; 256]; 8] {
    let mut table = [[0u32; 256]; 8];
    // SLICE_TABLE[0] is the standard byte-level CRC32C step table.
    let mut i = 0;
    while i < 256 {
        table[0][i] = TABLE[i];
        i += 1;
    }
    // For k = 1..8, SLICE_TABLE[k][b] = SLICE_TABLE[k-1][b] shifted by one
    // more byte position through the CRC.
    let mut k = 1;
    while k < 8 {
        let mut b = 0;
        while b < 256 {
            let prev = table[k - 1][b];
            table[k][b] = (prev >> 8) ^ table[0][(prev & 0xff) as usize];
            b += 1;
        }
        k += 1;
    }
    table
}

// -------------------- x86_64 hardware path --------------------

#[cfg(target_arch = "x86_64")]
fn crc32c_hw_x86_64_safe(crc: u32, data: &[u8]) -> u32 {
    // SAFETY: only stored in CRC_IMPL after `is_x86_feature_detected!("sse4.2")`
    // returned true in `select_impl`.
    unsafe { crc32c_hw_x86_64(crc, data) }
}

/// Bytes per parallel stream in the 3-way pipelined inner loop.  Chunk
/// size = `3 * STRIDE_BYTES`.  Chosen so the three accumulators live in
/// registers and the 4×256×u32 shift table (4 KiB) stays in L1.
const STRIDE_BYTES: usize = 1024;

/// Precomputed `x^(8 * STRIDE_BYTES) mod G(x)` byte-multiplication table
/// for CRC32C.  Used to combine three independent CRC streams into one.
const SHIFT_TABLE: [[u32; 256]; 4] = build_shift_table::<{ STRIDE_BYTES }>();

const fn build_shift_table<const ZB: usize>() -> [[u32; 256]; 4] {
    let mut table = [[0u32; 256]; 4];
    let mut k = 0;
    while k < 4 {
        let mut b = 0;
        while b < 256 {
            // Start the CRC register with byte `b` at slot `k`, zero elsewhere.
            let mut c = (b as u32) << (k * 8);
            // Run `ZB` zero bytes through the standard byte-table CRC32C step.
            let mut j = 0;
            while j < ZB {
                c = TABLE[(c & 0xff) as usize] ^ (c >> 8);
                j += 1;
            }
            table[k][b] = c;
            b += 1;
        }
        k += 1;
    }
    table
}

/// `shift_crc(c) = c * x^(8 * STRIDE_BYTES) mod G(x)`.  Equivalent to
/// advancing the CRC by `STRIDE_BYTES` zero bytes.
#[inline(always)]
fn shift_crc(crc: u32) -> u32 {
    SHIFT_TABLE[0][(crc & 0xff) as usize]
        ^ SHIFT_TABLE[1][((crc >> 8) & 0xff) as usize]
        ^ SHIFT_TABLE[2][((crc >> 16) & 0xff) as usize]
        ^ SHIFT_TABLE[3][((crc >> 24) & 0xff) as usize]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_hw_x86_64(crc: u32, data: &[u8]) -> u32 {
    use core::arch::x86_64::*;
    // SAFETY: `target_feature(enable = "sse4.2")` + CPUID-guarded dispatch
    // in `select_impl` make the intrinsic calls below sound.
    //
    // For inputs ≥ 3 × STRIDE_BYTES, run three parallel `_mm_crc32_u64`
    // chains so the CPU can keep the 3-cycle-latency / 1-cycle-throughput
    // crc32q pipeline saturated.  At chunk boundaries, combine via
    // `shift_crc`.  Algorithm matches Go's `crc32_amd64.s::castagnoliSSE42Triple`.
    let mut c = !crc;
    let mut data = data;
    while data.len() >= 3 * STRIDE_BYTES {
        let s0 = data.as_ptr();
        let s1 = unsafe { s0.add(STRIDE_BYTES) };
        let s2 = unsafe { s0.add(2 * STRIDE_BYTES) };
        let mut c0 = u64::from(c);
        let mut c1 = 0u64;
        let mut c2 = 0u64;
        let mut i = 0;
        while i < STRIDE_BYTES {
            // SAFETY: stream pointers each cover STRIDE_BYTES contiguous
            // bytes of `data`; `i` runs in 8-byte steps to `STRIDE_BYTES`.
            let v0 = unsafe { (s0.add(i) as *const u64).read_unaligned() };
            let v1 = unsafe { (s1.add(i) as *const u64).read_unaligned() };
            let v2 = unsafe { (s2.add(i) as *const u64).read_unaligned() };
            c0 = _mm_crc32_u64(c0, v0);
            c1 = _mm_crc32_u64(c1, v1);
            c2 = _mm_crc32_u64(c2, v2);
            i += 8;
        }
        // Fold: c_AB = shift(c0, STRIDE) XOR c1; c_ABC = shift(c_AB, STRIDE) XOR c2.
        let c_ab = shift_crc(c0 as u32) ^ (c1 as u32);
        c = shift_crc(c_ab) ^ (c2 as u32);
        data = &data[3 * STRIDE_BYTES..];
    }
    // Tail: single-stream, 8 bytes at a time, then byte-at-a-time remainder.
    let mut c64 = u64::from(c);
    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        // `chunks_exact(8)` guarantees an 8-byte slice, so the conversion
        // to `[u8; 8]` is infallible.
        let v = u64::from_le_bytes(chunk.try_into().unwrap());
        c64 = _mm_crc32_u64(c64, v);
    }
    let mut c = c64 as u32;
    for &b in chunks.remainder() {
        c = _mm_crc32_u8(c, b);
    }
    !c
}

// -------------------- aarch64 hardware path --------------------

#[cfg(target_arch = "aarch64")]
fn crc32c_hw_aarch64_safe(crc: u32, data: &[u8]) -> u32 {
    // SAFETY: only stored after `is_aarch64_feature_detected!("crc")` matched.
    unsafe { crc32c_hw_aarch64(crc, data) }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "crc")]
unsafe fn crc32c_hw_aarch64(crc: u32, data: &[u8]) -> u32 {
    use core::arch::aarch64::*;
    // SAFETY: `target_feature(enable = "crc")` plus the CPUID-style runtime
    // check in `select_impl` make the intrinsics below sound.
    //
    // Three-way pipelined inner loop mirrors the x86_64 path.  Each
    // `__crc32cd` instruction has ~3-cycle latency / 1-cycle throughput
    // on production aarch64 cores (Apple, Graviton, …), so three
    // independent accumulators saturate the pipeline.
    let mut c = !crc;
    let mut data = data;
    while data.len() >= 3 * STRIDE_BYTES {
        let s0 = data.as_ptr();
        let s1 = unsafe { s0.add(STRIDE_BYTES) };
        let s2 = unsafe { s0.add(2 * STRIDE_BYTES) };
        let mut c0 = c;
        let mut c1 = 0u32;
        let mut c2 = 0u32;
        let mut i = 0;
        while i < STRIDE_BYTES {
            let v0 = unsafe { (s0.add(i) as *const u64).read_unaligned() };
            let v1 = unsafe { (s1.add(i) as *const u64).read_unaligned() };
            let v2 = unsafe { (s2.add(i) as *const u64).read_unaligned() };
            c0 = __crc32cd(c0, v0);
            c1 = __crc32cd(c1, v1);
            c2 = __crc32cd(c2, v2);
            i += 8;
        }
        c = shift_crc(shift_crc(c0) ^ c1) ^ c2;
        data = &data[3 * STRIDE_BYTES..];
    }
    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        // `chunks_exact(8)` guarantees an 8-byte slice, so the conversion
        // to `[u8; 8]` is infallible.
        let v = u64::from_le_bytes(chunk.try_into().unwrap());
        c = __crc32cd(c, v);
    }
    for &b in chunks.remainder() {
        c = __crc32cb(c, b);
    }
    !c
}

/// Apply the MinLZ/S2/Snappy mask to a raw CRC32C value.
///
/// Go's source reads `c>>15 | c<<17 + 0xa282ead8`.  In Go, `|` and `+` are
/// same-precedence and associate left-to-right, so it means
/// `((c>>15) | (c<<17)) + 0xa282ead8`.  In Rust `+` binds tighter than `|`,
/// so the literal translation is wrong; we use `rotate_right(15)` (equivalent
/// to `(c>>15) | (c<<17)`) and `wrapping_add` to match Go's two's-complement
/// 32-bit overflow behavior.
#[inline]
pub(crate) fn mask_crc32c(c: u32) -> u32 {
    c.rotate_right(15).wrapping_add(0xa282_ead8)
}

/// Compute the masked CRC32C of `data` (raw CRC + mask).
#[inline]
pub(crate) fn masked_crc32c(data: &[u8]) -> u32 {
    mask_crc32c(crc32c(data))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(input: &[u8], raw: u32, masked: u32) {
        assert_eq!(crc32c(input), raw, "raw mismatch for {input:?}");
        assert_eq!(mask_crc32c(raw), masked, "mask fn mismatch for {input:?}");
        assert_eq!(
            masked_crc32c(input),
            masked,
            "masked mismatch for {input:?}"
        );
        // Also exercise the table impl directly so a wrong hardware path
        // doesn't mask itself by silently agreeing with itself.
        assert_eq!(
            crc32c_update_table(0, input),
            raw,
            "table mismatch for {input:?}"
        );
    }

    #[test]
    fn vectors_match_go() {
        // Vectors from minlz-rs/_helpers/crc_vectors/main.go.
        check(b"", 0x0000_0000, 0xa282_ead8);
        check(b"a", 0xc1d0_4330, 0x28e4_6e78);
        check(b"ab", 0xe2a2_2936, 0xf4f0_b01c);
        check(b"abc", 0x364b_3fb7, 0x21f1_576e);
        check(b"123456789", 0xe306_9283, 0xc78a_b0e5);
        check(
            b"The quick brown fox jumps over the lazy dog",
            0x2262_0404,
            0xaa8b_2f9c,
        );
        check(&[0u8; 32], 0x8a91_36aa, 0x0fd7_fffa);
        check(&[0xffu8; 32], 0x62a8_ab43, 0xf909_b029);
        let inc: Vec<u8> = (0u32..1024).map(|i| i as u8).collect();
        check(&inc, 0x2cdf_6e8f, 0x7fa1_4496);
        check(b"MinLz\x0b", 0x068b_74c0, 0x8c02_f7ee);
    }

    #[test]
    fn empty_masked_is_constant() {
        // crc32c(empty) = 0; rotate_right(0, 15) = 0; 0 + mask = mask.
        assert_eq!(masked_crc32c(&[]), 0xa282_ead8);
    }

    #[test]
    fn update_is_associative() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let one_shot = crc32c(data);
        let (a, b) = data.split_at(11);
        let split = crc32c_update(crc32c_update(0, a), b);
        assert_eq!(one_shot, split);
    }

    /// Differential test: at every offset/length combination across an odd
    /// size, the hardware and table impls must agree.  Catches bugs in the
    /// 8-byte-chunk + tail handling.
    #[test]
    fn hw_matches_table_at_every_tail() {
        let data: Vec<u8> = (0..257u32).map(|i| (i * 31) as u8).collect();
        for end in 0..=data.len() {
            let slice = &data[..end];
            let hw = crc32c(slice);
            let tbl = crc32c_update_table(0, slice);
            assert_eq!(hw, tbl, "length {end}");
        }
    }

    /// Cover the 3-way pipelined path (input ≥ 3 × STRIDE_BYTES).  Tests
    /// many lengths around the 3-stride boundary so the combine math is
    /// exercised at every tail.
    #[test]
    fn hw_matches_table_in_3way_range() {
        // Long-enough PRNG-style buffer; reuse a simple LCG.
        let n = 7 * super::STRIDE_BYTES + 257;
        let mut data = vec![0u8; n];
        let mut x: u64 = 0xdead_beef_cafe_babe;
        for slot in &mut data {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *slot = (x >> 24) as u8;
        }
        // Check at every length crossing the 3-way boundary.
        for end in [
            super::STRIDE_BYTES * 3 - 1,
            super::STRIDE_BYTES * 3,
            super::STRIDE_BYTES * 3 + 1,
            super::STRIDE_BYTES * 3 + 7,
            super::STRIDE_BYTES * 3 + 8,
            super::STRIDE_BYTES * 3 + 17,
            super::STRIDE_BYTES * 6,
            super::STRIDE_BYTES * 6 + 31,
            n - 1,
            n,
        ] {
            assert_eq!(
                crc32c(&data[..end]),
                crc32c_update_table(0, &data[..end]),
                "len {end}"
            );
        }
        // Also sweep length 0..STRIDE_BYTES*3+50 to catch any off-by-one
        // around the boundary.
        for end in (super::STRIDE_BYTES * 3 - 50..=super::STRIDE_BYTES * 3 + 50).step_by(7) {
            assert_eq!(
                crc32c(&data[..end]),
                crc32c_update_table(0, &data[..end]),
                "sweep len {end}"
            );
        }
    }
}
