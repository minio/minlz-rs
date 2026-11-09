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

//! The AVX2 operation loops and stream stats of SIMD blocks: a `core::arch` port of the
//! assembly that Go's `_generate/gen-simd.go` generates (`genSIMDExec`, `genSIMDStats`).
//!
//! Requirements: AVX2 and BMI2 ([`cpu::AVX2`](crate::cpu::AVX2)); [`Tier`](super::exec::Tier)
//! only calls these functions when the CPU has them. No AVX-512 instructions.
//!
//! The loops keep the [memory rules](super::exec) and need no more. One difference from Go:
//! the pattern table has 64 rows, rows 33 and up a copy of row 32, so tokens above 135, which
//! the decoder rejects before running a chunk, cannot index past it.
//!
//! The hot paths have Go's instruction counts (53, 62, 53 and 62 instructions per operation for
//! the plain, delta, flat and flat delta loops; Go 53, 62, 53, 63). [`pin`] and [`cold`] get
//! LLVM there: without them it sinks cursor updates, folds indexes into pointers and lays the
//! escape paths inside the loop, which cost 5-10% against Go.

use core::arch::{asm, x86_64::*};

use super::{MAX_LEN, OFF_TAB};

/// Rows of 32 bytes, indexed by min(offset, 32) (Go `simdPatternTable`).
/// Byte i of row r is 0x80 (keep the plain loaded byte) when i < r or r >= 32,
/// otherwise i % r, which indexes the first 16 source bytes broadcast to both lanes.
/// Row 0 is row 1: a delta reference at position 0 repeats dst[0], which is 0.
#[repr(align(32))]
struct Rows([[u8; 32]; 64]);

static PATTERN_TAB: Rows = {
    let mut t = [[0x80u8; 32]; 64];
    let mut r = 0;
    while r < MAX_LEN {
        let p = if r > 1 { r } else { 1 };
        let mut i = p;
        while i < 32 {
            t[r][i] = (i % p) as u8;
            i += 1;
        }
        r += 1;
    }
    Rows(t)
};

/// Returns v through an empty asm, which LLVM can neither see through nor move: the value is
/// computed where Go computes it. LLVM otherwise sinks the cursor updates to the end of the
/// loop, which keeps their inputs live across the match copy and spills a loop pointer, and
/// folds indexes into pointers, which Go keeps as indexes of dst.
#[inline(always)]
fn pin(mut v: usize) -> usize {
    // SAFETY: the empty asm only names the register.
    unsafe { asm!("/* {0} */", inout(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// [`pin`] for pointers. `map_addr(pin)` would add the pinned address minus the address,
/// which LLVM cannot fold.
#[inline(always)]
#[allow(
    clippy::pointers_in_nomem_asm_block,
    reason = "the asm only passes the pointer through a register"
)]
fn pin_ptr(mut p: *const u8) -> *const u8 {
    // SAFETY: the empty asm only names the register.
    unsafe { asm!("/* {0} */", inout(reg) p, options(nomem, nostack, preserves_flags)) };
    p
}

/// Marks the path that calls it as unlikely, so LLVM places it after the loop as Go does:
/// the common path then runs straight through.
#[cold]
#[inline(always)]
fn cold() {}

#[inline]
#[target_feature(enable = "avx2,bmi2")]
unsafe fn load(p: *const u8) -> __m256i {
    // SAFETY: the caller guarantees 32 readable bytes.
    unsafe { _mm256_loadu_si256(p.cast()) }
}

#[inline]
#[target_feature(enable = "avx2,bmi2")]
unsafe fn store(p: *mut u8, v: __m256i) {
    // SAFETY: the caller guarantees 32 writable bytes.
    unsafe { _mm256_storeu_si256(p.cast(), v) }
}

/// Returns the 32 bytes at dst+i repeated with period r, below 64 (Go's `pattern`): the bytes
/// before r are loaded, the others repeat the first 16.
///
/// i is pinned, so the callers' plain paths index dst as Go's do (LLVM otherwise adds dst+i
/// before the branch), and the two loads stay loads (LLVM otherwise shares the full load with
/// the plain path and broadcasts it with VPERMQ, which adds latency to the chain of stores
/// and loads that short offsets make).
#[inline]
#[target_feature(enable = "avx2,bmi2")]
unsafe fn pattern(r: usize, dst: *const u8, i: usize) -> __m256i {
    // SAFETY: the caller guarantees 32 readable bytes at dst+i; row r < 64 is in the table.
    unsafe {
        let p = dst.add(pin(i));
        let out = load(p);
        let bv = _mm256_broadcastsi128_si256(_mm_loadu_si128(p.cast()));
        let tv = load(PATTERN_TAB.0.as_ptr().add(r).cast());
        _mm256_blendv_epi8(_mm256_shuffle_epi8(bv, tv), out, tv)
    }
}

/// Copies the bytes of an escaped length after the first 32, from s+32 to o+32, and returns
/// the length: 33 plus the next escape byte (Go `simdEscapes`). Each step loads after the
/// previous store, as overlapping matches with offsets of 32 to 63 need.
///
/// The escape cursor lives in memory at escp, as in Go: only escapes touch it, and the loop
/// needs the registers (LLVM otherwise spills a pointer of the loop to keep it).
#[inline]
#[target_feature(enable = "avx2,bmi2")]
unsafe fn escape(s: *const u8, o: *mut u8, escp: *mut *const u8) -> usize {
    // SAFETY: the caller guarantees an escape byte at the cursor, and that s and o are
    // readable and writable for the length rounded up to 32 bytes.
    unsafe {
        let esc = escp.read_volatile();
        escp.write_volatile(esc.add(1));
        let n = *esc as usize + MAX_LEN + 1;
        // 32-byte steps after the first: (n-32+31)>>5 = (escape+32)>>5 of them, as Go's.
        // Pinned, or LLVM hoists dst+32 out of the loop into a register.
        let (s, o) = (pin_ptr(s), pin_ptr(o).cast_mut());
        let mut i = 32;
        loop {
            store(o.add(i), load(s.add(i)));
            i += 32;
            if i >= n {
                break;
            }
        }
        n
    }
}

/// Runs the first n operations of a chunk, see [`exec_go`](super::exec::exec_go)
/// (Go `genSIMDExec`). With DELTA, the first 32 literals of each operation are added to their
/// delta references. With FLAT, the chunk has the no-overlap flag, see
/// [`exec_flat_go`](super::exec::exec_flat_go), and its offset bits start at bit pos.
#[inline]
#[target_feature(enable = "avx2,bmi2")]
#[allow(clippy::too_many_arguments)]
unsafe fn exec<const DELTA: bool, const FLAT: bool>(
    dst: *mut u8,
    lits: *const u8,
    mut tok: *const u8,
    mut ll: *const u8,
    mut offc: *const u8,
    raw: *const u8,
    mut esc: *const u8,
    n: usize,
    mut pos: usize,
    st: &mut [usize; 4],
) {
    let escp = &raw mut esc;
    let end = tok.wrapping_add(n);
    let (mut d, mut last) = (st[0], st[2]);
    let mut litp = lits.wrapping_add(st[1]);
    while tok < end {
        // SAFETY: the memory rules (see super::exec). tok is below its n values. Literals are
        // read 32 bytes at a time from the next literal, which is at most the end of the
        // literals used (rule 2). Outputs are written 32 bytes at a time from positions up to
        // the end of the operation, and escaped lengths up to 31 bytes past it (rule 1).
        // Matches and delta references read from offsets clamped to the position, so from
        // dst[0] on, and up to 32 bytes past the position (rules 1 and 4). ll, offc and raw
        // are read 8 bytes at the next value or bit position, past which they stay readable
        // (rule 3), and esc once per escape. The pattern rows are below 64, as offsets in
        // pattern are below the match length code, at most 63, or 32 for delta references.
        unsafe {
            let t = *tok as usize;
            let mut lv = load(litp);
            if DELTA {
                // The references use the repeat offset from before this operation, clamped to
                // the position.
                let mut dr = last;
                if FLAT {
                    // Raised to 32 first. The position is at least 32, so the references
                    // don't repeat.
                    dr = dr.max(MAX_LEN);
                }
                let dr = dr.min(d);
                let rs = d - dr;
                let rv = if !FLAT && dr < MAX_LEN {
                    cold();
                    pattern(dr, dst, rs)
                } else {
                    load(dst.add(rs))
                };
                lv = _mm256_add_epi8(rv, lv);
            }
            store(dst.add(d), lv);
            let f = (t >> 1) & 1;
            let mut llv = *ll as usize & f.wrapping_neg();
            ll = pin_ptr(ll.add(f));
            if llv == MAX_LEN + 1 {
                cold();
                llv = escape(litp, dst.add(d), escp);
            }
            d = pin(d + llv);
            litp = pin_ptr(litp.add(llv));

            // The next offset symbol and its raw bits: o = base + bits (Go `simdOffset`).
            // They are only consumed for new offsets.
            let ot = OFF_TAB[*offc as usize];
            let nrep = (t & 1) ^ 1;
            offc = pin_ptr(offc.add(nrep));
            let v = raw.add(pos >> 3).cast::<u64>().read_unaligned() >> (pos & 7);
            let nb = ot >> 24;
            let o = (ot & 0xffffff) as usize + _bzhi_u64(v, nb) as usize;
            // Go's NEGQ nrep: the mask also selects the offset, so it needs no second register.
            let m = pin(nrep.wrapping_neg());
            pos += nb as usize & m;
            if m != 0 {
                last = o;
            }

            // Matches copy from the repeat offset clamped to the position.
            let mut r = last;
            let mut ml = t >> 2;
            if FLAT && r < ml {
                cold();
                // Encoders only take it with an escaped match (code 33) at offset 32, which keeps
                // r: escaped matches only raise r to 32.
                r = if ml == MAX_LEN + 1 {
                    r.max(MAX_LEN)
                } else {
                    ml
                };
            }
            let r = r.min(d);
            // Pinned, so the error check below stays d - src, as Go's, and the pattern path may
            // overwrite r (LLVM otherwise keeps r and spills).
            let src = if FLAT { d - r } else { pin(d - r) };
            if !FLAT && r < ml {
                // The clamped offset is below min(ml, 33). Row 32 is a plain copy.
                cold();
                store(dst.add(d), pattern(r, dst, src));
            } else {
                store(dst.add(d), load(dst.add(src)));
            }
            if ml == MAX_LEN + 1 {
                cold();
                // Matches longer than 32 bytes need an offset of at least 32.
                if !FLAT && d - src < MAX_LEN {
                    st[3] = 1;
                }
                ml = escape(dst.add(src), dst.add(d), escp);
            }
            d += ml;
            tok = tok.add(1);
        }
    }
    st[0] = d;
    st[1] = litp as usize - lits as usize;
    st[2] = last;
}

/// Runs the first n operations of a chunk (Go `simdExecAVX2`). st holds the output position,
/// the literal position, the repeat offset and an error flag.
///
/// # Safety
///
/// As for [`ExecFn`](super::exec::ExecFn).
#[target_feature(enable = "avx2,bmi2")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn exec_avx2(
    dst: *mut u8,
    lits: *const u8,
    tok: *const u8,
    ll: *const u8,
    offc: *const u8,
    raw: *const u8,
    esc: *const u8,
    n: usize,
    st: &mut [usize; 4],
) {
    // SAFETY: as for this function.
    unsafe { exec::<false, false>(dst, lits, tok, ll, offc, raw, esc, n, 0, st) }
}

/// Runs the first n operations of a chunk with delta literals (Go `simdExecDeltaAVX2`).
/// st holds the output position, the literal position, the repeat offset and an error flag.
///
/// # Safety
///
/// As for [`ExecFn`](super::exec::ExecFn).
#[target_feature(enable = "avx2,bmi2")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn exec_delta_avx2(
    dst: *mut u8,
    lits: *const u8,
    tok: *const u8,
    ll: *const u8,
    offc: *const u8,
    raw: *const u8,
    esc: *const u8,
    n: usize,
    st: &mut [usize; 4],
) {
    // SAFETY: as for this function.
    unsafe { exec::<true, false>(dst, lits, tok, ll, offc, raw, esc, n, 0, st) }
}

/// Runs the first n operations of a chunk with the no-overlap flag, from bit pos of raw
/// (Go `simdExecFlatAVX2`). Their operations must start at position 32 or later. st holds
/// the output position, the literal position, the repeat offset and an error flag.
///
/// # Safety
///
/// As for [`ExecFlatFn`](super::exec::ExecFlatFn).
#[target_feature(enable = "avx2,bmi2")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn exec_flat_avx2(
    dst: *mut u8,
    lits: *const u8,
    tok: *const u8,
    ll: *const u8,
    offc: *const u8,
    raw: *const u8,
    esc: *const u8,
    n: usize,
    pos: usize,
    st: &mut [usize; 4],
) {
    // SAFETY: as for this function.
    unsafe { exec::<false, true>(dst, lits, tok, ll, offc, raw, esc, n, pos, st) }
}

/// Runs the first n operations of a chunk with the no-overlap flag and delta literals, from
/// bit pos of raw (Go `simdExecFlatDeltaAVX2`). Their operations must start at position 32
/// or later. st holds the output position, the literal position, the repeat offset and an
/// error flag.
///
/// # Safety
///
/// As for [`ExecFlatFn`](super::exec::ExecFlatFn).
#[target_feature(enable = "avx2,bmi2")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn exec_flat_delta_avx2(
    dst: *mut u8,
    lits: *const u8,
    tok: *const u8,
    ll: *const u8,
    offc: *const u8,
    raw: *const u8,
    esc: *const u8,
    n: usize,
    pos: usize,
    st: &mut [usize; 4],
) {
    // SAFETY: as for this function.
    unsafe { exec::<true, true>(dst, lits, tok, ll, offc, raw, esc, n, pos, st) }
}

#[inline]
#[target_feature(enable = "avx2,bmi2")]
fn splat(c: u8) -> __m256i {
    _mm256_set1_epi8(c as i8)
}

/// Adds the sums of each 8 bytes of x to the quadwords of acc (Go `simdSum`).
#[inline]
#[target_feature(enable = "avx2,bmi2")]
fn sum(x: __m256i, acc: __m256i) -> __m256i {
    _mm256_add_epi64(_mm256_sad_epu8(x, _mm256_setzero_si256()), acc)
}

/// Returns the sum of the quadwords of v (Go `simdStatsEnd`).
#[inline]
#[target_feature(enable = "avx2,bmi2")]
fn hsum(v: __m256i) -> usize {
    let x = _mm_add_epi64(_mm256_extracti128_si256::<1>(v), _mm256_castsi256_si128(v));
    _mm_cvtsi128_si64(_mm_add_epi64(_mm_shuffle_epi32::<0x4e>(x), x)) as usize
}

/// Returns the token statistics of b, a multiple of 32 bytes (Go `simdTokStatsAVX2`):
/// the summed match length codes, the literals, the repeats and the escaped matches, and
/// whether every token is at most 135. lit and esc are counted 2 and 4 times.
///
/// # Safety
///
/// The CPU has AVX2 and BMI2.
#[target_feature(enable = "avx2,bmi2")]
pub(super) unsafe fn tok_stats_avx2(b: &[u8]) -> (usize, usize, usize, usize, bool) {
    let zero = _mm256_setzero_si256();
    let mut valid = _mm256_cmpeq_epi8(zero, zero);
    let [mut out, mut lit, mut rep, mut esc] = [zero; 4];
    let (m3f, m4, m2, m1, max) = (splat(0x3f), splat(4), splat(2), splat(1), splat(135));
    debug_assert!(b.len().is_multiple_of(32));
    for c in b.as_chunks::<32>().0 {
        // SAFETY: c holds 32 bytes.
        let t = unsafe { load(c.as_ptr()) };
        valid = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_min_epu8(max, t), t), valid);
        out = sum(_mm256_and_si256(_mm256_srli_epi16::<2>(t), m3f), out);
        lit = sum(_mm256_and_si256(m2, t), lit);
        rep = sum(_mm256_and_si256(m1, t), rep);
        // Match length code 33 has bits 7 and 2.
        let x = _mm256_and_si256(_mm256_srli_epi16::<5>(t), t);
        esc = sum(_mm256_and_si256(m4, x), esc);
    }
    let ok = _mm256_movemask_epi8(valid) == -1;
    (hsum(out), hsum(lit), hsum(rep), hsum(esc), ok)
}

/// Returns the sum of b, a multiple of 32 bytes, and how many bytes are 33
/// (Go `simdLLStatsAVX2`). ok is false if a byte is outside 1-33.
///
/// # Safety
///
/// The CPU has AVX2 and BMI2.
#[target_feature(enable = "avx2,bmi2")]
pub(super) unsafe fn ll_stats_avx2(b: &[u8]) -> (usize, usize, bool) {
    let zero = _mm256_setzero_si256();
    let mut valid = _mm256_cmpeq_epi8(zero, zero);
    let (mut sum_v, mut cnt) = (zero, zero);
    let (c1, c33) = (splat(1), splat(33));
    debug_assert!(b.len().is_multiple_of(32));
    for c in b.as_chunks::<32>().0 {
        // SAFETY: c holds 32 bytes.
        let t = unsafe { load(c.as_ptr()) };
        let x = _mm256_min_epu8(c33, _mm256_max_epu8(c1, t));
        valid = _mm256_and_si256(_mm256_cmpeq_epi8(x, t), valid);
        cnt = sum(_mm256_and_si256(c1, _mm256_cmpeq_epi8(c33, t)), cnt);
        sum_v = sum(t, sum_v);
    }
    let ok = _mm256_movemask_epi8(valid) == -1;
    (hsum(sum_v), hsum(cnt), ok)
}

/// Returns the sum of b>>3 over b, a multiple of 32 bytes, and how often it is 0
/// (Go `simdOffStatsAVX2`). ok is false if a byte is outside 1-175.
///
/// # Safety
///
/// The CPU has AVX2 and BMI2.
#[target_feature(enable = "avx2,bmi2")]
pub(super) unsafe fn off_stats_avx2(b: &[u8]) -> (usize, usize, bool) {
    let zero = _mm256_setzero_si256();
    let mut valid = _mm256_cmpeq_epi8(zero, zero);
    let (mut sum_v, mut cnt) = (zero, zero);
    let (c1, c175, m1f) = (splat(1), splat(175), splat(0x1f));
    debug_assert!(b.len().is_multiple_of(32));
    for c in b.as_chunks::<32>().0 {
        // SAFETY: c holds 32 bytes.
        let t = unsafe { load(c.as_ptr()) };
        let x = _mm256_min_epu8(c175, _mm256_max_epu8(c1, t));
        valid = _mm256_and_si256(_mm256_cmpeq_epi8(x, t), valid);
        let t = _mm256_and_si256(_mm256_srli_epi16::<3>(t), m1f);
        cnt = sum(_mm256_and_si256(c1, _mm256_cmpeq_epi8(zero, t)), cnt);
        sum_v = sum(t, sum_v);
    }
    let ok = _mm256_movemask_epi8(valid) == -1;
    (hsum(sum_v), hsum(cnt), ok)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::simd::exec::{Tier, simd_tiers};
    use crate::cpu;

    /// The rows are Go's `simdPatternTable`, and rows 33 and up are row 32.
    #[test]
    fn pattern_tab() {
        for (r, row) in PATTERN_TAB.0.iter().enumerate() {
            let p = r.clamp(1, MAX_LEN);
            for (i, &b) in row.iter().enumerate() {
                let want = if i < p || p >= MAX_LEN {
                    0x80
                } else {
                    (i % p) as u8
                };
                assert_eq!(b, want, "row {r} byte {i}");
            }
        }
    }

    /// The tests run the tier when the CPU has it, and the dispatch picks it below AVX-512,
    /// as Go's `simdTiers` forces it with the cpu flags.
    #[test]
    fn avx2_tier() {
        let _g = cpu::Disabled::new(cpu::AVX512);
        let has = cpu::has(cpu::AVX2);
        let mut seen = false;
        simd_tiers(|t| seen |= t == Tier::Avx2);
        assert!(seen == has && (Tier::best() == Tier::Avx2) == has);
        cpu::disable(cpu::AVX2);
        assert!(!Tier::Avx2.usable() && Tier::best() == Tier::Go);
    }
}
