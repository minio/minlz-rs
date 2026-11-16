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

//! The AVX-512 operation loops and stream stats of SIMD blocks: intrinsics ports of the
//! assembly Go's `_generate/gen-simd.go` generates (`genSIMDExec512`, `genSIMDStats512`), with
//! the contracts of `asm_amd64.go` and `decode_amd64.go`.
//!
//! Every function needs [`AVX512`](crate::cpu::AVX512): AVX-512 F/BW/VL/VBMI/VBMI2, AVX2,
//! BMI2 and POPCNT. The loops keep the [memory rules](super::exec). Their stream reads are
//! exact: tokens and raw bits are loaded masked (raw up to `nraw`), and literal lengths and
//! offset symbols by expand loads, which only read the values they place.

use core::arch::asm;
use core::arch::x86_64::*;

use super::MAX_LEN;

/// How many operations the loops expand at a time (Go `simdBatch`).
const BATCH: usize = 64;

/// VPERMB indexes by min(offset, 32) (Go `simdPermTable`): byte i of row r is i % r, and i in
/// row 32. Row 0 is row 1: a delta reference at position 0 repeats dst[0], which is 0.
/// Go has 33 rows. The rows after 32 repeat it, so the match length codes above 33 that
/// the decoder rejects before running a chunk still index inside the table.
static PERM_TAB: [[u8; 32]; 64] = {
    let mut t = [[0u8; 32]; 64];
    let mut r = 0;
    while r < 64 {
        let p = if r == 0 {
            1
        } else if r > MAX_LEN {
            MAX_LEN
        } else {
            r
        };
        let mut i = 0;
        while i < 32 {
            t[r][i] = (i % p) as u8;
            i += 1;
        }
        r += 1;
    }
    t
};

/// The literal lengths and offsets of a batch (Go's stack frame).
#[repr(C, align(64))]
struct Batch {
    llx: [u8; BATCH],
    offx: [u32; BATCH],
}

/// Marks the path that calls it as rare, so LLVM moves it out of the loop as Go's assembly does.
#[cold]
#[inline(always)]
fn cold() {}

/// Returns a zeroed vector that LLVM cannot see is zero, for the expand loads to merge into.
/// Zen 4/5 give the zero-masking form of VPEXPANDB a dependency on its old destination. Go's
/// lands in the register its tokens were just loaded into, but LLVM picks the copy loop's
/// register, so each expansion waited for the last copy of the batch before: the expansion
/// alone took 1.8 times Go's.
#[inline]
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
fn zreg() -> __m512i {
    let mut v = _mm512_setzero_si512();
    // SAFETY: the template is empty; it neither reads nor writes memory.
    unsafe { asm!("/* {0} */", inout(zmm_reg) v, options(pure, nomem, nostack, preserves_flags)) };
    v
}

/// Returns the mask k, which LLVM cannot see through. Without it, LLVM moves the new offset
/// mask of groups 2 and 3 through the stack into a vector register, as an 8-byte store that a
/// 16-byte load reads back, which cannot be forwarded, instead of shifting it with KSHIFTRQ.
#[inline]
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
fn kbar(mut k: u64) -> u64 {
    // SAFETY: the template is empty; it neither reads nor writes memory.
    unsafe { asm!("/* {0} */", inout(kreg) k, options(pure, nomem, nostack, preserves_flags)) };
    k
}

/// Returns x, which LLVM cannot see through.
#[inline(always)]
fn opaque(mut x: usize) -> usize {
    // SAFETY: the template is empty; it neither reads nor writes memory.
    unsafe { asm!("/* {0} */", inout(reg) x, options(pure, nomem, nostack, preserves_flags)) };
    x
}

/// # Safety
/// p has 32 readable bytes.
#[inline]
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
unsafe fn load32(p: *const u8) -> __m256i {
    // SAFETY: the caller's.
    unsafe { _mm256_loadu_si256(p.cast()) }
}

/// # Safety
/// p has 32 writable bytes.
#[inline]
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
unsafe fn store32(p: *mut u8, v: __m256i) {
    // SAFETY: the caller's.
    unsafe { _mm256_storeu_si256(p.cast(), v) }
}

/// Copies the bytes of an escaped length 33+e after the first 32 from s to p, in 32-byte steps:
/// (33+e-32+31)>>5 = (e+32)>>5 of them.
///
/// # Safety
/// s and p have 33+e bytes and 32 bytes of slack, readable and writable.
#[inline]
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
unsafe fn copy_tail(mut p: *mut u8, mut s: *const u8, e: usize) {
    let mut k = (e + 32) >> 5;
    loop {
        // SAFETY: the caller's.
        unsafe {
            (p, s) = (p.add(32), s.add(32));
            store32(p, load32(s));
        }
        k -= 1;
        if k == 0 {
            break;
        }
    }
}

/// Returns the 32 bytes at `dst[i..]` repeated with period r, which is below 64 (VPERMB row r).
///
/// # Safety
/// `dst[i..]` has 32 readable bytes.
#[inline]
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
unsafe fn pattern(r: usize, dst: *const u8, i: usize) -> __m256i {
    // SAFETY: rows have 32 bytes, and dst[i..] is the caller's. The row is loaded first, so
    // LLVM doesn't hoist the source address out of the branches, which costs an add.
    unsafe {
        let row = load32(PERM_TAB[r].as_ptr());
        _mm256_permutexvar_epi8(row, load32(dst.add(i)))
    }
}

/// Runs the first n operations of a chunk, see [`exec_go`](super::exec::exec_go) (Go
/// `genSIMDExec512`). Each batch of [`BATCH`] operations is first expanded into the literal
/// length and the resolved offset of each operation, so the copy loop does no stream
/// bookkeeping. With DELTA, the first 32 literals of each operation are added to their delta
/// references. With FLAT, the chunk has the no-overlap flag, see
/// [`exec_flat_go`](super::exec::exec_flat_go), and its offset bits start at bit pos.
///
/// # Safety
/// As for [`ExecFlatRawFn`](super::exec::ExecFlatRawFn).
#[inline]
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn exec<const DELTA: bool, const FLAT: bool>(
    dst: *mut u8,
    lits: *const u8,
    mut tok: *const u8,
    mut ll: *const u8,
    mut offc: *const u8,
    raw: *const u8,
    nraw: usize,
    mut esc: *const u8,
    n: usize,
    pos: usize,
    st: &mut [usize; 4],
) {
    // The copy loop runs at least one operation.
    if n == 0 {
        return;
    }
    let mut d = st[0];
    // SAFETY: the run has n tokens, and st[1] is a literal position.
    let (end, mut litp) = unsafe { (tok.add(n), lits.add(st[1])) };
    // The offset in effect after the last expanded operation, in every lane.
    let mut carry = _mm512_set1_epi32(st[2] as i32);
    let mut last = st[2];
    let mut s_pos = pos;
    let zero = _mm512_setzero_si512();
    let ones = _mm512_set1_epi32(-1);
    let (c1, c2, c1f) = (
        _mm512_set1_epi32(0x01010101),
        _mm512_set1_epi32(0x02020202),
        _mm512_set1_epi32(0x1f1f1f1f),
    );
    let (c7, c8, c15) = (
        _mm512_set1_epi32(7),
        _mm512_set1_epi32(8),
        _mm512_set1_epi32(15),
    );
    let iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    // Dword lanes 0-7 select qword 0, lanes 8-15 qword 1.
    let gidx = _mm512_maskz_mov_epi32(0xff00, c2);
    let c0123 = _mm512_set1_epi32(0x03020100);
    // Selects the low byte of each dword into all its bytes.
    let lowb = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12,
    ));
    // Without delta, the expansion raises the offsets, from the tokens of the batch.
    let raise = FLAT && !DELTA;
    let c32 = _mm512_set1_epi32(MAX_LEN as i32);
    let mut x = Batch {
        llx: [0; BATCH],
        offx: [0; BATCH],
    };
    loop {
        let m = (end as usize - tok as usize).min(BATCH);
        let kv = _bzhi_u64(u64::MAX, m as u32);
        // SAFETY: kv selects the next m tokens of the run.
        let tokv = unsafe { _mm512_maskz_loadu_epi8(kv, tok.cast()) };
        // The lanes of the batch's operations with literals and with new offsets.
        let k_lit = _mm512_test_epi8_mask(tokv, c2);
        let k_new = _mm512_mask_testn_epi8_mask(kv, tokv, c1);
        // SAFETY: the expand loads only read the values they place, the next literal lengths
        // and offset symbols of the run, and the batch's arrays have 64 bytes.
        let sym = unsafe {
            let l = _mm512_mask_expandloadu_epi8(zreg(), k_lit, ll.cast());
            ll = ll.add(k_lit.count_ones() as usize);
            _mm512_store_si512(x.llx.as_mut_ptr().cast(), l);
            let s = _mm512_mask_expandloadu_epi8(zreg(), k_new, offc.cast());
            offc = offc.add(k_new.count_ones() as usize);
            s
        };
        // The offset symbols of the batch with 0 for repeats, their raw bit counts, the bit
        // counts before each within its 8 operations, and the bit position of each 8 operations.
        // nb = max(s>>3 - 1, 0). Its sums within 8 operations fit in bytes.
        let nb8 = _mm512_subs_epu8(_mm512_and_si512(_mm512_srli_epi16::<3>(sym), c1f), c1);
        let mut y = nb8;
        y = _mm512_add_epi8(y, _mm512_slli_epi64::<8>(y));
        y = _mm512_add_epi8(y, _mm512_slli_epi64::<16>(y));
        y = _mm512_add_epi8(y, _mm512_slli_epi64::<32>(y));
        let exc8 = _mm512_sub_epi8(y, nb8);
        let t8 = _mm512_sad_epu8(nb8, zero);
        let mut q = t8;
        q = _mm512_add_epi64(q, _mm512_alignr_epi64::<7>(q, zero));
        q = _mm512_add_epi64(q, _mm512_alignr_epi64::<6>(q, zero));
        q = _mm512_add_epi64(q, _mm512_alignr_epi64::<4>(q, zero));
        let qabs = _mm512_add_epi64(_mm512_sub_epi64(q, t8), _mm512_set1_epi64(s_pos as i64));
        s_pos += _mm_cvtsi128_si64(_mm512_castsi512_si128(_mm512_alignr_epi64::<7>(q, q))) as usize;

        // Decodes the offsets of operations 16h to 16h+15 of the batch into offx.
        // Symbol 0 decodes as offset 0, which then takes the offset before it.
        macro_rules! group {
            ($h:literal) => {{
                // The lane each operation takes its offset from: the last new offset up to
                // it, or -1 for the carry.
                let kg = _kshiftri_mask64::<{ 16 * $h }>(kbar(k_new)) as u16;
                let mut from = _mm512_mask_mov_epi32(ones, kg, iota);
                from = _mm512_max_epi32(_mm512_alignr_epi32::<15>(from, ones), from);
                from = _mm512_max_epi32(_mm512_alignr_epi32::<14>(from, ones), from);
                from = _mm512_max_epi32(_mm512_alignr_epi32::<12>(from, ones), from);
                from = _mm512_max_epi32(_mm512_alignr_epi32::<8>(from, ones), from);
                let s = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<$h>(sym));
                let nb = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<$h>(nb8));
                let mut bp = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<$h>(exc8));
                let qs = _mm512_alignr_epi64::<{ 2 * $h }>(qabs, qabs);
                bp = _mm512_add_epi32(_mm512_permutexvar_epi32(gidx, qs), bp);
                // The 64 raw bytes from the byte holding the first bit position.
                let wb = _mm_cvtsi128_si64(_mm512_castsi512_si128(qs)) as usize >> 3;
                debug_assert!(wb <= nraw);
                let kr = _bzhi_u64(u64::MAX, (nraw - wb).min(64) as u32);
                // SAFETY: the bit positions of the run are inside raw, and kr selects the
                // bytes from wb up to nraw.
                let mut w = unsafe { _mm512_maskz_loadu_epi8(kr, raw.add(wb).cast()) };
                // Gather the 4 bytes from each bit position, then shift and mask.
                let mut xi = _mm512_sub_epi32(bp, _mm512_set1_epi32((wb << 3) as i32));
                xi = _mm512_shuffle_epi8(_mm512_srli_epi32::<3>(xi), lowb);
                xi = _mm512_add_epi8(c0123, xi);
                w = _mm512_permutexvar_epi8(xi, w);
                w = _mm512_srlv_epi32(w, _mm512_and_si512(c7, bp));
                w = _mm512_andnot_si512(_mm512_sllv_epi32(ones, nb), w);
                // The offset is (s&7 | min(s, 8)&8) << nb, plus the bits.
                let base = _mm512_ternarylogic_epi32::<0xd8>(_mm512_min_epu32(c8, s), s, c7);
                w = _mm512_add_epi32(_mm512_sllv_epi32(base, nb), w);
                let from = _mm512_permutex2var_epi32(w, from, carry);
                let o = if raise {
                    // The copy loop uses the offsets raised to min(ml, 32); the carry stays
                    // the repeat offset.
                    let ml = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32::<$h>(tokv));
                    let ml = _mm512_min_epu32(c32, _mm512_srli_epi32::<2>(ml));
                    _mm512_max_epu32(from, ml)
                } else {
                    from
                };
                // SAFETY: offx has 64 dwords.
                unsafe { _mm512_store_si512(x.offx.as_mut_ptr().add(16 * $h).cast(), o) };
                carry = _mm512_permutexvar_epi32(c15, from);
            }};
        }
        group!(0);
        if m > 16 {
            group!(1);
            if m > 32 {
                group!(2);
                if m > 48 {
                    group!(3);
                }
            }
        }

        // The operations keep the memory rules: they end SLACK bytes before the end of dst,
        // each 32-byte store ends at most 32 bytes past its operation, literals are readable
        // 32 bytes past those they use, the escapes are the run's, and offsets are clamped to
        // the position, so sources are inside dst.
        for j in 0..m {
            // SAFETY: j < m, so this is a token of the run, and literals are readable 32
            // bytes from the next one.
            let mut t = unsafe { *tok.add(j) } as usize;
            let mut lv = unsafe { load32(litp) };
            if DELTA {
                // The references use the offset from before this operation, clamped to the
                // position.
                let mut dr = last;
                if FLAT {
                    // Raised to 32 first. The position is at least 32, so the references
                    // don't repeat.
                    dr = dr.max(MAX_LEN);
                }
                dr = opaque(dr.min(d));
                let rs = d - dr;
                // SAFETY: rs is at most d, and the operation's 32 bytes at d are inside dst.
                let rv = if !FLAT && dr < MAX_LEN {
                    cold();
                    unsafe { pattern(dr, dst, rs) }
                } else {
                    unsafe { load32(dst.add(rs)) }
                };
                lv = _mm256_add_epi8(lv, rv);
            }
            // SAFETY: the operation's 32 bytes at d are inside dst.
            unsafe { store32(dst.add(d), lv) };
            let mut llv = x.llx[j] as usize;
            let mut o = x.offx[j] as usize;
            if llv == MAX_LEN + 1 {
                cold();
                // Escaped literal lengths are 33 plus an escape byte; the first 32 literals
                // are stored.
                // SAFETY: the run has an escape for each literal length of 33, and the
                // literals and output of the operation, 33+e bytes, have 32 bytes of slack.
                let e = unsafe { *esc } as usize;
                esc = unsafe { esc.add(1) };
                llv = e + MAX_LEN + 1;
                unsafe { copy_tail(dst.add(d), litp, e) };
            }
            d += llv;
            // SAFETY: the operation's literals are inside lits.
            litp = unsafe { litp.add(llv) };
            if DELTA {
                last = o;
            }
            if FLAT && DELTA {
                // Without delta, the expansion raised the offset. Encoders only take it with an
                // escaped match (code 33) at offset 32, which keeps o: escaped matches only raise
                // it to 32.
                t >>= 2;
                if o < t {
                    cold();
                    o = if t == MAX_LEN + 1 { o.max(MAX_LEN) } else { t };
                }
            }
            // Matches copy from the offset clamped to the position. Without the barriers, LLVM
            // turns the flat loops' d - min(o, d) into a saturating subtract, which costs a
            // register and a zeroing move, and keeps o for the check of long matches instead
            // of taking d - src, as Go does, which costs a register on the pattern path.
            o = opaque(o.min(d));
            let src = opaque(d - o);
            if !(FLAT && DELTA) {
                t >>= 2;
            }
            // SAFETY: src is at most d, and the match's first 32 bytes at d are inside dst.
            let v = if !FLAT && o < t {
                // Taken when the clamped offset is below min(ml, 33). Row 32 is a plain copy.
                cold();
                unsafe { pattern(o, dst, src) }
            } else {
                unsafe { load32(dst.add(src)) }
            };
            unsafe { store32(dst.add(d), v) };
            if t == MAX_LEN + 1 {
                cold();
                // Matches longer than 32 bytes need an offset of at least 32.
                if !FLAT && d - src < MAX_LEN {
                    st[3] = 1;
                }
                // SAFETY: the run has an escape for each match length code of 33, and the
                // match, 33+e bytes from d, has 32 bytes of slack; src is before d.
                let e = unsafe { *esc } as usize;
                esc = unsafe { esc.add(1) };
                unsafe { copy_tail(dst.add(d), dst.add(src), e) };
                // Added here, so the loop adds the length code without a merge.
                d += e;
            }
            d += t;
        }
        // SAFETY: the batch's m tokens are the run's.
        tok = unsafe { tok.add(m) };
        if tok >= end {
            break;
        }
    }
    st[0] = d;
    st[1] = litp as usize - lits as usize;
    st[2] = if DELTA {
        last
    } else {
        _mm_cvtsi128_si32(_mm512_castsi512_si128(carry)) as u32 as usize
    };
}

/// Runs the first n operations of a chunk (Go `simdExecAVX512`).
///
/// # Safety
/// As for [`ExecRawFn`](super::exec::ExecRawFn).
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn exec_avx512(
    dst: *mut u8,
    lits: *const u8,
    tok: *const u8,
    ll: *const u8,
    offc: *const u8,
    raw: *const u8,
    nraw: usize,
    esc: *const u8,
    n: usize,
    st: &mut [usize; 4],
) {
    // SAFETY: the caller's.
    unsafe { exec::<false, false>(dst, lits, tok, ll, offc, raw, nraw, esc, n, 0, st) }
}

/// Runs the first n operations of a chunk with delta literals (Go `simdExecDeltaAVX512`).
///
/// # Safety
/// As for [`ExecRawFn`](super::exec::ExecRawFn).
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn exec_delta_avx512(
    dst: *mut u8,
    lits: *const u8,
    tok: *const u8,
    ll: *const u8,
    offc: *const u8,
    raw: *const u8,
    nraw: usize,
    esc: *const u8,
    n: usize,
    st: &mut [usize; 4],
) {
    // SAFETY: the caller's.
    unsafe { exec::<true, false>(dst, lits, tok, ll, offc, raw, nraw, esc, n, 0, st) }
}

/// Runs the first n operations of a chunk with the no-overlap flag, from bit pos of raw (Go
/// `simdExecFlatAVX512`). The operations start at position 32 or later.
///
/// # Safety
/// As for [`ExecFlatRawFn`](super::exec::ExecFlatRawFn).
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn exec_flat_avx512(
    dst: *mut u8,
    lits: *const u8,
    tok: *const u8,
    ll: *const u8,
    offc: *const u8,
    raw: *const u8,
    nraw: usize,
    esc: *const u8,
    n: usize,
    pos: usize,
    st: &mut [usize; 4],
) {
    // SAFETY: the caller's.
    unsafe { exec::<false, true>(dst, lits, tok, ll, offc, raw, nraw, esc, n, pos, st) }
}

/// Runs the first n operations of a chunk with the no-overlap flag and delta literals, from
/// bit pos of raw (Go `simdExecFlatDeltaAVX512`). The operations start at position 32 or later.
///
/// # Safety
/// As for [`ExecFlatRawFn`](super::exec::ExecFlatRawFn).
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn exec_flat_delta_avx512(
    dst: *mut u8,
    lits: *const u8,
    tok: *const u8,
    ll: *const u8,
    offc: *const u8,
    raw: *const u8,
    nraw: usize,
    esc: *const u8,
    n: usize,
    pos: usize,
    st: &mut [usize; 4],
) {
    // SAFETY: the caller's.
    unsafe { exec::<true, true>(dst, lits, tok, ll, offc, raw, nraw, esc, n, pos, st) }
}

/// Runs f on the bytes of b, which is not empty, in steps of 64 with the lanes past its end 0,
/// and their lanes (Go `simdStats512Load`, `simdStats512Next`). Returns whether no step
/// returned a bad lane.
#[inline]
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
fn each64(b: &[u8], mut f: impl FnMut(__m512i, u64) -> u64) -> bool {
    let (mut p, mut rem, mut bad) = (b.as_ptr(), b.len() as isize, 0);
    loop {
        let k = _bzhi_u64(u64::MAX, (rem as usize).min(64) as u32);
        // SAFETY: rem > 0 bytes of b follow p, and k selects at most rem of them.
        bad |= f(unsafe { _mm512_maskz_loadu_epi8(k, p.cast()) }, k);
        p = p.wrapping_add(64);
        rem -= 64;
        if rem <= 0 {
            return bad == 0;
        }
    }
}

/// Returns the sum of the quadwords of v.
#[inline]
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
fn sum(v: __m512i) -> usize {
    _mm512_reduce_add_epi64(v) as usize
}

/// Returns the token statistics of b, which is not empty (Go `simdTokStatsAVX512`): the
/// summed match length codes, twice the operations with literals, the repeats, four times the
/// escaped matches, and whether every token is at most 135.
///
/// # Safety
/// The CPU has [`AVX512`](crate::cpu::AVX512).
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
pub(super) unsafe fn tok_stats_avx512(b: &[u8]) -> (usize, usize, usize, usize, bool) {
    let zero = _mm512_setzero_si512();
    let (mut out, mut lit, mut rep, mut esc) = (zero, zero, zero, zero);
    let (m3f, m4, m2, m1) = (
        _mm512_set1_epi8(0x3f),
        _mm512_set1_epi8(4),
        _mm512_set1_epi8(2),
        _mm512_set1_epi8(1),
    );
    let max = _mm512_set1_epi8(135u8 as i8);
    let ok = each64(b, |t, _| {
        let x = _mm512_and_si512(_mm512_srli_epi16::<2>(t), m3f);
        out = _mm512_add_epi64(_mm512_sad_epu8(x, zero), out);
        lit = _mm512_add_epi64(_mm512_sad_epu8(_mm512_and_si512(t, m2), zero), lit);
        rep = _mm512_add_epi64(_mm512_sad_epu8(_mm512_and_si512(t, m1), zero), rep);
        // Match length code 33 has bits 7 and 2.
        let x = _mm512_and_si512(_mm512_and_si512(_mm512_srli_epi16::<5>(t), t), m4);
        esc = _mm512_add_epi64(_mm512_sad_epu8(x, zero), esc);
        _mm512_cmpgt_epu8_mask(t, max)
    });
    (sum(out), sum(lit), sum(rep), sum(esc), ok)
}

/// Returns the sum of b, which is not empty, and how many bytes are 33, and whether every
/// byte is 1 to 33 (Go `simdLLStatsAVX512`).
///
/// # Safety
/// The CPU has [`AVX512`](crate::cpu::AVX512).
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
pub(super) unsafe fn ll_stats_avx512(b: &[u8]) -> (usize, usize, bool) {
    let zero = _mm512_setzero_si512();
    let (mut acc, mut cnt) = (zero, 0);
    let c33 = _mm512_set1_epi8(MAX_LEN as i8 + 1);
    let ok = each64(b, |t, kv| {
        cnt += _mm512_cmpeq_epi8_mask(t, c33).count_ones() as usize;
        acc = _mm512_add_epi64(_mm512_sad_epu8(t, zero), acc);
        _mm512_mask_cmpgt_epu8_mask(kv, t, c33) | _mm512_mask_testn_epi8_mask(kv, t, t)
    });
    (sum(acc), cnt, ok)
}

/// Returns the sum of b>>3 over b, which is not empty, how often it is 0, and whether every
/// byte is 1 to 175 (Go `simdOffStatsAVX512`).
///
/// # Safety
/// The CPU has [`AVX512`](crate::cpu::AVX512).
#[target_feature(enable = "avx2,bmi2,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,popcnt")]
pub(super) unsafe fn off_stats_avx512(b: &[u8]) -> (usize, usize, bool) {
    let zero = _mm512_setzero_si512();
    let (mut acc, mut cnt) = (zero, 0);
    let (c175, m1f) = (_mm512_set1_epi8(175u8 as i8), _mm512_set1_epi8(0x1f));
    let ok = each64(b, |t, kv| {
        let x = _mm512_and_si512(_mm512_srli_epi16::<3>(t), m1f);
        cnt += _mm512_mask_testn_epi8_mask(kv, x, x).count_ones() as usize;
        acc = _mm512_add_epi64(_mm512_sad_epu8(x, zero), acc);
        _mm512_mask_cmpgt_epu8_mask(kv, t, c175) | _mm512_mask_testn_epi8_mask(kv, t, t)
    });
    (sum(acc), cnt, ok)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::simd::decode::Chunk;
    use crate::block::simd::exec::{ExecFlatRawFn, exec_flat_go, exec_go};
    use crate::block::simd::{OFF_TAB, PAD, SLACK};
    use crate::guard::Buffer;

    /// The AVX-512 loops read the streams exactly: up to the last token, literal length,
    /// offset symbol and escape the operations use, and raw bytes up to nraw, the offset bits
    /// rounded up to whole bytes. Each stream ends at an inaccessible page; the Go loops run
    /// the same operations on padded copies. Flat runs start at a random operation, from its
    /// stream positions, as after `Chunk::head`.
    #[test]
    fn exec_exact_streams() {
        if !crate::cpu::has(crate::cpu::AVX512) {
            return;
        }
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut rng = |n: usize| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % n as u64) as usize
        };
        let mut guards: [Buffer; 5] = core::array::from_fn(|_| Buffer::new(1 << 16).unwrap());
        let mut lits = vec![0u8; 300 * (MAX_LEN + 256) + 1000 + PAD];
        for i in 0..2000 {
            let (delta, flat) = (i & 1 != 0, i & 2 != 0);
            // Wide chunks: mostly new offsets with 20 raw bits.
            let wide = i % 3 == 1;
            let d0 = MAX_LEN + rng(5000);
            let (l0, mut last) = (rng(1000), 1 + rng(1 << 24));
            let mut v: [Vec<u8>; 5] = Default::default();
            let (mut acc, mut nacc, mut bits, mut d) = (0u64, 0, 0, d0);
            let ops = 1 + rng(300);
            let k0 = if flat { rng(ops) } else { 0 };
            let mut cur = [0; 4];
            for j in 0..ops {
                if j == k0 {
                    cur = [v[1].len(), v[2].len(), v[3].len(), bits];
                }
                let mut t = rng(4) as u8;
                if wide && rng(16) > 0 {
                    t &= !1;
                }
                if t & 2 != 0 {
                    let n = 1 + rng(MAX_LEN + 1);
                    v[1].push(n as u8);
                    d += n;
                    if n > MAX_LEN {
                        let e = rng(256);
                        v[3].push(e as u8);
                        d += e;
                    }
                }
                if t & 1 == 0 {
                    let sym = if wide { 168 + rng(8) } else { 1 + rng(175) };
                    v[2].push(sym as u8);
                    let nb = (OFF_TAB[sym] >> 24) as usize;
                    let r = (rng(1 << 20) as u64) & ((1 << nb) - 1);
                    last = (OFF_TAB[sym] & 0xffffff) as usize + r as usize;
                    acc |= r << nacc;
                    (nacc, bits) = (nacc + nb, bits + nb);
                    while nacc >= 8 {
                        v[4].push(acc as u8);
                        (acc, nacc) = (acc >> 8, nacc - 8);
                    }
                }
                let mut ml = rng(MAX_LEN + 2);
                let r = if flat { last.max(MAX_LEN) } else { last }.min(d);
                if ml > MAX_LEN && r < MAX_LEN {
                    ml = MAX_LEN;
                } else if ml > MAX_LEN {
                    let e = rng(256);
                    v[3].push(e as u8);
                    d += e;
                }
                d += ml;
                v[0].push(t | (ml << 2) as u8);
            }
            if nacc > 0 {
                v[4].push(acc as u8);
            }
            for b in &mut lits {
                *b = rng(256) as u8;
            }
            let mut want = vec![0u8; d + SLACK];
            for b in &mut want[..d0] {
                *b = rng(256) as u8;
            }
            let mut got = want.clone();
            let n = v[0].len();
            let st0 = [d0, l0, rng(1 << 24) + 1, 0];
            let pad: Vec<Vec<u8>> = v.iter().map(|s| [&s[..], &[0; PAD]].concat()).collect();
            let c = Chunk {
                tok: &pad[0],
                ll: &pad[1],
                offc: &pad[2],
                esc: &pad[3],
                raw: &pad[4],
                n_tok: n,
                n_ll: v[1].len(),
                n_off: v[2].len(),
                n_esc: v[3].len(),
                bits,
                delta,
                flat,
            };
            let mut sw = st0;
            if flat {
                exec_flat_go(&mut want, &lits, &c, 0, n, [0; 4], &mut sw);
            } else {
                exec_go(&mut want, &lits, &c, n, &mut sw);
            }
            let p: Vec<*const u8> = (0..5)
                .map(|k| {
                    let g = guards[k].slice(v[k].len(), true);
                    g.copy_from_slice(&v[k]);
                    g.as_ptr()
                })
                .collect();
            let f: ExecFlatRawFn = match (delta, flat) {
                // SAFETY (both): as for the call below; pos is 0.
                (false, false) => |a, b, c, d, e, f, g, h, i, _, k| unsafe {
                    exec_avx512(a, b, c, d, e, f, g, h, i, k)
                },
                (true, false) => |a, b, c, d, e, f, g, h, i, _, k| unsafe {
                    exec_delta_avx512(a, b, c, d, e, f, g, h, i, k)
                },
                (false, true) => exec_flat_avx512,
                (true, true) => exec_flat_delta_avx512,
            };
            let mut sg = st0;
            if k0 > 0 {
                exec_flat_go(&mut got, &lits, &c, 0, k0, [0; 4], &mut sg);
            }
            // SAFETY: the CPU has AVX512; the streams hold exactly the values of the n
            // operations, k0 and cur are stream positions inside them, raw holds their bits,
            // got and lits have SLACK bytes past them.
            unsafe {
                f(
                    got.as_mut_ptr(),
                    lits.as_ptr(),
                    p[0].add(k0),
                    p[1].add(cur[0]),
                    p[2].add(cur[1]),
                    p[4],
                    v[4].len(),
                    p[3].add(cur[2]),
                    n - k0,
                    cur[3],
                    &mut sg,
                )
            };
            assert!(
                sw == sg && want[..sw[0]] == got[..sw[0]],
                "case {i} (delta {delta}, flat {flat}, operations {k0} to {n}): state {sg:?}, want {sw:?}"
            );
        }
    }
}
