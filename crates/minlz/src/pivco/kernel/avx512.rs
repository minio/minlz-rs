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

//! The AVX-512 tier: intrinsics ports of the assembly Go's `_generate/pivco` generates
//! (`enc_avx512.go`, `dec_avx512.go` with `dec.go`, `hist_avx512.go`), with the slice
//! contracts of `internal/pivco/kernel/avx512_amd64.go`. They port the AVX-512 primitives of
//! PivCo-Huffman (`pivco_huffman_primitives_avx512.h`, `pivco_huffman_x86_vertical.h`,
//! `pivco_huffman_avx512_pack.h`, `pivco_huffman_avx512_flat.h`).
//!
//! Every entry point is an `unsafe fn` whose caller has checked
//! [`PIVCO_AVX512`](crate::cpu::PIVCO_AVX512): AVX512 F/BW/VL/VBMI/VBMI2/VPOPCNTDQ (upstream's
//! requirement) plus BMI2/POPCNT for the scalar glue. [`histogram`] also needs
//! [`PIVCO_HIST_AVX512`](crate::cpu::PIVCO_HIST_AVX512) (GFNI, BITALG).
//!
//! Reads of src and ranks and stores to `ranks[..n]` and the packed region are exact;
//! right-side and bitmap stores of the partitions use [`RANK_SLACK`]. The decoder checks each
//! op before running it, reads the input and writes exactly (masked tails), and its merges
//! load 64 bytes at each child cursor, which stays inside the op's K-byte child, so they read
//! at most 63 bytes past it ([`ARENA_SLACK`]).
//!
//! Zen 5 is sensitive to code shape beyond instruction counts: [`reg`] keeps merge sources in
//! registers, [`disp_base`] gives the merge loads a displacement, and the partitions count
//! their loop down.

use core::arch::asm;
use core::arch::x86_64::*;

use super::generic_pack::pack_generic;
use super::{
    ARENA_SLACK, FLAG_DST, HIST_SCRATCH, OP_CC, OP_COPY, OP_CV, OP_NAT, OP_V128, OP_V512, OP_VV,
    Op, Program, R2S_LEN, RANK_SLACK, vert_split,
};

// The partitions store 64 bytes at right+nr, which can end 63 bytes past right[n], and 8
// bitmap bytes per 64 ranks.
const _: () = assert!(RANK_SLACK >= 63);

// hist rounds scratch up to 64 bytes and puts four 16 KiB bins there.
const _: () = assert!(HIST_SCRATCH >= 4 * BIN_SIZE + 63);

/// Returns the mask of the low `n` lanes, all of them for 64 <= n < 256 (BZHI).
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
fn low(n: usize) -> u64 {
    _bzhi_u64(u64::MAX, n as u32)
}

/// Loads a 64-byte table.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
fn ld(t: &[u8; 64]) -> __m512i {
    // SAFETY: t has 64 bytes.
    unsafe { _mm512_loadu_si512(t.as_ptr().cast()) }
}

/// # Safety
/// p has 8 readable bytes.
#[inline]
unsafe fn read64(p: *const u8) -> u64 {
    // SAFETY: the caller's; no alignment is required.
    unsafe { p.cast::<u64>().read_unaligned() }
}

/// Keeps `v` in a register. Zen 4/5 give the zero-masking forms of VPCOMPRESSB and VPEXPANDB
/// a dependency on the old destination (upstream issue #11, 46-54% on part_full/part_right),
/// and their memory forms are slow (vv 636 ns instead of 442 in dec_avx512.go's A/B), so they
/// merge into a register that holds a loaded, broadcast or freshly zeroed vector. LLVM folds a
/// zero merge source into the zero-masking form and a single-use load into the memory form;
/// this empty asm hides both from it. The same goes for the partitions' VPCMPUB, whose memory
/// form made part_none 1.25x Go's time on a Ryzen 9 9950X.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
fn reg(mut v: __m512i) -> __m512i {
    // SAFETY: the template is empty; it neither reads nor writes memory.
    unsafe { asm!("/* {0} */", inout(zmm_reg) v, options(pure, nomem, nostack, preserves_flags)) };
    v
}

/// # Safety
/// p has 256 readable bytes.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn load4(p: *const u8) -> [__m512i; 4] {
    // SAFETY: the caller's.
    unsafe {
        [
            _mm512_loadu_si512(p.cast()),
            _mm512_loadu_si512(p.add(64).cast()),
            _mm512_loadu_si512(p.add(128).cast()),
            _mm512_loadu_si512(p.add(192).cast()),
        ]
    }
}

/// Maps the bytes of c through the 256-entry table t: each VPERMI2B covers 128 entries
/// indexed by the low 7 bits; bit 7 picks the half.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
fn lookup256(c: __m512i, t: &[__m512i; 4]) -> __m512i {
    let hib = _mm512_movepi8_mask(c);
    let lo = _mm512_permutex2var_epi8(t[0], c, t[1]);
    let hi = _mm512_permutex2var_epi8(t[2], c, t[3]);
    _mm512_mask_blend_epi8(hib, lo, hi)
}

/// Sets `ranks[i] = s2r[src[i]]` and returns the largest rank written (Go `gatherAsm`; upstream
/// init_avx512).
///
/// # Safety
/// The CPU has [`PIVCO_AVX512`](crate::cpu::PIVCO_AVX512).
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
pub(super) unsafe fn gather(ranks: &mut [u8], src: &[u8], s2r: &[u8; 256]) -> u8 {
    let n = src.len();
    if n == 0 {
        return 0;
    }
    let (rp, sp) = (ranks[..n].as_mut_ptr(), src.as_ptr());
    // SAFETY: s2r has 256 bytes.
    let t = unsafe { load4(s2r.as_ptr()) };
    let mut mx = _mm512_setzero_si512();
    // Two vectors per iteration with one VPMAXUB into mx: a VPMAXUB chain through every
    // vector bounds the loop (about 18% at n=32768).
    let mut i = 0;
    while i + 128 <= n {
        // SAFETY: src and ranks hold n >= i+128 bytes.
        unsafe {
            let r0 = lookup256(_mm512_loadu_si512(sp.add(i).cast()), &t);
            let r1 = lookup256(_mm512_loadu_si512(sp.add(i + 64).cast()), &t);
            _mm512_storeu_si512(rp.add(i).cast(), r0);
            _mm512_storeu_si512(rp.add(i + 64).cast(), r1);
            mx = _mm512_max_epu8(_mm512_max_epu8(r0, r1), mx);
        }
        i += 128;
    }
    if i + 64 <= n {
        // SAFETY: src and ranks hold n >= i+64 bytes.
        unsafe {
            let r = lookup256(_mm512_loadu_si512(sp.add(i).cast()), &t);
            mx = _mm512_max_epu8(r, mx);
            _mm512_storeu_si512(rp.add(i).cast(), r);
        }
        i += 64;
    }
    if i < n {
        let kv = low(n - i);
        // SAFETY: kv selects src[i..n] and ranks[i..n].
        unsafe {
            let r = lookup256(_mm512_maskz_loadu_epi8(kv, sp.add(i).cast()), &t);
            mx = _mm512_mask_max_epu8(mx, kv, r, mx);
            _mm512_mask_storeu_epi8(rp.add(i).cast(), kv, r);
        }
    }
    let y = _mm256_max_epu8(
        _mm512_extracti64x4_epi64::<1>(mx),
        _mm512_castsi512_si256(mx),
    );
    let mut x = _mm_max_epu8(_mm256_extracti32x4_epi32::<1>(y), _mm256_castsi256_si128(y));
    x = _mm_max_epu8(_mm_srli_si128::<8>(x), x);
    x = _mm_max_epu8(_mm_srli_si128::<4>(x), x);
    x = _mm_max_epu8(_mm_srli_si128::<2>(x), x);
    x = _mm_max_epu8(_mm_srli_si128::<1>(x), x);
    _mm_cvtsi128_si32(x) as u8
}

/// The cursors of a partition (Go `genPart`).
struct Part {
    vt: __m512i,
    right: *mut u8,
    left: *mut u8,
    bm: *mut u8,
    nr: usize,
    nl: usize,
}

impl Part {
    /// Partitions one vector v whose valid lanes are kv, rem of them: writes its 8 bitmap
    /// bytes, compacts the right ranks to right+nr (RIGHT) and the left ranks in place to
    /// left+nl (LEFT).
    ///
    /// # Safety
    /// The CPU has the tier's features; bm has 8 writable bytes, right 64 at nr and left
    /// rem at nl.
    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn step<const RIGHT: bool, const LEFT: bool>(
        &mut self,
        v: __m512i,
        kv: u64,
        rem: usize,
    ) {
        let k = _mm512_mask_cmpgt_epu8_mask(kv, v, self.vt);
        // SAFETY: the caller's.
        unsafe { self.bm.cast::<u64>().write_unaligned(k) };
        self.bm = self.bm.wrapping_add(8);
        let p = k.count_ones() as usize;
        if RIGHT {
            let r = _mm512_mask_compress_epi8(v, k, v);
            // SAFETY: the caller's.
            unsafe { _mm512_storeu_si512(self.right.add(self.nr).cast(), r) };
        }
        if LEFT {
            let l = _mm512_mask_compress_epi8(v, !k & kv, v);
            // In place: the store ends at or before the end of the lanes just loaded.
            // SAFETY: the caller's.
            unsafe { _mm512_mask_storeu_epi8(self.left.add(self.nl).cast(), kv, l) };
            self.nl += rem - p;
        }
        self.nr += p;
    }
}

/// Partitions src[..n], n > 0, by rank > thr; see [`super::Tier::part_full`] (Go `genPart`;
/// upstream part_{full,right,none}_avx512). RIGHT compacts the right ranks to right, LEFT the
/// left ranks in place to left == src.
///
/// # Safety
/// The CPU has the tier's features; src has n readable bytes (writable for LEFT), right
/// n+RANK_SLACK writable (RIGHT) and bm ceil(n/8)+RANK_SLACK writable.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn part<const RIGHT: bool, const LEFT: bool>(
    src: *const u8,
    right: *mut u8,
    left: *mut u8,
    bm: *mut u8,
    n: usize,
    thr: u8,
) -> usize {
    let vt = _mm512_set1_epi8(thr as i8);
    let mut p = Part {
        vt,
        right,
        left,
        bm,
        nr: 0,
        nl: 0,
    };
    let (mut s, rem) = (src, n & 63);
    // Counted down, as Go's SUBQ/JAE: LLVM compiles `while rem >= 64` to ADD, CMP and JA,
    // and that loop made part_none 1.24 times Go's time on a Ryzen 9 9950X (0.96 counted).
    for _ in 0..n >> 6 {
        // SAFETY: the 64 bytes at s are inside src[..n]; nr, nl are at most the bytes
        // before s, so the stores fit the caller's slices.
        unsafe {
            p.step::<RIGHT, LEFT>(reg(_mm512_loadu_si512(s.cast())), u64::MAX, 64);
            s = s.add(64);
        }
    }
    if rem > 0 {
        let kv = low(rem);
        // SAFETY: kv selects the rem bytes left at s; as above.
        unsafe { p.step::<RIGHT, LEFT>(_mm512_maskz_loadu_epi8(kv, s.cast()), kv, rem) };
    }
    p.nr
}

/// Partitions `ranks` by rank > thr; see [`super::Tier::part_full`] (Go `partFullAsm`).
///
/// # Safety
/// The CPU has [`PIVCO_AVX512`](crate::cpu::PIVCO_AVX512).
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
pub(super) unsafe fn part_full(
    ranks: &mut [u8],
    right: &mut [u8],
    bm: &mut [u8],
    thr: u8,
) -> usize {
    let n = ranks.len();
    if n == 0 {
        return 0;
    }
    let (right, bm) = (
        &mut right[..n + RANK_SLACK],
        &mut bm[..n.div_ceil(8) + RANK_SLACK],
    );
    let r = ranks.as_mut_ptr();
    // SAFETY: the slices have the lengths part needs.
    unsafe { part::<true, true>(r, right.as_mut_ptr(), r, bm.as_mut_ptr(), n, thr) }
}

/// Partitions `ranks` like [`part_full`] without compacting the left ranks (Go
/// `partRightAsm`).
///
/// # Safety
/// The CPU has [`PIVCO_AVX512`](crate::cpu::PIVCO_AVX512).
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
pub(super) unsafe fn part_right(ranks: &[u8], right: &mut [u8], bm: &mut [u8], thr: u8) -> usize {
    let n = ranks.len();
    if n == 0 {
        return 0;
    }
    let (right, bm) = (
        &mut right[..n + RANK_SLACK],
        &mut bm[..n.div_ceil(8) + RANK_SLACK],
    );
    let (r, b) = (right.as_mut_ptr(), bm.as_mut_ptr());
    // SAFETY: the slices have the lengths part needs; left is not used.
    unsafe { part::<true, false>(ranks.as_ptr(), r, core::ptr::null_mut(), b, n, thr) }
}

/// Writes only the bitmap of `ranks` and returns the number of set bits (Go
/// `partNoneAsm`).
///
/// # Safety
/// The CPU has [`PIVCO_AVX512`](crate::cpu::PIVCO_AVX512).
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
pub(super) unsafe fn part_none(ranks: &[u8], bm: &mut [u8], thr: u8) -> usize {
    let n = ranks.len();
    if n == 0 {
        return 0;
    }
    let bm = &mut bm[..n.div_ceil(8) + RANK_SLACK];
    let null = core::ptr::null_mut();
    // SAFETY: the slices have the lengths part needs; right and left are not used.
    unsafe { part::<false, false>(ranks.as_ptr(), null, null, bm.as_mut_ptr(), n, thr) }
}

/// Runs `body` for `s` = 0..8, unrolled: the vertical kernels need constant shift counts
/// and column indexes, and LLVM does not unroll the larger depths by itself.
macro_rules! unroll8 {
    ($s:ident => $body:block) => {
        unroll8!(@ $s $body 0 1 2 3 4 5 6 7)
    };
    (@ $s:ident $body:block $($v:literal)*) => {
        $({
            let $s: usize = $v;
            $body
        })*
    };
}

/// One row of a vertical block: zmm for the 64-lane (512-value) blocks, xmm for the 16-lane
/// (128-value) ones.
///
/// Every method needs the tier's CPU features; load and store need N valid bytes.
trait Lanes: Copy {
    /// Lanes per row.
    const N: usize;
    unsafe fn load(p: *const u8) -> Self;
    unsafe fn store(p: *mut u8, v: Self);
    unsafe fn splat(b: u8) -> Self;
    unsafe fn add8(a: Self, b: Self) -> Self;
    unsafe fn and(a: Self, b: Self) -> Self;
    unsafe fn sll16(a: Self, n: usize) -> Self;
    unsafe fn srl16(a: Self, n: usize) -> Self;
    unsafe fn tern<const IMM: i32>(a: Self, b: Self, c: Self) -> Self;
    /// Maps the codes in v (junk above bit D allowed) to symbols (Go `lookup`).
    unsafe fn lookup<const D: usize>(v: Self, t: &Lut) -> Self;
}

impl Lanes for __m512i {
    const N: usize = 64;

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: the caller's.
        unsafe { _mm512_loadu_si512(p.cast()) }
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn store(p: *mut u8, v: Self) {
        // SAFETY: the caller's.
        unsafe { _mm512_storeu_si512(p.cast(), v) }
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn splat(b: u8) -> Self {
        _mm512_set1_epi8(b as i8)
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn add8(a: Self, b: Self) -> Self {
        _mm512_add_epi8(a, b)
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn and(a: Self, b: Self) -> Self {
        _mm512_and_si512(a, b)
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn sll16(a: Self, n: usize) -> Self {
        _mm512_sll_epi16(a, _mm_cvtsi64_si128(n as i64))
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn srl16(a: Self, n: usize) -> Self {
        _mm512_srl_epi16(a, _mm_cvtsi64_si128(n as i64))
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn tern<const IMM: i32>(a: Self, b: Self, c: Self) -> Self {
        _mm512_ternarylogic_epi32::<IMM>(a, b, c)
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn lookup<const D: usize>(v: Self, t: &Lut) -> Self {
        lookup::<D>(v, t)
    }
}

impl Lanes for __m128i {
    const N: usize = 16;

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: the caller's.
        unsafe { _mm_loadu_si128(p.cast()) }
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn store(p: *mut u8, v: Self) {
        // SAFETY: the caller's.
        unsafe { _mm_storeu_si128(p.cast(), v) };
        // Keeps the 16-byte stores apart: LLVM merges the ones of the D >= 5 lookups, which
        // are zmm and ymm results, into wider stores with VINSERTI shuffles that compete with
        // the lookups (v128/d7 1.47x Go's time).
        // SAFETY: the template is empty.
        unsafe { asm!("", options(nostack, preserves_flags)) };
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn splat(b: u8) -> Self {
        _mm_set1_epi8(b as i8)
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn add8(a: Self, b: Self) -> Self {
        _mm_add_epi8(a, b)
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn and(a: Self, b: Self) -> Self {
        _mm_and_si128(a, b)
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn sll16(a: Self, n: usize) -> Self {
        _mm_sll_epi16(a, _mm_cvtsi64_si128(n as i64))
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn srl16(a: Self, n: usize) -> Self {
        _mm_srl_epi16(a, _mm_cvtsi64_si128(n as i64))
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn tern<const IMM: i32>(a: Self, b: Self, c: Self) -> Self {
        _mm_ternarylogic_epi32::<IMM>(a, b, c)
    }

    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn lookup<const D: usize>(v: Self, t: &Lut) -> Self {
        match D {
            6 | 7 => _mm512_castsi512_si128(lookup::<D>(_mm512_castsi128_si512(v), t)),
            5 => _mm256_castsi256_si128(_mm256_permutexvar_epi8(
                _mm256_castsi128_si256(v),
                _mm512_castsi512_si256(t.0[0]),
            )),
            _ => _mm_permutexvar_epi8(v, _mm512_castsi512_si128(t.0[0])),
        }
    }
}

/// Builds a `[[u8; 64]; N]` table at compile time: `t[k][j] = e`.
macro_rules! table {
    ($n:expr, |$k:ident, $j:ident| $e:expr) => {{
        let mut t = [[0u8; 64]; $n];
        let mut $k = 0;
        while $k < $n {
            let mut $j = 0;
            while $j < 64 {
                t[$k][$j] = $e;
                $j += 1;
            }
            $k += 1;
        }
        t
    }};
}

/// packShufD2: 4 VPERMB index vectors: codes k, k+4, ..., k+60.
static PACK_SHUF_D2: [[u8; 64]; 4] = table!(4, |k, j| if j < 16 { (k + 4 * j) as u8 } else { 0 });
/// packShufD4: 2 VPERMB index vectors: even, odd codes.
static PACK_SHUF_D4: [[u8; 64]; 2] = table!(2, |k, j| if j < 32 { (k + 2 * j) as u8 } else { 0 });
/// packCompact, indexed by D (3, 5, 6, 7): bytes 0..D-1 of each qword, then zero.
static PACK_COMPACT: [[u8; 64]; 8] = table!(8, |d, j| {
    if matches!(d, 3 | 5 | 6 | 7) && j < 8 * d {
        (j / d * 8 + j % d) as u8
    } else {
        0
    }
});
/// packD3: qwords: masks A..D, multishift controls A..D.
const PACK_D3: [u64; 8] = [
    0x0000000700000007,
    0x0000070000000700,
    0x0007000000070000,
    0x0700000007000000,
    0x0000000000081C00,
    0x0000000000292105,
    0x00000000002E120A,
    0x0000000000331700,
];

/// Packs the codes `ranks[i]-base` of a flat region in the vertical layout to exactly
/// `dst[..(n*d+7)/8]` (n = ranks.len()), with zero padding bits (Go `packAsm`; upstream
/// prim_enc_pack_dN).
///
/// # Safety
/// The CPU has [`PIVCO_AVX512`](crate::cpu::PIVCO_AVX512).
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
pub(super) unsafe fn pack(dst: &mut [u8], ranks: &[u8], d: u8, base: u8) {
    let n = ranks.len();
    if n == 0 || !(2..=8).contains(&d) {
        return pack_generic(dst, ranks, d, base);
    }
    let dst = &mut dst[..(n * usize::from(d)).div_ceil(8)];
    let (n512, n128) = vert_split(n, d);
    let (o, s) = (dst.as_mut_ptr(), ranks.as_ptr());
    // SAFETY: dst holds the packed region of the n codes in ranks.
    unsafe {
        match d {
            2 => pack_d::<2>(o, s, n, n512, n128, base),
            3 => pack_d::<3>(o, s, n, n512, n128, base),
            4 => pack_d::<4>(o, s, n, n512, n128, base),
            5 => pack_d::<5>(o, s, n, n512, n128, base),
            6 => pack_d::<6>(o, s, n, n512, n128, base),
            7 => pack_d::<7>(o, s, n, n512, n128, base),
            _ => pack_d::<8>(o, s, n, n512, n128, base),
        }
    }
}

/// Packs n512 codes in 512-value vertical blocks, the next n128 in 128-value blocks and the
/// rest in natural order. The rank-to-code subtraction is a VPADDB of -base, so the load
/// folds into it.
///
/// # Safety
/// The CPU has the tier's features; src has n readable bytes, dst (n*D+7)/8 writable.
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn pack_d<const D: usize>(
    mut dst: *mut u8,
    mut src: *const u8,
    n: usize,
    n512: usize,
    n128: usize,
    base: u8,
) {
    let neg = _mm512_set1_epi8(base.wrapping_neg() as i8);
    // SAFETY: the vertical spans end on a byte boundary, so the three parts are consecutive
    // in dst and src.
    unsafe {
        if D < 8 && n512 | n128 != 0 {
            vert_pack::<__m512i, D>(dst, src, n512 / 512, neg);
            (dst, src) = (dst.add(n512 * D / 8), src.add(n512));
            vert_pack::<__m128i, D>(dst, src, n128 / 128, _mm512_castsi512_si128(neg));
            (dst, src) = (dst.add(n128 * D / 8), src.add(n128));
        }
        nat_pack::<D>(dst, src, n - n512 - n128, neg);
    }
}

/// Packs `blocks` vertical blocks of W::N lanes (Go `vertical`; upstream
/// vert512_pack_zmm_d*, vert_pack_x86_d*). Code v of a block of 8*N values sits in lane v%N at
/// bits D*(v/N)...; lane l's D bytes are stored at a stride of N. Step s adds codes s*N.. to
/// the columns j = s*D/8 (and j+1 when the field spans two bytes).
///
/// # Safety
/// The CPU has the tier's features; src has blocks*8*N readable bytes, dst blocks*D*N
/// writable.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn vert_pack<W: Lanes, const D: usize>(
    mut dst: *mut u8,
    mut src: *const u8,
    blocks: usize,
    neg: W,
) {
    for _ in 0..blocks {
        // SAFETY: the caller's, for this block.
        unsafe {
            let mut cols = [neg; 8];
            unroll8!(s => {
                let (j, off) = ((s * D) >> 3, (s * D) & 7);
                let v = W::add8(W::load(src.add(s * W::N)), neg);
                if off == 0 {
                    cols[j] = v;
                } else {
                    // cols[j] |= v<<off & hi
                    let t = W::sll16(v, off);
                    cols[j] = W::tern::<0xF8>(cols[j], t, W::splat(0xFF << off));
                    if off + D > 8 {
                        cols[j + 1] = W::and(W::srl16(v, 8 - off), W::splat(0xFF >> (8 - off)));
                    }
                }
            });
            for (j, &c) in cols.iter().enumerate().take(D) {
                W::store(dst.add(j * W::N), c);
            }
            (dst, src) = (dst.add(D * W::N), src.add(8 * W::N));
        }
    }
}

/// The constants of [`group`] for depth D (Go `natural`).
#[derive(Clone, Copy)]
struct Group {
    /// D = 2, 4: the VPERMB index vectors; D = 3: the masks A..D; D = 5..7: the
    /// multiply-as-shift factors w0, w1 and the qword mask.
    a: [__m512i; 4],
    /// D = 3: the multishift controls A..D.
    c3: [__m512i; 4],
    /// D = 3, 5, 6, 7: [`PACK_COMPACT`].
    cmp: __m512i,
}

impl Group {
    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    fn of<const D: usize>() -> Group {
        let z = _mm512_setzero_si512();
        let mut g = Group {
            a: [z; 4],
            c3: [z; 4],
            cmp: z,
        };
        match D {
            2 => {
                for (a, t) in g.a.iter_mut().zip(&PACK_SHUF_D2) {
                    *a = ld(t);
                }
            }
            4 => {
                for (a, t) in g.a.iter_mut().zip(&PACK_SHUF_D4) {
                    *a = ld(t);
                }
            }
            3 => {
                for k in 0..4 {
                    g.a[k] = _mm512_set1_epi64(PACK_D3[k] as i64);
                    g.c3[k] = _mm512_set1_epi64(PACK_D3[4 + k] as i64);
                }
                g.cmp = ld(&PACK_COMPACT[D]);
            }
            5..=7 => {
                g.cmp = ld(&PACK_COMPACT[D]);
                g.a[0] = _mm512_set1_epi16(((1 << D) << 8 | 1) as i16);
                g.a[1] = _mm512_set1_epi32((1 << (2 * D)) << 16 | 1);
                g.a[2] = _mm512_set1_epi64(((1u64 << (4 * D)) - 1) as i64);
            }
            _ => {}
        }
        g
    }
}

/// Packs the 64 codes in cb (zero past the valid ones) into the low 8*D bytes (Go `group`).
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
fn group<const D: usize>(g: &Group, cb: __m512i) -> __m512i {
    match D {
        2 => {
            let g0 = _mm512_permutexvar_epi8(g.a[0], cb);
            let g1 = _mm512_slli_epi32::<2>(_mm512_permutexvar_epi8(g.a[1], cb));
            let g2 = _mm512_slli_epi32::<4>(_mm512_permutexvar_epi8(g.a[2], cb));
            let g3 = _mm512_slli_epi32::<6>(_mm512_permutexvar_epi8(g.a[3], cb));
            _mm512_or_si512(_mm512_ternarylogic_epi32::<0xFE>(g0, g1, g2), g3)
        }
        4 => {
            let g0 = _mm512_permutexvar_epi8(g.a[0], cb);
            let g1 = _mm512_slli_epi32::<4>(_mm512_permutexvar_epi8(g.a[1], cb));
            _mm512_or_si512(g0, g1)
        }
        3 => {
            let g0 = _mm512_multishift_epi64_epi8(g.c3[0], _mm512_and_si512(g.a[0], cb));
            let g1 = _mm512_multishift_epi64_epi8(g.c3[1], _mm512_and_si512(g.a[1], cb));
            let g2 = _mm512_multishift_epi64_epi8(g.c3[2], _mm512_and_si512(g.a[2], cb));
            let g3 = _mm512_multishift_epi64_epi8(g.c3[3], _mm512_and_si512(g.a[3], cb));
            let o = _mm512_or_si512(_mm512_ternarylogic_epi32::<0xFE>(g0, g1, g2), g3);
            _mm512_permutexvar_epi8(g.cmp, o)
        }
        8 => cb,
        _ => {
            // ryg multiply-as-shift pyramid (pack_d5..7_avx512).
            let x = _mm512_maddubs_epi16(g.a[0], cb); // word = c[2i] + c[2i+1]<<D
            let x = _mm512_madd_epi16(x, g.a[1]); // dword = w[2i] + w[2i+1]<<2D
            let xs = _mm512_srl_epi64(x, _mm_cvtsi64_si128((32 - 4 * D) as i64));
            let x = _mm512_ternarylogic_epi64::<0xE4>(x, xs, g.a[2]); // qword = d[2i] | d[2i+1]<<4D
            _mm512_permutexvar_epi8(g.cmp, x)
        }
    }
}

/// Packs n codes in natural order, 64 per iteration and one masked tail (Go `natural`;
/// upstream pack_dN_avx512).
///
/// # Safety
/// The CPU has the tier's features; src has n readable bytes, dst (n*D+7)/8 writable.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn nat_pack<const D: usize>(
    mut dst: *mut u8,
    mut src: *const u8,
    mut n: usize,
    neg: __m512i,
) {
    if n == 0 {
        return;
    }
    let g = Group::of::<D>();
    let ks = low(8 * D);
    while n >= 64 {
        // SAFETY: 64 codes remain in src and 8*D bytes in dst.
        unsafe {
            let p = group::<D>(&g, _mm512_add_epi8(_mm512_loadu_si512(src.cast()), neg));
            match D {
                2 => _mm_storeu_si128(dst.cast(), _mm512_castsi512_si128(p)),
                4 => _mm256_storeu_si256(dst.cast(), _mm512_castsi512_si256(p)),
                8 => _mm512_storeu_si512(dst.cast(), p),
                _ => _mm512_mask_storeu_epi8(dst.cast(), ks, p),
            }
            (dst, src) = (dst.add(8 * D), src.add(64));
        }
        n -= 64;
    }
    if n > 0 {
        let kv = low(n);
        // SAFETY: kv selects the n codes left in src, and the store mask the ceil(n*D/8)
        // bytes left in dst.
        unsafe {
            // Zero lanes give zero padding bits.
            let cb = _mm512_maskz_add_epi8(kv, _mm512_maskz_loadu_epi8(kv, src.cast()), neg);
            let p = group::<D>(&g, cb);
            let k = if D == 8 { kv } else { low((n * D).div_ceil(8)) };
            _mm512_mask_storeu_epi8(dst.cast(), k, p);
        }
    }
}

/// decPeriodic: `PERIODIC[D-2][i] = i & (2^D-1)`; VPERMB with it repeats a 2^D-entry table
/// over 64 bytes, so lookups ignore the junk above bit D of an unmasked code.
static PERIODIC: [[u8; 64]; 4] = table!(4, |d, i| (i & ((4 << d) - 1)) as u8);
/// decNatShuf: qword k of the gather holds bytes D*k..D*k+7 = codes 8k..8k+7 at bit offsets
/// 0, D, ..., 7D (natural layout), indexed by D-2.
static NAT_SHUF: [[u8; 64]; 6] = table!(6, |d, i| ((d + 2) * (i / 8) + i % 8) as u8);
/// decNatCtrl: the VPMULTISHIFTQB controls that move those offsets to the bottom of each
/// byte (upstream pivco_huffman_avx512_flat.h flat_dN_unpack64), indexed by D-2.
const NAT_CTRL: [u64; 6] = {
    let mut c = [0; 6];
    let mut d = 0;
    while d < 6 {
        let mut k = 0;
        while k < 8 {
            c[d] |= (((d + 2) * k) as u64) << (8 * k);
            k += 1;
        }
        d += 1;
    }
    c
};

/// The flat lookup table at &R2S[P] (Go `table`, Z10..Z13): for D <= 5 in `0[0]`, repeated
/// every 2^D bytes so that VPERMB ignores code bits >= D; D = 6: `0[0]`; D = 7: `0[..2]`
/// (VPERMI2B ignores bit 7); D = 8: all four. Reads are exact.
#[derive(Clone, Copy)]
struct Lut([__m512i; 4]);

impl Lut {
    /// # Safety
    /// The CPU has the tier's features; t has 2^D readable bytes.
    #[inline]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
    )]
    unsafe fn at<const D: usize>(t: *const u8) -> Lut {
        let mut l = Lut([_mm512_setzero_si512(); 4]);
        if D <= 5 {
            // SAFETY: the mask selects the 2^D bytes.
            let v = unsafe { _mm512_maskz_loadu_epi8(low(1 << D), t.cast()) };
            l.0[0] = _mm512_permutexvar_epi8(ld(&PERIODIC[D - 2]), v);
        } else {
            for (i, v) in l.0.iter_mut().enumerate().take((1 << D) / 64) {
                // SAFETY: 64*i+64 <= 2^D.
                *v = unsafe { _mm512_loadu_si512(t.add(64 * i).cast()) };
            }
        }
        l
    }
}

/// Maps the 64 codes in v (junk above bit D allowed) to symbols (Go `lookup`): D = 7 with
/// VPERMI2B, which ignores bit 7, else VPERMB over the repeated table.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
fn lookup<const D: usize>(v: __m512i, t: &Lut) -> __m512i {
    if D == 7 {
        _mm512_permutex2var_epi8(t.0[0], v, t.0[1])
    } else {
        _mm512_permutexvar_epi8(v, t.0[0])
    }
}

/// The buffers of a [`Program`]. In-place merges alias their children, so every arena pointer
/// derives from one `arena.as_mut_ptr()`.
struct Bufs<'a> {
    arena: *mut u8,
    /// max(len(arena) - ARENA_SLACK, 0).
    alim: usize,
    dst: *mut u8,
    dst_len: usize,
    input: &'a [u8],
    r2s: Option<&'a [u8; R2S_LEN]>,
}

/// Runs `p.ops`, checking each op before it runs, so a malformed op fails closed (Go
/// `execAVX512`, `execAVX512Asm`). Returns the index of the first op that fails its checks.
///
/// # Safety
/// The CPU has [`PIVCO_AVX512`](crate::cpu::PIVCO_AVX512).
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
pub(super) unsafe fn exec(p: &mut Program<'_>) -> Result<(), usize> {
    let b = Bufs {
        arena: p.arena.as_mut_ptr(),
        alim: p.arena.len().saturating_sub(ARENA_SLACK),
        dst: p.dst.as_mut_ptr(),
        dst_len: p.dst.len(),
        input: p.input,
        r2s: p.r2s,
    };
    for (i, op) in p.ops.iter().enumerate() {
        // SAFETY: b holds p's buffers; the caller's features.
        if !unsafe { exec_op(&b, op) } {
            return Err(i);
        }
    }
    Ok(())
}

/// Checks op against the buffers and runs it; returns false if it fails a check.
///
/// # Safety
/// The CPU has the tier's features; b describes live buffers.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn exec_op(b: &Bufs<'_>, op: &Op) -> bool {
    let (k, d, bm) = (op.k as usize, usize::from(op.d), op.bm as usize);
    // Output: Out+K must fit Dst, or the arena less its slack.
    let (o, olim) = if op.flags & FLAG_DST != 0 {
        (b.dst, b.dst_len)
    } else {
        (b.arena, b.alim)
    };
    if op.out as usize + k > olim {
        return false;
    }
    // SAFETY: Out+K is inside the output buffer.
    let o = unsafe { o.add(op.out as usize) };
    // The n source bytes at Bm, if they are inside the input.
    let input = |n: usize| b.input.get(bm..bm + n).map(<[u8]>::as_ptr);
    // R2S[P..P+2^D], D <= 8, if R2S is set and holds it.
    let lut = || {
        let p = usize::from(op.p);
        b.r2s
            .and_then(|t| t.get(p..p + (1 << d)))
            .map(<[u8]>::as_ptr)
    };
    match op.kind {
        OP_VV | OP_CV => {
            let (vv, l, r) = (op.kind == OP_VV, op.l as usize, op.r as usize);
            let Some(bmp) = input(k.div_ceil(8)) else {
                return false;
            };
            if (vv && l + k > b.alim) || r + k > b.alim {
                return false;
            }
            // SAFETY: the output, the bitmap and the K-byte children are in range, and
            // ARENA_SLACK covers the 64-byte loads at the child cursors.
            let end = unsafe {
                if vv {
                    merge::<true>(o, k, bmp, b.arena, l, r, 0)
                } else {
                    merge::<false>(o, k, bmp, b.arena, 0, r, op.p)
                }
            };
            end - r == op.kr as usize
        }
        OP_CC => {
            let Some(bmp) = input(k.div_ceil(8)) else {
                return false;
            };
            // SAFETY: the output and the bitmap are in range.
            unsafe { merge_cc(o, k, bmp, op.p, op.q) };
            true
        }
        OP_V512 | OP_V128 => {
            let wide = op.kind == OP_V512;
            let blk = if wide { 512 } else { 128 };
            if !(2..=7).contains(&d) || k % blk != 0 {
                return false;
            }
            let (Some(src), Some(t)) = (input(k * d / 8), lut()) else {
                return false;
            };
            // SAFETY: the output, the region and the table are in range.
            unsafe {
                if wide {
                    vert::<__m512i>(o, src, k / blk, d, t)
                } else {
                    vert::<__m128i>(o, src, k / blk, d, t)
                }
            };
            true
        }
        OP_NAT => {
            if !(2..=8).contains(&d) {
                return false;
            }
            let (Some(src), Some(t)) = (input((k * d).div_ceil(8)), lut()) else {
                return false;
            };
            // SAFETY: the output, the region and the table are in range.
            unsafe { nat(o, src, k, d, t) };
            true
        }
        OP_COPY => {
            if d != 8 {
                return false;
            }
            let Some(src) = input(k) else {
                return false;
            };
            // SAFETY: the output and the region are in range.
            unsafe { copy(o, src, k) };
            true
        }
        _ => false,
    }
}

/// Loads the bitmap's last ceil(rem/8) bytes exactly; returns its rem valid bits and the
/// mask of rem lanes, 0 < rem < 64 (Go `tailMasks`).
///
/// # Safety
/// The CPU has the tier's features; bm has ceil(rem/8) readable bytes.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn tail_bits(bm: *const u8, rem: usize) -> (u64, u64) {
    // SAFETY: the mask selects the caller's bytes.
    let b = unsafe { _mm_maskz_loadu_epi8(low(rem.div_ceil(8)) as u16, bm.cast()) };
    let kv = low(rem);
    (_mm_cvtsi128_si64(b) as u64 & kv, kv)
}

/// What the merge loads add to their cursors, on the base [`disp_base`] lowers by it.
const DISP: usize = 64;

/// Returns `a - DISP` through an empty asm, which LLVM can't see through, so the merge loads
/// address the arena as base + cursor + DISP instead of folding DISP back into the base. Zen 5
/// runs these loads faster with a displacement: Go's CV loop as inline asm took 283 ns with
/// `[R13+R12+0]` (R13 needs one, as in Go's binary) or `[R8+R12+64]`, 309 ns with `[R8+R12]`;
/// the merges went from 0.88 (CV) and 0.95 (VV) of Go's speed to 0.99 and 1.0.
#[inline(always)]
#[allow(
    clippy::pointers_in_nomem_asm_block,
    reason = "the asm only passes the pointer through a register"
)]
fn disp_base(a: *const u8) -> *const u8 {
    let mut b = a.wrapping_sub(DISP);
    // SAFETY: the template is empty; it neither reads nor writes memory.
    unsafe { asm!("/* {0} */", inout(reg) b, options(pure, nomem, nostack, preserves_flags)) };
    b
}

/// Returns 64 merged symbols: R's next ones in the lanes of t, and L's next ones (VV) or c
/// (CV) in the lanes of nt (Go `expand`).
///
/// # Safety
/// The CPU has the tier's features; b is [`disp_base`] of the arena, which has 64 readable
/// bytes at r, and at l for VV.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn expand<const VV: bool>(
    t: u64,
    nt: u64,
    b: *const u8,
    l: usize,
    r: usize,
    c: __m512i,
) -> __m512i {
    // SAFETY: the caller's; b + DISP is the arena.
    unsafe {
        let z = if VV {
            let lv = reg(_mm512_loadu_si512(b.wrapping_add(DISP + l).cast()));
            _mm512_mask_expand_epi8(reg(_mm512_setzero_si512()), nt, lv)
        } else {
            c
        };
        _mm512_mask_expand_epi8(
            z,
            t,
            reg(_mm512_loadu_si512(b.wrapping_add(DISP + r).cast())),
        )
    }
}

/// One full 64-symbol step of [`merge`] (Go `vvStep`, `cvStep`); a is [`disp_base`] of the
/// arena.
///
/// # Safety
/// As for [`expand`]; bm has 8 readable bytes and out 64 writable.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn step<const VV: bool>(
    out: *mut u8,
    bm: *const u8,
    a: *const u8,
    l: &mut usize,
    r: &mut usize,
    c: __m512i,
) {
    // SAFETY: the caller's.
    unsafe {
        let t = read64(bm);
        _mm512_storeu_si512(out.cast(), expand::<VV>(t, !t, a, *l, *r, c));
        let p = t.count_ones() as usize;
        (*l, *r) = (*l + 64 - p, *r + p);
    }
}

/// Merges k symbols, `out[j] = bit j ? R[r++] : L[l++]` (VV, upstream
/// merge_vec_vec_avx512) or `bit j ? R[r++] : c` (CV, merge_cst_vec_avx512), and returns the
/// right cursor (Go `mergeVV`, `mergeCV`, `mergeLoop`). The full steps run two per iteration.
///
/// # Safety
/// The CPU has the tier's features; out has k writable bytes and bm ceil(k/8) readable; the
/// arena at a has k+63 readable bytes at r, and at l for VV. Each step loads its children
/// before it stores, so out may be one of them (in place).
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn merge<const VV: bool>(
    mut out: *mut u8,
    k: usize,
    mut bm: *const u8,
    a: *const u8,
    mut l: usize,
    mut r: usize,
    c: u8,
) -> usize {
    let (c, a) = (_mm512_set1_epi8(c as i8), disp_base(a));
    // SAFETY: the steps cover out[..k] and bm[..ceil(k/8)]; the cursors stay below the end
    // of their child, so the 64-byte loads stay inside k+63 bytes.
    unsafe {
        for _ in 0..k >> 7 {
            step::<VV>(out, bm, a, &mut l, &mut r, c);
            step::<VV>(out.add(64), bm.add(8), a, &mut l, &mut r, c);
            (out, bm) = (out.add(128), bm.add(16));
        }
        if k & 64 != 0 {
            step::<VV>(out, bm, a, &mut l, &mut r, c);
            (out, bm) = (out.add(64), bm.add(8));
        }
        let rem = k & 63;
        if rem != 0 {
            let (t, kv) = tail_bits(bm, rem);
            _mm512_mask_storeu_epi8(out.cast(), kv, expand::<VV>(t, !t & kv, a, l, r, c));
            r += t.count_ones() as usize;
        }
    }
    r
}

/// Writes `out[j] = bit j ? q : p` (Go `mergeCC`; upstream merge_cst_cst_avx512).
///
/// # Safety
/// The CPU has the tier's features; out has k writable bytes and bm ceil(k/8) readable.
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn merge_cc(mut out: *mut u8, k: usize, mut bm: *const u8, p: u8, q: u8) {
    let (pv, qv) = (_mm512_set1_epi8(p as i8), _mm512_set1_epi8(q as i8));
    // SAFETY: the steps cover out[..k] and bm[..ceil(k/8)].
    unsafe {
        for _ in 0..k >> 6 {
            _mm512_storeu_si512(out.cast(), _mm512_mask_blend_epi8(read64(bm), pv, qv));
            (out, bm) = (out.add(64), bm.add(8));
        }
        let rem = k & 63;
        if rem != 0 {
            let (t, kv) = tail_bits(bm, rem);
            _mm512_mask_storeu_epi8(out.cast(), kv, _mm512_mask_blend_epi8(t, pv, qv));
        }
    }
}

/// Unpacks `blocks` > 0 vertical blocks of W::N lanes and depth d (Go `vertical`).
///
/// # Safety
/// The CPU has the tier's features; 2 <= d <= 7; src has blocks*N*d readable bytes, out
/// blocks*8*N writable, and t 2^d readable.
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn vert<W: Lanes>(o: *mut u8, src: *const u8, blocks: usize, d: usize, t: *const u8) {
    if blocks == 0 {
        return;
    }
    // SAFETY: the caller's.
    unsafe {
        match d {
            2 => vert_dec::<W, 2>(o, src, blocks, t),
            3 => vert_dec::<W, 3>(o, src, blocks, t),
            4 => vert_dec::<W, 4>(o, src, blocks, t),
            5 => vert_dec::<W, 5>(o, src, blocks, t),
            6 => vert_dec::<W, 6>(o, src, blocks, t),
            _ => vert_dec::<W, 7>(o, src, blocks, t),
        }
    }
}

/// Unpacks the vertical blocks (Go `vertBlock`; upstream vert512_merge_zmm_d* and
/// vert_merge_x86_d*). Lane l of a block holds its 8 codes in D column bytes at a stride of
/// N lanes; step s extracts bit field s*D of every lane, which is codes s*N .. s*N+N-1.
///
/// # Safety
/// As for [`vert`].
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn vert_dec<W: Lanes, const D: usize>(
    mut out: *mut u8,
    mut src: *const u8,
    blocks: usize,
    t: *const u8,
) {
    // SAFETY: the caller's.
    let t = unsafe { Lut::at::<D>(t) };
    for _ in 0..blocks {
        // SAFETY: the caller's, for this block.
        unsafe {
            let mut c = [W::splat(0); 8];
            for (j, c) in c.iter_mut().enumerate().take(D) {
                *c = W::load(src.add(j * W::N));
            }
            unroll8!(s => {
                let (j, off) = ((s * D) >> 3, (s * D) & 7);
                let v = if off == 0 {
                    c[j]
                } else if off + D <= 8 {
                    W::srl16(c[j], off)
                } else {
                    // Bits off..7 of column j, then bits 0..off+D-9 of column j+1: the 16-bit
                    // shifts leave the neighbor byte's bits where the other column's go, so
                    // select per bit (0xE4: mask ? a : b) with the mask 0xFF>>off.
                    let (a, b) = (W::srl16(c[j], off), W::sll16(c[j + 1], 8 - off));
                    W::tern::<0xE4>(a, b, W::splat(0xFF >> off))
                };
                W::store(out.add(s * W::N), W::lookup::<D>(v, &t));
            });
            (src, out) = (src.add(W::N * D), out.add(8 * W::N));
        }
    }
}

/// Decodes k codes of depth d in the natural layout (Go `natural`).
///
/// # Safety
/// The CPU has the tier's features; 2 <= d <= 8; src has ceil(k*d/8) readable bytes, out k
/// writable and t 2^d readable.
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn nat(o: *mut u8, src: *const u8, k: usize, d: usize, t: *const u8) {
    // SAFETY: the caller's.
    unsafe {
        match d {
            2 => nat_dec::<2>(o, src, k, t),
            3 => nat_dec::<3>(o, src, k, t),
            4 => nat_dec::<4>(o, src, k, t),
            5 => nat_dec::<5>(o, src, k, t),
            6 => nat_dec::<6>(o, src, k, t),
            7 => nat_dec::<7>(o, src, k, t),
            _ => nat_dec::<8>(o, src, k, t),
        }
    }
}

/// The natural-layout kernel for depth D: 64 codes per step from an exact masked load of
/// 8*D bytes; the last step loads ceil(rest*D/8) bytes and stores rest symbols (upstream
/// merge_flat_avx512 64-wide loops, without the fast/safe split).
///
/// # Safety
/// As for [`nat`].
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn nat_dec<const D: usize>(
    mut out: *mut u8,
    mut src: *const u8,
    mut k: usize,
    t: *const u8,
) {
    // SAFETY: the caller's.
    let t = unsafe { Lut::at::<D>(t) };
    let (shuf, ctrl) = if D < 8 {
        let i = D - 2;
        (ld(&NAT_SHUF[i]), _mm512_set1_epi64(NAT_CTRL[i] as i64))
    } else {
        (_mm512_setzero_si512(), _mm512_setzero_si512())
    };
    let kd = low(8 * D);
    // SAFETY: each step reads its 8*D region bytes (ceil(rest*D/8) in the tail) and writes
    // its 64 (rest) symbols.
    unsafe {
        while k >= 64 {
            let z = if D < 8 {
                _mm512_maskz_loadu_epi8(kd, src.cast())
            } else {
                _mm512_loadu_si512(src.cast())
            };
            _mm512_storeu_si512(out.cast(), nat_codes::<D>(z, shuf, ctrl, &t));
            (src, out, k) = (src.add(8 * D), out.add(64), k - 64);
        }
        if k > 0 {
            let kv = low(k);
            let kz = if D < 8 { low((k * D).div_ceil(8)) } else { kv };
            let z = _mm512_maskz_loadu_epi8(kz, src.cast());
            _mm512_mask_storeu_epi8(out.cast(), kv, nat_codes::<D>(z, shuf, ctrl, &t));
        }
    }
}

/// Turns the region bytes in z into 64 symbols (Go `natCodes`).
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
fn nat_codes<const D: usize>(z: __m512i, shuf: __m512i, ctrl: __m512i, t: &Lut) -> __m512i {
    if D < 8 {
        let z = _mm512_multishift_epi64_epi8(ctrl, _mm512_permutexvar_epi8(shuf, z));
        return lookup::<D>(z, t);
    }
    // 256 entries: two 128-entry VPERMI2B lookups, picked by bit 7.
    lookup256(z, &t.0)
}

/// D = 8 flat: the codes are the symbols (Go `copyFlat`; upstream merge_flat_d8_avx512).
///
/// # Safety
/// The CPU has the tier's features; src has k readable bytes and out k writable.
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt"
)]
unsafe fn copy(mut out: *mut u8, mut src: *const u8, mut k: usize) {
    // SAFETY: each step copies bytes that remain of the k.
    unsafe {
        while k >= 256 {
            let v = load4(src);
            for (i, &v) in v.iter().enumerate() {
                _mm512_storeu_si512(out.add(64 * i).cast(), v);
            }
            (src, out, k) = (src.add(256), out.add(256), k - 256);
        }
        while k >= 64 {
            _mm512_storeu_si512(out.cast(), _mm512_loadu_si512(src.cast()));
            (src, out, k) = (src.add(64), out.add(64), k - 64);
        }
        if k > 0 {
            let kv = low(k);
            _mm512_mask_storeu_epi8(out.cast(), kv, _mm512_maskz_loadu_epi8(kv, src.cast()));
        }
    }
}

// AVX-512 GFNI/BITALG byte histogram: port of histogram_chunk_avx512 and
// hist_consume_bin_avx512 (pivco_huffman_primitives_avx512.h), which port Harold Aptroot's
// histogram (MIT, https://gitlab.com/-/snippets/3745720).
// Copyright (c) 2026 Harold Aptroot; MIT License, see LICENSE-histogram.
//
// Bytes are binned by their top two bits into four 16 KiB buffers with VPCOMPRESSB; the
// low six bits, permuted by one GF2P8AFFINEQB, select one of 64 one-bit counters per qword
// (VPSLLVQ of 1). Full-adder trees of VPTERNLOGD promote the counters, a VPERMB +
// GF2P8AFFINEQB 8x8 bit transpose and VPOPCNTB turn the bit planes into byte counts.
//
// Differences from C:
//   - The input is walked in 64-byte aligned rows; a partial first and last row are masked
//     loads counted at the start of the first round (exact reads, no scalar head or tail).
//     Masked-out lanes load as zero and are removed from the low bin with the same mask.
//   - Full rows run in chunks short enough that no bin can fill up, instead of testing
//     the four bin counts after every row.
//   - Each chunk starts in a loop for rows whose bytes are all below 0x40 (one bin, no
//     compress), and moves on to a loop for rows below 0x80 (two bins) and then to the
//     four-bin loop at the first row that does not fit. Skewed and ASCII inputs bin at a
//     fraction of the four-bin cost; other inputs pay two mispredicted tests per chunk.
//   - VPCOMPRESSB merges into a copy of the row instead of zero-masking (Zen 4/5 give the
//     zero-masking form a dependency on the old destination). The lanes past the count
//     are dead: the next store overwrites them, and consume pads the last partial row
//     of a bin with 0xFF, whose shift count >= 64 counts nothing.
//   - The two- and four-bin loops load the next row before they store the current one's
//     bins. C loads it after them; on a Ryzen 9 9950X (Zen 5) the load then often waits
//     for those stores' addresses, which depend on the current row's counts, and the
//     four-bin loop runs at the latency of that chain. Whether it waits depends on where
//     input and scratch sit in memory, so C's histogram is bimodal from run to run. A
//     binning-only C loop over 1.1 MiB of JPEG took 2.24-2.36 ns per row at 48 scratch
//     placements and 1.71-1.85 ns with the load moved up.
//   - The bin masks come from four byte compares of row&0xC0 instead of two VPMOVB2M and
//     mask register logic, whose throughput is low on Zen 5 (masks and counts of a row:
//     3.9 cycles instead of 5.2).

const BIN_SIZE: usize = 16 << 10;
/// A bin at this count is consumed before the next store.
const BIN_LIMIT: usize = BIN_SIZE - 64;
/// 65472 bytes: no u16 counter of a round can overflow.
const ROUND_ROWS: usize = 1023;
/// Go bounds one non-preemptible assembly call to about 70 µs with it. It is a whole number
/// of the kernel's 65472-byte rounds, so the input alignment of every chunk is the same.
const HIST_CHUNK: usize = 16 * 65472;

/// histTranspose: the VPERMB before the 8x8 bit transpose.
static HIST_TRANSPOSE: [u8; 64] = table!(1, |_k, i| (i % 8 * 8 + i / 8) as u8)[0];

/// The constants of the histogram.
struct HistConsts {
    aff: __m512i,
    gf: __m512i,
    one: __m512i,
    lo8: __m512i,
    tp: __m512i,
    ones: __m512i,
    top2: __m512i,
    b40: __m512i,
    b80: __m512i,
}

/// Adds the byte counts of src to h (Go `histogramAVX512`). scratch must hold
/// [`HIST_SCRATCH`] bytes.
///
/// # Safety
/// The CPU has [`PIVCO_AVX512`](crate::cpu::PIVCO_AVX512) and
/// [`PIVCO_HIST_AVX512`](crate::cpu::PIVCO_HIST_AVX512).
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt,gfni,avx512bitalg"
)]
pub(super) unsafe fn histogram(h: &mut [u32; 256], src: &[u8], scratch: &mut [u8]) {
    if src.is_empty() {
        return;
    }
    let scratch = &mut scratch[..HIST_SCRATCH];
    for c in src.chunks(HIST_CHUNK) {
        // SAFETY: the caller's features; c is not empty and scratch is long enough.
        unsafe { hist(h, c, scratch) };
    }
}

/// Adds the byte counts of src, which is not empty, to h (Go `histAsm`). Port of Harold
/// Aptroot's AVX-512 histogram (<https://gitlab.com/-/snippets/3745720>); MIT License, see
/// LICENSE-histogram.
///
/// # Safety
/// As for [`histogram`]; scratch has HIST_SCRATCH bytes.
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt,gfni,avx512bitalg"
)]
unsafe fn hist(h: &mut [u32; 256], src: &[u8], scratch: &mut [u8]) {
    let base = scratch.as_mut_ptr();
    // SAFETY: HIST_SCRATCH leaves 63 bytes for rounding up to 64.
    let base = unsafe { base.add(base.addr().wrapping_neg() & 63) };

    // The input is split in 64-byte aligned rows from src&^63; at most the first and the
    // last are partial and are counted with masked loads at the start of the first round.
    let off = src.as_ptr().addr() & 63;
    let span = src.len() + off; // from row
    let mut rows = span.div_ceil(64);
    let mut row = src.as_ptr().wrapping_sub(off);
    let (hm, tm) = (u64::MAX << off, low(((span - 1) & 63) + 1));
    // The partial rows: address, lane mask.
    let mut part = [(row, 0); 2];
    let mut npart = 0;
    if rows == 1 {
        part[0] = (row, hm & tm);
        (npart, rows) = (1, 0);
    } else {
        if off != 0 {
            part[0] = (row, hm);
            npart = 1;
            row = row.wrapping_add(64);
            rows -= 1;
        }
        if tm != u64::MAX {
            rows -= 1;
            part[npart] = (row.wrapping_add(rows * 64), tm);
            npart += 1;
        }
    }
    let last = row.wrapping_add(rows * 64).wrapping_sub(64);

    let c = HistConsts {
        aff: _mm512_set1_epi64(0x2010010204080000),
        gf: _mm512_set1_epi64(0x8040201008040201u64 as i64),
        one: _mm512_set1_epi64(1),
        lo8: _mm512_set1_epi16(0xFF),
        tp: ld(&HIST_TRANSPOSE),
        ones: _mm512_set1_epi8(-1),
        top2: _mm512_set1_epi8(0xC0u8 as i8),
        b40: _mm512_set1_epi8(0x40),
        b80: _mm512_set1_epi8(0x80u8 as i8),
    };
    // The [256]uint16 counters of the current round.
    let mut h16 = [0u16; 256];
    loop {
        h16.fill(0);
        let mut cnt = [0usize; 4];
        // The partial rows count against the first round.
        let mut rrem = (ROUND_ROWS - npart).min(rows);
        rows -= rrem;
        for &(pa, kv) in &part[..npart] {
            // SAFETY: kv selects the bytes of the row that are in src; the bins are empty
            // or hold one partial row.
            unsafe {
                bin4(
                    &c,
                    _mm512_maskz_loadu_epi8(kv, pa.cast()),
                    kv,
                    base,
                    &mut cnt,
                )
            };
        }
        npart = 0;

        // Run full rows in chunks short enough that no bin can overflow: a store at count c
        // needs c <= BIN_SIZE-64, and each row adds at most 64 to a bin.
        'recheck: while rrem > 0 {
            let mx = cnt.into_iter().fold(0, usize::max);
            if mx >= BIN_LIMIT {
                // SAFETY: the bins hold cnt <= BIN_SIZE bytes.
                unsafe { flush(&c, base, &mut cnt, BIN_LIMIT, &mut h16) };
                continue;
            }
            let mut left = (((BIN_LIMIT - mx) >> 6) + 1).min(rrem);
            rrem -= left;
            // The loops keep the row at row in d. The one-bin loop loads it; its store
            // addresses do not depend on the data. ahead loads the next row (the last full
            // one once there is none, for exact reads) before the current row's bins are
            // stored. A row that does not fit a loop is redone by the next one. The rows
            // are full rows of src and the bin counts stay at most BIN_LIMIT before a store.
            let mut d;
            loop {
                // SAFETY: row is a full, aligned row of src.
                d = unsafe { _mm512_load_si512(row.cast()) };
                if _mm512_test_epi8_mask(c.top2, d) != 0 {
                    break;
                }
                let x = _mm512_gf2p8affine_epi64_epi8::<0>(d, c.aff);
                // SAFETY: cnt[0] <= BIN_LIMIT.
                unsafe { _mm512_storeu_si512(base.add(cnt[0]).cast(), x) };
                cnt[0] += 64;
                row = row.wrapping_add(64);
                left -= 1;
                if left == 0 {
                    continue 'recheck;
                }
            }
            loop {
                // SAFETY: row and last are full rows of src.
                let nxt = unsafe { ahead(row, last) };
                if _mm512_movepi8_mask(d) != 0 {
                    break;
                }
                // SAFETY: cnt[0], cnt[1] <= BIN_LIMIT.
                unsafe { bin2(&c, d, base, &mut cnt) };
                d = nxt;
                row = row.wrapping_add(64);
                left -= 1;
                if left == 0 {
                    continue 'recheck;
                }
            }
            loop {
                // SAFETY: as above.
                let nxt = unsafe { ahead(row, last) };
                // SAFETY: cnt <= BIN_LIMIT.
                unsafe { bin4(&c, d, u64::MAX, base, &mut cnt) };
                d = nxt;
                row = row.wrapping_add(64);
                left -= 1;
                if left == 0 {
                    continue 'recheck;
                }
            }
        }
        // SAFETY: the bins hold cnt <= BIN_SIZE bytes.
        unsafe { flush(&c, base, &mut cnt, 1, &mut h16) };
        widen(h, &h16);
        if rows == 0 {
            return;
        }
    }
}

/// Loads the row after row, or the last full row once there is none (Go `ahead`).
///
/// # Safety
/// The CPU has the tier's features; row and last are full, aligned rows of the input.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt,gfni,avx512bitalg"
)]
unsafe fn ahead(row: *const u8, last: *const u8) -> __m512i {
    // SAFETY: the caller's.
    unsafe { _mm512_load_si512(row.wrapping_add(64).min(last).cast()) }
}

/// Bins the row d, whose valid lanes are kv, into the four bins (Go `body`).
///
/// # Safety
/// The CPU has the tier's features; every bin count is at most BIN_LIMIT.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt,gfni,avx512bitalg"
)]
unsafe fn bin4(c: &HistConsts, d: __m512i, kv: u64, base: *mut u8, cnt: &mut [usize; 4]) {
    let t = _mm512_and_si512(c.top2, d);
    let kc = [
        _mm512_mask_testn_epi8_mask(kv, t, t), // masked-out lanes load as zero
        _mm512_cmpeq_epi8_mask(c.b40, t),
        _mm512_cmpeq_epi8_mask(c.b80, t),
        _mm512_cmpeq_epi8_mask(c.top2, t),
    ];
    let d = _mm512_gf2p8affine_epi64_epi8::<0>(d, c.aff); // x = row, A = aff
    for (i, (&k, n)) in kc.iter().zip(cnt).enumerate() {
        // SAFETY: n <= BIN_LIMIT, so the 64 bytes are inside bin i.
        unsafe {
            let p = base.add(i * BIN_SIZE + *n);
            _mm512_storeu_si512(p.cast(), _mm512_mask_compress_epi8(d, k, d));
        }
        *n += k.count_ones() as usize;
    }
}

/// Bins the row d, whose bytes are all below 0x80, into bins 0 and 1 (Go `hist_row2`).
///
/// # Safety
/// The CPU has the tier's features; cnt[0] and cnt[1] are at most BIN_LIMIT.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt,gfni,avx512bitalg"
)]
unsafe fn bin2(c: &HistConsts, d: __m512i, base: *mut u8, cnt: &mut [usize; 4]) {
    let k6 = _mm512_test_epi8_mask(c.b40, d);
    let k0 = _mm512_testn_epi8_mask(c.b40, d);
    let d = _mm512_gf2p8affine_epi64_epi8::<0>(d, c.aff);
    // SAFETY: the caller's.
    unsafe {
        _mm512_storeu_si512(base.add(cnt[0]).cast(), _mm512_mask_compress_epi8(d, k0, d));
        let p1 = base.add(BIN_SIZE + cnt[1]);
        _mm512_storeu_si512(p1.cast(), _mm512_mask_compress_epi8(d, k6, d));
    }
    let p = k6.count_ones() as usize;
    cnt[1] += p;
    cnt[0] += 64 - p;
}

/// Consumes the bins holding at least thr bytes into h16 (Go `hist_flush`).
///
/// # Safety
/// The CPU has the tier's features; bin b holds cnt[b] <= BIN_SIZE bytes.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt,gfni,avx512bitalg"
)]
unsafe fn flush(
    c: &HistConsts,
    base: *mut u8,
    cnt: &mut [usize; 4],
    thr: usize,
    h16: &mut [u16; 256],
) {
    for (b, (n, hp)) in cnt.iter_mut().zip(h16.as_chunks_mut::<64>().0).enumerate() {
        if *n >= thr {
            // SAFETY: the caller's.
            unsafe { consume(c, base.add(b * BIN_SIZE), *n, hp) };
            *n = 0;
        }
    }
}

/// HIST_FA: c = a^b^c, b = majority(a, b, c); returns (b, c).
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt,gfni,avx512bitalg"
)]
fn fa(a: __m512i, b: __m512i, c: __m512i) -> (__m512i, __m512i) {
    let c = _mm512_ternarylogic_epi32::<0x96>(c, b, a);
    (_mm512_ternarylogic_epi32::<0xB2>(b, c, a), c) // from a, b and a^b^c
}

/// Turns the bit planes of v into byte counts: an 8x8 bit transpose per qword and VPOPCNTB.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt,gfni,avx512bitalg"
)]
fn transpose(c: &HistConsts, v: __m512i) -> __m512i {
    let v = _mm512_permutexvar_epi8(c.tp, v);
    // x = gf, A = v: transposes each 8x8 bit block.
    _mm512_popcnt_epi8(_mm512_gf2p8affine_epi64_epi8::<0>(c.gf, v))
}

/// Counts the nb > 0 six-bit morsels at data (64-byte aligned) into the 64 u16 counters at
/// hp (Go `consume`; upstream hist_consume_bin_avx512).
///
/// # Safety
/// The CPU has the tier's features; data has ceil(nb/64)*64 writable bytes.
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt,gfni,avx512bitalg"
)]
unsafe fn consume(c: &HistConsts, mut data: *mut u8, nb: usize, hp: &mut [u16; 64]) {
    let tail = nb & 63;
    if tail != 0 {
        // SAFETY: the padded row is inside the caller's rows.
        unsafe { _mm512_mask_storeu_epi8(data.add(nb & !63).cast(), u64::MAX << tail, c.ones) };
    }
    let mut rows = nb.div_ceil(64);
    let z = _mm512_setzero_si512();
    let (mut h0, mut h1, mut w0, mut w1, mut w2) = (z, z, z, z, z);
    loop {
        let mr = rows.min(31); // 31 rows of popcounts <= 8 fit a byte
        rows -= mr;
        let mut w = z;
        for _ in 0..mr {
            let mut x = [z; 8];
            for (r, x) in x.iter_mut().enumerate() {
                // SAFETY: the row is one of the caller's.
                let m = unsafe { _mm_loadl_epi64(data.add(8 * r).cast()) };
                *x = _mm512_sllv_epi64(c.one, _mm512_cvtepu8_epi64(m));
            }
            data = data.wrapping_add(64);
            let [x0, x1, x2, x3, x4, x5, x6, x7] = x;
            let (x1, x2) = fa(x0, x1, x2);
            let (x4, x5) = fa(x3, x4, x5);
            let (x7, n0) = fa(x6, x7, w0);
            let (x5, n0) = fa(x2, x5, n0);
            let (x4, x7) = fa(x1, x4, x7);
            let (x5, n1) = fa(x7, x5, w1);
            let (w3, n2) = fa(x5, x4, w2);
            (w0, w1, w2) = (n0, n1, n2);
            w = _mm512_add_epi8(transpose(c, w3), w);
        }
        h0 = _mm512_add_epi16(_mm512_and_si512(c.lo8, w), h0);
        h1 = _mm512_add_epi16(_mm512_srli_epi16::<8>(w), h1);
        if rows == 0 {
            break;
        }
    }
    let (t1, t2) = (transpose(c, w1), transpose(c, w2));
    let w0 = _mm512_add_epi8(_mm512_add_epi8(transpose(c, w0), t1), t1);
    let w0 = _mm512_add_epi8(_mm512_slli_epi64::<2>(t2), w0);
    let h0 = _mm512_add_epi16(_mm512_and_si512(c.lo8, w0), _mm512_slli_epi16::<3>(h0));
    let h1 = _mm512_add_epi16(_mm512_srli_epi16::<8>(w0), _mm512_slli_epi16::<3>(h1));
    let p = hp.as_mut_ptr();
    // SAFETY: hp has 64 u16s.
    unsafe {
        _mm512_storeu_si512(p.cast(), _mm512_add_epi16(_mm512_loadu_si512(p.cast()), h0));
        let p1 = p.add(32).cast();
        _mm512_storeu_si512(p1, _mm512_add_epi16(_mm512_loadu_si512(p1), h1));
    }
}

/// Adds h16, which is in consume's lane order (see histogram_chunk_avx512), to h.
#[inline]
#[target_feature(
    enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vbmi2,avx512vpopcntdq,bmi2,popcnt,gfni,avx512bitalg"
)]
fn widen(h: &mut [u32; 256], h16: &[u16; 256]) {
    let lo16 = _mm512_set1_epi32(0xFFFF);
    for (o, v) in h
        .as_chunks_mut::<32>()
        .0
        .iter_mut()
        .zip(h16.as_chunks::<32>().0)
    {
        // SAFETY: v has 32 u16s and o 32 u32s.
        unsafe {
            let v = _mm512_loadu_si512(v.as_ptr().cast());
            let (p0, p1) = (o.as_mut_ptr(), o.as_mut_ptr().add(16));
            let lo = _mm512_add_epi32(_mm512_loadu_si512(p0.cast()), _mm512_and_si512(lo16, v));
            let hi = _mm512_add_epi32(_mm512_loadu_si512(p1.cast()), _mm512_srli_epi32::<16>(v));
            _mm512_storeu_si512(p0.cast(), lo);
            _mm512_storeu_si512(p1.cast(), hi);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::pivco::kernel::{HIST_SCRATCH, Tier};
    use crate::pivco::testutil::{new_rng, rand_bytes};

    /// Go `TestAVX512Histogram`: the sizes of upstream test_histogram at several
    /// alignments, accumulated into a non-zero histogram twice, and 1 MiB of one byte so that
    /// a bin is consumed in the middle of a round. Inputs below 0x40 and below 0x80, also
    /// with sparse larger bytes, run the one- and two-bin row loops and their exits.
    #[test]
    fn avx512_histogram() {
        if !Tier::Avx512.hist_uses_scratch() {
            return; // no AVX-512 GFNI/BITALG
        }
        let mut rng = new_rng();
        let mut scratch = vec![0u8; HIST_SCRATCH];
        let sizes = [
            0,
            1,
            7,
            63,
            64,
            65,
            127,
            4095,
            4096,
            65472,
            65473,
            1 << 20,
            5 << 20,
        ];
        for n in sizes {
            for off in [0, 1, 63] {
                for kind in 0..6 {
                    let mut buf = vec![0u8; n + off + 64];
                    let src = &mut buf[off..off + n];
                    match kind {
                        0 => rand_bytes(&mut rng, src),
                        1 => src.fill(0xC3),
                        // one symbol per bin, skewed
                        2 => src
                            .iter_mut()
                            .for_each(|s| *s = (rng.int_n(4) << 6 | rng.int_n(2)) as u8),
                        // one bin
                        3 => src.iter_mut().for_each(|s| *s = rng.int_n(64) as u8),
                        // two bins
                        4 => src.iter_mut().for_each(|s| *s = rng.int_n(128) as u8),
                        // one bin, with sparse bytes of the other bins
                        _ => src.iter_mut().for_each(|s| {
                            *s = rng.int_n(64) as u8;
                            if rng.int_n(3000) == 0 {
                                *s = rng.int_n(256) as u8;
                            }
                        }),
                    }
                    let mut h: [u32; 256] = core::array::from_fn(|_| rng.uint32());
                    let mut want = h;
                    for _ in 0..2 {
                        for &c in src.iter() {
                            want[usize::from(c)] = want[usize::from(c)].wrapping_add(1);
                        }
                        Tier::Avx512.histogram(&mut h, src, &mut scratch);
                    }
                    assert!(h == want, "n={n} off={off} kind={kind}: histogram mismatch");
                }
            }
        }
    }

    /// Without its scratch, the AVX-512 histogram falls back to the generic one (features may
    /// be re-enabled between the caller's hist_uses_scratch check and the call).
    #[test]
    fn avx512_histogram_short_scratch() {
        if !Tier::Avx512.hist_uses_scratch() {
            return;
        }
        let mut src = vec![0u8; 100_000];
        rand_bytes(&mut new_rng(), &mut src);
        let (mut h, mut want) = ([0u32; 256], [0u32; 256]);
        Tier::Generic.histogram(&mut want, &src, &mut []);
        Tier::Avx512.histogram(&mut h, &src, &mut []);
        assert!(h == want);
    }
}
