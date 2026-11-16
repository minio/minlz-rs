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

//! The AVX2 tier: a `core::arch` port of the assembly that Go's `_generate/pivco`
//! (`enc_avx2.go`, `dec_avx2.go`, `avx2_util.go`, `dec.go`) generates. It ports upstream's
//! x86 tier as built with PIVCO_HAS_AVX2 and without AVX-512
//! (`pivco_huffman_primitives_x86.h`, `pivco_huffman_x86_vertical.h`,
//! `pivco_huffman_avx2_pack.h`). The `*_asm` functions port the generated functions of the
//! same Go names; the histogram is generic, as upstream's.
//!
//! Requirements: AVX2, BMI2, POPCNT ([`PIVCO_AVX2`](crate::cpu::PIVCO_AVX2)); every entry
//! point is an `unsafe fn` whose caller has checked them. No AVX-512 instructions.
//!
//! The entry points assert the lengths the kernels rely on: reads of src and ranks and
//! stores to ranks[..n] and the packed region are exact; right-side and bitmap stores use
//! [`RANK_SLACK`].
//!
//! Differences from C (encoder):
//!   - Reads of src and ranks are exact, and so are the in-place left stores of part_full
//!     and every store to the packed region: the tails run the vector step on the last
//!     ranks shifted down from a load that ends at n (or on an exact small load when
//!     n < 16) and store through exact 8/4/2/1-byte stores. C ends with scalar loops.
//!   - The partitions build their masks with VPMAXUB/VPCMPEQB, which also covers thr = 255.
//!   - The gather returns the largest rank, from a VPMAXUB pass over the stored ranks.
//!   - The natural packs subtract base with VPSUBUSB, so the zero lanes of a short tail
//!     stay zero and give zero padding bits.
//!
//! Differences from C (decoder):
//!   - The shuf1 table is kept as 16-byte rows with a zero low half, so the 32-lane merge
//!     index is two loads and two adds per lane instead of broadcasts and VPBLENDDs.
//!   - CV blends the constant by the sign of the merge index (one VPBLENDVB) instead of a
//!     second PSHUFB and an OR.
//!   - The tails run the 16-lane step on the masked last bitmap bits and store exactly;
//!     C ends with scalar loops.
//!   - The natural layout uses one unpack for D = 2..7 (PSHUFB pairs, VPMULLW to the top
//!     of each word, VPSRLW, VPACKUSWB), 32 codes per step on two 16-byte windows, with
//!     the vertical kernels' PSHUFB/VPBLENDVB lookups. C has a kernel per D.
//!
//! The 16-lane steps of the flat kernels run on the low lane of a YMM register (the
//! loads leave the high lane undefined, the stores drop it).
//!
//! The D = 8 copy and pack loops run at about 0.8 and 0.9 of Go's speed on a Ryzen 9 9950X.
//! Their speed depends on where the loop lands in a 64-byte block, which a Rust build doesn't
//! control: Go's copy loop as inline asm in a Rust binary ran 203 ns at offset 32 and 245-255
//! ns elsewhere, as fast as LLVM's 4x unrolled copy. D = 8 flats only occur for near-uniform
//! symbols.

use core::arch::x86_64::*;
use core::mem::transmute;

use super::generic_pack::pack_generic;
use super::{
    ARENA_SLACK, FLAG_DST, OP_CC, OP_COPY, OP_CV, OP_NAT, OP_V128, OP_V512, OP_VV, Op, Program,
    R2S_LEN, RANK_SLACK, vert_split,
};

// The partitions store 16 bytes at right+nr and 2 bitmap bytes at the tail's offset.
const _: () = assert!(RANK_SLACK >= 16);

/// Sets `ranks[i] = s2r[src[i]]` and returns the largest rank written.
///
/// # Safety
/// The CPU has [`PIVCO_AVX2`](crate::cpu::PIVCO_AVX2).
pub(super) unsafe fn gather_avx2(ranks: &mut [u8], src: &[u8], s2r: &[u8; 256]) -> u8 {
    let n = src.len();
    if n == 0 {
        return 0;
    }
    let ranks = &mut ranks[..n];
    // SAFETY: the CPU has the features; ranks and src hold n > 0 bytes.
    unsafe { gather_avx2_asm(ranks.as_mut_ptr(), src.as_ptr(), n, s2r) }
}

/// Partitions `ranks` by rank > thr; see [`super::Tier::part_full`].
///
/// # Safety
/// The CPU has [`PIVCO_AVX2`](crate::cpu::PIVCO_AVX2).
pub(super) unsafe fn part_full_avx2(
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
    // SAFETY: the CPU has the features; the slices have the lengths part_avx2_asm needs.
    unsafe { part_avx2_asm::<PART_FULL>(r, right.as_mut_ptr(), r, bm.as_mut_ptr(), n, thr) }
}

/// Partitions `ranks` like [`part_full_avx2`] without compacting the left ranks.
///
/// # Safety
/// The CPU has [`PIVCO_AVX2`](crate::cpu::PIVCO_AVX2).
pub(super) unsafe fn part_right_avx2(
    ranks: &[u8],
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
    let (r, b) = (right.as_mut_ptr(), bm.as_mut_ptr());
    // SAFETY: the CPU has the features; the slices have the lengths part_avx2_asm needs;
    // left is not used.
    unsafe { part_avx2_asm::<PART_RIGHT>(ranks.as_ptr(), r, core::ptr::null_mut(), b, n, thr) }
}

/// Writes only the bitmap of `ranks` and returns the number of set bits.
///
/// # Safety
/// The CPU has [`PIVCO_AVX2`](crate::cpu::PIVCO_AVX2).
pub(super) unsafe fn part_none_avx2(ranks: &[u8], bm: &mut [u8], thr: u8) -> usize {
    let n = ranks.len();
    if n == 0 {
        return 0;
    }
    let bm = &mut bm[..n.div_ceil(8) + RANK_SLACK];
    let null = core::ptr::null_mut();
    // SAFETY: the CPU has the features; the slices have the lengths part_avx2_asm needs;
    // right and left are not used.
    unsafe { part_avx2_asm::<PART_NONE>(ranks.as_ptr(), null, null, bm.as_mut_ptr(), n, thr) }
}

/// Packs the codes of a flat region; see [`super::Tier::pack`].
///
/// # Safety
/// The CPU has [`PIVCO_AVX2`](crate::cpu::PIVCO_AVX2).
pub(super) unsafe fn pack_avx2(dst: &mut [u8], ranks: &[u8], d: u8, base: u8) {
    let n = ranks.len();
    if n == 0 || !(2..=8).contains(&d) {
        pack_generic(dst, ranks, d, base);
        return;
    }
    let dst = &mut dst[..(n * usize::from(d)).div_ceil(8)];
    let (n512, n128) = vert_split(n, d);
    // SAFETY: the CPU has the features; dst holds the packed region and ranks n codes.
    unsafe { pack_avx2_asm(dst.as_mut_ptr(), ranks.as_ptr(), n, n512, n128, d, base) }
}

/// Runs the ops in one call. Like exec_generic it checks each op before running it, so a
/// malformed op fails closed.
///
/// # Safety
/// The CPU has [`PIVCO_AVX2`](crate::cpu::PIVCO_AVX2).
pub(super) unsafe fn exec_avx2(p: &mut Program<'_>) -> Result<(), usize> {
    // SAFETY: the CPU has the features.
    unsafe { exec_avx2_asm(p) }
}

// Raw loads and stores; the callers state why each access is in bounds.

#[inline(always)]
unsafe fn rd16(p: *const u8) -> u16 {
    // SAFETY: the caller guarantees 2 readable bytes.
    unsafe { p.cast::<u16>().read_unaligned() }
}

#[inline(always)]
unsafe fn rd32(p: *const u8) -> u32 {
    // SAFETY: the caller guarantees 4 readable bytes.
    unsafe { p.cast::<u32>().read_unaligned() }
}

#[inline(always)]
unsafe fn wr16(p: *mut u8, v: u16) {
    // SAFETY: the caller guarantees 2 writable bytes.
    unsafe { p.cast::<u16>().write_unaligned(v) }
}

#[inline(always)]
unsafe fn wr32(p: *mut u8, v: u32) {
    // SAFETY: the caller guarantees 4 writable bytes.
    unsafe { p.cast::<u32>().write_unaligned(v) }
}

#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn ld128(p: *const u8) -> __m128i {
    // SAFETY: the caller guarantees 16 readable bytes.
    unsafe { _mm_loadu_si128(p.cast()) }
}

#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn ld256(p: *const u8) -> __m256i {
    // SAFETY: the caller guarantees 32 readable bytes.
    unsafe { _mm256_loadu_si256(p.cast()) }
}

#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn st128(p: *mut u8, v: __m128i) {
    // SAFETY: the caller guarantees 16 writable bytes.
    unsafe { _mm_storeu_si128(p.cast(), v) }
}

#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn st256(p: *mut u8, v: __m256i) {
    // SAFETY: the caller guarantees 32 writable bytes.
    unsafe { _mm256_storeu_si256(p.cast(), v) }
}

/// Loads w (16 or 32) bytes; with 16 the high lane is undefined.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn ldw(p: *const u8, w: usize) -> __m256i {
    // SAFETY: the caller guarantees w readable bytes.
    unsafe {
        if w == 32 {
            ld256(p)
        } else {
            _mm256_castsi128_si256(ld128(p))
        }
    }
}

/// Stores the low w (16 or 32) bytes of v.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn stw(p: *mut u8, w: usize, v: __m256i) {
    // SAFETY: the caller guarantees w writable bytes.
    unsafe {
        if w == 32 {
            st256(p, v)
        } else {
            st128(p, _mm256_castsi256_si128(v))
        }
    }
}

// Shifts by a count the callers fold to a constant, as the generator's immediates; the
// intrinsics with immediates cannot take counts computed from const generics.

#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn sll16(v: __m256i, n: usize) -> __m256i {
    _mm256_sll_epi16(v, _mm_cvtsi64_si128(n as i64))
}

#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn srl16(v: __m256i, n: usize) -> __m256i {
    _mm256_srl_epi16(v, _mm_cvtsi64_si128(n as i64))
}

#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn srl64(v: __m256i, n: usize) -> __m256i {
    _mm256_srl_epi64(v, _mm_cvtsi64_si128(n as i64))
}

#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn splat(v: u8) -> __m256i {
    _mm256_set1_epi8(v as i8)
}

/// Runs $body for $s = 0..8, as constants: the flat kernels need their shifts and masks
/// folded, and LLVM leaves the larger bodies rolled.
macro_rules! unroll8 {
    ($s:ident => $body:block) => {{
        {
            let $s = 0usize;
            $body
        }
        {
            let $s = 1usize;
            $body
        }
        {
            let $s = 2usize;
            $body
        }
        {
            let $s = 3usize;
            $body
        }
        {
            let $s = 4usize;
            $body
        }
        {
            let $s = 5usize;
            $body
        }
        {
            let $s = 6usize;
            $body
        }
        {
            let $s = 7usize;
            $body
        }
    }};
}

/// Aligns a table to a cache line, as Go's RODATA blobs.
#[repr(C, align(64))]
struct Align64<T>(T);

/// The 48-byte shuffle table {0x80 x16, 0..15, 0x80 x16}: PSHUFB with the 16 bytes at
/// 16+k shifts down by k lanes, at 16-k up by k, zero filling.
static SHIFT: Align64<[u8; 48]> = Align64({
    let mut r = [0x80; 48];
    let mut i = 0;
    while i < 16 {
        r[16 + i] = i as u8;
        i += 1;
    }
    r
});

/// Returns the shuffle at `SHIFT[off..off+16]`.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn shift_row(off: usize) -> __m128i {
    let r: &[u8; 16] = SHIFT.0[off..off + 16].try_into().unwrap();
    // SAFETY: r holds 16 bytes.
    unsafe { ld128(r.as_ptr()) }
}

/// Loads the k (0..=16) bytes at p into the low lanes and zeroes the others, reading
/// nothing outside `p[..k]`.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn load_exact(p: *const u8, k: usize) -> __m128i {
    // SAFETY: every read lies in p[..k], which the caller guarantees readable.
    unsafe {
        if k >= 8 {
            let x = _mm_loadl_epi64(p.cast());
            let t = _mm_loadl_epi64(p.add(k - 8).cast());
            // Bytes k-8..k-1 go up by k-8 lanes.
            return _mm_or_si128(x, _mm_shuffle_epi8(t, shift_row(24 - k)));
        }
        let v = if k >= 4 {
            u64::from(rd32(p)) | u64::from(rd32(p.add(k - 4))) << (8 * (k - 4))
        } else if k >= 2 {
            u64::from(rd16(p)) | u64::from(rd16(p.add(k - 2))) << (8 * (k - 2))
        } else if k == 1 {
            u64::from(*p)
        } else {
            0
        };
        _mm_cvtsi64_si128(v as i64)
    }
}

/// Stores the low k (0..=16) bytes of x at p and nothing else; overlapping stores write
/// the same bytes twice.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn store_exact(p: *mut u8, k: usize, x: __m128i) {
    // SAFETY: every write lies in p[..k], which the caller guarantees writable.
    unsafe {
        if k >= 8 {
            _mm_storel_epi64(p.cast(), x);
            // Bytes k-8..k-1 come down by k-8 lanes.
            _mm_storel_epi64(p.add(k - 8).cast(), _mm_shuffle_epi8(x, shift_row(8 + k)));
            return;
        }
        let v = _mm_cvtsi128_si64(x) as u64;
        if k >= 4 {
            wr32(p, v as u32);
            wr32(p.add(k - 4), (v >> (8 * (k - 4))) as u32);
        } else if k >= 2 {
            wr16(p, v as u16);
            wr16(p.add(k - 2), (v >> (8 * (k - 2))) as u16);
        } else if k == 1 {
            *p = v as u8;
        }
    }
}

// ---------------------------------------------------------------------------------------
// Encoder kernels (enc_avx2.go).

/// The partition tables of x86_build_tabs (primitives_x86.h:907-952), per mask m: tabA
/// (lefts forward from lane 0, rights backward from 15), tabB0 (the same for lanes 8..15
/// at pc0 = 0, loaded at +pc0) and ctabR (rights first, 0x80 fill). x86_pc8 and
/// x86_ctab_l/pre_l are not needed: POPCNT, and no left compaction by table.
#[allow(clippy::type_complexity)]
const fn p16_tabs() -> ([[u8; 16]; 256], [[u8; 32]; 256], [[u8; 16]; 256]) {
    let (mut ta, mut tb, mut tc) = ([[0; 16]; 256], [[0; 32]; 256], [[0x80; 16]; 256]);
    let mut m = 0;
    while m < 256 {
        let (mut al, mut ar, mut bl, mut br, mut nr) = (0, 15, 8, 15, 0);
        let mut k = 0;
        while k < 8 {
            if m >> k & 1 != 0 {
                (ta[m][ar], tb[m][br], tc[m][nr]) = (k as u8, 8 + k as u8, k as u8);
                (ar, br, nr) = (ar - 1, br - 1, nr + 1);
            } else {
                (ta[m][al], tb[m][bl]) = (k as u8, 8 + k as u8);
                (al, bl) = (al + 1, bl + 1);
            }
            k += 1;
        }
        m += 1;
    }
    (ta, tb, tc)
}

/// preR[nlo][m]: the high byte's rights (8+k) from lane nlo on, 0x80 fill.
const fn pre_r() -> [[[u8; 16]; 256]; 9] {
    let mut t = [[[0x80; 16]; 256]; 9];
    let mut nlo = 0;
    while nlo < 9 {
        let mut m = 0;
        while m < 256 {
            let (mut p, mut k) = (nlo, 0);
            while k < 8 {
                if m >> k & 1 != 0 {
                    t[nlo][m][p] = 8 + k as u8;
                    p += 1;
                }
                k += 1;
            }
            m += 1;
        }
        nlo += 1;
    }
    t
}

#[allow(clippy::type_complexity)]
const P16_TABS: ([[u8; 16]; 256], [[u8; 32]; 256], [[u8; 16]; 256]) = p16_tabs();
// SAFETY (all three): same size, and every bit pattern is a valid __m128i.
static P16_A: [__m128i; 256] = unsafe { transmute(P16_TABS.0) };
static P16_B: Align64<[[u8; 32]; 256]> = Align64(P16_TABS.1);
static CTAB_R: [__m128i; 256] = unsafe { transmute(P16_TABS.2) };
static PRE_R: [[__m128i; 256]; 9] = unsafe { transmute(pre_r()) };

/// Ports prim_enc_init's 2tab loop (primitives_x86.h:1284-1314): each pair of symbols is
/// one u16 store of s2r[a] + hi[b], hi[s] = s2r[s]<<8 in a stack table built per call (C
/// keeps it in the table as enc_init_hi). Returns the largest of ranks[..n].
///
/// # Safety
/// The CPU has the features; ranks and src hold n > 0 bytes.
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn gather_avx2_asm(ranks: *mut u8, src: *const u8, n: usize, s2r: &[u8; 256]) -> u8 {
    let mut hi = [0u16; 256];
    for k in 0..16 {
        // SAFETY: s2r holds 16*16 bytes and hi 16*16 u16.
        unsafe {
            let v = _mm256_cvtepu8_epi16(ld128(s2r.as_ptr().add(16 * k)));
            st256(
                hi.as_mut_ptr().add(16 * k).cast(),
                _mm256_slli_epi16::<8>(v),
            );
        }
    }
    // Go loads 8 symbols and shifts the register by 16 per pair; LLVM turns that into a
    // shift per byte, so each pair is a 16-bit load here.
    let mut i = 0;
    while n - i >= 16 {
        for p in (0..8).step_by(2) {
            // SAFETY: i+16 <= n, the length of src and ranks.
            unsafe {
                let a = rd16(src.add(i + p));
                let x = u16::from(s2r[usize::from(a as u8)]) + hi[usize::from(a >> 8)];
                wr16(ranks.add(i + p), x);
                let b = rd16(src.add(i + 8 + p));
                let y = u16::from(s2r[usize::from(b as u8)]) + hi[usize::from(b >> 8)];
                wr16(ranks.add(i + 8 + p), y);
            }
        }
        i += 16;
    }
    while i < n {
        // SAFETY: i < n.
        unsafe { *ranks.add(i) = s2r[usize::from(*src.add(i))] };
        i += 1;
    }

    // The maximum of ranks[..n]; the last 32-byte load overlaps the loop's.
    if n < 32 {
        let mut m = 0;
        for i in 0..n {
            // SAFETY: i < n.
            m = m.max(unsafe { *ranks.add(i) });
        }
        return m;
    }
    let mut m = _mm256_setzero_si256();
    let mut i = 0;
    while i + 32 <= n {
        // SAFETY: i+32 <= n.
        m = _mm256_max_epu8(m, unsafe { ld256(ranks.add(i)) });
        i += 32;
    }
    // SAFETY: n >= 32.
    m = _mm256_max_epu8(m, unsafe { ld256(ranks.add(n - 32)) });
    let mut x = _mm_max_epu8(_mm256_castsi256_si128(m), _mm256_extracti128_si256::<1>(m));
    x = _mm_max_epu8(x, _mm_srli_si128::<8>(x));
    x = _mm_max_epu8(x, _mm_srli_si128::<4>(x));
    x = _mm_max_epu8(x, _mm_srli_si128::<2>(x));
    x = _mm_max_epu8(x, _mm_srli_si128::<1>(x));
    _mm_cvtsi128_si32(x) as u8
}

const PART_FULL: u8 = 0;
const PART_RIGHT: u8 = 1;
const PART_NONE: u8 = 2;

/// Returns the rank > thr bits of the 32 lanes of v.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn mask32(v: __m256i, thr: __m256i) -> u32 {
    !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(thr, _mm256_max_epu8(v, thr))) as u32)
}

/// Returns the rank > thr bits of the 16 lanes of v (zero lanes give 0).
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn mask16(v: __m128i, thr: __m128i) -> u32 {
    _mm_movemask_epi8(_mm_cmpeq_epi8(thr, _mm_max_epu8(v, thr))) as u32 ^ 0xFFFF
}

/// Compacts the 16 lanes of v by the mask bits sh..sh+15 of m (_P16REV): lefts in order
/// from lane 0, rights in reverse order down from lane 15.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn p16(v: __m128i, m: u32, sh: u32) -> __m128i {
    let (a, b) = (m >> sh & 0xFF, m >> (sh + 8) & 0xFF);
    let row = &P16_B.0[b as usize];
    // SAFETY: popcount(a) <= 8, so the 16 bytes lie in the 32-byte row.
    let tb = unsafe { ld128(row.as_ptr().add(a.count_ones() as usize)) };
    _mm_shuffle_epi8(v, _mm_or_si128(P16_A[a as usize], tb))
}

/// Compacts the right lanes of v by the mask bits sh..sh+15 of m in order to the low
/// lanes (_PC16).
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn pc16(v: __m128i, m: u32, sh: u32) -> __m128i {
    let (a, b) = (m >> sh & 0xFF, m >> (sh + 8) & 0xFF);
    // SAFETY: popcount(a) <= 8 < PRE_R.len().
    let pre = unsafe { PRE_R.get_unchecked(a.count_ones() as usize)[b as usize] };
    _mm_shuffle_epi8(v, _mm_min_epu8(CTAB_R[a as usize], pre))
}

/// A partition by rank > thr, 32 ranks per step with a 16-rank step and a tail after it
/// (part_full_x86 and part_core_x86, primitives_x86.h:997-1109). PART_FULL compacts the
/// lefts in place to left == src; PART_NONE never touches right.
///
/// # Safety
/// The CPU has the features; n > 0; src has n readable bytes (writable through left for
/// PART_FULL), right n+RANK_SLACK writable (not PART_NONE) and bm ceil(n/8)+RANK_SLACK.
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn part_avx2_asm<const KIND: u8>(
    src: *const u8,
    right: *mut u8,
    left: *mut u8,
    mut bm: *mut u8,
    n: usize,
    thr: u8,
) -> usize {
    let thr = _mm256_set1_epi8(thr as i8);
    let thr16 = _mm256_castsi256_si128(thr);
    let rev = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let (mut i, mut nl, mut nr) = (0, 0, 0);
    // The last 16 ranks, read before any in-place store; the tail shifts them down.
    let last = if n >= 16 {
        // SAFETY: n >= 16.
        unsafe { ld128(src.add(n - 16)) }
    } else {
        _mm_setzero_si128()
    };
    // In-place left stores land at nl <= i, on ranks a step has already loaded, and end
    // at or before its end; right stores end at nr+16 <= n+RANK_SLACK.
    for _ in 0..n >> 5 {
        // SAFETY: i+32 <= n; bm has 4 bytes per 32 ranks; see above for the stores.
        unsafe {
            let s = src.add(i);
            let v0 = ld256(s);
            let m = mask32(v0, thr);
            wr32(bm, m);
            if KIND != PART_NONE {
                let v1 = ld128(s.add(16));
                let lo = (m & 0xFFFF).count_ones() as usize;
                if KIND == PART_FULL {
                    let c = p16(_mm256_castsi256_si128(v0), m, 0);
                    st128(left.add(nl), c);
                    st128(right.add(nr), _mm_shuffle_epi8(c, rev));
                    let c = p16(v1, m, 16);
                    st128(left.add(nl + 16 - lo), c);
                    st128(right.add(nr + lo), _mm_shuffle_epi8(c, rev));
                } else {
                    st128(right.add(nr), pc16(_mm256_castsi256_si128(v0), m, 0));
                    st128(right.add(nr + lo), pc16(v1, m, 16));
                }
            }
            let pc = m.count_ones() as usize;
            (nr, nl) = (nr + pc, nl + 32 - pc);
            bm = bm.add(4);
        }
        i += 32;
    }
    if n - i >= 16 {
        // SAFETY: as in the loop, for 16 ranks and 2 bitmap bytes.
        unsafe {
            let v = ld128(src.add(i));
            let m = mask16(v, thr16);
            wr16(bm, m as u16);
            if KIND == PART_FULL {
                let c = p16(v, m, 0);
                st128(left.add(nl), c);
                st128(right.add(nr), _mm_shuffle_epi8(c, rev));
            } else if KIND == PART_RIGHT {
                st128(right.add(nr), pc16(v, m, 0));
            }
            let pc = m.count_ones() as usize;
            (nr, nl) = (nr + pc, nl + 16 - pc);
            bm = bm.add(2);
        }
        i += 16;
    }
    let t = n - i;
    if t > 0 {
        // t = 1..15 ranks remain at i; they become lanes 0..t-1 of v, the rest zero
        // (rank 0 is never right).
        let v = if i == 0 {
            // SAFETY: src holds t = n bytes.
            unsafe { load_exact(src, t) }
        } else {
            _mm_shuffle_epi8(last, shift_row(32 - t))
        };
        let m = mask16(v, thr16);
        // SAFETY: bm holds the tail's bitmap byte plus RANK_SLACK; the exact left store
        // ends at nl+t-pc <= n, the right store at nr+16 <= n+RANK_SLACK.
        unsafe {
            wr16(bm, m as u16);
            if KIND == PART_FULL {
                let c = p16(v, m, 0);
                // Exact: the bytes after the lefts belong to the caller.
                store_exact(left.add(nl), t - m.count_ones() as usize, c);
                st128(right.add(nr), _mm_shuffle_epi8(c, rev));
            } else if KIND == PART_RIGHT {
                st128(right.add(nr), pc16(v, m, 0));
            }
        }
        nr += m.count_ones() as usize;
    }
    nr
}

/// The constants of the multiply-as-shift pack (avx2_pack.h) for D: c0 bytes {1, 2^D},
/// c1 words {1, 2^2D}, c3 = 2^4D-1 per qword; RYG_CMP[D] compacts each 128-bit lane's two
/// 8D-bit qwords to 2D bytes (junk lanes zero).
static RYG_CMP: [__m256i; 9] = {
    let mut t = [[0u8; 32]; 9];
    let mut d = 2;
    while d <= 7 {
        let mut k = 0;
        while k < 32 {
            let e = k % 16;
            t[d][k] = if e < d {
                e as u8
            } else if e < 2 * d {
                (8 + e - d) as u8
            } else {
                0x80
            };
            k += 1;
        }
        d += 1;
    }
    // SAFETY: same size, and every bit pattern is a valid __m256i.
    unsafe { transmute(t) }
};

/// Packs the codes ranks[i]-base of a depth-d flat region (2 <= d <= 8, n > 0) into
/// exactly dst[..(n*d+7)/8]: n512 codes in 512-value vertical blocks, the next n128 in
/// 128-value blocks and the rest in natural order (prim_enc_pack_dN, pack_dN_x86).
///
/// # Safety
/// The CPU has the features; dst holds (n*d+7)/8 bytes and ranks n; (n512, n128) is
/// vert_split(n, d).
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn pack_avx2_asm(
    dst: *mut u8,
    ranks: *const u8,
    n: usize,
    n512: usize,
    n128: usize,
    d: u8,
    base: u8,
) {
    let base = _mm256_set1_epi8(base as i8);
    let nnat = n - n512 - n128;
    // SAFETY: forwarded.
    unsafe {
        match d {
            2 => pack_d::<2>(dst, ranks, nnat, n512, n128, base),
            3 => pack_d::<3>(dst, ranks, nnat, n512, n128, base),
            4 => pack_d::<4>(dst, ranks, nnat, n512, n128, base),
            5 => pack_d::<5>(dst, ranks, nnat, n512, n128, base),
            6 => pack_d::<6>(dst, ranks, nnat, n512, n128, base),
            7 => pack_d::<7>(dst, ranks, nnat, n512, n128, base),
            8 => pack_d::<8>(dst, ranks, nnat, n512, n128, base),
            _ => {}
        }
    }
}

/// # Safety
/// As for [`pack_avx2_asm`], with D = d.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn pack_d<const D: usize>(
    mut dst: *mut u8,
    mut src: *const u8,
    nnat: usize,
    n512: usize,
    n128: usize,
    base: __m256i,
) {
    // 512-value blocks run as two 32-lane halves (vert512_pack_ymm_d*), except D = 7,
    // which upstream runs as four 16-lane quarters on AMD (vert512_pack_x86_d7; 1.12 vs
    // 0.89 on c6a).
    // SAFETY: the blocks and the natural rest partition ranks[..n] and the region.
    unsafe {
        if D < 8 {
            for _ in 0..n512 >> 9 {
                if D == 7 {
                    vert_body::<D>(dst, src, 16, 64, 0, base);
                    vert_body::<D>(dst, src, 16, 64, 16, base);
                    vert_body::<D>(dst, src, 16, 64, 32, base);
                    vert_body::<D>(dst, src, 16, 64, 48, base);
                } else {
                    vert_body::<D>(dst, src, 32, 64, 0, base);
                    vert_body::<D>(dst, src, 32, 64, 32, base);
                }
                (src, dst) = (src.add(512), dst.add(512 * D / 8));
            }
            for _ in 0..n128 >> 7 {
                vert_body::<D>(dst, src, 16, 16, 0, base);
                (src, dst) = (src.add(128), dst.add(128 * D / 8));
            }
        }
        natural::<D>(dst, src, nnat, base);
    }
}

/// Packs w lanes at lane offset off of a vertical block whose rows (and columns) have the
/// given stride: code s*stride+l goes to bits s*D... of lane l.
///
/// Always inlined, as the generator unrolls the halves and quarters: LLVM keeps a loop
/// over them that rotates the masks through registers. It has no target features of its
/// own (Rust rejects them with `inline(always)`); its callers have them.
///
/// # Safety
/// The CPU has the features; src holds 8 rows and dst D columns of stride bytes.
#[inline(always)]
unsafe fn vert_body<const D: usize>(
    dst: *mut u8,
    src: *const u8,
    w: usize,
    stride: usize,
    off: usize,
    base: __m256i,
) {
    // SAFETY: the CPU has the features; the rows and columns lie in src and dst.
    unsafe {
        let mut cols = [_mm256_setzero_si256(); 8];
        unroll8!(s => {
            let (j, o) = ((s * D) >> 3, (s * D) & 7);
            let v = _mm256_subs_epu8(ldw(src.add(stride * s + off), w), base);
            if o == 0 {
                cols[j] = v;
            } else {
                let t = _mm256_and_si256(sll16(v, o), splat(0xFF << o));
                cols[j] = _mm256_or_si256(cols[j], t);
                if o + D > 8 {
                    cols[j + 1] = _mm256_and_si256(srl16(v, 8 - o), splat(0xFF >> (8 - o)));
                }
            }
        });
        for (j, &c) in cols.iter().enumerate().take(D) {
            stw(dst.add(stride * j + off), w, c);
        }
    }
}

/// Turns the 16 or 32 codes in v (zero in unused lanes) into 2D packed bytes per 128-bit
/// lane, or for D = 4 into 8 bytes per lane, or for D = 8 leaves them.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn code16<const D: usize>(v: __m256i) -> __m256i {
    match D {
        8 => v,
        4 => {
            let v = _mm256_maddubs_epi16(_mm256_set1_epi16(0x1001), v);
            _mm256_packus_epi16(v, v)
        }
        _ => {
            let c0 = _mm256_set1_epi16((1 | 1 << (D + 8)) as i16);
            let c1 = _mm256_set1_epi32(1 | 1 << (2 * D + 16));
            let c3 = _mm256_set1_epi64x(((1u64 << (4 * D)) - 1) as i64);
            let v = _mm256_madd_epi16(_mm256_maddubs_epi16(c0, v), c1);
            let xs = _mm256_andnot_si256(c3, srl64(v, 32 - 4 * D));
            let v = _mm256_or_si256(_mm256_and_si256(v, c3), xs);
            _mm256_shuffle_epi8(v, RYG_CMP[D])
        }
    }
}

/// Stores the low K (constant, 4..=16) bytes of x at o.
///
/// # Safety
/// o holds K writable bytes.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn store_const<const K: usize>(o: *mut u8, x: __m128i) {
    // SAFETY: the stores cover o[..K] exactly, overlapping for K = 6, 10, 12, 14.
    unsafe {
        match K {
            16 => st128(o, x),
            8 => _mm_storel_epi64(o.cast(), x),
            4 => wr32(o, _mm_cvtsi128_si32(x) as u32),
            6 => {
                wr32(o, _mm_cvtsi128_si32(x) as u32);
                wr32(o.add(2), _mm_cvtsi128_si32(_mm_srli_si128::<2>(x)) as u32);
            }
            10 => {
                _mm_storel_epi64(o.cast(), x);
                _mm_storel_epi64(o.add(2).cast(), _mm_srli_si128::<2>(x));
            }
            12 => {
                _mm_storel_epi64(o.cast(), x);
                _mm_storel_epi64(o.add(4).cast(), _mm_srli_si128::<4>(x));
            }
            14 => {
                _mm_storel_epi64(o.cast(), x);
                _mm_storel_epi64(o.add(6).cast(), _mm_srli_si128::<6>(x));
            }
            _ => unreachable!(),
        }
    }
}

/// pack_dN_x86's SIMD loops: 32 codes per step while the junk of its 16-byte stores stays
/// inside the region, then 16 codes per step with exact stores, then one partial step on
/// an exact load.
///
/// # Safety
/// src holds cnt codes and o their (cnt*D+7)/8 packed bytes.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn natural<const D: usize>(
    mut o: *mut u8,
    mut src: *const u8,
    mut cnt: usize,
    base: __m256i,
) {
    if cnt == 0 {
        return;
    }
    // Output bytes per 32- and 16-code step.
    let (per32, per16) = if D == 8 { (32, 16) } else { (4 * D, 2 * D) };
    // SAFETY: the steps read their codes inside src[..cnt] and store inside the region:
    // the 32-code step of D != 4, 8 stores 2D+16 bytes, so it runs while they fit.
    unsafe {
        let end = o.add((cnt * D).div_ceil(8));
        while cnt >= 32 {
            if D != 4 && D != 8 && (end as usize - o as usize) < 2 * D + 16 {
                break;
            }
            let v = code16::<D>(_mm256_subs_epu8(ld256(src), base));
            match D {
                8 => st256(o, v),
                4 => st128(
                    o,
                    _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0x08>(v)),
                ),
                _ => {
                    st128(o, _mm256_castsi256_si128(v));
                    st128(o.add(2 * D), _mm256_extracti128_si256::<1>(v));
                }
            }
            (src, o, cnt) = (src.add(32), o.add(per32), cnt - 32);
        }
        while cnt >= 16 {
            let v = code16::<D>(_mm256_subs_epu8(_mm256_castsi128_si256(ld128(src)), base));
            let v = _mm256_castsi256_si128(v);
            match per16 {
                4 => store_const::<4>(o, v),
                6 => store_const::<6>(o, v),
                8 => store_const::<8>(o, v),
                10 => store_const::<10>(o, v),
                12 => store_const::<12>(o, v),
                14 => store_const::<14>(o, v),
                _ => store_const::<16>(o, v),
            }
            (src, o, cnt) = (src.add(16), o.add(per16), cnt - 16);
        }
        if cnt > 0 {
            let v = _mm256_castsi128_si256(load_exact(src, cnt));
            let v = _mm256_castsi256_si128(code16::<D>(_mm256_subs_epu8(v, base)));
            store_exact(o, if D == 8 { cnt } else { (cnt * D).div_ceil(8) }, v);
        }
    }
}

// ---------------------------------------------------------------------------------------
// Decoder executor (dec_avx2.go, dec.go).

/// g_x86_merge_shuf0/shuf1 (primitives_x86.h:120-138). shuf0[m]: R lanes index R, L
/// lanes hold 0xFF minus their L index, bytes 8..15 popcount(m). shuf1[m]: the same for
/// lanes 8..15 relative to popcount of the low byte (added by the caller), low half zero.
const fn merge_tabs() -> [[[u8; 16]; 256]; 2] {
    let mut t = [[[0u8; 16]; 256]; 2];
    let mut m = 0;
    while m < 256 {
        let pop = (m as u32).count_ones() as u8;
        let (mut rs, mut rc) = (0u8, 0u8);
        let mut i = 0;
        while i < 8 {
            if m >> i & 1 != 0 {
                (t[0][m][i], rs) = (rs, rs + 1);
            } else {
                (t[0][m][i], rc) = (255 - rc, rc + 1);
            }
            t[0][m][8 + i] = pop;
            i += 1;
        }
        (rs, rc) = (0, 0);
        i = 0;
        while i < 8 {
            if m >> i & 1 != 0 {
                (t[1][m][8 + i], rs) = (rs, rs + 1);
            } else {
                (t[1][m][8 + i], rc) = (247 - rc, rc + 1);
            }
            i += 1;
        }
        m += 1;
    }
    t
}

// SAFETY: same size, and every bit pattern is a valid __m128i.
static SHUF: [[__m128i; 256]; 2] = unsafe { transmute(merge_tabs()) };

/// Natural unpack, per D: word e of a lane holds the bytes (b, b+1) of code e (idxA) or
/// 8+e (idxB) of its 16-byte window, b = D*e/8; the multiplier moves the code to the top.
static NAT: [[__m256i; 3]; 8] = {
    let mut t = [[[0u8; 32]; 3]; 8];
    let mut d = 2;
    while d <= 7 {
        let mut l = 0;
        while l < 32 {
            let mut e = 0;
            while e < 8 {
                let pos = d * e;
                let b = pos >> 3;
                (t[d][0][l + 2 * e], t[d][0][l + 2 * e + 1]) = (b as u8, (b + 1) as u8);
                (t[d][1][l + 2 * e], t[d][1][l + 2 * e + 1]) = ((d + b) as u8, (d + b + 1) as u8);
                let mul = 1u16 << (16 - d - (pos & 7));
                (t[d][2][l + 2 * e], t[d][2][l + 2 * e + 1]) = (mul as u8, (mul >> 8) as u8);
                e += 1;
            }
            l += 16;
        }
        d += 1;
    }
    // SAFETY: same size, and every bit pattern is a valid __m256i.
    unsafe { transmute(t) }
};

/// Runs p.ops and returns the index of the first op that fails its checks. Each op is
/// checked against the slices before it writes. Bitmap and region reads are exact: the
/// merges read 4, 2 and then 1 or 2 bitmap bytes, the natural unpack reads 16-byte
/// windows only while they end inside the source, then exact loads; stores are exact.
/// The merges load 16 bytes at child cursors that stay inside the child, so they read at
/// most 15 bytes past it, and the child lies in the op's K bytes at L or R (ARENA_SLACK).
///
/// # Safety
/// The CPU has the features.
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn exec_avx2_asm(p: &mut Program<'_>) -> Result<(), usize> {
    let alim = p.arena.len().saturating_sub(ARENA_SLACK);
    // Every arena pointer derives from a: in-place merges read and write the same bytes.
    let (a, dst, dlen) = (p.arena.as_mut_ptr(), p.dst.as_mut_ptr(), p.dst.len());
    let (input, r2s) = (p.input, p.r2s);
    for (i, op) in p.ops.iter().enumerate() {
        // Output: out+k must fit dst, or the arena less its slack.
        let k = op.k as usize;
        let (base, lim) = if op.flags & FLAG_DST != 0 {
            (dst, dlen)
        } else {
            (a, alim)
        };
        if op.out as usize + k > lim {
            return Err(i);
        }
        // SAFETY: out+k <= lim, inside base's buffer.
        let out = unsafe { base.add(op.out as usize) };
        // SAFETY: every kernel is run after the checks its contract names.
        let ok = unsafe {
            match op.kind {
                OP_VV | OP_CV | OP_CC => {
                    let (bm, l, r) = (op.bm as usize, op.l as usize, op.r as usize);
                    // The bitmap's ceil(k/8) bytes lie in the source; the children's k
                    // bytes in the arena less its slack.
                    if bm + k.div_ceil(8) > input.len() {
                        false
                    } else {
                        let bm = input.as_ptr().add(bm);
                        match op.kind {
                            OP_VV if l + k <= alim && r + k <= alim => {
                                merge_vv(out, k, bm, a.add(l), a.add(r)) == op.kr as usize
                            }
                            OP_CV if r + k <= alim => {
                                merge_cv(out, k, bm, a.add(r), op.p) == op.kr as usize
                            }
                            OP_CC => {
                                merge_cc(out, k, bm, op.p, op.q);
                                true
                            }
                            _ => false,
                        }
                    }
                }
                OP_V512 | OP_V128 => {
                    let (d, lanes) = (usize::from(op.d), if op.kind == OP_V512 { 64 } else { 16 });
                    match flat_checks(input, r2s, op, true) {
                        Some((src, tab)) if (2..=7).contains(&d) && k.is_multiple_of(8 * lanes) => {
                            vertical(out, src, tab, k / (8 * lanes), d, lanes == 64);
                            true
                        }
                        _ => false,
                    }
                }
                OP_NAT => match flat_checks(input, r2s, op, false) {
                    Some((src, tab)) => {
                        let end = input.as_ptr().add(input.len());
                        natural_unpack(out, src, end, tab, k, usize::from(op.d));
                        true
                    }
                    None => false,
                },
                OP_COPY => {
                    let bm = op.bm as usize;
                    if op.d != 8 || bm + k > input.len() {
                        false
                    } else {
                        copy_flat(out, input.as_ptr().add(bm), k);
                        true
                    }
                }
                _ => false,
            }
        };
        if !ok {
            return Err(i);
        }
    }
    Ok(())
}

/// Validates a flat op: D in 2..=8, the region of K*D/8 (exact) or ceil(K*D/8) bytes at
/// Bm must lie in the source, R2S must be set and R2S[P..P+2^D] inside it. It returns the
/// region and &R2S[P]. The vector lookups load 16-byte quarters from &R2S[P], which stay
/// inside R2S: P <= 255 and R2S_LEN = 256+64.
#[inline]
fn flat_checks(
    input: &[u8],
    r2s: Option<&[u8; R2S_LEN]>,
    op: &Op,
    exact: bool,
) -> Option<(*const u8, *const u8)> {
    let (k, d) = (op.k as usize, usize::from(op.d));
    if !(2..=8).contains(&d) {
        return None;
    }
    let bytes = if exact {
        k * d / 8
    } else {
        (k * d).div_ceil(8)
    };
    let (bm, p) = (op.bm as usize, usize::from(op.p));
    let r2s = r2s?;
    if bm + bytes > input.len() || p + (1 << d) > R2S_LEN {
        return None;
    }
    Some((input[bm..].as_ptr(), r2s[p..].as_ptr()))
}

/// Sets the merge index of the 16 bitmap bits at bit sh of m: shuf0[m0] + shuf1x[m1] (R
/// lanes: index into R, L lanes: 0x80 | complement of L's).
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn shuf16(m: u32, sh: u32) -> __m128i {
    _mm_add_epi8(
        SHUF[0][(m >> sh & 0xFF) as usize],
        SHUF[1][(m >> (sh + 8) & 0xFF) as usize],
    )
}

/// Returns the merge index of the 32 bitmap bits in m and the number of set bits in the
/// low 16.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn shuf32(m: u32) -> (__m256i, usize) {
    let x = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(shuf16(m, 0)), shuf16(m, 16));
    (x, (m & 0xFFFF).count_ones() as usize)
}

/// Loads the 16 bytes at p and at p+off into the two lanes.
///
/// # Safety
/// p..p+16 and p+off..p+off+16 are readable.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn ld2(p: *const u8, off: usize) -> __m256i {
    // SAFETY: the caller guarantees both 16-byte reads.
    unsafe { _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(ld128(p)), ld128(p.add(off))) }
}

/// Returns the bitmap bits of the last t = k&15 (1..=15) symbols, read exactly.
///
/// # Safety
/// bm holds ceil(t/8) bytes.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn tail_bits(bm: *const u8, t: usize) -> u32 {
    // SAFETY: t > 8 needs 2 bytes, else 1.
    let m = unsafe {
        if t > 8 {
            u32::from(rd16(bm))
        } else {
            u32::from(*bm)
        }
    };
    _bzhi_u32(m, t as u32)
}

// The merges load both children before storing a step's output. In the in-place layout
// the larger child ends at out[K) and starts K_small bytes in; the write cursor j = l+r
// then never passes its read cursor, so no unread byte is overwritten.

/// Writes `out[j] = bit j ? r[nr++] : l[nl++]` for j < k and returns nr (upstream
/// merge_vec_vec_x86): 32 per step, a 16 step, then the tail.
///
/// # Safety
/// out holds k writable bytes, bm ceil(k/8) bytes; l and r point at children of k bytes
/// followed by 16 readable bytes. Pointers into one buffer derive from one pointer.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn merge_vv(
    mut out: *mut u8,
    k: usize,
    mut bm: *const u8,
    mut l: *const u8,
    mut r: *const u8,
) -> usize {
    let r0 = r;
    let ones = _mm256_set1_epi8(-1);
    // SAFETY: each step consumes 32 bitmap bits and output bytes, pc of r and 32-pc of l;
    // the second lane loads start at r+lo and l+16-lo, the cursors after 16 symbols.
    let step = |out: *mut u8, bm: *const u8, l: *const u8, r: *const u8| unsafe {
        let m = rd32(bm);
        let (x, lo) = shuf32(m);
        let y = _mm256_shuffle_epi8(ld2(r, lo), x);
        let z = _mm256_shuffle_epi8(ld2(l, 16 - lo), _mm256_xor_si256(x, ones));
        st256(out, _mm256_or_si256(y, z));
        m.count_ones() as usize
    };
    // SAFETY: as above, for 16 symbols.
    let step16 = |m: u32, l: *const u8, r: *const u8| unsafe {
        let x = shuf16(m, 0);
        let y = _mm_shuffle_epi8(ld128(r), x);
        let z = _mm_shuffle_epi8(ld128(l), _mm_xor_si128(x, _mm256_castsi256_si128(ones)));
        _mm_or_si128(y, z)
    };
    // SAFETY: the cursors advance by what each step consumed, inside the buffers.
    unsafe {
        for _ in 0..k >> 6 {
            let pc = step(out, bm, l, r);
            (r, l) = (r.add(pc), l.add(32 - pc));
            let pc = step(out.add(32), bm.add(4), l, r);
            (r, l) = (r.add(pc), l.add(32 - pc));
            (bm, out) = (bm.add(8), out.add(64));
        }
        if k & 32 != 0 {
            let pc = step(out, bm, l, r);
            (r, l) = (r.add(pc), l.add(32 - pc));
            (bm, out) = (bm.add(4), out.add(32));
        }
        if k & 16 != 0 {
            let m = u32::from(rd16(bm));
            st128(out, step16(m, l, r));
            let pc = m.count_ones() as usize;
            (r, l) = (r.add(pc), l.add(16 - pc));
            (bm, out) = (bm.add(2), out.add(16));
        }
        let t = k & 15;
        if t != 0 {
            let m = tail_bits(bm, t);
            store_exact(out, t, step16(m, l, r));
            r = r.add(m.count_ones() as usize);
        }
        r.offset_from(r0) as usize
    }
}

/// Writes `out[j] = bit j ? r[nr++] : c` for j < k and returns nr (upstream
/// merge_cst_vec_x86).
///
/// # Safety
/// As for [`merge_vv`], without l.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn merge_cv(
    mut out: *mut u8,
    k: usize,
    mut bm: *const u8,
    mut r: *const u8,
    c: u8,
) -> usize {
    let r0 = r;
    let cv = _mm256_set1_epi8(c as i8);
    // SAFETY: as in merge_vv.
    let step = |out: *mut u8, bm: *const u8, r: *const u8| unsafe {
        let m = rd32(bm);
        let (x, lo) = shuf32(m);
        let y = _mm256_shuffle_epi8(ld2(r, lo), x);
        // L lanes have the sign bit of the index set.
        st256(out, _mm256_blendv_epi8(y, cv, x));
        m.count_ones() as usize
    };
    // SAFETY: as in merge_vv.
    let step16 = |m: u32, r: *const u8| unsafe {
        let x = shuf16(m, 0);
        _mm_blendv_epi8(_mm_shuffle_epi8(ld128(r), x), _mm256_castsi256_si128(cv), x)
    };
    // SAFETY: as in merge_vv.
    unsafe {
        for _ in 0..k >> 6 {
            r = r.add(step(out, bm, r));
            r = r.add(step(out.add(32), bm.add(4), r));
            (bm, out) = (bm.add(8), out.add(64));
        }
        if k & 32 != 0 {
            r = r.add(step(out, bm, r));
            (bm, out) = (bm.add(4), out.add(32));
        }
        if k & 16 != 0 {
            let m = u32::from(rd16(bm));
            st128(out, step16(m, r));
            r = r.add(m.count_ones() as usize);
            (bm, out) = (bm.add(2), out.add(16));
        }
        let t = k & 15;
        if t != 0 {
            let m = tail_bits(bm, t);
            store_exact(out, t, step16(m, r));
            r = r.add(m.count_ones() as usize);
        }
        r.offset_from(r0) as usize
    }
}

/// Writes `out[j] = bit j ? q : p` (upstream merge_cst_cst_x86): byte k of the bitmap
/// word to lanes 8k..8k+7, then lane j's bit to all of its bits.
///
/// # Safety
/// out holds k writable bytes, bm ceil(k/8) bytes.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn merge_cc(mut out: *mut u8, k: usize, mut bm: *const u8, p: u8, q: u8) {
    let (pv, qv) = (_mm256_set1_epi8(p as i8), _mm256_set1_epi8(q as i8));
    let shuf = _mm256_setr_epi64x(
        0,
        0x0101010101010101,
        0x0202020202020202,
        0x0303030303030303,
    );
    let bits = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
    let step16 = |m: u32| {
        let x = _mm_shuffle_epi8(_mm_cvtsi32_si128(m as i32), _mm256_castsi256_si128(shuf));
        let x = _mm_and_si128(x, _mm256_castsi256_si128(bits));
        let x = _mm_cmpeq_epi8(x, _mm256_castsi256_si128(bits));
        _mm_blendv_epi8(_mm256_castsi256_si128(pv), _mm256_castsi256_si128(qv), x)
    };
    // SAFETY: each step consumes its bitmap bytes and output bytes, inside the buffers.
    unsafe {
        for _ in 0..k >> 5 {
            let x = _mm256_shuffle_epi8(_mm256_set1_epi32(rd32(bm) as i32), shuf);
            let x = _mm256_cmpeq_epi8(_mm256_and_si256(x, bits), bits);
            st256(out, _mm256_blendv_epi8(pv, qv, x));
            (bm, out) = (bm.add(4), out.add(32));
        }
        if k & 16 != 0 {
            st128(out, step16(u32::from(rd16(bm))));
            (bm, out) = (bm.add(2), out.add(16));
        }
        let t = k & 15;
        if t != 0 {
            store_exact(out, t, step16(tail_bits(bm, t)));
        }
    }
}

/// Loads the lookup table of depth D from tab = &R2S[P]: max(1, 2^D/16) 16-byte quarters
/// broadcast to both lanes.
///
/// # Safety
/// tab holds the quarters (see [`flat_checks`]).
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn tables<const D: usize>(tab: *const u8) -> [__m256i; 8] {
    let mut t = [_mm256_setzero_si256(); 8];
    for (i, q) in t.iter_mut().enumerate().take((1 << D >> 4).max(1)) {
        // SAFETY: quarter i lies in the table.
        *q = _mm256_broadcastsi128_si256(unsafe { ld128(tab.add(16 * i)) });
    }
    t
}

/// Maps the D-bit codes in v (bits above D clear) to symbols with PSHUFB on 16-entry
/// quarters and a VPBLENDVB tree on bits 4..6 (the MAP forms of x86_vertical.h). Sibling
/// quarters are merged as soon as both are there, so few are live at once.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn lookup<const D: usize>(v: __m256i, t: &[__m256i; 8]) -> __m256i {
    if D <= 4 {
        return _mm256_shuffle_epi8(t[0], v);
    }
    // Bit b of each code at the sign of its byte.
    let (s4, s5, s6) = (sll16(v, 3), sll16(v, 2), sll16(v, 1));
    let pair = |i: usize| {
        let (a, b) = (
            _mm256_shuffle_epi8(t[i], v),
            _mm256_shuffle_epi8(t[i + 1], v),
        );
        _mm256_blendv_epi8(a, b, s4)
    };
    match D {
        5 => pair(0),
        6 => _mm256_blendv_epi8(pair(0), pair(2), s5),
        _ => {
            let lo = _mm256_blendv_epi8(pair(0), pair(2), s5);
            let hi = _mm256_blendv_epi8(pair(4), pair(6), s5);
            _mm256_blendv_epi8(lo, hi, s6)
        }
    }
}

/// Extracts field s of a vertical block's lanes: bits s*D... of the column bytes col(j).
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn codes<const D: usize>(s: usize, col: impl Fn(usize) -> __m256i) -> __m256i {
    let mask = ((1 << D) - 1) as u8;
    let (j, o) = ((s * D) >> 3, (s * D) & 7);
    let c = col(j);
    if o == 0 {
        return _mm256_and_si256(c, splat(mask));
    }
    let v = srl16(c, o);
    if o + D <= 8 {
        return _mm256_and_si256(v, splat(mask));
    }
    let v = _mm256_and_si256(v, splat(0xFF >> o));
    let t = _mm256_and_si256(
        sll16(col(j + 1), 8 - o),
        splat((0xFF << (8 - o)) as u8 & mask),
    );
    _mm256_or_si256(v, t)
}

/// Runs the 512-code (two 32-lane halves of 64 lanes) or 128-code (16 lanes) vertical
/// kernels (vert512_merge_ymm_d*, vert_merge_x86_d*) on blocks blocks.
///
/// # Safety
/// src holds the blocks' region, out their codes; tab as for [`tables`]; 2 <= d <= 7.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn vertical(
    out: *mut u8,
    src: *const u8,
    tab: *const u8,
    blocks: usize,
    d: usize,
    wide: bool,
) {
    // SAFETY: forwarded.
    unsafe {
        match (d, wide) {
            (2, true) => vert_d::<2, 64>(out, src, tab, blocks),
            (3, true) => vert_d::<3, 64>(out, src, tab, blocks),
            (4, true) => vert_d::<4, 64>(out, src, tab, blocks),
            (5, true) => vert_d::<5, 64>(out, src, tab, blocks),
            (6, true) => vert_d::<6, 64>(out, src, tab, blocks),
            (7, true) => vert_d::<7, 64>(out, src, tab, blocks),
            (2, false) => vert_d::<2, 16>(out, src, tab, blocks),
            (3, false) => vert_d::<3, 16>(out, src, tab, blocks),
            (4, false) => vert_d::<4, 16>(out, src, tab, blocks),
            (5, false) => vert_d::<5, 16>(out, src, tab, blocks),
            (6, false) => vert_d::<6, 16>(out, src, tab, blocks),
            (7, false) => vert_d::<7, 16>(out, src, tab, blocks),
            _ => {}
        }
    }
}

/// Lane l of a block holds its 8 codes in D column bytes at a stride of LANES.
///
/// # Safety
/// As for [`vertical`].
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn vert_d<const D: usize, const LANES: usize>(
    mut out: *mut u8,
    mut src: *const u8,
    tab: *const u8,
    blocks: usize,
) {
    // SAFETY: the table lies in R2S; each block reads LANES*D bytes and writes 8*LANES,
    // as one 16-lane step or two 32-lane halves.
    unsafe {
        let t = tables::<D>(tab);
        // The halves stay a loop: unrolled, LLVM spills table quarters for D = 7.
        let w = LANES.min(32);
        for _ in 0..blocks {
            for h in 0..LANES / w {
                vert_step::<D>(out.add(w * h), src.add(w * h), LANES, w, &t);
            }
            (src, out) = (src.add(LANES * D), out.add(8 * LANES));
        }
    }
}

/// Decodes w lanes of a vertical block whose rows and columns have the given stride.
/// Always inlined, as [`vert_body`] (LLVM passes the tables through memory otherwise).
///
/// # Safety
/// The CPU has the features; src holds D columns and out 8 rows of w bytes at the stride.
#[inline(always)]
unsafe fn vert_step<const D: usize>(
    out: *mut u8,
    src: *const u8,
    stride: usize,
    w: usize,
    t: &[__m256i; 8],
) {
    // Columns in registers for D < COLS_REG; above, they are read where they are used, as
    // the generator's memory operands (which it needs for D = 7 only): loaded up front for
    // D = 6 and 7, LLVM sinks the output stores past the lookups and spills.
    const COLS_REG: usize = 6;
    // SAFETY: the CPU has the features; the columns and rows lie in src and out.
    unsafe {
        let mut cols = [_mm256_setzero_si256(); 8];
        if D < COLS_REG {
            for (j, c) in cols.iter_mut().enumerate().take(D) {
                *c = ldw(src.add(stride * j), w);
            }
        }
        let col = |j: usize| {
            if D < COLS_REG {
                cols[j]
            } else {
                ldw(src.add(stride * j), w)
            }
        };
        unroll8!(s => {
            let v = lookup::<D>(codes::<D>(s, col), t);
            stw(out.add(stride * s), w, v);
        });
    }
}

/// Turns the window(s) in v into the codes of its 16 (per lane) D-bit fields; mul is
/// `NAT[D][2]`, passed through `black_box` by the caller: with the constant in sight,
/// LLVM turns VPMULLW and VPSRLW into VPMULHUW, VPBLENDW and VPAND.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
fn unpack<const D: usize>(v: __m256i, mul: __m256i) -> __m256i {
    let [ia, ib, _] = NAT[D];
    let a = srl16(_mm256_mullo_epi16(_mm256_shuffle_epi8(v, ia), mul), 16 - D);
    let b = srl16(_mm256_mullo_epi16(_mm256_shuffle_epi8(v, ib), mul), 16 - D);
    _mm256_packus_epi16(a, b)
}

/// The natural-layout kernels for D = 2..8.
///
/// # Safety
/// src holds the region of ceil(k*d/8) bytes and ends at or before end, the end of the
/// source; out holds k bytes; tab as for [`tables`].
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn natural_unpack(
    out: *mut u8,
    src: *const u8,
    end: *const u8,
    tab: *const u8,
    k: usize,
    d: usize,
) {
    // SAFETY: forwarded; D = 8 codes are bytes, so tab[c] lies in its 256 entries.
    unsafe {
        match d {
            2 => nat_d::<2>(out, src, end, tab, k),
            3 => nat_d::<3>(out, src, end, tab, k),
            4 => nat_d::<4>(out, src, end, tab, k),
            5 => nat_d::<5>(out, src, end, tab, k),
            6 => nat_d::<6>(out, src, end, tab, k),
            7 => nat_d::<7>(out, src, end, tab, k),
            _ => {
                for i in 0..k {
                    *out.add(i) = *tab.add(usize::from(*src.add(i)));
                }
            }
        }
    }
}

/// For D <= 7 a 16-code step reads a 16-byte window at 2D*step; the 32-code step does two
/// windows in the two lanes.
///
/// # Safety
/// As for [`natural_unpack`].
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn nat_d<const D: usize>(
    mut out: *mut u8,
    mut src: *const u8,
    end: *const u8,
    tab: *const u8,
    mut k: usize,
) {
    // SAFETY: the windows end inside the source (checked against end), the exact loads
    // inside the region; the stores cover out[..k].
    unsafe {
        let t = tables::<D>(tab);
        let mul = core::hint::black_box(NAT[D][2]);
        let room = |src: *const u8| end as usize - src as usize;
        while k >= 32 && room(src) >= 2 * D + 16 {
            let v = lookup::<D>(unpack::<D>(ld2(src, 2 * D), mul), &t);
            st256(out, v);
            (src, out, k) = (src.add(4 * D), out.add(32), k - 32);
        }
        while k >= 16 && room(src) >= 16 {
            let v = lookup::<D>(unpack::<D>(_mm256_castsi128_si256(ld128(src)), mul), &t);
            st128(out, _mm256_castsi256_si128(v));
            (src, out, k) = (src.add(2 * D), out.add(16), k - 16);
        }
        // Near the end of the source: exact loads of min(16, rest) codes.
        while k > 0 {
            let c = k.min(16);
            let x = _mm256_castsi128_si256(load_exact(src, (c * D).div_ceil(8)));
            store_exact(
                out,
                c,
                _mm256_castsi256_si128(lookup::<D>(unpack::<D>(x, mul), &t)),
            );
            if k <= 16 {
                break;
            }
            (src, out, k) = (src.add(2 * D), out.add(16), k - 16);
        }
    }
}

/// D = 8 flat: the codes are the symbols (upstream merge_flat_d8_x86).
///
/// # Safety
/// src and out hold k bytes.
#[inline]
#[target_feature(enable = "avx2,bmi2,popcnt")]
unsafe fn copy_flat(mut out: *mut u8, mut src: *const u8, mut k: usize) {
    // SAFETY: every load and store lies in the first k bytes.
    unsafe {
        while k >= 64 {
            // Both loads first, as Go: LLVM cannot move a load above a store that may alias.
            let (a, b) = (ld256(src), ld256(src.add(32)));
            st256(out, a);
            st256(out.add(32), b);
            (src, out, k) = (src.add(64), out.add(64), k - 64);
        }
        while k >= 16 {
            st128(out, ld128(src));
            (src, out, k) = (src.add(16), out.add(16), k - 16);
        }
        if k > 0 {
            store_exact(out, k, load_exact(src, k));
        }
    }
}
