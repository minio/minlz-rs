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

//! The NEON tier: Go's `neon_arm64.go`, `neon_arm64.s` and `neon_tables_arm64.go`, ports of
//! upstream `src/pivco_huffman_primitives_neon.h`, `pivco_huffman_neon_flat.h` and
//! `pivco_huffman_neon_pack.h`. ASIMD is part of ARMv8, so the tier is on unless
//! [`cpu::NEON`] is disabled. The encoder kernels run whole vector steps and the generic
//! kernels finish the tails; the decoder runs a whole [`Program`] in one call. Reads of src,
//! ranks and bitmaps and stores to ranks[..n], outputs and packed regions are exact, the
//! partitions store inside right[..n] and the natural unpack reads input windows that may
//! pass its region but not the input. Merges load up to 15 bytes past the K bytes of a
//! child, flats load 64 bytes of R2S from the first rank (128 for D=7, which the op checks
//! bound). The histogram is generic, as upstream's.
//!
//! The `*_asm` functions port the Go assembly functions of the same name with intrinsics
//! and its instruction choices; their helpers are named after its macros. Differences: the
//! cursors are counts, not pointers; scalar popcounts are `count_ones` (CNT+ADDV, as the
//! assembly's VCNT+UADDLV); the gather's max is one UMAXV, which Go's assembler lacks; the
//! COPY op is a memcpy. Little-endian only: integer loads and vector lanes share one byte
//! order.

use core::arch::aarch64::*;

use super::generic::{gather_generic, part_full_generic, part_none_generic, part_right_generic};
use super::generic_exec::exec_generic;
use super::generic_pack::{nat_pack, pack_generic};
use super::{
    ARENA_SLACK, FLAG_DST, OP_CC, OP_COPY, OP_CV, OP_NAT, OP_V128, OP_V512, OP_VV, Op, Program,
    R2S_LEN, RANK_SLACK, vert_split,
};
use crate::cpu;

// Merges load 15 bytes past a child; flats load 64 bytes of R2S at any first rank.
const _: () = assert!(ARENA_SLACK >= 15 && R2S_LEN >= 255 + 64);

fn has_neon() -> bool {
    cpu::has(cpu::NEON)
}

pub(super) fn gather_neon(ranks: &mut [u8], src: &[u8], s2r: &[u8; 256]) -> u8 {
    if !has_neon() {
        return gather_generic(ranks, src, s2r);
    }
    let n = src.len();
    let ranks = &mut ranks[..n];
    let (k, mut mx) = (n & !15, 0);
    if k > 0 {
        // SAFETY: the CPU has NEON; ranks and src hold n >= k bytes.
        mx = unsafe { gather_neon_asm(ranks.as_mut_ptr(), src.as_ptr(), k, s2r) };
    }
    mx.max(gather_generic(&mut ranks[k..], &src[k..], s2r))
}

pub(super) fn part_full_neon(
    ranks: &mut [u8],
    n: usize,
    right: &mut [u8],
    bm: &mut [u8],
    thr: u8,
) -> usize {
    if n == 0 || !has_neon() {
        return part_full_generic(ranks, n, right, bm, thr);
    }
    let (ranks, right, bm) = (
        &mut ranks[..n],
        &mut right[..n + RANK_SLACK],
        &mut bm[..n.div_ceil(8) + RANK_SLACK],
    );
    let (j, mut nr) = (n & !63, 0);
    if j > 0 {
        // SAFETY: the CPU has NEON; ranks and right hold n >= j bytes, bm j/8.
        nr = unsafe {
            part_full_neon_asm(
                ranks.as_mut_ptr(),
                right.as_mut_ptr(),
                bm.as_mut_ptr(),
                j,
                thr,
            )
        };
    }
    // The tail compacts its left ranks in place; they follow the j-nr vector ones.
    let tr = part_full_generic(
        &mut ranks[j..],
        n - j,
        &mut right[nr..],
        &mut bm[j / 8..],
        thr,
    );
    ranks.copy_within(j..n - tr, j - nr);
    nr + tr
}

pub(super) fn part_right_neon(
    ranks: &[u8],
    n: usize,
    right: &mut [u8],
    bm: &mut [u8],
    thr: u8,
) -> usize {
    if n == 0 || !has_neon() {
        return part_right_generic(ranks, n, right, bm, thr);
    }
    let (ranks, right, bm) = (
        &ranks[..n],
        &mut right[..n + RANK_SLACK],
        &mut bm[..n.div_ceil(8) + RANK_SLACK],
    );
    let (j, mut nr) = (n & !7, 0);
    if j > 0 {
        // SAFETY: the CPU has NEON; ranks and right hold n >= j bytes, bm j/8.
        nr = unsafe {
            part_right_neon_asm(ranks.as_ptr(), right.as_mut_ptr(), bm.as_mut_ptr(), j, thr)
        };
    }
    nr + part_right_generic(&ranks[j..], n - j, &mut right[nr..], &mut bm[j / 8..], thr)
}

pub(super) fn part_none_neon(ranks: &[u8], n: usize, bm: &mut [u8], thr: u8) -> usize {
    if n == 0 || !has_neon() {
        return part_none_generic(ranks, n, bm, thr);
    }
    let (ranks, bm) = (&ranks[..n], &mut bm[..n.div_ceil(8) + RANK_SLACK]);
    let (j, mut nr) = (n & !7, 0);
    if j > 0 {
        // SAFETY: the CPU has NEON; ranks holds n >= j bytes, bm j/8.
        nr = unsafe { part_none_neon_asm(ranks.as_ptr(), bm.as_mut_ptr(), j, thr) };
    }
    nr + part_none_generic(&ranks[j..], n - j, &mut bm[j / 8..], thr)
}

pub(super) fn pack_neon(dst: &mut [u8], ranks: &[u8], n: usize, d: u8, base: u8) {
    if n == 0 || !(2..=8).contains(&d) || !has_neon() {
        return pack_generic(dst, ranks, n, d, base);
    }
    let du = usize::from(d);
    let (dst, ranks) = (&mut dst[..(n * du).div_ceil(8)], &ranks[..n]);
    let (n512, n128) = vert_split(n, d);
    if n512 > 0 {
        // SAFETY: the CPU has NEON; 2 <= d <= 7 (vert_split), ranks holds n512 bytes and dst
        // their n512*d/8.
        unsafe { vert_pack_neon_asm(dst.as_mut_ptr(), ranks.as_ptr(), n512, 64, d, base) };
    }
    let (dst, ranks) = (&mut dst[n512 * du / 8..], &ranks[n512..]);
    if n128 > 0 {
        // SAFETY: as above, for n128.
        unsafe { vert_pack_neon_asm(dst.as_mut_ptr(), ranks.as_ptr(), n128, 16, d, base) };
    }
    let (dst, ranks) = (&mut dst[n128 * du / 8..], &ranks[n128..]);
    let mut i = 0;
    if ranks.len() >= 16 {
        // SAFETY: the CPU has NEON; the kernel reads at most ranks.len() ranks and stores
        // inside dst.len() bytes.
        i = unsafe {
            nat_pack_neon_asm(
                dst.as_mut_ptr(),
                ranks.as_ptr(),
                ranks.len(),
                dst.len(),
                d,
                base,
            )
        };
    }
    nat_pack(&mut dst[i * du / 8..], &ranks[i..], d, base);
}

/// Runs the ops in one call. Like exec_generic it checks each op before running it, so a
/// malformed op fails closed.
pub(super) fn exec_neon(p: &mut Program<'_>) -> Result<(), usize> {
    if !has_neon() {
        return exec_generic(p);
    }
    if p.ops.is_empty() {
        return Ok(());
    }
    // SAFETY: the CPU has NEON.
    unsafe { exec_neon_asm(p) }
}

// ---- Tables (Go neon_tables_arm64.go). The merge and partition tables are built as
// upstream's init_merge_tables and build_tabs (src/pivco_huffman_primitives_neon.h).

/// MERGE_TAB[0][m0] - MERGE_TAB[1][m1] is the merge index of 16 outputs with bitmap
/// m0|m1<<8: right lanes get their right cursor (0..15), left lanes 240 minus their left
/// cursor, so that one TBL and one TBX pick both sides.
static MERGE_TAB: [[[u8; 16]; 256]; 2] = merge_tabs();
/// P16_A[m0] | P16_B[m1][pc0..] is the index that puts the left ranks of 16 with mask
/// m0|m1<<8 in order at the front and the right ones reversed at the back;
/// pc0 = popcount(m0).
static P16_A: [[u8; 16]; 256] = p16_tabs().0;
static P16_B: [[u8; 32]; 256] = p16_tabs().1;
/// CTAB[0][m] (CTAB[1][m]) moves the right ranks of the low (high) 8 lanes of a vector to
/// the front; 0xFF lanes look up zero.
static CTAB: [[[u8; 8]; 256]; 2] = ctabs();
/// NAT_UNPACK_TAB[d] is the TBL gathering the byte pair of each of codes 0..7 and 8..15 of
/// a 16-code natural group into a u16 lane, the USHL counts that move the codes down, and
/// the code mask.
static NAT_UNPACK_TAB: [[[u8; 16]; 4]; 8] = nat_unpack_tabs();
/// NAT_PACK_TAB[d] is the compacting TBL and the USHL counts of upstream's converging
/// pyramid (src/pivco_huffman_neon_pack.h) at the 8, 16, 32 and 64-bit levels.
static NAT_PACK_TAB: [[[u8; 16]; 5]; 8] = nat_pack_tabs();

const fn merge_tabs() -> [[[u8; 16]; 256]; 2] {
    let mut t = [[[0u8; 16]; 256]; 2];
    let mut i = 0;
    while i < 256 {
        let mut pop = 0u8;
        let mut j = 0;
        while j < 8 {
            if i >> j & 1 != 0 {
                t[0][i][j] = pop;
                t[1][i][j + 8] = pop.wrapping_neg();
                pop += 1;
            } else {
                let v = pop.wrapping_sub(16).wrapping_sub(j as u8);
                t[0][i][j] = v;
                t[1][i][j + 8] = 8u8.wrapping_sub(v);
            }
            j += 1;
        }
        while j < 16 {
            t[0][i][j] = pop;
            j += 1;
        }
        i += 1;
    }
    t
}

const fn p16_tabs() -> ([[u8; 16]; 256], [[u8; 32]; 256]) {
    let (mut a, mut b) = ([[0u8; 16]; 256], [[0u8; 32]; 256]);
    let mut m = 0;
    while m < 256 {
        let (mut la, mut ra, mut lb, mut rb) = (0, 15, 8, 15);
        let mut k = 0;
        while k < 8 {
            if m >> k & 1 != 0 {
                (a[m][ra], b[m][rb]) = (k as u8, 8 + k as u8);
                (ra, rb) = (ra - 1, rb - 1);
            } else {
                (a[m][la], b[m][lb]) = (k as u8, 8 + k as u8);
                (la, lb) = (la + 1, lb + 1);
            }
            k += 1;
        }
        m += 1;
    }
    (a, b)
}

const fn ctabs() -> [[[u8; 8]; 256]; 2] {
    let mut t = [[[0u8; 8]; 256]; 2];
    let mut m = 0;
    while m < 256 {
        let mut q = 0;
        let mut k = 0;
        while k < 8 {
            if m >> k & 1 != 0 {
                (t[0][m][q], t[1][m][q]) = (k as u8, 8 + k as u8);
                q += 1;
            }
            k += 1;
        }
        while q < 8 {
            (t[0][m][q], t[1][m][q]) = (0xff, 0xff);
            q += 1;
        }
        m += 1;
    }
    t
}

const fn nat_unpack_tabs() -> [[[u8; 16]; 4]; 8] {
    let mut t = [[[0u8; 16]; 4]; 8];
    let mut d = 2;
    while d <= 7 {
        let mut k = 0;
        while k < 16 {
            let (b, s) = (k * d / 8, k * d % 8);
            (t[d][k / 8][k % 8 * 2], t[d][k / 8][k % 8 * 2 + 1]) = (b as u8, b as u8 + 1);
            t[d][2][k % 8 * 2] = (s as u8).wrapping_neg();
            t[d][3][k] = (1 << d) - 1;
            k += 1;
        }
        d += 1;
    }
    t
}

const fn nat_pack_tabs() -> [[[u8; 16]; 5]; 8] {
    let mut t = [[[0u8; 16]; 5]; 8];
    let mut d = 2;
    while d <= 7 {
        // The pyramid leaves the 8d bits of each u64 at bit 32-4d; odd d shift 4 more to
        // start on a byte.
        let (s, sr) = ((8 - d) as u8, (4 * (d & 1)) as u8);
        let b0 = (32 - 4 * d - sr as usize) / 8;
        t[d][0] = [0xff; 16];
        let mut i = 0;
        while i < d {
            (t[d][0][i], t[d][0][d + i]) = ((b0 + i) as u8, (8 + b0 + i) as u8);
            i += 1;
        }
        let mut i = 0;
        while i < 16 {
            (t[d][1][i], t[d][1][i + 2]) = (s, s);
            (t[d][2][i], t[d][2][i + 2]) = (s, s.wrapping_neg());
            i += 4;
        }
        let mut i = 0;
        while i < 16 {
            (t[d][3][i], t[d][3][i + 4]) = (2 * s, (2 * s).wrapping_neg());
            t[d][4][i] = sr.wrapping_neg();
            i += 8;
        }
        d += 1;
    }
    t
}

// ---- Encoder kernels.

/// Upstream init_neon: s2r lives in 16 registers, one TBL4 maps symbols 0..63 and three
/// TBX4 the other quarters.
///
/// # Safety
///
/// n > 0 is a multiple of 16, ranks and src hold n bytes; the CPU has NEON.
#[target_feature(enable = "neon")]
unsafe fn gather_neon_asm(ranks: *mut u8, src: *const u8, n: usize, s2r: &[u8; 256]) -> u8 {
    // SAFETY: the loads read the 256 bytes of s2r and src[i..i+16], the stores write
    // ranks[i..i+16], i+16 <= n.
    unsafe {
        let t = s2r.as_ptr();
        let (t0, t1) = (vld1q_u8_x4(t), vld1q_u8_x4(t.add(64)));
        let (t2, t3) = (vld1q_u8_x4(t.add(128)), vld1q_u8_x4(t.add(192)));
        let c64 = vdupq_n_u8(64);
        let mut mx = vdupq_n_u8(0);
        for i in (0..n).step_by(16) {
            let mut v = vld1q_u8(src.add(i));
            let mut r = vqtbl4q_u8(t0, v);
            v = vsubq_u8(v, c64);
            r = vqtbx4q_u8(r, t1, v);
            v = vsubq_u8(v, c64);
            r = vqtbx4q_u8(r, t2, v);
            v = vsubq_u8(v, c64);
            r = vqtbx4q_u8(r, t3, v);
            mx = vmaxq_u8(r, mx);
            vst1q_u8(ranks.add(i), r);
        }
        // UMAXV; Go's assembler has none and folds with EXT and UMAX.
        vmaxvq_u8(mx)
    }
}

/// The bit weights 1, 2, .., 128 of the lanes of each 8-lane chunk.
#[inline(always)]
unsafe fn bit_weights() -> uint8x16_t {
    // SAFETY: the CPU has NEON.
    unsafe { vreinterpretq_u8_u64(vdupq_n_u64(0x8040201008040201)) }
}

/// MASKS64: the partition masks of the 64 ranks in v (byte c for ranks 8c..8c+7): compare
/// with t, keep the bit weights w and sum each 8-lane chunk with a pairwise-add tree
/// (upstream masks64v_neon).
#[inline(always)]
unsafe fn masks64(v: uint8x16x4_t, t: uint8x16_t, w: uint8x16_t) -> uint8x8_t {
    // SAFETY: the CPU has NEON.
    unsafe {
        let a = vpaddq_u8(vandq_u8(vcgtq_u8(v.0, t), w), vandq_u8(vcgtq_u8(v.1, t), w));
        let b = vpaddq_u8(vandq_u8(vcgtq_u8(v.2, t), w), vandq_u8(vcgtq_u8(v.3, t), w));
        let a = vpaddq_u8(a, b);
        vget_low_u8(vpaddq_u8(a, a))
    }
}

/// MASK8: the partition mask of the 8 ranks in v's low half, whose high half is zero.
#[inline(always)]
unsafe fn mask8(v: uint8x16_t, t: uint8x16_t, w: uint8x16_t) -> u8 {
    // SAFETY: the CPU has NEON.
    unsafe { vaddv_u8(vget_low_u8(vandq_u8(vcgtq_u8(v, t), w))) }
}

/// Returns the low lane of the masks and the running popcounts of their bytes.
#[inline(always)]
unsafe fn mask_pops(mv: uint8x8_t) -> (u64, u64) {
    // SAFETY: the CPU has NEON.
    unsafe {
        let m = vget_lane_u64::<0>(vreinterpret_u64_u8(mv));
        (m, vget_lane_u64::<0>(vreinterpret_u64_u8(vcnt_u8(mv))))
    }
}

/// P16: scatters the 16 ranks of v, whose masks are the low 16 bits of m and the popcount
/// of the first mask the low byte of pops. One TBL puts the left ranks in order at the
/// front and the right ones reversed at the back, a second one (rev) reverses (upstream
/// part64_full_neon). The stores are 16 bytes at l and r; the junk after each side is
/// overwritten by the next group or ends inside the step's 64 ranks.
///
/// # Safety
///
/// l and r are writable for 16 bytes; the CPU has NEON.
#[inline(always)]
unsafe fn p16(v: uint8x16_t, m: u64, pops: u64, rev: uint8x16_t, l: *mut u8, r: *mut u8) {
    let (m0, m1, pc0) = (m as u8, (m >> 8) as u8, usize::from(pops as u8));
    // SAFETY: pc0 <= 8 keeps the P16_B load inside its 32-byte row; the caller guarantees the
    // stores.
    unsafe {
        let a = vld1q_u8(P16_A[usize::from(m0)].as_ptr());
        let b = vld1q_u8(P16_B[usize::from(m1)].as_ptr().add(pc0));
        let x = vqtbl1q_u8(v, vorrq_u8(a, b));
        vst1q_u8(l, x);
        vst1q_u8(r, vqtbl1q_u8(x, rev));
    }
}

/// # Safety
///
/// n > 0 is a multiple of 64, ranks and right hold n bytes and bm n/8; the CPU has NEON.
#[target_feature(enable = "neon")]
unsafe fn part_full_neon_asm(
    ranks: *mut u8,
    right: *mut u8,
    bm: *mut u8,
    n: usize,
    thr: u8,
) -> usize {
    // SAFETY: a step at i loads ranks[i..i+64] before it stores; chunk g with c right ranks
    // before it stores 16 bytes at ranks[nl+16g-c..] and right[nr+c..], nl, nr <= i, so
    // inside the step's 64 bytes.
    unsafe {
        let (t, w) = (vdupq_n_u8(thr), bit_weights());
        let rev = vcombine_u8(
            vcreate_u8(0x08090a0b0c0d0e0f),
            vcreate_u8(0x0001020304050607),
        );
        let (mut nl, mut nr) = (0, 0);
        for i in (0..n).step_by(64) {
            // The left stores land in the step's own ranks, so all 64 are loaded first.
            let v = vld1q_u8_x4(ranks.add(i));
            let (m, pops) = mask_pops(masks64(v, t, w));
            bm.add(i / 8).cast::<u64>().write_unaligned(m);
            let pc = pops.wrapping_mul(0x0101010101010101);
            for (g, x) in [v.0, v.1, v.2, v.3].into_iter().enumerate() {
                let c = usize::from((pc << 8 >> (16 * g)) as u8);
                let (l, r) = (ranks.add(nl + 16 * g - c), right.add(nr + c));
                p16(x, m >> (16 * g), pops >> (16 * g), rev, l, r);
            }
            let tr = (pc >> 56) as usize;
            (nl, nr) = (nl + 64 - tr, nr + tr);
        }
        nr
    }
}

/// C8: stores the right ranks of 8 lanes of v (mask m, CTAB half tab) at r; the junk after
/// them is overwritten by the next chunk.
///
/// # Safety
///
/// r is writable for 8 bytes; the CPU has NEON.
#[inline(always)]
unsafe fn c8(v: uint8x16_t, tab: &[[u8; 8]; 256], m: u8, r: *mut u8) {
    // SAFETY: the caller guarantees the store.
    unsafe { vst1_u8(r, vqtbl1_u8(v, vld1_u8(tab[usize::from(m)].as_ptr()))) }
}

/// Upstream part_core_neon with EMIT_RIGHT.
///
/// # Safety
///
/// n > 0 is a multiple of 8, ranks and right hold n bytes and bm n/8; the CPU has NEON.
#[target_feature(enable = "neon")]
unsafe fn part_right_neon_asm(
    ranks: *const u8,
    right: *mut u8,
    bm: *mut u8,
    n: usize,
    thr: u8,
) -> usize {
    // SAFETY: steps read ranks[i..] and write bm[i/8..] inside n; chunk h of a step stores 8
    // bytes at right[nr+c..], c right ranks before it, so below i+8h+8 <= n.
    unsafe {
        let (t, w) = (vdupq_n_u8(thr), bit_weights());
        let (mut i, mut nr) = (0, 0);
        while n - i >= 64 {
            let v = vld1q_u8_x4(ranks.add(i));
            let (m, pops) = mask_pops(masks64(v, t, w));
            bm.add(i / 8).cast::<u64>().write_unaligned(m);
            let pc = pops.wrapping_mul(0x0101010101010101);
            for (h, x) in [v.0, v.0, v.1, v.1, v.2, v.2, v.3, v.3]
                .into_iter()
                .enumerate()
            {
                let c = usize::from((pc << 8 >> (8 * h)) as u8);
                c8(x, &CTAB[h & 1], (m >> (8 * h)) as u8, right.add(nr + c));
            }
            (i, nr) = (i + 64, nr + (pc >> 56) as usize);
        }
        while i < n {
            let v = vcombine_u8(vld1_u8(ranks.add(i)), vdup_n_u8(0));
            let m = mask8(v, t, w);
            *bm.add(i / 8) = m;
            c8(v, &CTAB[0], m, right.add(nr));
            (i, nr) = (i + 8, nr + m.count_ones() as usize);
        }
        nr
    }
}

/// Upstream part_core_neon without EMIT_RIGHT.
///
/// # Safety
///
/// n > 0 is a multiple of 8, ranks holds n bytes and bm n/8; the CPU has NEON.
#[target_feature(enable = "neon")]
unsafe fn part_none_neon_asm(ranks: *const u8, bm: *mut u8, n: usize, thr: u8) -> usize {
    // SAFETY: steps read ranks[i..] and write bm[i/8..] inside n.
    unsafe {
        let (t, w) = (vdupq_n_u8(thr), bit_weights());
        let (mut i, mut nr) = (0, 0);
        while n - i >= 64 {
            let mv = masks64(vld1q_u8_x4(ranks.add(i)), t, w);
            vst1_u8(bm.add(i / 8), mv);
            (i, nr) = (i + 64, nr + usize::from(vaddlv_u8(vcnt_u8(mv))));
        }
        while i < n {
            let m = mask8(vcombine_u8(vld1_u8(ranks.add(i)), vdup_n_u8(0)), t, w);
            *bm.add(i / 8) = m;
            (i, nr) = (i + 8, nr + m.count_ones() as usize);
        }
        nr
    }
}

/// VSLI chain: a with each b inserted in turn at bit n, above the bits already there.
macro_rules! sli {
    ($a:expr $(, $n:literal: $b:expr)*) => {{
        let x = $a;
        $(let x = vsliq_n_u8::<$n>(x, $b);)*
        x
    }};
}

/// Upstream vert_pack_neon_dN, vert512_pack_neon_dN: each column is built with VSLI in step
/// order, which only inserts above the bits already there.
///
/// # Safety
///
/// n > 0 is a multiple of 8*ln, 2 <= d <= 7 and every rank-base < 1<<d; ranks holds n bytes
/// and dst n*d/8; the CPU has NEON.
#[target_feature(enable = "neon")]
unsafe fn vert_pack_neon_asm(dst: *mut u8, ranks: *const u8, n: usize, ln: usize, d: u8, base: u8) {
    // SAFETY: the caller sized dst and ranks as vp_loop requires.
    unsafe {
        let b = vdupq_n_u8(base);
        match d {
            2 => vp_loop::<2>(dst, ranks, n, ln, b, |v| {
                [
                    sli!(v[0], 2: v[1], 4: v[2], 6: v[3]),
                    sli!(v[4], 2: v[5], 4: v[6], 6: v[7]),
                ]
            }),
            3 => vp_loop::<3>(dst, ranks, n, ln, b, |v| {
                [
                    sli!(v[0], 3: v[1], 6: v[2]),
                    sli!(vshrq_n_u8::<2>(v[2]), 1: v[3], 4: v[4], 7: v[5]),
                    sli!(vshrq_n_u8::<1>(v[5]), 2: v[6], 5: v[7]),
                ]
            }),
            4 => vp_loop::<4>(dst, ranks, n, ln, b, |v| {
                [
                    sli!(v[0], 4: v[1]),
                    sli!(v[2], 4: v[3]),
                    sli!(v[4], 4: v[5]),
                    sli!(v[6], 4: v[7]),
                ]
            }),
            5 => vp_loop::<5>(dst, ranks, n, ln, b, |v| {
                [
                    sli!(v[0], 5: v[1]),
                    sli!(vshrq_n_u8::<3>(v[1]), 2: v[2], 7: v[3]),
                    sli!(vshrq_n_u8::<1>(v[3]), 4: v[4]),
                    sli!(vshrq_n_u8::<4>(v[4]), 1: v[5], 6: v[6]),
                    sli!(vshrq_n_u8::<2>(v[6]), 3: v[7]),
                ]
            }),
            6 => vp_loop::<6>(dst, ranks, n, ln, b, |v| {
                [
                    sli!(v[0], 6: v[1]),
                    sli!(vshrq_n_u8::<2>(v[1]), 4: v[2]),
                    sli!(vshrq_n_u8::<4>(v[2]), 2: v[3]),
                    sli!(v[4], 6: v[5]),
                    sli!(vshrq_n_u8::<2>(v[5]), 4: v[6]),
                    sli!(vshrq_n_u8::<4>(v[6]), 2: v[7]),
                ]
            }),
            _ => vp_loop::<7>(dst, ranks, n, ln, b, |v| {
                [
                    sli!(v[0], 7: v[1]),
                    sli!(vshrq_n_u8::<1>(v[1]), 6: v[2]),
                    sli!(vshrq_n_u8::<2>(v[2]), 5: v[3]),
                    sli!(vshrq_n_u8::<3>(v[3]), 4: v[4]),
                    sli!(vshrq_n_u8::<4>(v[4]), 3: v[5]),
                    sli!(vshrq_n_u8::<5>(v[5]), 2: v[6]),
                    sli!(vshrq_n_u8::<6>(v[6]), 1: v[7]),
                ]
            }),
        }
    }
}

/// The vertical pack walks 16-lane groups of ln-lane blocks: f builds the D columns of a
/// group from its 8 steps (VP_LOAD: base subtracted), which are stored at a stride of ln.
///
/// # Safety
///
/// n is a multiple of 8*ln, ln of 16; ranks holds n bytes and dst n*D/8; the CPU has NEON.
#[inline(always)]
unsafe fn vp_loop<const D: usize>(
    mut dst: *mut u8,
    mut ranks: *const u8,
    mut n: usize,
    ln: usize,
    base: uint8x16_t,
    f: impl Fn([uint8x16_t; 8]) -> [uint8x16_t; D],
) {
    // SAFETY: a block reads its 8*ln ranks and writes its D*ln column bytes.
    unsafe {
        while n > 0 {
            for g in (0..ln).step_by(16) {
                let v = core::array::from_fn(|s| vsubq_u8(vld1q_u8(ranks.add(g + s * ln)), base));
                for (j, c) in f(v).into_iter().enumerate() {
                    vst1q_u8(dst.add(g + j * ln), c);
                }
            }
            (dst, ranks, n) = (dst.add(D * ln), ranks.add(8 * ln), n - 8 * ln);
        }
    }
}

/// Packs 16-code groups of the natural layout while a 16-byte store fits in dst[..dst_len]
/// and returns the number of codes done, a multiple of 16. D < 8 is upstream's converging
/// pyramid (pivco_huffman_neon_pack.h) for every d: each USHL level shifts the halves of a
/// lane pair towards each other until u64 lane h holds codes 8h..8h+7, and a TBL compacts
/// their 2d bytes.
///
/// # Safety
///
/// n >= 16, 2 <= d <= 8 and every rank-base < 1<<d; ranks holds n bytes and dst dst_len,
/// at least n for d=8, which stores n & !15 bytes without consulting dst_len; the CPU has
/// NEON.
#[target_feature(enable = "neon")]
unsafe fn nat_pack_neon_asm(
    mut dst: *mut u8,
    mut ranks: *const u8,
    n: usize,
    mut dst_len: usize,
    d: u8,
    base: u8,
) -> usize {
    // SAFETY: every load reads 16 or 64 of the left ranks; every store writes 16 (d<8) or as
    // many bytes as codes (d=8, dst_len = n) inside dst_len.
    unsafe {
        let b = vdupq_n_u8(base);
        let mut left = n;
        if d == 8 {
            while left >= 64 {
                let v = vld1q_u8_x4(ranks);
                let v = uint8x16x4_t(
                    vsubq_u8(v.0, b),
                    vsubq_u8(v.1, b),
                    vsubq_u8(v.2, b),
                    vsubq_u8(v.3, b),
                );
                vst1q_u8_x4(dst, v);
                (dst, ranks, left) = (dst.add(64), ranks.add(64), left - 64);
            }
            while left >= 16 {
                vst1q_u8(dst, vsubq_u8(vld1q_u8(ranks), b));
                (dst, ranks, left) = (dst.add(16), ranks.add(16), left - 16);
            }
            return n - left;
        }
        let t = &NAT_PACK_TAB[usize::from(d)];
        let tb = vld1q_u8(t[0].as_ptr());
        let s8 = vreinterpretq_s8_u8(vld1q_u8(t[1].as_ptr()));
        let s16 = vreinterpretq_s16_u8(vld1q_u8(t[2].as_ptr()));
        let s32 = vreinterpretq_s32_u8(vld1q_u8(t[3].as_ptr()));
        let s64 = vreinterpretq_s64_u8(vld1q_u8(t[4].as_ptr()));
        let w = 2 * usize::from(d);
        while left >= 16 && dst_len >= 16 {
            let v = vshlq_u8(vsubq_u8(vld1q_u8(ranks), b), s8);
            let v = vshlq_u16(vreinterpretq_u16_u8(v), s16);
            let v = vshlq_u32(vreinterpretq_u32_u16(v), s32);
            let v = vshlq_u64(vreinterpretq_u64_u32(v), s64);
            vst1q_u8(dst, vqtbl1q_u8(vreinterpretq_u8_u64(v), tb));
            (dst, dst_len) = (dst.add(w), dst_len - w);
            (ranks, left) = (ranks.add(16), left - 16);
        }
        n - left
    }
}

// ---- Decoder: exec_neon_asm runs a whole Program with the op checks of exec_generic and of
// the amd64 executors (Go _generate/pivco dec.go), so the per-op cost is a dispatch, not a
// call. Like theirs, its writes are exact and its reads of the input stay inside it; only
// the natural unpack reads past an op's region.

/// A Program's buffers, every arena pointer derived from one `arena.as_mut_ptr()`.
struct Bufs<'a> {
    arena: *mut u8,
    /// max(len(arena) - ARENA_SLACK, 0): ops address arena[..alen].
    alen: u64,
    dst: *mut u8,
    dlen: u64,
    input: *const u8,
    ilen: u64,
    r2s: Option<&'a [u8; R2S_LEN]>,
}

/// Runs p.ops and returns the index of the first op that fails its checks.
///
/// # Safety
///
/// The CPU has NEON.
#[target_feature(enable = "neon")]
unsafe fn exec_neon_asm(p: &mut Program<'_>) -> Result<(), usize> {
    let b = Bufs {
        alen: p.arena.len().saturating_sub(ARENA_SLACK) as u64,
        arena: p.arena.as_mut_ptr(),
        dlen: p.dst.len() as u64,
        dst: p.dst.as_mut_ptr(),
        ilen: p.input.len() as u64,
        input: p.input.as_ptr(),
        r2s: p.r2s,
    };
    for (i, op) in p.ops.iter().enumerate() {
        // SAFETY: b holds the Program's buffers, the arena ARENA_SLACK bytes past alen, and
        // the CPU has NEON.
        if !unsafe { exec_op(&b, op) } {
            return Err(i);
        }
    }
    Ok(())
}

/// Checks op against the buffers and runs it; reports whether it passed.
///
/// # Safety
///
/// b holds valid buffers, the arena ARENA_SLACK bytes past alen; the CPU has NEON.
#[inline(always)]
unsafe fn exec_op(b: &Bufs<'_>, op: &Op) -> bool {
    // Output: Out+K must fit Dst, or the arena less its slack.
    let (k, bm) = (u64::from(op.k), u64::from(op.bm));
    let (o, lim) = if op.flags & FLAG_DST != 0 {
        (b.dst, b.dlen)
    } else {
        (b.arena, b.alen)
    };
    if u64::from(op.out) + k > lim {
        return false;
    }
    // SAFETY: out+k <= lim, the length of o's buffer.
    let o = unsafe { o.add(op.out as usize) };
    let ku = op.k as usize;
    match op.kind {
        OP_VV | OP_CV => {
            // BITMAP_CHECK and ARENA_CHECK of the children.
            let (l, r) = (u64::from(op.l), u64::from(op.r));
            let vv = op.kind == OP_VV;
            if bm + k.div_ceil(8) > b.ilen || (vv && l + k > b.alen) || r + k > b.alen {
                return false;
            }
            // SAFETY: o has k bytes, the bitmap ceil(k/8) in the input, and the children k
            // bytes below alen, so their loads 15 bytes past them stay in the arena.
            unsafe {
                let (bp, a) = (b.input.add(bm as usize), b.arena);
                if vv {
                    merge::<true>(o, ku, bp, a.add(l as usize), a.add(r as usize), 0, op.kr)
                } else {
                    merge::<false>(o, ku, bp, a, a.add(r as usize), op.p, op.kr)
                }
            }
        }
        OP_CC => {
            if bm + k.div_ceil(8) > b.ilen {
                return false;
            }
            // SAFETY: o has k bytes, the bitmap ceil(k/8) in the input.
            unsafe { merge_cc(o, ku, b.input.add(bm as usize), op.p, op.q) };
            true
        }
        OP_V512 | OP_V128 => {
            // Flat, vertical layout of ln lanes: D in 2..7, region K*D/8 bytes.
            let ln = if op.kind == OP_V512 { 64 } else { 16 };
            if !ku.is_multiple_of(8 * ln) || !(2..=7).contains(&op.d) {
                return false;
            }
            let Some((src, lut)) = flat_check(b, op, k * u64::from(op.d) / 8) else {
                return false;
            };
            // SAFETY: o has k bytes, the region k*d/8 and lut 1<<d, at least 64 (R2S_LEN).
            unsafe { vert_unpack(o, src, ku, ln, op.d, lut) };
            true
        }
        OP_NAT => {
            // Flat, natural layout: D in 2..8, region ceil(K*D/8) bytes.
            if !(2..=8).contains(&op.d) {
                return false;
            }
            let Some((src, lut)) = flat_check(b, op, (k * u64::from(op.d)).div_ceil(8)) else {
                return false;
            };
            // SAFETY: as for the vertical layout; the input holds ilen-bm bytes from src on.
            unsafe { nat_unpack(o, src, (b.ilen - bm) as usize, ku, op.d, lut) };
            true
        }
        OP_COPY => {
            // D = 8 flat: the codes are the symbols (upstream merge_flat_d8_neon).
            if op.d != 8 || bm + k > b.ilen {
                return false;
            }
            // SAFETY: o and the region have k bytes; the input is not an output buffer.
            unsafe { core::ptr::copy_nonoverlapping(b.input.add(bm as usize), o, ku) };
            true
        }
        _ => false,
    }
}

/// FLAT_CHECK: the need-byte region at Bm must lie in the input, R2S be set and
/// R2S[P..P+2^D] inside it; returns the region and R2S[P..]. D must be 2..8.
#[inline(always)]
fn flat_check(b: &Bufs<'_>, op: &Op, need: u64) -> Option<(*const u8, *const u8)> {
    let (bm, p) = (u64::from(op.bm), usize::from(op.p));
    let r2s = b.r2s?;
    if bm + need > b.ilen || p + (1 << op.d) > R2S_LEN {
        return None;
    }
    // SAFETY: bm <= ilen and p < R2S_LEN.
    Some(unsafe { (b.input.add(bm as usize), r2s.as_ptr().add(p)) })
}

/// MERGE_IDX: the merge index of the 16 outputs whose bitmap is the low 16 bits of m.
#[inline(always)]
unsafe fn merge_idx(m: u64) -> uint8x16_t {
    let (t0, t1) = (
        &MERGE_TAB[0][m as u8 as usize],
        &MERGE_TAB[1][(m >> 8) as u8 as usize],
    );
    // SAFETY: the rows are 16 bytes; the CPU has NEON.
    unsafe { vsubq_u8(vld1q_u8(t0.as_ptr()), vld1q_u8(t1.as_ptr())) }
}

/// STEP64: the 64 bitmap bits at bm and the running popcounts of their bytes.
#[inline(always)]
unsafe fn step64(bm: *const u8) -> (u64, u64) {
    // SAFETY: the caller guarantees 8 readable bytes; the CPU has NEON.
    unsafe {
        let m = bm.cast::<u64>().read_unaligned();
        let c = vget_lane_u64::<0>(vreinterpret_u64_u8(vcnt_u8(vcreate_u8(m))));
        (m, c.wrapping_mul(0x0101010101010101))
    }
}

/// TAIL_BITS: the bitmap bits at bm below n = 1..15, reading exactly ceil(n/8) bytes: the
/// second byte is the first one again unless n > 8.
#[inline(always)]
unsafe fn tail_bits(bm: *const u8, n: usize) -> u64 {
    // SAFETY: the caller guarantees ceil(n/8) readable bytes.
    let (b0, b1) = unsafe { (*bm, *bm.add(usize::from(n > 8))) };
    (u64::from(b0) | u64::from(b1) << 8) & ((1 << n) - 1)
}

/// tail_store: stores the low n = 0..15 bytes of v to o.
#[inline(always)]
unsafe fn tail_store(mut o: *mut u8, n: usize, v: uint8x16_t) {
    // SAFETY: the caller guarantees n writable bytes; each store is inside them.
    unsafe {
        let v = vreinterpretq_u64_u8(v);
        let mut w = vgetq_lane_u64::<0>(v);
        if n & 8 != 0 {
            o.cast::<u64>().write_unaligned(w);
            (o, w) = (o.add(8), vgetq_lane_u64::<1>(v));
        }
        if n & 4 != 0 {
            o.cast::<u32>().write_unaligned(w as u32);
            (o, w) = (o.add(4), w >> 32);
        }
        if n & 2 != 0 {
            o.cast::<u16>().write_unaligned(w as u16);
            (o, w) = (o.add(2), w >> 16);
        }
        if n & 1 != 0 {
            *o = w as u8;
        }
    }
}

/// Merges k outputs to o (upstream merge_vec_vec_neon, merge_cst_vec_neon): bit j of bm
/// takes the next right value, else the next left value (VV) or p (CV). It reports whether
/// the right cursor advanced by exactly kr.
///
/// The index of 16 outputs is a subtraction of two MERGE_TAB rows: right lanes index the
/// right child with a TBX, left lanes the left one with a TBL of 240 minus the index. Each
/// 16 outputs are loaded before they are stored, which keeps the in-place layout safe. In
/// a 64-output step, chunk g starts 16g-c left and c right values in, c being the running
/// popcount byte of the step's bitmap word before it. The last k&15 outputs are one
/// 16-output step on the bitmap bits below k, stored exactly; its 16-byte loads at the
/// cursors reach 15 bytes past the k bytes of the child.
///
/// # Safety
///
/// o is writable for k bytes, bm readable for ceil(k/8), r and (VV) l readable for k+15;
/// pointers into one buffer derive from one pointer; the CPU has NEON.
#[inline(always)]
unsafe fn merge<const VV: bool>(
    o: *mut u8,
    k: usize,
    bm: *const u8,
    l: *const u8,
    r: *const u8,
    p: u8,
    kr: u32,
) -> bool {
    // SAFETY: the cursors nl, nr <= j stay below the k values of their child, so a 16-byte
    // load ends at most 15 bytes past them; outputs j..j+16 are stored while j+16 <= k.
    unsafe {
        let (pv, v240) = (vdupq_n_u8(p), vdupq_n_u8(240));
        // VV16 and CV16 without the store.
        let m16 = |m: u64, nl: usize, nr: usize| {
            let i = merge_idx(m);
            let x = if VV {
                vqtbl1q_u8(vld1q_u8(l.add(nl)), vsubq_u8(v240, i))
            } else {
                pv
            };
            vqtbx1q_u8(x, vld1q_u8(r.add(nr)), i)
        };
        let (mut j, mut nr, mut bm) = (0, 0, bm);
        while k - j >= 64 {
            let (m, pc) = step64(bm);
            for g in 0..4 {
                let c = usize::from((pc << 8 >> (16 * g)) as u8);
                vst1q_u8(
                    o.add(j + 16 * g),
                    m16(m >> (16 * g), j - nr + 16 * g - c, nr + c),
                );
            }
            (j, nr, bm) = (j + 64, nr + (pc >> 56) as usize, bm.add(8));
        }
        while k - j >= 16 {
            let m = u64::from(bm.cast::<u16>().read_unaligned());
            vst1q_u8(o.add(j), m16(m, j - nr, nr));
            (j, nr, bm) = (j + 16, nr + m.count_ones() as usize, bm.add(2));
        }
        let mut v = vdupq_n_u8(0);
        if j < k {
            let m = tail_bits(bm, k - j);
            v = m16(m, j - nr, nr);
            nr += m.count_ones() as usize;
        }
        // The tail is stored after the check, so a failing op leaves its last k&15 outputs
        // unwritten.
        if nr as u64 != u64::from(kr) {
            return false;
        }
        tail_store(o.add(j), k - j, v);
    }
    true
}

/// out[j] = bit j ? q : p (upstream merge_cst_cst_neon): each bitmap bit is spread to its
/// output byte and selects q over p.
///
/// # Safety
///
/// o is writable for k bytes, bm readable for ceil(k/8); the CPU has NEON.
#[inline(always)]
unsafe fn merge_cc(o: *mut u8, k: usize, bm: *const u8, p: u8, q: u8) {
    // SAFETY: 16 outputs are stored while j+16 <= k, from the 2 bitmap bytes at j/8.
    unsafe {
        let (pv, qv, w) = (vdupq_n_u8(p), vdupq_n_u8(q), bit_weights());
        let sel = vcombine_u8(vdup_n_u8(0), vdup_n_u8(1));
        let spread = |m: u16| {
            let x = vqtbl1q_u8(vreinterpretq_u8_u16(vdupq_n_u16(m)), sel);
            vbslq_u8(vtstq_u8(x, w), qv, pv)
        };
        let mut j = 0;
        while k - j >= 16 {
            vst1q_u8(
                o.add(j),
                spread(bm.add(j / 8).cast::<u16>().read_unaligned()),
            );
            j += 16;
        }
        if j < k {
            tail_store(
                o.add(j),
                k - j,
                spread(tail_bits(bm.add(j / 8), k - j) as u16),
            );
        }
    }
}

// Field extraction for the vertical unpack: the d-bit field at bit r of column a, continued
// in column b when it crosses into it; m is the code mask.
macro_rules! low {
    ($a:expr, $m:expr) => {
        vandq_u8($a, $m)
    };
}
macro_rules! mid {
    ($r:literal, $a:expr, $m:expr) => {
        vandq_u8(vshrq_n_u8::<$r>($a), $m)
    };
}
macro_rules! top {
    ($r:literal, $a:expr) => {
        vshrq_n_u8::<$r>($a)
    };
}
macro_rules! cross {
    ($r:literal, $a:expr, $b:expr, $m:expr) => {
        vandq_u8(vsliq_n_u8::<{ 8 - $r }>(vshrq_n_u8::<$r>($a), $b), $m)
    };
}

/// Flat, vertical layout of ln lanes (upstream vert512_merge_neon, vert_merge_neon): step s
/// of a group is cut from its columns with uniform shifts and mapped through lut.
///
/// # Safety
///
/// k is a multiple of 8*ln, 2 <= d <= 7; o is writable for k bytes, src readable for
/// k*d/8 and lut for 64 (d=7: 128); the CPU has NEON.
#[inline(always)]
unsafe fn vert_unpack(o: *mut u8, src: *const u8, k: usize, ln: usize, d: u8, lut: *const u8) {
    // SAFETY: the caller's sizes are those vu_loop requires.
    unsafe {
        let m = vdupq_n_u8((1 << d) - 1);
        match d {
            2 => vu_loop::<2>(o, src, k, ln, lut, |c| {
                [
                    low!(c[0], m),
                    mid!(2, c[0], m),
                    mid!(4, c[0], m),
                    top!(6, c[0]),
                    low!(c[1], m),
                    mid!(2, c[1], m),
                    mid!(4, c[1], m),
                    top!(6, c[1]),
                ]
            }),
            3 => vu_loop::<3>(o, src, k, ln, lut, |c| {
                [
                    low!(c[0], m),
                    mid!(3, c[0], m),
                    cross!(6, c[0], c[1], m),
                    mid!(1, c[1], m),
                    mid!(4, c[1], m),
                    cross!(7, c[1], c[2], m),
                    mid!(2, c[2], m),
                    top!(5, c[2]),
                ]
            }),
            4 => vu_loop::<4>(o, src, k, ln, lut, |c| {
                [
                    low!(c[0], m),
                    top!(4, c[0]),
                    low!(c[1], m),
                    top!(4, c[1]),
                    low!(c[2], m),
                    top!(4, c[2]),
                    low!(c[3], m),
                    top!(4, c[3]),
                ]
            }),
            5 => vu_loop::<5>(o, src, k, ln, lut, |c| {
                [
                    low!(c[0], m),
                    cross!(5, c[0], c[1], m),
                    mid!(2, c[1], m),
                    cross!(7, c[1], c[2], m),
                    cross!(4, c[2], c[3], m),
                    mid!(1, c[3], m),
                    cross!(6, c[3], c[4], m),
                    top!(3, c[4]),
                ]
            }),
            6 => vu_loop::<6>(o, src, k, ln, lut, |c| {
                [
                    low!(c[0], m),
                    cross!(6, c[0], c[1], m),
                    cross!(4, c[1], c[2], m),
                    top!(2, c[2]),
                    low!(c[3], m),
                    cross!(6, c[3], c[4], m),
                    cross!(4, c[4], c[5], m),
                    top!(2, c[5]),
                ]
            }),
            _ => vu_loop::<7>(o, src, k, ln, lut, |c| {
                [
                    low!(c[0], m),
                    cross!(7, c[0], c[1], m),
                    cross!(6, c[1], c[2], m),
                    cross!(5, c[2], c[3], m),
                    cross!(4, c[3], c[4], m),
                    cross!(3, c[4], c[5], m),
                    cross!(2, c[5], c[6], m),
                    top!(1, c[6]),
                ]
            }),
        }
    }
}

/// The vertical unpack walks 16-lane groups of ln-lane blocks: f cuts the 8 steps of a group
/// from its D columns (at a stride of ln), which are mapped and stored at a stride of ln.
///
/// # Safety
///
/// k is a multiple of 8*ln, ln of 16; o is writable for k bytes, src readable for k*D/8
/// and lut as [`luts`] requires; the CPU has NEON.
#[inline(always)]
unsafe fn vu_loop<const D: usize>(
    mut o: *mut u8,
    mut src: *const u8,
    mut k: usize,
    ln: usize,
    lut: *const u8,
    f: impl Fn([uint8x16_t; D]) -> [uint8x16_t; 8],
) {
    // SAFETY: a block reads its D*ln column bytes and writes its 8*ln outputs.
    unsafe {
        let (lo, hi) = luts::<D>(lut);
        while k > 0 {
            for g in (0..ln).step_by(16) {
                let c = core::array::from_fn(|j| vld1q_u8(src.add(g + j * ln)));
                for (s, x) in f(c).into_iter().enumerate() {
                    vst1q_u8(o.add(g + s * ln), lut_map::<D>(x, lo, hi));
                }
            }
            (o, src, k) = (o.add(8 * ln), src.add(D * ln), k - 8 * ln);
        }
    }
}

/// The R2S rows at lut: 64 bytes, and for D=7 the next 64.
///
/// # Safety
///
/// lut is readable for 64 bytes (D=7: 128); the CPU has NEON.
#[inline(always)]
unsafe fn luts<const D: usize>(lut: *const u8) -> (uint8x16x4_t, uint8x16x4_t) {
    // SAFETY: the caller guarantees the bytes.
    unsafe {
        let lo = vld1q_u8_x4(lut);
        (lo, if D == 7 { vld1q_u8_x4(lut.add(64)) } else { lo })
    }
}

/// Code to symbol lookups: TBL1 for d <= 4, TBL2, TBL4, and for d=7 the OR of a TBL4 of
/// the codes and one of the codes minus 64. Upstream prefers these independent lookups to
/// a TBX4, which waits for the first TBL4.
#[inline(always)]
unsafe fn lut_map<const D: usize>(x: uint8x16_t, lo: uint8x16x4_t, hi: uint8x16x4_t) -> uint8x16_t {
    // SAFETY: the CPU has NEON.
    unsafe {
        match D {
            0..=4 => vqtbl1q_u8(lo.0, x),
            5 => vqtbl2q_u8(uint8x16x2_t(lo.0, lo.1), x),
            6 => vqtbl4q_u8(lo, x),
            _ => vorrq_u8(
                vqtbl4q_u8(hi, vsubq_u8(x, vdupq_n_u8(64))),
                vqtbl4q_u8(lo, x),
            ),
        }
    }
}

/// Flat, natural layout: D in 2..8, k codes from src, left the input bytes from src on.
///
/// # Safety
///
/// o is writable for k bytes, src readable for ceil(k*d/8) <= left bytes and lut for 1<<d
/// (at least 64); the CPU has NEON.
#[inline(always)]
unsafe fn nat_unpack(o: *mut u8, src: *const u8, left: usize, k: usize, d: u8, lut: *const u8) {
    // SAFETY: the caller's sizes are those nat_loop requires; d=8 reads k codes and maps
    // them through the 256-byte lut.
    unsafe {
        match d {
            2 => nat_loop::<2>(o, src, left, k, lut),
            3 => nat_loop::<3>(o, src, left, k, lut),
            4 => nat_loop::<4>(o, src, left, k, lut),
            5 => nat_loop::<5>(o, src, left, k, lut),
            6 => nat_loop::<6>(o, src, left, k, lut),
            7 => nat_loop::<7>(o, src, left, k, lut),
            // nat_d8: 256-entry table, one lookup per code.
            _ => {
                for i in 0..k {
                    *o.add(i) = *lut.add(usize::from(*src.add(i)));
                }
            }
        }
    }
}

/// The natural unpack of D < 8: 16-code groups load a window of 16 bytes (8 for d <= 4,
/// whose groups span at most 8) while it fits in the input. NAT_CODES maps the 16 codes in
/// a window's low 2d bytes: a TBL gathers the byte pair of each code into a u16 lane, USHL
/// shifts the code down, UZP1 keeps the low bytes and lut_map maps them (upstream
/// merge_flat_d7_neon, for every d).
///
/// # Safety
///
/// As [`nat_unpack`].
#[inline(always)]
unsafe fn nat_loop<const D: usize>(
    mut o: *mut u8,
    mut src: *const u8,
    mut left: usize,
    mut k: usize,
    lut: *const u8,
) {
    // SAFETY: a group of 16 codes uses 2D bytes; windows load w <= left bytes and exact loads
    // the group's bytes; 16 outputs are stored while 16 codes are left, else exactly k.
    unsafe {
        let t = &NAT_UNPACK_TAB[D];
        let (g0, g1) = (vld1q_u8(t[0].as_ptr()), vld1q_u8(t[1].as_ptr()));
        let (sh, m) = (
            vreinterpretq_s16_u8(vld1q_u8(t[2].as_ptr())),
            vld1q_u8(t[3].as_ptr()),
        );
        let (lo, hi) = luts::<D>(lut);
        let codes = |v: uint8x16_t| {
            let a = vshlq_u16(vreinterpretq_u16_u8(vqtbl1q_u8(v, g0)), sh);
            let b = vshlq_u16(vreinterpretq_u16_u8(vqtbl1q_u8(v, g1)), sh);
            let x = vuzp1q_u8(vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b));
            lut_map::<D>(vandq_u8(x, m), lo, hi)
        };
        // NAT_FITS
        let w = if D <= 4 { 8 } else { 16 };
        while k >= 16 && left >= w {
            let v = if D <= 4 {
                vcombine_u8(vld1_u8(src), vdup_n_u8(0))
            } else {
                vld1q_u8(src)
            };
            vst1q_u8(o, codes(v));
            (o, src, left, k) = (o.add(16), src.add(2 * D), left - 2 * D, k - 16);
        }
        // nat_tail: the last group, and those whose window would pass the end of the input:
        // a full window while 16 input bytes are left, else exactly the
        // ceil(min(16, k)*D/8) bytes the group uses.
        while k > 0 {
            let v = if left >= 16 {
                vld1q_u8(src)
            } else {
                load_exact(src, (k.min(16) * D).div_ceil(8))
            };
            let x = codes(v);
            if k < 16 {
                tail_store(o, k, x);
                return;
            }
            vst1q_u8(o, x);
            (o, src, left, k) = (o.add(16), src.add(2 * D), left - 2 * D, k - 16);
        }
    }
}

/// nat_exact: the n = 1..14 bytes at p in a vector's low lanes, the others zero.
#[inline(always)]
unsafe fn load_exact(mut p: *const u8, n: usize) -> uint8x16_t {
    // SAFETY: the caller guarantees n readable bytes; each load is inside them.
    unsafe {
        let mut lo = 0;
        if n & 8 != 0 {
            (lo, p) = (p.cast::<u64>().read_unaligned(), p.add(8));
        }
        // The 0..7 bytes after the first 8 gather in hi, b being their bit count.
        let (mut hi, mut b) = (0u64, 0);
        if n & 4 != 0 {
            (hi, p, b) = (u64::from(p.cast::<u32>().read_unaligned()), p.add(4), 32);
        }
        if n & 2 != 0 {
            hi |= u64::from(p.cast::<u16>().read_unaligned()) << b;
            (p, b) = (p.add(2), b + 16);
        }
        if n & 1 != 0 {
            hi |= u64::from(*p) << b;
        }
        if n & 8 == 0 {
            (lo, hi) = (hi, 0);
        }
        vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(lo), vcreate_u64(hi)))
    }
}
