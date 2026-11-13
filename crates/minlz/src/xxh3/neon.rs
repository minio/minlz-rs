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

//! NEON stripe loop (baseline on aarch64). `vmovn`/`vshrn` split each lane into its 32-bit halves
//! for `vmull`, and `vext #1` swaps adjacent lanes for `acc[i ^ 1] += data[i]`. Little-endian
//! only: the byte loads are reinterpreted as 64-bit lanes.

use core::arch::aarch64::*;
use core::mem::transmute;

use super::{KEY, Kernel, PRIME32_1, STRIPE, key};

/// [`super::accum`] on NEON.
///
/// # Safety
///
/// The CPU has NEON.
#[target_feature(enable = "neon")]
pub(super) unsafe fn accum_neon(a: &mut [u64; 8], p: &[u8]) {
    // SAFETY: the caller checked NEON.
    unsafe { super::accum::<Neon>(a, p) }
}

/// [`super::blocks`] on NEON.
///
/// # Safety
///
/// The CPU has NEON.
#[target_feature(enable = "neon")]
pub(super) unsafe fn blocks_neon(a: &mut [u64; 8], p: &[u8], n: usize) {
    // SAFETY: the caller checked NEON.
    unsafe { super::blocks::<Neon>(a, p, n) }
}

/// Loads 16 bytes at p as two little-endian 64-bit lanes.
///
/// # Safety
///
/// p is readable for 16 bytes and the CPU has NEON.
#[inline(always)]
unsafe fn vld(p: *const u8) -> uint64x2_t {
    // SAFETY: forwarded caller contract.
    unsafe { vreinterpretq_u64_u8(vld1q_u8(p)) }
}

struct Neon;

impl Kernel for Neon {
    type Acc = [uint64x2_t; 4];

    #[inline(always)]
    unsafe fn load(a: &[u64; 8]) -> Self::Acc {
        // SAFETY: same size, and every bit pattern is valid for both; lane 0 is the lower address
        // on little-endian.
        unsafe { transmute::<[u64; 8], Self::Acc>(*a) }
    }

    #[inline(always)]
    unsafe fn store(acc: Self::Acc, a: &mut [u64; 8]) {
        // SAFETY: as in load.
        *a = unsafe { transmute::<Self::Acc, [u64; 8]>(acc) };
    }

    #[inline(always)]
    unsafe fn zero() -> Self::Acc {
        // SAFETY: the caller checked NEON.
        [unsafe { vdupq_n_u64(0) }; 4]
    }

    #[inline(always)]
    unsafe fn add(a: Self::Acc, b: Self::Acc) -> Self::Acc {
        // SAFETY: the caller checked NEON.
        core::array::from_fn(|i| unsafe { vaddq_u64(a[i], b[i]) })
    }

    #[inline(always)]
    unsafe fn stripe(acc: &mut Self::Acc, p: &[u8; STRIPE], k: &[u8; STRIPE]) {
        for (i, a) in acc.iter_mut().enumerate() {
            // SAFETY: the caller checked NEON; the loads read 16 bytes inside p and k.
            unsafe {
                let d = vld(p.as_ptr().add(16 * i));
                let dk = veorq_u64(d, vld(k.as_ptr().add(16 * i)));
                let prod = vmull_u32(vmovn_u64(dk), vshrn_n_u64::<32>(dk));
                *a = vaddq_u64(*a, vaddq_u64(prod, vextq_u64::<1>(d, d)));
            }
        }
    }

    #[inline(always)]
    unsafe fn scramble(acc: &mut Self::Acc) {
        for (i, a) in acc.iter_mut().enumerate() {
            // SAFETY: the caller checked NEON; the load reads 16 bytes inside KEY.
            unsafe {
                let prime = vdup_n_u32(PRIME32_1 as u32);
                let k = vld(key(KEY.len() - STRIPE).as_ptr().add(16 * i));
                let x = veorq_u64(veorq_u64(*a, vshrq_n_u64::<47>(*a)), k);
                let lo = vmull_u32(vmovn_u64(x), prime);
                let hi = vmull_u32(vshrn_n_u64::<32>(x), prime);
                *a = vaddq_u64(lo, vshlq_n_u64::<32>(hi));
            }
        }
    }
}
