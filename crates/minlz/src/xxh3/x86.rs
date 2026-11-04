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

//! SSE2, AVX2 and AVX-512 stripe loops. Each 64-bit lane computes the scalar kernel: `shuffle 0x31`
//! puts the high half of a lane next to its low half for `mul_epu32`, and `shuffle 0x4e` swaps
//! adjacent lanes for `acc[i ^ 1] += data[i]`.

use core::arch::x86_64::*;
use core::mem::transmute;

use super::{KEY, Kernel, PRIME32_1, STRIPE, key};

macro_rules! entries {
    ($feature:literal, $k:ty, $accum:ident, $blocks:ident) => {
        /// [`super::accum`] on this unit.
        ///
        /// # Safety
        ///
        /// The CPU has the target features.
        #[target_feature(enable = $feature)]
        pub(super) unsafe fn $accum(a: &mut [u64; 8], p: &[u8]) {
            // SAFETY: the caller checked the features.
            unsafe { super::accum::<$k>(a, p) }
        }

        /// [`super::blocks`] on this unit.
        ///
        /// # Safety
        ///
        /// The CPU has the target features.
        #[target_feature(enable = $feature)]
        pub(super) unsafe fn $blocks(a: &mut [u64; 8], p: &[u8], n: usize) {
            // SAFETY: the caller checked the features.
            unsafe { super::blocks::<$k>(a, p, n) }
        }
    };
}

entries!("sse2", Sse2, accum_sse2, blocks_sse2);
entries!("avx2", Avx2, accum_avx2, blocks_avx2);
entries!("avx512f", Avx512, accum_avx512, blocks_avx512);

/// The scramble key, past the stripe keys.
const SCRAMBLE: usize = KEY.len() - STRIPE;

struct Sse2;

impl Kernel for Sse2 {
    type Acc = [__m128i; 4];

    #[inline(always)]
    unsafe fn load(a: &[u64; 8]) -> Self::Acc {
        // SAFETY: same size, and every bit pattern is valid for both.
        unsafe { transmute::<[u64; 8], Self::Acc>(*a) }
    }

    #[inline(always)]
    unsafe fn store(acc: Self::Acc, a: &mut [u64; 8]) {
        // SAFETY: same size, and every bit pattern is valid for both.
        *a = unsafe { transmute::<Self::Acc, [u64; 8]>(acc) };
    }

    #[inline(always)]
    unsafe fn stripe(acc: &mut Self::Acc, p: &[u8; STRIPE], k: &[u8; STRIPE]) {
        for (i, a) in acc.iter_mut().enumerate() {
            // SAFETY: SSE2 is baseline on x86-64; the loads read 16 bytes inside p and k.
            unsafe {
                let d = _mm_loadu_si128(p.as_ptr().add(16 * i).cast());
                let dk = _mm_xor_si128(d, _mm_loadu_si128(k.as_ptr().add(16 * i).cast()));
                let prod = _mm_mul_epu32(dk, _mm_shuffle_epi32::<0x31>(dk));
                *a = _mm_add_epi64(*a, _mm_add_epi64(prod, _mm_shuffle_epi32::<0x4e>(d)));
            }
        }
    }

    #[inline(always)]
    unsafe fn scramble(acc: &mut Self::Acc) {
        for (i, a) in acc.iter_mut().enumerate() {
            // SAFETY: SSE2 is baseline on x86-64; the load reads 16 bytes inside KEY.
            unsafe {
                let prime = _mm_set1_epi64x(PRIME32_1 as i64);
                let k = _mm_loadu_si128(key(SCRAMBLE).as_ptr().add(16 * i).cast());
                let x = _mm_xor_si128(_mm_xor_si128(*a, _mm_srli_epi64::<47>(*a)), k);
                let lo = _mm_mul_epu32(x, prime);
                let hi = _mm_mul_epu32(_mm_shuffle_epi32::<0x31>(x), prime);
                *a = _mm_add_epi64(lo, _mm_slli_epi64::<32>(hi));
            }
        }
    }
}

struct Avx2;

impl Kernel for Avx2 {
    type Acc = [__m256i; 2];

    #[inline(always)]
    unsafe fn load(a: &[u64; 8]) -> Self::Acc {
        // SAFETY: same size, and every bit pattern is valid for both.
        unsafe { transmute::<[u64; 8], Self::Acc>(*a) }
    }

    #[inline(always)]
    unsafe fn store(acc: Self::Acc, a: &mut [u64; 8]) {
        // SAFETY: same size, and every bit pattern is valid for both.
        *a = unsafe { transmute::<Self::Acc, [u64; 8]>(acc) };
    }

    #[inline(always)]
    unsafe fn stripe(acc: &mut Self::Acc, p: &[u8; STRIPE], k: &[u8; STRIPE]) {
        for (i, a) in acc.iter_mut().enumerate() {
            // SAFETY: the caller checked AVX2; the loads read 32 bytes inside p and k.
            unsafe {
                let d = _mm256_loadu_si256(p.as_ptr().add(32 * i).cast());
                let dk = _mm256_xor_si256(d, _mm256_loadu_si256(k.as_ptr().add(32 * i).cast()));
                let prod = _mm256_mul_epu32(dk, _mm256_shuffle_epi32::<0x31>(dk));
                *a = _mm256_add_epi64(*a, _mm256_add_epi64(prod, _mm256_shuffle_epi32::<0x4e>(d)));
            }
        }
    }

    #[inline(always)]
    unsafe fn scramble(acc: &mut Self::Acc) {
        for (i, a) in acc.iter_mut().enumerate() {
            // SAFETY: the caller checked AVX2; the load reads 32 bytes inside KEY.
            unsafe {
                let prime = _mm256_set1_epi64x(PRIME32_1 as i64);
                let k = _mm256_loadu_si256(key(SCRAMBLE).as_ptr().add(32 * i).cast());
                let x = _mm256_xor_si256(_mm256_xor_si256(*a, _mm256_srli_epi64::<47>(*a)), k);
                let lo = _mm256_mul_epu32(x, prime);
                let hi = _mm256_mul_epu32(_mm256_shuffle_epi32::<0x31>(x), prime);
                *a = _mm256_add_epi64(lo, _mm256_slli_epi64::<32>(hi));
            }
        }
    }

    // zeebo/xxh3 prefetches a block ahead in its AVX2 and AVX-512 loops.
    #[inline(always)]
    unsafe fn prefetch(p: *const u8) {
        // SAFETY: prefetching never faults.
        unsafe { _mm_prefetch::<_MM_HINT_T0>(p.cast()) }
    }
}

struct Avx512;

impl Kernel for Avx512 {
    type Acc = __m512i;

    #[inline(always)]
    unsafe fn load(a: &[u64; 8]) -> Self::Acc {
        // SAFETY: same size, and every bit pattern is valid for both.
        unsafe { transmute::<[u64; 8], Self::Acc>(*a) }
    }

    #[inline(always)]
    unsafe fn store(acc: Self::Acc, a: &mut [u64; 8]) {
        // SAFETY: same size, and every bit pattern is valid for both.
        *a = unsafe { transmute::<Self::Acc, [u64; 8]>(acc) };
    }

    #[inline(always)]
    unsafe fn stripe(acc: &mut Self::Acc, p: &[u8; STRIPE], k: &[u8; STRIPE]) {
        // SAFETY: the caller checked AVX-512F; the loads read p and k.
        unsafe {
            let d = _mm512_loadu_si512(p.as_ptr().cast());
            let dk = _mm512_xor_si512(d, _mm512_loadu_si512(k.as_ptr().cast()));
            let prod = _mm512_mul_epu32(dk, _mm512_shuffle_epi32::<0x31>(dk));
            *acc = _mm512_add_epi64(
                *acc,
                _mm512_add_epi64(prod, _mm512_shuffle_epi32::<0x4e>(d)),
            );
        }
    }

    #[inline(always)]
    unsafe fn scramble(acc: &mut Self::Acc) {
        // SAFETY: the caller checked AVX-512F; the load reads 64 bytes inside KEY.
        unsafe {
            let prime = _mm512_set1_epi64(PRIME32_1 as i64);
            let k = _mm512_loadu_si512(key(SCRAMBLE).as_ptr().cast());
            let x = _mm512_xor_si512(_mm512_xor_si512(*acc, _mm512_srli_epi64::<47>(*acc)), k);
            let lo = _mm512_mul_epu32(x, prime);
            let hi = _mm512_mul_epu32(_mm512_shuffle_epi32::<0x31>(x), prime);
            *acc = _mm512_add_epi64(lo, _mm512_slli_epi64::<32>(hi));
        }
    }

    #[inline(always)]
    unsafe fn prefetch(p: *const u8) {
        // SAFETY: prefetching never faults.
        unsafe { _mm_prefetch::<_MM_HINT_T0>(p.cast()) }
    }
}
