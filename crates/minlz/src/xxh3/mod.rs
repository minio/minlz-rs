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

//! XXH3-64 with seed 0 and the default secret: the checksum of MinLZ SIMD stream chunks (0x10).
//!
//! Implements XXH3 as defined by the xxHash specification by Yann Collet
//! (<https://github.com/Cyan4973/xxHash/blob/dev/doc/xxhash_spec.md>), bit-exact with
//! `github.com/zeebo/xxh3`, which MinLZ Go uses. The stripe loop of inputs over 240 bytes runs on
//! the best vector unit of the CPU, picked once: AVX-512, AVX2 or SSE2 on x86-64, NEON on aarch64,
//! portable code elsewhere. As in zeebo/xxh3, AVX-512 only takes inputs of at least 1 KiB.

use std::sync::OnceLock;

#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
mod neon;
#[cfg(test)]
mod tests;
#[cfg(all(target_arch = "x86_64", not(miri)))]
mod x86;

const STRIPE: usize = 64;
const BLOCK: usize = 1024;
/// Inputs of at least this size take AVX-512 instead of AVX2: bigger inputs benefit more.
const AVX512_SWITCH: usize = 1 << 10;

const PRIME32_1: u64 = 0x9E37_79B1;
const PRIME32_2: u64 = 0x85EB_CA77;
const PRIME32_3: u64 = 0xC2B2_AE3D;
const PRIME64_1: u64 = 0x9E37_79B1_85EB_CA87;
const PRIME64_2: u64 = 0xC2B2_AE3D_27D4_EB4F;
const PRIME64_3: u64 = 0x1656_67B1_9E37_79F9;
const PRIME64_4: u64 = 0x85EB_CA77_C2B2_AE63;
const PRIME64_5: u64 = 0x27D4_EB2F_1656_67C5;

const INIT_ACC: [u64; 8] = [
    PRIME32_3, PRIME64_1, PRIME64_2, PRIME64_3, PRIME64_4, PRIME32_2, PRIME64_5, PRIME32_1,
];

/// The default secret.
const KEY: [u8; 192] = [
    0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c, 0xf7, 0x21, 0xad, 0x1c,
    0xde, 0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb, 0x72, 0x40, 0xa4, 0xa4, 0xb7, 0xb3, 0x67, 0x1f,
    0xcb, 0x79, 0xe6, 0x4e, 0xcc, 0xc0, 0xe5, 0x78, 0x82, 0x5a, 0xd0, 0x7d, 0xcc, 0xff, 0x72, 0x21,
    0xb8, 0x08, 0x46, 0x74, 0xf7, 0x43, 0x24, 0x8e, 0xe0, 0x35, 0x90, 0xe6, 0x81, 0x3a, 0x26, 0x4c,
    0x3c, 0x28, 0x52, 0xbb, 0x91, 0xc3, 0x00, 0xcb, 0x88, 0xd0, 0x65, 0x8b, 0x1b, 0x53, 0x2e, 0xa3,
    0x71, 0x64, 0x48, 0x97, 0xa2, 0x0d, 0xf9, 0x4e, 0x38, 0x19, 0xef, 0x46, 0xa9, 0xde, 0xac, 0xd8,
    0xa8, 0xfa, 0x76, 0x3f, 0xe3, 0x9c, 0x34, 0x3f, 0xf9, 0xdc, 0xbb, 0xc7, 0xc7, 0x0b, 0x4f, 0x1d,
    0x8a, 0x51, 0xe0, 0x4b, 0xcd, 0xb4, 0x59, 0x31, 0xc8, 0x9f, 0x7e, 0xc9, 0xd9, 0x78, 0x73, 0x64,
    0xea, 0xc5, 0xac, 0x83, 0x34, 0xd3, 0xeb, 0xc3, 0xc5, 0x81, 0xa0, 0xff, 0xfa, 0x13, 0x63, 0xeb,
    0x17, 0x0d, 0xdd, 0x51, 0xb7, 0xf0, 0xda, 0x49, 0xd3, 0x16, 0x55, 0x26, 0x29, 0xd4, 0x68, 0x9e,
    0x2b, 0x16, 0xbe, 0x58, 0x7d, 0x47, 0xa1, 0xfc, 0x8f, 0xf8, 0xb8, 0xd1, 0x7a, 0xd0, 0x31, 0xce,
    0x45, 0xcb, 0x3a, 0x8f, 0x95, 0x16, 0x04, 0x28, 0xaf, 0xd7, 0xfb, 0xca, 0xbb, 0x4b, 0x40, 0x7e,
];

/// Returns the XXH3-64 hash of b.
pub fn hash(b: &[u8]) -> u64 {
    hash_tier(Tier::best(), b)
}

fn hash_tier(t: Tier, b: &[u8]) -> u64 {
    let l = b.len();
    let len = l as u64;
    match l {
        0 => xxh64_avalanche(k64(56) ^ k64(64)),
        1..=3 => {
            let c = u32::from(b[0]) << 16
                | u32::from(b[l >> 1]) << 24
                | u32::from(b[l - 1])
                | (l as u32) << 8;
            xxh64_avalanche(u64::from(c ^ (k32(0) ^ k32(4))))
        }
        4..=8 => {
            let x = u64::from(r32(b, l - 4)) | u64::from(r32(b, 0)) << 32;
            rrmxmx(x ^ (k64(8) ^ k64(16)), len)
        }
        9..=16 => {
            let lo = r64(b, 0) ^ (k64(24) ^ k64(32));
            let hi = r64(b, l - 8) ^ (k64(40) ^ k64(48));
            avalanche(
                len.wrapping_add(lo.swap_bytes())
                    .wrapping_add(hi)
                    .wrapping_add(mul_fold(lo, hi)),
            )
        }
        17..=128 => {
            let mut acc = len.wrapping_mul(PRIME64_1);
            let mut add = |o: usize, k: usize| acc = acc.wrapping_add(mix16(b, o, k));
            if l > 32 {
                if l > 64 {
                    if l > 96 {
                        add(48, 96);
                        add(l - 64, 112);
                    }
                    add(32, 64);
                    add(l - 48, 80);
                }
                add(16, 32);
                add(l - 32, 48);
            }
            add(0, 0);
            add(l - 16, 16);
            avalanche(acc)
        }
        129..=240 => {
            let mut acc = len.wrapping_mul(PRIME64_1);
            for i in 0..8 {
                acc = acc.wrapping_add(mix16(b, 16 * i, 16 * i));
            }
            acc = avalanche(acc);
            for i in 8..l / 16 {
                acc = acc.wrapping_add(mix16(b, 16 * i, 16 * (i - 8) + 3));
            }
            avalanche(acc.wrapping_add(mix16(b, l - 16, 119)))
        }
        _ => {
            let mut acc = INIT_ACC;
            let t = if l < AVX512_SWITCH { t.small() } else { t };
            t.accum(&mut acc, b);
            merge(&acc, len.wrapping_mul(PRIME64_1))
        }
    }
}

/// A streaming XXH3-64 hasher. It does not allocate: input is buffered in a fixed 1088-byte buffer.
pub struct Hasher {
    acc: [u64; 8],
    blk: u64,
    len: usize,
    tier: Tier,
    /// Pending input. A block is accumulated only once more than a stripe follows it, so sum64
    /// always has the last stripe of the input at hand.
    buf: [u8; BLOCK + STRIPE],
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher {
    /// Returns a hasher of no input.
    pub fn new() -> Self {
        Self::with_tier(Tier::best())
    }

    fn with_tier(tier: Tier) -> Self {
        Self {
            acc: INIT_ACC,
            blk: 0,
            len: 0,
            tier,
            buf: [0; BLOCK + STRIPE],
        }
    }

    /// Resets the hasher to no input.
    pub fn reset(&mut self) {
        self.acc = INIT_ACC;
        self.blk = 0;
        self.len = 0;
    }

    /// Adds b to the input.
    pub fn write(&mut self, mut b: &[u8]) {
        if b.is_empty() {
            return;
        }
        if self.len > 0 {
            let need = self.buf.len() - self.len;
            if b.len() <= need {
                self.buf[self.len..][..b.len()].copy_from_slice(b);
                self.len += b.len();
                return;
            }
            self.buf[self.len..].copy_from_slice(&b[..need]);
            b = &b[need..];
            self.tier.blocks(&mut self.acc, &self.buf, 1);
            self.blk += 1;
            self.buf.copy_within(BLOCK.., 0);
            self.len = STRIPE;
            if b.len() <= BLOCK {
                self.buf[STRIPE..][..b.len()].copy_from_slice(b);
                self.len += b.len();
                return;
            }
            // The retained stripe and the next input bytes straddle one block. Consuming only the
            // missing bytes leaves a complete trailing stripe in b, so later blocks need no copy.
            const BRIDGE: usize = BLOCK - STRIPE;
            self.buf[STRIPE..BLOCK].copy_from_slice(&b[..BRIDGE]);
            self.tier.blocks(&mut self.acc, &self.buf, 1);
            self.blk += 1;
            b = &b[BRIDGE..];
        }
        if b.len() > self.buf.len() {
            let n = (b.len() - self.buf.len()).div_ceil(BLOCK);
            self.tier.blocks(&mut self.acc, b, n);
            self.blk += n as u64;
            b = &b[n * BLOCK..];
        }
        self.buf[..b.len()].copy_from_slice(b);
        self.len = b.len();
    }

    /// Returns the hash of the input so far. More input may follow.
    pub fn sum64(&self) -> u64 {
        if self.blk == 0 {
            return hash_tier(self.tier, &self.buf[..self.len]);
        }
        debug_assert!(self.len > STRIPE);
        let mut acc = self.acc;
        self.tier.small().accum(&mut acc, &self.buf[..self.len]);
        let l = self.blk * BLOCK as u64 + self.len as u64;
        merge(&acc, l.wrapping_mul(PRIME64_1))
    }
}

/// A unit that runs the stripe loop. A variant other than Scalar is only made by [`tiers`], once the
/// CPU is known to have its features: the unsafe kernels rely on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tier {
    Scalar,
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    Sse2,
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    Avx2,
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    Avx512,
    #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
    Neon,
}

/// Returns the tiers this CPU runs, slowest first.
fn tiers() -> Vec<Tier> {
    #[allow(unused_mut)]
    let mut t = vec![Tier::Scalar];
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    {
        t.push(Tier::Sse2);
        if std::is_x86_feature_detected!("avx2") {
            t.push(Tier::Avx2);
            if std::is_x86_feature_detected!("avx512f") {
                t.push(Tier::Avx512);
            }
        }
    }
    #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
    t.push(Tier::Neon);
    t
}

impl Tier {
    fn best() -> Tier {
        static BEST: OnceLock<Tier> = OnceLock::new();
        *BEST.get_or_init(|| *tiers().last().unwrap())
    }

    /// Returns the tier for inputs below [`AVX512_SWITCH`].
    fn small(self) -> Tier {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        if self == Tier::Avx512 {
            return Tier::Avx2;
        }
        self
    }

    /// Accumulates p, which is at least a stripe long, as the end of the input.
    fn accum(self, a: &mut [u64; 8], p: &[u8]) {
        // SAFETY: tiers() made self once the CPU had its features; Scalar needs none.
        unsafe {
            match self {
                Tier::Scalar => accum::<Scalar>(a, p),
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                Tier::Sse2 => x86::accum_sse2(a, p),
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                Tier::Avx2 => x86::accum_avx2(a, p),
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                Tier::Avx512 => x86::accum_avx512(a, p),
                #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
                Tier::Neon => neon::accum_neon(a, p),
            }
        }
    }

    /// Accumulates the first n blocks of p, which input follows.
    fn blocks(self, a: &mut [u64; 8], p: &[u8], n: usize) {
        // SAFETY: tiers() made self once the CPU had its features; Scalar needs none.
        unsafe {
            match self {
                Tier::Scalar => blocks::<Scalar>(a, p, n),
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                Tier::Sse2 => x86::blocks_sse2(a, p, n),
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                Tier::Avx2 => x86::blocks_avx2(a, p, n),
                #[cfg(all(target_arch = "x86_64", not(miri)))]
                Tier::Avx512 => x86::blocks_avx512(a, p, n),
                #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
                Tier::Neon => neon::blocks_neon(a, p, n),
            }
        }
    }
}

/// The spec's accumulate_512 and scramble on one vector unit, holding the 8 lanes in `Acc`.
///
/// # Safety
///
/// The methods use the unit's CPU features: callers must have checked them.
trait Kernel {
    type Acc: Copy;
    unsafe fn load(a: &[u64; 8]) -> Self::Acc;
    unsafe fn store(acc: Self::Acc, a: &mut [u64; 8]);
    unsafe fn stripe(acc: &mut Self::Acc, p: &[u8; STRIPE], k: &[u8; STRIPE]);
    unsafe fn scramble(acc: &mut Self::Acc);
    #[inline(always)]
    unsafe fn prefetch(_p: *const u8) {}
}

/// Returns the secret stripe at o.
#[inline(always)]
fn key(o: usize) -> &'static [u8; STRIPE] {
    const K: &[u8; 192] = &KEY;
    K[o..o + STRIPE].try_into().unwrap()
}

/// Accumulates the stripes of n blocks of p, scrambling after each.
///
/// # Safety
///
/// The CPU has K's features.
#[inline(always)]
unsafe fn blocks_acc<K: Kernel>(acc: &mut K::Acc, p: &[u8], n: usize) {
    for b in p[..n * BLOCK].chunks_exact(BLOCK) {
        for (s, st) in b.chunks_exact(STRIPE).enumerate() {
            // SAFETY: the caller checked K's features; prefetching never faults.
            unsafe {
                K::prefetch(st.as_ptr().wrapping_add(BLOCK));
                K::stripe(acc, st.try_into().unwrap(), key(8 * s));
            }
        }
        // SAFETY: the caller checked K's features.
        unsafe { K::scramble(acc) };
    }
}

/// Accumulates the first n blocks of p.
///
/// # Safety
///
/// The CPU has K's features.
#[inline(always)]
unsafe fn blocks<K: Kernel>(a: &mut [u64; 8], p: &[u8], n: usize) {
    // SAFETY: the caller checked K's features.
    unsafe {
        let mut acc = K::load(a);
        blocks_acc::<K>(&mut acc, p, n);
        K::store(acc, a);
    }
}

/// Accumulates p, at least a stripe long, as the end of the input: the blocks that more input
/// follows, the remaining whole stripes but the last, then the last 64 bytes.
///
/// # Safety
///
/// The CPU has K's features.
#[inline(always)]
unsafe fn accum<K: Kernel>(a: &mut [u64; 8], p: &[u8]) {
    let n = (p.len() - 1) / BLOCK;
    let rest = &p[n * BLOCK..];
    // SAFETY: the caller checked K's features.
    unsafe {
        let mut acc = K::load(a);
        blocks_acc::<K>(&mut acc, p, n);
        for (s, st) in rest[..rest.len() - 1].chunks_exact(STRIPE).enumerate() {
            K::stripe(&mut acc, st.try_into().unwrap(), key(8 * s));
        }
        K::stripe(
            &mut acc,
            p[p.len() - STRIPE..].try_into().unwrap(),
            key(KEY.len() - STRIPE - 7),
        );
        K::store(acc, a);
    }
}

struct Scalar;

impl Kernel for Scalar {
    type Acc = [u64; 8];

    #[inline(always)]
    unsafe fn load(a: &[u64; 8]) -> [u64; 8] {
        *a
    }

    #[inline(always)]
    unsafe fn store(acc: [u64; 8], a: &mut [u64; 8]) {
        *a = acc;
    }

    #[inline(always)]
    unsafe fn stripe(acc: &mut [u64; 8], p: &[u8; STRIPE], k: &[u8; STRIPE]) {
        for i in 0..8 {
            let v = r64(p, 8 * i);
            let dk = v ^ r64(k, 8 * i);
            acc[i ^ 1] = acc[i ^ 1].wrapping_add(v);
            acc[i] = acc[i].wrapping_add((dk & 0xFFFF_FFFF) * (dk >> 32));
        }
    }

    #[inline(always)]
    unsafe fn scramble(acc: &mut [u64; 8]) {
        let k = key(KEY.len() - STRIPE);
        for (i, a) in acc.iter_mut().enumerate() {
            *a = (*a ^ *a >> 47 ^ r64(k, 8 * i)).wrapping_mul(PRIME32_1);
        }
    }
}

/// The spec's mergeAccs, with the secret at 11.
fn merge(a: &[u64; 8], start: u64) -> u64 {
    let mut r = start;
    for i in 0..4 {
        r = r.wrapping_add(mul_fold(
            a[2 * i] ^ k64(11 + 16 * i),
            a[2 * i + 1] ^ k64(19 + 16 * i),
        ));
    }
    avalanche(r)
}

#[inline(always)]
fn mix16(b: &[u8], o: usize, k: usize) -> u64 {
    mul_fold(r64(b, o) ^ k64(k), r64(b, o + 8) ^ k64(k + 8))
}

#[inline(always)]
fn mul_fold(a: u64, b: u64) -> u64 {
    let p = u128::from(a) * u128::from(b);
    p as u64 ^ (p >> 64) as u64
}

fn avalanche(mut h: u64) -> u64 {
    h ^= h >> 37;
    h = h.wrapping_mul(0x1656_6791_9E37_79F9);
    h ^ h >> 32
}

fn xxh64_avalanche(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(PRIME64_2);
    h ^= h >> 29;
    h = h.wrapping_mul(PRIME64_3);
    h ^ h >> 32
}

fn rrmxmx(mut h: u64, len: u64) -> u64 {
    h ^= h.rotate_left(49) ^ h.rotate_left(24);
    h = h.wrapping_mul(0x9FB2_1C65_1E98_DF25);
    h ^= (h >> 35).wrapping_add(len);
    h = h.wrapping_mul(0x9FB2_1C65_1E98_DF25);
    h ^ h >> 28
}

#[inline(always)]
fn r64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

#[inline(always)]
fn r32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

#[inline(always)]
fn k64(o: usize) -> u64 {
    r64(&KEY, o)
}

#[inline(always)]
fn k32(o: usize) -> u32 {
    r32(&KEY, o)
}
