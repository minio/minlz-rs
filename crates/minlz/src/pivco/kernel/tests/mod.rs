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

//! Ports of Go's `internal/pivco/kernel` tests. This file holds the obviously-correct
//! references and buffer helpers of Go's `ref_test.go`.
//!
//! The vector conformance tests (`vec_*`) compare every tier of [`Tier::tiers`] but the
//! generic one against the generic tier, so new tiers are tested by being listed; so do the
//! page guard tests. Go's race detector tests are not ported.

#[cfg(all(target_arch = "x86_64", not(miri)))]
mod avx2;
mod encode;
mod exec;

use super::Tier;
pub(super) use crate::pivco::testutil::{Pcg, new_rng, rand_bits, rand_bytes};

/// 0..=300 and the sizes around the layout blocks.
pub(super) fn test_sizes() -> Vec<usize> {
    if cfg!(miri) {
        return MIRI_SIZES.to_vec();
    }
    let mut s: Vec<usize> = (0..=300).collect();
    s.extend([
        511, 512, 513, 639, 640, 641, 1023, 1024, 1025, 4095, 4096, 65535,
    ]);
    s
}

/// A few sizes that reach every layout and tail path, for Miri, which is slow.
pub(super) const MIRI_SIZES: [usize; 4] = [0, 9, 129, 640];

pub(super) const GUARD_BYTE: u8 = 0xAA;
pub(super) const GUARD_LEN: usize = 64;

/// Returns the usable tiers other than [`Tier::Generic`].
pub(super) fn vec_tiers() -> Vec<Tier> {
    Tier::tiers()
        .into_iter()
        .filter(|&t| t != Tier::Generic)
        .collect()
}

/// Returns page-guarded memory of at least n bytes.
pub(super) fn guard_buf(n: usize) -> crate::guard::Buffer {
    crate::guard::Buffer::new(n).expect("guard: mapping failed")
}

pub(super) fn filled(n: usize, v: u8) -> Vec<u8> {
    vec![v; n]
}

pub(super) fn rand_vec(rng: &mut Pcg, n: usize) -> Vec<u8> {
    let mut b = vec![0; n];
    rand_bytes(rng, &mut b);
    b
}

/// Returns a copy of b followed by GUARD_LEN guard bytes.
pub(super) fn guarded(b: &[u8]) -> Vec<u8> {
    let mut all = b.to_vec();
    all.resize(b.len() + GUARD_LEN, GUARD_BYTE);
    all
}

pub(super) fn intact(b: &[u8]) -> bool {
    b.iter().all(|&c| c == GUARD_BYTE)
}

pub(super) fn bit(bm: &[u8], j: usize) -> bool {
    bm[j / 8] >> (j % 8) & 1 != 0
}

pub(super) fn ref_part(ranks: &[u8], thr: u8) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut bm = vec![0u8; ranks.len().div_ceil(8)];
    let (mut left, mut right) = (Vec::new(), Vec::new());
    for (j, &v) in ranks.iter().enumerate() {
        if v > thr {
            bm[j / 8] |= 1 << (j % 8);
            right.push(v);
        } else {
            left.push(v);
        }
    }
    (bm, left, right)
}

/// Returns out[j] = bit j ? r[nr++] : l[nl++] for a k-bit bitmap, and nr.
pub(super) fn ref_merge(bm: &[u8], k: usize, l: &[u8], r: &[u8]) -> (Vec<u8>, usize) {
    let (mut nl, mut nr) = (0, 0);
    let out = (0..k)
        .map(|j| {
            if bit(bm, j) {
                nr += 1;
                r[nr - 1]
            } else {
                nl += 1;
                l[nl - 1]
            }
        })
        .collect();
    (out, nr)
}

/// Returns the byte and bit that hold bit t of code i in an n-code flat region of the
/// vertical layout, straight from the layout description in upstream
/// `src/pivco_huffman_vertical.h`.
pub(super) fn ref_bit_pos(n: usize, d: usize, i: usize, t: usize) -> (usize, usize) {
    let (mut n512, mut n128) = (0, 0);
    if (2..=7).contains(&d) {
        if n >= 512 {
            n512 = n / 512 * 512;
        }
        if n - n512 >= 128 {
            n128 = (n - n512) / 128 * 128;
        }
    }
    let (base, i, ln) = if i < n512 {
        (0, i, 64)
    } else if i < n512 + n128 {
        (n512 * d / 8, i - n512, 16)
    } else {
        let pos = i * d + t; // the vertical spans end on a byte boundary
        return (pos / 8, pos % 8);
    };
    let (blk, v) = (i / (8 * ln), i % (8 * ln));
    let (lane, pos) = (v % ln, v / ln * d + t);
    (base + blk * ln * d + ln * (pos / 8) + lane, pos % 8)
}

pub(super) fn ref_pack(codes: &[u8], d: usize) -> Vec<u8> {
    let mut out = vec![0u8; (codes.len() * d).div_ceil(8)];
    for (i, &c) in codes.iter().enumerate() {
        for t in 0..d {
            if c >> t & 1 != 0 {
                let (by, bi) = ref_bit_pos(codes.len(), d, i, t);
                out[by] |= 1 << bi;
            }
        }
    }
    out
}

pub(super) fn ref_unpack(region: &[u8], n: usize, d: usize) -> Vec<u8> {
    (0..n)
        .map(|i| {
            (0..d).fold(0u8, |c, t| {
                let (by, bi) = ref_bit_pos(n, d, i, t);
                c | (region[by] >> bi & 1) << t
            })
        })
        .collect()
}

/// Transliterates the upstream scalar packers (`prim_enc_pack_dN` in
/// `src/pivco_huffman_primitives_scalar.h`) as an independent check of ref_bit_pos.
pub(super) fn c_pack(ranks: &[u8], d: usize, base: u8) -> Vec<u8> {
    let n = ranks.len();
    let mut out = vec![0u8; (n * d).div_ceil(8)];
    let vert_n = |n: usize| {
        if (2..=7).contains(&d) && n >= 128 {
            n & !127
        } else {
            0
        }
    };
    let vpack = |out: &mut [u8], ranks: &[u8], nv: usize, ln: usize| {
        let mask = (1u32 << d) - 1;
        let bv = ln * 8;
        for b in 0..nv / bv {
            let blk = &mut out[b * ln * d..];
            for v in 0..bv {
                let val = u32::from(ranks[b * bv + v].wrapping_sub(base)) & mask;
                let (l, bit) = (v % ln, v / ln * d);
                let (j, off) = (bit >> 3, bit & 7);
                blk[ln * j + l] |= (val << off) as u8;
                if off + d > 8 {
                    blk[ln * (j + 1) + l] |= (val >> (8 - off)) as u8;
                }
            }
        }
    };
    let nat_pack = |out: &mut [u8], ranks: &[u8]| {
        let (mut buf, mut bits, mut idx) = (0u64, 0, 0);
        for &r in ranks {
            buf |= u64::from(r.wrapping_sub(base)) << bits;
            bits += d;
            while bits >= 8 {
                out[idx] = buf as u8;
                idx += 1;
                buf >>= 8;
                bits -= 8;
            }
        }
        if bits > 0 {
            out[idx] = (buf & ((1 << bits) - 1)) as u8;
        }
    };
    let n5 = if (2..=7).contains(&d) && n >= 512 {
        n & !511
    } else {
        0
    };
    vpack(&mut out, ranks, n5, 64);
    let r = n - n5;
    let nv = vert_n(r);
    let o1 = &mut out[(n5 * d) >> 3..];
    vpack(o1, &ranks[n5..], nv, 16);
    if r > nv {
        nat_pack(&mut o1[(nv * d) >> 3..], &ranks[n5 + nv..]);
    }
    out
}

/// Returns the number of set bits in the first k bits of b.
pub(super) fn dec_popcnt(b: &[u8], k: usize) -> u32 {
    (0..k).map(|j| u32::from(b[j / 8] >> (j % 8) & 1)).sum()
}

/// The PCG port draws Go's `rand.New(rand.NewPCG(1, 2))` sequence, so inputs match Go's.
#[test]
fn pcg_matches_go() {
    let mut r = new_rng();
    assert_eq!(
        (
            r.uint64(),
            r.uint32(),
            r.int_n(251),
            r.int_n(1000003),
            r.int_n(256)
        ),
        (14192431797130687760, 2647573421, 196, 796598, 80)
    );
    assert_eq!(
        (r.float64(), r.perm(6), r.int_n(3)),
        (0.3883664855410056, vec![4, 3, 0, 1, 5, 2], 1)
    );
}
