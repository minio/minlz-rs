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

//! Ports of Go's `encode_test.go` and the tier-agnostic part of `encode_vec_test.go`.

use super::*;
use crate::pivco::kernel::generic::gather_generic;
use crate::pivco::kernel::generic_pack::pack_generic;
use crate::pivco::kernel::{HIST_SCRATCH, RANK_SLACK, vert_split};

#[test]
fn gather() {
    for tier in Tier::tiers() {
        let mut rng = new_rng();
        for n in test_sizes() {
            let mut s2r = [0u8; 256];
            rand_bytes(&mut rng, &mut s2r);
            if n % 3 == 0 {
                s2r[0] = 255; // sentinel-like rank must be reported as the max
            }
            let mut src = rand_vec(&mut rng, n);
            if n % 2 == 0 {
                src.iter_mut().for_each(|s| *s &= 7);
            }
            let src = &guarded(&src)[..n];
            let want: Vec<u8> = src.iter().map(|&s| s2r[usize::from(s)]).collect();
            let wmx = want.iter().copied().max().unwrap_or(0);
            let mut buf = filled(n + RANK_SLACK + GUARD_LEN, GUARD_BYTE);
            let mx = tier.gather(&mut buf[..n + RANK_SLACK], src, &s2r);
            assert!(
                mx == wmx && buf[..n] == want && intact(&buf[n + RANK_SLACK..]),
                "{tier} n={n}: max {mx} want {wmx}"
            );
        }
    }
}

#[test]
fn partition() {
    for tier in Tier::tiers() {
        let mut rng = new_rng();
        for n in test_sizes() {
            let r = rng.int_n(251);
            for thr in [0u8, 1, 127, (2 + r) as u8, 253, 254, 255] {
                let mut ranks = rand_vec(&mut rng, n);
                if n % 2 == 1 {
                    // Ranks around thr, so both sides and long runs occur.
                    for v in &mut ranks {
                        *v = (i32::from(thr) + i32::from(*v % 7) - 3).clamp(0, 255) as u8;
                    }
                }
                test_partition(tier, &ranks, thr);
            }
        }
    }
}

/// Every tier returns 0 for n = 0 without touching its buffers.
#[test]
fn partition_empty() {
    for tier in Tier::tiers() {
        assert!(
            tier.part_full(&mut [], 0, &mut [], &mut [], 0) == 0
                && tier.part_right(&[], 0, &mut [], &mut [], 0) == 0
                && tier.part_none(&[], 0, &mut [], 0) == 0,
            "{tier}: nonzero count"
        );
    }
}

/// Runs the partition checks: bitmap with zero padding, both sides in order, guard bytes
/// right after ranks[..n] and after the slack of right and bm, and, adjacent, right ==
/// ranks[n..] as for a right child in the encoder's walk; ranks are then exact.
pub(super) fn test_partition(tier: Tier, ranks: &[u8], thr: u8) {
    let (n, nb) = (ranks.len(), ranks.len().div_ceil(8));
    let (wbm, wl, wr) = ref_part(ranks, thr);
    let new_buf = |n: usize| filled(n + RANK_SLACK + GUARD_LEN, GUARD_BYTE);
    let check_bm = |name: &str, bm: &[u8]| {
        assert!(
            bm[..nb] == wbm && intact(&bm[nb + RANK_SLACK..]),
            "{tier} {name} n={n} thr={thr}: bitmap"
        );
    };

    for adjacent in [false, true] {
        let mut rk = new_buf(if adjacent { 2 * n } else { n });
        let mut sep = new_buf(n);
        let mut bm = new_buf(nb);
        rk[..n].copy_from_slice(ranks);
        let nr = {
            let (rkn, right) = if adjacent {
                rk.split_at_mut(n)
            } else {
                (&mut rk[..n + RANK_SLACK], &mut sep[..])
            };
            tier.part_full(
                rkn,
                n,
                &mut right[..n + RANK_SLACK],
                &mut bm[..nb + RANK_SLACK],
                thr,
            )
        };
        check_bm("part_full", &bm);
        let right = if adjacent { &rk[n..] } else { &sep[..] };
        let ok = nr == wr.len()
            && rk[..n - nr] == wl
            && right[..nr] == wr
            && intact(&right[n + RANK_SLACK..]);
        assert!(
            ok,
            "{tier} part_full adjacent={adjacent} n={n} thr={thr}: nr {nr} want {}",
            wr.len()
        );
        if !adjacent {
            assert!(
                intact(&rk[n..]),
                "{tier} part_full n={n} thr={thr}: wrote past ranks"
            );
        }

        rk[..n].copy_from_slice(ranks);
        let mut bm = new_buf(nb);
        let mut sep = new_buf(n);
        if adjacent {
            rk[n..].fill(GUARD_BYTE);
        }
        let nr = {
            let (rkn, right): (&[u8], &mut [u8]) = if adjacent {
                let (h, t) = rk.split_at_mut(n);
                (h, t)
            } else {
                (&rk[..n + RANK_SLACK], &mut sep[..])
            };
            tier.part_right(
                rkn,
                n,
                &mut right[..n + RANK_SLACK],
                &mut bm[..nb + RANK_SLACK],
                thr,
            )
        };
        check_bm("part_right", &bm);
        let right = if adjacent { &rk[n..] } else { &sep[..] };
        assert!(
            nr == wr.len()
                && rk[..n] == *ranks
                && right[..nr] == wr
                && intact(&right[n + RANK_SLACK..]),
            "{tier} part_right adjacent={adjacent} n={n} thr={thr}: nr {nr} want {}",
            wr.len()
        );
    }

    let mut rk = new_buf(n);
    rk[..n].copy_from_slice(ranks);
    let mut bm = new_buf(nb);
    let nr = tier.part_none(&rk[..n + RANK_SLACK], n, &mut bm[..nb + RANK_SLACK], thr);
    check_bm("part_none", &bm);
    assert!(
        nr == wr.len() && rk[..n] == *ranks && intact(&rk[n..]),
        "{tier} part_none n={n} thr={thr}: nr {nr} want {}",
        wr.len()
    );
}

#[test]
fn vert_split_cases() {
    for (n, d, n512, n128) in [
        (1000, 3, 512, 384),
        (1000, 8, 0, 0),
        (1000, 1, 0, 0),
        (511, 7, 0, 384),
        (639, 2, 512, 0),
        (640, 2, 512, 128),
        (127, 4, 0, 0),
        (65535, 5, 65024, 384),
    ] {
        assert_eq!(vert_split(n, d), (n512, n128), "vert_split({n}, {d})");
    }
}

/// Pins a few addresses of the wire layout by hand.
#[test]
fn layout_example() {
    // "abcd" with four 2-bit codes 0..3 packs naturally to 0xE4.
    assert_eq!(ref_pack(&[0, 1, 2, 3], 2), [0xe4], "natural abcd");
    // One 128-value block, D=3: code v lives in lane v%16 at bits 3*(v/16)...
    let mut codes = vec![0u8; 128];
    (codes[0], codes[1], codes[16], codes[48]) = (1, 2, 7, 5);
    let mut want = vec![0u8; 48];
    want[0] = 1 | 7 << 3; // lane 0: v=0 at bit 0, v=16 at bits 3..5
    want[1] = 2; // lane 1: v=1 at bit 0
    want[16] = 5 << 1; // lane 0: v=48 at bits 9..11, column 1
    assert_eq!(ref_pack(&codes, 3), want, "vertical128 block");
    assert_eq!(c_pack(&codes, 3, 0), want, "vertical128 block (C)");
}

#[test]
fn pack() {
    for tier in Tier::tiers() {
        let mut rng = new_rng();
        for n in test_sizes() {
            for d in 2..=8usize {
                let base = rng.int_n(257 - (1 << d)) as u8;
                let mut codes = rand_vec(&mut rng, n);
                codes.iter_mut().for_each(|c| *c &= ((1u16 << d) - 1) as u8);
                let mut ranks = filled(n + RANK_SLACK + GUARD_LEN, GUARD_BYTE);
                for (r, &c) in ranks.iter_mut().zip(&codes) {
                    *r = base + c;
                }
                let want = ref_pack(&codes, d);
                if n <= 1100 || n == 65535 {
                    assert_eq!(
                        c_pack(&ranks[..n], d, base),
                        want,
                        "reference mismatch n={n} d={d}"
                    );
                }
                let size = want.len();
                // Junk in dst: every byte must be written.
                let mut dst = rand_vec(&mut rng, size + GUARD_LEN);
                dst[size..].fill(GUARD_BYTE);
                tier.pack(&mut dst[..size], &ranks[..n + RANK_SLACK], n, d as u8, base);
                assert!(
                    dst[..size] == want && intact(&dst[size..]) && intact(&ranks[n..]),
                    "{tier} n={n} d={d}: region or guard"
                );
            }
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn histogram() {
    let mut scratch = vec![0u8; HIST_SCRATCH];
    for tier in Tier::tiers() {
        let mut rng = new_rng();
        let mut sizes = test_sizes();
        sizes.push((1 << 20) + 13);
        for n in sizes {
            for kind in 0..3 {
                let mut src = rand_vec(&mut rng, n);
                match kind {
                    1 => src.fill(7),
                    2 => src.iter_mut().for_each(|s| *s = 200 + (*s & 1)),
                    _ => {}
                }
                let src = &guarded(&src)[..n];
                let mut h: [u32; 256] = core::array::from_fn(|_| rng.uint32());
                let mut want = h;
                for &c in src {
                    want[usize::from(c)] = want[usize::from(c)].wrapping_add(1);
                }
                tier.histogram(&mut h, src, &mut scratch);
                assert!(h == want, "{tier} n={n} kind={kind}: histogram mismatch");
            }
        }
    }
}

fn enc_sizes() -> Vec<usize> {
    if cfg!(miri) {
        return MIRI_SIZES.to_vec();
    }
    let mut s: Vec<usize> = (0..=300).collect();
    s.extend([511, 512, 513, 1023, 1024, 4095, 4096, 65535]);
    s
}

/// The vector gather against the generic one, with the tails at every alignment: gather
/// writes exactly ranks[..n].
#[test]
fn vec_gather() {
    for tier in vec_tiers() {
        let mut rng = new_rng();
        for n in enc_sizes() {
            for alpha in [1usize, 7, 128, 256] {
                // Symbols >= alpha are absent: rank 0xFF, as in Table.s2r. A full table has
                // the real rank 255.
                let mut s2r = [0u8; 256];
                for (s, r) in rng.perm(256).into_iter().enumerate() {
                    s2r[s] = if alpha < 256 {
                        (r % 255) as u8
                    } else {
                        r as u8
                    };
                }
                s2r[alpha.min(256)..].fill(0xFF);
                let mut src = vec![0u8; n];
                for s in &mut src {
                    *s = rng.int_n(alpha) as u8;
                }
                if n > 0 && rng.int_n(4) == 0 {
                    let i = rng.int_n(n);
                    src[i] = 255;
                }
                let src = &guarded(&src)[..n];
                let mut want = vec![0u8; n];
                let wmx = gather_generic(&mut want, src, &s2r);
                let mut buf = filled(n + GUARD_LEN, GUARD_BYTE);
                let mx = tier.gather(&mut buf[..n], src, &s2r);
                assert!(
                    mx == wmx && buf[..n] == want && intact(&buf[n..]),
                    "{tier} n={n} alpha={alpha}: max {mx} want {wmx}"
                );
            }
        }
    }
}

/// Runs test_partition on all-left, all-right and mixed inputs.
#[test]
fn vec_partition() {
    let tiers = vec_tiers();
    if tiers.is_empty() {
        return;
    }
    let mut rng = new_rng();
    for n in enc_sizes() {
        let r = rng.int_n(251);
        for thr in [0u8, 1, (2 + r) as u8, 253, 254] {
            for pat in 0..4 {
                let mut ranks = rand_vec(&mut rng, n);
                for (i, v) in ranks.iter_mut().enumerate() {
                    match pat {
                        0 => *v = rng.int_n(usize::from(thr) + 1) as u8, // all left
                        1 => *v = (usize::from(thr) + 1 + rng.int_n(255 - usize::from(thr))) as u8, // all right
                        2 => *v = if (i / 37) & 1 == 0 { thr } else { thr + 1 }, // runs around thr
                        _ => {}
                    }
                }
                for &tier in &tiers {
                    test_partition(tier, &ranks, thr);
                }
            }
        }
    }
}

/// The vector pack against the generic one; pack writes exactly the region.
#[test]
fn vec_pack() {
    let tiers = vec_tiers();
    if tiers.is_empty() {
        return;
    }
    let mut rng = new_rng();
    let mut sizes = enc_sizes();
    sizes.extend([639, 640, 641, 1025, 1535, 1536, 1537]);
    for n in sizes {
        for d in 2..=8u8 {
            let base = rng.int_n(257 - (1 << d)) as u8;
            let mut ranks = filled(n + GUARD_LEN, GUARD_BYTE);
            for r in &mut ranks[..n] {
                *r = base + (rng.uint32() as u8 & ((1u16 << d) - 1) as u8);
            }
            let size = (n * usize::from(d)).div_ceil(8);
            let mut want = vec![0u8; size];
            pack_generic(&mut want, &ranks, n, d, base);
            for &tier in &tiers {
                // Junk in dst: every byte must be written. Guard bytes right after it.
                let mut dst = rand_vec(&mut rng, size + GUARD_LEN);
                dst[size..].fill(GUARD_BYTE);
                tier.pack(&mut dst[..size], &ranks[..n], n, d, base);
                assert!(
                    dst[..size] == want && intact(&dst[size..]) && intact(&ranks[n..]),
                    "{tier} n={n} d={d} base={base}: region or guard"
                );
            }
        }
    }
}

/// Places every exactly-accessed buffer flush against an inaccessible page, after and
/// before it, so that any read or write outside faults; for every tier.
#[test]
#[cfg_attr(miri, ignore = "no page guards under Miri")]
fn vec_page_guard() {
    let mut rng = new_rng();
    const MAX_N: usize = 4096 + 300;
    let (mut mem_a, mut mem_b) = (guard_buf(MAX_N), guard_buf(MAX_N));
    let mut s2r = [0u8; 256];
    rand_bytes(&mut rng, &mut s2r);
    let mut scratch = vec![0u8; HIST_SCRATCH];
    let mut right = vec![0u8; MAX_N + RANK_SLACK];
    let mut bm = vec![0u8; MAX_N / 8 + 1 + RANK_SLACK];
    // 0..=80 reaches every branch of the exact 0..16-byte tail loads and stores.
    let mut sizes = vec![127, 128, 200, 511, 512, 513, 640, 1000, 4095, 4096 + 300];
    sizes.extend(0..=80);
    for tier in Tier::tiers() {
        for &n in &sizes {
            for at_end in [true, false] {
                // Gather: src and ranks exact.
                let src = mem_a.slice(n, at_end);
                rand_bytes(&mut rng, src);
                let ranks = mem_b.slice(n, at_end);
                tier.gather(ranks, src, &s2r);
                // Partitions: ranks exact.
                for thr in [0u8, 100, 254] {
                    tier.part_none(ranks, n, &mut bm, thr);
                    tier.part_right(ranks, n, &mut right, &mut bm, thr);
                    tier.part_full(ranks, n, &mut right, &mut bm, thr);
                }
                // Pack: ranks and region exact.
                for d in 2..=8u8 {
                    ranks.iter_mut().for_each(|r| *r &= ((1u16 << d) - 1) as u8);
                    let dst = mem_a.slice((n * usize::from(d)).div_ceil(8), at_end);
                    tier.pack(dst, ranks, n, d, 0);
                }
                // Histogram: src exact, with all bytes and with bytes below 0x80.
                let mut h = [0u32; 256];
                let src = mem_a.slice(n, at_end);
                rand_bytes(&mut rng, src);
                tier.histogram(&mut h, src, &mut scratch);
                src.iter_mut().for_each(|s| *s &= 0x7F);
                tier.histogram(&mut h, src, &mut scratch);
            }
        }
    }
}
