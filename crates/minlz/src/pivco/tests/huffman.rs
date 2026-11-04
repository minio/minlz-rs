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

//! Port of Go's `huffman_test.go`.

use crate::pivco::build_lengths;
use crate::pivco::huffman::{Leaf, sort_leaves};
use crate::pivco::testutil::*;

fn fib_hist(r: &mut Xorshift64, k: usize, extra: usize) -> [u64; 256] {
    let mut h = [0u64; 256];
    let p = r.perm();
    let (mut a, mut b) = (1u64, 1u64);
    for &s in &p[..k] {
        h[usize::from(s)] = a;
        (a, b) = (b, a.wrapping_add(b));
    }
    for &s in &p[k..(k + extra).min(256)] {
        h[usize::from(s)] = 1 + r.intn(3) as u64;
    }
    h
}

/// Returns histograms whose totals stay below 2^40, so the Huffman depth stays below 64
/// and C (whose limiter has a [64] count array) is well defined on them.
pub(super) fn lens_test_hists() -> Vec<[u64; 256]> {
    let mut r = Xorshift64(0x9E3779B97F4A7C15);
    let mut out = Vec::new();
    for _ in 0..2000 {
        let mut h = [0u64; 256];
        let mut n = 1 + r.intn(256);
        if r.intn(4) == 0 {
            n = 1 + r.intn(8);
        }
        let shift = [3usize, 8, 16, 24, 31][r.intn(5)];
        let p = r.perm();
        for &s in &p[..n] {
            h[usize::from(s)] = match r.intn(3) {
                0 => 1 + r.next() % (1 << shift),
                1 => 1 + r.intn(4) as u64,
                _ => 1 << r.intn(shift + 1),
            };
        }
        out.push(h);
    }
    for k in 12..=56 {
        for extra in (0..200).step_by(25) {
            out.push(fib_hist(&mut r, k, extra));
        }
    }
    for k in 2..=38 {
        let mut h = [0u64; 256];
        for i in 0..k {
            h[i * 3 % 256] = 1 << i;
        }
        out.push(h);
    }
    for n in 1..=256 {
        let (mut h, mut g) = ([0u64; 256], [0u64; 256]);
        for i in 0..n {
            h[255 - i] = 7;
            g[i] = 1 + i as u64 % 3;
        }
        out.push(h);
        out.push(g);
    }
    out
}

/// Digest of build_lengths over lens_test_hists, verified against C pivco_build_table
/// (src/huffman_table.c at the commit in the module doc).
const LENS_TEST_DIGEST: u64 = 0x9291ff8d45ccc2c1;

#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn build_lengths_c() {
    let mut all = Vec::new();
    for h in lens_test_hists() {
        let mut lens = [0u8; 256];
        assert!(build_lengths(&h, &mut lens), "empty");
        check_lengths(&h, &lens);
        let mut short = [0u8; 256];
        assert!(
            build_lengths(&h[..used_syms(&h)], &mut short) && short == lens,
            "lengths up to the last used symbol: {short:?}, want {lens:?}"
        );
        all.extend_from_slice(&lens);
    }
    assert_eq!(fnv1a64(&all), LENS_TEST_DIGEST);
}

#[test]
fn build_lengths_edges() {
    let mut h = [0u64; 256];
    let mut lens = [0u8; 256];
    lens[3] = 9;
    assert!(
        !build_lengths(&h, &mut lens) && lens == [0; 256],
        "empty histogram accepted or lens not cleared"
    );
    h[200] = 1 << 63;
    assert!(
        build_lengths(&h, &mut lens) && lens[200] == 1 && kraft_sum(&lens) == 1 << 10,
        "single symbol"
    );
    h[7] = 3;
    build_lengths(&h, &mut lens);
    assert!(lens[7] == 1 && lens[200] == 1, "two symbols");
    h.fill(1000);
    build_lengths(&h, &mut lens);
    assert!(lens.iter().all(|&l| l == 8), "uniform 256");
    // Equal frequencies: the leaf wins ties, so the shorter codes go to the first symbols.
    h = [0; 256];
    (
        h[usize::from(b'a')],
        h[usize::from(b'b')],
        h[usize::from(b'c')],
    ) = (1, 1, 1);
    build_lengths(&h, &mut lens);
    assert_eq!(
        (
            lens[usize::from(b'a')],
            lens[usize::from(b'b')],
            lens[usize::from(b'c')]
        ),
        (2, 2, 1),
        "ties"
    );
}

/// Covers inputs where C is undefined (depth > 63) or the frequency sum wraps:
/// build_lengths must still return a complete code.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn build_lengths_extreme() {
    let mut r = Xorshift64(1);
    let mut hs = Vec::new();
    for k in 60..=93 {
        let a = fib_hist(&mut r, k, 0);
        let b = fib_hist(&mut r, k, 100);
        hs.push(a);
        hs.push(b);
    }
    for _ in 0..300 {
        let mut h = [0u64; 256];
        let p = r.perm();
        let n = 2 + r.intn(255);
        for &s in &p[..n] {
            let s = usize::from(s);
            h[s] = (1 << 63) + (r.next() >> 2);
            if r.intn(3) == 0 {
                h[s] = u64::MAX - r.intn(5) as u64;
            }
            if r.intn(5) == 0 {
                h[s] = 1 + r.intn(3) as u64;
            }
        }
        hs.push(h);
    }
    hs.push([u64::MAX; 256]);
    for h in &hs {
        let mut lens = [0u8; 256];
        build_lengths(h, &mut lens);
        check_lengths(h, &lens);
    }
}

#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn sort_leaves_stable() {
    let mut r = Xorshift64(7);
    for _ in 0..500 {
        let n = 1 + r.intn(256);
        let mut bits = 0;
        let mask = (1u64 << (1 + r.intn(63))) - 1;
        let mut l: Vec<Leaf> = (0..n)
            .map(|i| {
                let mut x = Leaf {
                    freq: 1 + (r.next() & mask),
                    sym: i as u8,
                };
                if r.intn(2) == 0 {
                    let a = r.intn(4) as u64;
                    x.freq = 1 + (a << r.intn(56));
                }
                bits |= x.freq;
                x
            })
            .collect();
        let mut want = l.clone();
        want.sort_by_key(|x| x.freq);
        sort_leaves(&mut l, bits);
        assert!(l == want, "n={n}: sort mismatch");
    }
}

/// Go FuzzLengths, over its seeds and seeded random inputs.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn fuzz_lengths_seeded() {
    fuzz_lengths(&[0, b'a', 1, 0, b'b', 1, 0, b'c', 2, 0]);
    fuzz_lengths(&[3, 0, 255, 255, 1, 255, 255, 2, 255, 255, 3, 255, 255]);
    let mut x = Xorshift64(0x5EED);
    for _ in 0..300 {
        let n = x.intn(800);
        let data: Vec<u8> = (0..n).map(|_| x.next() as u8).collect();
        fuzz_lengths(&data);
    }
}
