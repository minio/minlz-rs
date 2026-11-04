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

//! Port of Go's `block_test.go`.

use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::pivco::encode::HIST_DIRECT;
use crate::pivco::kernel::{KIND_BOTH, KIND_FLAT, KIND_FULL, KIND_LEAF_LEFT, Rec};
use crate::pivco::testutil::*;
use crate::pivco::{Error, MAX_BLOCK_SIZE, MAX_CODE_LEN, Table};

#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn block_round_trip() {
    let mut ds: Vec<[u64; 256]> = TEST_DISTS.iter().map(|&n| dist_of(n)).collect();
    ds.extend((0..6u64).map(|seed| rhist(0x1234 + seed)));
    let mut cs = coders();
    for h in &ds {
        let tb = must_table(&lens_of(h));
        for n in BLOCK_SIZES {
            let src = sample(h, n, (n as u64).wrapping_mul(0x9E3779B97F4A7C15) + 1);
            for c in &mut cs {
                round_trip(c, &src, &tb);
            }
        }
    }
}

/// The lone record's bitmap is all zeros.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn block_single_symbol() {
    let mut cs = coders();
    for s in [0u8, 42, 255] {
        for l in [1u8, 5, 11] {
            let mut lens = [0u8; 256];
            lens[usize::from(s)] = l;
            let tb = must_table(&lens);
            for n in [1, 7, 8, 9, 255, 256, 32768, 65535] {
                let src = vec![s; n];
                for c in &mut cs {
                    let blk = round_trip(c, &src, &tb);
                    assert!(
                        blk == vec![0; n.div_ceil(8)],
                        "{} sym {s} len {l} n {n}",
                        c.tier
                    );
                }
            }
        }
    }
}

fn table_of(lens: &[(u8, u8)]) -> Table {
    let mut l = [0u8; 256];
    for &(v, n) in lens {
        l[usize::from(v)] = n;
    }
    must_table(&l)
}

/// Encodes the examples of SPEC_PIVCO.md appendix A.
#[test]
fn block_spec_examples() {
    let a1 = table_of(&[
        (b'a', 3),
        (b'b', 3),
        (b'c', 3),
        (b'd', 3),
        (b'e', 2),
        (b'f', 3),
        (b'g', 3),
    ]);
    assert_eq!(&a1.r2s[..usize::from(a1.nranks)], b"abcdefg", "A.1 ranks");
    let want = [
        Rec {
            op: KIND_FULL,
            param: 3,
            right: 2,
        },
        Rec {
            op: KIND_FLAT | 2 << 2,
            param: 0,
            right: 0,
        },
        Rec {
            op: KIND_LEAF_LEFT,
            param: 4,
            right: 1,
        },
        Rec {
            op: KIND_BOTH,
            param: 5,
            right: 0,
        },
    ];
    assert_eq!(a1.recs(), want, "A.1 records");
    let a5 = table_of(&[
        (b'a', 3),
        (b'b', 3),
        (b'c', 3),
        (b'd', 3),
        (b'e', 3),
        (b'f', 3),
        (b'g', 3),
        (b'h', 3),
    ]);
    let mut a5src: Vec<u8> = (b'a'..=b'h').flat_map(|v| [v; 16]).collect();
    a5src.extend_from_slice(b"aa");
    let mut a5blk = [[0x88u8; 16], [0xc6; 16], [0xfa; 16]].concat();
    a5blk.push(0);
    let a4 = table_of(&[(b'x', 1)]);
    let cases: [(&str, &Table, &[u8], &[u8]); 5] = [
        (
            "A.2",
            &a1,
            b"fedcbafeedbeef",
            &[0x08, 0x00, 0x03, 0x00, 0x00, 0x85, 0x1b, 0x07, 0xc3, 0x39],
        ),
        (
            "A.3 eee",
            &a1,
            b"eee",
            &[0x03, 0x00, 0x00, 0x00, 0x00, 0x07],
        ),
        ("A.3 abba", &a1, b"abba", &[0x00, 0x00, 0x14, 0x00]),
        ("A.4", &a4, b"xxxxxxxxxxx", &[0, 0]),
        ("A.5", &a5, &a5src, &a5blk),
    ];
    for (name, tb, src, blk) in cases {
        for c in &mut coders() {
            assert_eq!(round_trip(c, src, tb), blk, "{name} {}", c.tier);
        }
    }
}

/// Codes every symbol at the bottom of a chain of LEAF_LEFT nodes, the deepest stack of
/// right ranks the encoder's scratch must hold.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn block_deep_chain() {
    let mut lens = [0u8; 256];
    for (i, l) in lens[..MAX_CODE_LEN].iter_mut().enumerate() {
        *l = i as u8 + 1;
    }
    lens[MAX_CODE_LEN] = MAX_CODE_LEN as u8;
    let tb = must_table(&lens);
    let src: Vec<u8> = (0..MAX_BLOCK_SIZE)
        .map(|i| (MAX_CODE_LEN - 1 + i % 2) as u8)
        .collect();
    for c in &mut coders() {
        round_trip(c, &src, &tb);
    }
}

/// 256 symbols of length 8 are one depth-8 flat, stored verbatim.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn block_flat_root() {
    let tb = must_table(&[8; 256]);
    let mut cs = coders();
    for n in BLOCK_SIZES {
        let src = sample(&dist_of("sym0_255"), n, n as u64);
        for c in &mut cs {
            assert!(
                round_trip(c, &src, &tb) == src,
                "{} n {n}: not stored verbatim",
                c.tier
            );
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn block_flat_depths() {
    let mut cs = coders();
    for d in 2..=8usize {
        for place in ["root", "sub", "deep"] {
            if d == 8 && place != "root" {
                continue;
            }
            let tb = must_table(&flat_lens(d, place));
            for k in [0, 1, 127, 128, 129, 511, 512, 513, 639, 640, 641, 1151] {
                // Exactly k symbols of the flat, shuffled with 300 others.
                let mut x = Xorshift64((k * 16 + d) as u64 | 1);
                let mut src: Vec<u8> = (0..k)
                    .map(|_| (16 * b2i(d < 8) + (x.next() % (1 << d)) as usize) as u8)
                    .collect();
                let top = match place {
                    "sub" => 1,
                    "deep" => 3,
                    _ => 0,
                };
                if top > 0 {
                    src.extend((0..300).map(|_| (x.next() % top) as u8));
                    for i in (1..src.len()).rev() {
                        let j = (x.next() % (i as u64 + 1)) as usize;
                        src.swap(i, j);
                    }
                }
                if src.is_empty() {
                    continue;
                }
                for c in &mut cs {
                    let blk = round_trip(c, &src, &tb);
                    let mut nf = 0;
                    for f in walk_block(&tb, &blk, src.len()) {
                        if f.kind == b'f' {
                            assert_eq!(
                                f.size,
                                (k * d).div_ceil(8),
                                "D{d} {place} k={k}: flat region size"
                            );
                            nf += 1;
                        }
                    }
                    assert_eq!(nf, b2i(k > 0), "D{d} {place} k={k}: flat regions");
                }
            }
        }
    }
}

#[test]
fn block_errors() {
    let tb = table_of(&[(b'a', 1), (b'b', 2), (b'c', 2)]);
    let empty = Table::default();
    let big = vec![0u8; MAX_BLOCK_SIZE + 1];
    for c in &mut coders() {
        let cases: [(&[u8], &Table, Error); 5] = [
            (&[], &tb, Error::BlockSize),
            (&big, &tb, Error::BlockSize),
            (b"abc", &empty, Error::InvalidTable),
            (b"abcd", &tb, Error::SymbolNotInTable),
            (b"dabc", &tb, Error::SymbolNotInTable),
        ];
        for (src, t, want) in cases {
            let mut all = guarded(100, 1);
            let got = c.enc.encode(&mut all[1..101], src, t);
            assert!(
                got == Err(want) && intact(&all),
                "{} {src:?}: err {got:?}, want {want}",
                c.tier
            );
        }
        for (n, t, want) in [
            (3, &empty, Error::InvalidTable),
            (0, &tb, Error::BlockSize),
            (MAX_BLOCK_SIZE + 1, &tb, Error::BlockSize),
        ] {
            let got = c.dec.decode(&mut vec![0; n], &[1, 0, 0, 0], t);
            assert_eq!(got, Err(want), "decode {n} symbols");
        }
        // A dst of the wrong size is a programming error.
        for size in [1, 3] {
            let r = catch_unwind(AssertUnwindSafe(|| {
                c.enc.encode(&mut vec![0; size], b"abc", &tb)
            }));
            assert!(r.is_err(), "dst of {size} bytes for a block of 4 accepted");
        }
    }
    let mut h = [0u64; 256];
    h[usize::from(b'd')] = 1;
    assert!(
        tb.block_size(&h).is_none() && empty.block_size(&h).is_none(),
        "block_size of a symbol without a code or an invalid table"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn histogram() {
    let src = sample(&dist_of("english"), 100000, 1);
    for c in &mut coders() {
        // Inputs below HIST_DIRECT are counted without the kernels.
        for n in [HIST_DIRECT - 1, HIST_DIRECT, src.len()] {
            let mut want = [0u64; 256];
            for &b in &src[..n] {
                want[usize::from(b)] += 2;
            }
            for off in [0, 1, 63] {
                let mut h: [u64; 256] = core::array::from_fn(|i| b2i(i < off) as u64);
                c.enc.histogram(&mut h, &src[..n]);
                c.enc.histogram(&mut h, &src[..n]);
                for i in 0..256 {
                    assert_eq!(
                        h[i],
                        want[i] + b2i(i < off) as u64,
                        "{}, {n} bytes: symbol {i}",
                        c.tier
                    );
                }
            }
        }
        assert_eq!(
            !c.enc.hscratch.is_empty(),
            c.tier.hist_uses_scratch(),
            "{}: histogram scratch allocated",
            c.tier
        );
    }
}

/// Shares one Table between threads, each with its own coders.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn concurrent_table() {
    let h = dist_of("english");
    let tb = must_table(&lens_of(&h));
    let srcs = [
        sample(&h, 300, 1),
        sample(&h, 5000, 2),
        sample(&h, 40000, 3),
    ];
    let want: Vec<Vec<u8>> = srcs
        .iter()
        .map(|src| round_trip(&mut coders().swap_remove(0), src, &tb))
        .collect();
    std::thread::scope(|s| {
        for g in 0..8 {
            let (tb, srcs, want) = (&tb, &srcs, &want);
            s.spawn(move || {
                // Tier selection tests may disable tiers in parallel.
                let mut cs = coders();
                let mut c = cs.swap_remove(g % cs.len());
                for _ in 0..4 {
                    for (src, want) in srcs.iter().zip(want) {
                        let mut blk = vec![0; want.len()];
                        assert!(
                            c.enc.encode(&mut blk, src, tb).is_ok() && blk == *want,
                            "{}: encode differs",
                            c.tier
                        );
                        let mut got = vec![0; src.len()];
                        assert!(
                            c.dec.decode(&mut got, &blk, tb).is_ok() && got == *src,
                            "{}: decode differs",
                            c.tier
                        );
                    }
                }
            });
        }
    });
}

/// Go FuzzRoundTripBlock, over its seeds and seeded random inputs.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn fuzz_round_trip_block_seeded() {
    let mut cs = coders();
    let with = |flags: u8, b: Vec<u8>| [vec![flags], b].concat();
    let mut inputs = vec![
        b"\x00abcab".to_vec(),
        with(5, sample(&dist_of("english"), 3000, 1)),
        with(8, sample(&dist_of("proba80"), 9000, 1)),
        with(1, sample(&dist_of("sym0_255"), 700, 1)),
    ];
    let mut x = Xorshift64(0xB10C);
    for i in 0..60 {
        let n = [2, 9, 100, 1000, 20000][i % 5] + x.intn(50);
        let alpha = 1 + x.intn(256);
        let mut d: Vec<u8> = (0..n).map(|_| (x.next() % alpha as u64) as u8).collect();
        d[0] = x.next() as u8;
        inputs.push(d);
    }
    for d in &inputs {
        fuzz_round_trip_block(&mut cs, d);
    }
}
