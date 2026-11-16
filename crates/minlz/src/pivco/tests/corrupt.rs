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

//! Port of Go's `corrupt_test.go`.

use crate::pivco::kernel::{KIND_FULL, KIND_LEAF_LEFT};
use crate::pivco::testutil::*;
use crate::pivco::{
    ERR_MERGE, ERR_PADDING, ERR_SINGLE, ERR_SPLIT, ERR_TRAILING, ERR_TRUNCATED, Error,
    MAX_BLOCK_SIZE,
};

fn first_coder() -> Coder {
    coders().swap_remove(0)
}

#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn corrupt_truncated() {
    let mut c = first_coder();
    for cc in corrupt_cases() {
        for i in 0..cc.blk.len() {
            assert!(
                decode_err(&mut c, &cc.blk[..i], cc.src.len(), &cc.tb).is_err(),
                "{}: truncated to {i} of {} bytes decodes",
                cc.name,
                cc.blk.len()
            );
        }
        let mut b = cc.blk.clone();
        b.push(0);
        assert_eq!(
            decode_err(&mut c, &b, cc.src.len(), &cc.tb),
            Err(ERR_TRAILING),
            "{}: trailing byte",
            cc.name
        );
    }
}

/// Flips every bit of every case: the decoder must reject the result or decode it to
/// symbols that encode back to the flipped block.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn corrupt_bit_flips() {
    let mut c = first_coder();
    for cc in corrupt_cases() {
        let mut b = cc.blk.clone();
        for i in 0..b.len() * 8 {
            b[i / 8] ^= 1 << (i % 8);
            if let Ok(got) = decode_err(&mut c, &b, cc.src.len(), &cc.tb) {
                canon(&mut c, &cc.tb, &b, &got);
            }
            b[i / 8] ^= 1 << (i % 8);
        }
    }
}

/// Decodes blocks with other symbol counts than they hold, which the format does not
/// store: the decoder must reject them or decode symbols that encode back to the same block.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn corrupt_count() {
    let mut c = first_coder();
    let (mut accepted, mut rejected) = (0, 0);
    for cc in corrupt_cases() {
        let n = cc.src.len() as isize;
        for m in [n - 8, n - 1, n + 1, n + 7, n + 8, n / 2, 2 * n] {
            if m < 1 || m == n {
                continue;
            }
            match decode_err(&mut c, &cc.blk, m as usize, &cc.tb) {
                Err(_) => rejected += 1,
                Ok(got) => {
                    canon(&mut c, &cc.tb, &cc.blk, &got);
                    accepted += 1;
                }
            }
        }
    }
    // A block is only also valid for another count when the extra or missing symbols take
    // the leftmost path at every node without changing a region's size.
    assert!(
        rejected != 0,
        "{accepted} other counts accepted, {rejected} rejected"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn corrupt_fields() {
    let mut c = first_coder();
    for cc in corrupt_cases() {
        for f in walk_block(&cc.tb, &cc.blk, cc.src.len()) {
            let mut mutate = |what: &str, want: Error, edit: &dyn Fn(&mut Vec<u8>)| {
                let mut b = cc.blk.clone();
                edit(&mut b);
                let got = decode_err(&mut c, &b, cc.src.len(), &cc.tb);
                assert_eq!(
                    got,
                    Err(want),
                    "{}: {what} of {} at {}",
                    cc.name,
                    f.kind as char,
                    f.pos
                );
            };
            if f.kind == b'k' {
                let k1 = (f.k + 1) as u16;
                mutate("K+1", ERR_SPLIT, &|b| {
                    b[f.pos..f.pos + 2].copy_from_slice(&k1.to_le_bytes())
                });
                mutate("0xFFFF", ERR_SPLIT, &|b| b[f.pos..f.pos + 2].fill(0xFF));
            }
            if (f.kind == b'b' || f.kind == b'f') && f.bits % 8 != 0 {
                mutate("padding", ERR_PADDING, &|b| b[f.pos + f.size - 1] |= 0x80);
            }
        }
    }
}

#[test]
fn corrupt_short() {
    let mut c = first_coder();
    let mut lens = [0u8; 256];
    (
        lens[usize::from(b'a')],
        lens[usize::from(b'b')],
        lens[usize::from(b'c')],
    ) = (1, 2, 2);
    let tb = must_table(&lens);
    let cases: [(usize, &[u8]); 5] = [
        (5, &[]),
        (5, &[5]),
        (5, &[0, 0]),
        (5, &[3, 0, 2]),
        (MAX_BLOCK_SIZE, &[0xFF, 0xFF, 0]),
    ];
    for (n, b) in cases {
        assert!(decode_err(&mut c, b, n, &tb).is_err(), "{b:x?} decodes");
    }
}

/// Samples bit flips and truncations of maximum size blocks.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn corrupt_large() {
    let mut c = first_coder();
    let mut x = Xorshift64(99);
    for name in ["english", "zipfian", "odd_sigma"] {
        let h = dist_of(name);
        let tb = must_table(&lens_of(&h));
        let src = sample(&h, MAX_BLOCK_SIZE, 3);
        let mut b = round_trip(&mut c, &src, &tb);
        for _ in 0..300 {
            let i = (x.next() % (b.len() * 8) as u64) as usize;
            b[i / 8] ^= 1 << (i % 8);
            if let Ok(got) = decode_err(&mut c, &b, src.len(), &tb) {
                canon(&mut c, &tb, &b, &got);
            }
            b[i / 8] ^= 1 << (i % 8);
            let cut = (x.next() % b.len() as u64) as usize;
            assert!(
                decode_err(&mut c, &b[..cut], src.len(), &tb).is_err(),
                "{name}: truncated block decodes"
            );
        }
    }
}

/// Checks that no op runs before the whole block is parsed: a bad bitmap closing the first
/// child of the root is a merge error in the full block, but truncation when the block is
/// cut right after it.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn decode_parse_first() {
    let mut c = first_coder();
    let mut tested = 0;
    for name in TEST_DISTS {
        let h = dist_of(name);
        let tb = must_table(&lens_of(&h));
        let r = tb.sched[0];
        if r.kind() != KIND_FULL {
            continue;
        }
        let src = sample(&h, MAX_BLOCK_SIZE, 5);
        let blk = round_trip(&mut c, &src, &tb);
        let kr = usize::from(blk[0]) | usize::from(blk[1]) << 8;
        let first = if kr > src.len() - kr {
            usize::from(r.right)
        } else {
            1
        };
        let k = tb.sched[first].kind();
        if k != KIND_FULL && k != KIND_LEAF_LEFT {
            continue;
        }
        for f in walk_block(&tb, &blk, src.len()) {
            if f.kind != b'b' || f.rec != first {
                continue;
            }
            let mut b = blk.clone();
            b[f.pos] ^= 1;
            assert_eq!(
                decode_err(&mut c, &b, src.len(), &tb),
                Err(ERR_MERGE),
                "{name}"
            );
            assert_eq!(
                decode_err(&mut c, &b[..f.pos + f.size], src.len(), &tb),
                Err(ERR_TRUNCATED),
                "{name}: cut block"
            );
            tested += 1;
        }
    }
    assert!(
        tested != 0,
        "no distribution has a merging first child under a FULL root"
    );
}

/// A single-symbol table's bitmap must be all zeros.
#[test]
fn corrupt_single() {
    let mut c = first_coder();
    let mut x = [0u8; 256];
    x[usize::from(b'x')] = 1;
    assert_eq!(
        decode_err(&mut c, &[0xff, 1], 9, &must_table(&x)),
        Err(ERR_SINGLE)
    );
}
