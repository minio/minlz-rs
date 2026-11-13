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

//! Port of Go's `table_test.go`.

use super::huffman::lens_test_hists;
use crate::pivco::kernel::{KIND_FLAT, Rec};
use crate::pivco::testutil::*;
use crate::pivco::{
    ERR_CODE_LEN, ERR_KRAFT, ERR_NO_SYMBOLS, Error, MAX_CODE_LEN, Table, build_lengths,
};

/// Returns the records upstream can mark for FSE coding: all but the flats.
fn marks(tb: &Table) -> Vec<u8> {
    tb.recs()
        .iter()
        .enumerate()
        .filter(|(_, r)| r.kind() != KIND_FLAT)
        .map(|(i, _)| i as u8)
        .collect()
}

fn sched_of(ops: &[[u8; 3]]) -> Vec<Rec> {
    ops.iter()
        .map(|o| Rec {
            op: o[0],
            param: o[1],
            right: o[2],
        })
        .collect()
}

/// (symbol, length) pairs.
type Pairs<'a> = &'a [(usize, u8)];

fn lens_from(pairs: Pairs) -> [u8; 256] {
    let mut l = [0u8; 256];
    for &(s, v) in pairs {
        l[s] = v;
    }
    l
}

#[test]
fn table_examples() {
    let (a, b) = (usize::from(b'A'), usize::from(b'B'));
    let mut lens = [0u8; 256];
    (lens[a], lens[b]) = (1, 3);
    lens[usize::from(b'C')..=usize::from(b'H')].fill(4);
    let tb = must_table(&lens);
    assert_eq!(
        tb.recs(),
        sched_of(&[[3, 0, 1], [0, 4, 2], [9, 1, 0], [3, 5, 1], [2, 6, 0]])
    );
    assert_eq!(&tb.r2s[..usize::from(tb.nranks)], b"ACDEFBGH");

    let cases: [(&str, Pairs, Vec<Rec>); 5] = [
        (
            "abc",
            &[(97, 1), (98, 2), (99, 2)],
            sched_of(&[[3, 0, 1], [2, 1, 0]]),
        ),
        ("two", &[(0, 1), (255, 1)], sched_of(&[[2, 0, 0]])),
        (
            "flat2",
            &[(97, 2), (98, 2), (99, 2), (100, 2)],
            sched_of(&[[9, 0, 0]]),
        ),
        ("single", &[(usize::from(b'x'), 5)], sched_of(&[[2, 0, 0]])),
        ("single11", &[(0, 11)], sched_of(&[[2, 0, 0]])),
    ];
    for (name, pairs, sched) in cases {
        let l = lens_from(pairs);
        let tb = must_table(&l);
        assert!(
            tb.recs() == sched && tb.lens == l,
            "{name}: sched {:?}",
            tb.recs()
        );
    }

    lens.fill(8);
    let tb = must_table(&lens);
    assert!(
        tb.recs() == sched_of(&[[33, 0, 0]]) && tb.nranks == 256,
        "256x8: {:?}",
        tb.recs()
    );
    for i in 0..256 {
        assert!(
            tb.r2s[i] == i as u8 && tb.s2r[i] == i as u8,
            "256x8: not identity"
        );
    }
}

/// Returns valid and invalid length sets: lengths of lens_test_hists, random complete codes
/// and random junk.
pub(super) fn table_test_lens() -> Vec<[u8; 256]> {
    let mut out: Vec<[u8; 256]> = lens_test_hists()
        .iter()
        .map(|h| {
            let mut l = [0u8; 256];
            build_lengths(h, &mut l);
            l
        })
        .collect();
    let mut r = Xorshift64(0xC0FFEE);
    for _ in 0..3000 {
        let mut l = [0u8; 256];
        let p = r.perm();
        if r.intn(4) == 0 {
            let n = r.intn(256);
            for &s in &p[..n] {
                l[usize::from(s)] = r.intn(13) as u8;
            }
            out.push(l);
            continue;
        }
        // Random complete code: split random leaves until there are n.
        let mut depths = vec![0u8];
        let n = 1 + r.intn(256);
        while depths.len() < n {
            let mut i = r.intn(depths.len());
            if usize::from(depths[i]) == MAX_CODE_LEN {
                match depths.iter().position(|&d| usize::from(d) < MAX_CODE_LEN) {
                    Some(j) => i = j,
                    None => break,
                }
            }
            depths[i] += 1;
            depths.push(depths[i]);
        }
        for (i, &d) in depths.iter().enumerate() {
            l[usize::from(p[i])] = d;
        }
        if depths.len() == 1 {
            l[usize::from(p[0])] = 1 + r.intn(MAX_CODE_LEN) as u8;
        }
        out.push(l);
    }
    out
}

fn table_digest(tb: &Table) -> Vec<u8> {
    let mut b = vec![tb.nsched as u8, (tb.nranks >> 8) as u8, tb.nranks as u8];
    for r in tb.recs() {
        b.extend_from_slice(&[r.op, r.param, r.right]);
    }
    b.extend_from_slice(&tb.r2s[..usize::from(tb.nranks)]);
    for (s, &l) in tb.lens.iter().enumerate() {
        if l != 0 {
            b.push(tb.s2r[s]);
        }
    }
    b.extend(marks(tb));
    b
}

/// Digest of table_digest over table_test_lens (0xFF for rejected sets), verified against
/// C pivco_build_codec_table at the commit in the module doc.
const TABLE_TEST_DIGEST: u64 = 0x310e97615f352508;

#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn table_c() {
    let mut all = Vec::new();
    let mut nvalid = 0;
    for l in table_test_lens() {
        let mut tb = Table::default();
        if let Err(e) = tb.set_lengths(&l) {
            assert!(matches!(e, Error::Corrupt(_)), "{e}");
            all.push(0xFF);
            continue;
        }
        nvalid += 1;
        check_table(&tb);
        all.extend(table_digest(&tb));
    }
    assert_eq!(fnv1a64(&all), TABLE_TEST_DIGEST, "{nvalid} valid");
}

#[test]
fn table_reject() {
    let good = lens_from(&[(1, 1), (2, 1)]);
    let cases: [(&str, Pairs, Error); 9] = [
        ("empty", &[], ERR_NO_SYMBOLS),
        ("len12", &[(0, 1), (1, 12)], ERR_CODE_LEN),
        // build reads 4 bits, which make 16 an unused symbol of a complete code and 31 length 15.
        ("len16", &[(0, 1), (1, 1), (2, 16)], ERR_CODE_LEN),
        ("len31", &[(0, 1), (1, 1), (2, 31)], ERR_CODE_LEN),
        ("len255", &[(9, 255)], ERR_CODE_LEN),
        ("single12", &[(9, 12)], ERR_CODE_LEN),
        ("incomplete", &[(0, 1), (1, 2)], ERR_KRAFT),
        ("overfull", &[(0, 1), (1, 1), (2, 1)], ERR_KRAFT),
        ("overfull11", &[(0, 1), (1, 2), (2, 2), (3, 11)], ERR_KRAFT),
    ];
    for (name, pairs, want) in cases {
        let mut tb = must_table(&good);
        let got = tb.set_lengths(&lens_from(pairs));
        assert!(
            got == Err(want) && matches!(want, Error::Corrupt(_)),
            "{name}: set_lengths = {got:?}, want {want}"
        );
        assert!(
            !tb.valid() && tb.block_size(&[0, 1]).is_none(),
            "{name}: table still valid"
        );
    }
}

#[test]
fn table_rebuild() {
    // Reusing a Table must not leak state from a previous table with more ranks or records.
    let small = lens_from(&[(97, 1), (98, 2), (99, 2)]);
    let fresh = must_table(&small);
    let state = |t: &Table| (*t.lens(), t.recs().to_vec(), *t.r2s(), *t.s2r(), t.nranks());
    for prev in [[8u8; 256], lens_of(&dist_of("english"))] {
        let mut tb = must_table(&prev);
        assert!(tb.set_lengths(&small).is_ok(), "build");
        assert!(
            state(&tb) == state(&fresh),
            "rebuilt table differs from a fresh one"
        );
    }
}

/// Go FuzzTable's seeds: 128 bytes of length nibbles, then a payload.
pub(super) fn table_seeds() -> Vec<Vec<u8>> {
    let mut seeds = Vec::new();
    for (i, l) in table_test_lens()[..40].iter().enumerate() {
        let mut b = vec![0u8; 128];
        for (s, &v) in l.iter().enumerate() {
            b[s / 2] |= v << (4 * (s & 1));
        }
        b.extend(core::iter::repeat_n(i as u8, i));
        seeds.push(b);
    }
    seeds
}

/// Go FuzzTable, over its seeds and seeded random mutations of them.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn fuzz_table_seeded() {
    let mut cs = coders();
    let mut x = Xorshift64(0xF022);
    for seed in &table_seeds() {
        fuzz_table_lens(&mut cs, seed);
        for _ in 0..5 {
            let mut d = seed.clone();
            for _ in 0..1 + x.intn(4) {
                let i = x.intn(d.len());
                d[i] ^= 1 << x.intn(8);
            }
            d.extend((0..x.intn(3000)).map(|_| x.next() as u8));
            fuzz_table_lens(&mut cs, &d);
        }
    }
}
