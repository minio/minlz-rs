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

//! Block tests of Go's `decode-simd_test.go`, and the SIMD parts of the block API (`decode.go`).

use std::collections::HashMap;

use super::*;
use crate::Error;
use crate::block::format::get_uvarint;
use crate::block::simd::decode::{Decoder, Tables, ll_stats, off_stats, tok_stats};
use crate::block::simd::exec::{Tier, simd_tiers};
use crate::block::simd::{MAX_CHUNK, MAX_CHUNKS, OFF_TAB, PAD, rvarint};
use crate::guard::Buffer;
use crate::reference::decode_simd_block;

/// Returns the MinLZ block of a SIMD block body: the header of size bytes, then body.
fn with_header(size: u64, body: &[u8]) -> Vec<u8> {
    let mut b = vec![0];
    put_uvarint(&mut b, BLOCK_TYPE << 24 | size);
    b.extend_from_slice(body);
    b
}

/// Decodes block from a copy that ends at an inaccessible page.
fn guarded_decode(block: &[u8]) -> Result<Vec<u8>, Error> {
    let mut g = Buffer::new(block.len()).unwrap();
    let s = g.slice(block.len(), true);
    s.copy_from_slice(block);
    let mut got = Vec::new();
    crate::decode(&mut got, s).map(|()| got)
}

#[test]
fn simd_rvarint() {
    for (v, b) in [
        (5u64, &[0x05u8][..]),
        (300, &[0x02, 0xac]),
        (100000, &[0x06, 0x8d, 0xa0]),
    ] {
        let mut got = Vec::new();
        test_rvarint(&mut got, v);
        assert_eq!(got, b, "{v}");
        assert_eq!(
            rvarint(&[&[0x80, 0xff], b].concat()),
            Some((v, b.len())),
            "{b:x?}"
        );
    }
    for v in [
        0,
        127,
        128,
        16383,
        16384,
        1 << 21,
        (1 << 35) + 1,
        (1 << 63) + 5,
        u64::MAX,
    ] {
        let mut b = vec![0x80];
        test_rvarint(&mut b, v);
        assert_eq!(rvarint(&b), Some((v, b.len() - 1)), "{v}");
        assert!(
            rvarint(&b[2..]).is_none() || v < 128,
            "{v}: truncated varint accepted"
        );
    }
    assert_eq!(rvarint(&[0x80; 10]), None, "unterminated varint accepted");
    assert_eq!(
        rvarint(&[&[2][..], &[0xff; 9]].concat()),
        None,
        "overflow accepted"
    );
}

#[test]
fn simd_offsets() {
    let mut next = 1;
    for (s, &t) in OFF_TAB.iter().enumerate().take(176).skip(1) {
        let (base, n) = ((t & 0xffffff) as usize, (t >> 24) as usize);
        assert_eq!(base, next, "symbol {s}");
        next = base + (1 << n);
        for off in [base, base + (1 << n) - 1] {
            let (sym, raw, nb) = test_offset(off);
            assert!(
                sym as usize == s && nb == n && raw as usize == off - base,
                "offset {off}: symbol {sym}/{nb} bits, want {s}/{n}"
            );
        }
    }
    assert_eq!(next, 1 << 24, "offsets end");
    for (s, want) in [(16, [16, 17]), (24, [32, 35]), (175, [15728640, 16777215])] {
        let (base, n) = ((OFF_TAB[s] & 0xffffff) as usize, OFF_TAB[s] >> 24);
        assert_eq!([base, base + (1 << n) - 1], want, "symbol {s}");
    }
    assert_eq!(OFF_TAB[0], 1, "symbol 0 must decode as offset 1");
}

/// Parses "Class, values 1-33:" and "In-class length, symbols 1-175:".
fn caption(line: &str) -> Option<(&str, usize, usize)> {
    let (kind, rest) = line.split_once(", ")?;
    let range = rest.strip_suffix(':')?.rsplit(' ').next()?;
    let (lo, hi) = range.split_once('-')?;
    let ok = matches!(kind, "Class" | "In-class length")
        && rest
            .bytes()
            .all(|c| c.is_ascii_lowercase() || b" 0123456789-:".contains(&c));
    ok.then_some((kind, lo.parse().ok()?, hi.parse().ok()?))
}

/// Compares the class layouts with the arrays in SPEC_SIMD.md appendix A.
#[test]
#[cfg_attr(miri, ignore = "parses the spec text slowly and runs no unsafe code")]
fn simd_layouts_spec() {
    let spec = include_str!("../../../reference/testdata/SPEC_SIMD_appendix_a.md");
    let streams = [
        ("A.1", LL),
        ("A.2", ESC),
        ("A.3", OFFC),
        ("A.4", TOK),
        ("A.5", LITS),
        ("A.6", DELTA),
    ];
    let mut found: HashMap<String, isize> = HashMap::new();
    let (mut section, mut kind, mut lo, mut idx) = ("", "", 0, 0);
    for line in spec.lines().map(str::trim) {
        if line.starts_with("## A.") {
            (section, kind) = (&line[3..6], "");
            continue;
        }
        let Some(&(_, k)) = streams.iter().find(|s| s.0 == section) else {
            continue;
        };
        if let Some((kd, l, hi)) = caption(line) {
            (kind, idx, lo) = (kd, 0, l);
            found.insert(format!("{section}{kind}"), (hi - l + 1) as isize);
            continue;
        }
        if line.is_empty() || line == "```" {
            kind = "";
            continue;
        }
        if kind.is_empty() {
            continue;
        }
        for f in line.split(',').map(str::trim).filter(|f| !f.is_empty()) {
            let want: u8 = f.parse().unwrap_or_else(|_| panic!("{section}: {line:?}"));
            let (v, l) = (lo + idx, &LAYOUTS[k]);
            let got = if kind == "Class" {
                l.class[v]
            } else {
                l.inlen[v]
            };
            assert_eq!(got, want, "{section} {kind} of {v}");
            idx += 1;
            *found.get_mut(&format!("{section}{kind}")).unwrap() -= 1;
        }
    }
    for (s, _) in streams {
        for kind in ["Class", "In-class length"] {
            assert_eq!(
                found.get(&format!("{s}{kind}")),
                Some(&0),
                "{s} {kind}: values missing"
            );
        }
    }
    // Every in-class code is complete, and the table sizes are those of 4.2.
    for (k, want) in [
        (TOK, 7),
        (LL, 6),
        (OFFC, 12),
        (ESC, 8),
        (LITS, 7),
        (DELTA, 5),
    ] {
        let l = &LAYOUTS[k];
        let mut kraft = [0f64; 24];
        for (v, &c) in l.class.iter().enumerate() {
            if c != 0xff {
                kraft[c as usize] += 1.0 / (1u64 << l.inlen[v]) as f64;
            }
        }
        for (c, &x) in kraft[..l.n].iter().enumerate() {
            assert_eq!(x, 1.0, "stream {k} class {c}: Kraft sum");
        }
        assert_eq!(l.n.div_ceil(2), want, "stream {k}: bytes");
    }
}

#[test]
#[cfg_attr(miri, ignore = "40 blocks of up to 5000 operations: over 30 minutes")]
fn simd_decode_built() {
    simd_tiers(|_| {
        // Go's draws make blocks of more than MAX_CHUNKS chunks, or chunks of 1 operation with
        // code length tables, which never pad to 64 bytes below their output. Go's seed avoids
        // both; so does this one.
        let mut rng = Rng(4);
        let mut b = TestBuilder::default();
        let modes = [STORED, CODE_LENS, CURRENT, CLASS_LENS, CURRENT];
        let lit_modes = [STORED, CODE_LENS, CURRENT, CLASS_LENS, DELTA_LENS];
        for i in 0..40 {
            let n = 1 + rng.intn(5000);
            let max_off = 1 << rng.intn(25);
            let (ops, lits) = test_ops(&mut rng, n, b"aaabbcdef\x00\xff", 288, max_off);
            let limit = rng.intn(7);
            let mut tb = TestBlock {
                ops,
                lits,
                limit,
                chunk: (1 + rng.intn(1500)).min(1024 << limit),
                lit_block: 1 + rng.intn(9000),
                modes: modes[..1 + i % modes.len()].to_vec(),
                lit_modes: lit_modes[..1 + i % lit_modes.len()].to_vec(),
                ..Default::default()
            };
            if i % 3 == 1 {
                tb.delta = rng.uint64();
            }
            if i % 4 >= 2 {
                // Not from rng, which would change the other blocks.
                tb.flat = (i as u64).wrapping_mul(0x9e3779b97f4a7c15);
            }
            let (e, want) = test_pad(&mut b, &mut tb);
            let block = e.bytes();
            test_decode(&block, &want);
            assert_eq!(decode_simd_block(&block), Ok(want), "block {i}: reference");
        }
    });
}

/// A stored entry keeps the current table (SPEC_SIMD.md 2.2), in the chunks and the literal
/// blocks.
#[test]
fn simd_stored_keeps_table() {
    // Miri: 2 KiB of output, not 39 KiB; the builder and reference are slow per output byte.
    let (chunk, ml, lit_block) = if cfg!(miri) {
        (16, 40, 64)
    } else {
        (64, 200, 256)
    };
    let mut ops = vec![op(4, ml, 40); 3 * chunk];
    // The first match must not be clamped below 32 (4.3).
    ops[0].ll = 40;
    let mut b = TestBuilder::default();
    let modes = vec![CODE_LENS, STORED, CURRENT];
    let (e, want) = b.build(&TestBlock {
        lits: b"abcd".repeat(ops.len() + 9),
        ops,
        chunk,
        lit_block,
        modes: modes.clone(),
        lit_modes: modes.clone(),
        ..Default::default()
    });
    for (i, &m) in modes.iter().enumerate() {
        assert!(
            e.chunks[i].s[TOK].mode == m && e.lits[i].mode == m,
            "entry {i}: token mode {}, literal mode {}, want {m}",
            e.chunks[i].s[TOK].mode,
            e.lits[i].mode
        );
    }
    test_decode(&e.bytes(), &want);
    assert_eq!(decode_simd_block(&e.bytes()), Ok(want), "reference");
}

#[test]
fn simd_delta_example() {
    // SPEC_SIMD.md 4.5: the output ends with 41 42 43 07 00 41 42 43, and the repeat offset is 5.
    let lits = vec![
        0x41, 0x42, 0x43, 0x07, 0x00, 0x41, 0x42, 0x43, 0x07, 0x01, 0x41, 0x42, 0x44, 0x07,
    ];
    let mut tb = TestBlock {
        ops: vec![op(8, 0, 5), op(6, 4, 0)],
        lits,
        chunk: 1,
        delta: 2,
        ..Default::default()
    };
    let (e, want) = test_pad(&mut TestBuilder::default(), &mut tb);
    assert_eq!(
        e.stored[8..],
        [0x00, 0x01, 0x00, 0x00, 0x01, 0x00],
        "stored"
    );
    test_decode(&e.bytes(), &want);
}

/// The long match rule (4.3) on the last operation of a block, which runs in exec_exact.
#[test]
fn simd_long_match_tail() {
    simd_tiers(|_| {
        let mut b = TestBuilder::default();
        for (last, ok) in [
            (op(1, 40, 32), true),
            (op(1, 40, 20), false),
            (op(1, 40, 0), false), // The repeat offset is 20.
        ] {
            let ops = vec![
                op(40, 288, 32),
                op(0, 288, 0),
                op(0, 288, 0),
                op(1, 4, 20),
                last,
            ];
            // No literals follow the operations, so the last one ends the output.
            let (e, want) = b.build(&TestBlock {
                ops,
                lits: b"abcdef".repeat(7),
                ..Default::default()
            });
            let block = e.bytes();
            assert!(
                block.len() - 1 + MIN_SAVING <= want.len(),
                "block too large"
            );
            if ok {
                test_decode(&block, &want);
            } else {
                let r = crate::decode(&mut Vec::new(), &block);
                assert!(
                    matches!(r, Err(Error::Corrupt(_))),
                    "last operation {last:?}: {r:?}"
                );
            }
        }
    });
}

/// Moves the end of the block over the last operations, so the exact loop takes over at each
/// of them.
#[test]
fn simd_tail() {
    // Miri: shorter blocks and every eighth tail.
    let (n_ops, step) = if cfg!(miri) { (60, 8) } else { (200, 1) };
    simd_tiers(|_| {
        let mut rng = Rng(3);
        let mut b = TestBuilder::default();
        for tail in (0..40).step_by(step) {
            for delta in [0, 1] {
                let (mut ops, mut lits) = test_ops(&mut rng, n_ops, b"abcd", 288, 1000);
                for _ in 0..20 {
                    let (ll, ml) = (rng.intn(3), rng.intn(4));
                    let off = rng.intn(2) * (1 + rng.intn(10));
                    ops.push(op(ll, ml, off));
                }
                let n: usize = ops.iter().map(|o| o.ll).sum();
                lits.resize(lits.len().max(n + tail), b'x');
                lits.truncate(n + tail);
                let mut tb = TestBlock {
                    ops,
                    lits,
                    delta,
                    ..Default::default()
                };
                let (e, want) = test_pad(&mut b, &mut tb);
                test_decode(&e.bytes(), &want);
            }
        }
    });
}

/// The SIMD fixtures of the Go encoder up to max bytes, with their reference output.
pub(super) fn go_fixtures(max: usize) -> Vec<(String, Vec<u8>)> {
    let mut out: Vec<(String, Vec<u8>)> = crate::unzip::simd_fixtures()
        .into_iter()
        .filter(|(n, b)| n.ends_with(".mzb") && b.len() <= max)
        .collect();
    out.sort();
    assert!(!out.is_empty(), "no fixtures");
    out
}

/// Decodes blocks that end at an inaccessible page. With few literals, the offset bits of the
/// last chunk are close to the end of the block. Go also encodes copies of earlier data at 4
/// and 64 KiB at every SIMD level; the Go fixtures stand in for them.
#[test]
fn simd_guard_page() {
    simd_tiers(|_| {
        let mut b = TestBuilder::default();
        let mut blocks = Vec::new();
        // Miri: the shorter blocks, whose last raw bits still end at each bit position.
        for n in (1..=if cfg!(miri) { 92 } else { 300 }).step_by(13) {
            // 17 raw bits per operation.
            let mut ops = vec![op(1, 32, 1 << 20)];
            ops.extend((0..n).map(|i| op(0, 32, (1 << 20) + i)));
            let mut tb = TestBlock {
                ops,
                lits: b"a".to_vec(),
                delta: (n & 1) as u64,
                ..Default::default()
            };
            let (e, want) = test_pad(&mut b, &mut tb);
            blocks.push((format!("{n} operations"), e.bytes(), want));
        }
        // Miri isolates the file system.
        let fixtures = if cfg!(miri) {
            Vec::new()
        } else {
            go_fixtures(64 << 10)
        };
        for (name, block) in fixtures {
            let want = decode_simd_block(&block).unwrap();
            blocks.push((name, block, want));
        }
        for (name, block, want) in &blocks {
            assert_eq!(guarded_decode(block).as_ref(), Ok(want), "{name}");
        }
    });
}

type Change = Box<dyn Fn(&mut TestBuilder, &mut TestEnc, Vec<u8>) -> Vec<u8>>;
type Raw = Box<dyn Fn(Vec<u8>) -> Vec<u8>>;

/// A case of [`simd_validation`]. Empty modes are stored.
#[derive(Default)]
struct VCase {
    name: &'static str,
    modes: Vec<usize>,
    ok: bool,
    /// Replaces the operations.
    ops: Option<Vec<TestOp>>,
    chunk: usize,
    /// Changes the block, and returns its output.
    change: Option<Change>,
    raw: Option<Raw>,
}

fn same(f: impl Fn(&mut TestEnc) + 'static) -> Option<Change> {
    Some(Box::new(move |_, e, want| {
        f(e);
        want
    }))
}

fn chunk0(e: &mut TestEnc, k: usize) -> &mut TestStream {
    &mut e.chunks[0].s[k]
}

/// Codes stream k of the first chunk with lens, in an entry of mode and table.
fn recode(
    b: &mut TestBuilder,
    e: &mut TestEnc,
    k: usize,
    mode: usize,
    table: Vec<u8>,
    lens: &[u8; 256],
) {
    let mut tab = pivco::Table::default();
    tab.set_lengths(lens).unwrap();
    let data = b.code(k, &e.chunks[0].vals[k], &tab).unwrap();
    *chunk0(e, k) = TestStream { mode, table, data };
}

/// Returns the start of the LZ section, its size and the size of L.
fn sections(b: &[u8]) -> (usize, usize, usize) {
    let (_, h) = get_uvarint(&b[1..]).unwrap();
    let (lz, n) = rvarint(b).unwrap();
    (2 + h, lz as usize, n)
}

/// Sets the reverse varint that ends at end to 1 more than the bytes from start to it.
fn past(b: &[u8], start: usize, end: usize) -> Vec<u8> {
    let (_, n) = rvarint(&b[..end]).unwrap();
    let mut out = b[..end - n].to_vec();
    test_rvarint(&mut out, (end - n - start + 1) as u64);
    out.extend_from_slice(&b[end..]);
    out
}

fn validation_cases() -> Vec<VCase> {
    let coded = || vec![CODE_LENS, CURRENT, CLASS_LENS];
    let class = || vec![CLASS_LENS];
    let lits_only = |no_lz: bool| -> Option<Change> {
        Some(Box::new(move |_, e, _| {
            let lits = b"aaab".repeat(400);
            *e = TestEnc {
                typ: BLOCK_TYPE,
                size: lits.len(),
                no_lz,
                lits: vec![TestBuilder::default().stream(LITS, CODE_LENS, &lits)],
                ..Default::default()
            };
            lits
        }))
    };
    let trim = |k: usize| {
        same(move |e| {
            chunk0(e, k).data.pop();
        })
    };
    let extend = |k: usize| same(move |e| chunk0(e, k).data.push(1));
    // Codes stream k with a valid delta class table, which only literals may use.
    let delta = |k: usize| -> Option<Change> {
        Some(Box::new(move |b, e, want| {
            let (cl, lens) = test_class_lens(LITS, DELTA_LENS);
            recode(b, e, k, DELTA_LENS, test_nibbles(&cl), &lens);
            want
        }))
    };
    // Codes stream k with a complete code lengths table that also has a code for 0.
    let code_for_0 = |k: usize| -> Option<Change> {
        Some(Box::new(move |b, e, want| {
            let mut h = [0u64; 256];
            for &v in &e.chunks[0].vals[k] {
                h[v as usize] += 1;
            }
            h[0] = 1;
            let mut lens = [0u8; 256];
            pivco::build_lengths(&h, &mut lens);
            let table = test_nibbles(&lens[..2 * CODE_BYTES[k]]);
            recode(b, e, k, CODE_LENS, table, &lens);
            want
        }))
    };
    // Codes escape 0 as the only value, in 1 bit.
    let escape_table = || {
        let mut t = vec![0; CODE_BYTES[ESC]];
        t[0] = 1;
        t
    };
    // A block of one chunk with 32768 operations and 65536 escapes, stored or coded.
    let escapes = move |coded: bool| -> Option<Change> {
        Some(Box::new(move |b, e, _| {
            let mut ops = vec![op(MAX_LEN + 1, MAX_LEN + 1, 0); 1 << 15];
            ops[0].off = MAX_LEN;
            let (ne, want) = b.build(&TestBlock {
                lits: b"abc".repeat(11 * ops.len()),
                ops,
                limit: 5,
                ..Default::default()
            });
            *e = ne;
            if coded {
                *chunk0(e, ESC) = TestStream {
                    mode: CODE_LENS,
                    table: escape_table(),
                    data: vec![0; (1 << 16) / 8],
                };
            }
            want
        }))
    };
    let ones = |n| Some(vec![TestOp::default(); n]);
    let raw = |f: fn(Vec<u8>) -> Vec<u8>| -> Option<Raw> { Some(Box::new(f)) };
    vec![
        VCase {
            name: "valid",
            ok: true,
            ..Default::default()
        },
        VCase {
            name: "valid coded",
            modes: coded(),
            ok: true,
            ..Default::default()
        },
        VCase {
            name: "valid class",
            modes: class(),
            ok: true,
            ..Default::default()
        },
        VCase {
            name: "type 2",
            change: same(|e| e.typ = 2),
            ..Default::default()
        },
        VCase {
            name: "type 255",
            change: same(|e| e.typ = 255),
            ..Default::default()
        },
        VCase {
            name: "size above 8MiB",
            change: same(|e| e.size = crate::MAX_BLOCK_SIZE + 1),
            ..Default::default()
        },
        VCase {
            name: "size+1",
            change: same(|e| e.size += 1),
            ..Default::default()
        },
        VCase {
            name: "size-1",
            change: same(|e| e.size -= 1),
            ..Default::default()
        },
        VCase {
            name: "flags bit 5",
            change: same(|e| e.flags |= 1 << 5),
            ..Default::default()
        },
        VCase {
            name: "flags bit 6",
            change: same(|e| e.flags |= 1 << 6),
            ..Default::default()
        },
        VCase {
            name: "flags bit 7",
            change: same(|e| e.flags |= 1 << 7),
            ..Default::default()
        },
        VCase {
            name: "chunk limit 5",
            ok: true,
            change: same(|e| e.flags |= 5),
            ..Default::default()
        },
        VCase {
            name: "chunk limit 6",
            ok: true,
            change: same(|e| e.flags |= 6),
            ..Default::default()
        },
        VCase {
            name: "chunk limit 7",
            change: same(|e| e.flags |= 7),
            ..Default::default()
        },
        VCase {
            name: "hint bits",
            ok: true,
            change: same(|e| e.flags |= 3 << 3),
            ..Default::default()
        },
        VCase {
            name: "no flags",
            raw: raw(|b| b[..5].to_vec()),
            ..Default::default()
        },
        VCase {
            name: "L too large",
            raw: raw(|b| {
                let (_, n) = rvarint(&b).unwrap();
                let mut c = b[..b.len() - n].to_vec();
                test_rvarint(&mut c, b.len() as u64);
                c
            }),
            ..Default::default()
        },
        VCase {
            name: "L unterminated",
            raw: raw(|b| [&b[..6], &[0x80, 0x80]].concat()),
            ..Default::default()
        },
        VCase {
            name: "L past its bytes",
            raw: raw(|b| {
                let (start, _, _) = sections(&b);
                past(&b, start, b.len())
            }),
            ..Default::default()
        },
        VCase {
            name: "literal records past their bytes",
            raw: raw(|b| {
                let (start, lz, n) = sections(&b);
                past(&b, start + lz, b.len() - n)
            }),
            ..Default::default()
        },
        VCase {
            name: "chunk records past their bytes",
            raw: raw(|b| {
                let (start, lz, n) = sections(&b);
                let mut c = past(&b, start, start + lz);
                let grown = lz + c.len() - b.len();
                c.truncate(c.len() - n);
                test_rvarint(&mut c, grown as u64);
                c
            }),
            ..Default::default()
        },
        VCase {
            name: "literal record extra",
            change: same(|e| e.lit_rec_extra = vec![0x80]),
            ..Default::default()
        },
        VCase {
            name: "literal record for missing data",
            change: same(|e| e.lit_rec_extra = vec![1 << 3]),
            ..Default::default()
        },
        VCase {
            name: "literal data extra",
            change: same(|e| e.lit_extra = vec![1]),
            ..Default::default()
        },
        VCase {
            name: "chunk data extra",
            change: same(|e| e.lz_extra = vec![1]),
            ..Default::default()
        },
        VCase {
            name: "chunk record truncated",
            change: same(|e| e.chunk_rec_extra = vec![0]),
            ..Default::default()
        },
        VCase {
            name: "no literal blocks",
            change: same(|e| e.lits.clear()),
            ..Default::default()
        },
        VCase {
            name: "literal block empty",
            change: same(|e| e.lits.push(TestStream::default())),
            ..Default::default()
        },
        VCase {
            name: "literal block of 65536",
            change: same(|e| {
                e.size += 65536;
                e.lits.push(TestStream {
                    data: vec![0; 65536],
                    ..Default::default()
                });
            }),
            ..Default::default()
        },
        VCase {
            name: "literal block of 65535",
            ok: true,
            change: Some(Box::new(|_, e, mut want| {
                e.size += 65535;
                e.lits.push(TestStream {
                    data: vec![0; 65535],
                    ..Default::default()
                });
                want.resize(want.len() + 65535, 0);
                want
            })),
            ..Default::default()
        },
        VCase {
            name: "literals above size",
            change: same(|e| {
                e.lits.push(TestStream {
                    data: vec![0; e.size],
                    ..Default::default()
                })
            }),
            ..Default::default()
        },
        VCase {
            name: "chunk without operations",
            change: same(|e| e.chunks.push(TestChunk::default())),
            ..Default::default()
        },
        VCase {
            name: "chunk above limit",
            ops: ones(1025),
            ..Default::default()
        },
        VCase {
            name: "chunk at limit",
            ok: true,
            ops: ones(1024),
            ..Default::default()
        },
        VCase {
            name: "chunk flag bit 4",
            change: same(|e| e.chunks[0].flags |= 1 << 4),
            ..Default::default()
        },
        VCase {
            name: "chunk flag bit 7",
            change: same(|e| e.chunks[0].flags |= 1 << 7),
            ..Default::default()
        },
        VCase {
            name: "chunk hint bits",
            ok: true,
            change: same(|e| e.chunks[0].flags |= 3 << 2),
            ..Default::default()
        },
        // The flag raises offsets below min(ml, 32) (SPEC_SIMD.md 4.3).
        VCase {
            name: "chunk no-overlap flag",
            ok: true,
            change: Some(Box::new(|_, e, want| {
                e.chunks[0].flags |= 2;
                let out = decode_simd_block(&e.bytes()).expect("reference");
                assert!(out != want, "reference: flag ignored");
                out
            })),
            ..Default::default()
        },
        VCase {
            name: "literal mode 5",
            change: same(|e| e.lits[0].mode = 5),
            ..Default::default()
        },
        VCase {
            name: "literal mode 7",
            change: same(|e| e.lits[0].mode = 7),
            ..Default::default()
        },
        VCase {
            name: "token mode 6",
            change: same(|e| chunk0(e, TOK).mode = 6),
            ..Default::default()
        },
        // Tokens read a token class table in mode 4 too, so only the mode is wrong.
        VCase {
            name: "token mode 4",
            modes: class(),
            change: same(|e| chunk0(e, TOK).mode = DELTA_LENS),
            ..Default::default()
        },
        VCase {
            name: "ll mode 4",
            change: delta(LL),
            ..Default::default()
        },
        VCase {
            name: "offset mode 4",
            change: delta(OFFC),
            ..Default::default()
        },
        VCase {
            name: "escape mode 4",
            change: delta(ESC),
            ..Default::default()
        },
        VCase {
            name: "offset bits mode 1",
            change: same(|e| chunk0(e, 4).mode = CURRENT),
            ..Default::default()
        },
        VCase {
            name: "offset bits mode 2",
            change: same(|e| chunk0(e, 4).mode = CODE_LENS),
            ..Default::default()
        },
        VCase {
            name: "literal mode 1 first",
            modes: coded(),
            change: same(|e| (e.lits[0].mode, e.lits[0].table) = (CURRENT, Vec::new())),
            ..Default::default()
        },
        VCase {
            name: "token mode 1 first",
            modes: coded(),
            change: same(|e| (chunk0(e, TOK).mode, chunk0(e, TOK).table) = (CURRENT, Vec::new())),
            ..Default::default()
        },
        VCase {
            name: "table length 12",
            modes: coded(),
            change: same(|e| e.lits[0].table[0x61 / 2] = 0xcc),
            ..Default::default()
        },
        VCase {
            name: "class length 12",
            modes: class(),
            change: same(|e| chunk0(e, OFFC).table[0] = 0xcc),
            ..Default::default()
        },
        // Literal class 10 has in-class lengths of 6.
        VCase {
            name: "class plus in-class above 11",
            modes: class(),
            change: same(|e| e.lits[0].table[5] = 0x66),
            ..Default::default()
        },
        VCase {
            name: "class pad nibble",
            modes: class(),
            change: same(|e| e.lits[0].table[6] |= 0x10),
            ..Default::default()
        },
        VCase {
            name: "incomplete table",
            modes: coded(),
            change: same(|e| {
                let s = chunk0(e, TOK);
                s.table.fill(0);
                s.table[0] = 0x22;
            }),
            ..Default::default()
        },
        VCase {
            name: "single value of length 2",
            modes: coded(),
            change: same(|e| {
                let s = chunk0(e, TOK);
                s.table.fill(0);
                s.table[0] = 0x02;
            }),
            ..Default::default()
        },
        VCase {
            name: "ll code for 0",
            change: code_for_0(LL),
            ..Default::default()
        },
        VCase {
            name: "offset code for 0",
            change: code_for_0(OFFC),
            ..Default::default()
        },
        VCase {
            name: "pivco block truncated",
            modes: coded(),
            change: trim(TOK),
            ..Default::default()
        },
        VCase {
            name: "pivco block extra",
            modes: coded(),
            change: extend(TOK),
            ..Default::default()
        },
        VCase {
            name: "token count missing",
            modes: coded(),
            change: same(|e| chunk0(e, TOK).data.clear()),
            ..Default::default()
        },
        VCase {
            name: "token count truncated",
            modes: coded(),
            change: same(|e| chunk0(e, TOK).data.truncate(1)),
            ..Default::default()
        },
        VCase {
            name: "literal count missing",
            modes: coded(),
            change: same(|e| e.lits[0].data.clear()),
            ..Default::default()
        },
        VCase {
            name: "literal count truncated",
            modes: coded(),
            change: same(|e| e.lits[0].data.truncate(1)),
            ..Default::default()
        },
        // No length is above 32, so there are no escapes.
        VCase {
            name: "coded empty stream",
            ops: Some(vec![op(1, 4, 3), op(0, 4, 0)]),
            change: same(move |e| {
                *chunk0(e, ESC) = TestStream {
                    mode: CODE_LENS,
                    table: escape_table(),
                    data: Vec::new(),
                };
            }),
            ..Default::default()
        },
        VCase {
            name: "65536 escapes",
            ok: true,
            change: escapes(false),
            ..Default::default()
        },
        VCase {
            name: "coded 65536 escapes",
            change: escapes(true),
            ..Default::default()
        },
        VCase {
            name: "token 136",
            change: same(|e| chunk0(e, TOK).data[1] = 136),
            ..Default::default()
        },
        VCase {
            name: "ll 0",
            change: same(|e| chunk0(e, LL).data[1] = 0),
            ..Default::default()
        },
        VCase {
            name: "ll 34",
            change: same(|e| chunk0(e, LL).data[1] = 34),
            ..Default::default()
        },
        VCase {
            name: "offset symbol 0",
            change: same(|e| chunk0(e, OFFC).data[1] = 0),
            ..Default::default()
        },
        // Symbol 2 has no raw bits, like 176. The last of 13 symbols is outside the 8-byte words.
        VCase {
            name: "offset symbol 176",
            ops: Some(vec![op(1, 4, 2); 13]),
            change: same(|e| *chunk0(e, OFFC).data.last_mut().unwrap() = 176),
            ..Default::default()
        },
        VCase {
            name: "ll extra",
            change: extend(LL),
            ..Default::default()
        },
        VCase {
            name: "ll missing",
            change: trim(LL),
            ..Default::default()
        },
        VCase {
            name: "offset symbol extra",
            change: extend(OFFC),
            ..Default::default()
        },
        VCase {
            name: "offset symbol missing",
            change: trim(OFFC),
            ..Default::default()
        },
        VCase {
            name: "escape extra",
            change: extend(ESC),
            ..Default::default()
        },
        VCase {
            name: "escape missing",
            change: trim(ESC),
            ..Default::default()
        },
        // The pivco block of a literal length stream does not store its count: one more value
        // may code to the same bytes, but eight more make the root bitmap longer.
        VCase {
            name: "coded ll count",
            modes: coded(),
            change: same(|e| {
                let mut v = e.chunks[0].vals[LL].clone();
                v.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
                e.chunks[0].s[LL] = TestBuilder::default().stream(LL, CODE_LENS, &v);
            }),
            ..Default::default()
        },
        VCase {
            name: "coded ll extra",
            modes: coded(),
            change: extend(LL),
            ..Default::default()
        },
        VCase {
            name: "coded ll truncated",
            modes: coded(),
            change: trim(LL),
            ..Default::default()
        },
        VCase {
            name: "coded escape truncated",
            modes: coded(),
            change: trim(ESC),
            ..Default::default()
        },
        VCase {
            name: "offset bits extra",
            change: extend(4),
            ..Default::default()
        },
        VCase {
            name: "offset bits missing",
            change: trim(4),
            ..Default::default()
        },
        VCase {
            name: "offset bits cut",
            change: Some(Box::new(|b, e, _| {
                // The offset bits of the last chunk are more than the bytes after them.
                let mut ops = vec![op(1, 32, 1 << 20)];
                ops.extend((0..3000).map(|i| op(0, 32, (1 << 20) + i)));
                let (ne, want) = b.build(&TestBlock {
                    ops,
                    lits: b"a".to_vec(),
                    ..Default::default()
                });
                *e = ne;
                e.lz_cut = e.chunks.last().unwrap().s[4].data.len() - 8;
                want
            })),
            ..Default::default()
        },
        VCase {
            name: "offset bits unused",
            ok: true,
            change: same(|e| {
                for c in &mut e.chunks {
                    let n: usize = c.vals[OFFC]
                        .iter()
                        .map(|&s| (OFF_TAB[s as usize] >> 24) as usize)
                        .sum();
                    if !n.is_multiple_of(8) {
                        *c.s[4].data.last_mut().unwrap() |= 0xff << (n % 8);
                    }
                }
            }),
            ..Default::default()
        },
        VCase {
            name: "long match offset 20",
            ops: Some(vec![op(100, 40, 20)]),
            ..Default::default()
        },
        VCase {
            name: "long match clamped",
            ops: Some(vec![op(20, 40, 40)]),
            ..Default::default()
        },
        VCase {
            name: "long match repeat 20",
            ops: Some(vec![op(100, 30, 20), op(0, 33, 0)]),
            ..Default::default()
        },
        VCase {
            name: "long match offset 32",
            ok: true,
            ops: Some(vec![op(100, 288, 32)]),
            ..Default::default()
        },
        VCase {
            name: "offset beyond position",
            ok: true,
            ops: Some(vec![op(10, 20, 1000)]),
            ..Default::default()
        },
        VCase {
            name: "first op without literals",
            ops: Some(vec![op(0, 20, 1), op(10, 20, 1)]),
            ..Default::default()
        },
        VCase {
            name: "chunk output below its operations",
            chunk: 8,
            ops: Some([vec![op(7, 0, 0)], vec![TestOp::default(); 7]].concat()),
            ..Default::default()
        },
        VCase {
            name: "chunk output at its operations",
            ok: true,
            chunk: 8,
            ops: Some([vec![op(8, 0, 0)], vec![TestOp::default(); 7]].concat()),
            ..Default::default()
        },
        VCase {
            name: "literal over-read",
            change: same(|e| _ = e.lits.pop()),
            ..Default::default()
        },
        VCase {
            name: "no operations",
            ok: true,
            change: lits_only(true),
            ..Default::default()
        },
        VCase {
            name: "no chunks",
            ok: true,
            change: lits_only(false),
            ..Default::default()
        },
    ]
}

#[test]
fn simd_validation() {
    // Miri: 60 short operations, still in three chunks and literal blocks. It skips the cases
    // of 64 KiB or more, and the slowest valid blocks, whose decoding simd_reference and
    // simd_class_len_16 cover. Literal blocks of 80 leave the last one longer than the literals
    // after the operations, so "literal over-read" still fails in an operation.
    let (n_ops, max_ml, chunk, lit_block) = if cfg!(miri) {
        (60, 64, 24, 80)
    } else {
        (2000, 288, 600, 3000)
    };
    let miri_skip = [
        "literal block of 65536",
        "literal block of 65535",
        "65536 escapes",
        "coded 65536 escapes",
        "offset bits cut",
        "valid coded",
        "valid class",
        "chunk at limit",
    ];
    simd_tiers(|_| {
        let mut b = TestBuilder::default();
        for tc in validation_cases() {
            let name = tc.name;
            if cfg!(miri) && miri_skip.contains(&name) {
                continue;
            }
            let modes = if tc.modes.is_empty() {
                vec![STORED]
            } else {
                tc.modes.clone()
            };
            let (ops, lits) = test_ops(&mut Rng(4), n_ops, b"abcdefgh", max_ml, 4000);
            let mut tb = TestBlock {
                ops,
                lits,
                modes: modes.clone(),
                lit_modes: modes,
                lit_block,
                chunk,
                ..Default::default()
            };
            if let Some(ops) = &tc.ops {
                tb = TestBlock {
                    ops: ops.clone(),
                    lits: b"abc".repeat(3000),
                    chunk: tc.chunk,
                    ..Default::default()
                };
                if ops[0] == TestOp::default() {
                    // One chunk of ops.len() operations, with chunk limit 0.
                    tb.ops.fill(op(1, 4, 3));
                    tb.chunk = ops.len();
                }
            }
            let (mut e, mut want) = test_pad(&mut b, &mut tb);
            if let Some(f) = &tc.change {
                want = f(&mut b, &mut e, want);
            }
            let mut block = e.bytes();
            if let Some(f) = &tc.raw {
                block = f(block);
            }
            // Reads past the block fault.
            let mut g = Buffer::new(block.len()).unwrap();
            let block = &*{
                let s = g.slice(block.len(), true);
                s.copy_from_slice(&block);
                s
            };
            let rf = decode_simd_block(block);
            if tc.ok {
                test_decode(block, &want);
                assert_eq!(rf, Ok(want), "{name}: reference");
                continue;
            }
            assert!(rf.is_err(), "{name}: reference: no error");
            let r = crate::decode(&mut Vec::new(), block);
            assert!(
                matches!(
                    r,
                    Err(Error::Corrupt(_) | Error::Unsupported | Error::TooLarge)
                ),
                "{name}: {r:?}"
            );
            if let Ok((n, body)) = simd_header(&block[1..]) {
                let mut dst = vec![0xfe; n + 64];
                assert!(
                    decode_simd(&mut dst[..n], body).is_err(),
                    "{name}: no error"
                );
                assert!(dst[..n].iter().all(|&v| v == 0), "{name}: dst not cleared");
                assert!(
                    dst[n..].iter().all(|&v| v == 0xfe),
                    "{name}: wrote past dst"
                );
            }
        }
    });
}

#[test]
fn simd_validation_chunks() {
    simd_tiers(|_| {
        let mut b = TestBuilder::default();
        for n in [MAX_CHUNKS, MAX_CHUNKS + 1] {
            let mut ops = vec![op(1, MAX_LEN, 1)];
            ops.resize(n, op(0, MAX_LEN, 0));
            let (e, want) = b.build(&TestBlock {
                ops,
                lits: b"a".to_vec(),
                chunk: 1,
                ..Default::default()
            });
            let block = e.bytes();
            assert!(
                e.chunks.len() == n && block.len() - 1 + MIN_SAVING <= want.len(),
                "{} chunks, block of {} bytes for {}",
                e.chunks.len(),
                block.len(),
                want.len()
            );
            let rf = decode_simd_block(&block);
            if n <= MAX_CHUNKS {
                test_decode(&block, &want);
                assert!(rf.is_ok(), "{n} chunks: reference: {rf:?}");
                continue;
            }
            let r = crate::decode(&mut Vec::new(), &block);
            assert!(
                matches!(r, Err(Error::Corrupt(_))) && rf.is_err(),
                "{n} chunks: {r:?}, reference {rf:?}"
            );
        }
    });
}

#[test]
#[cfg_attr(miri, ignore = "blocks of 65535 operations, up to 8 MiB")]
fn simd_validation_65535_operations() {
    // Value 6 is 65535 operations, not 1024<<6: a stored token stream could hold 65536.
    // Escapes are stored: two per operation don't fit a pivco block.
    let coded = vec![CODE_LENS, CODE_LENS, CODE_LENS, STORED];
    let check = |name: &str, e: &TestEnc, want: &[u8], coded: bool| {
        assert_eq!(e.chunks[0].ops.len(), MAX_CHUNK, "{name}: first chunk");
        for k in 0..ESC {
            assert!(
                !coded || e.chunks[0].s[k].mode != STORED,
                "{name}: stream {k} is stored"
            );
        }
        let block = e.bytes();
        simd_tiers(|_| test_decode(&block, want));
        // The reference doesn't depend on the tier.
        let rf = decode_simd_block(&block);
        assert!(
            rf.as_ref().is_ok_and(|r| r == want),
            "{name}: reference: {:?}",
            rf.err()
        );
    };
    let rejected = |name: &str, block: &[u8]| {
        assert!(
            decode_simd_block(block).is_err(),
            "{name}: reference: no error"
        );
        let (n, body) = simd_header(&block[1..]).unwrap();
        simd_tiers(|tier| {
            let mut dst = vec![0xfe; n + 64];
            let r = decode_simd(&mut dst[..n], body);
            assert!(
                matches!(r, Err(Error::Corrupt(_))),
                "{name} {tier:?}: {r:?}"
            );
            assert!(
                dst[n..].iter().all(|&v| v == 0xfe),
                "{name} {tier:?}: wrote past dst"
            );
        });
    };
    let cases = [
        ("at limit", MAX_CHUNK, 1, 4, vec![], true),
        ("above limit", MAX_CHUNK + 1, 1, 4, vec![], false),
        ("at limit coded", MAX_CHUNK, 1, 4, coded.clone(), true),
        // Two escapes of 30 per operation: 8257410 bytes of matches and literals.
        (
            "most escapes",
            MAX_CHUNK,
            MAX_LEN + 31,
            MAX_LEN + 31,
            coded,
            true,
        ),
    ];
    let mut b = TestBuilder::default();
    for (name, n, ll, ml, modes, ok) in cases {
        let mut ops = vec![op(ll, ml, 0); n];
        ops[0].off = ml.max(3);
        let (mut e, want) = b.build(&TestBlock {
            ops,
            lits: b"abc".repeat(n * ll / 3 + 1),
            limit: 6,
            chunk: n,
            modes: modes.clone(),
            lit_modes: modes.clone(),
            ..Default::default()
        });
        if !ok {
            rejected(name, &e.bytes());
            continue;
        }
        check(name, &e, &want, !modes.is_empty());
        if ml <= MAX_LEN {
            continue;
        }
        // Escapes of 255 add up to far more than the decoded size.
        let esc = &mut e.chunks[0].s[ESC].data;
        assert_eq!(esc.len(), 2 * MAX_CHUNK, "{name}: escapes");
        esc.fill(255);
        rejected(&format!("{name}: escapes of 255"), &e.bytes());
    }
    // Coded streams with offset bits and escapes, in every loop.
    let modes = [
        vec![CODE_LENS],
        vec![CLASS_LENS],
        vec![CODE_LENS, CURRENT],
        vec![CLASS_LENS, CURRENT],
    ];
    for (i, modes) in modes.into_iter().enumerate() {
        let (ops, lits) = test_ops(
            &mut Rng(i as u64),
            MAX_CHUNK,
            b"aaabbcdef\x00\xff",
            40,
            1 << 22,
        );
        let mut tb = TestBlock {
            ops,
            lits,
            limit: 6,
            chunk: MAX_CHUNK,
            delta: i as u64 & 1,
            flat: i as u64 >> 1,
            lit_modes: modes.clone(),
            modes,
            ..Default::default()
        };
        let (e, want) = test_pad(&mut b, &mut tb);
        check(&format!("random {i}"), &e, &want, true);
    }
    simd_tiers(|tier| {
        let (out, n_lit, n_off, n_esc, ok) = tok_stats(tier, &vec![135; MAX_CHUNK]);
        assert!(
            ok && (out, n_lit, n_off, n_esc) == (33 * MAX_CHUNK, MAX_CHUNK, 0, MAX_CHUNK),
            "{tier:?}: tokens: {out} {n_lit} {n_off} {n_esc} {ok}"
        );
        let (sum, n, ok) = ll_stats(tier, &vec![33; MAX_CHUNK]);
        assert!(
            ok && (sum, n) == (33 * MAX_CHUNK, MAX_CHUNK),
            "{tier:?}: literal lengths: {sum} {n} {ok}"
        );
        let (sum, _, _) = ll_stats(tier, &vec![255; 2 * MAX_CHUNK]);
        assert_eq!(sum, 255 * 2 * MAX_CHUNK, "{tier:?}: escapes");
        let (n, ok) = off_stats(tier, &vec![175; MAX_CHUNK]);
        assert!(
            ok && n == 20 * MAX_CHUNK,
            "{tier:?}: offset symbols: {n} {ok}"
        );
    });
}

#[test]
#[cfg_attr(
    miri,
    ignore = "builds and decodes 8 MiB blocks; simd_block_api checks the size limit"
)]
fn simd_validation_max_size() {
    let max = crate::MAX_BLOCK_SIZE;
    let mut ops = vec![op(40, 288, 32)];
    ops.resize((max - 40) / 288, op(0, 288, 0));
    let mut b = TestBuilder::default();
    for size in [max, max + 1] {
        let (e, want) = b.build(&TestBlock {
            lits: b"a".repeat(size - 288 * ops.len()),
            ops: ops.clone(),
            ..Default::default()
        });
        let block = e.bytes();
        let rf = decode_simd_block(&block);
        // Building is the slow part, so only the decodes run per tier.
        simd_tiers(|_| {
            if size <= max {
                test_decode(&block, &want);
                assert!(rf.as_ref() == Ok(&want), "size {size}: reference");
                return;
            }
            let r = crate::decode(&mut Vec::new(), &block);
            assert!(
                r == Err(Error::TooLarge) && rf.is_err(),
                "size {size}: {r:?}, reference {rf:?}"
            );
        });
    }
}

#[test]
fn simd_validation_64_bytes() {
    for n in [100, 1000] {
        let mut src = Vec::new();
        put_uvarint(&mut src, 1 << 24 | n as u64);
        src.resize(src.len() + n - 64 - 4, 0);
        assert!(simd_header(&src).is_ok(), "size {n}");
        src.push(0);
        assert!(
            simd_header(&src).is_err(),
            "size {n}: 63 bytes smaller accepted"
        );
    }
    let mut src = Vec::new();
    put_uvarint(&mut src, 1 << 32 | 100);
    assert!(simd_header(&src).is_err(), "varint above 32 bits accepted");
}

/// A failed decode leaves nothing in dst, such as the literals of an earlier block that the
/// cached decoder keeps.
#[test]
fn simd_corrupt_clears() {
    // Miri: a twentieth of the operations, still in three chunks and literal blocks, and Z
    // bytes past their literals.
    let (zs, n_ops, chunk, lit_block) = if cfg!(miri) {
        (2000, 100, 40, 150)
    } else {
        (80000, 2000, 600, 3000)
    };
    simd_tiers(|_| {
        let mut b = TestBuilder::default();
        let (e, _) = b.build(&TestBlock {
            lits: b"Z".repeat(zs),
            lit_modes: vec![CODE_LENS],
            ..Default::default()
        });
        crate::decode(&mut Vec::new(), &e.bytes()).unwrap();
        // The "literal over-read" block of simd_validation.
        let (ops, lits) = test_ops(&mut Rng(4), n_ops, b"abcdefgh", 288, 4000);
        let mut tb = TestBlock {
            ops,
            lits,
            lit_block,
            chunk,
            ..Default::default()
        };
        let (mut e, want) = test_pad(&mut b, &mut tb);
        e.lits.pop();
        let block = e.bytes();
        let (n, body) = simd_header(&block[1..]).unwrap();
        assert_eq!(n, want.len());
        let mut dst = vec![0; n];
        let r = decode_simd(&mut dst, body);
        let z = dst.iter().filter(|&&v| v == b'Z').count();
        assert!(
            r.is_err() && dst.iter().all(|&v| v == 0),
            "error {r:?}, {z} Z bytes in dst"
        );
    });
}

/// Returns the lengths of a random complete code over all or some of n values. Lengths are at
/// most 6, so most class tables stay within 11 bits per value.
fn random_code(rng: &mut Rng, n: usize) -> Vec<u8> {
    let (mut depth, mut m) = (vec![0u8], n);
    if rng.intn(2) == 0 {
        m = 2 + rng.intn(n - 2);
    }
    while depth.len() < m {
        let i = rng.intn(depth.len());
        if depth[i] < 6 {
            depth[i] += 1;
            depth.push(depth[i]);
        }
    }
    let mut l = vec![0; n];
    for (i, v) in rng.perm(n).into_iter().take(depth.len()).enumerate() {
        l[v] = depth[i];
    }
    l
}

fn nib(tab: &[u8], i: usize) -> u8 {
    (tab[i / 2] >> (4 * (i & 1))) & 15
}

fn set_nib(tab: &mut [u8], i: usize, v: u8) {
    let s = 4 * (i & 1);
    tab[i / 2] = (tab[i / 2] & !(15 << s)) | (v << s);
}

/// Makes one random change to the parts of a block (Go `testSIMDReference`'s change).
fn change_block(
    rng: &mut Rng,
    b: &mut TestBuilder,
    sd: &mut Tables,
    e: &mut TestEnc,
    lit_block: usize,
) {
    let op = rng.intn(13);
    // Table changes need a table.
    let has = |s: &TestStream| op != 1 && op != 2 || !s.table.is_empty();
    let nlits = e.lits.iter().filter(|s| has(s)).count();
    let nall = nlits
        + e.chunks
            .iter()
            .flat_map(|c| &c.s)
            .filter(|s| has(s))
            .count();
    if nall == 0 {
        return;
    }
    let pick = rng.intn(nall);
    let s = if pick < nlits {
        e.lits.iter_mut().filter(|s| has(s)).nth(pick)
    } else {
        e.chunks
            .iter_mut()
            .flat_map(|c| &mut c.s)
            .filter(|s| has(s))
            .nth(pick - nlits)
    }
    .unwrap();
    match op {
        0 => s.mode = rng.intn(8),
        1 => {
            let n = 2 * s.table.len();
            let (mut x, y) = (rng.intn(n), rng.intn(n));
            if rng.intn(2) == 0 {
                // Value 0 is outside the range of some streams.
                x = 0;
            }
            let (vx, vy) = (nib(&s.table, x), nib(&s.table, y));
            set_nib(&mut s.table, x, vy);
            set_nib(&mut s.table, y, vx);
        }
        2 => {
            // The last nibble may be a pad nibble.
            let mut used = vec![2 * s.table.len() - 1];
            used.extend((0..2 * s.table.len()).filter(|&x| nib(&s.table, x) != 0));
            let x = used[rng.intn(used.len())];
            set_nib(&mut s.table, x, rng.intn(16) as u8);
        }
        3 => _ = s.data.pop(),
        4 => s.data.push(rng.intn(256) as u8),
        5 => {
            if !s.data.is_empty() {
                let i = rng.intn(s.data.len());
                s.data[i] = rng.intn(256) as u8;
            }
        }
        6 => e.size = [e.size - 1, e.size + 1, crate::MAX_BLOCK_SIZE + 1][rng.intn(3)],
        7 => e.typ = 1 + rng.intn(255) as u64,
        8 => e.flags ^= 1 << rng.intn(8),
        9 => {
            let i = rng.intn(e.chunks.len());
            e.chunks[i].flags ^= 1 << rng.intn(8);
        }
        10 => _ = e.lits.remove(rng.intn(e.lits.len())),
        11 => {
            let tok = &mut e.chunks[0].s[TOK];
            if tok.mode == STORED {
                tok.data[0] &= !2;
            }
        }
        _ => {
            // A random class table, coded with the decoder's table, so the reference checks its
            // mapping.
            let k = rng.intn(LITS + 1);
            let (mut mode, mut lay) = (CLASS_LENS, k);
            let (st, vals) = if k == LITS {
                let i = rng.intn(e.lits.len());
                if rng.intn(2) == 0 {
                    (mode, lay) = (DELTA_LENS, DELTA);
                }
                let end = (i * lit_block + lit_block).min(e.stored.len());
                (&mut e.lits[i], e.stored[i * lit_block..end].to_vec())
            } else {
                let c = rng.intn(e.chunks.len());
                let c = &mut e.chunks[c];
                (&mut c.s[k], c.vals[k].clone())
            };
            let mut cl = random_code(rng, LAYOUTS[lay].n);
            if k == TOK {
                cl.extend(random_code(rng, 4));
                cl.extend(random_code(rng, 4));
            }
            (st.mode, st.table) = (mode, test_nibbles(&cl));
            if sd.set_table(k, mode, &st.table).is_some()
                && !vals.is_empty()
                && let Ok(d) = b.code(k, &vals, &sd.tab[k])
            {
                st.data = d;
            }
        }
    }
}

/// Compares the decoder with the reference decoder on built blocks, and on changed ones.
#[test]
fn simd_reference() {
    simd_tiers(|_| {
        let mut rng = Rng(6);
        let mut b = TestBuilder::default();
        let mut sd = Tables::default();
        let modes = [STORED, CODE_LENS, CURRENT, CLASS_LENS, CURRENT];
        let lit_modes = [STORED, CODE_LENS, CURRENT, CLASS_LENS, DELTA_LENS];
        // Operations at the edge of validity, not all valid.
        let edge = [
            vec![op(100, 33, 20)],
            vec![op(100, 33, 32)],
            vec![op(20, 33, 40)],
            vec![op(10, 20, 1000)],
            vec![op(0, 20, 1), op(10, 20, 1)],
        ];
        // Literals of every value, so the literal class layouts matter.
        let all: Vec<u8> = (0..=255).collect();
        // Miri: the edge blocks and one of each kind after them, each small.
        let (blocks, changes, max_ops) = if cfg!(miri) {
            (10, 3, 40)
        } else {
            (100, 60, 1500)
        };
        let mut valid = 0;
        for i in 0..blocks {
            let alphabet: &[u8] = if i % 2 == 0 {
                &all
            } else {
                b"aaabbcdef\x00\xff"
            };
            let n = 1 + rng.intn(max_ops);
            let max_off = 1 << rng.intn(25);
            let (mut ops, mut lits) = test_ops(&mut rng, n, alphabet, 288, max_off);
            if i < edge.len() {
                (ops, lits) = (edge[i].clone(), b"abc".repeat(100));
            } else if i % 5 == 3 {
                // Single-value streams.
                ops = vec![op(1, 4, 3); 1 + rng.intn(500)];
                lits = b"a".repeat(ops.len());
            }
            let limit = rng.intn(7);
            let mut tb = TestBlock {
                ops,
                lits,
                limit,
                chunk: 32 + rng.intn(300),
                lit_block: 64 + rng.intn(2000),
                modes: modes[..1 + i % modes.len()].to_vec(),
                lit_modes: lit_modes[..1 + i % lit_modes.len()].to_vec(),
                delta: rng.uint64(),
                ..Default::default()
            };
            if i % 4 == 1 {
                tb.chunk = (1024 << limit).min(MAX_CHUNK);
            }
            let orig = test_pad(&mut b, &mut tb).0.bytes();
            for j in 0..changes {
                let (mut e, want) = b.build(&tb);
                if j % 2 == 1 {
                    change_block(&mut rng, &mut b, &mut sd, &mut e, tb.lit_block);
                }
                let mut block = e.bytes();
                if j > 0 && j % 2 == 0 {
                    let (_, n) = get_uvarint(&block[1..]).unwrap();
                    let p = 1 + n + rng.intn(block.len() - 1 - n);
                    match rng.intn(4) {
                        0 => block[p] ^= 1 << rng.intn(8),
                        1 => block[p] = rng.intn(256) as u8,
                        2 => _ = block.remove(p),
                        _ => block.truncate(p),
                    }
                }
                let mut got = Vec::new();
                let r = crate::decode(&mut got, &block);
                let rf = decode_simd_block(&block);
                let same = rf.as_ref().is_ok_and(|rf| *rf == got);
                assert!(
                    r.is_ok() == rf.is_ok()
                        && (r.is_err() || same)
                        && (j > 0 || i < edge.len() || rf.as_ref() == Ok(&want)),
                    "block {i} change {j}: decode: {r:?}, reference: {:?}, same output: {same}",
                    rf.err()
                );
                if r.is_ok() && block != orig {
                    valid += 1;
                }
            }
        }
        eprintln!("{valid} changed blocks decoded");
    });
}

/// The group lengths and both flag codes of a token class table must be complete codes, even
/// when the token code they give is (A.4).
#[test]
fn simd_token_components() {
    let (mut ops, mut short, mut long) = (vec![op(1, 0, 0)], vec![op(1, 4, 0)], vec![op(1, 32, 0)]);
    // Miri: fewer operations, still every short match length.
    for i in 0..if cfg!(miri) { 60 } else { 200 } {
        ops.push(op(0, 32, 0));
        short.push(op(0, 4 + i % 28, 0));
        long.push(op(0, 32, 0));
    }
    let g = |pairs: &[(usize, u8)]| {
        let mut g = [0u8; 22];
        for &(i, v) in pairs {
            g[i] = v;
        }
        g
    };
    let mut b = TestBuilder::default();
    for (ok, g, ops) in [
        // Both give the same token lengths.
        (
            true,
            g(&[
                (0, 1),
                (12, 1),
                (14, 1),
                (15, 2),
                (16, 3),
                (17, 3),
                (18, 2),
                (19, 2),
                (20, 2),
                (21, 2),
            ]),
            &ops,
        ),
        (
            false,
            g(&[
                (0, 1),
                (12, 2),
                (14, 1),
                (15, 2),
                (16, 3),
                (17, 3),
                (18, 1),
                (19, 1),
                (20, 1),
                (21, 1),
            ]),
            &ops,
        ),
        // Only one flag code is incomplete, and no group uses it.
        (
            false,
            [
                3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0,
            ],
            &short,
        ),
        (
            false,
            g(&[
                (12, 1),
                (13, 1),
                (14, 1),
                (18, 2),
                (19, 2),
                (20, 2),
                (21, 2),
            ]),
            &long,
        ),
    ] {
        let (mut e, want) = b.build(&TestBlock {
            ops: ops.clone(),
            lits: b"abc".to_vec(),
            ..Default::default()
        });
        let mut lens = [0u8; 256];
        let grp = &LAYOUTS[TOK];
        for (v, l) in lens[..136].iter_mut().enumerate() {
            let f = if v >> 2 >= 32 {
                g[18 + (v & 3)]
            } else {
                g[14 + (v & 3)]
            };
            let gl = g[grp.class[v >> 2] as usize];
            if gl != 0 && f != 0 {
                *l = gl + grp.inlen[v >> 2] + f;
            }
        }
        let mut tab = pivco::Table::default();
        tab.set_lengths(&lens).unwrap();
        let data = b.code(TOK, &e.chunks[0].vals[TOK], &tab).unwrap();
        e.chunks[0].s[TOK] = TestStream {
            mode: CLASS_LENS,
            table: test_nibbles(&g),
            data,
        };
        let block = e.bytes();
        let mut got = Vec::new();
        let r = crate::decode(&mut got, &block);
        let rf = decode_simd_block(&block);
        assert!(
            r.is_ok() == ok && rf.is_ok() == ok,
            "complete components {ok}: decode {r:?}, reference {rf:?}"
        );
        if ok {
            assert!(got == want && rf == Ok(want), "output mismatch");
        }
    }
}

/// A literal class table where class 10 (0x80-0xbf, in-class length 6) has class length 10.
/// pivco reads lengths as 4 bits, so 16 would be an unused value of the complete code of
/// classes 5 and 6 (the lowercase letters), and the block would decode.
#[test]
fn simd_class_len_16() {
    simd_tiers(|_| {
        let lay = &LAYOUTS[LITS];
        let mut cl = vec![0u8; lay.n];
        (cl[5], cl[6]) = (1, 1);
        let mut lens = [0u8; 256];
        for (v, &c) in lay.class.iter().enumerate() {
            if c == 5 || c == 6 {
                lens[v] = 1 + lay.inlen[v];
            }
        }
        let mut tab = pivco::Table::default();
        tab.set_lengths(&lens).unwrap();
        let lits = b"aaab".repeat(400);
        let data = TestBuilder::default().code(LITS, &lits, &tab).unwrap();
        for c10 in [0, 10] {
            cl[10] = c10;
            let e = TestEnc {
                typ: BLOCK_TYPE,
                size: lits.len(),
                no_lz: true,
                lits: vec![TestStream {
                    mode: CLASS_LENS,
                    table: test_nibbles(&cl),
                    data: data.clone(),
                }],
                ..Default::default()
            };
            let block = e.bytes();
            let rf = decode_simd_block(&block);
            if c10 == 0 {
                test_decode(&block, &lits);
                assert!(rf.is_ok(), "reference: {rf:?}");
                continue;
            }
            let r = crate::decode(&mut Vec::new(), &block);
            assert!(
                matches!(r, Err(Error::Corrupt(_))) && rf.is_err(),
                "length 16: {r:?}, reference {rf:?}"
            );
        }
    });
}

/// Decodes block with a new decoder. Its literal buffer holds bytes of an earlier block, and
/// ends right after the size+PAD bytes that decode needs (Go `simdTestLitDecode`).
fn test_lit_decode(tier: Tier, block: &[u8]) -> Result<Vec<u8>, Error> {
    let (size, body) = simd_header(&block[1..])?;
    let mut s = Decoder::new();
    s.tier = tier;
    s.lits.buf = (0..size + PAD).map(|i| i as u8 | 0x80).collect();
    let mut dst = vec![0xfe; size + 64];
    let ok = s.decode(&mut dst[..size], body, false);
    assert!(dst[size..].iter().all(|&v| v == 0xfe), "wrote past dst");
    ok.map(|()| dst[..size].to_vec()).ok_or(Error::Corrupt(""))
}

/// Blocks whose chunks take their literals from literal blocks decoded just before them:
/// chunks that need many literal blocks, literal blocks that serve many chunks, stored and coded
/// literal blocks, delta chunks, and literals after the last operation. Changed blocks, among
/// them blocks whose literals run out or are more than the block holds, must be accepted and
/// decoded as the reference does.
#[test]
#[cfg_attr(miri, ignore = "blocks of thousands of operations: over 25 minutes")]
fn simd_lit_lazy() {
    simd_tiers(|tier| {
        let mut rng = Rng(11);
        let mut b = TestBuilder::default();
        let lit_modes = [STORED, CODE_LENS, CURRENT, CLASS_LENS, DELTA_LENS, STORED];
        let mut cases: Vec<(String, TestEnc, Vec<u8>)> = Vec::new();
        let mut add = |b: &mut TestBuilder, name: String, mut tb: TestBlock| {
            let (e, want) = test_pad(b, &mut tb);
            cases.push((name, e, want));
        };
        for i in 0..6 {
            let n = 3000 + rng.intn(3000);
            let max_off = 1 << rng.intn(20);
            let (ops, lits) = test_ops(&mut rng, n, b"abcdefgh\x00", 288, max_off);
            let delta = (i & 1) as u64 * rng.uint64();
            let tb = TestBlock {
                ops: ops.clone(),
                lits: lits.clone(),
                limit: 2,
                chunk: 500 + rng.intn(3000),
                lit_block: 1 + rng.intn(200),
                modes: vec![CODE_LENS, CURRENT],
                lit_modes: lit_modes[i..].to_vec(),
                delta,
                ..Default::default()
            };
            add(&mut b, format!("many literal blocks per chunk {i}"), tb);
            let tb = TestBlock {
                chunk: ops.len() / 90 + rng.intn(20),
                ops,
                lits,
                lit_block: 5000 + rng.intn(60000),
                lit_modes: lit_modes[i..].to_vec(),
                delta,
                ..Default::default()
            };
            add(&mut b, format!("many chunks per literal block {i}"), tb);
        }
        // Few literals in the operations, many after them.
        let tail: Vec<u8> = (0..30000).map(|_| b"abcd"[rng.intn(4)]).collect();
        let tb = TestBlock {
            ops: vec![op(40, 32, 7), op(2, 30, 0)],
            lits: tail.clone(),
            chunk: 1,
            lit_block: 2000,
            lit_modes: vec![STORED, CODE_LENS, CURRENT],
            delta: 1,
            ..Default::default()
        };
        add(&mut b, "literals after the operations".into(), tb);
        let mut e = TestEnc {
            typ: BLOCK_TYPE,
            size: tail.len(),
            no_lz: true,
            ..Default::default()
        };
        for (i, v) in tail.chunks(7000).enumerate() {
            e.lits.push(b.stream(LITS, [CODE_LENS, CURRENT][i % 2], v));
        }
        cases.push(("no operations".into(), e, tail));

        let extra = b"yz".repeat(500);
        let changes = 60;
        for (name, te, want0) in &cases {
            let block = te.bytes();
            assert_eq!(
                decode_simd_block(&block).as_ref(),
                Ok(want0),
                "{name}: reference"
            );
            test_decode(&block, want0);
            assert_eq!(test_lit_decode(tier, &block).as_ref(), Ok(want0), "{name}");
            for j in 0..changes {
                let mut e = te.clone();
                let mut want = None;
                match j % 6 {
                    // The literals run out, mostly in the operations.
                    0 => e.lits.truncate(rng.intn(e.lits.len())),
                    1 => _ = e.lits.remove(rng.intn(e.lits.len())),
                    // One literal more than the block holds.
                    2 => e.lits.push(TestStream {
                        data: b"x".to_vec(),
                        ..Default::default()
                    }),
                    3 => {
                        e.size += extra.len();
                        e.lits.push(b.stream(LITS, CODE_LENS, &extra));
                        want = Some([&want0[..], &extra].concat());
                    }
                    4 => {
                        let i = rng.intn(e.lits.len());
                        let s = &mut e.lits[i];
                        let n = s.data.len().saturating_sub(1 + rng.intn(3));
                        s.data.truncate(n);
                    }
                    _ => {}
                }
                let mut block = e.bytes();
                if j % 6 == 5 {
                    let (_, h) = get_uvarint(&block[1..]).unwrap();
                    let p = 1 + h + rng.intn(block.len() - 1 - h);
                    block[p] ^= 1 << rng.intn(8);
                }
                let got = test_lit_decode(tier, &block);
                let rf = decode_simd_block(&block);
                assert!(
                    got.is_ok() == rf.is_ok()
                        && (got.is_err() || got.as_ref().ok() == rf.as_ref().ok())
                        && (want.is_none() || got.as_ref().ok() == want.as_ref()),
                    "{name} change {j}: {:?}, reference {:?}",
                    got.err(),
                    rf.err()
                );
            }
        }
    });
}

/// fill decodes literal blocks only until the literals a chunk needs are decoded, and zeros
/// follow them.
#[test]
fn simd_lit_fill() {
    let mut rng = Rng(12);
    let mut b = TestBuilder::default();
    // Miri: five literal blocks, in every mode.
    let n = if cfg!(miri) { 12000 } else { 50000 };
    let lits: Vec<u8> = (0..n).map(|_| b"abcdefgh"[rng.intn(8)]).collect();
    let mut e = TestEnc {
        typ: BLOCK_TYPE,
        size: lits.len(),
        no_lz: true,
        ..Default::default()
    };
    let mut ends: Vec<usize> = Vec::new();
    let mut i = 0;
    while i < lits.len() {
        let v = &lits[i..(i + 1 + rng.intn(4000)).min(lits.len())];
        e.lits
            .push(b.stream(LITS, [STORED, CODE_LENS, CURRENT][ends.len() % 3], v));
        i += v.len();
        ends.push(i);
    }
    let raw = e.bytes();
    let (_, h) = get_uvarint(&raw[1..]).unwrap();
    let body = &raw[2 + h..];
    let (_, n) = rvarint(body).unwrap();
    let mut s = Decoder::new();
    s.lits.buf = vec![0xa5; lits.len() + PAD];
    let mut r = s
        .literals(&body[..body.len() - n], lits.len())
        .expect("literal section");
    let mut l = 0;
    while !r.recs.is_empty() {
        let need = rng.intn(6000);
        let mut want = s.lits.n;
        for &end in &ends {
            if want - l >= need {
                break;
            }
            want = want.max(end);
        }
        assert!(
            s.lits.fill(&mut s.t, &mut r, l, need, lits.len()).is_some(),
            "fill failed"
        );
        let got = &s.lits.buf;
        assert!(
            s.lits.n == want && got[..want] == lits[..want] && got[want..want + PAD] == [0; PAD],
            "{need} literals from {l}: decoded {}, want {want}",
            s.lits.n
        );
        l += rng.intn(need.min(want - l) + 1);
    }
    assert_eq!(r.pos, r.end, "literal data left");
}

/// Reserved block types, and SIMD headers out of range, through the block API (Go
/// `isMinLZ`).
#[test]
fn simd_block_api() {
    let mut b = TestBuilder::default();
    let mut tb = TestBlock {
        ops: vec![op(10, 200, 3); 20],
        lits: b"0123456789".repeat(20),
        ..Default::default()
    };
    let (mut e, want) = test_pad(&mut b, &mut tb);
    assert_eq!(crate::is_minlz(&e.bytes()), Ok(Some(want.len())));
    for (typ, err) in [(2, Error::Unsupported), (255, Error::Unsupported)] {
        e.typ = typ;
        let block = e.bytes();
        assert_eq!(crate::decoded_len(&block), Err(err), "type {typ}");
        assert_eq!(crate::is_minlz(&block), Err(err), "type {typ}");
        let mut dst = b"keep".to_vec();
        assert_eq!(
            crate::append_decoded(&mut dst, &block),
            Err(err),
            "type {typ}"
        );
        assert_eq!(dst, b"keep", "type {typ}: dst changed");
    }
    // A size above MAX_BLOCK_SIZE in a SIMD header is too large; a header above 32 bits is corrupt.
    let block = e.bytes();
    let (_, h) = get_uvarint(&block[1..]).unwrap();
    let body = &block[1 + h..];
    let big = with_header(crate::MAX_BLOCK_SIZE as u64 + 1, body);
    assert_eq!(crate::decoded_len(&big), Err(Error::TooLarge));
    let huge = with_header(1 << 32, body);
    assert!(matches!(crate::decoded_len(&huge), Err(Error::Corrupt(_))));
    // A failed SIMD decode leaves dst as it was.
    e.typ = BLOCK_TYPE;
    e.lits.pop();
    let mut dst = b"keep".to_vec();
    assert!(crate::append_decoded(&mut dst, &e.bytes()).is_err());
    assert_eq!(dst, b"keep");
}
