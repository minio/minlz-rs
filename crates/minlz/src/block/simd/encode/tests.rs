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

//! Ports of the SIMD encoder tests of Go's `encode_test.go`, of the Go parser part of
//! TestSIMDParseAsmBounds (`encode-simd_asm_test.go`), and of FuzzEncodingBlocks' seed run
//! (`fuzz_test.go`). Blocks round trip through `decode` and the strict reference decoder. Go's
//! asm parser tiers (simdEncTiers) have no Rust counterpart.

use super::super::parse_better::better_max_off;
use super::super::{rvarint, simd_header};
use super::*;
use crate::block::format::get_uvarint;
use crate::block::{
    Level, append_encoded, decode, decoded_len, encode, max_encoded_len, try_encode,
};
use crate::reference::{SimdWalker, decode_simd_block_strict, walk_simd_block};

/// Go's simdLevels.
const LEVELS: [i32; 7] = [
    LEVEL_SUPER_FAST,
    LEVEL_FASTEST,
    LEVEL_BALANCED,
    LEVEL_SMALLEST,
    LEVEL_SUPER_FAST | LEVEL_DELTA,
    LEVEL_FASTEST | LEVEL_DELTA,
    LEVEL_BALANCED | LEVEL_DELTA,
];

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn intn(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn exp_f64(&mut self) -> f64 {
        -(1.0 - self.f64()).ln()
    }

    /// Returns n random bytes (Go's `genEncRandom`).
    fn random(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }

    /// Returns size bytes of random words (Go's `genEncText`).
    fn text(&mut self, size: usize) -> Vec<u8> {
        const WORDS: [&str; 15] = [
            "lorem",
            "ipsum",
            "dolor",
            "sit",
            "amet",
            "consectetur",
            "adipiscing",
            "elit",
            "sed",
            "do",
            "eiusmod",
            "tempor",
            "incididunt",
            "ut",
            "labore",
        ];
        let mut out = Vec::with_capacity(size + 16);
        while out.len() < size {
            out.extend_from_slice(WORDS[self.intn(WORDS.len())].as_bytes());
            out.push(b' ');
        }
        out.truncate(size);
        out
    }

    /// Alternates incompressible and compressible runs (Go's `genEncMixed`).
    fn mixed(&mut self, size: usize) -> Vec<u8> {
        const RUN: usize = 4 << 10;
        let pattern = b"the quick brown fox jumps over the lazy dog. ";
        let mut out = Vec::new();
        while out.len() < size {
            out.extend_from_slice(&self.random(RUN));
            out.extend(pattern.iter().cycle().take(RUN));
        }
        out.truncate(size);
        out
    }

    /// Returns n bytes of runs of length bytes of one value, which differs from the run before
    /// (Go's `simdRuns`).
    fn runs(&mut self, length: usize, n: usize) -> Vec<u8> {
        let mut b: Vec<u8> = Vec::with_capacity(n + length);
        while b.len() < n {
            let mut c = self.intn(255) as u8;
            if let Some(&l) = b.last()
                && c >= l
            {
                c += 1;
            }
            b.extend(std::iter::repeat_n(c, length));
        }
        b.truncate(n);
        b
    }
}

fn testdata(name: &str) -> Vec<u8> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata");
    std::fs::read(p.join(name)).unwrap()
}

fn twain() -> Vec<u8> {
    testdata("Mark.Twain-Tom.Sawyer.txt")
}

/// The files of testdata/bench (or $MINLZ_TESTDATA/bench), when present.
fn bench_files() -> Vec<(String, Vec<u8>)> {
    let dir = std::env::var_os("MINLZ_TESTDATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata"))
        .join("bench");
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    rd.flatten()
        .map(|e| (e.file_name().into_string().unwrap(), e.path()))
        .filter(|(n, _)| !n.ends_with(".partial") && !n.ends_with(".jpeg"))
        .map(|(n, p)| (n, std::fs::read(p).unwrap()))
        .collect()
}

fn level(l: i32) -> Level {
    Level::try_from(l).unwrap()
}

fn encode_level(src: &[u8], l: i32) -> Vec<u8> {
    let mut dst = Vec::new();
    encode(&mut dst, src, level(l)).unwrap();
    dst
}

fn is_simd(b: &[u8]) -> bool {
    b.len() > 1 && get_uvarint(&b[1..]).is_some_and(|(v, _)| v >> 24 == BLOCK_TYPE)
}

/// Decodes a block of encode with decode, and a SIMD block also with the strict reference
/// decoder, which must agree.
fn decode_block(b: &[u8]) -> Result<Vec<u8>, String> {
    let mut d = Vec::new();
    decode(&mut d, b).map_err(|e| format!("decode: {e}"))?;
    if is_simd(b) && decode_simd_block_strict(b)? != d {
        return Err("decode and the reference decoder differ".into());
    }
    Ok(d)
}

/// encode with the test knobs k.
fn encode_knobs(src: &[u8], l: i32, k: Knobs) -> Vec<u8> {
    let mut dst = vec![0; max_encoded_len(src.len()).unwrap()];
    let mut e = Encoder::new(&mut dst[1..]);
    e.knobs = k;
    let n = if src.len() > MIN_SAVING {
        e.parse(src, l);
        e.finish()
    } else {
        0
    };
    if n == 0 {
        return [&[0, 0], src].concat();
    }
    dst.truncate(1 + n);
    dst
}

impl<'a> Encoder<'a> {
    fn new(out: &'a mut [u8]) -> Self {
        Self::with_scratch(out, Scratch::new())
    }
}

/// Calls f with the flags of each chunk of block b.
fn chunk_flags(b: &[u8], f: impl FnMut(u8)) {
    struct W<F>(F);
    impl<F: FnMut(u8)> SimdWalker for W<F> {
        fn chunk(&mut self, flags: u8, _: usize) {
            (self.0)(flags);
        }
    }
    walk_simd_block(b, &mut W(f)).unwrap();
}

/// Adds the entry modes of the streams of SIMD block b to modes, with the literals as LITS, and how
/// many chunks have each chunk flag to flags (Go's `simdTestStats`).
fn test_stats(b: &[u8], modes: &mut [[usize; 8]; LITS + 1], flags: &mut [usize; 2]) {
    let (_, body) = simd_header(&b[1..]).unwrap();
    let body = &body[1..];
    let mut entry = |tab: &[u8], k: usize| -> usize {
        let (e, n) = get_uvarint(tab).unwrap();
        let mode = (e & 7) as usize;
        modes[k][mode] += 1;
        n + match mode {
            CODE_LENS => CODE_BYTES[k],
            DELTA_LENS => LAYOUTS[DELTA].n.div_ceil(2),
            CLASS_LENS if k == TOK => 11,
            CLASS_LENS => LAYOUTS[k].n.div_ceil(2),
            _ => 0,
        }
    };
    let table = |sec: &[u8]| -> Vec<u8> {
        let (size, n) = rvarint(sec).unwrap();
        sec[sec.len() - n - size as usize..sec.len() - n].to_vec()
    };
    let (lz, n) = rvarint(body).unwrap();
    let lz = lz as usize;
    let tab = table(&body[lz..body.len() - n]);
    let mut t = &tab[..];
    while !t.is_empty() {
        t = &t[entry(t, LITS)..];
    }
    if lz == 0 {
        return;
    }
    let tab = table(&body[..lz]);
    let mut t = &tab[..];
    while !t.is_empty() {
        for (i, f) in flags.iter_mut().enumerate() {
            *f += (t[0] >> i) as usize & 1;
        }
        t = &t[1..];
        for k in 0..LITS {
            t = &t[entry(t, k)..];
        }
        t = &t[get_uvarint(t).unwrap().1..];
    }
}

/// Every level round-trips short inputs, from just over MIN_SAVING bytes. The other encoder tests
/// skip Miri, so this is what checks the parsers' unchecked loads there.
#[test]
fn simd_encode_short() {
    let mut src = [
        Rng(5).text(2 << 10),
        Rng(6).runs(9, 600),
        Rng(7).random(300),
    ]
    .concat();
    src.extend_from_within(100..900);
    for n in [MIN_SAVING + 1, MIN_SAVING + 9, 700, src.len()] {
        for l in LEVELS {
            let b = encode_level(&src[..n], l);
            assert!(
                decode_block(&b).unwrap() == src[..n],
                "level {l}, {n} bytes"
            );
        }
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "Miri adds error to log2, so exact powers of two round up"
)]
fn tables_match_go_formulas() {
    for (k, &n) in CODE_BYTES.iter().enumerate() {
        assert_eq!(
            MAX_ENTROPY[k],
            (256.0 * ((2 * n) as f64).log2()).ceil() as usize
        );
    }
    for (i, &v) in LOG2_FRAC.iter().enumerate() {
        assert_eq!(v, (256.0 * (1.0 + i as f64 / 256.0).log2()).round() as u8);
    }
    for (k, &n) in CLASS_BYTES.iter().enumerate() {
        let classes = if k == TOK { 22 } else { LAYOUTS[k].n };
        assert_eq!(n, classes.div_ceil(2));
    }
}

/// TestSIMDEncode
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode() {
    let mut rng = Rng(1);
    let n = 3 << 20;
    let mut period =
        |p: usize| -> Vec<u8> { rng.random(p).iter().copied().cycle().take(n).collect() };
    let (p3, p100) = (period(3), period(100));
    let mut rng = Rng(2);
    // records changes a few bytes of each record a little, which delta literals code well.
    let mut records = |n: usize| -> Vec<u8> {
        let mut rec = rng.random(64);
        let mut b = Vec::new();
        while b.len() < n {
            for _ in 0..4 {
                let i = rng.intn(rec.len());
                rec[i] = rec[i].wrapping_add(rng.intn(5) as u8).wrapping_sub(2);
            }
            b.extend_from_slice(&rec);
        }
        b
    };
    let (rec, rec16) = (records(n), records(16 << 10));
    let text = twain();
    let tom: Vec<u8> = text.iter().copied().cycle().take(n).collect();
    let mut rng = Rng(3);
    let mut inputs = vec![
        ("zeros-1k".to_string(), vec![0; 1024]),
        ("zeros".into(), vec![0; n]),
        ("text-2k".into(), text[..2048].to_vec()),
        ("tom-64k".into(), tom[..64 << 10].to_vec()),
        ("tom-64k+1".into(), tom[..(64 << 10) + 1].to_vec()),
        ("tom-sawyer".into(), tom.clone()),
        ("words".into(), rng.text(n)),
        ("mixed".into(), rng.mixed(n)),
        ("period-3".into(), p3),
        ("period-100".into(), p100),
        ("records".into(), rec),
        ("records-16k".into(), rec16),
    ];
    inputs.extend(bench_files());
    let mut modes = [[0; 8]; LITS + 1];
    let mut flags = [0; 2];
    for (name, src) in &inputs {
        for l in LEVELS {
            let enc = encode_level(src, l);
            assert!(
                decode_block(&enc).as_ref() == Ok(src),
                "{name} level {l}: roundtrip mismatch"
            );
            if !is_simd(&enc) {
                // Delta literals can make small inputs incompressible.
                assert!(l & LEVEL_DELTA != 0, "{name} level {l}: not a SIMD block");
                continue;
            }
            let mut t = Vec::new();
            assert!(
                try_encode(&mut t, src, level(l)).unwrap() && t == enc,
                "{name} level {l}: try_encode mismatch"
            );
            let mut app = vec![1];
            append_encoded(&mut app, src, level(l)).unwrap();
            assert!(
                app[1..] == enc[..],
                "{name} level {l}: append_encoded mismatch"
            );
            test_stats(&enc, &mut modes, &mut flags);
        }
    }
    for m in 0..=DELTA_LENS {
        assert!(
            modes.iter().any(|s| s[m] > 0),
            "mode {m} not used: {modes:?}"
        );
    }
    assert!(
        flags[0] > 0 && flags[1] > 0,
        "chunk flags not used: {flags:?}"
    );
}

/// Go's int levels convert to Level and back; all other ints are invalid.
#[test]
fn level_ints() {
    for l in -2..=300 {
        let valid = matches!(l, -1 | 1..=7 | 260..=262);
        match Level::try_from(l) {
            Ok(v) => assert!(valid && i32::from(v) == l, "level {l}"),
            Err(e) => assert!(!valid && e == crate::Error::InvalidLevel, "level {l}"),
        }
    }
}

/// Blocks don't depend on what the thread's pooled encoder and tables coded before: Go's pool
/// can hand out a new encoder at any time.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_reuse() {
    let mut rng = Rng(1);
    let inputs = [
        rng.text(100_000),
        rng.mixed(300_000),
        vec![0; 70_000],
        Rng(2).runs(3, 50_000),
    ];
    let cases: Vec<_> = inputs
        .iter()
        .flat_map(|src| LEVELS.map(|l| (src, l)))
        .collect();
    let fresh: Vec<Vec<u8>> = cases
        .iter()
        .map(|&(src, l)| {
            let src = src.clone();
            std::thread::spawn(move || encode_level(&src, l))
                .join()
                .unwrap()
        })
        .collect();
    for (&(src, l), want) in cases.iter().zip(&fresh).rev() {
        assert!(
            encode_level(src, l) == *want,
            "level {l}, {} bytes: differs from a fresh thread's block",
            src.len()
        );
    }
}

/// TestSIMDEncodeRoom checks that blocks don't depend on spare room in dst, and save the level's
/// minimum. Skewed random bytes code close to the size limits of the levels.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_room() {
    let mut rng = Rng(1);
    let mut big = vec![0; max_encoded_len(16 << 10).unwrap() + (64 << 10)];
    for size in [1 << 10, 4 << 10, 16 << 10] {
        let mut tight = vec![0; max_encoded_len(size).unwrap() - 1];
        let mut p = 0.01;
        while p < 0.08 {
            let src: Vec<u8> = (0..size)
                .map(|_| (rng.exp_f64() / p).min(255.0) as u8)
                .collect();
            for l in LEVELS {
                let n = encode_block_simd(&mut tight, &src, l);
                let m = encode_block_simd(&mut big, &src, l);
                assert!(
                    m == n && big[..m] == tight[..n],
                    "size {size} p {p:.3} level {l}: {m} bytes with spare room, {n} without"
                );
                let save = match l & !LEVEL_DELTA {
                    LEVEL_SUPER_FAST => MIN_SAVING.max(size / 8),
                    LEVEL_FASTEST | LEVEL_BALANCED => MIN_SAVING.max(size / 32),
                    _ => MIN_SAVING,
                };
                assert!(
                    n == 0 || size - n >= save,
                    "size {size} p {p:.3} level {l}: block of {n} bytes saves less than {save}"
                );
            }
            p += 0.002;
        }
    }
}

/// TestSIMDEncodeMaxChunks checks that a block fails when it would have more than MAX_CHUNKS
/// chunks. The parsers can't reach that, so this ends chunks early.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_max_chunks() {
    let src = vec![0u8; 8 << 10];
    for n in [MAX_CHUNKS, MAX_CHUNKS + 1] {
        let mut dst = vec![0; max_encoded_len(src.len()).unwrap()];
        let mut e = Encoder::new(&mut dst[1..]);
        e.reset(&src, 0, 0);
        e.emit_copy_lits(&src[..1], 1, MAX_LEN);
        e.end_chunk();
        for _ in 0..n - 1 {
            e.emit_repeat_lits(&[], MAX_LEN);
            e.end_chunk();
        }
        e.emit_literals(&src[e.pos..]);
        let size = e.finish();
        assert_eq!(
            size > 0,
            n <= MAX_CHUNKS,
            "{n} chunks: block of {size} bytes"
        );
        if size > 0 {
            assert_eq!(decode_simd_block_strict(&dst[..1 + size]), Ok(src.clone()));
        }
    }
}

/// TestSIMDEncodeOps decodes encoded blocks with the strict reference decoder, which also checks
/// the no-overlap flags.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_ops() {
    let mut rng = Rng(1);
    let five = rng.random(5);
    let inputs = [
        vec![0; 100000],
        twain(),
        rng.text(300000),
        rng.mixed(300000),
        five.repeat(50000),
    ];
    for (i, src) in inputs.iter().enumerate() {
        for l in LEVELS {
            let enc = encode_level(src, l);
            assert!(
                decode_block(&enc).as_ref() == Ok(src),
                "input {i} level {l}"
            );
        }
    }
}

/// TestSIMDEncodeNoOverlap checks that with the no_overlap knob, every chunk has the no-overlap
/// flag, and no match relies on it to raise its offset. It also checks the long_runs knob.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_no_overlap() {
    let mut rng = Rng(1);
    let mut inputs = vec![
        vec![0; 100000],
        twain(),
        rng.text(300000),
        rng.mixed(300000),
    ];
    // Runs of every period up to 40, and of several lengths, between random bytes.
    for p in 1..=40 {
        let mut b = Vec::new();
        for n in [p, p + 1, 2 * p, 3 * p + 1, 33, 64, 100, 289, 1000] {
            b.extend_from_slice(&rng.random(p));
            for _ in 0..n {
                b.push(b[b.len() - p]);
            }
            b.extend_from_slice(&rng.random(3));
        }
        inputs.push(b.repeat(4));
    }
    for (no_overlap, long_runs) in [(false, false), (true, false), (false, true), (true, true)] {
        let k = Knobs {
            no_overlap,
            long_runs,
            ..Knobs::default()
        };
        for (i, src) in inputs.iter().enumerate() {
            for l in LEVELS {
                let enc = encode_knobs(src, l, k);
                if !is_simd(&enc) {
                    continue;
                }
                let id = format!("knobs {no_overlap} {long_runs} input {i} level {l}");
                assert!(decode_simd_block_strict(&enc).as_ref() == Ok(src), "{id}");
                chunk_flags(&enc, |f| {
                    assert!(
                        f & 2 != 0 || !no_overlap,
                        "{id}: chunk without the no-overlap flag"
                    );
                });
            }
        }
    }
}

/// TestSIMDEncodeSplitChunkEnd checks that a split that ends a chunk before its last operation
/// sets no repeat offset after it: the next chunk would output fewer bytes than it has operations.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_split_chunk_end() {
    let src = vec![b'a'; CHUNK_OPS + 20];
    let mut out = vec![0; 2 * src.len() + (1 << 16)];
    let mut e = Encoder::new(&mut out[1..]);
    e.reset(&src, 0, 0);
    (e.out.limit, e.no_overlap) = (isize::MAX / 2, true);
    e.emit_copy_lits(&src[..1], 1, 1);
    for _ in 0..CHUNK_OPS - 3 {
        e.emit_repeat_lits(&[], 1);
    }
    // Operations of 1, 2 and 1 bytes, the chunk ending after the second.
    e.emit_repeat_lits(&[], 4);
    assert!(
        e.chunks == 1 && e.n_tok == 1,
        "{} chunks, {} operations",
        e.chunks,
        e.n_tok
    );
    e.emit_literals(&src[e.pos..]);
    e.end_chunk();
    let n = e.finish();
    assert_eq!(decode_simd_block_strict(&out[..1 + n]), Ok(src));
}

/// The state of an Encoder after a block is parsed, and the block (Go's `simdParseResult`).
struct ParseResult {
    chunks: usize,
    size: usize,
    lits: Vec<u8>,
    block: Vec<u8>,
}

/// Parses src with the SuperFast parser, encode_fast_block_64k if k64, as parse does but without
/// the size limit (Go's `simdParseRun`). The block starts with a zero byte.
fn parse_run(src: &[u8], k64: bool, no_overlap: bool, budget: bool) -> ParseResult {
    let mut out = vec![0; 1 + 2 * src.len() + (1 << 16)];
    let mut e = Encoder::new(&mut out[1..]);
    (e.split_lits, e.fast_code, e.no_overlap, e.budget) = (false, 1, no_overlap, budget);
    e.reset(src, 0, 0);
    e.out.limit = isize::MAX / 2;
    if k64 {
        e.encode_fast_block_64k(src);
    } else {
        e.encode_fast_block(src);
    }
    if e.n_tok > 0 {
        e.end_chunk();
    }
    let lits = [&e.s.buf[..e.n_lits], e.tail].concat();
    let chunks = e.chunks;
    let size = e.finish();
    out.truncate(1 + size);
    ParseResult {
        chunks,
        size,
        lits,
        block: out,
    }
}

/// TestSIMDEncodeSplitBudget checks that the split of LevelSuperFastSIMD keeps the parse of its
/// parsers, and that its budget keeps short runs within 5% of the size without the split, and
/// lets sparse runs split.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_split_budget() {
    let mut rng = Rng(1);
    // Gaps of zeros after a random byte.
    let mut sparse = |gap: usize, n: usize| -> Vec<u8> {
        let mut b = Vec::with_capacity(n + gap + 1);
        while b.len() < n {
            b.push(1 + rng.intn(255) as u8);
            b.resize(b.len() + gap, 0);
        }
        b.truncate(n);
        b
    };
    let sparse24 = sparse(24, 1 << 20);
    let mut rng = Rng(2);
    let (runs6, runs9, small) = (
        rng.runs(6, 1 << 20),
        rng.runs(9, 1 << 20),
        rng.runs(9, 16 << 10),
    );
    // Chunks that the short runs stop splitting, then one of long runs, whose split saves
    // operations.
    let runs_long = [rng.runs(9, 256 << 10), rng.runs(2000, 2 << 20)].concat();
    // name, src, the budget stops the split, every chunk is split, fewer operations than a chunk
    // holds so no split ends one, the last chunk is split.
    let inputs = [
        ("runs-6", runs6, true, false, false, false),
        ("runs-9", runs9, true, false, false, false),
        ("runs-9-small", small, true, false, true, false),
        ("sparse-24", sparse24, false, true, false, true),
        ("text", twain(), false, false, true, false),
        ("runs-long", runs_long, false, false, false, true),
    ];
    for (parser, k64, max) in [
        ("SuperFast", false, crate::MAX_BLOCK_SIZE),
        ("SuperFast64K", true, 64 << 10),
    ] {
        for (name, src, short, all_split, same_parse, last_split) in &inputs {
            let src = &src[..src.len().min(max)];
            let b = parse_run(src, k64, true, true);
            let g = parse_run(src, k64, false, false);
            let a = parse_run(src, k64, true, false);
            assert!(
                !same_parse || b.lits == g.lits && a.lits == g.lits,
                "{parser} {name}: the split changes the parse"
            );
            let (mut flagged, mut last) = (0, 0);
            chunk_flags(&b.block, |f| {
                flagged += (f >> 1) as usize & 1;
                last = f;
            });
            assert!(
                !short || b.size <= g.size + g.size / 20 && 3 * b.size <= 2 * a.size,
                "{parser} {name}: {} bytes, {} without the split, {} without the budget",
                b.size,
                g.size,
                a.size
            );
            assert!(
                !(*all_split && flagged != b.chunks
                    || *last_split && b.chunks > 1 && last & 2 == 0),
                "{parser} {name}: {flagged} of {} chunks split, the last {}",
                b.chunks,
                last & 2 != 0
            );
            assert!(
                decode_simd_block_strict(&b.block).as_deref() == Ok(src),
                "{parser} {name}: roundtrip mismatch"
            );
        }
    }
}

/// TestSIMDEncodeSplitBudgetChunk checks that the budget holds per chunk: a chunk that runs stop
/// splitting doesn't stop the next.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_split_budget_chunk() {
    let src = Rng(1).runs(9, 1 << 20);
    let mut out = vec![0; 2 * src.len() + (1 << 16)];
    let mut e = Encoder::new(&mut out);
    e.reset(&src, 0, 0);
    (e.out.limit, e.no_overlap, e.budget) = (isize::MAX / 2, true, true);
    let mut stopped = false;
    while e.pos + 9 <= src.len() && e.chunks == 0 {
        e.emit_copy_lits(&src[e.pos..e.pos + 1], 1, 8);
        stopped = stopped || !e.no_overlap;
    }
    assert!(
        stopped && e.chunks == 1,
        "stopped {stopped}, {} chunks",
        e.chunks
    );
    e.emit_copy_lits(&src[e.pos..e.pos + 1], 1, 8);
    assert!(!e.overlap, "the next chunk doesn't split");
    // With delta literals, LevelSuperFastSIMD doesn't split: the budget would leave raised
    // references in chunks without the flag.
    let enc = encode_level(&src, LEVEL_SUPER_FAST | LEVEL_DELTA);
    assert!(decode_block(&enc) == Ok(src), "delta: roundtrip mismatch");
}

/// TestSIMDEncodeShortPeriod checks that LevelSmallestSIMD continues long matches with an offset
/// below 32 at a multiple of the offset, in operations of up to 288 bytes instead of 32.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_short_period() {
    struct Ops(usize);
    impl SimdWalker for Ops {
        fn op(&mut self, _: usize, _: usize, _: usize, _: bool) {
            self.0 += 1;
        }
    }
    let mut rng = Rng(1);
    for r in [1, 2, 7, 21, 31] {
        // A short run, a byte that breaks the period, then a long run whose match is a repeat.
        let mut src = rng.random(64);
        while src.len() < 64 + 50 {
            src.push(src[src.len() - r]);
        }
        src.push(!src[src.len() - r]);
        while src.len() < 64 + 50 + 1 + 2000 {
            src.push(src[src.len() - r]);
        }
        src.extend(rng.random(64));
        let enc = encode_level(&src, LEVEL_SMALLEST);
        assert!(decode_block(&enc) == Ok(src), "period {r}");
        let mut ops = Ops(0);
        walk_simd_block(&enc, &mut ops).unwrap();
        assert!(
            ops.0 <= 20,
            "period {r}: {} operations, want at most 20",
            ops.0
        );
    }
}

/// TestSIMDEncodeChunkEdge
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_chunk_edge() {
    for delta in [0, -1, 256] {
        for before in CHUNK_OPS - 3..=CHUNK_OPS {
            let mut rng = Rng(before as u64);
            // Literals, offset and length of each operation.
            let mut ops: Vec<(usize, usize, usize)> =
                (0..before).map(|_| (1, 1 + rng.intn(8), 4)).collect();
            ops.extend([(300, 100, 40), (600, 40, 600), (0, 3, 70), (2, 3, 33)]);
            let mut src = Vec::new();
            for &(lits, off, ml) in &ops {
                for _ in 0..lits {
                    src.push(rng.next() as u8);
                }
                for _ in 0..ml {
                    src.push(src[src.len() - off.min(src.len())]);
                }
            }
            let mut block = vec![0; max_encoded_len(src.len()).unwrap()];
            let mut e = Encoder::new(&mut block[1..]);
            e.reset(&src, delta, 0);
            let mut p = 0;
            for &(lits, off, ml) in &ops {
                let l = &src[p..p + lits];
                p += lits + ml;
                if lits == 0 {
                    e.emit_copy(off, ml);
                } else if off == e.rep {
                    e.emit_repeat_lits(l, ml);
                } else {
                    e.emit_copy_lits(l, off, ml);
                }
            }
            if e.n_tok > 0 {
                e.end_chunk();
            }
            let n = e.finish();
            assert!(n > 0, "block not encoded");
            assert!(
                decode_simd_block_strict(&block[..n + 1]) == Ok(src),
                "delta {delta}, {before} operations before: roundtrip mismatch"
            );
        }
    }
}

/// Returns 1 MiB of literal runs and matches around the lengths where the parsers leave operations
/// to the emitters (Go's `simdTestParseInput`).
fn test_parse_input(rng: &mut Rng) -> Vec<u8> {
    const LENS: [usize; 12] = [0, 1, 4, 8, 31, 32, 33, 287, 288, 289, 290, 600];
    const OFFS: [usize; 10] = [1, 3, 8, 15, 16, 31, 32, 33, 1000, 100000];
    let mut src = vec![0];
    while src.len() < 1 << 20 {
        for _ in 0..LENS[rng.intn(LENS.len())] {
            src.push(rng.next() as u8);
        }
        let off = OFFS[rng.intn(OFFS.len())].min(src.len());
        for _ in 0..LENS[rng.intn(LENS.len())].max(4) {
            src.push(src[src.len() - off]);
        }
    }
    src.truncate(1 << 20);
    src
}

/// TestSIMDEncodeParse checks that the parsers' operations cover the input, which blocks without
/// delta literals don't show, with literal runs and matches around the lengths where the parsers
/// leave operations to the emitters.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_parse() {
    let mut rng = Rng(1);
    let src = test_parse_input(&mut rng);
    let text = rng.text(1 << 20);
    for input in [&src[..64 << 10], &src[..], &text[..]] {
        for l in LEVELS {
            let mut block = vec![0; max_encoded_len(input.len()).unwrap()];
            let mut e = Encoder::new(&mut block[1..]);
            e.parse(input, l);
            assert_eq!(e.pos, input.len(), "size {} level {l}", input.len());
            let n = e.finish();
            assert!(
                n > 0 && decode_simd_block_strict(&block[..n + 1]).as_deref() == Ok(input),
                "size {} level {l}: block of {n} bytes, roundtrip mismatch",
                input.len()
            );
        }
    }
}

/// TestSIMDEncodeDelta
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_delta() {
    let src = Rng(1).text(100000);
    for l in [LEVEL_SMALLEST | LEVEL_DELTA, 2 | LEVEL_DELTA, LEVEL_DELTA] {
        assert_eq!(
            Level::try_from(l),
            Err(crate::Error::InvalidLevel),
            "level {l}"
        );
    }
    for l in LEVELS {
        let enc = encode_level(&src, l);
        let mut modes = [[0; 8]; LITS + 1];
        let mut flags = [0; 2];
        test_stats(&enc, &mut modes, &mut flags);
        let chunks: usize = modes[TOK].iter().sum();
        let want = chunks * (l / LEVEL_DELTA) as usize;
        assert!(
            chunks > 0 && flags[0] == want,
            "level {l}: {} of {chunks} chunks with delta literals, want {want}",
            flags[0]
        );
    }

    // Chunks of literals that are their references plus 0 or 1, then random literals, then
    // again, should take delta, plain and delta literals, in separate literal blocks.
    let mut rng = Rng(1);
    let mut src2: Vec<u8> = Vec::new();
    for delta in [true, false, true] {
        for _ in 0..CHUNK_OPS {
            let p = src2.len();
            for j in 0..6 {
                let mut v = rng.next() as u8;
                if delta && p >= 8 {
                    v = src2[p - 8 + j].wrapping_add(rng.intn(2) as u8);
                }
                src2.push(v);
            }
            let r = 8.min(src2.len());
            for _ in 0..4 {
                src2.push(src2[src2.len() - r]);
            }
        }
    }
    let mut block = vec![0; max_encoded_len(src2.len()).unwrap()];
    let mut e = Encoder::new(&mut block[1..]);
    e.reset(&src2, 256, 0);
    for p in (0..src2.len()).step_by(10) {
        if e.rep == 8 {
            e.emit_repeat_lits(&src2[p..p + 6], 4);
        } else {
            e.emit_copy_lits(&src2[p..p + 6], 8, 4);
        }
    }
    let n = e.finish();
    assert!(n > 0, "block not encoded");
    let switches = e.s.switches.clone();
    let enc = &block[..n + 1];
    assert_eq!(decode_simd_block_strict(enc).as_ref(), Ok(&src2));
    let mut modes = [[0; 8]; LITS + 1];
    let mut flags = [0; 2];
    test_stats(enc, &mut modes, &mut flags);
    let lit_blocks: usize = modes[LITS].iter().sum();
    // Each run of 3*CHUNK_OPS*2 literals needs 2 literal blocks.
    assert!(
        switches[0] == 0 && flags[0] == 2 && lit_blocks == 6,
        "switches {switches:?}, {} delta chunks, {lit_blocks} literal blocks; want delta, plain, delta in 6 literal blocks",
        flags[0]
    );
}

/// TestSIMDEntropyBits checks the bound that put skips new tables with: entropy_bits is at most
/// the entropy, which no table codes below, and the table sizes it adds are those of the tables.
/// It also checks that the values of each stream are below syms and that hist[..syms] holds
/// their counts.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_entropy_bits() {
    let mut rng = Rng(1);
    let mut checks = 0;
    let mut check = |c: &mut Coder, name: &str, layout: usize, v: &[u8]| {
        let k = layout.min(LITS);
        let syms = 2 * CODE_BYTES[k];
        if v.len() <= MIN_CODED {
            return;
        }
        checks += 1;
        c.histogram(v, syms);
        let mut cnt = [0u64; 256];
        for &b in v {
            cnt[b as usize] += 1;
        }
        let n = v.len() as f64;
        let mut h = 0.0;
        for (s, &f) in cnt.iter().enumerate() {
            assert!(
                if s < syms { c.hist[s] == f } else { f == 0 },
                "{name} layout {layout}: value {s}: count {f}, histogram {}",
                c.hist[s]
            );
            if f > 0 {
                h -= f as f64 * (f as f64 / n).log2();
            }
        }
        let low = entropy_bits(&c.hist[..syms], v.len());
        assert!(
            low as f64 <= h,
            "{name} layout {layout}: bound {low} above entropy {h:.3}"
        );
        c.tmp[0].set_code(k, &c.hist);
        c.tmp[1].set_class(layout, &c.hist, syms);
        for (i, size) in [CODE_BYTES[k], CLASS_BYTES[layout]].into_iter().enumerate() {
            let (cost, ok) = c.tmp[i].cost(&c.hist, syms);
            assert!(
                ok && cost as f64 >= h - 1e-6 && c.tmp[i].n == size,
                "{name} layout {layout} table {i}: cost {cost} (ok {ok}) below entropy {h:.3}, or {} table bytes, want {size}",
                c.tmp[i].n
            );
        }
    };
    let mut s = Scratch::new();
    let lo = [0, 1, 1, 0, 0, 0];
    for layout in 0..=DELTA {
        let syms = 2 * CODE_BYTES[layout.min(LITS)];
        for size in [37, 64, 100, 300, 1000, 4096, 16384, 65535] {
            for _ in 0..20 {
                let mut v = Vec::with_capacity(size);
                match rng.intn(3) {
                    0 => {
                        // Few values, uniform.
                        let r = 1 + rng.intn(8.min(syms - lo[layout]));
                        while v.len() < size {
                            v.push((lo[layout] + rng.intn(r)) as u8);
                        }
                    }
                    1 => {
                        // Skewed.
                        let sc = 0.5 + 8.0 * rng.f64();
                        while v.len() < size {
                            let x = lo[layout] + (rng.exp_f64() * sc) as usize;
                            v.push(x.min(syms - 1) as u8);
                        }
                    }
                    _ => {
                        // Counts just below where log2 truncates most, 2^b + 2^(b-8) - 1.
                        while v.len() < size {
                            let b = 9 + rng.intn(8);
                            let f = ((1 << b) + (1 << (b - 8)) - 1).min(size - v.len());
                            let sym = (lo[layout] + rng.intn(syms - lo[layout])) as u8;
                            v.extend(std::iter::repeat_n(sym, f));
                        }
                    }
                }
                check(&mut s.c, &format!("random size {size}"), layout, &v);
            }
        }
    }
    // The streams of real data.
    let mut inputs = vec![twain(), rng.text(1 << 20), rng.mixed(1 << 20)];
    inputs.extend(bench_files().into_iter().map(|(_, b)| b));
    for (i, input) in inputs.iter().enumerate() {
        for src in input.chunks(64 << 10) {
            for level in 0..3 {
                let mut out = vec![0; max_encoded_len(src.len()).unwrap()];
                let mut e = Encoder::with_scratch(&mut out, s);
                e.reset(src, 0, 0);
                match level {
                    0 => e.encode_fast_block_64k(src),
                    1 => e.encode_block_64k(src),
                    _ => e.encode_block_better_64k(src),
                }
                let name = format!("input {i} level {level}");
                let (n_tok, n_ll, n_offc, n_esc, n_lits) =
                    (e.n_tok, e.n_ll, e.n_offc, e.n_esc, e.n_lits);
                s = e.s;
                check(&mut s.c, &name, TOK, &s.tok[..n_tok]);
                check(&mut s.c, &name, LL, &s.ll[..n_ll]);
                check(&mut s.c, &name, OFFC, &s.offc[..n_offc]);
                check(&mut s.c, &name, ESC, &s.esc[..n_esc]);
                check(&mut s.c, &name, LITS, &s.buf[..n_lits]);
                check(&mut s.c, &name, DELTA, &s.buf[..n_lits]);
            }
        }
    }
    assert!(checks > 1000, "{checks} histograms");
}

/// TestEncodeHuge at the SIMD levels.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_huge() {
    let data = vec![0u8; crate::MAX_BLOCK_SIZE];
    for l in LEVELS {
        let enc = encode_level(&data, l);
        assert!(is_simd(&enc), "level {l}");
        assert!(enc.len() <= max_encoded_len(data.len()).unwrap());
        assert!(decode_block(&enc) == Ok(data.clone()), "level {l}");
    }
}

/// Go's `FuzzEncodingBlocks` over the seeds `go test -short` runs it on: the two
/// `buildMaxOffsetBoundary` inputs and, with `MINLZ_TESTDATA`, `enc_regressions.zip` and
/// `fuzz/FuzzEncodingBlocks`. Tom Sawyer is added; Go's bench files are left out, as
/// `simd_encode` runs them. Every level encodes from a source against either guard page, as the
/// parsers read ahead and extend matches back, within `max_encoded_len`; SIMD blocks decode on
/// every exec tier into a `dst` holding garbage, and with the strict reference decoder.
/// `decoded_len` agrees, as Go's `FuzzDecodeBlock` checks on its seeds, SIMD blocks of the same
/// inputs.
#[test]
#[cfg_attr(miri, ignore)]
fn fuzz_encoding_blocks_seeded() {
    use crate::fuzz_seeds::go_seeds;
    let mut inputs = vec![twain()];
    inputs.extend(go_seeds("enc_regressions.zip"));
    inputs.extend(go_seeds("fuzz/FuzzEncodingBlocks"));
    inputs.extend((1..=2).filter_map(|d| crate::block::tests::build_max_offset_boundary(0, d)));
    let mut dec = Vec::with_capacity(crate::MAX_BLOCK_SIZE);
    for data in inputs.iter().filter(|d| d.len() <= crate::MAX_BLOCK_SIZE) {
        let mut g = crate::guard::Buffer::new(data.len()).unwrap();
        let src = g.slice(data.len(), data.len() & 1 == 0);
        src.copy_from_slice(data);
        for l in [1, 2, 3].into_iter().chain(LEVELS) {
            let what = format!("{} bytes, level {l}", data.len());
            let enc = encode_level(src, l);
            assert!(enc.len() <= max_encoded_len(data.len()).unwrap(), "{what}");
            assert_eq!(decoded_len(&enc), Ok(data.len()), "{what}");
            if !is_simd(&enc) {
                assert!(decode_block(&enc).as_ref() == Ok(data), "{what}");
                continue;
            }
            super::super::exec::simd_tiers(|t| {
                dec.clear();
                dec.resize(data.len(), 0xfe);
                decode(&mut dec, &enc).unwrap_or_else(|e| panic!("{what}, {t:?}: {e}"));
                assert!(dec == *data, "{what}, {t:?}: mismatch");
            });
            let r = decode_simd_block_strict(&enc);
            assert!(r.as_ref() == Ok(data), "{what}: reference {:?}", r.err());
        }
    }
}

/// TestEncodeRandomStored at the SIMD levels, all of which store random data.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_encode_random_stored() {
    let mut rng = Rng(0);
    for size in [100, 1000, 10000, 100000, 1000000] {
        let src = rng.random(size);
        let want = [&[0, 0], &src[..]].concat();
        for l in [4, 5, 6, 7, 260, 261, 262] {
            assert!(encode_level(&src, l) == want, "{size} level {l}");
        }
    }
}

impl Rng {
    /// Returns a random byte that is not in not (Go's `simdOtherByte`).
    fn other(&mut self, not: &[u8]) -> u8 {
        loop {
            let c = self.next() as u8;
            if !not.contains(&c) {
                return c;
            }
        }
    }
}

/// Parses src with the Balanced parser, or with the 64 KB one, as parse does but without the size
/// limit, and returns the repeat offset and the literals.
fn better_parse_run(src: &[u8], big: bool) -> (usize, Vec<u8>) {
    let mut out = vec![0; 2 * src.len() + (1 << 16)];
    let mut e = Encoder::new(&mut out);
    e.reset(src, 0, 0);
    e.out.limit = isize::MAX / 2;
    if big {
        e.encode_block_better(src);
    } else {
        e.encode_block_better_64k(src);
    }
    if e.n_tok > 0 {
        e.end_chunk();
    }
    (e.rep, [&e.s.buf[..e.n_lits], e.tail].concat())
}

/// Returns n random bytes with a copy of length bytes at offset off, whose ends Balanced's search
/// checks, so it finds the copy unless a random match comes first. For other lengths only the bytes
/// after the copy differ.
fn short_input(n: usize, off: usize, length: usize) -> Vec<u8> {
    // Not Go's seed n+off: its bytes differ here, and with n+off a short table entry of the copy's
    // source is overwritten before the search reaches the copy.
    let mut rng = Rng((n + off + 1) as u64);
    let mut b = rng.random(n);
    // The positions that the search checks while it finds no match.
    let mut checked = vec![false; n];
    let mut s = 1;
    while s < n {
        checked[s] = true;
        s += 100.min((s >> 7) + 1);
    }
    let mut p = off + 8;
    while !checked[p] || !checked[p - off] {
        p += 1;
    }
    b.copy_within(p - off..p - off + length, p);
    b[p - 1] = rng.other(&[b[p - off - 1]]);
    b[p + length] = rng.other(&[b[p - off + length]]);
    b
}

/// Returns a block where Balanced finds a copy of 4 bytes at offset far+210 that is the repeat
/// offset, which it keeps, where it passes over those at other offsets over its bound, and the
/// position of the copy. The repeat check finds such copies at the positions that the search checks
/// one after another, so this one is at a position that it reaches with a step of 2.
fn far_repeat_input(rng: &mut Rng, far: usize) -> (Vec<u8>, usize) {
    // Zeros end an operation at q, so the search checks q to q+128, then every other position.
    let mut b = rng.random(16);
    b.resize(16 + 300, 0);
    b[15] = rng.other(&[0]);
    let q = b.len();
    b.extend(rng.random(200));
    (b[q], b[q + 199]) = (rng.other(&[0]), rng.other(&[0]));
    b.resize(b.len() + far, 0);
    b.extend(rng.random(20));
    b[q + 200 + far] = rng.other(&[0]);
    // A copy of 30 bytes at q+10 sets the repeat offset. 140 literals later, the search checks
    // with a step of 2, where 4 bytes at the repeat offset follow, those at q+180.
    let t = b.len();
    b.extend_from_within(q + 10..q + 40);
    b.extend(rng.random(200));
    (b[t - 1], b[t + 30]) = (rng.other(&[b[q + 9]]), rng.other(&[b[q + 40]]));
    let s = t + 30 + 140;
    b.copy_within(q + 180..q + 184, s);
    (b[s - 1], b[s + 4]) = (rng.other(&[b[q + 179]]), rng.other(&[b[q + 184]]));
    (b, s)
}

/// TestSIMDParseAsmBounds without the assembly parsers: the Balanced parsers pass over new
/// matches of 4 and 5 bytes over the offsets of better_max_off, at each bound and one over, and
/// keep them at the repeat offset. Unlike Go's, the bounds are Go's numbers, and the sizes include
/// the first of the next class, so that a wrong bound fails too.
#[test]
#[cfg_attr(miri, ignore)]
fn simd_parse_better_bounds() {
    let classes = [
        (2 << 10, [127, 511]),
        ((2 << 10) + 1, [2047, 16383]),
        (8 << 10, [2047, 16383]),
        ((8 << 10) + 1, [1023, 16383]),
        (64 << 10, [1023, 16383]),
    ];
    for (big, max) in [(true, crate::MAX_BLOCK_SIZE), (false, 64 << 10)] {
        for (n, m) in classes {
            assert_eq!(better_max_off(n), m, "{n} bytes");
            for off in [m[0], m[0] + 1, m[1], m[1] + 1] {
                // The copy of 6 bytes, which is kept, shows that the search finds the shorter
                // ones.
                for length in (4..=6).take_while(|_| off < n / 2) {
                    let (rep, _) = better_parse_run(&short_input(n, off, length), big);
                    let want = length > 5 || off <= m[length & 1];
                    assert!(
                        (rep == off) == want,
                        "big {big}: {n} bytes, copy of {length} at {off}: kept {}",
                        !want
                    );
                }
            }
        }
        // Copies at the repeat offset, over the bound of each size class. Without the copy the
        // block has 4 literals more.
        for far in [0, 2000, 10000, 70000] {
            let (mut src, at) = far_repeat_input(&mut Rng(1), far);
            if src.len() > max {
                continue;
            }
            let (_, g) = better_parse_run(&src, big);
            src[at + 3] = src[at + 3].wrapping_add(1);
            let (_, c) = better_parse_run(&src, big);
            assert!(
                c.len() == g.len() + 4,
                "big {big}: far repeat {far}: {} literals, without the copy {}",
                g.len(),
                c.len()
            );
        }
    }
}
