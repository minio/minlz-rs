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

//! Seeds of the cargo-fuzz targets in `fuzz/`, derived as Go's fuzzers derive theirs with
//! `f.Add`. The ignored `write_fuzz_seeds` tests (here, in `block::tests` and
//! `pivco::tests::fuzz`) add them to `fuzz/corpus/<target>/`, named by content hash:
//!
//! ```text
//! MINLZ_TESTDATA=<Go repo>/testdata cargo test -p minlz --release --lib write_fuzz_seeds -- --ignored
//! ```
//!
//! The raw inputs are Tom Sawyer and the inputs of the SIMD fixtures, and with `MINLZ_TESTDATA`
//! also Go's seed zips and its `testdata/fuzz/FuzzEncodingBlocks` corpus.
//!
//! [`go_seeds`] also gives the tests that run Go's seed corpora their inputs.

use std::io::Write;
use std::path::Path;

use crate::block::simd::{MIN_SAVING, simd_header};
use crate::block::{Level, MAX_BLOCK_SIZE, encode};
use crate::stream::WriterBuilder;
use crate::unzip::{simd_fixtures, unzip};

fn manifest() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Adds `seeds` to `fuzz/corpus/<target>/`.
pub(crate) fn save(target: &str, seeds: impl IntoIterator<Item = Vec<u8>>) {
    let dir = manifest().join("fuzz/corpus").join(target);
    std::fs::create_dir_all(&dir).unwrap();
    let mut n = 0;
    for s in seeds {
        let name = format!("seed-{:016x}", crate::xxh3::hash(&s));
        std::fs::write(dir.join(name), &s).unwrap();
        n += 1;
    }
    eprintln!("{target}: {n} seeds");
}

/// The seeds in Go's `testdata/<name>`, a zip or a Go fuzzer's corpus directory, with
/// `go test fuzz v1` files expanded to their values. Empty without `MINLZ_TESTDATA` or the
/// directory.
pub(crate) fn go_seeds(name: &str) -> Vec<Vec<u8>> {
    let Some(dir) = std::env::var_os("MINLZ_TESTDATA") else {
        return Vec::new();
    };
    let p = Path::new(&dir).join(name);
    let files = if name.ends_with(".zip") {
        unzip(&std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
            .into_iter()
            .map(|(_, f)| f)
            .collect()
    } else if let Ok(rd) = std::fs::read_dir(&p) {
        rd.map(|e| std::fs::read(e.unwrap().path()).unwrap())
            .collect()
    } else {
        Vec::new()
    };
    files.iter().flat_map(|f| go_values(f)).collect()
}

/// The `[]byte` values of a `go test fuzz v1` file, or the file itself, as Go's
/// `internal/fuzz` reads seeds.
fn go_values(f: &[u8]) -> Vec<Vec<u8>> {
    if !f.starts_with(b"go test fuzz") {
        return vec![f.to_vec()];
    }
    f.split(|&c| c == b'\n')
        .skip(1)
        .filter_map(|l| l.trim_ascii().strip_prefix(b"[]byte(")?.strip_suffix(b")"))
        .map(unquote)
        .collect()
}

/// Go's `strconv.Unquote` of a double-quoted string literal.
fn unquote(q: &[u8]) -> Vec<u8> {
    let s = &q[1..q.len() - 1];
    let num =
        |b: &[u8], radix| u32::from_str_radix(std::str::from_utf8(b).unwrap(), radix).unwrap();
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        if s[i] != b'\\' {
            out.push(s[i]);
            i += 1;
            continue;
        }
        let e = s[i + 1];
        i += 2;
        match e {
            b'x' => {
                out.push(num(&s[i..i + 2], 16) as u8);
                i += 2;
            }
            b'0'..=b'7' => {
                out.push(num(&s[i - 1..i + 2], 8) as u8);
                i += 2;
            }
            b'u' | b'U' => {
                let n = if e == b'u' { 4 } else { 8 };
                let c = char::from_u32(num(&s[i..i + n], 16)).unwrap();
                out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
                i += n;
            }
            b'a' => out.push(7),
            b'b' => out.push(8),
            b'f' => out.push(12),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'v' => out.push(11),
            c => out.push(c),
        }
    }
    out
}

/// The raw inputs the seeds derive from, at most `MAX_BLOCK_SIZE` each.
fn inputs() -> Vec<Vec<u8>> {
    let mut v =
        vec![std::fs::read(manifest().join("../../testdata/Mark.Twain-Tom.Sawyer.txt")).unwrap()];
    // Stored fixtures have no block file; the others repeat their input at each level.
    for (name, block) in simd_fixtures() {
        if name.ends_with(".mzb") {
            v.push(crate::reference::decode_simd_block(&block).unwrap());
        }
    }
    for name in [
        "enc_regressions.zip",
        "fuzz/block-corpus-raw.zip",
        "fuzz/block-corpus-enc.zip",
        "fuzz/FuzzEncodingBlocks",
    ] {
        v.extend(go_seeds(name));
    }
    v.retain(|b| b.len() <= MAX_BLOCK_SIZE);
    v.sort();
    v.dedup();
    v
}

/// A seed of the `simd_decode` target, `extra` and the body, for a SIMD block.
fn simd_seed(block: &[u8]) -> Option<Vec<u8>> {
    let (size, body) = simd_header(block.get(1..)?).ok()?;
    let extra = (size - body.len() - 4 - MIN_SAVING) as u32;
    Some([&extra.to_le_bytes()[..], body].concat())
}

fn stream(data: &[u8], b: WriterBuilder) -> Vec<u8> {
    let mut w = b.build(Vec::new()).unwrap();
    w.write_all(data).unwrap();
    w.finish().unwrap()
}

/// Seeds of the block and stream targets, as Go's `FuzzEncodingBlocks`, `FuzzDecodeBlock`,
/// `FuzzDecodeSIMD`, `FuzzStreamEncode` and `FuzzStreamDecode` add them.
#[test]
#[ignore = "writes fuzz/corpus, see fuzz/RUNBOOK.md §1"]
fn write_fuzz_seeds() {
    let inputs = inputs();
    let (mut dec, mut simd, mut sdec) = (Vec::new(), Vec::new(), Vec::new());
    for b in &inputs {
        let enc = |l| {
            let mut d = Vec::new();
            encode(&mut d, b, l).unwrap();
            d
        };
        let (bal, bal_simd, small_simd) = (
            enc(Level::Balanced),
            enc(Level::BalancedSimd),
            enc(Level::SmallestSimd),
        );
        simd.extend(
            [
                &enc(Level::SuperFastSimd),
                &bal_simd,
                &small_simd,
                &enc(Level::FastestSimdDelta),
            ]
            .into_iter()
            .filter_map(|b| simd_seed(b)),
        );
        dec.extend([bal, bal_simd, small_simd]);
        let w = || WriterBuilder::new().block_size(8 << 10);
        for s in [
            stream(b, w()),
            stream(b, w().level(Level::BalancedSimd).append_index()),
            stream(b, w().uncompressed().append_index()),
        ] {
            // The mode of Go's reader: two threads, no checksums, 1 MiB blocks.
            sdec.push([&[0x0d][..], &s].concat());
        }
    }
    dec.extend(go_seeds("dec-block-regressions.zip"));
    dec.extend(go_seeds("fuzz/block-corpus-dec.zip"));
    save("decode_arbitrary", dec);
    save("simd_decode", simd);
    save("stream_decode_arbitrary", sdec);
    // Go's four stream encoders as (mode, level index), one per input, with its write method.
    const ENCODERS: [(u8, u8); 4] = [(0x04, 0), (0x44, 4), (0xc5, 9), (0x05, 3)];
    save(
        "stream_roundtrip",
        inputs.iter().enumerate().map(|(i, b)| {
            let (mode, level) = ENCODERS[i % 4];
            let method = ((b.len() + i) % 4) as u8;
            [&[mode | method << 4, level][..], b].concat()
        }),
    );
    save("roundtrip", inputs);
}
