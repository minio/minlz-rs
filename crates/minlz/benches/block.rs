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

//! Criterion benchmarks for the block codec.
//!
//! Mirrors Go's `BenchmarkTwainEncode1eN` / `BenchmarkTwainDecode1eN` shape
//! (`benchmarks_test.go:123` / `:108`).  Reads
//! `testdata/Mark.Twain-Tom.Sawyer.txt` from the upstream Go repo and runs
//! encode + decode at each level for several scaled-up corpus sizes.
//!
//! Run with:
//!
//! ```
//! cargo bench --bench block
//! cargo bench --bench block -- --quick      # rough estimate, ~5 s/case
//! cargo bench --bench block twain_decode_1e5/L1   # one case
//! ```
//!
//! HTML reports land in `target/criterion/`.
//!
//! The reference Go numbers (from `go test -tags=noasm -bench=BenchmarkTwain*`)
//! are listed in the README so a CI job can compare percentages.

#![allow(missing_docs)] // criterion_group! emits an undocumented fn

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use minlz::{decode, encode, Level};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Upstream Snappy testdata corpus.  Files are not committed to this repo
/// (see `.gitignore`); `ensure_bench_file` downloads them on demand into
/// `testdata/bench/`.
const BENCH_URL: &str = "https://raw.githubusercontent.com/google/snappy/master/testdata/";

/// Locate the Twain corpus.  Honours the `MINLZ_TESTDATA` env override so
/// the bench can be run from a copy of the repo that doesn't include
/// `testdata/`.
fn load_twain() -> Vec<u8> {
    let candidates: Vec<PathBuf> = {
        let mut v = Vec::new();
        if let Ok(d) = std::env::var("MINLZ_TESTDATA") {
            v.push(PathBuf::from(d).join("Mark.Twain-Tom.Sawyer.txt"));
        }
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        v.push(manifest.join("../../../testdata/Mark.Twain-Tom.Sawyer.txt"));
        v.push(manifest.join("../../testdata/Mark.Twain-Tom.Sawyer.txt"));
        v
    };
    for p in &candidates {
        if let Ok(b) = std::fs::read(p) {
            return b;
        }
    }
    panic!(
        "could not locate Mark.Twain-Tom.Sawyer.txt; tried: {candidates:?}.  \
         Set MINLZ_TESTDATA=<dir> to override."
    );
}

/// Port of Go's `expand(src, n)` (`minlz_test.go:1461`).  Tiles `src` to
/// fill an `n`-byte buffer, XOR-ing each copy with an incrementing counter
/// so adjacent runs don't compress trivially.
fn expand(src: &[u8], n: usize) -> Vec<u8> {
    let mut dst = vec![0u8; n];
    let mut cnt: u8 = 0;
    let mut start = 0;
    while start < n {
        let end = (start + src.len()).min(n);
        let copy_len = end - start;
        dst[start..end].copy_from_slice(&src[..copy_len]);
        for b in dst[start..end].iter_mut() {
            *b ^= cnt;
        }
        start += src.len();
        cnt = cnt.wrapping_add(1);
    }
    dst
}

fn level_label(l: Level) -> &'static str {
    match l {
        Level::Fastest => "L1",
        Level::Balanced => "L2",
        Level::Smallest => "L3",
    }
}

/// Canonical write location for downloaded bench data.  Honours
/// `MINLZ_TESTDATA` for the same override convention used elsewhere.
fn bench_dir() -> PathBuf {
    if let Ok(d) = std::env::var("MINLZ_TESTDATA") {
        return PathBuf::from(d).join("bench");
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/bench")
}

/// Ensure `testdata/bench/<filename>` exists, downloading from the
/// upstream Snappy testdata corpus if missing.  Mirrors Go's
/// `downloadBenchmarkFiles` (minlz-go `benchmarks_test.go`).
fn ensure_bench_file(filename: &str) -> std::io::Result<PathBuf> {
    let dir = bench_dir();
    let path = dir.join(filename);
    if let Ok(m) = std::fs::metadata(&path) {
        if m.len() > 0 {
            return Ok(path);
        }
    }
    std::fs::create_dir_all(&dir)?;
    let url = format!("{BENCH_URL}{filename}");
    eprintln!("minlz bench: downloading {url} -> {}", path.display());
    let resp = ureq::get(&url)
        .call()
        .map_err(|e| std::io::Error::other(format!("GET {url}: {e}")))?;
    let mut body = Vec::new();
    resp.into_reader().read_to_end(&mut body)?;
    let tmp = dir.join(format!("{filename}.partial"));
    let res = (|| -> std::io::Result<()> {
        std::fs::write(&tmp, &body)?;
        std::fs::rename(&tmp, &path)
    })();
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res.map(|()| path)
}

fn load_bench_file(filename: &str) -> Vec<u8> {
    let path =
        ensure_bench_file(filename).unwrap_or_else(|e| panic!("ensure bench file {filename}: {e}"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", Path::new(&path).display()))
}

/// Mirrors `benchmarks_test.go:testFiles` — the Snappy benchmark corpus.
struct SnappyFile {
    label: &'static str,
    filename: &'static str,
    size_limit: usize, // 0 = no limit
}

const SNAPPY_FILES: &[SnappyFile] = &[
    SnappyFile {
        label: "html",
        filename: "html",
        size_limit: 0,
    },
    SnappyFile {
        label: "urls",
        filename: "urls.10K",
        size_limit: 0,
    },
    SnappyFile {
        label: "jpg",
        filename: "fireworks.jpeg",
        size_limit: 0,
    },
    SnappyFile {
        label: "jpg_200b",
        filename: "fireworks.jpeg",
        size_limit: 200,
    },
    SnappyFile {
        label: "pdf",
        filename: "paper-100k.pdf",
        size_limit: 0,
    },
    SnappyFile {
        label: "html4",
        filename: "html_x_4",
        size_limit: 0,
    },
    SnappyFile {
        label: "txt1",
        filename: "alice29.txt",
        size_limit: 0,
    },
    SnappyFile {
        label: "txt2",
        filename: "asyoulik.txt",
        size_limit: 0,
    },
    SnappyFile {
        label: "txt3",
        filename: "lcet10.txt",
        size_limit: 0,
    },
    SnappyFile {
        label: "txt4",
        filename: "plrabn12.txt",
        size_limit: 0,
    },
    SnappyFile {
        label: "pb",
        filename: "geo.protodata",
        size_limit: 0,
    },
    SnappyFile {
        label: "gaviota",
        filename: "kppkn.gtb",
        size_limit: 0,
    },
    SnappyFile {
        label: "txt1_128b",
        filename: "alice29.txt",
        size_limit: 128,
    },
    SnappyFile {
        label: "txt1_1000b",
        filename: "alice29.txt",
        size_limit: 1000,
    },
    SnappyFile {
        label: "txt1_10000b",
        filename: "alice29.txt",
        size_limit: 10000,
    },
    SnappyFile {
        label: "txt1_20000b",
        filename: "alice29.txt",
        size_limit: 20000,
    },
];

fn bench_encode(c: &mut Criterion) {
    let twain = load_twain();
    let sizes: &[(usize, &str)] = &[(100_000, "1e5"), (1_000_000, "1e6")];
    let levels = [Level::Fastest, Level::Balanced, Level::Smallest];

    let mut g = c.benchmark_group("twain_encode");
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(2));
    for (n, label) in sizes {
        let src = expand(&twain, *n);
        g.throughput(Throughput::Bytes(*n as u64));
        for &level in &levels {
            let id = BenchmarkId::new(*label, level_label(level));
            g.bench_with_input(id, &src, |b, src| {
                let mut buf = Vec::with_capacity(src.len() + 16);
                b.iter(|| {
                    encode(&mut buf, src, level).expect("encode");
                });
            });
        }
    }
    g.finish();
}

fn bench_decode(c: &mut Criterion) {
    let twain = load_twain();
    let sizes: &[(usize, &str)] = &[(100_000, "1e5"), (1_000_000, "1e6")];
    let levels = [Level::Fastest, Level::Balanced, Level::Smallest];

    let mut g = c.benchmark_group("twain_decode");
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(2));
    for (n, label) in sizes {
        let src = expand(&twain, *n);
        g.throughput(Throughput::Bytes(*n as u64));
        for &level in &levels {
            let mut enc = Vec::new();
            encode(&mut enc, &src, level).expect("encode for bench prep");
            let id = BenchmarkId::new(*label, level_label(level));
            g.bench_with_input(id, &enc, |b, enc| {
                let mut buf = Vec::with_capacity(src.len() + 16);
                b.iter(|| {
                    decode(&mut buf, enc).expect("decode");
                });
            });
        }
    }
    g.finish();
}

/// Per-(file, level) decode bench mirroring Go's
/// `BenchmarkDecodeBlockSingle` over the Snappy testdata corpus.
fn bench_snappy_decode(c: &mut Criterion) {
    let levels = [Level::Fastest, Level::Balanced, Level::Smallest];
    let mut g = c.benchmark_group("snappy_decode");
    g.measurement_time(Duration::from_secs(2));
    g.warm_up_time(Duration::from_secs(1));
    for tf in SNAPPY_FILES {
        let mut data = load_bench_file(tf.filename);
        if tf.size_limit > 0 && data.len() > tf.size_limit {
            data.truncate(tf.size_limit);
        }
        g.throughput(Throughput::Bytes(data.len() as u64));
        for &level in &levels {
            let mut enc = Vec::new();
            encode(&mut enc, &data, level).expect("encode for bench prep");
            let id = BenchmarkId::new(tf.label, level_label(level));
            g.bench_with_input(id, &enc, |b, enc| {
                let mut buf = Vec::with_capacity(data.len() + 16);
                b.iter(|| {
                    decode(&mut buf, enc).expect("decode");
                });
            });
        }
    }
    g.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_snappy_decode);
criterion_main!(benches);
