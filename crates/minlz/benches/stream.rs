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

//! Criterion benchmarks for the streaming codec.
//!
//! Mirrors the shape of `block.rs` — Twain corpus at a couple of sizes,
//! encode + decode at every level.
//!
//! Run with:
//!
//! ```
//! cargo bench --bench stream
//! cargo bench --bench stream -- --quick
//! cargo bench --bench stream stream_encode/1e6/L2
//! ```

#![allow(missing_docs)]

use std::io::{Read, Write};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;

use std::io::Cursor;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use minlz::stream::{MtWriterBuilder, ReadSeeker, Reader, WriterBuilder};
use minlz::Level;

fn load_twain() -> Vec<u8> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(d) = std::env::var("MINLZ_TESTDATA") {
        candidates.push(PathBuf::from(d).join("Mark.Twain-Tom.Sawyer.txt"));
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(manifest.join("../../../testdata/Mark.Twain-Tom.Sawyer.txt"));
    candidates.push(manifest.join("../../testdata/Mark.Twain-Tom.Sawyer.txt"));
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

fn encode_stream(src: &[u8], level: Level) -> Vec<u8> {
    let mut w = WriterBuilder::new().level(level).build(Vec::new());
    w.write_all(src).unwrap();
    w.finish().unwrap()
}

fn bench_encode(c: &mut Criterion) {
    let twain = load_twain();
    let sizes: &[(usize, &str)] = &[(100_000, "1e5"), (1_000_000, "1e6")];
    let levels = [Level::Fastest, Level::Balanced, Level::Smallest];

    let mut g = c.benchmark_group("stream_encode");
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(2));
    for (n, label) in sizes {
        let src = expand(&twain, *n);
        g.throughput(Throughput::Bytes(*n as u64));
        for &level in &levels {
            let id = BenchmarkId::new(*label, level_label(level));
            g.bench_with_input(id, &src, |b, src| {
                b.iter(|| {
                    let mut w = WriterBuilder::new().level(level).build(Vec::new());
                    w.write_all(src).unwrap();
                    let _ = w.finish().unwrap();
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

    let mut g = c.benchmark_group("stream_decode");
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(2));
    for (n, label) in sizes {
        let src = expand(&twain, *n);
        g.throughput(Throughput::Bytes(*n as u64));
        for &level in &levels {
            let stream = encode_stream(&src, level);
            let id = BenchmarkId::new(*label, level_label(level));
            g.bench_with_input(id, &stream, |b, stream| {
                let mut out = Vec::with_capacity(src.len());
                b.iter(|| {
                    out.clear();
                    Reader::new(&stream[..]).read_to_end(&mut out).unwrap();
                });
            });
        }
    }
    g.finish();
}

fn bench_encode_mt(c: &mut Criterion) {
    let twain = load_twain();
    // MT shines on larger inputs.  5e6 ≈ a few blocks per thread on
    // 8-core, 5e7 ≈ many blocks per thread.
    let sizes: &[(usize, &str)] = &[(5_000_000, "5e6"), (50_000_000, "5e7")];
    let levels = [Level::Fastest, Level::Balanced];
    let mut g = c.benchmark_group("stream_encode_mt");
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(2));
    let concurrency = std::thread::available_parallelism().unwrap_or(NonZeroUsize::new(1).unwrap());
    for (n, label) in sizes {
        let src = expand(&twain, *n);
        g.throughput(Throughput::Bytes(*n as u64));
        for &level in &levels {
            let id = BenchmarkId::new(*label, level_label(level));
            g.bench_with_input(id, &src, |b, src| {
                b.iter(|| {
                    let mut w = MtWriterBuilder::new()
                        .level(level)
                        .concurrency(concurrency)
                        .build(Vec::with_capacity(src.len() / 8));
                    w.write_all(src).unwrap();
                    let _ = w.finish().unwrap();
                });
            });
        }
    }
    g.finish();
}

fn bench_decode_mt(c: &mut Criterion) {
    let twain = load_twain();
    let sizes: &[(usize, &str)] = &[(5_000_000, "5e6"), (50_000_000, "5e7")];
    let levels = [Level::Fastest, Level::Balanced];
    let mut g = c.benchmark_group("stream_decode_mt");
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(2));
    let concurrency = std::thread::available_parallelism()
        .unwrap_or(NonZeroUsize::new(1).unwrap())
        .get();
    for (n, label) in sizes {
        let src = expand(&twain, *n);
        g.throughput(Throughput::Bytes(*n as u64));
        for &level in &levels {
            let stream = encode_stream(&src, level);
            let id = BenchmarkId::new(*label, level_label(level));
            g.bench_with_input(id, &stream, |b, stream| {
                b.iter(|| {
                    let mut reader = Reader::new(&stream[..]);
                    let (_, _w) = reader
                        .decode_concurrent(Vec::<u8>::new(), concurrency)
                        .unwrap();
                });
            });
        }
    }
    g.finish();
}

/// `index_seek_random` — 1000 random read_at probes against a 256 MiB
/// stream, matching the F-stage plan's analytical-query target.  4 KiB
/// reads at random uncompressed offsets; throughput is reported in
/// bytes returned, so higher is better.
fn bench_index_seek_random(c: &mut Criterion) {
    let twain = load_twain();
    // 256 MiB.  Smaller than the plan's 1 GiB to keep CI bearable; the
    // shape of the per-probe cost is identical above ~32 MiB since the
    // index always has the est_block_uncomp ≥ 1 MiB floor.
    const STREAM_BYTES: usize = 256 << 20;
    const PROBES: usize = 1000;
    const READ_BYTES: usize = 4096;

    let src = expand(&twain, STREAM_BYTES);
    // Index-bearing stream at 32 KiB blocks (the value the example
    // settled on).  Encoded once; the benchmark only times the
    // ReadSeeker creation + probe loop.
    let mut compressed: Vec<u8> = Vec::with_capacity(src.len());
    let mut w = WriterBuilder::new()
        .block_size(32 << 10)
        .level(Level::Balanced)
        .append_index()
        .build(&mut compressed);
    w.write_all(&src).unwrap();
    let _ = w.finish().unwrap();

    // Pre-generate offsets via a deterministic LCG so iterations cover
    // the same distribution.
    let mut offsets = Vec::with_capacity(PROBES);
    let mut x: u64 = 0x00c0_ffee_5eed;
    let max_off = (STREAM_BYTES - READ_BYTES) as u64;
    for _ in 0..PROBES {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        offsets.push((x >> 33) % max_off);
    }

    let mut g = c.benchmark_group("index_seek_random");
    g.measurement_time(Duration::from_secs(8));
    g.warm_up_time(Duration::from_secs(2));
    g.throughput(Throughput::Bytes((PROBES * READ_BYTES) as u64));
    g.bench_with_input(
        BenchmarkId::new("256MiB_4KiB_x1000", "L2"),
        &compressed,
        |b, compressed| {
            let mut buf = vec![0u8; READ_BYTES];
            b.iter(|| {
                let reader = Reader::new(Cursor::new(compressed.as_slice()));
                let mut rs = ReadSeeker::new(reader, &[]).expect("ReadSeeker::new");
                for &off in &offsets {
                    let n = rs.read_at(&mut buf, off).expect("read_at");
                    criterion::black_box(&buf[..n]);
                }
            });
        },
    );
    g.finish();
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_encode_mt,
    bench_decode_mt,
    bench_index_seek_random,
);
criterion_main!(benches);
