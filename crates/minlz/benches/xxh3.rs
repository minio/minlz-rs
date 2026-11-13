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

//! Criterion benchmarks for XXH3-64: one-shot, and streaming in 64 KiB writes.
//!
//! ```
//! cargo bench --bench xxh3
//! cargo bench --bench xxh3 xxh3/hasher_64k
//! ```

#![allow(missing_docs)] // criterion_group! emits an undocumented fn

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use minlz::xxh3::{Hasher, hash};

fn bench_xxh3(c: &mut Criterion) {
    let mut g = c.benchmark_group("xxh3");
    for size in [1 << 10, 64 << 10, 1 << 20, 4 << 20] {
        let src: Vec<u8> = (0..size as u32)
            .map(|i| (i.wrapping_mul(0x9E37_79B1) >> 24) as u8)
            .collect();
        g.throughput(Throughput::Bytes(size as u64));
        g.bench_with_input(BenchmarkId::new("hash", size), &src, |b, s| {
            b.iter(|| hash(black_box(s)))
        });
        g.bench_with_input(BenchmarkId::new("hasher_64k", size), &src, |b, s| {
            let mut h = Hasher::new();
            b.iter(|| {
                h.reset();
                for c in black_box(s).chunks(64 << 10) {
                    h.write(c);
                }
                h.sum64()
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench_xxh3);
criterion_main!(benches);
