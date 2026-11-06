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

//! Criterion benchmarks for pivco, per kernel tier.
//!
//! Mirrors Go's `BenchmarkBlock` (`internal/pivco/block_test.go`) and `BenchmarkGather`,
//! `BenchmarkPartition`, `BenchmarkPack`, `BenchmarkExec`, `BenchmarkHistogram`
//! (`internal/pivco/kernel/bench_test.go`): same inputs (Go's PCG seeds), sizes and
//! sub-benchmark names, so `Block/generic/enc/english` here is Go's
//! `BenchmarkBlock/generic/enc/english`. Throughput is reported in MB/s (10^6), as Go does.
//!
//! ```text
//! cargo bench --bench pivco
//! cargo bench --bench pivco -- Exec/generic/vv
//! ```

#![allow(missing_docs)] // criterion_group! emits an undocumented fn

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use minlz::pivco::kernel::{
    ARENA_SLACK, HIST_SCRATCH, OP_CC, OP_COPY, OP_CV, OP_NAT, OP_V128, OP_V512, OP_VV, Op, Program,
    R2S_LEN, RANK_SLACK, Tier,
};
use minlz::pivco::{
    Decoder, Encoder, Error, MAX_BLOCK_SIZE, MAX_CODE_LEN, Table, build_lengths, kernel,
};

#[path = "../src/pivco/testutil.rs"]
mod testutil;

use testutil::*;

const BENCH_N: usize = 32768;

/// Zero bytes at a page boundary, as Go's large allocations, so that the buffers are
/// placed relative to cache lines and pages as in Go's benchmarks.
struct Aligned(&'static mut [u8]);

impl core::ops::Deref for Aligned {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.0
    }
}

impl core::ops::DerefMut for Aligned {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.0
    }
}

fn aligned(n: usize) -> Aligned {
    let v = vec![0u8; n + 4096].leak();
    let o = v.as_ptr().align_offset(4096);
    Aligned(&mut v[o..o + n])
}

fn bench_block(c: &mut Criterion) {
    let mut g = c.benchmark_group("Block");
    for name in ["english", "proba80", "uniform", "sparse_16"] {
        let h = dist_of(name);
        let tb = must_table(&lens_of(&h));
        let src = sample(&h, 32768, 1);
        g.throughput(Throughput::BytesDecimal(src.len() as u64));
        for mut cd in coders() {
            let mut blk = aligned(block_size(&mut cd, &src, &tb));
            cd.enc.encode(&mut blk, &src, &tb).unwrap();
            g.bench_function(format!("{}/enc/{name}", cd.tier), |b| {
                b.iter(|| cd.enc.encode(&mut blk, &src, &tb).unwrap())
            });
            let mut dst = aligned(src.len());
            g.bench_function(format!("{}/dec/{name}", cd.tier), |b| {
                b.iter(|| cd.dec.decode(&mut dst, &blk, &tb).unwrap())
            });
        }
    }
    g.finish();
}

fn bench_gather(c: &mut Criterion) {
    let mut rng = new_rng();
    let mut src = aligned(BENCH_N);
    rand_bytes(&mut rng, &mut src);
    let mut ranks = aligned(BENCH_N + RANK_SLACK);
    let mut s2r = [0u8; 256];
    rand_bytes(&mut rng, &mut s2r);
    let mut g = c.benchmark_group("Gather");
    g.throughput(Throughput::BytesDecimal(BENCH_N as u64));
    for t in Tier::tiers() {
        g.bench_function(t.to_string(), |b| {
            b.iter(|| t.gather(&mut ranks, &src, &s2r))
        });
    }
    g.finish();
}

fn bench_partition(c: &mut Criterion) {
    let mut rng = new_rng();
    let mut src = aligned(BENCH_N);
    rand_bytes(&mut rng, &mut src);
    let mut ranks = aligned(BENCH_N + RANK_SLACK);
    let mut right = aligned(BENCH_N + RANK_SLACK);
    let mut bm = aligned(BENCH_N / 8 + RANK_SLACK);
    let mut g = c.benchmark_group("Partition");
    g.throughput(Throughput::BytesDecimal(BENCH_N as u64));
    for t in Tier::tiers() {
        g.bench_function(format!("{t}/full"), |b| {
            b.iter(|| {
                // part_full consumes its input; the copy is part of the time.
                ranks[..BENCH_N].copy_from_slice(&src);
                t.part_full(&mut ranks, BENCH_N, &mut right, &mut bm, 127)
            })
        });
        ranks[..BENCH_N].copy_from_slice(&src);
        g.bench_function(format!("{t}/right"), |b| {
            b.iter(|| t.part_right(&ranks, BENCH_N, &mut right, &mut bm, 127))
        });
        g.bench_function(format!("{t}/none"), |b| {
            b.iter(|| t.part_none(&ranks, BENCH_N, &mut bm, 127))
        });
    }
    g.finish();
}

fn bench_pack(c: &mut Criterion) {
    let mut src = aligned(BENCH_N);
    rand_bytes(&mut new_rng(), &mut src);
    let mut ranks = aligned(BENCH_N + RANK_SLACK);
    let mut dst = aligned(BENCH_N);
    let mut g = c.benchmark_group("Pack");
    g.throughput(Throughput::BytesDecimal(BENCH_N as u64));
    for t in Tier::tiers() {
        for d in 2..=8u8 {
            for (r, &s) in ranks.iter_mut().zip(src.iter()) {
                *r = s & ((1u16 << d) - 1) as u8;
            }
            let size = (BENCH_N * usize::from(d)).div_ceil(8);
            g.bench_function(format!("{t}/d{d}"), |b| {
                b.iter(|| t.pack(&mut dst[..size], &ranks, BENCH_N, d, 0))
            });
        }
    }
    g.finish();
}

fn bench_exec(c: &mut Criterion) {
    let mut rng = new_rng();
    let (bm, kr) = rand_bits(&mut rng, BENCH_N, 0.5);
    let mut tail = aligned(BENCH_N);
    rand_bytes(&mut rng, &mut tail);
    let mut input = aligned(bm.len() + tail.len());
    input[..bm.len()].copy_from_slice(&bm);
    input[bm.len()..].copy_from_slice(&tail);
    let mut arena = aligned(3 * BENCH_N + ARENA_SLACK);
    rand_bytes(&mut rng, &mut arena);
    let mut r2s = [0u8; R2S_LEN];
    rand_bytes(&mut rng, &mut r2s);
    let (n, kr) = (BENCH_N as u32, kr as u32);
    let mut cases = vec![
        (
            "vv".to_string(),
            Op {
                kind: OP_VV,
                kr,
                l: n,
                r: 2 * n,
                ..Op::default()
            },
        ),
        (
            "vv/inplace".to_string(),
            Op {
                kind: OP_VV,
                kr,
                l: 2 * n,
                r: n - kr,
                ..Op::default()
            },
        ),
        (
            "cv".to_string(),
            Op {
                kind: OP_CV,
                kr,
                p: 7,
                r: n,
                ..Op::default()
            },
        ),
        (
            "cc".to_string(),
            Op {
                kind: OP_CC,
                p: 7,
                q: 9,
                ..Op::default()
            },
        ),
        (
            "copy".to_string(),
            Op {
                kind: OP_COPY,
                d: 8,
                ..Op::default()
            },
        ),
    ];
    for d in 2..=7u8 {
        for (kind, name) in [(OP_V512, "v512"), (OP_V128, "v128"), (OP_NAT, "nat")] {
            cases.push((
                format!("{name}/d{d}"),
                Op {
                    kind,
                    d,
                    ..Op::default()
                },
            ));
        }
    }
    let mut g = c.benchmark_group("Exec");
    g.throughput(Throughput::BytesDecimal(BENCH_N as u64));
    for t in Tier::tiers() {
        for (name, op) in &cases {
            let mut op = Op { k: n, ..*op };
            if op.kind >= OP_V512 {
                op.bm = bm.len() as u32;
            }
            let ops = [op];
            g.bench_function(format!("{t}/{name}"), |b| {
                let mut p = Program {
                    ops: &ops,
                    input: &input,
                    arena: &mut arena,
                    dst: &mut [],
                    r2s: Some(&r2s),
                };
                b.iter(|| assert_eq!(t.exec(&mut p), Ok(()), "exec failed"))
            });
        }
    }
    g.finish();
}

fn bench_histogram(c: &mut Criterion) {
    let mut src = aligned(BENCH_N);
    rand_bytes(&mut new_rng(), &mut src);
    let mut scratch = vec![0u8; HIST_SCRATCH];
    let mut g = c.benchmark_group("Histogram");
    g.throughput(Throughput::BytesDecimal(BENCH_N as u64));
    for t in Tier::tiers() {
        let mut h = [0u32; 256];
        g.bench_function(t.to_string(), |b| {
            b.iter(|| t.histogram(&mut h, &src, &mut scratch))
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_block,
    bench_gather,
    bench_partition,
    bench_pack,
    bench_exec,
    bench_histogram
);
criterion_main!(benches);
