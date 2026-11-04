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

//! Port of Go's `TestSIMDDecodeAllocs` (`decode-simd_test.go`): after warmup, decoding SIMD
//! blocks does not allocate. A counting global allocator needs its own test binary. Go builds
//! its block with the test builder; Go encoder fixtures with coded and class tables, delta
//! literals, several chunks and literal blocks stand in for it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

/// Counts the allocations of the current thread.
struct Counting;

fn count() {
    let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
}

// SAFETY: forwards to the system allocator; counting does not allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller upholds the GlobalAlloc contract.
        unsafe { System.alloc(l) }
    }

    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        count();
        // SAFETY: as for alloc.
        unsafe { System.alloc_zeroed(l) }
    }

    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        count();
        // SAFETY: as for alloc.
        unsafe { System.realloc(p, l, n) }
    }

    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: as for alloc.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn fixture(name: &str) -> Vec<u8> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/simd/");
    std::fs::read(format!("{p}{name}")).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn simd_decode_allocs() {
    let blocks: Vec<Vec<u8>> = [
        "mixed-1126400.L262.mzb",
        "sparse-524288.L5.mzb",
        "sawyer-65536.L7.mzb",
        "textlike-4096.L260.mzb",
    ]
    .into_iter()
    .map(fixture)
    .collect();
    let mut dst = Vec::with_capacity(minlz::MAX_BLOCK_SIZE);
    let mut run = || {
        for b in &blocks {
            minlz::decode(&mut dst, b).unwrap();
        }
    };
    // The best tier, then the lower ones, as Go's simdTiers.
    for mask in [
        0,
        minlz::cpu::AVX512,
        minlz::cpu::AVX512 | minlz::cpu::AVX2 | minlz::cpu::NEON,
    ] {
        minlz::cpu::disable(mask);
        run(); // warm up
        let before = ALLOCS.with(Cell::get);
        for _ in 0..20 {
            run();
        }
        let n = ALLOCS.with(Cell::get) - before;
        minlz::cpu::enable_all();
        assert_eq!(n, 0, "tiers without {mask:#x}: {n} allocs in 20 runs");
    }
    // The counter works.
    let before = ALLOCS.with(Cell::get);
    drop(std::hint::black_box(vec![0u8; 100]));
    assert!(ALLOCS.with(Cell::get) > before);
}
