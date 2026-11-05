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

//! Ports of Go's `TestSIMDEncodeAllocs` and `TestEncodeAllocs` (`encode_test.go`): after warmup,
//! encoding into a large enough dst must not allocate. A counting global allocator needs its own
//! test binary.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use minlz::{Level, encode, max_encoded_len};

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

fn check_allocs(what: &str, mut f: impl FnMut()) {
    f(); // warm up
    let before = ALLOCS.with(Cell::get);
    for _ in 0..5 {
        f();
    }
    let n = ALLOCS.with(Cell::get) - before;
    assert_eq!(n, 0, "{what}: {n} allocs in 5 runs");
}

/// Go's genEncText of 1 MiB, from SplitMix64.
fn text() -> Vec<u8> {
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
    let mut s = 1u64;
    let mut out = Vec::new();
    while out.len() < 1 << 20 {
        s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        out.extend_from_slice(WORDS[((z ^ (z >> 31)) % 15) as usize].as_bytes());
        out.push(b' ');
    }
    out.truncate(1 << 20);
    out
}

#[test]
fn counter_counts() {
    let before = ALLOCS.with(Cell::get);
    drop(std::hint::black_box(vec![0u8; 100]));
    assert!(ALLOCS.with(Cell::get) > before);
}

/// TestSIMDEncodeAllocs, also at 64 KB, where the parsers have other tables.
#[test]
fn simd_encode_allocs() {
    let src = text();
    let mut dst = Vec::with_capacity(max_encoded_len(src.len()).unwrap());
    for n in [64 << 10, src.len()] {
        for l in [4, 5, 6, 7, 260, 261, 262] {
            let level = Level::try_from(l).unwrap();
            check_allocs(&format!("size {n} level {l}"), || {
                encode(&mut dst, &src[..n], level).unwrap();
            });
        }
    }
}

/// TestEncodeAllocs checks that the classic levels take their hash tables from pools.
#[test]
fn encode_allocs() {
    let src = text();
    let mut dst = Vec::with_capacity(max_encoded_len(src.len()).unwrap());
    for n in [1 << 10, 64 << 10, src.len()] {
        for level in [
            Level::SuperFast,
            Level::Fastest,
            Level::Balanced,
            Level::Smallest,
        ] {
            check_allocs(&format!("size {n} level {level:?}"), || {
                encode(&mut dst, &src[..n], level).unwrap();
            });
        }
    }
}
