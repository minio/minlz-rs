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

//! Index parser fuzz target.
//!
//! * Arbitrary bytes → `Index::load` must never panic.
//! * For randomly-shaped indices, `append_to` → `load` must round-trip.
//!
//! Run with `cargo +nightly fuzz run index_load -- -max_total_time=60`.

#![no_main]

use libfuzzer_sys::fuzz_target;
use minlz::Index;

fuzz_target!(|data: &[u8]| {
    // Half the time, parse raw bytes; half the time, treat the input as
    // a seed for a synthetic round-trip.
    if data.first().copied().unwrap_or(0) & 1 == 0 {
        let mut idx = Index::default();
        let _ = idx.load(data);
        return;
    }

    // Build a small synthetic index seeded by `data` and verify the
    // round-trip.  Bail out cheaply on under-sized inputs.
    if data.len() < 8 {
        return;
    }
    let est = ((data[0] as u64) << 10).max(1 << 12);
    let mut idx = Index::default();
    idx.reset(est as usize);
    let n = (data[1] as usize) % 64;
    let mut comp = 0u64;
    let mut uncomp = 0u64;
    for i in 0..n {
        comp += (data.get(2 + i).copied().unwrap_or(1) as u64) * 100 + 1;
        uncomp += est + (data.get(2 + i).copied().unwrap_or(1) as u64);
        let _ = idx.add(comp, uncomp);
    }
    let total_u = uncomp;
    let total_c = comp;
    let mut buf = Vec::new();
    idx.append_to(&mut buf, Some(total_u), Some(total_c))
        .expect("synthetic totals fit i64");

    let mut idx2 = Index::default();
    idx2.load(&buf).expect("round-trip load");
    assert_eq!(idx2.total_uncompressed(), Some(total_u));
    assert_eq!(idx2.total_compressed(), Some(total_c));
    assert_eq!(idx2.offsets(), idx.offsets());
});
