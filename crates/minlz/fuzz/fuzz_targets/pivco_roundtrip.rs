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

//! pivco round trip on every kernel tier (Go's `FuzzRoundTripBlock`): a flags byte (ghost
//! symbols in the table), then up to 65535 data bytes coded with their own Huffman table.
//! All tiers must write the same block, nothing outside it, and decode it back.
//!
//! Run with `cargo +nightly fuzz run pivco_roundtrip -- -max_total_time=60`.

#![no_main]

use std::cell::RefCell;

use libfuzzer_sys::fuzz_target;
use minlz::pivco::{
    Decoder, Encoder, Error, MAX_BLOCK_SIZE, MAX_CODE_LEN, Table, build_lengths, kernel,
};

#[path = "../../src/pivco/testutil.rs"]
mod testutil;

thread_local! {
    static CODERS: RefCell<Vec<testutil::Coder>> = RefCell::new(testutil::coders());
}

fuzz_target!(|data: &[u8]| {
    CODERS.with_borrow_mut(|cs| testutil::fuzz_round_trip_block(cs, data));
});
