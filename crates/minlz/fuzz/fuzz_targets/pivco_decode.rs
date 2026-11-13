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

//! pivco block decoding on every kernel tier: Go's `FuzzDecodeBlock` (data[0] even) and
//! `FuzzDecodeBlockFixed` (data[0] odd) on data[1..]. Tiers must agree, errors must be
//! `Corrupt` without writing outside dst, and an accepted block must be what the encoder
//! writes for its output.
//!
//! Run with `cargo +nightly fuzz run pivco_decode -- -max_total_time=60`.

#![no_main]

use std::cell::RefCell;

use libfuzzer_sys::fuzz_target;
use minlz::pivco::{
    Decoder, Encoder, Error, MAX_BLOCK_SIZE, MAX_CODE_LEN, Table, build_lengths, kernel,
};

#[path = "../../src/pivco/testutil.rs"]
mod testutil;

thread_local! {
    static STATE: RefCell<(Vec<testutil::Coder>, Vec<Table>)> =
        RefCell::new((testutil::coders(), testutil::fixed_tables()));
}

fuzz_target!(|data: &[u8]| {
    let Some((&mode, data)) = data.split_first() else {
        return;
    };
    STATE.with_borrow_mut(|(cs, tabs)| {
        if mode & 1 == 0 {
            testutil::fuzz_decode_block(cs, data);
        } else {
            testutil::fuzz_decode_block_fixed(cs, tabs, data);
        }
    });
});
