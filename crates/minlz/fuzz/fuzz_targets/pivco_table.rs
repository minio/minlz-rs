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

//! pivco tables: Go's `FuzzTable` (data[0] even: 128 bytes of length nibbles, then a
//! payload round-tripped on every tier) and `FuzzLengths` (data[0] odd: a frequency scale,
//! then (symbol, u16 count) triples) on data[1..].
//!
//! Run with `cargo +nightly fuzz run pivco_table -- -max_total_time=60`.

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
    let Some((&mode, data)) = data.split_first() else {
        return;
    };
    if mode & 1 == 0 {
        CODERS.with_borrow_mut(|cs| testutil::fuzz_table_lens(cs, data));
    } else {
        testutil::fuzz_lengths(data);
    }
});
