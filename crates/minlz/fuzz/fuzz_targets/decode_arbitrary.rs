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

//! Go's `FuzzDecodeBlock`: decode arbitrary bytes as a block. The decoder must never panic or
//! read out of bounds, a decode into a dst filled with garbage must give the same output, and
//! `decoded_len` must agree with a successful decode. A SIMD block must decode as the reference
//! decoder does, and when that succeeds, on every exec tier.
//!
//! Seeds: blocks, see `RUNBOOK.md` §1.
//!
//! Run with `cargo +nightly fuzz run decode_arbitrary -- -max_total_time=60`.

#![no_main]

mod common;

use libfuzzer_sys::fuzz_target;
use minlz::reference::decode_simd_block;
use minlz::{MAX_BLOCK_SIZE, decode, decoded_len, max_encoded_len};

fuzz_target!(|data: &[u8]| {
    // A block is never larger than max_encoded_len of its output, which is at most MAX_BLOCK_SIZE.
    if data.len() > max_encoded_len(MAX_BLOCK_SIZE).unwrap() {
        return;
    }
    let mut base = Vec::new();
    let res = decode(&mut base, data);
    if res.is_ok() {
        assert_eq!(decoded_len(data).ok(), Some(base.len()), "decoded_len");
    }
    // The capacity decode asked for, so the garbage is not reallocated away. Only successful
    // decodes use it, and failed ones may have reserved a large declared size.
    let new_dst = || Vec::with_capacity(base.capacity());
    if common::is_simd(data) {
        let want = decode_simd_block(data);
        assert_eq!(
            res.is_ok(),
            want.is_ok(),
            "err {res:?}, reference {:?}",
            want.as_ref().err()
        );
        if let Ok(w) = &want {
            assert!(base == *w, "reference mismatch");
            common::decode_tiers(&"simd", data, Ok(w), &mut new_dst(), 0xfe);
        }
    } else if res.is_ok() {
        let mut dst = new_dst();
        common::garbage(&mut dst, 0xfe);
        decode(&mut dst, data).expect("decode into garbage");
        assert!(dst == base, "garbage in dst changed the output");
    }
});
