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

//! Go's `FuzzEncodingBlocks`: encode arbitrary bytes (up to `MAX_BLOCK_SIZE`) at every level,
//! classic and SIMD, and decode them back. A SIMD block must decode on every exec tier and with
//! the strict reference decoder, and no block may exceed `max_encoded_len`. As in Go, outputs
//! are filled with garbage first. Each has exactly the capacity its coder asks for, so ASan
//! flags writes past it.
//!
//! Seeds: raw inputs, see `RUNBOOK.md` §1.
//!
//! Run with `cargo +nightly fuzz run roundtrip -- -max_total_time=60`.

#![no_main]

mod common;

use libfuzzer_sys::fuzz_target;
use minlz::reference::decode_simd_block_strict;
use minlz::{Level, encode, max_encoded_len};

const LEVELS: [Level; 10] = [
    Level::Fastest,
    Level::Balanced,
    Level::Smallest,
    Level::SuperFastSimd,
    Level::FastestSimd,
    Level::BalancedSimd,
    Level::SmallestSimd,
    Level::SuperFastSimdDelta,
    Level::FastestSimdDelta,
    Level::BalancedSimdDelta,
];

fuzz_target!(|data: &[u8]| {
    if data.len() > minlz::MAX_BLOCK_SIZE {
        return;
    }
    let mel = max_encoded_len(data.len()).unwrap();
    let mut enc = Vec::with_capacity(mel);
    // The SIMD decoder writes exactly the output, so this never grows.
    let mut sdec = Vec::with_capacity(data.len());
    for level in LEVELS {
        common::garbage(&mut enc, 0xff);
        encode(&mut enc, data, level).expect("encode");
        assert!(
            enc.len() <= mel,
            "{level:?}: {} bytes, max_encoded_len {mel}",
            enc.len()
        );
        if common::is_simd(&enc) {
            common::decode_tiers(&level, &enc, Ok(data), &mut sdec, 0xfe);
            let r = decode_simd_block_strict(&enc);
            assert!(
                r.as_deref() == Ok(data),
                "{level:?}: reference {:?}",
                r.err()
            );
        } else {
            // The classic decoder reserves room to overshoot, which would double a dst sized for
            // the output. A new one gets exactly that room; the second decode finds garbage in it.
            let mut dec = Vec::new();
            for _ in 0..2 {
                common::garbage(&mut dec, 0xfe);
                minlz::decode(&mut dec, &enc).expect("decode");
                assert!(dec == data, "{level:?}: mismatch");
            }
        }
    }
});
