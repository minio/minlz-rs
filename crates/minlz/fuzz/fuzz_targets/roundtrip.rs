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

//! Encode arbitrary bytes at every level, decode, assert equality.
//!
//! Seed corpus from `testdata/fuzz/block-corpus-enc.zip` in the Go repo.

#![no_main]

use libfuzzer_sys::fuzz_target;
use minlz::{decode, encode, Level};

fuzz_target!(|data: &[u8]| {
    // Block codec only handles inputs up to `MAX_BLOCK_SIZE` (8 MiB).
    // Cap at that; the stream fuzzer (`stream_decode_arbitrary`) covers
    // larger inputs via the multi-block path.
    if data.len() > minlz::MAX_BLOCK_SIZE {
        return;
    }
    for level in [Level::Fastest, Level::Balanced, Level::Smallest] {
        let mut enc = Vec::new();
        encode(&mut enc, data, level).expect("encode");
        let mut dec = Vec::new();
        decode(&mut dec, &enc).expect("decode");
        assert_eq!(dec.as_slice(), data, "level {level:?}");
    }
});
