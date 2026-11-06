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

//! Go's `FuzzDecodeSIMD`: data[..4] is a little-endian `extra`, data[4..] a SIMD block body
//! from its flags byte. The block gets a valid header of `body.len() + 4 + 64 + extra` bytes, and
//! every exec tier must decode it as the reference decoder does. dst has exactly the decoded
//! size as capacity, so writes past it are heap overflows under ASan, and is filled with garbage
//! that varies with `extra` first.
//!
//! Seeds: SIMD blocks as `extra` and body, see `RUNBOOK.md` §1.
//!
//! Run with `cargo +nightly fuzz run simd_decode -- -max_total_time=60`.

#![no_main]

mod common;

use libfuzzer_sys::fuzz_target;
use minlz::reference::decode_simd_block;

/// Larger blocks only cost memory: every check is per chunk or per literal block.
const MAX_N: usize = 1 << 20;

fuzz_target!(|data: &[u8]| {
    let Some((extra, body)) = data.split_first_chunk::<4>() else {
        return;
    };
    let n = body.len() + 4 + 64 + u32::from_le_bytes(*extra) as usize;
    if body.is_empty() || n > MAX_N {
        return;
    }
    let mut src = vec![0];
    let mut v = 1 << 24 | n as u64;
    while v >= 0x80 {
        src.push(v as u8 | 0x80);
        v >>= 7;
    }
    src.push(v as u8);
    src.extend_from_slice(body);
    let want = decode_simd_block(&src);
    let want = want.as_deref().map_err(String::as_str);
    let mut dst = Vec::with_capacity(n);
    common::decode_tiers(&"reference", &src, want, &mut dst, 0xfe ^ extra[0]);
});
