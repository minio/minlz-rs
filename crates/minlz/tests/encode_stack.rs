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

//! Encoding must fit a small thread stack: the encoders keep only hash tables of up to 16 KiB on
//! the stack and the rest of their state on the heap. A stack overflow aborts the process, so
//! this needs its own test binary.

use minlz::{Level, encode};

#[test]
fn encode_fits_small_stack() {
    let src: Vec<u8> = (0u32..256 << 10)
        .map(|i| (i.wrapping_mul(2654435761) >> 29) as u8 ^ (i as u8 & 0x0f))
        .collect();
    // A new thread, so the encoder state is not pooled yet. On x86_64 Linux every level needs at
    // most 36 KiB in debug and 28 KiB in release builds, and a 64 KiB table on the stack 72 KiB
    // or more. Windows rounds the size up to 128 KiB.
    std::thread::Builder::new()
        .stack_size(68 << 10)
        .spawn(move || {
            for l in [
                Level::SuperFast,
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
            ] {
                // Up to 64 KiB and above it the encoders use different tables.
                for n in [64 << 10, src.len()] {
                    encode(&mut Vec::new(), &src[..n], l).unwrap();
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
