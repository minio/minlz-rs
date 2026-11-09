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

//! Encoding must fit a small thread stack: the encoders keep their large state on the heap. A
//! stack overflow aborts the process, so this needs its own test binary.

use minlz::{Level, encode};

#[test]
fn encode_fits_small_stack() {
    let src: Vec<u8> = (0u32..64 << 10)
        .map(|i| (i.wrapping_mul(2654435761) >> 29) as u8 ^ (i as u8 & 0x0f))
        .collect();
    // A new thread, so the encoder state is not pooled yet. Every level needs under 96 KiB in
    // debug and release builds.
    std::thread::Builder::new()
        .stack_size(128 << 10)
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
                encode(&mut Vec::new(), &src, l).unwrap();
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
