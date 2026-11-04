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

//! Clean-room reference decoder of SIMD blocks (`SPEC_SIMD.md`) with its own pivco decoder
//! (`SPEC_PIVCO.md`), ported from Go's `internal/reference/{simd,pivco}.go`.
//!
//! It shares no code with the main implementation, so tests and fuzzers can check one against the
//! other. It is written to be read next to the specs, not to be fast. Errors are messages.

mod pivco;
mod simd;

pub use simd::{SimdWalker, decode_simd_block, decode_simd_block_strict, walk_simd_block};

/// The largest decoded block.
const MAX_BLOCK_SIZE: usize = 8 << 20;

const MAX_CODE_LEN: usize = 11;

const MAX_VARINT_LEN64: usize = 10;

/// Go's `binary.Uvarint`: the value and its size, or `None` where Go returns a size `<= 0`.
fn uvarint(buf: &[u8]) -> Option<(u64, usize)> {
    let mut x = 0u64;
    for (i, &b) in buf.iter().enumerate() {
        if i == MAX_VARINT_LEN64 {
            return None;
        }
        if b < 0x80 {
            if i == MAX_VARINT_LEN64 - 1 && b > 1 {
                return None;
            }
            return Some((x | ((b as u64) << (7 * i)), i + 1));
        }
        x |= ((b & 0x7f) as u64) << (7 * i);
    }
    None
}
