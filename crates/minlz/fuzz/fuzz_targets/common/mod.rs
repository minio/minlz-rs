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

//! Checks shared by the block targets. ASan's redzones around libFuzzer's exact-size input
//! and around exact-capacity outputs stand in for the guard pages of Go's fuzzers.

// Each target uses a subset.
#![allow(dead_code)]

use std::fmt::Debug;

use minlz::cpu;

/// Reports whether `b` has a SIMD block header: a zero byte, then a size varint of at least
/// 1 << 24, as the decoder tells them apart.
pub fn is_simd(b: &[u8]) -> bool {
    let mut v = 0u64;
    for (i, &c) in b.iter().skip(1).take(10).enumerate() {
        v |= u64::from(c & 0x7f) << (7 * i);
        if c < 0x80 {
            return b[0] == 0 && v >= 1 << 24;
        }
    }
    false
}

/// Fills `dst` with `fill` to its capacity, so output must not depend on what dst held, as Go's
/// fuzzers fill dst before decoding. `decode` decodes SIMD blocks over the fill, and classic
/// blocks into the memory under it; `encode` zeroes its output first.
pub fn garbage(dst: &mut Vec<u8>, fill: u8) {
    dst.clear();
    dst.resize(dst.capacity(), fill);
}

/// Decodes `block` into `dst` on every exec tier, each with the pivco tier of its instruction
/// set, which must all return `want`. `ctx` names the case in failures.
pub fn decode_tiers(
    ctx: &dyn Debug,
    block: &[u8],
    want: Result<&[u8], &str>,
    dst: &mut Vec<u8>,
    fill: u8,
) {
    cpu::fuzz_tiers(|tiers| {
        garbage(dst, fill);
        let got = minlz::decode(dst, block);
        assert_eq!(
            got.is_ok(),
            want.is_ok(),
            "{ctx:?}, tiers {tiers:?}: err {got:?}, want err {:?}",
            want.err()
        );
        if let Ok(w) = want {
            assert!(dst[..] == *w, "{ctx:?}, tiers {tiers:?}: output mismatch");
        }
    });
}
