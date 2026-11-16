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

//! Port of Go's `TestAVX512HistogramFallback` (`kernel/hist_avx512_test.go`). It turns a CPU
//! feature off for the whole process, which would race the other histogram tests, so it
//! needs its own test binary.

use minlz::cpu;
use minlz::pivco::{
    Decoder, Encoder, Error, MAX_BLOCK_SIZE, MAX_CODE_LEN, Table, build_lengths, kernel,
};

#[path = "../src/pivco/testutil.rs"]
mod testutil;

use kernel::Tier;
use testutil::*;

#[test]
fn avx512_histogram_fallback() {
    if !Tier::Avx512.hist_uses_scratch() {
        return; // no AVX-512 GFNI/BITALG
    }
    cpu::disable(cpu::PIVCO_HIST_AVX512);
    assert!(
        !Tier::Avx512.hist_uses_scratch(),
        "scratch reported without GFNI"
    );
    let mut src = vec![0u8; 1000];
    rand_bytes(&mut new_rng(), &mut src);
    let (mut h, mut want) = ([0u32; 256], [0u32; 256]);
    Tier::Generic.histogram(&mut want, &src, &mut []);
    Tier::Avx512.histogram(&mut h, &src, &mut []);
    assert!(h == want, "fallback histogram mismatch");
    cpu::enable_all();
}
