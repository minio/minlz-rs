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

//! Port of Go's `internal/pivco/avx2_test.go`.

use crate::cpu;
use crate::pivco::encode::HIST_DIRECT;
use crate::pivco::kernel::Tier;
use crate::pivco::testutil::*;
use crate::pivco::{Decoder, Encoder};

/// Hides AVX-512 and checks that zero coders then run the AVX2 tier, without histogram
/// scratch.
#[test]
fn zero_coders_avx2() {
    let _d = cpu::Disabled::new(cpu::PIVCO_AVX512 | cpu::PIVCO_HIST_AVX512);
    if !Tier::tiers().contains(&Tier::Avx2) {
        return;
    }
    let (mut e, mut d) = (Encoder::default(), Decoder::default());
    let mut h = [0u64; 256];
    // Shorter inputs are counted without a kernel.
    e.histogram(&mut h, &[0; HIST_DIRECT]);
    // Decoding sets the decoder up before it reads the block.
    let tb = must_table(&lens_of(&dist_of("english")));
    let _ = d.decode(&mut [0], &[], &tb);
    assert!(
        e.tier == Tier::Avx2 && d.tier == Tier::Avx2 && e.hscratch.is_empty(),
        "zero coders: encoder {} (scratch {}), decoder {}",
        e.tier,
        !e.hscratch.is_empty(),
        d.tier
    );
}
