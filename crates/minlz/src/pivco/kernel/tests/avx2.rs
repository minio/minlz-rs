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

//! Port of Go's `kernel/avx2_test.go`. Go's TestAVX2AsmISA scans the generated assembly;
//! here the compiler emits only the enabled `avx2,bmi2,popcnt` features unless the crate is
//! built for an AVX-512 target CPU. To check a default build, emit its assembly
//! (`cargo rustc --release -p minlz --lib -- --emit asm`) and search the
//! `pivco::kernel::avx2` functions for `zmm`, `%k0`..`%k7`, `xmm16`..`ymm31` and `{`
//! (EVEX masking and broadcast): there are none.

use super::Tier;
use crate::cpu;

/// Checks that the AVX2 tier is the best one without AVX-512 and that it needs its
/// features.
#[test]
fn avx2_selected() {
    let _d = cpu::Disabled::new(cpu::PIVCO_AVX512 | cpu::PIVCO_HIST_AVX512);
    if !cpu::has(cpu::PIVCO_AVX2) {
        return;
    }
    assert!(
        std::arch::is_x86_feature_detected!("bmi2")
            && std::arch::is_x86_feature_detected!("popcnt"),
        "the AVX2 tier needs BMI2 and POPCNT"
    );
    assert_eq!(Tier::best(), Tier::Avx2, "without AVX-512");
    assert_eq!(
        Tier::tiers(),
        [Tier::Avx2, Tier::Generic],
        "without AVX-512"
    );
    assert!(
        !Tier::Avx2.hist_uses_scratch(),
        "the AVX2 histogram is generic"
    );
    cpu::disable(cpu::PIVCO_AVX2);
    assert_eq!(Tier::best(), Tier::Generic, "without the AVX2 tier");
    assert_eq!(Tier::tiers(), [Tier::Generic], "without the AVX2 tier");
}
