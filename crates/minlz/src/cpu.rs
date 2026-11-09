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

//! CPU features the SIMD tiers dispatch on (Go `internal/cpu` plus the pivco kernel checks).
//!
//! A tier is only used when the CPU has every feature its `#[target_feature]` functions
//! enable, so calling them is sound. Tests and tools may [`disable`] tiers to run the lower
//! ones; disabling can only remove tiers, never add them. Miri always gets the generic code.

use core::sync::atomic::{AtomicU32, Ordering};

/// The SIMD exec loops and stats on x86-64: AVX2 and BMI2 (Go `cpu.AVX2`).
pub const AVX2: u32 = 1;
/// The AVX-512 exec loops and stats (Go `cpu.AVX512`).
pub const AVX512: u32 = 2;
/// The NEON exec loops and kernels on aarch64 (Go `cpu.NEON`).
pub const NEON: u32 = 4;
/// The pivco AVX2 tier (Go kernel `hasAVX2`).
pub const PIVCO_AVX2: u32 = 8;
/// The pivco AVX-512 tier (Go kernel `hasAVX512`).
pub const PIVCO_AVX512: u32 = 16;
/// The pivco AVX-512 GFNI/BITALG histogram (Go kernel `hasHistAVX512`).
pub const PIVCO_HIST_AVX512: u32 = 32;

static DISABLED: AtomicU32 = AtomicU32::new(0);

/// Turns off the features in `mask` (a union of the constants above) until [`enable_all`].
pub fn disable(mask: u32) {
    DISABLED.fetch_or(mask, Ordering::Relaxed);
}

/// Turns every detected feature back on.
pub fn enable_all() {
    DISABLED.store(0, Ordering::Relaxed);
}

/// Reports whether all features in `mask` are present and not disabled.
#[inline]
pub fn has(mask: u32) -> bool {
    detected() & mask == mask && DISABLED.load(Ordering::Relaxed) & mask == 0
}

/// Runs f once per SIMD exec tier this CPU has, with the SIMD block decodes of this thread forced
/// to it and to the pivco tier of the same instruction set, and passes their names. It panics if
/// the last pivco block f decoded was set to another pivco tier. Only fuzz builds have it: a
/// decoder keeps the pivco tier it took on first use, so disabling features can't select one.
#[cfg(fuzzing)]
pub fn fuzz_tiers(mut f: impl FnMut(&dyn core::fmt::Debug)) {
    crate::block::simd::fuzz_tiers(|t, p| f(&(t, p)));
}

/// Disables features for a test until dropped. Holders run one at a time, so a test that
/// checks tier selection is not re-enabled by another; tests that only use tiers may see
/// fewer while one is held, which they tolerate.
#[cfg(test)]
// Only the SIMD tier tests (x86_64 and little-endian aarch64) use it.
#[cfg_attr(not(all(target_arch = "x86_64", not(miri))), allow(dead_code))]
pub(crate) struct Disabled {
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
#[cfg_attr(not(all(target_arch = "x86_64", not(miri))), allow(dead_code))]
impl Disabled {
    pub(crate) fn new(mask: u32) -> Disabled {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        disable(mask);
        Disabled { _lock }
    }
}

#[cfg(test)]
impl Drop for Disabled {
    fn drop(&mut self) {
        enable_all();
    }
}

#[inline]
fn detected() -> u32 {
    use std::sync::OnceLock;
    static D: OnceLock<u32> = OnceLock::new();
    *D.get_or_init(detect)
}

#[cfg(all(target_arch = "x86_64", not(miri)))]
fn detect() -> u32 {
    use std::arch::is_x86_feature_detected as f;
    let mut m = 0;
    if f!("avx2") && f!("bmi2") {
        m |= AVX2;
        if f!("avx512f")
            && f!("avx512bw")
            && f!("avx512vl")
            && f!("avx512vbmi")
            && f!("avx512vbmi2")
            && f!("popcnt")
        {
            m |= AVX512;
        }
    }
    if f!("avx2") && f!("bmi2") && f!("popcnt") {
        m |= PIVCO_AVX2;
    }
    if m & AVX512 != 0 && f!("avx512vpopcntdq") {
        m |= PIVCO_AVX512;
        if f!("gfni") && f!("avx512bitalg") {
            m |= PIVCO_HIST_AVX512;
        }
    }
    m
}

#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
fn detect() -> u32 {
    // ASIMD is part of ARMv8.
    NEON
}

#[cfg(any(
    miri,
    not(any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_endian = "little")
    ))
))]
fn detect() -> u32 {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_nest() {
        let d = detected();
        if d & AVX512 != 0 {
            assert!(d & AVX2 != 0);
        }
        if d & PIVCO_HIST_AVX512 != 0 {
            assert!(d & PIVCO_AVX512 != 0);
        }
        if d & PIVCO_AVX512 != 0 {
            assert!(d & PIVCO_AVX2 != 0 && d & AVX512 != 0);
        }
    }
}
