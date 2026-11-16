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

//! Histograms past the u32 range of the kernels.
#![cfg(target_pointer_width = "64")]

#[test]
#[ignore = "needs 4 GiB of RAM"]
fn histogram_4gib() {
    let src = vec![7u8; 1 << 32];
    let mut h = [0u64; 256];
    minlz::pivco::Encoder::default().histogram(&mut h, &src);
    assert_eq!(h[7], 1 << 32);
}
