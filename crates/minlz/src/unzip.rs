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

//! A minimal zip reader for test data: the SIMD fixtures of the Go encoder and Go's seed zips.
//! The integration tests include this file with `#[path]`.

use std::io::Read;

/// The files of a zip archive, stored or deflated, as (name, contents).
pub fn unzip(z: &[u8]) -> Vec<(String, Vec<u8>)> {
    let u16_at = |i: usize| u16::from_le_bytes([z[i], z[i + 1]]) as usize;
    let u32_at = |i: usize| u32::from_le_bytes(z[i..i + 4].try_into().unwrap()) as usize;
    // The end of central directory record, before the archive comment.
    let end = (0..=z.len() - 22)
        .rev()
        .find(|&i| u32_at(i) == 0x0605_4b50)
        .expect("zip end record");
    let mut p = u32_at(end + 16);
    let mut out = Vec::new();
    for _ in 0..u16_at(end + 10) {
        assert_eq!(u32_at(p), 0x0201_4b50, "zip central directory");
        let (method, size, name) = (u16_at(p + 10), u32_at(p + 20), u16_at(p + 28));
        let local = u32_at(p + 42);
        let start = local + 30 + u16_at(local + 26) + u16_at(local + 28);
        let data = &z[start..start + size];
        let name_bytes = &z[p + 46..p + 46 + name];
        if !name_bytes.ends_with(b"/") {
            let contents = match method {
                0 => data.to_vec(),
                8 => {
                    let mut b = Vec::new();
                    flate2::read::DeflateDecoder::new(data)
                        .read_to_end(&mut b)
                        .unwrap();
                    b
                }
                m => panic!("zip method {m}"),
            };
            out.push((String::from_utf8_lossy(name_bytes).into_owned(), contents));
        }
        p += 46 + name + u16_at(p + 30) + u16_at(p + 32);
    }
    out
}

/// The files of `tests/fixtures/simd.zip`: the SIMD blocks of the Go encoder,
/// `<input>.L<level>.mzb`, and their `manifest.txt`, written by `_helpers/simdgo`.
pub fn simd_fixtures() -> Vec<(String, Vec<u8>)> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/simd.zip");
    unzip(&std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}")))
}
