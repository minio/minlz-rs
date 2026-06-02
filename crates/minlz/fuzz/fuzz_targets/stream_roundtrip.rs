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

//! Encoder correctness fuzzer.  Treats input bytes as a *user payload*,
//! encodes them via both single-threaded `Writer` and multi-threaded
//! `MtWriter`, then decodes (matching path) and asserts byte-equality.
//!
//! The first byte is a control byte encoding:
//!   - bits 0-1: encode concurrency (1..=4)
//!   - bits 2-3: decode concurrency (1..=4)
//!   - bits 4-5: compression level (Fastest / Balanced / Smallest)
//!   - bit 6:    block size — 4 KiB (small) or 2 MiB (default)
//!
//! Cap is 9 MiB so the payload crosses ≥ 1 block boundary at
//! `MAX_BLOCK_SIZE` (8 MiB), exercising the multi-block path.

#![no_main]

use std::io::{Read, Write};

use libfuzzer_sys::fuzz_target;
use minlz::Level;
use minlz::stream::{ConcurrentDecode, MtWriterBuilder, Reader, WriterBuilder};

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > 9 * 1024 * 1024 {
        return;
    }
    let mode = data[0];
    let payload = &data[1..];

    let enc_conc = (mode & 0x3) as usize + 1;
    let dec_conc = ((mode >> 2) & 0x3) as usize + 1;
    let level = match (mode >> 4) & 0x3 {
        0 => Level::Fastest,
        1 => Level::Balanced,
        2 => Level::Smallest,
        _ => Level::Balanced,
    };
    let block_size = if mode & 0x40 != 0 {
        4 * 1024
    } else {
        2 * 1024 * 1024
    };

    // 1) Single-threaded encode + decode.
    {
        let mut buf = Vec::new();
        let mut w = WriterBuilder::new()
            .level(level)
            .block_size(block_size)
            .build(&mut buf)
            .unwrap();
        w.write_all(payload).expect("ST write");
        let _ = w.finish().expect("ST finish");
        let mut dec = Vec::with_capacity(payload.len());
        Reader::new(&buf[..])
            .read_to_end(&mut dec)
            .expect("ST decode");
        assert_eq!(dec.as_slice(), payload, "ST round-trip mismatch");
    }
    // 2) Multi-threaded encode + decode.
    {
        let mut w = MtWriterBuilder::new()
            .level(level)
            .block_size(block_size)
            .concurrency(enc_conc)
            .build(Vec::<u8>::new())
            .unwrap();
        w.write_all(payload).expect("MT write");
        let stream = w.finish().expect("MT finish");
        let mut reader = Reader::new(&stream[..]);
        let ConcurrentDecode {
            bytes_written: n,
            writer: dec,
        } = reader
            .decode_concurrent(Vec::<u8>::with_capacity(payload.len()), dec_conc)
            .expect("MT decode");
        assert_eq!(n as usize, payload.len(), "MT length mismatch");
        assert_eq!(dec.as_slice(), payload, "MT round-trip mismatch");
    }
});
