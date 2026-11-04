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

//! Encoder correctness fuzzer (Go's `FuzzStreamEncode`).  Treats input bytes
//! as a *user payload*, encodes them via both single-threaded `Writer` and
//! multi-threaded `MtWriter`, then decodes (matching path) and asserts
//! byte-equality and the writers' byte counts.
//!
//! The first byte is a control byte encoding:
//!   - bits 0-1: encode concurrency (1..=4)
//!   - bits 2-3: decode concurrency (1..=4)
//!   - bits 4-5: write method, Go's four: one write, 1000-byte writes,
//!     `io::copy`'s 8 KiB writes, or a third, a flush (`MtWriter`:
//!     `encode_buffer`) of the next third, a flush and the rest, which gives
//!     short blocks
//!   - bit 6:    block size — 4 KiB (small) or 2 MiB (default)
//!   - bit 7:    1 KiB padding and an appended index
//!
//! The second byte picks the compression level from [`LEVELS`], the classic
//! and the SIMD levels.
//!
//! Cap is 9 MiB so the payload crosses ≥ 1 block boundary at
//! `MAX_BLOCK_SIZE` (8 MiB), exercising the multi-block path.

#![no_main]

use std::io::{self, Read, Write};

use libfuzzer_sys::fuzz_target;
use minlz::Level;
use minlz::stream::{ConcurrentDecode, MtWriterBuilder, Reader, WriterBuilder};

const LEVELS: [Level; 10] = [
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
];

/// Writes `data` to `w` with write method `m`; `mid` writes the middle third of method 3.
fn write<W: Write>(
    w: &mut W,
    data: &[u8],
    m: u8,
    mid: impl FnOnce(&mut W, &[u8]) -> io::Result<()>,
) -> io::Result<()> {
    match m {
        0 => w.write_all(data),
        1 => data.chunks(1000).try_for_each(|c| w.write_all(c)),
        // From a slice, io::copy would make one write_all.
        2 => io::copy(&mut io::Cursor::new(data), w).map(drop),
        _ => {
            let k = data.len() / 3;
            w.write_all(&data[..k])?;
            mid(w, &data[k..2 * k])?;
            w.flush()?;
            w.write_all(&data[2 * k..])
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > 9 * 1024 * 1024 {
        return;
    }
    let mode = data[0];
    let level = LEVELS[data[1] as usize % LEVELS.len()];
    let payload = &data[2..];

    let enc_conc = (mode & 0x3) as usize + 1;
    let dec_conc = ((mode >> 2) & 0x3) as usize + 1;
    let method = (mode >> 4) & 0x3;
    let block_size = if mode & 0x40 != 0 {
        4 * 1024
    } else {
        2 * 1024 * 1024
    };
    let (padding, index) = if mode & 0x80 != 0 {
        (1 << 10, true)
    } else {
        (0, false)
    };
    let check_padding = |stream: &[u8]| {
        if padding != 0 {
            assert_eq!(stream.len() % padding as usize, 0, "padding");
        }
    };

    // 1) Single-threaded encode + decode.
    {
        let mut b = WriterBuilder::new()
            .level(level)
            .block_size(block_size)
            .padding(padding);
        if index {
            b = b.append_index();
        }
        let mut w = b.build(Vec::new()).unwrap();
        write(&mut w, payload, method, |w, third| {
            w.flush()?;
            w.write_all(third)
        })
        .expect("ST write");
        // The writer counts blocks as it emits them.
        w.flush().expect("ST flush");
        assert_eq!(w.written().uncompressed, payload.len() as u64, "ST written");
        let buf = w.finish().expect("ST finish");
        check_padding(&buf);
        let mut dec = Vec::with_capacity(payload.len());
        Reader::new(&buf[..])
            .read_to_end(&mut dec)
            .expect("ST decode");
        assert_eq!(dec.as_slice(), payload, "ST round-trip mismatch");
    }
    // 2) Multi-threaded encode + decode.
    {
        let mut b = MtWriterBuilder::new()
            .level(level)
            .block_size(block_size)
            .padding(padding)
            .concurrency(enc_conc);
        if index {
            b = b.append_index();
        }
        let mut w = b.build(Vec::<u8>::new()).unwrap();
        write(&mut w, payload, method, |w, third| w.encode_buffer(third)).expect("MT write");
        w.flush().expect("MT flush");
        assert_eq!(w.uncompressed_written(), payload.len() as u64, "MT written");
        let stream = w.finish().expect("MT finish");
        check_padding(&stream);
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
