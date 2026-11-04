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

//! Decoder robustness fuzzer.  Feeds arbitrary bytes to both the
//! single-threaded `Reader::read_to_end` and the multi-threaded
//! `Reader::decode_concurrent`.  Either must return `Ok` or `Err` —
//! never panic / OOB-read / deadlock — and, as Go's `FuzzStreamDecode`,
//! they agree on success and output.  On streams that decode, `skip` to
//! the middle continues as the full decode does.
//!
//! The first byte picks the worker count for the MT path (bits 0-1) and
//! whether checksums and hashes are ignored (bit 2), which lets mutated
//! chunks reach the block decoders.  The rest is fed verbatim as a
//! potential MinLZ stream.  Cap is 9 MiB so libfuzzer mutations can reach
//! the > MAX_BLOCK_SIZE multi-block territory when `-max_len` is set
//! accordingly.
//!
//! Corpus: seeded from the upstream tarball; see fuzz/RUNBOOK.md §1.

#![no_main]

use std::io::Read;
use std::time::{Duration, Instant};

use libfuzzer_sys::fuzz_target;
use minlz::stream::ReaderBuilder;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > 9 * 1024 * 1024 {
        return;
    }
    let threads = ((data[0] & 0x3) as usize) + 1; // 1..=4
    let ignore_crc = data[0] & 0x4 != 0;
    let payload = &data[1..];
    let reader = || {
        let b = ReaderBuilder::new();
        let b = if ignore_crc { b.ignore_crc() } else { b };
        b.build(payload).unwrap()
    };

    // 1) Single-threaded decoder.
    let mut out = Vec::new();
    let read = reader().read_to_end(&mut out);
    // 2) Multi-threaded decoder, with a wall-clock guard to catch
    //    pipeline deadlocks (libfuzzer's own 25s timeout also fires).
    let start = Instant::now();
    let conc = reader().decode_concurrent(Vec::new(), threads);
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "MT decode_concurrent ran > 10s on payload len={} — possible deadlock",
        payload.len()
    );
    assert_eq!(read.is_ok(), conc.is_ok(), "{read:?} vs {conc:?}");
    let Ok(conc) = conc else { return };
    assert!(
        conc.writer == out,
        "Read and decode_concurrent output differ"
    );

    // 3) Skip doesn't decode skipped blocks, so it only must match Read on valid streams.
    let skip = payload.len() / 2;
    let mut r = reader();
    let res = r.skip(skip as u64);
    if skip > out.len() {
        assert!(res.is_err(), "skip past the end");
        return;
    }
    res.expect("skip");
    let mut tail = Vec::new();
    r.read_to_end(&mut tail).expect("read after skip");
    assert!(tail == out[skip..], "skip mismatch");
});
