//! Decoder robustness fuzzer.  Feeds arbitrary bytes to both the
//! single-threaded `Reader::read_to_end` and the multi-threaded
//! `Reader::decode_concurrent`.  Either must return `Ok` or `Err` —
//! never panic / OOB-read / deadlock.
//!
//! The first byte picks the worker count for the MT path.  The rest is
//! fed verbatim as a potential MinLZ stream.  Cap is 9 MiB so libfuzzer
//! mutations can reach the > MAX_BLOCK_SIZE multi-block territory when
//! `-max_len` is set accordingly.
//!
//! Corpus: seeded from the upstream tarball; see fuzz/RUNBOOK.md §1.

#![no_main]

use std::io::Read;
use std::time::{Duration, Instant};

use libfuzzer_sys::fuzz_target;
use minlz::stream::Reader;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > 9 * 1024 * 1024 {
        return;
    }
    let threads = ((data[0] & 0x3) as usize) + 1; // 1..=4
    let payload = &data[1..];

    // 1) Single-threaded decoder.
    {
        let mut out = Vec::new();
        let _ = Reader::new(payload).read_to_end(&mut out);
    }
    // 2) Multi-threaded decoder, with a wall-clock guard to catch
    //    pipeline deadlocks (libfuzzer's own 25s timeout also fires).
    {
        let start = Instant::now();
        let mut reader = Reader::new(payload);
        let _ = reader.decode_concurrent(std::io::sink(), threads);
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "MT decode_concurrent ran > 10s on payload len={} — possible deadlock",
            payload.len()
        );
    }
});
