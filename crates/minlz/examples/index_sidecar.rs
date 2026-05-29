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

//! Build an index for a stream that doesn't carry one, store the index
//! separately (sidecar), and use it over a non-seekable reader.
//!
//! Run with `cargo run --release --example index_sidecar`.
//!
//! Demonstrates:
//! * `index_stream(r)` to construct an index from a stream that was
//!   written without `append_index()`.
//! * `remove_index_headers` / `restore_index_headers` to ship the
//!   index in compact form (saves ~20 bytes of envelope).
//! * `Reader::skip` paired with `ReaderBuilder::ignore_stream_id()`
//!   for the recipe used when the input can *only* `Read` (no
//!   `Seek`) — what you'd do over an HTTP body, a pipe, or any
//!   sequential source.
//!
//! Mirrors Go's `ExampleIndexStream`.

use std::io::{Cursor, Read, Write};

use minlz::index::{index_stream, remove_index_headers, restore_index_headers};
use minlz::stream::{ReaderBuilder, WriterBuilder};
use minlz::Index;

fn main() {
    // -------------------- 1. Produce sample data --------------------
    let payload = sample_payload(2 << 20);

    // -------------------- 2. Compress without an appended index --------------------
    //
    // `.generate_index(false)` skips even the in-memory bookkeeping —
    // useful when you know you'll build the index later or never need
    // one.  We could also leave generation on but skip `append_index()`.
    let mut compressed: Vec<u8> = Vec::new();
    let mut w = WriterBuilder::new()
        .block_size(64 << 10)
        .generate_index(false)
        .build(&mut compressed);
    w.write_all(&payload).expect("write");
    let _ = w.finish().expect("finish");
    println!(
        "compressed (no index): {} -> {} bytes",
        payload.len(),
        compressed.len(),
    );

    // -------------------- 3. Build the index after the fact --------------------
    //
    // `index_stream` walks the compressed bytes, parses chunk headers
    // only (no block decoding), and emits a self-contained 0x40 chunk.
    // The caller decides what to do with those bytes.
    let idx_chunk = index_stream(Cursor::new(&compressed)).expect("index_stream");
    println!("index chunk: {} bytes", idx_chunk.len());

    // -------------------- 4. Optional: trim the envelope --------------------
    //
    // `remove_index_headers` strips the 0x40 framing + trailing size +
    // trailer to save ~20 bytes when persisting the index in a sidecar
    // (e.g. database row, separate file).  `restore_index_headers` is
    // the inverse for re-parsing later.
    let slim = remove_index_headers(&idx_chunk).expect("remove_index_headers");
    println!(
        "sidecar slim form: {} bytes (saved {})",
        slim.len(),
        idx_chunk.len() - slim.len(),
    );

    // -------------------- 5. Use the index over a non-seekable reader --------------------
    //
    // Imagine the compressed stream is being served over a one-shot
    // source (HTTP body, pipe, …) that cannot `Seek`.  The standard
    // recipe:
    //   a. Restore + load the index.
    //   b. Look up the (compressed_offset, uncompressed_offset) pair
    //      for the byte position you want.
    //   c. Position the input source at `compressed_offset` (range
    //      request, fseek before passing to Reader, etc.).
    //   d. Build a Reader with `ignore_stream_id()` — the stream now
    //      starts mid-frame, so the stream identifier is absent.
    //   e. `Reader::skip(target - uncompressed_offset)` to land at the
    //      exact byte; subsequent reads return the payload.
    let restored = restore_index_headers(slim);
    let mut idx = Index::default();
    idx.load(&restored).expect("Index::load");

    let want_off: u64 = 1_000_000;
    let (c_off, u_off) = idx.find(want_off as i64).expect("find");
    println!("Index::find({want_off}) -> comp_off={c_off} uncomp_off={u_off}");

    // Simulate "position the source at c_off" by slicing the bytes.
    let mut reader = ReaderBuilder::new()
        .ignore_stream_id()
        .build(Cursor::new(&compressed[c_off as usize..]));
    reader.skip(want_off - u_off as u64).expect("skip");

    // Read a few bytes from the requested position.
    let mut got = [0u8; 64];
    reader.read_exact(&mut got).expect("read");
    assert_eq!(&got, &payload[want_off as usize..want_off as usize + 64]);
    println!("read 64 bytes at uncompressed offset {want_off} — matches source");
}

fn sample_payload(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut x = 0xc0ffee_u64;
    while out.len() < n {
        x = x.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
        out.push(((x >> 24) as u8) & 0x3f);
    }
    out
}
