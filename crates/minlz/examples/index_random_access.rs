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

//! Random-access reads over a MinLZ-compressed stream using an
//! appended index.
//!
//! Run with `cargo run --release --example index_random_access`.
//!
//! Demonstrates:
//! * Compressing with `WriterBuilder::append_index()` so the index chunk
//!   is written to the end of the stream.
//! * Wrapping the resulting `Read + Seek` source in `ReadSeeker` to do
//!   `read_at` / `Seek::seek` by *uncompressed* offset.
//!
//! Mirrors Go's `ExampleIndex_Load`-style usage when the input *can*
//! seek (see `index_sidecar.rs` for the non-seekable path).

use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::time::Instant;

use minlz::stream::{ReadSeeker, Reader, WriterBuilder};

const PAYLOAD_BYTES: usize = 100 << 20; // 100 MiB

fn main() {
    // -------------------- 1. Produce sample data --------------------
    //
    // 100 MiB of pseudo-text bytes (alphabet of 4 → highly compressible).
    // Big enough that the index actually has multiple entries to
    // traverse on a `find()` lookup.
    let t = Instant::now();
    let payload = sample_payload(PAYLOAD_BYTES);
    println!(
        "generated {} MiB payload in {:.2?}",
        payload.len() >> 20,
        t.elapsed()
    );

    // -------------------- 2. Compress with an index --------------------
    //
    // `Index::reset()` floors `est_block_uncomp` to 1 MiB regardless of
    // the writer's actual `block_size`, so for a 100 MiB stream we get
    // ~100 index entries — plenty for the binary-search lookup path
    // (`find` switches from linear to `partition_point` above 200
    // entries; well above that for larger payloads).
    let t = Instant::now();
    let mut compressed: Vec<u8> = Vec::with_capacity(payload.len());
    let mut w = WriterBuilder::new()
        .block_size(32 << 10) // 32 KiB: small enough that Reader::skip's
        // whole-block fast path still applies, but not so small that
        // per-chunk overhead dominates (4 KiB minimum is excessive).
        .append_index() // write the index chunk after EOF
        .build(&mut compressed);
    w.write_all(&payload).expect("write");
    let _ = w.finish().expect("finish");
    let enc_elapsed = t.elapsed();
    println!(
        "compressed {} MiB -> {} MiB ({:.1}%) in {:.2?}  [{:.1} MiB/s]",
        payload.len() >> 20,
        compressed.len() >> 20,
        100.0 * compressed.len() as f64 / payload.len() as f64,
        enc_elapsed,
        (payload.len() as f64 / (1 << 20) as f64) / enc_elapsed.as_secs_f64(),
    );

    // -------------------- 3. Open with ReadSeeker --------------------
    let t = Instant::now();
    let reader = Reader::new(Cursor::new(compressed.clone()));
    let mut rs = ReadSeeker::new(reader, &[]).expect("ReadSeeker::new");
    let open_elapsed = t.elapsed();
    let idx = rs.index();
    println!(
        "opened ReadSeeker in {:.2?}  (index: {} entries, total_uncompressed={}, total_compressed={})",
        open_elapsed,
        idx.offsets.len(),
        idx.total_uncompressed,
        idx.total_compressed,
    );

    // -------------------- 4. Random reads (timed) --------------------
    //
    // `read_at` seeks via the index, then reads `buf.len()` bytes.
    // Each call is independent; per-call latency reflects:
    //   * index lookup (O(log n) for >200 entries),
    //   * underlying `Seek::seek` on the source,
    //   * decoding *one* block (the one containing `offset`),
    //   * copying `buf.len()` bytes.
    let mut buf = [0u8; 4096];
    let probes: &[u64] = &[
        0,
        1 << 20,  // 1 MiB
        50 << 20, // middle
        (PAYLOAD_BYTES as u64) - 4096,
        1234567,  // arbitrary mid-stream
        99 << 20, // near the end
    ];
    for &off in probes {
        let t = Instant::now();
        let n = rs.read_at(&mut buf, off).expect("read_at");
        let dt = t.elapsed();
        let want =
            &payload[off as usize..off as usize + buf.len().min(payload.len() - off as usize)];
        assert_eq!(&buf[..n], want, "off={off}");
        println!("  read_at({off:>10}) = {n:>4} bytes  in {dt:>10.2?}");
    }

    // -------------------- 5. Seek + sequential read --------------------
    //
    // `ReadSeeker` impls `Seek`, so once positioned the reader streams
    // forward at full decode bandwidth.  Compare:
    //   * a cold seek to mid-stream + 1 MiB sequential read
    //   * tail seek (`SeekFrom::End`) + 32 KiB read
    let mut chunk = vec![0u8; 1 << 20];
    let t = Instant::now();
    rs.seek(SeekFrom::Start(50 << 20)).expect("seek mid");
    let seek_dt = t.elapsed();
    let t = Instant::now();
    rs.read_exact(&mut chunk).expect("read 1 MiB");
    let read_dt = t.elapsed();
    println!(
        "seek to 50 MiB in {:.2?}; then read 1 MiB in {:.2?}  [{:.1} MiB/s]",
        seek_dt,
        read_dt,
        1.0 / read_dt.as_secs_f64(),
    );

    let mut tail = [0u8; 32 << 10];
    let t = Instant::now();
    rs.seek(SeekFrom::End(-(tail.len() as i64)))
        .expect("seek end");
    rs.read_exact(&mut tail).expect("read tail");
    let tail_dt = t.elapsed();
    assert_eq!(&tail[..], &payload[payload.len() - tail.len()..]);
    println!("SeekFrom::End(-32K) + read in {:.2?}", tail_dt);
}

fn sample_payload(n: usize) -> Vec<u8> {
    // Pseudo-text payload — a small alphabet repeated through an LCG
    // mod 4, which is highly compressible (matches Go's `expand` helper
    // in `index_test.go`).
    let alphabet = b"0123";
    let mut out = Vec::with_capacity(n);
    let mut x = 0xdeadbeef_u64;
    while out.len() < n {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.push(alphabet[((x >> 33) as usize) & 3]);
    }
    out
}
