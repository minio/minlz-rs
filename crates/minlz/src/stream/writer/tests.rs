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

use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};

use crate::index::{Index, index_stream};
use crate::stream::crc::masked_crc32c;
use crate::stream::format::{
    CHECKSUM_SIZE, CHUNK_TYPE_MINLZ_COMPRESSED_DATA, CHUNK_TYPE_MINLZ_SIMD,
    CHUNK_TYPE_UNCOMPRESSED_DATA, MAGIC_BODY_LEN, SIMD_HASH_SIZE, STREAM_FLAG_SIMD,
};
use crate::stream::{
    MIN_USER_SKIPPABLE_CHUNK, MtWriter, MtWriterBuilder, ReadSeeker, Reader, ReaderBuilder, Writer,
    WriterBuilder,
};
use crate::{Level, try_encode, xxh3};

fn round_trip(payload: &[u8]) -> Vec<u8> {
    let buf: Vec<u8> = Vec::new();
    let mut w = Writer::new(buf);
    w.write_all(payload).unwrap();
    let out = w.finish().unwrap();
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&out))
        .read_to_end(&mut decoded)
        .unwrap();
    decoded
}

#[test]
fn empty_round_trip() {
    let got = round_trip(b"");
    assert!(got.is_empty());
}

#[test]
fn small_payload_round_trip() {
    let got = round_trip(b"Hello, MinLZ!");
    assert_eq!(got, b"Hello, MinLZ!");
}

#[test]
fn highly_compressible_round_trip() {
    let payload = vec![b'a'; 10_000];
    let got = round_trip(&payload);
    assert_eq!(got, payload);
}

#[test]
fn incompressible_round_trip() {
    // Pseudo-random data that won't compress.
    let mut payload = Vec::with_capacity(2048);
    let mut x: u32 = 0x1234_5678;
    for _ in 0..2048 {
        x = x.wrapping_mul(1664525).wrapping_add(1013904223);
        payload.push((x >> 16) as u8);
    }
    let got = round_trip(&payload);
    assert_eq!(got, payload);
}

#[test]
fn multi_block_payload_round_trip() {
    // Force at least 2 blocks at 4 KiB block size.
    let payload: Vec<u8> = (0..16 * 1024).map(|i| (i % 256) as u8).collect();
    let buf: Vec<u8> = Vec::new();
    let mut w = WriterBuilder::new().block_size(4 << 10).build(buf).unwrap();
    w.write_all(&payload).unwrap();
    let out = w.finish().unwrap();
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&out))
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn uncompressed_mode_round_trip() {
    let payload = vec![b'b'; 5000];
    let buf: Vec<u8> = Vec::new();
    let mut w = WriterBuilder::new().uncompressed().build(buf).unwrap();
    w.write_all(&payload).unwrap();
    let out = w.finish().unwrap();
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&out))
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn flush_on_write_emits_one_chunk_per_write() {
    let buf: Vec<u8> = Vec::new();
    let mut w = WriterBuilder::new().flush_on_write().build(buf).unwrap();
    w.write_all(b"a").unwrap();
    w.write_all(b"b").unwrap();
    w.write_all(b"c").unwrap();
    let out = w.finish().unwrap();
    // Each write produces a separate chunk; reader should still round-trip.
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&out))
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, b"abc");
}

#[test]
fn add_user_chunk_round_trips_via_callback() {
    use std::sync::Arc;
    use std::sync::Mutex;
    let buf: Vec<u8> = Vec::new();
    let mut w = Writer::new(buf);
    w.write_all(b"prefix").unwrap();
    w.add_user_chunk(0x80, b"meta-a").unwrap();
    w.write_all(b"infix").unwrap();
    w.add_user_chunk(0xc0, b"meta-b").unwrap();
    w.write_all(b"suffix").unwrap();
    let out = w.finish().unwrap();

    type Seen = Arc<Mutex<Vec<(u8, Vec<u8>)>>>;
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let s1 = seen.clone();
    let s2 = seen.clone();
    let mut decoded = Vec::new();
    crate::stream::ReaderBuilder::new()
        .user_chunk_callback(0x80, move |id, body| {
            s1.lock().unwrap().push((id, body.to_vec()));
            Ok(())
        })
        .user_chunk_callback(0xc0, move |id, body| {
            s2.lock().unwrap().push((id, body.to_vec()));
            Ok(())
        })
        .build(Cursor::new(&out))
        .unwrap()
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, b"prefixinfixsuffix");
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0], (0x80, b"meta-a".to_vec()));
    assert_eq!(seen[1], (0xc0, b"meta-b".to_vec()));
}

#[test]
fn add_user_chunk_invalid_id_errors() {
    let buf: Vec<u8> = Vec::new();
    let mut w = Writer::new(buf);
    assert!(w.add_user_chunk(0x7f, b"oops").is_err());
    assert!(w.add_user_chunk(0xfe, b"oops").is_err());
}

#[test]
fn padding_produces_multiple_of_n() {
    let payload = b"abc";
    let pad: u32 = 64;
    let buf: Vec<u8> = Vec::new();
    let mut w = WriterBuilder::new().padding(pad).build(buf).unwrap();
    w.write_all(payload).unwrap();
    let out = w.finish().unwrap();
    assert_eq!(out.len() % pad as usize, 0);
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&out))
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn padding_with_already_aligned_output_is_noop() {
    // Build a payload whose output happens to land on a multiple of `pad`.
    // We can't easily predict the output size, so just check that the
    // padding-enabled writer never produces a stream shorter than the
    // non-padded version.
    let payload = vec![b'x'; 100];
    let pad = 32u32;
    let buf: Vec<u8> = Vec::new();
    let mut w = WriterBuilder::new().padding(pad).build(buf).unwrap();
    w.write_all(&payload).unwrap();
    let out_padded = w.finish().unwrap();

    let buf: Vec<u8> = Vec::new();
    let mut w = WriterBuilder::new().build(buf).unwrap();
    w.write_all(&payload).unwrap();
    let out_plain = w.finish().unwrap();

    assert!(out_padded.len() >= out_plain.len());
    assert_eq!(out_padded.len() % pad as usize, 0);
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&out_padded))
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn reset_reuses_state() {
    let buf: Vec<u8> = Vec::new();
    let mut w = Writer::new(buf);
    w.write_all(b"first").unwrap();
    let first = w.finish().unwrap();

    let buf: Vec<u8> = Vec::new();
    let mut w = Writer::new(buf);
    w.write_all(b"first").unwrap();
    // Now reset and write a different payload.
    w.reset(Vec::new());
    w.write_all(b"second").unwrap();
    let second = w.finish().unwrap();

    let mut d1 = Vec::new();
    Reader::new(Cursor::new(&first))
        .read_to_end(&mut d1)
        .unwrap();
    let mut d2 = Vec::new();
    Reader::new(Cursor::new(&second))
        .read_to_end(&mut d2)
        .unwrap();
    assert_eq!(d1, b"first");
    assert_eq!(d2, b"second");
}

#[test]
fn written_reports_byte_counts() {
    let buf: Vec<u8> = Vec::new();
    let mut w = Writer::new(buf);
    let payload = vec![b'q'; 8 << 10];
    w.write_all(&payload).unwrap();
    w.flush().unwrap();
    let written = w.written();
    assert_eq!(written.uncompressed, payload.len() as u64);
    assert!(written.compressed > 0);
    // After finish there's the EOF chunk too.
    let out = w.finish().unwrap();
    assert!(out.len() as u64 > written.compressed);
}

#[test]
fn flush_does_not_emit_eof() {
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut w = Writer::new(&mut buf);
        w.write_all(b"partial").unwrap();
        w.flush().unwrap();
    }
    // Reader expects EOF — without finish() the stream is truncated.
    let mut decoded = Vec::new();
    let err = Reader::new(Cursor::new(&buf))
        .read_to_end(&mut decoded)
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn level_smallest_round_trip() {
    let payload: Vec<u8> = (0..4096).map(|i| (i * 31) as u8).collect();
    let buf: Vec<u8> = Vec::new();
    let mut w = WriterBuilder::new()
        .level(crate::Level::Smallest)
        .build(buf)
        .unwrap();
    w.write_all(&payload).unwrap();
    let out = w.finish().unwrap();
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&out))
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn write_returns_full_length() {
    let buf: Vec<u8> = Vec::new();
    let mut w = Writer::new(buf);
    let n = w.write(b"hello").unwrap();
    assert_eq!(n, 5);
}

/// Deterministic LCG to feed varied input shapes through the stream codec
/// without pulling in `rand`.  Keeps runs short enough for `cargo test`.
fn lcg(seed: u64) -> impl FnMut() -> u8 {
    let mut x = seed;
    move || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) as u8
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "200 trials × 3 levels × ≤40 KiB is impractical under miri"
)]
fn stress_random_inputs_roundtrip() {
    // 200 inputs at sizes 0..40 KiB (crosses the 4 KiB minimum block size at
    // L1 etc.).  Three compression levels to keep runtime moderate.
    for trial in 0..200u64 {
        let mut rng = lcg(0xc0ffee ^ trial.wrapping_mul(0x9e37));
        let size = (trial as usize * 211) % (40 * 1024);
        let mut input = vec![0u8; size];
        // Bias: half of trials get repetitive data, half get pseudo-random.
        if trial & 1 == 0 {
            for chunk in input.chunks_mut(7) {
                let v = rng();
                for b in chunk {
                    *b = v;
                }
            }
        } else {
            for b in &mut input {
                *b = rng();
            }
        }
        for level in [
            crate::Level::SuperFast,
            crate::Level::Fastest,
            crate::Level::Smallest,
        ] {
            let mut w = WriterBuilder::new()
                .block_size(crate::stream::MIN_BLOCK_SIZE)
                .level(level)
                .build(Vec::new())
                .unwrap();
            w.write_all(&input).unwrap();
            let stream = w.finish().unwrap();
            let mut decoded = Vec::new();
            Reader::new(Cursor::new(&stream))
                .read_to_end(&mut decoded)
                .unwrap();
            assert_eq!(decoded, input, "trial={trial} size={size} level={level:?}");
        }
    }
}

#[test]
fn empty_writes_are_noop() {
    let buf: Vec<u8> = Vec::new();
    let mut w = Writer::new(buf);
    assert_eq!(w.write(b"").unwrap(), 0);
    let out = w.finish().unwrap();
    // Empty stream: the EOF chunk alone, with no stream identifier (SPEC 4.6).
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&out))
        .read_to_end(&mut decoded)
        .unwrap();
    assert!(decoded.is_empty());
}

#[cfg(miri)]
#[test]
fn stress_random_inputs_roundtrip_miri() {
    // 6 trials × 2 levels.  Sizes 500..5620 span the 4 KiB
    // MIN_BLOCK_SIZE boundary (3 single-block, 3 multi-block) and
    // cover both repetitive and pseudo-random inputs through L1 + L3.
    for trial in 0..6u64 {
        let mut rng = lcg(0xc0ffee ^ trial.wrapping_mul(0x9e37));
        let size = (trial as usize) * 1024 + 500;
        let mut input = vec![0u8; size];
        if trial & 1 == 0 {
            for chunk in input.chunks_mut(7) {
                let v = rng();
                for b in chunk {
                    *b = v;
                }
            }
        } else {
            for b in &mut input {
                *b = rng();
            }
        }
        for level in [crate::Level::Fastest, crate::Level::Smallest] {
            let mut w = WriterBuilder::new()
                .block_size(crate::stream::MIN_BLOCK_SIZE)
                .level(level)
                .build(Vec::new())
                .unwrap();
            w.write_all(&input).unwrap();
            let stream = w.finish().unwrap();
            let mut decoded = Vec::new();
            Reader::new(Cursor::new(&stream))
                .read_to_end(&mut decoded)
                .unwrap();
            assert_eq!(decoded, input, "trial={trial} size={size} level={level:?}");
        }
    }
}

// -------------------- SIMD chunks (ports of writer_test.go) --------------------

/// Go's `simdLevels`.
const SIMD_LEVELS: [Level; 7] = [
    Level::SuperFastSimd,
    Level::FastestSimd,
    Level::BalancedSimd,
    Level::SmallestSimd,
    Level::SuperFastSimdDelta,
    Level::FastestSimdDelta,
    Level::BalancedSimdDelta,
];

fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

/// Go's `makeTestData`: n bytes of JSON log lines.
fn make_test_data(n: usize) -> Vec<u8> {
    const LINES: [&str; 5] = [
        "{\"timestamp\":\"2024-01-15T10:11:12Z\",\"level\":\"INFO\",\"msg\":\"hello world\"}\n",
        "{\"timestamp\":\"2024-01-15T10:11:13Z\",\"level\":\"WARN\",\"msg\":\"needle in the haystack\"}\n",
        "{\"timestamp\":\"2024-01-15T10:11:14Z\",\"level\":\"ERROR\",\"msg\":\"something failed\"}\n",
        "{\"timestamp\":\"2024-01-15T10:11:15Z\",\"level\":\"DEBUG\",\"msg\":\"verbose log entry\"}\n",
        "{\"timestamp\":\"2024-01-15T10:11:16Z\",\"level\":\"INFO\",\"msg\":\"shutting down\"}\n",
    ];
    let mut rng = xorshift(0xdead);
    let mut b = Vec::with_capacity(n + 100);
    while b.len() < n {
        b.extend_from_slice(LINES[rng() as usize % LINES.len()].as_bytes());
    }
    b.truncate(n);
    b
}

/// Go's `simdTestData`: n bytes of text, with a random part that doesn't compress and a table
/// of numbers for delta literals.
fn simd_test_data(n: usize) -> Vec<u8> {
    let mut b = make_test_data(n);
    let mut rng = xorshift(0x51);
    for v in &mut b[n / 3..n / 3 + (n / 8).min(48 << 10)] {
        *v = rng() as u8;
    }
    let mut i = 2 * n / 3;
    while i + 4 <= n.min(2 * n / 3 + (64 << 10)) {
        b[i..i + 4].copy_from_slice(&(((i * i) >> 10) as u32).to_le_bytes());
        i += 4;
    }
    b
}

/// The number of chunks of each type in the MinLZ stream b.
fn stream_chunks(mut b: &[u8]) -> [usize; 256] {
    let mut n = [0; 256];
    while !b.is_empty() {
        assert!(b.len() >= 4, "truncated chunk header");
        let l = 4 + (b[1] as usize | (b[2] as usize) << 8 | (b[3] as usize) << 16);
        assert!(l <= b.len(), "truncated chunk");
        n[b[0] as usize] += 1;
        b = &b[l..];
    }
    n
}

/// Go's `NewWriter` with `WriterConcurrency`: concurrency 1 is the synchronous [`Writer`].
/// A `None` level is `WriterUncompressed`.
enum AnyWriter {
    St(Writer<Vec<u8>>),
    Mt(MtWriter<Vec<u8>>),
}

impl AnyWriter {
    fn new(level: Option<Level>, block_size: usize, conc: usize, padding: u32) -> Self {
        if conc == 1 {
            let b = WriterBuilder::new().block_size(block_size).padding(padding);
            let b = match level {
                Some(l) => b.level(l),
                None => b.uncompressed(),
            };
            return Self::St(b.append_index().build(Vec::new()).unwrap());
        }
        let b = MtWriterBuilder::new()
            .block_size(block_size)
            .padding(padding)
            .concurrency(conc);
        let b = match level {
            Some(l) => b.level(l),
            None => b.uncompressed(),
        };
        Self::Mt(b.append_index().build(Vec::new()).unwrap())
    }

    /// Go's `EncodeBuffer`: buf is encoded as blocks of its own.
    fn encode_buffer(&mut self, buf: &[u8]) -> io::Result<()> {
        match self {
            Self::St(w) => {
                w.flush()?;
                w.write_all(buf)?;
                w.flush()
            }
            Self::Mt(w) => w.encode_buffer(buf),
        }
    }

    fn add_user_chunk(&mut self, id: u8, data: &[u8]) -> io::Result<()> {
        match self {
            Self::St(w) => w.add_user_chunk(id, data),
            Self::Mt(w) => w.add_user_chunk(id, data),
        }
    }

    /// The finished stream, and the uncompressed size Go's `Written` reports.
    fn finish(mut self) -> (Vec<u8>, u64) {
        // The writers count blocks as they emit them.
        self.flush().unwrap();
        match self {
            Self::St(w) => {
                let n = w.written().uncompressed;
                (w.finish().unwrap(), n)
            }
            Self::Mt(w) => {
                let n = w.written().uncompressed;
                (w.finish().unwrap(), n)
            }
        }
    }
}

impl Write for AnyWriter {
    fn write(&mut self, p: &[u8]) -> io::Result<usize> {
        match self {
            Self::St(w) => w.write(p),
            Self::Mt(w) => w.write(p),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::St(w) => w.flush(),
            Self::Mt(w) => w.flush(),
        }
    }
}

fn read_all<R: Read>(mut r: R) -> io::Result<Vec<u8>> {
    let mut v = Vec::new();
    r.read_to_end(&mut v)?;
    Ok(v)
}

/// Go's `testSIMDStreamRead`: comp decodes to want with every reader API. comp must have an
/// index.
fn test_simd_stream_read(comp: &[u8], want: &[u8]) {
    let got = read_all(Reader::new(comp)).expect("Read");
    assert!(got == want, "Read: output mismatch");
    let got = Reader::new(comp)
        .decode_concurrent(Vec::new(), 4)
        .expect("decode_concurrent")
        .writer;
    assert!(got == want, "decode_concurrent: output mismatch");
    let idx = index_stream(comp).expect("index_stream");
    let mut index = Index::default();
    index.load(&idx).unwrap();
    let mut rs = ReadSeeker::new(Reader::new(Cursor::new(comp))).expect("ReadSeeker");
    let mut tmp = [0u8; 1000];
    let n = want.len();
    for off in [0, n / 3 + 7, n / 2, n - 100] {
        let want_at = &want[off..(off + tmp.len()).min(n)];
        let tmp = &mut tmp[..want_at.len()];
        rs.seek(SeekFrom::Start(off as u64)).expect("Seek");
        rs.read_exact(tmp)
            .unwrap_or_else(|e| panic!("Seek {off}: {e}"));
        assert!(tmp == want_at, "Seek {off}: output mismatch");
        let k = rs.read_at(tmp, off as u64).unwrap();
        assert!(tmp[..k] == want_at[..], "ReadAt {off}: output mismatch");
        let mut dec = Reader::new(comp);
        dec.skip(off as u64).expect("Skip");
        let got = read_all(dec).unwrap_or_else(|e| panic!("Skip {off}: {e}"));
        assert!(got == want[off..], "Skip {off}: output mismatch");
        let e = index.find(off as u64);
        let mut dec = ReaderBuilder::new()
            .ignore_stream_id()
            .build(&comp[e.compressed as usize..])
            .unwrap();
        dec.skip(off as u64 - e.uncompressed)
            .expect("Skip from index");
        dec.read_exact(tmp)
            .unwrap_or_else(|e| panic!("read from index {off}: {e}"));
        assert!(tmp == want_at, "read from index {off}: output mismatch");
    }
}

/// Ports `TestWriterSIMD` at Go's `-short` size and case selection, with the Writer at
/// concurrency 1 and the MtWriter at 4.
#[test]
#[cfg_attr(miri, ignore = "600 KiB streams too slow for miri")]
fn writer_simd() {
    let n = 600 << 10;
    let data = simd_test_data(n);
    type WriteFn = fn(&mut AnyWriter, &[u8]) -> io::Result<()>;
    let writes: [(&str, u32, WriteFn); 4] = [
        ("write", 1000, |w, data| {
            let mut rng = xorshift(1);
            let mut p = data;
            while !p.is_empty() {
                let k = p.len().min(rng() as usize % (200 << 10) + 1);
                w.write_all(&p[..k])?;
                p = &p[k..];
            }
            Ok(())
        }),
        ("write-all", 0, |w, data| w.write_all(data)),
        ("copy", 0, |w, data| io::copy(&mut &*data, w).map(drop)),
        ("encodebuffer-flush", 0, |w, data| {
            w.encode_buffer(&data[..data.len() / 2])?;
            w.flush()?;
            w.encode_buffer(&data[data.len() / 2..])
        }),
    ];
    let mut k = 0;
    for level in SIMD_LEVELS {
        for bs in [4 << 10, 64 << 10, 1 << 20, 8 << 20] {
            for conc in [1, 4] {
                for (name, padding, write) in writes {
                    // Go's -short runs every fifth case. 5 shares no factor with the loop
                    // sizes, so every pair of settings still runs.
                    k += 1;
                    if k % 5 != 0 {
                        continue;
                    }
                    let what = format!("level-{}/bs-{bs}/c-{conc}/{name}", i32::from(level));
                    let mut w = AnyWriter::new(Some(level), bs, conc, padding);
                    write(&mut w, &data).unwrap_or_else(|e| panic!("{what}: {e}"));
                    let (comp, in_n) = w.finish();
                    assert_eq!(in_n, n as u64, "{what}: written");
                    assert!(
                        comp[MAGIC_BODY_LEN + 3] & STREAM_FLAG_SIMD != 0,
                        "{what}: stream identifier has no SIMD flag"
                    );
                    let c = stream_chunks(&comp);
                    assert_eq!(c[CHUNK_TYPE_MINLZ_COMPRESSED_DATA as usize], 0, "{what}");
                    assert!(c[CHUNK_TYPE_MINLZ_SIMD as usize] > 0, "{what}: SIMD chunks");
                    test_simd_stream_read(&comp, &data);
                }
            }
        }
    }
}

/// Ports `TestWriterSIMDChunks`: the layout of SIMD chunks against the block encoder. The
/// classic levels are added, since only the `encode` goldens pin their encoders to Go.
#[test]
#[cfg_attr(miri, ignore = "300 KiB streams too slow for miri")]
fn writer_simd_chunks() {
    let data = simd_test_data(300 << 10);
    const BS: usize = 16 << 10;
    let classic = [
        Level::SuperFast,
        Level::Fastest,
        Level::Balanced,
        Level::Smallest,
    ];
    for level in classic.into_iter().chain(SIMD_LEVELS) {
        let (kind, sum_size) = if classic.contains(&level) {
            (CHUNK_TYPE_MINLZ_COMPRESSED_DATA, CHECKSUM_SIZE)
        } else {
            (CHUNK_TYPE_MINLZ_SIMD, SIMD_HASH_SIZE)
        };
        for conc in [1, 2] {
            let mut w = AnyWriter::new(Some(level), BS, conc, 0);
            w.encode_buffer(&data).unwrap();
            let (stream, _) = w.finish();
            let mut comp = &stream[MAGIC_BODY_LEN + 4..];
            let (mut off, mut types) = (0, [0usize; 256]);
            let (mut block, mut whole) = (Vec::new(), Vec::new());
            while !comp.is_empty() {
                let l = comp[1] as usize | (comp[2] as usize) << 8 | (comp[3] as usize) << 16;
                let payload = &comp[4..4 + l];
                types[comp[0] as usize] += 1;
                let src = &data[off..(off + BS).min(data.len())];
                let what = format!("level {}, concurrency {conc}", i32::from(level));
                match comp[0] {
                    t if t == kind => {
                        block.clear();
                        assert!(try_encode(&mut block, src, level).unwrap(), "{what}");
                        crate::encode(&mut whole, src, level).unwrap();
                        let sum = match kind {
                            CHUNK_TYPE_MINLZ_SIMD => xxh3::hash(src),
                            _ => masked_crc32c(src).into(),
                        };
                        assert!(
                            block == whole
                                && payload[..sum_size] == sum.to_le_bytes()[..sum_size]
                                && payload[sum_size..] == block[1..],
                            "{what}: chunk at {off} differs from encode"
                        );
                        off += src.len();
                    }
                    CHUNK_TYPE_UNCOMPRESSED_DATA => {
                        assert!(
                            !try_encode(&mut block, src, level).unwrap(),
                            "{what}: uncompressed chunk at {off} compresses with try_encode"
                        );
                        off += l - CHECKSUM_SIZE;
                    }
                    _ => {}
                }
                comp = &comp[4 + l..];
            }
            assert_eq!(off, data.len());
            assert!(types[CHUNK_TYPE_UNCOMPRESSED_DATA as usize] > 0);
            assert!(types[kind as usize] > 0, "level {}", i32::from(level));
        }
    }
}

/// Ports `TestWriterSIMDIdentifier` without the sidecar: the stream identifier has the SIMD
/// flag exactly for SIMD levels, also when a user chunk makes `add_user_chunk` write it.
#[test]
#[cfg_attr(miri, ignore = "40 SIMD and classic streams are slow under miri")]
fn writer_simd_identifier() {
    let data = make_test_data(10 << 10);
    let levels = [
        None,
        Some(Level::SuperFast),
        Some(Level::Fastest),
        Some(Level::Balanced),
    ]
    .into_iter()
    .chain([Some(Level::Smallest)])
    .chain(SIMD_LEVELS.map(Some));
    for level in levels {
        for i in 0..4 {
            let (user_first, conc) = (i & 1 != 0, 1 + (i >> 1));
            let mut w = AnyWriter::new(level, 64 << 10, conc, 0);
            if user_first {
                w.add_user_chunk(MIN_USER_SKIPPABLE_CHUNK, b"user").unwrap();
            }
            w.write_all(&data).unwrap();
            let (main, _) = w.finish();
            let simd = level.is_some_and(Level::is_simd);
            assert_eq!(
                main[MAGIC_BODY_LEN + 3] & STREAM_FLAG_SIMD != 0,
                simd,
                "level {level:?}, user chunk {user_first}, concurrency {conc}"
            );
            assert!(
                read_all(Reader::new(&main[..])).unwrap() == data,
                "{level:?}"
            );
        }
    }
}

/// Ports `TestWriterSIMDHash`: the hash of SIMD chunks against XXH3 64 bit, seed 0, values of
/// the C implementation.
#[test]
fn writer_simd_hash() {
    for (n, want) in [(1000, 0x30e4c5246f300d11u64), (4095, 0x268198759d7bdf74)] {
        let data: Vec<u8> = (0..n).map(|i| ((i + 1) % 251) as u8).collect();
        let mut w = WriterBuilder::new()
            .level(Level::FastestSimd)
            .build(Vec::new())
            .unwrap();
        w.write_all(&data).unwrap();
        let out = w.finish().unwrap();
        let chunk = &out[MAGIC_BODY_LEN + 4..];
        let hash = u64::from_le_bytes(chunk[4..12].try_into().unwrap());
        assert!(
            chunk[0] == CHUNK_TYPE_MINLZ_SIMD && hash == want,
            "{n} bytes: chunk type {:#x}, hash {hash:#x}, want {want:#x}",
            chunk[0]
        );
    }
}

/// Ports the stream part of `TestDataRoundtrips` (`minlz_test.go`) for its generated input,
/// at every level and Go's `testOptions` additions (`simd`, `simd-d`). The zip inputs are in
/// [`encoder_regression`], whose classic levels skip the two over 10000 bytes, and the SIMD
/// encode tests' `fuzz_encoding_blocks_seeded`; the zstd one needs a decoder this crate doesn't
/// depend on.
#[test]
#[cfg_attr(miri, ignore = "1 MiB streams too slow for miri")]
fn data_roundtrips_longblock() {
    let data = vec![0u8; 1 << 20];
    let levels = [
        Level::SuperFast,
        Level::Fastest,
        Level::Balanced,
        Level::Smallest,
    ]
    .into_iter()
    .chain(SIMD_LEVELS);
    for level in levels {
        for conc in [1, 4] {
            let mut w = AnyWriter::new(Some(level), crate::stream::DEFAULT_BLOCK_SIZE, conc, 0);
            w.write_all(&data).unwrap();
            let (comp, _) = w.finish();
            let got = read_all(Reader::new(&comp[..])).unwrap();
            assert!(got == data, "level {level:?}, concurrency {conc}");
        }
        let mut block = Vec::new();
        crate::encode(&mut block, &data, level).unwrap();
        let mut got = Vec::new();
        crate::decode(&mut got, &block).unwrap();
        assert!(got == data, "block level {level:?}");
    }
}

/// Ports `TestEncoderRegression` (`writer_test.go`): the inputs of Go's `enc_regressions.zip`
/// (with `MINLZ_TESTDATA`) through the writers with Go's `testOptions`, written whole and with
/// `io::copy` for Go's `ReadFrom`. The 4 MiB window (no different from the default for these
/// inputs) and 4 MiB padding are left out. Inputs over 10000 bytes, which Go's `-short` mode
/// skips, skip only the classic levels. One reader decodes every stream, reset between them.
#[test]
#[cfg_attr(miri, ignore = "hundreds of streams too slow for miri")]
fn encoder_regression() {
    #[derive(Clone, Copy)]
    enum Opt {
        Default,
        Level(Level),
        Uncompressed,
    }
    let opts = [
        ("default", Opt::Default),
        ("fast", Opt::Level(Level::Fastest)),
        ("best", Opt::Level(Level::Smallest)),
        ("none", Opt::Uncompressed),
        ("simd", Opt::Level(Level::BalancedSimd)),
        ("simd-d", Opt::Level(Level::FastestSimdDelta)),
    ];
    let new = |opt, c1: bool, bs: usize, pad: u32| {
        macro_rules! with_opt {
            ($b:expr) => {{
                let b = $b.block_size(bs).padding(pad);
                match opt {
                    Opt::Default => b.append_index(),
                    Opt::Level(l) => b.level(l),
                    Opt::Uncompressed => b.uncompressed(),
                }
                .build(Vec::new())
                .unwrap()
            }};
        }
        if c1 {
            AnyWriter::St(with_opt!(WriterBuilder::new()))
        } else {
            AnyWriter::Mt(with_opt!(MtWriterBuilder::new()))
        }
    };
    let mut dec = Reader::new(Cursor::new(Vec::new()));
    for (i, data) in crate::fuzz_seeds::go_seeds("enc_regressions.zip")
        .iter()
        .enumerate()
    {
        for (name, opt) in opts {
            if data.len() > 10000 && !matches!(opt, Opt::Level(l) if l.is_simd()) {
                continue;
            }
            for c1 in [false, true] {
                for bs in [crate::stream::DEFAULT_BLOCK_SIZE, 4 << 10] {
                    for pad in [0, 2, 8000] {
                        for copy in [false, true] {
                            let what = format!(
                                "input {i}: {name}, c1 {c1}, bs {bs}, pad {pad}, copy {copy}"
                            );
                            let mut w = new(opt, c1, bs, pad);
                            if copy {
                                // Not a &[u8], which io::copy hands to write_all whole.
                                io::copy(&mut Cursor::new(data), &mut w).map(drop)
                            } else {
                                w.write_all(data)
                            }
                            .unwrap_or_else(|e| panic!("{what}: {e}"));
                            let (comp, _) = w.finish();
                            assert!(pad == 0 || comp.len() % pad as usize == 0, "{what}");
                            dec.reset(Cursor::new(comp));
                            let got = read_all(&mut dec).unwrap_or_else(|e| panic!("{what}: {e}"));
                            assert!(got == *data, "{what}: decoder mismatch");
                        }
                    }
                }
            }
        }
    }
}

/// Ports `TestWriterConcurrencyIndex` (`writer_test.go`) without the search tables, also at a
/// SIMD level: the stream with its index doesn't depend on the concurrency, and the index
/// entries point at blocks, not at the stream header or user chunks before them.
#[test]
#[cfg_attr(miri, ignore = "5 MiB streams too slow for miri")]
fn writer_concurrency_index() {
    const BS: usize = 256 << 10;
    let data = make_test_data(5 << 20);
    for level in [Level::Balanced, Level::BalancedSimd] {
        for api in ["write", "buffer", "copy", "user"] {
            let mut want: Option<Vec<u8>> = None;
            for conc in [1, 4] {
                let what = format!("{level:?}/{api}: concurrency {conc}");
                let mut w = AnyWriter::new(Some(level), BS, conc, 0);
                match api {
                    "write" => w.write_all(&data),
                    "buffer" => w.encode_buffer(&data),
                    "copy" => io::copy(&mut Cursor::new(&data), &mut w).map(drop),
                    _ => [&data[..4 * BS], &data[4 * BS..]]
                        .into_iter()
                        .zip(MIN_USER_SKIPPABLE_CHUNK..)
                        .try_for_each(|(part, id)| {
                            w.add_user_chunk(id, b"user")?;
                            w.write_all(part)
                        }),
                }
                .unwrap_or_else(|e| panic!("{what}: {e}"));
                let (comp, _) = w.finish();
                let mut idx = Index::default();
                idx.load_stream(&mut Cursor::new(&comp))
                    .unwrap_or_else(|e| panic!("{what}: {e}"));
                let o = idx.offsets();
                assert!(o.len() >= 3 && o[0].compressed != 0, "{what}: index {o:?}");
                for e in o {
                    assert!(
                        matches!(
                            comp[e.compressed as usize],
                            CHUNK_TYPE_UNCOMPRESSED_DATA
                                | CHUNK_TYPE_MINLZ_COMPRESSED_DATA
                                | CHUNK_TYPE_MINLZ_SIMD
                        ),
                        "{what}: entry {e:?} is not at a block"
                    );
                }
                match &want {
                    None => want = Some(comp),
                    Some(w) => assert!(
                        comp == *w,
                        "{what}: {} bytes differ from concurrency 1 ({} bytes)",
                        comp.len(),
                        w.len()
                    ),
                }
            }
        }
    }
}

/// Runs Go's `cmd/mz` with `args`, `input` on stdin, and returns stdout.
fn go_mz(go: &str, args: &[&str], input: &[u8]) -> Vec<u8> {
    use std::process::{Command, Stdio};
    let mut child = Command::new(go)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn go mz");
    let mut stdin = child.stdin.take().unwrap();
    let input = input.to_vec();
    let feed = std::thread::spawn(move || stdin.write_all(&input));
    let out = child.wait_with_output().expect("wait");
    feed.join().unwrap().unwrap();
    assert!(
        out.status.success(),
        "go mz {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// Cross-impl: SIMD streams at every SIMD level, written by Go `cmd/mz` (`-cpu` 1 and 4) and
/// decoded by Rust, and written by Rust (Writer, MtWriter) and decoded by Go (`-cpu` 1 and 4).
/// Once Rust writes SIMD chunks at a level, its streams match Go's `-cpu=1` streams byte for
/// byte. Gated on `MINLZ_GO_CMD`, a Go `cmd/mz` of the simd-blocks branch.
#[test]
fn simd_streams_cross_go() {
    let Ok(go) = std::env::var("MINLZ_GO_CMD") else {
        return;
    };
    let data = simd_test_data(600 << 10);
    for level in SIMD_LEVELS {
        let l = i32::from(level);
        let mut args = vec![
            "c",
            "-c",
            "-bs=64K",
            ["-xfast", "-1", "-2", "-3"][(l & 255) as usize - 4],
        ];
        if l & 256 != 0 {
            args.push("-simd.delta");
        }
        let mut w = AnyWriter::new(Some(level), 64 << 10, 1, 0);
        w.write_all(&data).unwrap();
        let (rust, _) = w.finish();
        for cpu in ["-cpu=1", "-cpu=4"] {
            let go_stream = go_mz(&go, &[&args[..], &[cpu, "-"]].concat(), &data);
            assert!(
                go_stream[MAGIC_BODY_LEN + 3] & STREAM_FLAG_SIMD != 0
                    && stream_chunks(&go_stream)[CHUNK_TYPE_MINLZ_SIMD as usize] > 0,
                "level {l} {cpu}: Go wrote no SIMD chunks"
            );
            test_simd_stream_read(&go_stream, &data);
            if cpu == "-cpu=1" && stream_chunks(&rust)[CHUNK_TYPE_MINLZ_SIMD as usize] > 0 {
                assert!(
                    rust == go_stream,
                    "level {l}: Rust stream differs from Go's"
                );
            }
        }
        for conc in [1, 4] {
            let mut w = AnyWriter::new(Some(level), 64 << 10, conc, 0);
            w.write_all(&data).unwrap();
            let (rust, _) = w.finish();
            for cpu in ["-cpu=1", "-cpu=4"] {
                let got = go_mz(&go, &["d", "-c", cpu, "-"], &rust);
                assert!(got == data, "level {l}, concurrency {conc}, Go {cpu}");
            }
        }
    }
}
