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

use std::collections::HashSet;
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::sync::{Arc, Mutex, Weak, mpsc};
use std::time::Instant;

use super::{BlockJob, ENGINE_IDLE, Engine, IDLE, OrderItem};
use crate::Index;
use crate::stream::mt_pool::BufferPool;
use crate::stream::{
    ConcurrentDecode, MAX_BLOCK_SIZE, MIN_BLOCK_SIZE, MtWriter, MtWriterBuilder, ReadSeeker, Reader,
};

fn round_trip_concurrency(payload: &[u8], concurrency: usize) -> Vec<u8> {
    let n = concurrency;
    let mut w = MtWriterBuilder::new()
        .concurrency(n)
        .build(Vec::new())
        .unwrap();
    w.write_all(payload).unwrap();
    let buf = w.finish().unwrap();
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&buf))
        .read_to_end(&mut decoded)
        .unwrap();
    decoded
}

#[test]
fn empty_round_trip() {
    let got = round_trip_concurrency(b"", 4);
    assert!(got.is_empty());
}

#[test]
fn small_payload_round_trip() {
    let got = round_trip_concurrency(b"Hello, MinLZ!", 4);
    assert_eq!(got, b"Hello, MinLZ!");
}

#[test]
fn highly_compressible_round_trip() {
    let payload = vec![b'a'; 10_000];
    let got = round_trip_concurrency(&payload, 4);
    assert_eq!(got, payload);
}

#[test]
fn multi_block_payload_round_trip() {
    let payload: Vec<u8> = (0..16 * 1024).map(|i| (i % 256) as u8).collect();
    let mut w = MtWriterBuilder::new()
        .block_size(4 << 10)
        .concurrency(4)
        .build(Vec::new())
        .unwrap();
    w.write_all(&payload).unwrap();
    let buf = w.finish().unwrap();
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&buf))
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn encode_buffer_round_trip() {
    let payload: Vec<u8> = (0..40 * 1024).map(|i| (i * 31) as u8).collect();
    let mut w = MtWriterBuilder::new()
        .block_size(4 << 10)
        .concurrency(4)
        .build(Vec::new())
        .unwrap();
    w.encode_buffer(&payload).unwrap();
    let buf = w.finish().unwrap();
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&buf))
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn uncompressed_mode_round_trip() {
    let payload = vec![b'b'; 5_000];
    let mut w = MtWriterBuilder::new()
        .uncompressed()
        .concurrency(4)
        .build(Vec::new())
        .unwrap();
    w.write_all(&payload).unwrap();
    let buf = w.finish().unwrap();
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&buf))
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn concurrency_auto_and_one_work() {
    let payload = vec![b'q'; 8 << 10];
    for n in [0, 1] {
        assert_eq!(
            round_trip_concurrency(&payload, n),
            payload,
            "concurrency {n}"
        );
    }
}

#[test]
fn written_reports_byte_counts() {
    let mut mt = MtWriterBuilder::new()
        .block_size(64 << 10)
        .concurrency(2)
        .build(Vec::<u8>::new())
        .unwrap();
    assert_eq!(mt.written(), Default::default());
    let mut st = crate::stream::WriterBuilder::new()
        .block_size(64 << 10)
        .build(Vec::<u8>::new())
        .unwrap();
    let payload: Vec<u8> = (0..300_000u32).map(|i| (i % 251) as u8).collect();
    for w in [&mut mt as &mut dyn Write, &mut st] {
        w.write_all(&payload).unwrap();
        w.flush().unwrap();
    }
    let written = mt.written();
    assert_eq!(written.uncompressed, payload.len() as u64);
    assert_eq!(written, st.written());
    // After finish there's the EOF chunk too.
    let out = mt.finish().unwrap();
    assert!(out.len() as u64 > written.compressed);
}

#[test]
fn finish_returns_inner_writer() {
    let payload = b"return me".to_vec();
    let mut w = MtWriter::new(Vec::<u8>::new());
    w.write_all(&payload).unwrap();
    let buf = w.finish().unwrap();
    assert!(!buf.is_empty());
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&buf))
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, payload);
}

#[test]
fn add_user_chunk_round_trip_via_callback() {
    let mut w = MtWriterBuilder::new()
        .concurrency(2)
        .build(Vec::<u8>::new())
        .unwrap();
    w.write_all(b"prefix").unwrap();
    w.add_user_chunk(0x80, b"meta-a").unwrap();
    w.write_all(b"infix").unwrap();
    w.add_user_chunk(0xc0, b"meta-b").unwrap();
    w.write_all(b"suffix").unwrap();
    let buf = w.finish().unwrap();

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
        .build(Cursor::new(&buf))
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
fn cross_concurrency_round_trips() {
    let payload: Vec<u8> = (0..32 * 1024).map(|i| (i * 17) as u8).collect();
    // Miri: one worker and a pool of four.
    let ns: &[usize] = if cfg!(miri) { &[1, 4] } else { &[1, 2, 4, 8] };
    for &n in ns {
        let got = round_trip_concurrency(&payload, n);
        assert_eq!(got, payload, "concurrency = {n}");
    }
}

#[test]
fn drop_without_finish_does_not_block() {
    let payload = vec![b'x'; 64 * 1024];
    let buf: Vec<u8> = Vec::new();
    let mut w = MtWriterBuilder::new()
        .block_size(4 << 10)
        .concurrency(4)
        .build(buf)
        .unwrap();
    w.write_all(&payload).unwrap();
    drop(w);
    // If we got here without hanging, the test passes.
}

#[test]
fn drop_without_finish_writes_no_trailer() {
    #[derive(Clone, Default)]
    struct Shared(Arc<Mutex<Vec<u8>>>);
    impl Write for Shared {
        fn write(&mut self, p: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(p);
            Ok(p.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let out = Shared::default();
    let mut w = MtWriterBuilder::new()
        .block_size(4 << 10)
        .concurrency(2)
        .append_index()
        .padding(1 << 10)
        .build(out.clone())
        .unwrap();
    w.write_all(&[b'x'; 10 << 10]).unwrap();
    drop(w);
    let out = out.0.lock().unwrap();
    assert!(!out.ends_with(b"\x00xdi2s"), "dropped stream got an index");
    assert!(
        Index::default()
            .load_stream(&mut Cursor::new(&out[..]))
            .is_err()
    );
}

/// Determinism / stress: random PRNG inputs at varied sizes + cross
/// encode/decode concurrencies (1×16 = 16 combinations) for many
/// iterations.  Surfaces scheduling-dependent bugs that single-iteration
/// tests miss — substitute for `loom` while staying std-only.
#[test]
#[cfg_attr(miri, ignore = "50 trials × 16 (N,M) combos is impractical under miri")]
fn stress_random_cross_concurrency() {
    fn lcg(seed: u64) -> impl FnMut() -> u8 {
        let mut x = seed;
        move || {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 33) as u8
        }
    }
    // 50 random shapes × 16 (N,M) combos × 3 levels = 2400 round-trips.
    let combos: &[(usize, usize)] = &[
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (4, 8),
        (8, 1),
        (8, 2),
        (8, 4),
        (8, 8),
    ];
    for trial in 0..50u64 {
        let mut rng = lcg(0xdeadbeef ^ trial.wrapping_mul(0xC0FFEE));
        let size = ((trial as usize * 211 + 4) % (24 * 1024)).max(1);
        let mut input = vec![0u8; size];
        for b in &mut input {
            *b = rng();
        }
        for &(enc_n, dec_n) in combos {
            let mut w = MtWriterBuilder::new()
                .block_size(crate::stream::MIN_BLOCK_SIZE)
                .concurrency(enc_n)
                .build(Vec::<u8>::new())
                .unwrap();
            w.write_all(&input).unwrap();
            let stream = w.finish().unwrap();
            let ConcurrentDecode {
                bytes_written: n,
                writer: dec,
            } = Reader::new(&stream[..])
                .decode_concurrent(Vec::<u8>::with_capacity(input.len()), dec_n)
                .unwrap();
            assert_eq!(
                n as usize,
                input.len(),
                "trial={trial} enc={enc_n} dec={dec_n}"
            );
            assert_eq!(dec, input, "trial={trial} enc={enc_n} dec={dec_n}");
        }
    }
}

/// Cross-impl: encode with Rust MT, decode through Go `cmd/mz`.  Gated
/// on `MINLZ_GO_CMD` so CI without a Go binary still runs the suite.
#[test]
fn rust_mt_encoded_decodes_in_go() {
    use std::io::Write as _;
    use std::process::{Command, Stdio};
    let Ok(go_bin) = std::env::var("MINLZ_GO_CMD") else {
        return;
    };
    let payload: Vec<u8> = (0..96 * 1024)
        .map(|i| ((i * 31) ^ (i >> 5)) as u8)
        .collect();
    for concurrency in [1usize, 4, 8] {
        let mut w = MtWriterBuilder::new()
            .concurrency(concurrency)
            .block_size(8 << 10)
            .build(Vec::<u8>::new())
            .unwrap();
        w.write_all(&payload).unwrap();
        let stream = w.finish().unwrap();

        let mut child = Command::new(&go_bin)
            .args(["d", "-cpu=4", "-c", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn go mz");
        child.stdin.as_mut().unwrap().write_all(&stream).unwrap();
        let out = child.wait_with_output().expect("wait");
        assert!(
            out.status.success(),
            "Go decode failed at concurrency {concurrency}"
        );
        assert_eq!(out.stdout, payload, "concurrency {concurrency}");
    }
}

// -------------------- index integration --------------------

#[test]
#[cfg_attr(
    miri,
    ignore = "4 MiB stream × 3 concurrencies is impractical under miri"
)]
fn append_index_round_trips_via_read_seeker() {
    // Multi-block payload at small block_size so we get several index
    // entries (est_block_uncomp floors at 1 MiB → still 1 entry for
    // sub-MiB streams; use a large stream to actually exercise lookups).
    let payload: Vec<u8> = (0..(4 << 20))
        .map(|i| b'a' + ((i / 17) as u8 & 0x0f))
        .collect();
    for concurrency in [1usize, 4, 8] {
        let mut w = MtWriterBuilder::new()
            .block_size(64 << 10)
            .concurrency(concurrency)
            .append_index()
            .build(Vec::<u8>::new())
            .unwrap();
        w.write_all(&payload).unwrap();
        let stream = w.finish().unwrap();

        // Decode forward — sanity check.
        let mut decoded = Vec::new();
        Reader::new(Cursor::new(&stream))
            .read_to_end(&mut decoded)
            .unwrap();
        assert_eq!(decoded, payload, "concurrency={concurrency}");

        // Seek probes via ReadSeeker.
        let reader = Reader::new(Cursor::new(stream.clone()));
        let mut rs = ReadSeeker::new(reader)
            .unwrap_or_else(|e| panic!("ReadSeeker::new (n={concurrency}): {e}"));
        assert_eq!(rs.index().total_uncompressed(), Some(payload.len() as u64));

        for &off in &[0u64, 100_000, (2 << 20), (4 << 20) - 64] {
            if off as usize >= payload.len() {
                continue;
            }
            let want_len = 64.min(payload.len() - off as usize);
            let mut buf = vec![0u8; want_len];
            let n = rs.read_at(&mut buf, off).unwrap();
            assert_eq!(n, want_len, "n={concurrency} off={off}");
            assert_eq!(
                &buf[..],
                &payload[off as usize..off as usize + want_len],
                "n={concurrency} off={off}"
            );
        }

        // SeekFrom::End must work because total_uncompressed is known.
        let mut tail = [0u8; 32];
        let abs = rs.seek(SeekFrom::End(-32)).unwrap();
        assert_eq!(abs, payload.len() as u64 - 32);
        rs.read_exact(&mut tail).unwrap();
        assert_eq!(&tail, &payload[payload.len() - 32..]);
    }
}

#[test]
#[cfg_attr(miri, ignore = "256 KiB stream is impractical under miri")]
fn append_index_off_produces_smaller_stream() {
    let payload = vec![b'z'; 256 << 10];
    let n = 4;
    let mut indexed = MtWriterBuilder::new()
        .block_size(16 << 10)
        .concurrency(n)
        .append_index()
        .build(Vec::<u8>::new())
        .unwrap();
    indexed.write_all(&payload).unwrap();
    let with_idx = indexed.finish().unwrap();

    let mut bare = MtWriterBuilder::new()
        .block_size(16 << 10)
        .concurrency(n)
        .generate_index(false)
        .build(Vec::<u8>::new())
        .unwrap();
    bare.write_all(&payload).unwrap();
    let no_idx = bare.finish().unwrap();

    assert!(
        with_idx.len() > no_idx.len(),
        "indexed should be larger by the index chunk size"
    );

    // Both decode to the same payload.
    let mut a = Vec::new();
    Reader::new(Cursor::new(&with_idx))
        .read_to_end(&mut a)
        .unwrap();
    let mut b = Vec::new();
    Reader::new(Cursor::new(&no_idx))
        .read_to_end(&mut b)
        .unwrap();
    assert_eq!(a, payload);
    assert_eq!(b, payload);
}

#[test]
#[cfg_attr(miri, ignore = "2 MiB stream is impractical under miri")]
fn mt_padding_aligns_total_size() {
    // Encode with padding=1024 and verify the resulting stream is a
    // multiple of 1024 bytes.  Decoding must still produce the original
    // payload (decoder skips the 0xfe padding chunk).
    let payload: Vec<u8> = (0..(2 << 20)).map(|i| (i as u8).wrapping_mul(7)).collect();
    let mut w = MtWriterBuilder::new()
        .block_size(64 << 10)
        .concurrency(4)
        .padding(1024)
        .build(Vec::<u8>::new())
        .unwrap();
    w.write_all(&payload).unwrap();
    let buf = w.finish().unwrap();
    assert_eq!(
        buf.len() % 1024,
        0,
        "stream len {} not a multiple of 1024",
        buf.len()
    );
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&buf))
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, payload);
}

#[test]
#[cfg_attr(miri, ignore = "1 MiB stream is impractical under miri")]
fn mt_padding_plus_index_aligns_and_reads() {
    // padding + append_index — padding precedes the index but its size
    // accounts for the index length, so the final stream is aligned.
    let payload: Vec<u8> = (0..(1 << 20)).map(|i| (i as u8).wrapping_mul(13)).collect();
    let multiple = 4096u32;
    let mut w = MtWriterBuilder::new()
        .block_size(64 << 10)
        .concurrency(4)
        .padding(multiple)
        .append_index()
        .build(Vec::<u8>::new())
        .unwrap();
    w.write_all(&payload).unwrap();
    let buf = w.finish().unwrap();
    assert_eq!(
        buf.len() % multiple as usize,
        0,
        "stream len {} not a multiple of {}",
        buf.len(),
        multiple
    );
    // ReadSeeker still finds the index at the tail.
    let reader = Reader::new(Cursor::new(buf));
    let mut rs = ReadSeeker::new(reader).expect("ReadSeeker::new");
    assert_eq!(rs.index().total_uncompressed(), Some(payload.len() as u64));
    // total_compressed must be unknown (None) when padding is in play.
    assert_eq!(rs.index().total_compressed(), None);
    // Random read still works.
    let mut buf = [0u8; 512];
    let off = (payload.len() / 2) as u64;
    let n = rs.read_at(&mut buf, off).unwrap();
    assert_eq!(&buf[..n], &payload[off as usize..off as usize + n]);
}

#[test]
#[cfg_attr(miri, ignore = "3 MiB stream is impractical under miri")]
fn mt_index_offsets_match_st_index_offsets() {
    // For an identical input + identical block layout, ST and MT
    // writers must produce indices that agree on every (comp, uncomp)
    // pair.  Run at concurrency=4 to exercise the writer-thread index
    // accumulation across worker submissions.
    let payload: Vec<u8> = (0..(3 << 20))
        .map(|i| ((i * 17) ^ (i >> 4)) as u8)
        .collect();
    let block_size = 64 << 10;

    let mut st = crate::stream::WriterBuilder::new()
        .block_size(block_size)
        .build(Vec::<u8>::new())
        .unwrap();
    st.write_all(&payload).unwrap();
    let (_, st_idx) = st.close_index().unwrap();

    let mut mt = MtWriterBuilder::new()
        .block_size(block_size)
        .concurrency(4)
        .append_index()
        .build(Vec::<u8>::new())
        .unwrap();
    mt.write_all(&payload).unwrap();
    let mt_stream = mt.finish().unwrap();

    // Extract the MT-emitted index by reading the tail.
    let reader = Reader::new(Cursor::new(mt_stream.clone()));
    let rs = ReadSeeker::new(reader).unwrap();
    let mt_idx = rs.index().clone();

    let mut parsed_st = Index::default();
    parsed_st.load(&st_idx).unwrap();

    assert_eq!(
        parsed_st.total_uncompressed(),
        mt_idx.total_uncompressed(),
        "total uncompressed mismatch"
    );
    // ST and MT must enumerate the same blocks at the same uncomp
    // offsets (compressed offsets diverge by the stream header byte
    // count differences if any — but ST and MT use the same header).
    assert_eq!(
        parsed_st.offsets().len(),
        mt_idx.offsets().len(),
        "entry count mismatch"
    );
    for (a, b) in parsed_st.offsets().iter().zip(mt_idx.offsets().iter()) {
        assert_eq!(a.uncompressed, b.uncompressed, "uncomp offset");
        assert_eq!(a.compressed, b.compressed, "comp offset");
    }
}

#[test]
#[cfg_attr(miri, ignore = "1 MiB stream is impractical under miri")]
fn mt_close_index_matches_st_close_index() {
    let payload: Vec<u8> = (0..(1 << 20))
        .map(|i| ((i * 17) ^ (i >> 4)) as u8)
        .collect();
    let mut st = crate::stream::WriterBuilder::new()
        .block_size(64 << 10)
        .build(Vec::<u8>::new())
        .unwrap();
    st.write_all(&payload).unwrap();
    let (_, st_idx) = st.close_index().unwrap();
    let build = || {
        MtWriterBuilder::new()
            .block_size(64 << 10)
            .concurrency(4)
            .build(Vec::<u8>::new())
            .unwrap()
    };
    let mut mt = build();
    mt.write_all(&payload).unwrap();
    let (stream, mt_idx) = mt.close_index().unwrap();
    assert_eq!(mt_idx, st_idx);
    let mut plain = build();
    plain.write_all(&payload).unwrap();
    assert_eq!(
        plain.finish().unwrap(),
        stream,
        "close_index changed the stream"
    );

    let off = MtWriterBuilder::new()
        .concurrency(1)
        .generate_index(false)
        .build(Vec::<u8>::new())
        .unwrap();
    assert_eq!(
        off.close_index().unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
}

// -------------------- engine reuse --------------------
// Each test uses its own block size, so parallel tests never share an engine.

fn idle_ids() -> Vec<u64> {
    IDLE.lock().unwrap().iter().map(|e| e.id).collect()
}

/// The writer's engine id, and a handle that dies with the engine.
fn engine_of<W: Write + Send + 'static>(w: &MtWriter<W>) -> (u64, Weak<BufferPool>) {
    (w.engine().id, Arc::downgrade(&w.engine().input_pool))
}

fn shaped(block_size: usize) -> MtWriter<Vec<u8>> {
    MtWriterBuilder::new()
        .block_size(block_size)
        .concurrency(2)
        .build(Vec::new())
        .unwrap()
}

fn decode(stream: &[u8]) -> Vec<u8> {
    let mut decoded = Vec::new();
    Reader::new(stream).read_to_end(&mut decoded).unwrap();
    decoded
}

/// Alive while `token` has more than one strong count.
struct Sink {
    fail: bool,
    _token: Arc<()>,
}

impl Write for Sink {
    fn write(&mut self, p: &[u8]) -> io::Result<usize> {
        if self.fail {
            return Err(io::Error::other("sink failed"));
        }
        Ok(p.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Queues a job for worker 0 the way `submit_block` does.
fn inject(w: &mut MtWriter<Vec<u8>>, uncompressed: Vec<u8>, obuf_pool: Arc<BufferPool>) {
    let (result_tx, result_rx) = mpsc::sync_channel(1);
    w.order_tx
        .as_ref()
        .unwrap()
        .send(OrderItem::Block(result_rx))
        .unwrap();
    let job = BlockJob {
        uncompressed,
        level: crate::block::Level::Balanced,
        uncompressed_mode: false,
        result_tx,
        obuf_pool,
    };
    w.engine().submit_txs[0].send(job).unwrap();
}

#[test]
#[cfg_attr(miri, ignore = "Miri never parks engines")]
fn engine_reused_by_same_shape() {
    const BS: usize = MIN_BLOCK_SIZE + 1;
    let payload: Vec<u8> = (0..3 * BS + 5).map(|i| (i / 7) as u8).collect();
    let mut w = shaped(BS);
    let (id, _) = engine_of(&w);
    w.write_all(&payload).unwrap();
    assert_eq!(decode(&w.finish().unwrap()), payload);
    assert!(idle_ids().contains(&id), "finished engine not parked");

    let mut w = shaped(BS);
    assert_eq!(engine_of(&w).0, id);
    assert!(!idle_ids().contains(&id));
    w.write_all(&payload).unwrap();
    assert_eq!(decode(&w.finish().unwrap()), payload);
}

#[test]
#[cfg_attr(miri, ignore = "Miri never parks engines")]
fn idle_engine_expires() {
    const BS: usize = MIN_BLOCK_SIZE + 2;
    let mut w = shaped(BS);
    let (id, pool) = engine_of(&w);
    w.write_all(b"expire").unwrap();
    w.finish().unwrap();
    assert!(idle_ids().contains(&id), "finished engine not parked");
    let start = Instant::now();
    while pool.strong_count() > 0 {
        assert!(start.elapsed() < 10 * ENGINE_IDLE, "engine never expired");
        std::thread::sleep(ENGINE_IDLE / 20);
    }
    assert!(!idle_ids().contains(&id));
}

#[test]
fn failing_writer_drops_engine() {
    const BS: usize = MIN_BLOCK_SIZE + 3;
    let token = Arc::new(());
    let sink = Sink {
        fail: true,
        _token: token.clone(),
    };
    let mut w = MtWriterBuilder::new()
        .block_size(BS)
        .concurrency(2)
        .build(sink)
        .unwrap();
    let (id, pool) = engine_of(&w);
    let _ = w.write_all(&vec![b'f'; 2 * BS]);
    assert!(w.finish().is_err());
    assert_eq!(Arc::strong_count(&token), 1, "W outlived finish");
    assert_eq!(pool.strong_count(), 0, "engine still alive");
    assert!(!idle_ids().contains(&id));
}

#[test]
#[cfg_attr(miri, ignore = "an 8 MiB block is impractical under miri")]
fn worker_error_drops_engine() {
    const BS: usize = MIN_BLOCK_SIZE + 4;
    let mut w = shaped(BS);
    let (id, pool) = engine_of(&w);
    w.write_all(b"before").unwrap();
    w.flush().unwrap();
    // Too large to encode: the worker reports an error.
    let obuf_pool = w.engine().obuf_pool.clone();
    inject(&mut w, vec![0; MAX_BLOCK_SIZE + 1], obuf_pool);
    let err = w.finish().unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    assert!(err.to_string().starts_with("encode: "), "{err}");
    assert_eq!(pool.strong_count(), 0, "engine still alive");
    assert!(!idle_ids().contains(&id));
}

#[test]
fn worker_panic_drops_engine() {
    const BS: usize = MIN_BLOCK_SIZE + 5;
    let mut w = shaped(BS);
    let (id, pool) = engine_of(&w);
    // Allocating an obuf of this capacity panics in the worker.
    inject(
        &mut w,
        b"boom".to_vec(),
        Arc::new(BufferPool::new(0, usize::MAX)),
    );
    let _ = w.write_all(&vec![b'p'; 2 * BS]);
    assert!(w.finish().is_err());
    assert_eq!(pool.strong_count(), 0, "engine still alive");
    assert!(!idle_ids().contains(&id));

    let mut w = shaped(BS);
    assert_ne!(engine_of(&w).0, id);
    w.write_all(b"after").unwrap();
    assert_eq!(decode(&w.finish().unwrap()), b"after");
}

#[test]
fn drop_without_finish_drops_engine_and_writer() {
    const BS: usize = MIN_BLOCK_SIZE + 6;
    let token = Arc::new(());
    let sink = Sink {
        fail: false,
        _token: token.clone(),
    };
    let mut w = MtWriterBuilder::new()
        .block_size(BS)
        .concurrency(2)
        .build(sink)
        .unwrap();
    let (id, pool) = engine_of(&w);
    w.write_all(&vec![b'd'; 5 * BS + 7]).unwrap();
    drop(w);
    assert_eq!(Arc::strong_count(&token), 1, "W outlived the writer");
    assert_eq!(pool.strong_count(), 0, "engine still alive");
    assert!(!idle_ids().contains(&id));
}

#[test]
#[cfg_attr(miri, ignore = "Miri never parks engines")]
fn concurrent_writers_share_bounded_engines() {
    const BS: usize = MIN_BLOCK_SIZE + 7;
    const THREADS: usize = 8;
    const ROUNDS: usize = 4;
    let ids = Mutex::new(HashSet::new());
    std::thread::scope(|s| {
        for t in 0..THREADS {
            let ids = &ids;
            s.spawn(move || {
                for r in 0..ROUNDS {
                    let payload: Vec<u8> = (0..3 * BS + 100 * t + r)
                        .map(|i| (i * 7 / (t + 1)) as u8)
                        .collect();
                    let mut w = shaped(BS);
                    ids.lock().unwrap().insert(engine_of(&w).0);
                    w.write_all(&payload).unwrap();
                    assert_eq!(decode(&w.finish().unwrap()), payload, "t={t} r={r}");
                }
            });
        }
    });
    // Writers only build an engine when none is parked, so at most one per
    // concurrent writer exists.
    let ids = ids.into_inner().unwrap();
    assert!(
        ids.len() <= THREADS,
        "{} engines for {THREADS} writers",
        ids.len()
    );
    let parked = IDLE
        .lock()
        .unwrap()
        .iter()
        .filter(|e| e.key == (2, BS))
        .count();
    assert!(parked <= THREADS, "{parked} parked engines");
}

#[test]
#[cfg_attr(miri, ignore = "leaks the inherited engine's threads")]
fn forked_child_skips_inherited_engines() {
    const BS: usize = MIN_BLOCK_SIZE + 8;
    let mut inherited = Engine::new((2, BS));
    inherited.pid = std::process::id().wrapping_add(1);
    let id = inherited.id;
    IDLE.lock().unwrap().push(inherited);
    let mut w = shaped(BS);
    assert_ne!(engine_of(&w).0, id);
    assert!(!idle_ids().contains(&id));
    w.write_all(b"child").unwrap();
    assert_eq!(decode(&w.finish().unwrap()), b"child");
}

// -------------------- flush --------------------

/// Records what reached it and how often it was flushed; each write is slow.
struct SlowSink(Arc<Mutex<(Vec<u8>, usize)>>);

impl Write for SlowSink {
    fn write(&mut self, p: &[u8]) -> io::Result<usize> {
        std::thread::sleep(std::time::Duration::from_millis(20));
        self.0.lock().unwrap().0.extend_from_slice(p);
        Ok(p.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.lock().unwrap().1 += 1;
        Ok(())
    }
}

#[test]
#[cfg_attr(miri, ignore = "300 KiB stream is impractical under miri")]
fn flush_reaches_sink() {
    let state = Arc::new(Mutex::new((Vec::new(), 0)));
    let mut w = MtWriterBuilder::new()
        .block_size(64 << 10)
        .concurrency(2)
        .generate_index(false)
        .build(SlowSink(state.clone()))
        .unwrap();
    let data: Vec<u8> = (0..300u32 << 10).map(|i| (i * 7 / 13) as u8).collect();
    w.write_all(&data).unwrap();
    w.flush().unwrap();
    let (at_flush, flushes) = {
        let s = state.lock().unwrap();
        (s.0.len(), s.1)
    };
    assert!(flushes >= 1, "W::flush not called");
    w.finish().unwrap();
    // finish only adds the EOF chunk: a 4-byte header and uvarint(300 KiB).
    assert_eq!(state.lock().unwrap().0.len() - at_flush, 7);
}

#[test]
fn flush_reports_sink_error() {
    let mut w = MtWriterBuilder::new()
        .block_size(MIN_BLOCK_SIZE + 9)
        .concurrency(2)
        .build(Sink {
            fail: true,
            _token: Arc::new(()),
        })
        .unwrap();
    w.write_all(b"data").unwrap();
    assert!(w.flush().is_err());
}
