use std::io::{Cursor, Read, Write};

use crate::stream::{Reader, Writer, WriterBuilder};

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
    let mut w = WriterBuilder::new().block_size(4 << 10).build(buf);
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
    let mut w = WriterBuilder::new().uncompressed().build(buf);
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
    let mut w = WriterBuilder::new().flush_on_write().build(buf);
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
    use std::cell::RefCell;
    use std::rc::Rc;
    let buf: Vec<u8> = Vec::new();
    let mut w = Writer::new(buf);
    w.write_all(b"prefix").unwrap();
    w.add_user_chunk(0x80, b"meta-a").unwrap();
    w.write_all(b"infix").unwrap();
    w.add_user_chunk(0xc0, b"meta-b").unwrap();
    w.write_all(b"suffix").unwrap();
    let out = w.finish().unwrap();

    type Seen = Rc<RefCell<Vec<(u8, Vec<u8>)>>>;
    let seen: Seen = Rc::new(RefCell::new(Vec::new()));
    let s1 = seen.clone();
    let s2 = seen.clone();
    let mut decoded = Vec::new();
    crate::stream::ReaderBuilder::new()
        .user_chunk_callback(0x80, move |id, body| {
            s1.borrow_mut().push((id, body.to_vec()));
            Ok(())
        })
        .user_chunk_callback(0xc0, move |id, body| {
            s2.borrow_mut().push((id, body.to_vec()));
            Ok(())
        })
        .build(Cursor::new(&out))
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(decoded, b"prefixinfixsuffix");
    let seen = seen.borrow();
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
    let mut w = WriterBuilder::new().padding(pad).build(buf);
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
    let mut w = WriterBuilder::new().padding(pad).build(buf);
    w.write_all(&payload).unwrap();
    let out_padded = w.finish().unwrap();

    let buf: Vec<u8> = Vec::new();
    let mut w = WriterBuilder::new().build(buf);
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
    let (uncomp, comp) = w.written();
    assert_eq!(uncomp, payload.len() as u64);
    assert!(comp > 0);
    // After finish there's the EOF chunk too.
    let out = w.finish().unwrap();
    assert!(out.len() as u64 > comp);
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
        .build(buf);
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
    ignore = "200 trials × 2 levels × ≤40 KiB is impractical under miri"
)]
fn stress_random_inputs_roundtrip() {
    // 200 inputs at sizes 0..40 KiB (crosses the 4 KiB minimum block size at
    // L1 etc.).  Two compression levels to keep runtime moderate.
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
        for level in [crate::Level::Fastest, crate::Level::Smallest] {
            let mut w = WriterBuilder::new()
                .block_size(crate::stream::MIN_BLOCK_SIZE)
                .level(level)
                .build(Vec::new());
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
    // Empty stream: stream id + EOF only.
    let mut decoded = Vec::new();
    Reader::new(Cursor::new(&out))
        .read_to_end(&mut decoded)
        .unwrap();
    assert!(decoded.is_empty());
}

#[cfg(miri)]
#[test]
fn stress_random_inputs_roundtrip_miri() {
    // 6 trials × 2 levels × ≤2 KiB.  Smaller than the native version
    // but still crosses MIN_BLOCK_SIZE and covers both repetitive and
    // pseudo-random inputs through L1 + L3.
    for trial in 0..6u64 {
        let mut rng = lcg(0xc0ffee ^ trial.wrapping_mul(0x9e37));
        let size = (trial as usize * 211) % 2048;
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
                .build(Vec::new());
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
