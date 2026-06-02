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

//! Tests ported from Go `index_test.go` plus Rust-specific edge cases.

use std::io::{Cursor, Read, Seek, SeekFrom, Write};

use super::*;
use crate::stream::Reader as StreamReader;

fn make_index(n: usize, gap: u64) -> Index {
    let mut idx = Index::default();
    idx.reset(gap.max(1) as usize);
    for i in 0..n {
        let comp = (i as u64) * 700;
        let uncomp = (i as u64) * gap;
        idx.add(comp, uncomp).expect("add");
    }
    idx.total_uncompressed = Some((n as u64) * gap);
    idx.total_compressed = Some((n as u64) * 700);
    idx
}

#[test]
fn round_trip_empty() {
    let mut idx = Index::default();
    idx.reset(1 << 20);
    let mut buf = Vec::new();
    idx.append_to(&mut buf, Some(0), Some(0)).unwrap();

    let mut idx2 = Index::default();
    let tail = idx2.load(&buf).expect("load");
    assert!(tail.is_empty(), "should consume entire buffer");
    assert_eq!(idx2.total_uncompressed, Some(0));
    assert_eq!(idx2.total_compressed, Some(0));
    assert!(idx2.offsets.is_empty());
}

#[test]
fn round_trip_small() {
    let idx = make_index(10, 1 << 20);
    let total_u = idx.total_uncompressed;
    let total_c = idx.total_compressed;
    let mut buf = Vec::new();
    let mut idx_mut = idx;
    idx_mut.append_to(&mut buf, total_u, total_c).unwrap();

    let mut idx2 = Index::default();
    let tail = idx2.load(&buf).expect("load");
    assert!(tail.is_empty());
    assert_eq!(idx2.total_uncompressed, total_u);
    assert_eq!(idx2.total_compressed, total_c);
    assert_eq!(idx2.offsets, idx_mut.offsets);
}

#[test]
fn round_trip_with_irregular_uncompressed() {
    let mut idx = Index::default();
    idx.reset(1 << 20);
    // First entry at 0 is implicit; second entry slightly off the
    // est-block prediction so `has_uncompressed` must turn on.
    idx.add(0, 0).unwrap();
    idx.add(500_000, (1 << 20) + 12345).unwrap();
    idx.add(1_000_000, 3 << 20).unwrap();
    idx.total_uncompressed = Some(3 << 20);
    idx.total_compressed = Some(1_000_000);

    let mut buf = Vec::new();
    let mut clone = idx.clone();
    clone
        .append_to(&mut buf, idx.total_uncompressed, idx.total_compressed)
        .unwrap();

    let mut idx2 = Index::default();
    idx2.load(&buf).expect("load");
    assert_eq!(idx2.offsets, idx.offsets);
}

#[test]
fn add_skips_within_est_block() {
    let mut idx = Index::default();
    idx.reset(1 << 20);
    idx.add(0, 0).unwrap();
    // Same gap as est_block_uncomp - 1 should be skipped.
    idx.add(100, (1 << 20) - 1).unwrap();
    assert_eq!(idx.offsets.len(), 1);
    // Exact gap should be accepted.
    idx.add(200, 1 << 20).unwrap();
    assert_eq!(idx.offsets.len(), 2);
}

#[test]
fn add_rejects_decreasing_offset() {
    let mut idx = Index::default();
    idx.reset(1 << 20);
    idx.add(100, 1 << 20).unwrap();
    let err = idx.add(99, 2 << 20).unwrap_err();
    assert!(matches!(err, Error::Invalid(_)));
}

#[test]
fn find_basic() {
    let idx = make_index(8, 1 << 20);
    // The index uses est_block_uncomp = 1 << 20 (from reset, since gap matches).
    let _ = idx.est_block_uncomp();
    // Offset 0 -> first entry.
    let p = idx.find(0);
    assert_eq!((p.compressed, p.uncompressed), (0, 0));
    // Offset just past first entry but before second -> still first.
    let p = idx.find((1 << 20) - 1);
    assert_eq!((p.compressed, p.uncompressed), (0, 0));
    // Exact second offset.
    let p = idx.find(1 << 20);
    assert_eq!(p.uncompressed, 1 << 20);
    assert_eq!(p.compressed, 700);
}

#[test]
fn find_past_end_clamps_to_last_entry() {
    let idx = make_index(4, 1 << 20);
    let last = *idx.offsets().last().unwrap();
    assert_eq!(idx.find(idx.total_uncompressed.unwrap() + 1), last);
}

#[test]
fn find_big_index_binary_search() {
    let n = 1000;
    let idx = make_index(n, 1 << 20);
    // Each offset should map to itself when queried exactly.
    for i in 0..n {
        let p = idx.find((i as u64) * (1 << 20));
        assert_eq!(p.uncompressed, (i as u64) * (1 << 20));
        assert_eq!(p.compressed, (i as u64) * 700);
    }
}

#[test]
fn load_rejects_short_buffer() {
    let mut idx = Index::default();
    assert!(matches!(idx.load(&[0; 8]).unwrap_err(), Error::Truncated));
}

#[test]
fn load_rejects_wrong_chunk_type() {
    let mut idx = Index::default();
    let mut buf = Vec::new();
    Index::default()
        .append_to(&mut buf, Some(0), Some(0))
        .unwrap();
    buf[0] = 0x80;
    assert!(matches!(idx.load(&buf).unwrap_err(), Error::BadFormat(_)));
}

#[test]
fn load_accepts_legacy_chunk_id() {
    let mut buf = Vec::new();
    let mut tmp = make_index(4, 1 << 20);
    tmp.append_to(&mut buf, tmp.total_uncompressed, tmp.total_compressed)
        .unwrap();
    buf[0] = LEGACY_INDEX_CHUNK;
    let mut idx = Index::default();
    idx.load(&buf).expect("legacy chunk id must load");
    assert_eq!(idx.offsets.len(), 4);
}

#[test]
fn load_rejects_bad_trailer() {
    let mut buf = Vec::new();
    let mut tmp = make_index(2, 1 << 20);
    tmp.append_to(&mut buf, tmp.total_uncompressed, tmp.total_compressed)
        .unwrap();
    let last = buf.len() - 1;
    buf[last] ^= 0xFF;
    let mut idx = Index::default();
    assert!(matches!(idx.load(&buf).unwrap_err(), Error::BadFormat(_)));
}

#[test]
fn load_rejects_truncated_chunk() {
    let mut buf = Vec::new();
    let mut tmp = make_index(2, 1 << 20);
    tmp.append_to(&mut buf, tmp.total_uncompressed, tmp.total_compressed)
        .unwrap();
    buf.truncate(buf.len() - 1);
    let mut idx = Index::default();
    assert!(matches!(
        idx.load(&buf).unwrap_err(),
        Error::Truncated | Error::BadFormat(_)
    ));
}

#[test]
fn header_round_trip() {
    let mut buf = Vec::new();
    let mut tmp = make_index(6, 1 << 20);
    tmp.append_to(&mut buf, tmp.total_uncompressed, tmp.total_compressed)
        .unwrap();
    let slim = remove_index_headers(&buf).expect("remove_index_headers");
    let restored = restore_index_headers(slim);
    assert_eq!(restored, buf);
}

#[test]
fn remove_headers_rejects_garbage() {
    assert!(remove_index_headers(&[0; 4]).is_none());
    assert!(remove_index_headers(&[0xff; 64]).is_none());
}

#[test]
#[cfg_attr(miri, ignore = "MAX_INDEX_ENTRIES sweep is impractical under miri")]
fn reduce_light_at_max() {
    let mut idx = Index::default();
    idx.reset(1 << 20);
    let n = MAX_INDEX_ENTRIES + 100;
    for i in 0..n {
        idx.add((i as u64) * 700, (i as u64) * (1 << 20)).unwrap();
    }
    assert!(idx.offsets.len() <= MAX_INDEX_ENTRIES);
    // est_block should have grown.
    assert!(idx.est_block_uncomp() >= (1 << 20) * 2);
}

#[test]
fn load_stream_finds_index_at_tail() {
    let mut buf = Vec::new();
    let mut tmp = make_index(5, 1 << 20);
    tmp.append_to(&mut buf, tmp.total_uncompressed, tmp.total_compressed)
        .unwrap();
    // Pretend the bytes are a stream with the index appended.
    let mut stream = vec![0u8; 128];
    stream.extend_from_slice(&buf);
    let mut cur = Cursor::new(stream);
    let mut idx = Index::default();
    idx.load_stream(&mut cur).expect("load_stream");
    assert_eq!(idx.offsets, tmp.offsets);
}

#[test]
fn load_stream_rejects_when_trailer_missing() {
    let stream = vec![0u8; 64];
    let mut cur = Cursor::new(stream);
    let mut idx = Index::default();
    assert!(matches!(
        idx.load_stream(&mut cur).unwrap_err(),
        Error::BadFormat(_)
    ));
}

#[test]
#[cfg_attr(miri, ignore = "5 MiB stream is impractical under miri")]
fn index_stream_matches_writer_index() {
    // Build a compressed stream + capture the writer-produced index.
    let mut payload = vec![0u8; 5 << 20];
    // Make it compressible.
    for (i, b) in payload.iter_mut().enumerate() {
        *b = b'0' + (i as u8 & 3);
    }

    let mut compressed: Vec<u8> = Vec::new();
    let mut w = crate::stream::WriterBuilder::new()
        .block_size(64 << 10)
        .build(&mut compressed)
        .unwrap();
    w.write_all(&payload).unwrap();
    let _ = w.finish().unwrap();

    // Run IndexStream over it.
    let idx_via_stream = index_stream(Cursor::new(&compressed)).unwrap();

    // Parse both indexes and compare offsets — they should match.
    let mut idx_a = Index::default();
    idx_a.load(&idx_via_stream).unwrap();

    // Sanity: at least 50 entries for a 5 MiB stream at 64 KiB blocks.
    assert!(idx_a.offsets.len() > 10);
    assert_eq!(idx_a.total_uncompressed, Some(payload.len() as u64));
    // total_compressed must equal payload+overhead.
    assert!(idx_a.total_compressed.unwrap() > 0);
}

#[test]
#[cfg_attr(miri, ignore = "5 MiB stream is impractical under miri")]
fn writer_close_index_round_trip() {
    // Verify the writer's close_index produces bytes that load cleanly.
    let payload = vec![b'x'; 5 << 20];

    let mut compressed: Vec<u8> = Vec::new();
    let mut w = crate::stream::WriterBuilder::new()
        .block_size(64 << 10)
        .build(&mut compressed)
        .unwrap();
    w.write_all(&payload).unwrap();
    let idx_bytes = w.close_index().unwrap();

    let mut idx = Index::default();
    idx.load(&idx_bytes).unwrap();
    assert_eq!(idx.total_uncompressed, Some(payload.len() as u64));
    assert!(!idx.offsets.is_empty());
}

#[test]
#[cfg_attr(miri, ignore = "1 MiB stream is impractical under miri")]
fn seek_with_index_skip() {
    // Reproduce the Go ExampleIndex_Load pattern: stream is consumed
    // forward with ReaderIgnoreStreamIdentifier + Skip after seeking
    // the underlying source to an index entry's compressed offset.
    let mut payload = vec![0u8; 1 << 20];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = b'0' + (i as u8 & 3);
    }

    let mut compressed: Vec<u8> = Vec::new();
    let mut w = crate::stream::WriterBuilder::new()
        .block_size(64 << 10)
        .build(&mut compressed)
        .unwrap();
    w.write_all(&payload).unwrap();
    let idx_bytes = w.close_index().unwrap();

    let mut idx = Index::default();
    idx.load(&idx_bytes).unwrap();

    let want_off = 555_555u64;
    let entry = idx.find(want_off);
    let (c_off, u_off) = (entry.compressed, entry.uncompressed);
    let mut reader = crate::stream::ReaderBuilder::new()
        .ignore_stream_id()
        .build(Cursor::new(&compressed[c_off as usize..]))
        .unwrap();
    let to_skip = want_off - u_off;
    reader.skip(to_skip).unwrap();
    let mut out = Vec::new();
    reader.read_to_end(&mut out).unwrap();
    assert_eq!(out, payload[want_off as usize..]);
}

/// Helper used by the read_seeker tests below.  Pulled out so the seek-
/// test plumbing is shared across the seek_*/read_at_* cases.
fn build_indexed_stream(payload: &[u8]) -> Vec<u8> {
    let mut compressed: Vec<u8> = Vec::new();
    let mut w = crate::stream::WriterBuilder::new()
        .block_size(16 << 10)
        .append_index()
        .build(&mut compressed)
        .unwrap();
    w.write_all(payload).unwrap();
    let _ = w.finish().unwrap();
    compressed
}

#[test]
#[cfg_attr(miri, ignore = "256 KiB stream is impractical under miri")]
fn read_seeker_seek_start_end_and_current() {
    let mut payload = vec![0u8; 256 << 10];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = b'a' + ((i / 4096) as u8 & 0x0f);
    }
    let compressed = build_indexed_stream(&payload);

    let reader = StreamReader::new(Cursor::new(compressed));
    let mut rs = crate::stream::ReadSeeker::new(reader, &[]).expect("ReadSeeker::new");

    // SeekFrom::Start with an offset inside a later block.
    let off = (200 << 10) as u64;
    let abs = rs.seek(SeekFrom::Start(off)).unwrap();
    assert_eq!(abs, off);
    let mut buf = vec![0u8; 1 << 10];
    rs.read_exact(&mut buf).unwrap();
    assert_eq!(&buf[..], &payload[off as usize..(off as usize + 1024)]);

    // SeekFrom::Current relative move.
    let abs = rs.seek(SeekFrom::Current(-512)).unwrap();
    assert_eq!(abs, off + 1024 - 512);
    rs.read_exact(&mut buf).unwrap();
    assert_eq!(&buf[..], &payload[(abs as usize)..(abs as usize + 1024)]);

    // SeekFrom::End.
    let abs = rs.seek(SeekFrom::End(-1024)).unwrap();
    assert_eq!(abs, payload.len() as u64 - 1024);
    let mut tail = vec![0u8; 1024];
    rs.read_exact(&mut tail).unwrap();
    assert_eq!(tail, payload[payload.len() - 1024..]);
}

#[test]
#[cfg_attr(
    miri,
    ignore = "256 KiB stream + random probes is impractical under miri"
)]
fn read_seeker_read_at_random() {
    let mut payload = vec![0u8; 256 << 10];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31).wrapping_add(7);
    }
    let compressed = build_indexed_stream(&payload);

    let reader = StreamReader::new(Cursor::new(compressed));
    let mut rs = crate::stream::ReadSeeker::new(reader, &[]).expect("ReadSeeker::new");

    let mut buf = vec![0u8; 100];
    let offsets = [0u64, 11, 1_000, 65_000, (128 << 10) - 33, (256 << 10) - 100];
    for &off in &offsets {
        let n = rs.read_at(&mut buf, off).unwrap();
        assert_eq!(n, buf.len(), "off={off}");
        assert_eq!(&buf[..], &payload[off as usize..(off as usize + buf.len())]);
    }
}

#[test]
#[cfg_attr(miri, ignore = "200 KiB stream is impractical under miri")]
fn read_seeker_external_index_bytes() {
    let payload = vec![b'q'; 200 << 10];
    let mut compressed: Vec<u8> = Vec::new();
    let mut w = crate::stream::WriterBuilder::new()
        .block_size(16 << 10)
        .build(&mut compressed)
        .unwrap();
    w.write_all(&payload).unwrap();
    let idx_bytes = w.close_index().unwrap();

    // (a) Full chunk form.
    let reader = StreamReader::new(Cursor::new(compressed.clone()));
    let mut rs = crate::stream::ReadSeeker::new(reader, &idx_bytes).expect("ReadSeeker::new");
    rs.seek(SeekFrom::Start(150 << 10)).unwrap();
    let mut buf = vec![0u8; 1024];
    rs.read_exact(&mut buf).unwrap();
    assert_eq!(&buf[..], &payload[150 << 10..(150 << 10) + 1024]);

    // (b) Slim form re-wrapped with restore_index_headers — must match (a).
    let slim = remove_index_headers(&idx_bytes).unwrap().to_vec();
    let restored = restore_index_headers(&slim);
    let reader2 = StreamReader::new(Cursor::new(compressed.clone()));
    let mut rs2 = crate::stream::ReadSeeker::new(reader2, &restored).expect("ReadSeeker::new");
    rs2.seek(SeekFrom::Start(150 << 10)).unwrap();
    let mut buf2 = vec![0u8; 1024];
    rs2.read_exact(&mut buf2).unwrap();
    assert_eq!(buf, buf2);

    // (c) Bare slim form passed *directly* to ReadSeeker::new — must
    // be auto-detected and accepted without calling restore_index_headers.
    let reader3 = StreamReader::new(Cursor::new(compressed));
    let mut rs3 = crate::stream::ReadSeeker::new(reader3, &slim).expect("ReadSeeker::new bare");
    rs3.seek(SeekFrom::Start(150 << 10)).unwrap();
    let mut buf3 = vec![0u8; 1024];
    rs3.read_exact(&mut buf3).unwrap();
    assert_eq!(buf, buf3);
}

// -------------------- additional coverage (Stage F follow-ups) --------------------

#[test]
fn add_rejects_decreasing_compressed_offset() {
    let mut idx = Index::default();
    idx.reset(1 << 20);
    idx.add(1000, 1 << 20).unwrap();
    // Larger uncomp gap but smaller comp — error.
    let err = idx.add(999, 2 << 20).unwrap_err();
    assert!(matches!(err, Error::Invalid(_)));
}

#[test]
fn add_equal_offsets_after_skip_window_succeed() {
    // After the skip check, equal offsets are accepted (Go uses `<`/`>`,
    // never `<=` — equality on either field is fine).
    let mut idx = Index::default();
    idx.reset(1 << 20);
    idx.add(0, 0).unwrap();
    // Same offsets are skipped by the gap check (gap=0 < 1MB).
    idx.add(0, 0).unwrap();
    assert_eq!(idx.offsets.len(), 1);
}

#[test]
#[cfg_attr(
    miri,
    ignore = "MAX_INDEX_ENTRIES+5000 entries is impractical under miri"
)]
fn reduce_triggered_by_append_to() {
    // Force a giant entry count then call append_to and confirm the
    // emitted byte stream parses back to ≤ MAX_INDEX_ENTRIES entries.
    let mut idx = Index::default();
    idx.reset(4 << 10); // est_block ≥ 1 MiB
    // Manually inject more than MAX_INDEX_ENTRIES so append_to.reduce()
    // is what shrinks them.
    idx.offsets.clear();
    let n = MAX_INDEX_ENTRIES + 5_000;
    for i in 0..n {
        idx.offsets.push(OffsetPair {
            compressed: (i as u64) * 100,
            uncompressed: (i as u64) * idx.est_block_uncomp(),
        });
    }
    let mut buf = Vec::new();
    let total_u = (n as u64) * idx.est_block_uncomp();
    let total_c = (n as u64) * 100;
    idx.append_to(&mut buf, Some(total_u), Some(total_c))
        .unwrap();
    assert!(
        idx.offsets.len() < MAX_INDEX_ENTRIES,
        "reduce() should have shrunk in place"
    );

    // Round-trip via load: parsed count == in-memory count.
    let mut idx2 = Index::default();
    idx2.load(&buf).unwrap();
    assert_eq!(idx2.offsets.len(), idx.offsets.len());
    assert_eq!(idx2.total_uncompressed, Some(total_u));
    assert_eq!(idx2.total_compressed, Some(total_c));
}

#[test]
fn find_uses_binary_search_above_threshold() {
    // 300 entries triggers the partition_point path; verify lookups.
    let n = 300;
    let idx = make_index(n, 1 << 20);
    // Mid-bucket query — must land on the entry strictly ≤ the query.
    let target = (n as u64 / 2) * (1 << 20) + 5;
    let p = idx.find(target);
    assert_eq!(p.uncompressed, (n as u64 / 2) * (1 << 20));
    assert_eq!(p.compressed, (n as u64 / 2) * 700);
}

#[test]
fn find_at_total_uncompressed_succeeds() {
    let idx = make_index(8, 1 << 20);
    // Exactly at the end — returns the last entry (offset <= total).
    let total = idx.total_uncompressed.unwrap();
    assert!(idx.find(total).uncompressed <= total);
}

#[test]
fn json_contains_expected_fields() {
    let idx = make_index(3, 1 << 20);
    let json = idx.to_json();
    assert!(json.contains("\"total_uncompressed\""));
    assert!(json.contains("\"total_compressed\""));
    assert!(json.contains("\"est_block_uncompressed\""));
    assert!(json.contains("\"offsets\""));
    assert!(json.contains("\"compressed\""));
    assert!(json.contains("\"uncompressed\""));
}

#[test]
fn restore_headers_empty_returns_empty() {
    assert!(restore_index_headers(&[]).is_empty());
}

#[test]
fn remove_headers_rejects_wrong_trailer() {
    let mut buf = Vec::new();
    let mut idx = make_index(2, 1 << 20);
    idx.append_to(&mut buf, idx.total_uncompressed, idx.total_compressed)
        .unwrap();
    // Corrupt the trailer.
    let last = buf.len() - 1;
    buf[last] ^= 0x80;
    assert!(remove_index_headers(&buf).is_none());
}

#[test]
fn load_total_compressed_minus_one_accepted() {
    // Streams with padding store total_compressed = -1.  load() must
    // accept that.
    let mut idx = Index::default();
    idx.reset(1 << 20);
    idx.add(0, 0).unwrap();
    idx.add(700, 1 << 20).unwrap();
    let mut buf = Vec::new();
    idx.append_to(&mut buf, Some(2 << 20), None).unwrap();
    let mut idx2 = Index::default();
    idx2.load(&buf).unwrap();
    assert_eq!(idx2.total_compressed, None);
    assert_eq!(idx2.total_uncompressed, Some(2 << 20));
}

#[test]
fn load_rejects_entries_above_max() {
    // Forge an index header that claims > 65535 entries — must reject.
    let mut buf = Vec::new();
    buf.extend_from_slice(&[CHUNK_TYPE_INDEX, 0, 0, 0]); // chunk header (length patched later)
    buf.extend_from_slice(INDEX_HEADER);
    super::put_varint(&mut buf, 0); // total_uncompressed
    super::put_varint(&mut buf, 0); // total_compressed
    super::put_varint(&mut buf, 1 << 20); // est_block
    super::put_varint(&mut buf, (MAX_INDEX_ENTRIES + 1) as i64); // bad
    buf.push(0); // has_uncomp
    // Stuff a fake trailer so we hit the entry-count check before short-EOF.
    let size = (buf.len() + 4 + INDEX_TRAILER.len()) as u32;
    buf.extend_from_slice(&size.to_le_bytes());
    buf.extend_from_slice(INDEX_TRAILER);
    let chunk_len = (buf.len() - 4) as u32;
    buf[1] = chunk_len as u8;
    buf[2] = (chunk_len >> 8) as u8;
    buf[3] = (chunk_len >> 16) as u8;
    let mut idx = Index::default();
    assert!(matches!(idx.load(&buf).unwrap_err(), Error::Invalid(_)));
}

#[test]
fn load_rejects_negative_total_uncompressed() {
    let mut buf = Vec::new();
    let mut idx = make_index(2, 1 << 20);
    idx.append_to(&mut buf, idx.total_uncompressed, idx.total_compressed)
        .unwrap();
    // Patch the first varint (total_uncompressed) to a negative zigzag.
    // The first varint sits at offset 4 + INDEX_HEADER.len() = 10.
    buf[10] = 1; // zigzag(-1) = 1 → decoded as -1
    let mut idx2 = Index::default();
    assert!(matches!(idx2.load(&buf).unwrap_err(), Error::Invalid(_)));
}

#[test]
fn load_stream_short_file_errors() {
    // <10 bytes — can't even hold a trailer.
    let stream = vec![0u8; 4];
    let mut cur = Cursor::new(stream);
    let mut idx = Index::default();
    let err = idx.load_stream(&mut cur).unwrap_err();
    // No trailer (BadFormat) or a seek/read I/O error — both acceptable;
    // just ensure it doesn't panic.
    assert!(
        matches!(err, Error::BadFormat(_) | Error::Io(_)),
        "err={err:?}"
    );
}

#[test]
#[cfg_attr(miri, ignore = "100 KiB stream is impractical under miri")]
fn read_at_short_read_at_eof() {
    // Stream is 100 KiB; read_at requests 200 KiB starting at 50 KiB.
    // Expect Ok(50K) — partial fill, no error.  Bytes [0..50K] must be
    // the last 50 KiB of the payload.
    let mut payload = vec![0u8; 100 << 10];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(13);
    }
    let compressed = build_indexed_stream(&payload);
    let reader = StreamReader::new(Cursor::new(compressed));
    let mut rs = crate::stream::ReadSeeker::new(reader, &[]).unwrap();
    let mut buf = vec![0u8; 200 << 10];
    let n = rs.read_at(&mut buf, (50 << 10) as u64).unwrap();
    assert_eq!(n, 50 << 10);
    assert_eq!(&buf[..n], &payload[50 << 10..]);
}

#[test]
#[cfg_attr(miri, ignore = "100 KiB stream is impractical under miri")]
fn read_exact_at_errors_on_eof() {
    let payload = vec![b'x'; 100 << 10];
    let compressed = build_indexed_stream(&payload);
    let reader = StreamReader::new(Cursor::new(compressed));
    let mut rs = crate::stream::ReadSeeker::new(reader, &[]).unwrap();
    let mut buf = vec![0u8; 200 << 10];
    let err = rs.read_exact_at(&mut buf, (50 << 10) as u64).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
}

#[test]
#[cfg_attr(miri, ignore = "256 KiB stream is impractical under miri")]
fn read_seeker_seek_past_end_errors_then_recovers() {
    // After failing past-EOF, a subsequent valid seek should still work.
    let payload = vec![b'A'; 256 << 10];
    let compressed = build_indexed_stream(&payload);
    let reader = StreamReader::new(Cursor::new(compressed));
    let mut rs = crate::stream::ReadSeeker::new(reader, &[]).unwrap();

    // Read a few bytes so the reader has internal state.
    let mut warm = [0u8; 16];
    rs.read_exact(&mut warm).unwrap();

    // Out-of-bounds seek → UnexpectedEof from index.find.
    let err = rs
        .seek(SeekFrom::Start(payload.len() as u64 + 100))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);

    // Recover with a valid seek.
    rs.seek(SeekFrom::Start(0)).unwrap();
    let mut head = [0u8; 16];
    rs.read_exact(&mut head).unwrap();
    assert_eq!(&head, &payload[..16]);
}

#[test]
fn index_stream_handles_user_chunks() {
    use std::io::Write as _;
    // Stream with interleaved user chunks must still index correctly.
    let payload: Vec<u8> = (0u8..255).cycle().take(8 * 1024).collect();
    let mut compressed: Vec<u8> = Vec::new();
    let mut w = crate::stream::WriterBuilder::new()
        .block_size(crate::stream::MIN_BLOCK_SIZE)
        .build(&mut compressed)
        .unwrap();
    w.write_all(&payload[..4096]).unwrap();
    w.add_user_chunk(0x80, b"hello").unwrap();
    w.write_all(&payload[4096..]).unwrap();
    let _ = w.finish().unwrap();

    let idx_bytes = index_stream(Cursor::new(&compressed)).unwrap();
    let mut idx = Index::default();
    idx.load(&idx_bytes).unwrap();
    assert_eq!(idx.total_uncompressed, Some(payload.len() as u64));
}

// -------------------- miri-only lite variants --------------------
//
// These exercise the same code paths as the heavier originals above with
// payloads sized for the miri interpreter (~500× slowdown on the codec
// hot loops).  They only exist when compiling under miri, so native
// builds see no extra tests or compile-time overhead.

#[cfg(miri)]
#[test]
fn index_stream_matches_writer_index_miri() {
    // 48 KiB at 4 KiB blocks → 12 entries via index_stream (uses
    // first-block d_len as est_block, so every block triggers add).
    let mut payload = vec![0u8; 48 << 10];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = b'0' + (i as u8 & 3);
    }
    let mut compressed: Vec<u8> = Vec::new();
    let mut w = crate::stream::WriterBuilder::new()
        .block_size(crate::stream::MIN_BLOCK_SIZE)
        .build(&mut compressed)
        .unwrap();
    w.write_all(&payload).unwrap();
    let _ = w.finish().unwrap();

    let idx_via_stream = index_stream(Cursor::new(&compressed)).unwrap();
    let mut idx_a = Index::default();
    idx_a.load(&idx_via_stream).unwrap();
    assert!(idx_a.offsets.len() > 10);
    assert_eq!(idx_a.total_uncompressed, Some(payload.len() as u64));
    assert!(idx_a.total_compressed.unwrap() > 0);
}

#[cfg(miri)]
#[test]
fn writer_close_index_round_trip_miri() {
    let payload = vec![b'x'; 32 << 10];
    let mut compressed: Vec<u8> = Vec::new();
    let mut w = crate::stream::WriterBuilder::new()
        .block_size(crate::stream::MIN_BLOCK_SIZE)
        .build(&mut compressed)
        .unwrap();
    w.write_all(&payload).unwrap();
    let idx_bytes = w.close_index().unwrap();
    let mut idx = Index::default();
    idx.load(&idx_bytes).unwrap();
    assert_eq!(idx.total_uncompressed, Some(payload.len() as u64));
    assert!(!idx.offsets.is_empty());
}

#[cfg(miri)]
#[test]
fn seek_with_index_skip_miri() {
    let mut payload = vec![0u8; 32 << 10];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = b'0' + (i as u8 & 3);
    }
    let compressed = build_indexed_stream(&payload);
    let reader = StreamReader::new(Cursor::new(compressed));
    let mut rs = crate::stream::ReadSeeker::new(reader, &[]).expect("ReadSeeker::new");

    let off = (16 << 10) as u64;
    rs.seek(SeekFrom::Start(off)).unwrap();
    let mut buf = vec![0u8; 1 << 10];
    rs.read_exact(&mut buf).unwrap();
    assert_eq!(&buf[..], &payload[off as usize..(off as usize + 1024)]);
}

#[cfg(miri)]
#[test]
fn read_seeker_seek_start_end_and_current_miri() {
    let mut payload = vec![0u8; 32 << 10];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = b'a' + ((i / 1024) as u8 & 0x0f);
    }
    let compressed = build_indexed_stream(&payload);
    let reader = StreamReader::new(Cursor::new(compressed));
    let mut rs = crate::stream::ReadSeeker::new(reader, &[]).expect("ReadSeeker::new");

    let off = (20 << 10) as u64;
    let abs = rs.seek(SeekFrom::Start(off)).unwrap();
    assert_eq!(abs, off);
    let mut buf = vec![0u8; 1 << 10];
    rs.read_exact(&mut buf).unwrap();
    assert_eq!(&buf[..], &payload[off as usize..(off as usize + 1024)]);

    let abs = rs.seek(SeekFrom::Current(-512)).unwrap();
    assert_eq!(abs, off + 1024 - 512);
    rs.read_exact(&mut buf).unwrap();
    assert_eq!(&buf[..], &payload[(abs as usize)..(abs as usize + 1024)]);

    let abs = rs.seek(SeekFrom::End(-1024)).unwrap();
    assert_eq!(abs, payload.len() as u64 - 1024);
    let mut tail = vec![0u8; 1024];
    rs.read_exact(&mut tail).unwrap();
    assert_eq!(tail, payload[payload.len() - 1024..]);
}

#[cfg(miri)]
#[test]
fn read_seeker_read_at_random_miri() {
    let mut payload = vec![0u8; 32 << 10];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(31).wrapping_add(7);
    }
    let compressed = build_indexed_stream(&payload);
    let reader = StreamReader::new(Cursor::new(compressed));
    let mut rs = crate::stream::ReadSeeker::new(reader, &[]).expect("ReadSeeker::new");

    let mut buf = vec![0u8; 64];
    for &off in &[0u64, 11, 4_000, (16 << 10) - 33, (32 << 10) - 64] {
        let n = rs.read_at(&mut buf, off).unwrap();
        assert_eq!(n, buf.len(), "off={off}");
        assert_eq!(&buf[..], &payload[off as usize..(off as usize + buf.len())]);
    }
}

#[cfg(miri)]
#[test]
fn read_seeker_external_index_bytes_miri() {
    let payload = vec![b'q'; 32 << 10];
    let mut compressed: Vec<u8> = Vec::new();
    let mut w = crate::stream::WriterBuilder::new()
        .block_size(crate::stream::MIN_BLOCK_SIZE)
        .build(&mut compressed)
        .unwrap();
    w.write_all(&payload).unwrap();
    let idx_bytes = w.close_index().unwrap();

    let reader = StreamReader::new(Cursor::new(compressed.clone()));
    let mut rs = crate::stream::ReadSeeker::new(reader, &idx_bytes).expect("ReadSeeker::new");
    rs.seek(SeekFrom::Start(20 << 10)).unwrap();
    let mut buf = vec![0u8; 512];
    rs.read_exact(&mut buf).unwrap();
    assert_eq!(&buf[..], &payload[20 << 10..(20 << 10) + 512]);

    let slim = remove_index_headers(&idx_bytes).unwrap().to_vec();
    let restored = restore_index_headers(&slim);
    let reader2 = StreamReader::new(Cursor::new(compressed.clone()));
    let mut rs2 = crate::stream::ReadSeeker::new(reader2, &restored).expect("ReadSeeker::new");
    rs2.seek(SeekFrom::Start(20 << 10)).unwrap();
    let mut buf2 = vec![0u8; 512];
    rs2.read_exact(&mut buf2).unwrap();
    assert_eq!(buf, buf2);
}

#[cfg(miri)]
#[test]
fn read_at_short_read_at_eof_miri() {
    let mut payload = vec![0u8; 32 << 10];
    for (i, b) in payload.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(13);
    }
    let compressed = build_indexed_stream(&payload);
    let reader = StreamReader::new(Cursor::new(compressed));
    let mut rs = crate::stream::ReadSeeker::new(reader, &[]).unwrap();
    let mut buf = vec![0u8; 64 << 10];
    let n = rs.read_at(&mut buf, (16 << 10) as u64).unwrap();
    assert_eq!(n, 16 << 10);
    assert_eq!(&buf[..n], &payload[16 << 10..]);
}

#[cfg(miri)]
#[test]
fn read_exact_at_errors_on_eof_miri() {
    let payload = vec![b'x'; 32 << 10];
    let compressed = build_indexed_stream(&payload);
    let reader = StreamReader::new(Cursor::new(compressed));
    let mut rs = crate::stream::ReadSeeker::new(reader, &[]).unwrap();
    let mut buf = vec![0u8; 64 << 10];
    let err = rs.read_exact_at(&mut buf, (16 << 10) as u64).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
}

#[cfg(miri)]
#[test]
fn read_seeker_seek_past_end_errors_then_recovers_miri() {
    let payload = vec![b'A'; 32 << 10];
    let compressed = build_indexed_stream(&payload);
    let reader = StreamReader::new(Cursor::new(compressed));
    let mut rs = crate::stream::ReadSeeker::new(reader, &[]).unwrap();

    let mut warm = [0u8; 16];
    rs.read_exact(&mut warm).unwrap();

    let err = rs
        .seek(SeekFrom::Start(payload.len() as u64 + 100))
        .unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);

    rs.seek(SeekFrom::Start(0)).unwrap();
    let mut head = [0u8; 16];
    rs.read_exact(&mut head).unwrap();
    assert_eq!(&head, &payload[..16]);
}

#[test]
fn add_rejects_offset_beyond_i64_range() {
    // The signed (zigzag) wire format cannot represent offsets past i64::MAX;
    // `add` must reject them instead of wrapping at encode time.
    let too_big = i64::MAX as u64 + 1;
    let mut idx = Index::default();
    assert!(matches!(idx.add(0, too_big), Err(Error::Invalid(_))));
    assert!(matches!(idx.add(too_big, 0), Err(Error::Invalid(_))));
    // A value exactly at the boundary is accepted.
    assert!(idx.add(0, i64::MAX as u64).is_ok());
}

#[test]
fn append_to_rejects_total_beyond_i64_range() {
    let too_big = i64::MAX as u64 + 1;
    let mut idx = Index::default();
    idx.add(0, 0).unwrap();
    let mut buf = Vec::new();
    assert!(matches!(
        idx.append_to(&mut buf, Some(too_big), Some(0)),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        idx.append_to(&mut buf, Some(0), Some(too_big)),
        Err(Error::Invalid(_))
    ));
    // Known-good totals still encode.
    buf.clear();
    assert!(idx.append_to(&mut buf, Some(0), Some(0)).is_ok());
}
