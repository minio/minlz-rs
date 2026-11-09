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

use std::cell::RefCell;
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::rc::Rc;

use crate::Error as BlockError;
use crate::block;
use crate::stream::crc::masked_crc32c;
use crate::stream::format::{
    CHECKSUM_SIZE, CHUNK_HEADER_SIZE, CHUNK_TYPE_EOF, CHUNK_TYPE_MINLZ_COMPRESSED_DATA,
    CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC, CHUNK_TYPE_MINLZ_SIMD, CHUNK_TYPE_PADDING,
    CHUNK_TYPE_UNCOMPRESSED_DATA, DEFAULT_BLOCK_SIZE, MAGIC_BODY_LEN, MAX_BLOCK_SIZE,
    STREAM_FLAG_SIMD, get_uvarint, make_stream_header, put_uvarint, read_chunk_len,
};
use crate::stream::{Error, MtWriterBuilder, ReadSeeker, Reader, ReaderBuilder, WriterBuilder};

// -------------------- builders --------------------

fn push_chunk(out: &mut Vec<u8>, chunk_type: u8, payload: &[u8]) {
    let len = payload.len();
    assert!(len < 1 << 24);
    out.push(chunk_type);
    out.push(len as u8);
    out.push((len >> 8) as u8);
    out.push((len >> 16) as u8);
    out.extend_from_slice(payload);
}

fn push_eof(out: &mut Vec<u8>, total_uncompressed: u64) {
    let mut varint = [0u8; 10];
    let n = put_uvarint(&mut varint, total_uncompressed);
    push_chunk(out, CHUNK_TYPE_EOF, &varint[..n]);
}

fn push_uncompressed_block(out: &mut Vec<u8>, payload: &[u8]) {
    let mut body = Vec::with_capacity(CHECKSUM_SIZE + payload.len());
    body.extend_from_slice(&masked_crc32c(payload).to_le_bytes());
    body.extend_from_slice(payload);
    push_chunk(out, CHUNK_TYPE_UNCOMPRESSED_DATA, &body);
}

fn push_minlz_block(out: &mut Vec<u8>, payload: &[u8]) {
    let mut chunk_body = Vec::new();
    let n = block::encode_chunk_body(&mut chunk_body, 0, payload, block::Level::Balanced)
        .expect("encode");
    assert!(n > 0, "test payload should compress");
    chunk_body.truncate(n);
    let mut body = Vec::with_capacity(CHECKSUM_SIZE + chunk_body.len());
    body.extend_from_slice(&masked_crc32c(payload).to_le_bytes());
    body.extend_from_slice(&chunk_body);
    push_chunk(out, CHUNK_TYPE_MINLZ_COMPRESSED_DATA, &body);
}

fn build_stream(blocks: &[&[u8]]) -> Vec<u8> {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    let mut total = 0u64;
    for b in blocks {
        push_minlz_block(&mut s, b);
        total += b.len() as u64;
    }
    push_eof(&mut s, total);
    s
}

fn build_stream_uncompressed(blocks: &[&[u8]]) -> Vec<u8> {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    let mut total = 0u64;
    for b in blocks {
        push_uncompressed_block(&mut s, b);
        total += b.len() as u64;
    }
    push_eof(&mut s, total);
    s
}

fn decode_to_end<R: Read>(mut r: R) -> io::Result<Vec<u8>> {
    let mut v = Vec::new();
    r.read_to_end(&mut v)?;
    Ok(v)
}

// -------------------- core round-trip --------------------

#[test]
fn empty_stream_decodes_to_empty() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    push_eof(&mut s, 0);
    let got = decode_to_end(Reader::new(Cursor::new(&s))).unwrap();
    assert!(got.is_empty());
}

#[test]
fn single_uncompressed_block() {
    let payload = b"Hello, MinLZ!".to_vec();
    let s = build_stream_uncompressed(&[&payload]);
    let got = decode_to_end(Reader::new(Cursor::new(&s))).unwrap();
    assert_eq!(got, payload);
}

#[test]
fn single_compressed_block() {
    // A highly-compressible payload.
    let payload = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_vec();
    let s = build_stream(&[&payload]);
    let got = decode_to_end(Reader::new(Cursor::new(&s))).unwrap();
    assert_eq!(got, payload);
}

#[test]
fn multi_block_round_trip() {
    let payloads: Vec<Vec<u8>> = (0..4)
        .map(|i| {
            let mut v = Vec::new();
            for j in 0..200 {
                v.extend_from_slice(format!("block{i}-line{j}\n").as_bytes());
            }
            v
        })
        .collect();
    let refs: Vec<&[u8]> = payloads.iter().map(|v| v.as_slice()).collect();
    let s = build_stream(&refs);
    let got = decode_to_end(Reader::new(Cursor::new(&s))).unwrap();
    let expected: Vec<u8> = payloads.into_iter().flatten().collect();
    assert_eq!(got, expected);
}

// -------------------- error paths --------------------

#[test]
fn rejects_truncated_stream_header() {
    let s = b"\xff\x06\x00\x00MinL"; // 9 bytes, magic short by 1
    let err = Reader::new(Cursor::new(&s[..]))
        .read(&mut [0u8; 1])
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn rejects_wrong_magic() {
    let mut s = Vec::new();
    s.extend_from_slice(&[0xff, 0x06, 0x00, 0x00]);
    s.extend_from_slice(b"sNaPpY"); // Snappy magic
    let err = Reader::new(Cursor::new(&s))
        .read(&mut [0u8; 1])
        .unwrap_err();
    // Snappy fallback is out of scope -> Unsupported -> InvalidData.
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn rejects_data_before_header() {
    // Uncompressed-data chunk with no preceding stream identifier.
    let mut s = Vec::new();
    push_uncompressed_block(&mut s, b"oops");
    let err = Reader::new(Cursor::new(&s))
        .read(&mut [0u8; 1])
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn rejects_bad_crc() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    // Hand-write an uncompressed chunk with a deliberately wrong checksum.
    let payload = b"abcde";
    let bad_crc = 0xdeadbeef_u32;
    let mut body = Vec::new();
    body.extend_from_slice(&bad_crc.to_le_bytes());
    body.extend_from_slice(payload);
    push_chunk(&mut s, CHUNK_TYPE_UNCOMPRESSED_DATA, &body);
    push_eof(&mut s, payload.len() as u64);
    let err = Reader::new(Cursor::new(&s))
        .read(&mut [0u8; 8])
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn ignore_crc_skips_check() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    let payload = b"abcde";
    let bad_crc = 0xdeadbeef_u32;
    let mut body = Vec::new();
    body.extend_from_slice(&bad_crc.to_le_bytes());
    body.extend_from_slice(payload);
    push_chunk(&mut s, CHUNK_TYPE_UNCOMPRESSED_DATA, &body);
    push_eof(&mut s, payload.len() as u64);
    let got = decode_to_end(
        ReaderBuilder::new()
            .ignore_crc()
            .build(Cursor::new(&s))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(got, payload);
}

#[test]
fn eof_with_wrong_total_size() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    push_uncompressed_block(&mut s, b"hello");
    // EOF declares 100 bytes; we wrote 5.
    push_eof(&mut s, 100);
    let mut buf = Vec::new();
    let err = Reader::new(Cursor::new(&s))
        .read_to_end(&mut buf)
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn reserved_unskippable_errors() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    push_chunk(&mut s, 0x04, b"dummy"); // reserved unskippable
    let err = Reader::new(Cursor::new(&s))
        .read(&mut [0u8; 1])
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn user_non_skippable_without_callback_errors() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    push_chunk(&mut s, 0xc0, b"unskippable"); // user non-skippable
    let err = Reader::new(Cursor::new(&s))
        .read(&mut [0u8; 1])
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

/// Fails once with `kind` when it reaches `fail_at`, then serves the rest.
struct FailOnce<'a> {
    data: &'a [u8],
    pos: usize,
    fail_at: usize,
    kind: io::ErrorKind,
    failed: bool,
}

impl Read for FailOnce<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if !self.failed && self.pos == self.fail_at {
            self.failed = true;
            return Err(io::Error::new(self.kind, std::fmt::Error));
        }
        let end = if self.failed {
            self.data.len()
        } else {
            self.fail_at
        };
        let n = buf.len().min(end - self.pos);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

impl Seek for FailOnce<'_> {
    fn seek(&mut self, _: SeekFrom) -> io::Result<u64> {
        unreachable!("seeks within the decoded block do no source I/O")
    }
}

/// A chunk header read that fails part-way must keep failing: a retry would
/// parse the next header from the wrong offset.
#[test]
fn header_read_errors_are_sticky() {
    let payload: Vec<u8> = (0..10_000u32).map(|i| (i * 7) as u8).collect();
    let s = build_multi_block_stream(&payload, 4 << 10);
    // Stream identifier (10 bytes), then the first chunk's header and body.
    let first_len = s[11] as usize | (s[12] as usize) << 8 | (s[13] as usize) << 16;
    let fail_at = 10 + 4 + first_len + 2;
    for skip in [false, true] {
        let inner = FailOnce {
            data: &s,
            pos: 0,
            fail_at,
            kind: io::ErrorKind::TimedOut,
            failed: false,
        };
        let mut r = Reader::new(inner);
        let mut buf = vec![0u8; 4096];
        let err = if skip {
            r.skip(4096 + 10).unwrap_err()
        } else {
            r.read_exact(&mut buf).unwrap();
            r.read(&mut buf).unwrap_err()
        };
        assert_eq!(err.kind(), io::ErrorKind::TimedOut, "skip={skip}");
        assert!(
            err.get_ref().is_some_and(|e| e.is::<std::fmt::Error>()),
            "skip={skip}: first error keeps its payload"
        );
        let err = r.read(&mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut, "skip={skip}: retry");
    }

    // An `UnexpectedEof` from the source is corruption, as in a chunk body.
    // Seeking clears a sticky `UnexpectedEof` without moving the source, so
    // a read after a seek within the block would resume mid-header.
    let idx = crate::index::index_stream(&s[..]).unwrap();
    let inner = FailOnce {
        data: &s,
        pos: 0,
        fail_at,
        kind: io::ErrorKind::UnexpectedEof,
        failed: false,
    };
    let mut rs = ReadSeeker::new(Reader::new(inner), &idx).unwrap();
    let mut buf = vec![0u8; 4096];
    rs.read_exact(&mut buf).unwrap();
    let err = rs.read(&mut buf).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    let err = rs.seek(SeekFrom::Start(0)).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData, "seek");
    let err = rs.read(&mut buf).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData, "read after seek");

    // A truncated header after a complete stream must not turn into a
    // clean EOF on retry.
    let t = [&s[..], &[0xff, 0x06]].concat();
    let mut r = Reader::new(Cursor::new(&t));
    let err = r.read_to_end(&mut Vec::new()).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    let err = r.read(&mut [0u8; 1]).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    let mut r = Reader::new(Cursor::new(&t));
    let err = r.skip(payload.len() as u64 + 1).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData, "skip");
    let err = r.read(&mut [0u8; 1]).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData, "read after skip");
}

// -------------------- skippables --------------------

#[test]
fn padding_is_silently_skipped() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    push_chunk(&mut s, CHUNK_TYPE_PADDING, &[0u8; 32]);
    let payload = b"after-pad";
    push_uncompressed_block(&mut s, payload);
    push_chunk(&mut s, CHUNK_TYPE_PADDING, &[0u8; 16]);
    push_eof(&mut s, payload.len() as u64);
    let got = decode_to_end(Reader::new(Cursor::new(&s))).unwrap();
    assert_eq!(got, payload);
}

#[test]
fn reserved_skippable_is_silently_skipped() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    push_chunk(&mut s, 0x55, b"reserved"); // in 0x40..=0x7f
    let payload = b"after-skip";
    push_uncompressed_block(&mut s, payload);
    push_eof(&mut s, payload.len() as u64);
    let got = decode_to_end(Reader::new(Cursor::new(&s))).unwrap();
    assert_eq!(got, payload);
}

#[test]
fn user_skippable_without_callback_is_skipped() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    push_chunk(&mut s, 0x80, b"user-meta");
    let payload = b"data";
    push_uncompressed_block(&mut s, payload);
    push_eof(&mut s, payload.len() as u64);
    let got = decode_to_end(Reader::new(Cursor::new(&s))).unwrap();
    assert_eq!(got, payload);
}

#[test]
fn user_chunk_callback_receives_payload() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    push_chunk(&mut s, 0x80, b"hello-user");
    push_chunk(&mut s, 0xc0, b"non-skippable-data");
    let payload = b"data";
    push_uncompressed_block(&mut s, payload);
    push_eof(&mut s, payload.len() as u64);

    type Seen = Rc<RefCell<Vec<(u8, Vec<u8>)>>>;
    let seen: Seen = Rc::new(RefCell::new(Vec::new()));
    let seen2 = seen.clone();
    let seen3 = seen.clone();
    let reader = ReaderBuilder::new()
        .user_chunk_callback(0x80, move |id, body| {
            seen2.borrow_mut().push((id, body.to_vec()));
            Ok(())
        })
        .user_chunk_callback(0xc0, move |id, body| {
            seen3.borrow_mut().push((id, body.to_vec()));
            Ok(())
        })
        .build(Cursor::new(&s))
        .unwrap();
    let got = decode_to_end(reader).unwrap();
    assert_eq!(got, payload);
    let seen = seen.borrow();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0], (0x80, b"hello-user".to_vec()));
    assert_eq!(seen[1], (0xc0, b"non-skippable-data".to_vec()));
}

#[test]
fn user_chunk_callback_error_aborts_decode() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    push_chunk(&mut s, 0x80, b"trigger");
    push_uncompressed_block(&mut s, b"unreachable");
    push_eof(&mut s, 11);
    let reader = ReaderBuilder::new()
        .user_chunk_callback(0x80, |_, _| Err(io::Error::other("denied")))
        .build(Cursor::new(&s))
        .unwrap();
    let mut buf = Vec::new();
    let err = { reader }.read_to_end(&mut buf).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::Other);
}

// -------------------- options --------------------

#[test]
fn ignore_stream_id_allows_starting_mid_stream() {
    let mut s = Vec::new();
    push_uncompressed_block(&mut s, b"abc");
    // No EOF — but ignore_stream_id also disables EOF check, so we still
    // need a trailing EOF chunk; the reader returns its bytes and then
    // gracefully ends at the underlying EOF only if the EOF chunk was seen.
    // Add one with non-matching total — should be tolerated.
    push_eof(&mut s, 999);
    let got = decode_to_end(
        ReaderBuilder::new()
            .ignore_stream_id()
            .build(Cursor::new(&s))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(got, b"abc");
}

#[test]
fn max_block_size_rejects_oversized() {
    // Build a stream identifier declaring a small block size but follow with
    // an uncompressed block that exceeds the reader's option ceiling.
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(crate::stream::MIN_BLOCK_SIZE, false));
    // Reader's option is 4 KiB. Write an 8 KiB uncompressed block.
    push_uncompressed_block(&mut s, &vec![0u8; 8 << 10]);
    let err = ReaderBuilder::new()
        .max_block_size(crate::stream::MIN_BLOCK_SIZE)
        .build(Cursor::new(&s))
        .unwrap()
        .read(&mut [0u8; 4])
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn block_start_advances() {
    let payloads: [&[u8]; 3] = [b"AAA", b"BBBB", b"CCCCC"];
    let s = build_stream_uncompressed(&payloads);
    let mut reader = Reader::new(Cursor::new(&s));
    let mut buf = [0u8; 1];
    // First read: drains the first block, block_start moves to 0 (still
    // current block).
    reader.read_exact(&mut buf).unwrap();
    assert_eq!(buf[0], b'A');
    assert_eq!(reader.block_start(), 0);
    // Drain to the end of block 1: 2 more bytes in current block, then
    // reading more pulls block 2.
    let mut rest = [0u8; 2];
    reader.read_exact(&mut rest).unwrap();
    assert_eq!(&rest, b"AA");
    // Trigger advance to block 2.
    reader.read_exact(&mut buf).unwrap();
    assert_eq!(buf[0], b'B');
    assert_eq!(reader.block_start(), 3);
}

// -------------------- concatenation --------------------

#[test]
fn concatenated_streams() {
    let mut s = build_stream_uncompressed(&[b"first"]);
    s.extend_from_slice(&build_stream_uncompressed(&[b"second"]));
    let got = decode_to_end(Reader::new(Cursor::new(&s))).unwrap();
    assert_eq!(got, b"firstsecond");
}

#[test]
fn extra_bytes_after_eof_errors() {
    let mut s = build_stream_uncompressed(&[b"hello"]);
    // Append a truncated 4-byte header that isn't a valid stream identifier
    // and isn't EOF / stream-id of a follow-up stream.
    s.extend_from_slice(&[0x10, 0x00, 0x00, 0x00]); // reserved unskippable
    let err = Reader::new(Cursor::new(&s))
        .read_to_end(&mut Vec::new())
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

// -------------------- reset --------------------

#[test]
fn reset_reuses_state() {
    let s1 = build_stream_uncompressed(&[b"first"]);
    let s2 = build_stream_uncompressed(&[b"second"]);
    let mut reader = Reader::new(Cursor::new(&s1[..]));
    let mut out = Vec::new();
    reader.read_to_end(&mut out).unwrap();
    assert_eq!(out, b"first");

    reader.reset(Cursor::new(&s2[..]));
    let mut out = Vec::new();
    reader.read_to_end(&mut out).unwrap();
    assert_eq!(out, b"second");
}

// -------------------- fuzz regressions --------------------

/// Crash inputs found by `cargo fuzz run stream_decode_arbitrary` on
/// 2026-05-27.  Pre-fix root cause: `block::decoded_len_chunk_body`
/// returned `0` for the literal-inline sentinel (`varint(dlen) = 0`),
/// which then tripped `debug_assert_eq!(self.decoded.len(), dlen)` in
/// `handle_compressed_chunk`.  The decoder must accept these without
/// panicking and either return `Ok` or `Err`.
#[test]
fn fuzz_crashes_2026_05_27_do_not_panic() {
    use std::io::BufRead;
    let crashes: &[&[u8]] = &[
        &[
            0xff, 0x06, 0x00, 0x00, b'M', b'i', b'n', b'L', b'z', 0x00, 0x02, 0x06, 0x00, 0x00,
            0x02, 0x00, 0xff, 0x0f, 0x00, b'M', b'i', b'n', b'L', b'z', 0x00, 0x40, 0x00, 0x00,
            0x00, 0xff, 0x06, 0x30, 0x86, 0x00, b'M', b'i', b'n', b'L', b'z', 0x00, 0x01, 0x00,
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc6, 0xc6,
        ],
        &[
            0xff, 0x06, 0x00, 0x00, b'M', b'i', b'n', b'L', b'z', 0x29, 0x03, 0x0a, 0x00, 0x00,
            0x00, 0xf5, 0xb8, 0x03, 0x00, 0x06, 0x00, 0x00, b'M', b'i', b'n', b'L', b'z', 0x29,
            0x03, 0x0a, 0x00, 0x00, 0x00, 0xf5, 0xb8, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00,
        ],
        &[
            0xff, 0x06, 0x00, 0x00, b'M', b'i', b'n', b'L', b'z', 0x04, 0x02, 0x06, 0x00, 0x00,
            b'M', b'i', b'n', b'L', 0x00, 0x80, 0x00, 0x20, 0x09, 0x00, 0x00, 0xe8, 0xff, 0xff,
            0xff, 0xff, 0xfe, 0xa1, 0xb1, 0x3a, b'z', 0x04, 0x02, 0x00, 0xc6, 0x00, 0xc6, 0x86,
        ],
        &{
            let mut v = vec![
                0xff, 0x06, 0x00, 0x00, b'M', b'i', b'n', b'L', b'z', 0x00, 0x03, 0x3c, 0x00, 0x00,
                0x00, 0x00, 0x3c, 0x00, 0x00, 0x00,
            ];
            v.extend(std::iter::repeat_n(0u8, 140));
            v
        },
        &[
            0x41, 0x00, 0x00, 0x00, 0xbd, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0xff, 0x06,
            0x00, 0x00, b'M', b'i', b'n', b'L', b'z', 0x00, 0x03, 0x3c, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0xc4, 0x00, 0x01, 0x00, 0xc6, 0xbd, 0x00, 0x00, 0x00, 0xbd, 0x00,
            0x00, 0x00, 0xbd, 0x00, 0x00, 0x00, 0xbd, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00,
            0xbd, 0x00, 0x00, 0x00, 0xbd, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x8f, 0x00,
            0x00, 0x00, 0xbd, 0x00, 0x00, 0x00, 0xbd, 0x00, 0x00, 0x00, 0xbd, 0x00, 0x00, 0x00,
            0x41, 0x00, 0x00, 0x00, 0xbd, 0x00, 0x00, 0x00, 0xbd, 0x00, 0x00, 0x00, 0xbd, 0x00,
            0x00, 0x00, 0xbd, 0x00, 0x00, 0x00, 0xbd, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00,
            0xbd, 0x00, 0x00, 0x00, 0xbd, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0x8f, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x00, 0xd7, 0x00, 0x00,
        ],
        &[
            0xff, 0x06, 0x00, 0x00, b'M', b'i', b'n', b'L', b'z', 0x00, 0x03, 0x08, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
        ],
        &[
            0xff, 0x06, 0x00, 0x00, b'M', b'i', b'n', b'L', b'z', 0x01, 0x03, 0x11, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x00, b'M', b'i',
            b'n', b'L', b'z', 0x00, 0x01, 0xff, 0x1d, 0xc6,
        ],
        &[
            0xff, 0x06, 0x00, 0x00, b'M', b'i', b'n', b'L', b'z', 0x29, 0x03, 0x0a, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf5, 0xb8, 0x03,
            0x00, 0x00, 0x00, 0x00,
        ],
    ];
    for (i, input) in crashes.iter().enumerate() {
        // Both read paths must not panic.  Result may be Ok or Err.
        let mut out = Vec::new();
        let _ = Reader::new(*input).read_to_end(&mut out);
        let mut r = Reader::new(*input);
        while let Ok(buf) = r.fill_buf() {
            let n = buf.len();
            if n == 0 {
                break;
            }
            r.consume(n);
        }
        let _ = i; // silence in case the loop body changes
    }
}

// -------------------- cross-impl smoke --------------------

#[test]
fn decodes_go_generated_stream() {
    // 67 'a' bytes compressed by `cmd/mz c -cpu=1 -1` (Go encoder, single
    // thread, fastest level).  The trailing 0x40 chunk is Go's index chunk
    // (silently skipped by stage-C reader).
    let bytes = decode_hex(
        "ff0600004d696e4c7a0d02090000a91e7170430061ec24200100004340\
         1a000073326964780086013880808008020014 1e00000000786469 3273",
    );
    let got = decode_to_end(Reader::new(Cursor::new(&bytes))).unwrap();
    assert_eq!(got, vec![b'a'; 67]);
}

fn decode_hex(s: &str) -> Vec<u8> {
    let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    let mut out = Vec::with_capacity(cleaned.len() / 2);
    let bytes = cleaned.as_bytes();
    for pair in bytes.chunks(2) {
        let s = std::str::from_utf8(pair).unwrap();
        out.push(u8::from_str_radix(s, 16).unwrap());
    }
    out
}

// -------------------- skip() tests --------------------

fn build_multi_block_stream(payload: &[u8], block_size: usize) -> Vec<u8> {
    use std::io::Write as _;
    let mut compressed: Vec<u8> = Vec::new();
    let mut w = crate::stream::WriterBuilder::new()
        .block_size(block_size)
        .build(&mut compressed)
        .unwrap();
    w.write_all(payload).unwrap();
    let _ = w.finish().unwrap();
    compressed
}

#[test]
fn skip_zero_is_noop() {
    let payload = b"hello, MinLZ".to_vec();
    let stream = build_multi_block_stream(&payload, crate::stream::MIN_BLOCK_SIZE);
    let mut reader = Reader::new(Cursor::new(stream));
    reader.skip(0).unwrap();
    let mut out = Vec::new();
    reader.read_to_end(&mut out).unwrap();
    assert_eq!(out, payload);
}

#[test]
fn skip_within_first_block_no_io() {
    // After buffering the first block via fill_buf, skip(k) where k < block
    // should not advance the underlying reader past the first chunk body.
    let payload: Vec<u8> = (0u8..255).cycle().take(8 * 1024).collect();
    let stream = build_multi_block_stream(&payload, crate::stream::MIN_BLOCK_SIZE);
    let mut reader = Reader::new(Cursor::new(stream));
    // Prime the buffer.
    let mut throwaway = [0u8; 1];
    reader.read_exact(&mut throwaway).unwrap();
    reader.skip(100).unwrap();
    let mut tail = Vec::new();
    reader.read_to_end(&mut tail).unwrap();
    assert_eq!(tail, payload[101..]);
}

#[test]
fn skip_across_block_boundary() {
    // Build payload of 3 blocks at 4 KiB.
    let payload: Vec<u8> = (0..12 * 1024).map(|i| (i as u8).wrapping_mul(17)).collect();
    let stream = build_multi_block_stream(&payload, 4 << 10);
    let mut reader = Reader::new(Cursor::new(stream));
    reader.skip(5_000).unwrap();
    let mut tail = Vec::new();
    reader.read_to_end(&mut tail).unwrap();
    assert_eq!(tail, payload[5_000..]);
}

#[test]
fn skip_entire_block_then_partial() {
    // 4 blocks, skip 2 full blocks + half of the third.
    let block_size = 4 << 10;
    let payload: Vec<u8> = (0..4 * block_size)
        .map(|i| (i as u8).wrapping_add(7))
        .collect();
    let stream = build_multi_block_stream(&payload, block_size);
    let mut reader = Reader::new(Cursor::new(stream));
    // Skip beyond first 2 blocks + 1 KiB into the third.
    reader.skip((2 * block_size + 1024) as u64).unwrap();
    let mut tail = Vec::new();
    reader.read_to_end(&mut tail).unwrap();
    assert_eq!(tail, payload[2 * block_size + 1024..]);
}

#[test]
fn skip_past_eof_errors() {
    let payload = vec![b'x'; 1024];
    let stream = build_multi_block_stream(&payload, crate::stream::MIN_BLOCK_SIZE);
    let mut reader = Reader::new(Cursor::new(stream));
    let err = reader.skip(payload.len() as u64 + 1).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
    // Subsequent reads continue to surface the sticky error.
    let mut buf = [0u8; 1];
    let err2 = reader.read(&mut buf).unwrap_err();
    assert_eq!(err2.kind(), io::ErrorKind::UnexpectedEof);
}

#[test]
fn skip_across_concatenated_streams() {
    let payload: Vec<u8> = (0..10_000u32).map(|i| (i * 7) as u8).collect();
    let one = build_multi_block_stream(&payload, 4 << 10);
    let both = [one.clone(), one].concat();
    let want = [&payload[..], &payload[..]].concat();
    for n in [9_999, 10_000, 10_001, 15_000, 20_000] {
        let mut reader = Reader::new(Cursor::new(&both));
        reader.skip(n as u64).unwrap();
        let mut tail = Vec::new();
        reader.read_to_end(&mut tail).unwrap();
        assert_eq!(tail, want[n..], "skip({n})");
    }
    let err = Reader::new(Cursor::new(&both)).skip(20_001).unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::UnexpectedEof);
}

#[test]
fn skip_then_read_matches_full_decode() {
    // Differential: full decode vs (skip k + read).
    let block_size = 4 << 10;
    let payload: Vec<u8> = (0..5 * block_size)
        .map(|i| ((i * 31) ^ (i >> 3)) as u8)
        .collect();
    let stream = build_multi_block_stream(&payload, block_size);
    for &skip in &[
        0u64,
        1,
        1024,
        4096,
        4097,
        8192,
        16_000,
        payload.len() as u64 - 1,
    ] {
        let mut reader = Reader::new(Cursor::new(&stream));
        reader.skip(skip).unwrap();
        let mut tail = Vec::new();
        reader.read_to_end(&mut tail).unwrap();
        assert_eq!(tail, payload[skip as usize..], "skip={skip}");
    }
}

#[test]
fn skip_full_block_uses_fast_path() {
    // Whole-block skip must NOT decode (per Go's "CRC not checked on
    // skipped blocks" guarantee).  We verify by corrupting the CRC of
    // an early block and confirming a skip-past-it still succeeds.
    let payload = vec![b'M'; 8 * 1024];
    let mut stream = build_multi_block_stream(&payload, crate::stream::MIN_BLOCK_SIZE);

    // Find the first compressed chunk and flip a CRC byte.  The 10-byte
    // stream identifier is followed by the first data chunk's 4-byte
    // chunk header, then its CRC.
    let crc_off = make_stream_header(DEFAULT_BLOCK_SIZE, false).len() + 4;
    stream[crc_off] ^= 0xff;

    // Reading would fail with CRC error.
    {
        let mut r = Reader::new(Cursor::new(&stream));
        let mut sink = Vec::new();
        let err = r.read_to_end(&mut sink).unwrap_err();
        // Could be CRC or InvalidData wrapping Crc.
        let s = format!("{err:?}");
        assert!(
            s.to_lowercase().contains("crc") || err.kind() == io::ErrorKind::InvalidData,
            "expected CRC error, got: {s}"
        );
    }

    // But skipping past the whole first block should succeed without
    // touching CRC.
    let mut r = Reader::new(Cursor::new(&stream));
    // The first MinLZ block is sized to `block_size` = MIN_BLOCK_SIZE.
    // Skip exactly that many bytes — the fast path covers it.
    r.skip(crate::stream::MIN_BLOCK_SIZE as u64).unwrap();
    // current_offset must reflect the skip.
    assert_eq!(r.current_offset(), crate::stream::MIN_BLOCK_SIZE as u64);
}

#[test]
fn current_offset_tracks_reads_and_skips() {
    let payload: Vec<u8> = (0..16 * 1024).map(|i| (i as u8).wrapping_mul(3)).collect();
    let stream = build_multi_block_stream(&payload, 4 << 10);
    let mut reader = Reader::new(Cursor::new(stream));
    assert_eq!(reader.current_offset(), 0);
    let mut buf = [0u8; 1000];
    reader.read_exact(&mut buf).unwrap();
    assert_eq!(reader.current_offset(), 1000);
    reader.skip(2000).unwrap();
    assert_eq!(reader.current_offset(), 3000);
    reader.read_exact(&mut buf).unwrap();
    assert_eq!(reader.current_offset(), 4000);
}

#[test]
fn comp_crc_chunk_round_trip() {
    // 0x03 = "compressed CRC" — CRC over compressed bytes after the varint.
    let payload = b"the quick brown fox the quick brown fox the quick brown fox";
    let mut chunk_body = Vec::new();
    let n = block::encode_chunk_body(&mut chunk_body, 0, payload, block::Level::Balanced)
        .expect("encode");
    chunk_body.truncate(n);
    // Strip the leading varint(dlen) from chunk_body to get the raw
    // compressed bytes that the CRC covers for the 0x03 chunk type.
    let (_, n) = crate::stream::format::get_uvarint(&chunk_body).unwrap();
    let crc_input = &chunk_body[n..];
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE, false));
    let mut body = Vec::new();
    body.extend_from_slice(&masked_crc32c(crc_input).to_le_bytes());
    body.extend_from_slice(&chunk_body);
    push_chunk(&mut s, CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC, &body);
    push_eof(&mut s, payload.len() as u64);
    let got = decode_to_end(Reader::new(Cursor::new(&s))).unwrap();
    assert_eq!(got, payload);
}

// -------------------- ports of Go reader tests --------------------

/// Sticky errors are copies that keep only kind and message, so errors are
/// compared by message.
fn is_err(e: &io::Error, want: Error) -> bool {
    e.to_string() == want.to_string()
}

fn rand_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

fn write_stream(data: &[u8], b: WriterBuilder) -> Vec<u8> {
    let mut w = b.build(Vec::new()).unwrap();
    w.write_all(data).unwrap();
    w.finish().unwrap()
}

/// Offset of the `nth` (1-based) chunk whose type `pick` accepts.
fn find_chunk(b: &[u8], nth: usize, pick: impl Fn(u8) -> bool) -> usize {
    let (mut off, mut seen) = (0, 0);
    loop {
        if pick(b[off]) {
            seen += 1;
            if seen == nth {
                return off;
            }
        }
        off += CHUNK_HEADER_SIZE + read_chunk_len(&b[off + 1..]);
    }
}

/// Ports `TestReaderEmptyStream` (`minlz_test.go`): an EOF chunk directly
/// after an EOF chunk is an empty stream, which must have size 0. The writer
/// writes only an EOF chunk (and the index) for empty input.
#[test]
#[cfg_attr(miri, ignore = "200 KB streams too slow for miri")]
fn reader_empty_stream() {
    let data = b"Hello empty stream. ".repeat(10000);
    let a = write_stream(
        &data,
        WriterBuilder::new().block_size(64 << 10).append_index(),
    );
    let empty = write_stream(b"", WriterBuilder::new().append_index());
    assert!(
        empty.starts_with(&[CHUNK_TYPE_EOF, 1, 0, 0, 0]),
        "empty: {empty:x?}"
    );
    let mt = MtWriterBuilder::new()
        .concurrency(2)
        .append_index()
        .build(Vec::new())
        .unwrap()
        .finish()
        .unwrap();
    assert!(mt == empty, "MtWriter: {mt:x?}");
    let mut rs = ReadSeeker::new(Reader::new(Cursor::new(&empty)), &[]).unwrap();
    assert_eq!(rs.seek(SeekFrom::End(0)).unwrap(), 0, "ReadSeeker: Seek");
    assert_eq!(rs.read_at(&mut [0; 8], 0).unwrap(), 0, "ReadSeeker: ReadAt");
    let cases = [
        ("a+empty", [&a[..], &empty].concat(), Some(data.clone())),
        ("empty+a", [&empty[..], &a].concat(), Some(data.clone())),
        (
            "empty+empty",
            [&empty[..], &empty].concat(),
            Some(Vec::new()),
        ),
        (
            "a+empty+a",
            [&a[..], &empty, &a].concat(),
            Some(data.repeat(2)),
        ),
        (
            "a+nonzero-size",
            [&a[..], &[CHUNK_TYPE_EOF, 1, 0, 0, 1]].concat(),
            None,
        ),
    ];
    for (name, stream, want) in cases {
        let read = decode_to_end(Reader::new(Cursor::new(&stream)));
        let conc = Reader::new(Cursor::new(&stream))
            .decode_concurrent(Vec::new(), 2)
            .map(|c| c.writer);
        // Skipping past the end reads every EOF chunk.
        let skip = Reader::new(Cursor::new(&stream)).skip(u64::MAX);
        let index = crate::index::index_stream(Cursor::new(&stream));
        let Some(want) = want else {
            // index_stream doesn't check EOF sizes.
            for (api, res) in [
                ("Read", read.map(drop)),
                ("DecodeConcurrent", conc.map(drop)),
                ("Skip", skip),
            ] {
                let e = res.expect_err(name);
                assert!(is_err(&e, Error::Corrupt), "{name}/{api}: {e}");
            }
            continue;
        };
        assert!(read.unwrap() == want, "{name}/Read");
        assert!(conc.unwrap() == want, "{name}/DecodeConcurrent");
        let e = skip.expect_err(name);
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof, "{name}/Skip: {e}");
        index.unwrap_or_else(|e| panic!("{name}/IndexStream: {e}"));
    }
}

/// Ports `TestReaderSkipCRC` (`reader_test.go`), without the S2 rows: chunk
/// data kept after `skip` is CRC-checked like `read`.
#[test]
fn reader_skip_crc() {
    let random = rand_bytes(0, 16 << 10);
    let text = b"compressible text ".repeat(1 << 10);
    for data in [random, text] {
        let mut b = write_stream(
            &data,
            WriterBuilder::new().block_size(4 << 10).append_index(),
        );
        // Find the second data chunk, which holds offset 4096.
        let chunk = find_chunk(&b, 2, |t| t <= CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC);
        let typ = b[chunk];
        b[chunk + CHUNK_HEADER_SIZE] ^= 1;

        let e = Reader::new(Cursor::new(&b)).skip(4096 + 100).unwrap_err();
        assert!(is_err(&e, Error::Crc), "chunk {typ:#x}: Skip: {e}");

        let mut r = ReaderBuilder::new()
            .ignore_crc()
            .build(Cursor::new(&b))
            .unwrap();
        r.skip(4096 + 100).unwrap();
        let got = decode_to_end(&mut r).unwrap();
        assert!(
            got == data[4096 + 100..],
            "chunk {typ:#x}: Read with ignore_crc"
        );

        // Skipping to the exact end of the corrupted chunk must not leave it
        // in the buffer for a later seek back into it.
        let mut rs = ReadSeeker::new(Reader::new(Cursor::new(&b)), &[]).unwrap();
        rs.seek(SeekFrom::Start(8192)).unwrap();
        let e = rs.read_at(&mut [0; 100], 4096).unwrap_err();
        assert!(
            is_err(&e, Error::Crc),
            "chunk {typ:#x}: ReadAt after Seek: {e}"
        );
    }
}

/// Ports `TestDecodeConcurrentUncompressedCRC` (`minlz_test.go`): a bad
/// checksum of an uncompressed chunk is found at every concurrency.
#[test]
#[cfg_attr(miri, ignore = "100 KB streams too slow for miri")]
fn decode_concurrent_uncompressed_crc() {
    let data = rand_bytes(1, 100 << 10);
    let mut stream = write_stream(&data, WriterBuilder::new().block_size(16 << 10));
    let off = find_chunk(&stream, 3, |t| t == CHUNK_TYPE_UNCOMPRESSED_DATA);
    stream[off + CHUNK_HEADER_SIZE + CHECKSUM_SIZE + 10] ^= 1;
    for c in [1, 2, 8] {
        let e = Reader::new(Cursor::new(&stream))
            .decode_concurrent(io::sink(), c)
            .unwrap_err();
        assert!(is_err(&e, Error::Crc), "concurrency {c}: {e}");
        let n = ReaderBuilder::new()
            .ignore_crc()
            .build(Cursor::new(&stream))
            .unwrap()
            .decode_concurrent(io::sink(), c)
            .unwrap()
            .bytes_written;
        assert_eq!(
            n,
            data.len() as u64,
            "concurrency {c}, ignoring the checksum"
        );
    }
}

/// read_uncompressed reads and checksums in pieces: a flip in any piece must fail the checksum.
#[test]
#[cfg_attr(miri, ignore = "checksums 2 MB")]
fn read_uncompressed_pieces() {
    let data = rand_bytes(2, (2 << 20) + 1000);
    let sum = masked_crc32c(&data);
    let mut buf = vec![0xa5; data.len()];
    super::read_uncompressed(&mut &data[..], &mut buf, sum, false).unwrap();
    assert!(buf == data);
    for pos in [0, 1 << 20, data.len() - 1] {
        let mut bad = data.clone();
        bad[pos] ^= 1;
        let e = super::read_uncompressed(&mut &bad[..], &mut buf, sum, false).unwrap_err();
        assert!(matches!(e, Error::Crc), "flip at {pos}: {e}");
        super::read_uncompressed(&mut &bad[..], &mut buf, sum, true).unwrap();
    }
}

/// Port of Go's `crcTestCompressedStream`: `stream` with every 0x02 chunk
/// turned into a 0x03 chunk, and the offsets of their checksums.
fn crc_test_compressed_stream(stream: &[u8]) -> (Vec<u8>, Vec<usize>) {
    let mut converted = stream.to_vec();
    let mut checksums = Vec::new();
    let mut off = 0;
    while off < converted.len() {
        let end = off + CHUNK_HEADER_SIZE + read_chunk_len(&converted[off + 1..]);
        if converted[off] == CHUNK_TYPE_MINLZ_COMPRESSED_DATA {
            let payload = &mut converted[off + CHUNK_HEADER_SIZE..end];
            let (_, hdr_len) = get_uvarint(&payload[CHECKSUM_SIZE..]).unwrap();
            let crc = masked_crc32c(&payload[CHECKSUM_SIZE + hdr_len..]);
            payload[..CHECKSUM_SIZE].copy_from_slice(&crc.to_le_bytes());
            converted[off] = CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC;
            checksums.push(off + CHUNK_HEADER_SIZE);
        }
        off = end;
    }
    assert!(!checksums.is_empty(), "fixture has no compressed chunks");
    (converted, checksums)
}

/// Ports `TestConcurrentChunkCRC` (`reader_test.go`) at Go's `-short` size:
/// ordered decoding, CRC rejection and input-buffer reuse.
#[test]
#[cfg_attr(miri, ignore = "3 MiB stream too slow for miri")]
fn concurrent_chunk_crc() {
    const BLOCK_SIZE: usize = 256 << 10;
    const BLOCKS: usize = 12;
    let mut input = vec![0u8; BLOCKS * BLOCK_SIZE];
    for (block, b) in input.chunks_mut(BLOCK_SIZE).enumerate() {
        b.fill(block as u8);
        if block % 2 != 0 {
            for (i, v) in b[..BLOCK_SIZE / 2].iter_mut().enumerate() {
                *v = (i % 64 + block) as u8;
            }
            b[BLOCK_SIZE / 2..].copy_from_slice(&rand_bytes(82 + block as u64, BLOCK_SIZE / 2));
        }
    }
    let stream = write_stream(
        &input,
        WriterBuilder::new()
            .block_size(BLOCK_SIZE)
            .level(block::Level::Fastest),
    );
    let (comp_stream, checksums) = crc_test_compressed_stream(&stream);
    assert_eq!(checksums.len(), BLOCKS, "compressed blocks");
    for (kind, valid) in [(2, &stream), (3, &comp_stream)] {
        let got = decode_to_end(Reader::new(Cursor::new(valid))).unwrap();
        assert!(got == input, "chunk{kind:02x} sequential control");
        for concurrent in [1, 2, 4] {
            for mode in ["valid", "corrupt-first", "corrupt-last", "ignore-corrupt"] {
                let what = format!("chunk{kind:02x}/cpu{concurrent}/{mode}");
                let mut encoded = valid.clone();
                if mode != "valid" {
                    let idx = if mode == "corrupt-last" {
                        checksums.len() - 1
                    } else {
                        0
                    };
                    encoded[checksums[idx]] ^= 1;
                }
                let mut b = ReaderBuilder::new();
                if mode == "ignore-corrupt" {
                    b = b.ignore_crc();
                }
                let res = b
                    .build(Cursor::new(encoded))
                    .unwrap()
                    .decode_concurrent(Vec::new(), concurrent);
                if mode.starts_with("corrupt") {
                    let e = res.unwrap_err();
                    assert!(is_err(&e, Error::Crc), "{what}: invalid checksum: got {e}");
                    continue;
                }
                let r = res.unwrap_or_else(|e| panic!("{what}: {e}"));
                assert_eq!(r.bytes_written, input.len() as u64, "{what}");
                assert!(r.writer == input, "{what}");
            }
        }
    }
}

/// A 0x02 or 0x03 chunk body declaring a decoded length of 0 is corrupt:
/// unlike a standalone block, a chunk has no literal-only form.
#[test]
fn chunk_zero_decoded_len_is_corrupt() {
    for typ in [
        CHUNK_TYPE_MINLZ_COMPRESSED_DATA,
        CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC,
    ] {
        let payload = b"literal bytes";
        let mut body = masked_crc32c(payload).to_le_bytes().to_vec();
        body.push(0); // varint(0)
        body.extend_from_slice(payload);
        let mut s = make_stream_header(DEFAULT_BLOCK_SIZE, false).to_vec();
        push_chunk(&mut s, typ, &body);
        push_eof(&mut s, payload.len() as u64);

        let corrupt = |e: io::Error| e.kind() == io::ErrorKind::InvalidData;
        let read = decode_to_end(Reader::new(Cursor::new(&s)));
        assert!(read.is_err_and(corrupt), "{typ:#x}/Read");
        let skip = Reader::new(Cursor::new(&s)).skip(1);
        assert!(skip.is_err_and(corrupt), "{typ:#x}/Skip");
        // Skipping the whole chunk validates it too.
        let skip = Reader::new(Cursor::new(&s)).skip(100);
        assert!(skip.is_err_and(corrupt), "{typ:#x}/Skip past");
        let conc = Reader::new(Cursor::new(&s)).decode_concurrent(Vec::new(), 2);
        assert!(conc.is_err_and(corrupt), "{typ:#x}/DecodeConcurrent");
    }
}

/// `decode_concurrent` after a `skip` over whole blocks counts them toward the
/// EOF size, as Go's `streamSize` does.
#[test]
#[cfg_attr(miri, ignore = "200 KB streams too slow for miri")]
fn skip_then_decode_concurrent() {
    let data = rand_bytes(5, 200 << 10);
    let s = write_stream(&data, WriterBuilder::new().block_size(64 << 10));
    for skip in [64 << 10, 128 << 10] {
        let mut r = Reader::new(Cursor::new(&s));
        r.skip(skip as u64).unwrap();
        let got = r.decode_concurrent(Vec::new(), 2).unwrap().writer;
        assert!(got == data[skip..], "skip {skip}");
    }
}

// -------------------- SIMD chunks (ports of minlz_test.go, seek_simd_test.go) --------------------

/// Go's `ErrCorrupt`: the stream's, or that of a block decode the stream error wraps.
fn is_corrupt(e: &io::Error) -> bool {
    is_err(e, Error::Corrupt)
        || e.to_string()
            .starts_with(&Error::Block(BlockError::Corrupt("")).to_string())
}

/// Go's `makeHeader`.
fn make_header(block_size: usize, simd: bool) -> Vec<u8> {
    make_stream_header(block_size, simd).to_vec()
}

fn eof_chunk(size: usize) -> Vec<u8> {
    let mut b = Vec::new();
    push_eof(&mut b, size as u64);
    b
}

/// A SIMD block of data.
fn simd_block(data: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    block::encode(&mut b, data, block::Level::BalancedSimd).unwrap();
    assert!(
        get_uvarint(&b[1..]).unwrap().0 >> 24 == 1,
        "not a SIMD block"
    );
    b
}

fn hello_simd() -> Vec<u8> {
    b"Hello SIMD chunk, hello SIMD stream. ".repeat(300)
}

/// Ports `TestReaderSIMDChunks`: every chunk reader takes SIMD chunks only where the format
/// allows them. Go's BlockSearcher and sidecar readers aren't in this crate.
#[test]
#[cfg_attr(
    miri,
    ignore = "SIMD encoders and 16 MiB index_stream buffers are slow under miri"
)]
fn reader_simd_chunks() {
    let data = hello_simd();
    let simd = simd_block(&data);
    let mut min_lz = Vec::new();
    block::encode(&mut min_lz, &data, block::Level::Balanced).unwrap();
    let chunk = |typ: u8, check: &[u8], block: &[u8]| {
        let mut b = Vec::new();
        push_chunk(&mut b, typ, &[check, &block[1..]].concat());
        b
    };
    let hash = crate::xxh3::hash(&data).to_le_bytes();
    let bad_hash = (crate::xxh3::hash(&data) ^ 1).to_le_bytes();
    let sum = masked_crc32c(&data).to_le_bytes();
    let eof = eof_chunk(data.len());
    let simd_chunk = chunk(CHUNK_TYPE_MINLZ_SIMD, &hash, &simd);
    let hdr = make_header(64 << 10, true);
    let no_flag = make_header(64 << 10, false);
    let bit7 = |id: &[u8]| {
        let mut id = id.to_vec();
        *id.last_mut().unwrap() |= 1 << 7;
        id
    };
    let (_, h) = get_uvarint(&min_lz[1..]).unwrap();
    let mut too_large = vec![0];
    let mut v = [0; 10];
    let n = put_uvarint(&mut v, MAX_BLOCK_SIZE as u64 + 1);
    too_large.extend_from_slice(&v[..n]);
    too_large.extend_from_slice(&min_lz[1 + h..]);
    let max_enc = block::max_encoded_len(4 << 10).unwrap();
    let s2_id = b"\xff\x06\x00\x00S2sTwO";

    let cat = |parts: &[&[u8]]| parts.concat();
    // (name, stream, ok, the error every reader must return)
    let tests: Vec<(&str, Vec<u8>, bool, Option<Error>)> = vec![
        ("valid", cat(&[&hdr, &simd_chunk, &eof]), true, None),
        (
            "concatenated",
            cat(&[
                &no_flag,
                &chunk(CHUNK_TYPE_MINLZ_COMPRESSED_DATA, &sum, &min_lz),
                &eof,
                &hdr,
                &simd_chunk,
                &eof,
            ]),
            true,
            None,
        ),
        (
            "no-flag",
            cat(&[&no_flag, &simd_chunk, &eof]),
            false,
            Some(Error::Corrupt),
        ),
        (
            "no-flag-concatenated",
            cat(&[&hdr, &simd_chunk, &eof, &no_flag, &simd_chunk, &eof]),
            false,
            Some(Error::Corrupt),
        ),
        // This crate has no S2 fallback, so the S2 stream fails before its chunk.
        (
            "s2-concatenated",
            cat(&[&hdr, &simd_chunk, &eof, s2_id, &simd_chunk, &eof]),
            false,
            None,
        ),
        (
            "bad-hash",
            cat(&[&hdr, &chunk(CHUNK_TYPE_MINLZ_SIMD, &bad_hash, &simd), &eof]),
            false,
            Some(Error::Crc),
        ),
        (
            "truncated",
            cat(&[&hdr, &simd_chunk[..simd_chunk.len() - 10]]),
            false,
            None,
        ),
        (
            "short",
            cat(&[&hdr, b"\x10\x04\x00\x00", &hash[..4], &eof]),
            false,
            None,
        ),
        (
            "too-large",
            cat(&[&make_header(4 << 10, true), &simd_chunk, &eof]),
            false,
            Some(Error::TooLarge),
        ),
        (
            "too-large-minlz",
            cat(&[
                &make_header(MAX_BLOCK_SIZE, false),
                &chunk(CHUNK_TYPE_MINLZ_COMPRESSED_DATA, &sum, &too_large),
                &eof,
            ]),
            false,
            Some(Error::TooLarge),
        ),
        // One byte longer than the largest chunk of the block size; chunk drops the first byte.
        (
            "long-chunk",
            cat(&[
                &make_header(4 << 10, true),
                &chunk(
                    CHUNK_TYPE_MINLZ_SIMD,
                    &[],
                    &vec![0; 1 + max_enc + CHECKSUM_SIZE + 1],
                ),
                &eof,
            ]),
            false,
            Some(Error::Corrupt),
        ),
        (
            "bit-7",
            cat(&[
                &bit7(&no_flag),
                &chunk(CHUNK_TYPE_MINLZ_COMPRESSED_DATA, &sum, &min_lz),
                &eof,
            ]),
            false,
            Some(Error::Corrupt),
        ),
        (
            "bit-7-simd",
            cat(&[&bit7(&hdr), &simd_chunk, &eof]),
            false,
            Some(Error::Corrupt),
        ),
    ];
    type ReadDataFn = fn(&[u8], &[u8]) -> io::Result<()>;
    // (name, whether the hash is checked, reader)
    let readers: [(&str, bool, ReadDataFn); 4] = [
        ("Read", true, |s, data| {
            let got = decode_to_end(Reader::new(s))?;
            if got != data.repeat(got.len() / data.len()) {
                return Err(io::Error::other("output mismatch"));
            }
            Ok(())
        }),
        ("DecodeConcurrent", true, |s, _| {
            Reader::new(Cursor::new(s.to_vec()))
                .decode_concurrent(io::sink(), 2)
                .map(drop)
        }),
        ("Skip", true, |s, _| {
            let mut dec = Reader::new(s);
            // Skips decode the block they end in, so every chunk is checked.
            loop {
                match dec.skip(1000) {
                    Ok(()) => {}
                    Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
                    Err(e) => return Err(e),
                }
            }
        }),
        ("IndexStream", false, |s, _| {
            crate::index::index_stream(s).map(drop).map_err(Into::into)
        }),
    ];
    for (name, stream, ok, want) in &tests {
        for (r, hash, read) in readers {
            // index_stream doesn't check the identifier's block size or reserved bit, and
            // returns index errors.
            let index = r == "IndexStream";
            if *name == "bad-hash" && !hash
                || index && (*name == "too-large" || name.starts_with("bit-7"))
            {
                continue;
            }
            let res = read(stream, &data);
            assert_eq!(res.is_ok(), *ok, "{name}/{r}: got {res:?}");
            if let (Some(want), false) = (want, index) {
                let e = res.unwrap_err();
                let matches = match want {
                    Error::Corrupt => is_corrupt(&e),
                    want => e.to_string() == want.to_string(),
                };
                assert!(matches, "{name}/{r}: got error {e}, want {want}");
            }
        }
    }

    // Ignoring checksums also ignores the hash.
    let bad = cat(&[&hdr, &chunk(CHUNK_TYPE_MINLZ_SIMD, &bad_hash, &simd), &eof]);
    let r = ReaderBuilder::new().ignore_crc().build(&bad[..]).unwrap();
    decode_to_end(r).expect("ignore_crc");
    let mut r = ReaderBuilder::new()
        .ignore_crc()
        .build(Cursor::new(bad.clone()))
        .unwrap();
    r.decode_concurrent(io::sink(), 2)
        .expect("ignore_crc, decode_concurrent");
    // Without a stream identifier, the flag is unknown, so SIMD chunks are taken until one is read.
    let ignore_id = |s: Vec<u8>| {
        decode_to_end(
            ReaderBuilder::new()
                .ignore_stream_id()
                .build(Cursor::new(s))
                .unwrap(),
        )
    };
    let got = ignore_id(cat(&[&simd_chunk, &eof])).expect("ignore_stream_id");
    assert!(got == data, "ignore_stream_id: output mismatch");
    let e = ignore_id(cat(&[&simd_chunk, &eof, &no_flag, &simd_chunk, &eof])).unwrap_err();
    assert!(is_corrupt(&e), "ignore_stream_id, no flag: got {e}");
    // Reset forgets the flag of the last identifier (Go's Reader.Reset).
    let mut r = ReaderBuilder::new()
        .ignore_stream_id()
        .build(Cursor::new(cat(&[&no_flag, &eof])))
        .unwrap();
    decode_to_end(&mut r).expect("ignore_stream_id, no flag, no chunks");
    r.reset(Cursor::new(cat(&[&simd_chunk, &eof])));
    let got = decode_to_end(&mut r).expect("ignore_stream_id after reset");
    assert!(got == data, "ignore_stream_id after reset: output mismatch");
}

/// Go's `blockTypeMismatches`: SIMD blocks in MinLZ chunks, a MinLZ block in a SIMD chunk and
/// reserved block types in both, all of data, as (name, bad, good). good is a valid chunk of the
/// same data; both are padded to the same length, so a stream with good can provide the index
/// of the stream with bad.
fn block_type_mismatches(data: &[u8]) -> Vec<(String, Vec<u8>, Vec<u8>)> {
    let simd = simd_block(data);
    let mut min_lz = Vec::new();
    block::encode(&mut min_lz, data, block::Level::Balanced).unwrap();
    // Streams store blocks without the indicator byte.
    let (simd, min_lz) = (&simd[1..], &min_lz[1..]);
    let chunk = |typ: u8, block: &[u8]| {
        let check = if typ == CHUNK_TYPE_MINLZ_SIMD {
            crate::xxh3::hash(data).to_le_bytes().to_vec()
        } else {
            masked_crc32c(data).to_le_bytes().to_vec()
        };
        let mut b = Vec::new();
        push_chunk(&mut b, typ, &[&check, block].concat());
        b
    };
    let good = chunk(CHUNK_TYPE_MINLZ_SIMD, simd);
    let mut m = vec![
        (
            "simd-in-0x02".to_string(),
            chunk(CHUNK_TYPE_MINLZ_COMPRESSED_DATA, simd),
            good.clone(),
        ),
        (
            "simd-in-0x03".to_string(),
            chunk(CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC, simd),
            good.clone(),
        ),
        (
            "minlz-in-0x10".to_string(),
            chunk(CHUNK_TYPE_MINLZ_SIMD, min_lz),
            chunk(CHUNK_TYPE_MINLZ_COMPRESSED_DATA, min_lz),
        ),
    ];
    let (_, h) = get_uvarint(simd).unwrap();
    for typ in [2u64, 255] {
        let mut reserved = vec![0; 10];
        let n = put_uvarint(&mut reserved, typ << 24 | data.len() as u64);
        reserved.truncate(n);
        reserved.extend_from_slice(&simd[h..]);
        for ct in [
            CHUNK_TYPE_MINLZ_COMPRESSED_DATA,
            CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC,
            CHUNK_TYPE_MINLZ_SIMD,
        ] {
            m.push((
                format!("type-{typ}-in-0x{ct:02x}"),
                chunk(ct, &reserved),
                good.clone(),
            ));
        }
    }
    let pad = |b: &[u8], n: usize| {
        let mut b = b.to_vec();
        let zeros = vec![0; n - b.len() - 4];
        push_chunk(&mut b, CHUNK_TYPE_PADDING, &zeros);
        b
    };
    for (_, bad, good) in &mut m {
        // A padding chunk has at least its 4 byte header.
        let n = bad.len().max(good.len()) + 4;
        (*bad, *good) = (pad(bad, n), pad(good, n));
    }
    m
}

type ReadFn<'a> = dyn Fn(&[u8]) -> io::Result<()> + 'a;

/// Ports `TestReaderBlockTypeMismatch`: every reader rejects a block whose type doesn't match
/// its chunk type as corrupt.
#[test]
#[cfg_attr(
    miri,
    ignore = "SIMD encoders and 16 MiB index_stream buffers are slow under miri"
)]
fn reader_block_type_mismatch() {
    let data = hello_simd();
    let eof = eof_chunk(2 * data.len());
    // Inside the second block, so ReadAt seeks to it.
    let off = data.len() as u64 + 10;
    for (name, bad_chunk, good_chunk) in block_type_mismatches(&data) {
        // The first block is valid, so the readers get past it.
        let first = [&make_header(64 << 10, true)[..], &good_chunk].concat();
        let good = [&first[..], &good_chunk, &eof].concat();
        let bad = [&first[..], &bad_chunk, &eof].concat();
        let index = crate::index::index_stream(&good[..]).unwrap();
        let readers: [(&str, &ReadFn); 5] = [
            ("Read", &|s| decode_to_end(Reader::new(s)).map(drop)),
            ("DecodeConcurrent", &|s| {
                Reader::new(Cursor::new(s.to_vec()))
                    .decode_concurrent(io::sink(), 2)
                    .map(drop)
            }),
            ("Skip", &|s| Reader::new(s).skip(off)),
            ("ReadAt", &|s| {
                let mut rs = ReadSeeker::new(Reader::new(Cursor::new(s)), &index)?;
                rs.read_at(&mut [0; 10], off).map(drop)
            }),
            ("IndexStream", &|s| {
                crate::index::index_stream(s).map(drop).map_err(Into::into)
            }),
        ];
        for (r, read) in readers {
            read(&good).unwrap_or_else(|e| panic!("{name}/{r}: valid stream: {e}"));
            let e = read(&bad).expect_err(&name);
            assert!(
                r == "IndexStream" || is_corrupt(&e),
                "{name}/{r}: got error {e}, want corrupt"
            );
        }
    }
}

/// Go's `seekTestData`.
fn seek_test_data(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut b = Vec::with_capacity(n + 32);
    while b.len() < n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        b.extend_from_slice(format!("line {}: {:x}\n", b.len(), x % (1 << 12)).as_bytes());
    }
    b.truncate(n);
    b
}

/// Go's `seekTestStream`.
fn seek_test_stream(level: block::Level, b: &[u8]) -> Vec<u8> {
    write_stream(b, WriterBuilder::new().level(level).block_size(64 << 10))
}

/// Ports `TestReadSeekerSIMDConcat` (`seek_simd_test.go`): seeks into concatenated streams with
/// and without the SIMD flag. Index jumps skip the identifier of the target stream.
#[test]
#[cfg_attr(miri, ignore = "700 KiB streams too slow for miri")]
fn read_seeker_simd_concat() {
    let (a, b) = (seek_test_data(400 << 10, 1), seek_test_data(300 << 10, 2));
    let want = [&a[..], &b].concat();
    const N: usize = 1000;
    let mut offs: Vec<usize> = (0..want.len()).step_by(16 << 10).collect();
    offs.push(a.len() - N / 2);
    offs.sort();
    for (name, lv) in [
        (
            "minlz+simd",
            [block::Level::Balanced, block::Level::BalancedSimd],
        ),
        (
            "simd+minlz",
            [block::Level::BalancedSimd, block::Level::Balanced],
        ),
    ] {
        let stream = [seek_test_stream(lv[0], &a), seek_test_stream(lv[1], &b)].concat();
        let idx = crate::index::index_stream(&stream[..]).unwrap();
        let new_rs = || ReadSeeker::new(Reader::new(Cursor::new(&stream)), &idx).unwrap();
        let mut got = [0u8; N];
        // Each direction on a new reader: seeks go to either stream before and after the read
        // across the boundary has read the second identifier.
        let mut rs = new_rs();
        for &off in &offs {
            let m = N.min(want.len() - off);
            rs.read_exact_at(&mut got[..m], off as u64)
                .unwrap_or_else(|e| panic!("{name}: ReadAt({off}): {e}"));
            assert!(
                got[..m] == want[off..off + m],
                "{name}: ReadAt({off}): wrong data"
            );
        }
        let mut rs = new_rs();
        for &off in offs.iter().rev() {
            let m = N.min(want.len() - off);
            rs.seek(SeekFrom::Start(off as u64)).unwrap();
            rs.read_exact(&mut got[..m])
                .unwrap_or_else(|e| panic!("{name}: Seek({off}): read: {e}"));
            assert!(
                got[..m] == want[off..off + m],
                "{name}: Seek({off}): wrong data"
            );
        }
    }
}

/// An index jump into a later concatenated stream skips its identifier: neither that stream's
/// EOF size (counted from its identifier) nor its block size may be checked against the first.
#[test]
#[cfg_attr(miri, ignore = "multi-MiB streams too slow for miri")]
fn read_seeker_concat_jump_into_later_stream() {
    let (a, b) = (seek_test_data(400 << 10, 1), seek_test_data(3 << 20, 2));
    let want = [&a[..], &b].concat();
    let sb = write_stream(
        &b,
        WriterBuilder::new()
            .level(block::Level::BalancedSimd)
            .block_size(1 << 20),
    );
    let stream = [seek_test_stream(block::Level::Balanced, &a), sb].concat();
    let idx = crate::index::index_stream(&stream[..]).unwrap();
    let mut rs = ReadSeeker::new(Reader::new(Cursor::new(&stream)), &idx).unwrap();
    let off = a.len() + (2 << 20) + 5;
    rs.seek(SeekFrom::Start(off as u64)).unwrap();
    let mut got = Vec::new();
    rs.read_to_end(&mut got).unwrap();
    assert!(got == want[off..]);
}

/// Two concatenated 400 and 300 KiB streams, their decoded bytes and index.
fn seek_test_concat() -> (Vec<u8>, Vec<u8>, usize, Vec<u8>) {
    let (a, b) = (seek_test_data(400 << 10, 1), seek_test_data(300 << 10, 2));
    let stream = [
        seek_test_stream(block::Level::Balanced, &a),
        seek_test_stream(block::Level::Balanced, &b),
    ]
    .concat();
    let idx = crate::index::index_stream(&stream[..]).unwrap();
    (stream, [&a[..], &b].concat(), a.len(), idx)
}

/// Seek offsets count from the start of the first stream, not of the stream being read: the
/// in-buffer fast path and `SeekFrom::Current` must not mix up the two streams.
#[test]
#[cfg_attr(miri, ignore = "700 KiB streams too slow for miri")]
fn read_seeker_concat_offsets_are_global() {
    let (stream, want, a_len, idx) = seek_test_concat();
    let mut rs = ReadSeeker::new(Reader::new(Cursor::new(&stream)), &idx).unwrap();
    let mut got = [0u8; 1000];
    let cross = a_len - 500;
    rs.read_exact_at(&mut got, cross as u64).unwrap();
    assert!(got[..] == want[cross..cross + 1000], "ReadAt({cross})");
    rs.read_exact_at(&mut got, 10).unwrap();
    assert!(got[..] == want[10..1010], "ReadAt(10) after crossing");

    rs.read_exact_at(&mut got, cross as u64).unwrap();
    let end = (cross + 1000) as u64;
    assert_eq!(rs.stream_position().unwrap(), end);
    assert_eq!(rs.seek(SeekFrom::Current(-1000)).unwrap(), cross as u64);
    rs.read_exact(&mut got).unwrap();
    assert!(
        got[..] == want[cross..cross + 1000],
        "read after Current(-1000)"
    );
    assert_eq!(rs.seek(SeekFrom::Current(100)).unwrap(), end + 100);
    rs.read_exact(&mut got).unwrap();
    assert!(
        got[..] == want[end as usize + 100..][..1000],
        "read after Current(100)"
    );
}

/// Reads through the second stream's EOF chunk, which is checked when its identifier was read
/// and not after an index jump skipped it.
#[test]
#[cfg_attr(miri, ignore = "700 KiB streams too slow for miri")]
fn read_seeker_concat_reads_through_eof() {
    let (stream, want, a_len, idx) = seek_test_concat();
    let mut bad = stream.clone();
    let eof = find_chunk(&bad, 2, |t| t == CHUNK_TYPE_EOF);
    bad[eof + CHUNK_HEADER_SIZE] ^= 1;
    // A sequential crossing reads the second identifier, a jump past its first block doesn't.
    for (off, checked) in [(a_len - 500, true), (a_len + (100 << 10), false)] {
        let mut rs = ReadSeeker::new(Reader::new(Cursor::new(&stream)), &idx).unwrap();
        rs.seek(SeekFrom::Start(off as u64)).unwrap();
        let mut got = Vec::new();
        rs.read_to_end(&mut got).unwrap();
        assert!(got == want[off..], "Seek({off})");
        // Past the last EOF chunk a seek rereads the first identifier.
        assert_eq!(rs.read_at(&mut [0; 10], want.len() as u64).unwrap(), 0);
        let mut got = [0; 1000];
        rs.read_exact_at(&mut got, 10).unwrap();
        assert!(
            got[..] == want[10..1010],
            "Seek({off}): ReadAt(10) after EOF"
        );

        let mut rs = ReadSeeker::new(Reader::new(Cursor::new(&bad)), &idx).unwrap();
        rs.seek(SeekFrom::Start(off as u64)).unwrap();
        let res = rs.read_to_end(&mut Vec::new());
        if checked {
            let e = res.expect_err("bad EOF size");
            assert!(is_corrupt(&e), "Seek({off}): {e}");
        } else {
            res.unwrap();
        }
    }
    // After reading the second stream from a crossing, the fast path must find the first.
    let mut rs = ReadSeeker::new(Reader::new(Cursor::new(&stream)), &idx).unwrap();
    rs.read_exact_at(&mut [0; 1000], (a_len - 500) as u64)
        .unwrap();
    rs.seek(SeekFrom::Start(10)).unwrap();
    let mut got = Vec::new();
    rs.read_to_end(&mut got).unwrap();
    assert!(got == want[10..], "Seek(10) after crossing");

    let e = decode_to_end(Reader::new(Cursor::new(&bad))).unwrap_err();
    assert!(is_corrupt(&e), "Reader: {e}");
    assert!(decode_to_end(Reader::new(Cursor::new(&stream))).unwrap() == want);
    let skip = a_len + (64 << 10);
    let mut r = Reader::new(Cursor::new(&stream));
    r.skip(skip as u64).unwrap();
    assert_eq!(r.current_offset(), skip as u64);
    let got = r.decode_concurrent(Vec::new(), 2).unwrap().writer;
    assert!(got == want[skip..], "skip then decode_concurrent");
}

/// Ports `TestReaderSIMDFlagStrict` (`seek_simd_test.go`): reads still require the flag of the
/// last identifier they read.
#[test]
#[cfg_attr(miri, ignore = "200 KiB streams too slow for miri")]
fn reader_simd_flag_strict() {
    let a = seek_test_data(200 << 10, 1);
    let simd = seek_test_stream(block::Level::BalancedSimd, &a);
    let mut no_flag = simd.clone();
    no_flag[CHUNK_HEADER_SIZE + MAGIC_BODY_LEN - 1] &= !STREAM_FLAG_SIMD;
    let e = decode_to_end(Reader::new(&no_flag[..])).unwrap_err();
    assert!(is_corrupt(&e), "sequential read: got {e}");

    // index_stream rejects no_flag, so index the stream with the flag; the offsets are the same.
    let idx = crate::index::index_stream(&[&simd[..], &simd].concat()[..]).unwrap();
    let stream = [&simd[..], &no_flag].concat();
    let mut rs = ReadSeeker::new(Reader::new(Cursor::new(&stream)), &idx).unwrap();
    let e = rs
        .read_at(&mut [0; 1000], a.len() as u64 - 500)
        .unwrap_err();
    assert!(
        is_corrupt(&e),
        "read on into a stream without the flag: got {e}"
    );
}

// -------------------- seeded ports of Go's stream fuzzers --------------------

fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

/// Inputs for the stream fuzzers: text, random bytes, runs and a mix, of sizes around the
/// block sizes.
fn fuzz_inputs() -> Vec<Vec<u8>> {
    let mut rng = xorshift(0xf022);
    let mut out = vec![Vec::new(), b"x".to_vec()];
    for n in [100, 5000, 20 << 10, 70 << 10] {
        let text = seek_test_data(n, n as u64);
        let random: Vec<u8> = (0..n).map(|_| rng() as u8).collect();
        let runs: Vec<u8> = (0..n).map(|i| (i / 300 % 7) as u8).collect();
        let mix = [&text[..n / 2], &random[n / 2..]].concat();
        out.extend([text, random, runs, mix]);
    }
    out
}

/// Applies 1 to 4 random edits: bit flips, byte overwrites, cuts and short appends.
fn mutate(rng: &mut impl FnMut() -> u64, b: &[u8]) -> Vec<u8> {
    let mut d = b.to_vec();
    for _ in 0..1 + rng() % 4 {
        if d.is_empty() {
            d.push(rng() as u8);
            continue;
        }
        let i = rng() as usize % d.len();
        match rng() % 4 {
            0 => d[i] ^= 1 << (rng() % 8),
            1 => d[i] = rng() as u8,
            2 => d.truncate(i),
            _ => d.extend((0..1 + rng() % 8).map(|_| rng() as u8)),
        }
    }
    d
}

/// Go's `FuzzStreamEncode` over [`fuzz_inputs`]: the encoders vary the stream options, with
/// four write methods, and Read or decode_concurrent.
#[test]
#[cfg_attr(miri, ignore = "too slow for miri")]
fn fuzz_stream_encode_seeded() {
    // (level, block size, concurrency, padding)
    let encoders = [
        (block::Level::SuperFast, 128 << 10, 1, 0),
        (block::Level::FastestSimd, 4 << 10, 1, 0),
        (block::Level::BalancedSimdDelta, 16 << 10, 2, 1 << 10),
        (block::Level::SuperFastSimd, 64 << 10, 2, 0),
    ];
    for data in fuzz_inputs() {
        for (i, &(level, bs, conc, padding)) in encoders.iter().enumerate() {
            let k = data.len() / 3;
            let write_method = (data.len() + i) % 4;
            let comp = if conc == 1 {
                let mut w = WriterBuilder::new()
                    .level(level)
                    .block_size(bs)
                    .padding(padding)
                    .append_index()
                    .build(Vec::new())
                    .unwrap();
                match write_method {
                    0 => w.write_all(&data).unwrap(),
                    1 => {
                        for c in data.chunks(1000) {
                            w.write_all(c).unwrap();
                        }
                    }
                    2 => drop(io::copy(&mut &data[..], &mut w).unwrap()),
                    _ => {
                        // Buffered data, then a flush: short blocks.
                        w.write_all(&data[..k]).unwrap();
                        w.flush().unwrap();
                        w.write_all(&data[k..2 * k]).unwrap();
                        w.flush().unwrap();
                        w.write_all(&data[2 * k..]).unwrap();
                    }
                }
                // The writer counts blocks as it emits them.
                w.flush().unwrap();
                assert_eq!(w.written().uncompressed, data.len() as u64, "encoder {i}");
                w.finish().unwrap()
            } else {
                let mut w = MtWriterBuilder::new()
                    .level(level)
                    .block_size(bs)
                    .padding(padding)
                    .concurrency(conc)
                    .append_index()
                    .build(Vec::new())
                    .unwrap();
                match write_method {
                    0 => w.write_all(&data).unwrap(),
                    1 => {
                        for c in data.chunks(1000) {
                            w.write_all(c).unwrap();
                        }
                    }
                    2 => drop(io::copy(&mut &data[..], &mut w).unwrap()),
                    _ => {
                        // EncodeBuffer after buffered data, and a Flush: short blocks.
                        w.write_all(&data[..k]).unwrap();
                        w.encode_buffer(&data[k..2 * k]).unwrap();
                        w.flush().unwrap();
                        w.write_all(&data[2 * k..]).unwrap();
                    }
                }
                w.flush().unwrap();
                assert_eq!(w.uncompressed_written(), data.len() as u64, "encoder {i}");
                w.finish().unwrap()
            };
            if padding > 1 {
                assert_eq!(comp.len() % padding as usize, 0, "encoder {i}");
            }
            let got = if (i + comp.len()) % 2 == 0 {
                decode_to_end(Reader::new(&comp[..]))
            } else {
                Reader::new(Cursor::new(comp))
                    .decode_concurrent(Vec::new(), 2)
                    .map(|c| c.writer)
            };
            let got = got.unwrap_or_else(|e| panic!("encoder {i}, write {write_method}: {e}"));
            assert!(got == data, "encoder {i}, write {write_method}: mismatch");
        }
    }
}

/// Go's `FuzzStreamDecode` over streams of [`fuzz_inputs`] (block size 8 KiB: Balanced,
/// BalancedSimd with an index, uncompressed with an index) and seeded mutations of them: Read
/// and decode_concurrent agree, and on valid streams Skip matches Read.
#[test]
#[cfg_attr(miri, ignore = "too slow for miri")]
fn fuzz_stream_decode_seeded() {
    let mut seeds = Vec::new();
    for data in fuzz_inputs() {
        for b in [
            WriterBuilder::new().block_size(8 << 10),
            WriterBuilder::new()
                .block_size(8 << 10)
                .level(block::Level::BalancedSimd)
                .append_index(),
            WriterBuilder::new()
                .block_size(8 << 10)
                .uncompressed()
                .append_index(),
        ] {
            seeds.push(write_stream(&data, b));
        }
    }
    let new_reader = |data: &[u8]| {
        ReaderBuilder::new()
            .ignore_crc()
            .max_block_size(1 << 20)
            .build(Cursor::new(data.to_vec()))
            .unwrap()
    };
    let mut rng = xorshift(0x5eed);
    for seed in &seeds {
        for m in 0..20 {
            let data = if m == 0 {
                seed.clone()
            } else {
                mutate(&mut rng, seed)
            };
            let read = decode_to_end(new_reader(&data));
            let conc = new_reader(&data)
                .decode_concurrent(Vec::new(), 2)
                .map(|c| c.writer);
            assert_eq!(read.is_ok(), conc.is_ok(), "{read:?} vs {conc:?}");
            let Ok(read) = read else { continue };
            assert!(read == conc.unwrap(), "output mismatch");

            // Skip doesn't decode skipped blocks, so it only must match Read on valid streams.
            let skip = data.len() / 2;
            let mut dec = new_reader(&data);
            let res = dec.skip(skip as u64);
            if skip > read.len() {
                assert!(res.is_err(), "skip past end");
                continue;
            }
            res.unwrap_or_else(|e| panic!("skip {skip} of {}: {e}", read.len()));
            let tail = decode_to_end(dec).unwrap();
            assert!(tail == read[skip..], "skip mismatch");
        }
    }
}
// -------------------- chunk body validation --------------------

/// A compressed chunk has no `dlen == 0` literal form (only standalone
/// blocks do), and its compressed bytes never exceed `dlen`.  Every decode
/// path must reject both, including a skip that drops the whole chunk.
#[test]
fn rejects_chunk_with_zero_or_short_dlen() {
    for (dlen, rest) in [(0u64, &b"hello world"[..]), (1, &[0xaa; 10][..])] {
        let mut varint = [0u8; 10];
        let n = put_uvarint(&mut varint, dlen);
        let mut body = masked_crc32c(rest).to_le_bytes().to_vec();
        body.extend_from_slice(&varint[..n]);
        body.extend_from_slice(rest);
        let mut s = make_stream_header(DEFAULT_BLOCK_SIZE, false).to_vec();
        push_chunk(&mut s, CHUNK_TYPE_MINLZ_COMPRESSED_DATA, &body);
        push_eof(&mut s, rest.len() as u64);

        let err = decode_to_end(Reader::new(Cursor::new(&s))).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData, "dlen={dlen}: read");
        let err = Reader::new(Cursor::new(&s)).skip(1).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData, "dlen={dlen}: skip");
        let err = Reader::new(Cursor::new(s.clone()))
            .decode_concurrent(Vec::new(), 2)
            .unwrap_err();
        assert_eq!(
            err.kind(),
            io::ErrorKind::InvalidData,
            "dlen={dlen}: concurrent"
        );
        assert!(
            crate::index_stream(Cursor::new(&s)).is_err(),
            "dlen={dlen}: index"
        );
    }
}
