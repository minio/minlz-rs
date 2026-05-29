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
use std::io::{self, Cursor, Read};
use std::rc::Rc;

use crate::block;
use crate::stream::crc::masked_crc32c;
use crate::stream::format::{
    make_stream_header, put_uvarint, CHECKSUM_SIZE, CHUNK_TYPE_EOF,
    CHUNK_TYPE_MINLZ_COMPRESSED_DATA, CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC,
    CHUNK_TYPE_PADDING, CHUNK_TYPE_UNCOMPRESSED_DATA, DEFAULT_BLOCK_SIZE,
};
use crate::stream::{Reader, ReaderBuilder};

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
    let compressed =
        block::append_encoded_chunk_body(&mut chunk_body, payload, block::Level::Balanced)
            .expect("encode");
    assert!(compressed, "test payload should compress");
    let mut body = Vec::with_capacity(CHECKSUM_SIZE + chunk_body.len());
    body.extend_from_slice(&masked_crc32c(payload).to_le_bytes());
    body.extend_from_slice(&chunk_body);
    push_chunk(out, CHUNK_TYPE_MINLZ_COMPRESSED_DATA, &body);
}

fn build_stream(blocks: &[&[u8]]) -> Vec<u8> {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
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
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
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
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
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
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
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
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
    let payload = b"abcde";
    let bad_crc = 0xdeadbeef_u32;
    let mut body = Vec::new();
    body.extend_from_slice(&bad_crc.to_le_bytes());
    body.extend_from_slice(payload);
    push_chunk(&mut s, CHUNK_TYPE_UNCOMPRESSED_DATA, &body);
    push_eof(&mut s, payload.len() as u64);
    let got = decode_to_end(ReaderBuilder::new().ignore_crc().build(Cursor::new(&s))).unwrap();
    assert_eq!(got, payload);
}

#[test]
fn eof_with_wrong_total_size() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
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
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
    push_chunk(&mut s, 0x04, b"dummy"); // reserved unskippable
    let err = Reader::new(Cursor::new(&s))
        .read(&mut [0u8; 1])
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn user_non_skippable_without_callback_errors() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
    push_chunk(&mut s, 0xc0, b"unskippable"); // user non-skippable
    let err = Reader::new(Cursor::new(&s))
        .read(&mut [0u8; 1])
        .unwrap_err();
    assert_eq!(err.kind(), io::ErrorKind::InvalidData);
}

// -------------------- skippables --------------------

#[test]
fn padding_is_silently_skipped() {
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
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
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
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
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
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
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
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
        .build(Cursor::new(&s));
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
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
    push_chunk(&mut s, 0x80, b"trigger");
    push_uncompressed_block(&mut s, b"unreachable");
    push_eof(&mut s, 11);
    let reader = ReaderBuilder::new()
        .user_chunk_callback(0x80, |_, _| Err(io::Error::other("denied")))
        .build(Cursor::new(&s));
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
            .build(Cursor::new(&s)),
    )
    .unwrap();
    assert_eq!(got, b"abc");
}

#[test]
fn max_block_size_rejects_oversized() {
    // Build a stream identifier declaring a small block size but follow with
    // an uncompressed block that exceeds the reader's option ceiling.
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(crate::stream::MIN_BLOCK_SIZE));
    // Reader's option is 4 KiB. Write an 8 KiB uncompressed block.
    push_uncompressed_block(&mut s, &vec![0u8; 8 << 10]);
    let err = ReaderBuilder::new()
        .max_block_size(crate::stream::MIN_BLOCK_SIZE)
        .build(Cursor::new(&s))
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
            v.extend(std::iter::repeat(0u8).take(140));
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
        .build(&mut compressed);
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
    let crc_off = make_stream_header(DEFAULT_BLOCK_SIZE).len() + 4;
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
    block::append_encoded_chunk_body(&mut chunk_body, payload, block::Level::Balanced)
        .expect("encode");
    // Strip the leading varint(dlen) from chunk_body to get the raw
    // compressed bytes that the CRC covers for the 0x03 chunk type.
    let (_, n) = crate::stream::format::get_uvarint(&chunk_body).unwrap();
    let crc_input = &chunk_body[n..];
    let mut s = Vec::new();
    s.extend_from_slice(&make_stream_header(DEFAULT_BLOCK_SIZE));
    let mut body = Vec::new();
    body.extend_from_slice(&masked_crc32c(crc_input).to_le_bytes());
    body.extend_from_slice(&chunk_body);
    push_chunk(&mut s, CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC, &body);
    push_eof(&mut s, payload.len() as u64);
    let got = decode_to_end(Reader::new(Cursor::new(&s))).unwrap();
    assert_eq!(got, payload);
}
