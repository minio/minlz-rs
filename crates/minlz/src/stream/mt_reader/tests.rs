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

use std::io::{Cursor, Write};

use crate::stream::{ConcurrentDecode, MtWriterBuilder, Reader};

fn encode_mt(payload: &[u8], concurrency: usize, block_size: Option<usize>) -> Vec<u8> {
    let n = concurrency;
    let mut b = MtWriterBuilder::new().concurrency(n);
    if let Some(bs) = block_size {
        b = b.block_size(bs);
    }
    let mut w = b.build(Vec::new()).unwrap();
    w.write_all(payload).unwrap();
    w.finish().unwrap()
}

#[test]
fn decode_concurrent_empty() {
    let stream = encode_mt(b"", 4, None);
    let mut reader = Reader::new(Cursor::new(stream));
    let ConcurrentDecode {
        bytes_written: n,
        writer: sink,
    } = reader.decode_concurrent(Vec::<u8>::new(), 4).unwrap();
    assert_eq!(n, 0);
    assert!(sink.is_empty());
}

#[test]
fn decode_concurrent_small() {
    let payload = b"Hello, MinLZ!".to_vec();
    let stream = encode_mt(&payload, 4, None);
    let mut reader = Reader::new(Cursor::new(stream));
    let ConcurrentDecode {
        bytes_written: n,
        writer: sink,
    } = reader.decode_concurrent(Vec::<u8>::new(), 4).unwrap();
    assert_eq!(n as usize, payload.len());
    assert_eq!(sink, payload);
}

#[test]
fn decode_concurrent_multi_block() {
    let payload: Vec<u8> = (0..32 * 1024).map(|i| (i * 17) as u8).collect();
    let stream = encode_mt(&payload, 4, Some(4 << 10));
    let mut reader = Reader::new(Cursor::new(stream));
    let ConcurrentDecode {
        bytes_written: n,
        writer: sink,
    } = reader.decode_concurrent(Vec::<u8>::new(), 4).unwrap();
    assert_eq!(n as usize, payload.len());
    assert_eq!(sink, payload);
}

#[test]
fn decode_concurrent_matches_st_for_st_encoded() {
    // Encode with the single-threaded Writer, decode with MT reader.
    let payload: Vec<u8> = (0..40 * 1024).map(|i| (i * 31) as u8).collect();
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut w = crate::stream::WriterBuilder::new()
            .block_size(4 << 10)
            .build(&mut buf)
            .unwrap();
        w.write_all(&payload).unwrap();
        let _ = w.finish().unwrap();
    }
    let mut reader = Reader::new(Cursor::new(&buf));
    let ConcurrentDecode {
        bytes_written: n,
        writer: sink,
    } = reader.decode_concurrent(Vec::<u8>::new(), 4).unwrap();
    assert_eq!(n as usize, payload.len());
    assert_eq!(sink, payload);
}

#[test]
fn cross_concurrency_round_trips() {
    let payload: Vec<u8> = (0..64 * 1024).map(|i| (i & 0xff) as u8).collect();
    for &enc_n in &[1, 2, 4, 8] {
        for &dec_n in &[1, 2, 4, 8] {
            let stream = encode_mt(&payload, enc_n, Some(4 << 10));
            let mut reader = Reader::new(Cursor::new(&stream));
            let ConcurrentDecode {
                bytes_written: n,
                writer: sink,
            } = reader.decode_concurrent(Vec::<u8>::new(), dec_n).unwrap();
            assert_eq!(n as usize, payload.len(), "enc={enc_n} dec={dec_n}");
            assert_eq!(sink, payload, "enc={enc_n} dec={dec_n}");
        }
    }
}

#[test]
fn decode_concurrent_rejects_after_read() {
    use std::io::Read;
    let payload = b"hello".to_vec();
    let stream = encode_mt(&payload, 2, None);
    let mut reader = Reader::new(Cursor::new(stream));
    // Drain one byte via the single-threaded path so decoded_pos < decoded.len().
    let mut tmp = [0u8; 1];
    reader.read_exact(&mut tmp).unwrap();
    let err = reader.decode_concurrent(Vec::<u8>::new(), 4).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
}

#[test]
fn decode_concurrent_propagates_crc_error() {
    let payload = b"some payload".to_vec();
    let mut stream = encode_mt(&payload, 2, None);
    // Flip a byte in the body region (past the stream header, past chunk
    // header + CRC).  This should fail CRC at decode time.
    let pos = stream.len() - 5;
    stream[pos] ^= 0xff;
    let mut reader = Reader::new(Cursor::new(stream));
    let err = reader.decode_concurrent(Vec::<u8>::new(), 4).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

/// Cross-impl: decode a Go-MT-encoded stream with Rust MT.  Gated on
/// `MINLZ_GO_CMD` so the test is skipped when Go isn't available.
#[test]
fn go_mt_encoded_decodes_in_rust_mt() {
    use std::io::Write as _;
    use std::process::{Command, Stdio};
    let Ok(go_bin) = std::env::var("MINLZ_GO_CMD") else {
        return;
    };
    let payload: Vec<u8> = (0..96 * 1024)
        .map(|i| ((i * 17) ^ (i >> 3)) as u8)
        .collect();
    for cpu in [1, 4, 8] {
        let mut child = Command::new(&go_bin)
            .args(["c", &format!("-cpu={cpu}"), "-2", "-c", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn go mz");
        child.stdin.as_mut().unwrap().write_all(&payload).unwrap();
        let out = child.wait_with_output().expect("wait");
        assert!(out.status.success(), "Go encode failed at cpu={cpu}");
        let stream = out.stdout;

        let mut reader = Reader::new(Cursor::new(&stream));
        let ConcurrentDecode {
            bytes_written: n,
            writer: dec,
        } = reader
            .decode_concurrent(Vec::<u8>::with_capacity(payload.len()), 4)
            .expect("rust decode");
        assert_eq!(n as usize, payload.len(), "cpu={cpu}");
        assert_eq!(dec, payload, "cpu={cpu}");
    }
}
