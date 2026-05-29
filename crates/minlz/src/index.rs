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

//! MinLZ stream index: build, encode, decode, and seek.
//!
//! Mirrors `index.go` from the Go reference.  The module is self-contained:
//! it depends on `std` only, so callers wanting just to read or rewrite an
//! index never pay for the block codec.
//!
//! Wire format: `SPEC.md` §4.12.

use std::io::{self, Read, Seek, SeekFrom};

use crate::stream::{MAX_BLOCK_SIZE, MAX_USER_CHUNK_SIZE};

/// MinLZ index chunk ID.
pub const CHUNK_TYPE_INDEX: u8 = 0x40;
/// Legacy S2 / Snappy index chunk ID.  Accepted on [`Index::load`]
/// but never emitted.
pub const LEGACY_INDEX_CHUNK: u8 = 0x99;

const INDEX_HEADER: &[u8; 6] = b"s2idx\x00";
const INDEX_TRAILER: &[u8; 6] = b"\x00xdi2s";
const MAX_INDEX_ENTRIES: usize = 1 << 16;
const MIN_INDEX_DIST: i64 = 1 << 20;
const SKIPPABLE_FRAME_HEADER: usize = 4;
const MAX_VARINT_LEN_64: usize = 10;

// Chunk-type values mirrored locally so this module stays free of cross-
// module visibility tweaks.  Wire-format constants, not behavior.
const CHUNK_TYPE_LEGACY_COMPRESSED_DATA: u8 = 0x00;
const CHUNK_TYPE_UNCOMPRESSED_DATA: u8 = 0x01;
const CHUNK_TYPE_MINLZ_COMPRESSED_DATA: u8 = 0x02;
const CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC: u8 = 0x03;
const CHUNK_TYPE_EOF: u8 = 0x20;
const CHUNK_TYPE_STREAM_IDENTIFIER: u8 = 0xff;
const MAX_NON_SKIPPABLE_CHUNK: u8 = 0x3f;
const CHECKSUM_SIZE: usize = 4;
const MAGIC_BODY_MINLZ: &[u8] = b"MinLz";
const MAGIC_BODY_S2: &[u8] = b"sNaPpY";
const MAGIC_BODY_LEN: usize = 6;

/// Pair of `(compressed, uncompressed)` stream offsets.  Sorted ascending
/// by both fields inside an [`Index`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OffsetPair {
    /// Compressed stream offset, counted from the first byte of the stream.
    pub compressed: i64,
    /// Uncompressed offset corresponding to the same point in the stream.
    pub uncompressed: i64,
}

/// MinLZ stream index.
///
/// Built incrementally by the writer (one `(compressed, uncompressed)` pair
/// every ~`est_block_uncomp` bytes) or recovered from the end of an
/// existing stream via [`Index::load_stream`] / [`Index::load`].
#[derive(Debug, Clone, Default)]
pub struct Index {
    /// Total uncompressed size, or `-1` if unknown.
    pub total_uncompressed: i64,
    /// Total compressed size, or `-1` if unknown (e.g. padded streams).
    pub total_compressed: i64,
    /// Index entries.  Sorted ascending by both offsets.
    pub offsets: Vec<OffsetPair>,
    est_block_uncomp: i64,
}

impl Index {
    /// Empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset to an empty index ready to receive entries for a stream whose
    /// maximum block size is `max_block`.  Matches Go `Index.reset`.
    pub fn reset(&mut self, max_block: usize) {
        let mut mb = max_block as i64;
        while mb < MIN_INDEX_DIST {
            mb *= 2;
        }
        self.est_block_uncomp = mb;
        self.total_uncompressed = -1;
        self.total_compressed = -1;
        self.offsets.clear();
    }

    /// Estimated uncompressed block size used by the delta encoder.
    pub fn est_block_uncomp(&self) -> i64 {
        self.est_block_uncomp
    }

    /// Append an entry to the index.  Entries *must* arrive in increasing
    /// order.
    ///
    /// Matches Go `Index.add`:
    /// * If the uncompressed gap from the last entry is `< est_block_uncomp`,
    ///   the call is a no-op (returns `Ok(())`) — used to thin the index
    ///   during writing.
    /// * Otherwise the new offsets must be ≥ the last ones (decreasing
    ///   offsets are rejected with `io::ErrorKind::InvalidData`).
    /// * After insertion, if the entry count would exceed 65 535, the
    ///   internal `reduce_light` shrinker doubles `est_block_uncomp` and
    ///   re-thins entries in place.
    pub fn add(&mut self, compressed: i64, uncompressed: i64) -> io::Result<()> {
        if let Some(&latest) = self.offsets.last() {
            let gap = uncompressed
                .checked_sub(latest.uncompressed)
                .ok_or_else(|| corrupt("index: uncompressed offset overflow"))?;
            if gap < self.est_block_uncomp {
                return Ok(());
            }
            if latest.uncompressed > uncompressed {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "minlz: index uncompressed offset decreased ({} > {})",
                        latest.uncompressed, uncompressed
                    ),
                ));
            }
            if latest.compressed > compressed {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "minlz: index compressed offset decreased ({} > {})",
                        latest.compressed, compressed
                    ),
                ));
            }
        }
        self.offsets.push(OffsetPair {
            compressed,
            uncompressed,
        });
        if self.offsets.len() > MAX_INDEX_ENTRIES {
            self.reduce_light();
        }
        Ok(())
    }

    /// Look up the index entry at or before `offset` (an uncompressed
    /// position).
    ///
    /// `offset < 0` is interpreted as a distance from the end (`-1` is the
    /// last byte) and requires [`total_uncompressed`](Self::total_uncompressed)
    /// to be known (`!= -1`).
    ///
    /// Returns `(compressed_offset, uncompressed_offset)`.  Maps Go
    /// `Index.Find`.
    pub fn find(&self, mut offset: i64) -> io::Result<(i64, i64)> {
        if self.total_uncompressed < 0 {
            return Err(corrupt("index: total uncompressed unknown"));
        }
        if offset < 0 {
            offset = match offset.checked_add(self.total_uncompressed) {
                Some(o) if o >= 0 => o,
                _ => return Err(io::Error::from(io::ErrorKind::UnexpectedEof)),
            };
        }
        if offset > self.total_uncompressed {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        if self.offsets.is_empty() {
            return Ok((0, 0));
        }
        if self.offsets.len() > 200 {
            let n = self.offsets.partition_point(|p| p.uncompressed <= offset);
            let i = if n == 0 { 0 } else { n - 1 };
            return Ok((self.offsets[i].compressed, self.offsets[i].uncompressed));
        }
        let mut c = 0i64;
        let mut u = 0i64;
        for p in &self.offsets {
            if p.uncompressed > offset {
                break;
            }
            c = p.compressed;
            u = p.uncompressed;
        }
        Ok((c, u))
    }

    /// Reduce entry count to ≤ 65 535 by dropping entries while widening
    /// the implicit block size.  Matches Go `Index.reduce`.
    fn reduce(&mut self) {
        if self.offsets.len() < MAX_INDEX_ENTRIES {
            return;
        }
        let mut remove_n = (self.offsets.len() + 1) / MAX_INDEX_ENTRIES;
        while self.est_block_uncomp * (remove_n as i64 + 1) < MIN_INDEX_DIST
            && self.offsets.len() / (remove_n + 1) > 1000
        {
            remove_n += 1;
        }
        let mut j = 0usize;
        let mut idx = 0usize;
        while idx < self.offsets.len() {
            self.offsets[j] = self.offsets[idx];
            j += 1;
            idx += 1 + remove_n;
        }
        self.offsets.truncate(j);
        self.est_block_uncomp += self.est_block_uncomp * remove_n as i64;
    }

    /// Halve the in-memory footprint by doubling `est_block_uncomp` and
    /// dropping entries that fall within the new spacing.  Matches Go
    /// `Index.reduceLight`.
    fn reduce_light(&mut self) {
        self.est_block_uncomp *= 2;
        let mut j = 0usize;
        let mut idx = 0usize;
        while idx < self.offsets.len() {
            let base = self.offsets[idx];
            self.offsets[j] = base;
            j += 1;
            idx += 1;
            while idx < self.offsets.len()
                && self.offsets[idx].uncompressed - base.uncompressed < self.est_block_uncomp
            {
                idx += 1;
            }
        }
        self.offsets.truncate(j);
    }

    /// Encode this index into `buf` as a self-contained 0x40 chunk.
    /// The chunk includes header, trailer, and chunk-length prefix — the
    /// output can be concatenated to a closed MinLZ stream.
    ///
    /// `total_uncomp` / `total_comp` are recorded in the index.  Pass
    /// `-1` for either when the value is unknown (e.g. padding rewrites
    /// the compressed total).
    ///
    /// Matches Go `Index.appendTo`.
    pub fn append_to(&mut self, buf: &mut Vec<u8>, total_uncomp: i64, total_comp: i64) {
        self.reduce();
        let init = buf.len();
        buf.extend_from_slice(&[CHUNK_TYPE_INDEX, 0, 0, 0]);
        buf.extend_from_slice(INDEX_HEADER);
        put_varint(buf, total_uncomp);
        put_varint(buf, total_comp);
        put_varint(buf, self.est_block_uncomp);
        put_varint(buf, self.offsets.len() as i64);

        let mut has_uncomp: u8 = 0;
        for (idx, info) in self.offsets.iter().enumerate() {
            if idx == 0 {
                if info.uncompressed != 0 {
                    has_uncomp = 1;
                    break;
                }
                continue;
            }
            let prev = self.offsets[idx - 1].uncompressed;
            if info.uncompressed != prev + self.est_block_uncomp {
                has_uncomp = 1;
                break;
            }
        }
        buf.push(has_uncomp);

        if has_uncomp == 1 {
            for (idx, info) in self.offsets.iter().enumerate() {
                let mut u = info.uncompressed;
                if idx > 0 {
                    let prev = self.offsets[idx - 1].uncompressed;
                    u -= prev + self.est_block_uncomp;
                }
                put_varint(buf, u);
            }
        }

        let mut c_predict = self.est_block_uncomp / 2;
        for (idx, info) in self.offsets.iter().enumerate() {
            let mut c = info.compressed;
            if idx > 0 {
                let prev = self.offsets[idx - 1].compressed;
                c -= prev + c_predict;
                c_predict += c / 2;
            }
            put_varint(buf, c);
        }

        let size = (buf.len() - init + 4 + INDEX_TRAILER.len()) as u32;
        buf.extend_from_slice(&size.to_le_bytes());
        buf.extend_from_slice(INDEX_TRAILER);

        let chunk_len = (buf.len() - init - SKIPPABLE_FRAME_HEADER) as u32;
        buf[init + 1] = chunk_len as u8;
        buf[init + 2] = (chunk_len >> 8) as u8;
        buf[init + 3] = (chunk_len >> 16) as u8;
    }

    /// Decode an index from `bytes`.  Accepts both the canonical 0x40
    /// chunk and the legacy 0x99 chunk; matches Go `Index.Load`.
    ///
    /// Returns the unconsumed tail (bytes after the chunk's trailer).
    pub fn load<'a>(&mut self, bytes: &'a [u8]) -> io::Result<&'a [u8]> {
        if bytes.len() <= 4 + INDEX_HEADER.len() + INDEX_TRAILER.len() {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        if bytes[0] != CHUNK_TYPE_INDEX && bytes[0] != LEGACY_INDEX_CHUNK {
            return Err(corrupt("index: bad chunk type"));
        }
        let chunk_len =
            (bytes[1] as usize) | ((bytes[2] as usize) << 8) | ((bytes[3] as usize) << 16);
        let mut b = &bytes[4..];
        if b.len() < chunk_len {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        if &b[..INDEX_HEADER.len()] != INDEX_HEADER.as_slice() {
            return Err(unsupported("index: bad header"));
        }
        b = &b[INDEX_HEADER.len()..];

        let (v, n) = read_varint(b)?;
        if v < 0 {
            return Err(corrupt("index: negative total uncompressed"));
        }
        self.total_uncompressed = v;
        b = &b[n..];

        let (v, n) = read_varint(b)?;
        self.total_compressed = v;
        b = &b[n..];

        let (v, n) = read_varint(b)?;
        if v < 0 {
            return Err(corrupt("index: negative est_block"));
        }
        self.est_block_uncomp = v;
        b = &b[n..];

        let (v, n) = read_varint(b)?;
        if v < 0 || v > MAX_INDEX_ENTRIES as i64 {
            return Err(corrupt("index: entry count out of range"));
        }
        let entries = v as usize;
        b = &b[n..];

        if entries > self.offsets.capacity() {
            self.offsets = Vec::with_capacity(entries);
        } else {
            self.offsets.clear();
        }
        self.offsets.resize(entries, OffsetPair::default());

        if b.is_empty() {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        let has_uncomp = b[0];
        b = &b[1..];
        if has_uncomp > 1 {
            return Err(corrupt("index: invalid has_uncompressed byte"));
        }

        for idx in 0..entries {
            let mut u_off = 0i64;
            if has_uncomp != 0 {
                let (v, n) = read_varint(b)?;
                u_off = v;
                b = &b[n..];
            }
            if idx > 0 {
                let prev = self.offsets[idx - 1].uncompressed;
                u_off = prev
                    .checked_add(self.est_block_uncomp)
                    .and_then(|s| u_off.checked_add(s))
                    .ok_or_else(|| corrupt("index: uncompressed offset overflow"))?;
                if u_off <= prev {
                    return Err(corrupt("index: non-monotonic uncompressed offset"));
                }
            }
            if u_off < 0 {
                return Err(corrupt("index: negative uncompressed offset"));
            }
            self.offsets[idx].uncompressed = u_off;
        }

        let mut c_predict = self.est_block_uncomp / 2;
        for idx in 0..entries {
            let (mut c_off, n) = read_varint(b)?;
            b = &b[n..];
            if idx > 0 {
                let predict_next = c_predict
                    .checked_add(c_off / 2)
                    .ok_or_else(|| corrupt("index: compressed predictor overflow"))?;
                let prev = self.offsets[idx - 1].compressed;
                c_off = prev
                    .checked_add(c_predict)
                    .and_then(|s| c_off.checked_add(s))
                    .ok_or_else(|| corrupt("index: compressed offset overflow"))?;
                if c_off <= prev {
                    return Err(corrupt("index: non-monotonic compressed offset"));
                }
                c_predict = predict_next;
            }
            if c_off < 0 {
                return Err(corrupt("index: negative compressed offset"));
            }
            self.offsets[idx].compressed = c_off;
        }

        if b.len() < 4 + INDEX_TRAILER.len() {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        b = &b[4..]; // stored size, redundant with chunk header
        if &b[..INDEX_TRAILER.len()] != INDEX_TRAILER.as_slice() {
            return Err(corrupt("index: bad trailer"));
        }
        Ok(&b[INDEX_TRAILER.len()..])
    }

    /// Load an index by seeking to the end of `rs`, finding the trailer,
    /// then reading the index chunk.  Matches Go `Index.LoadStream`.
    pub fn load_stream<R: Read + Seek>(&mut self, rs: &mut R) -> io::Result<()> {
        rs.seek(SeekFrom::End(-10))?;
        let mut tail = [0u8; 10];
        rs.read_exact(&mut tail)?;
        if &tail[4..4 + INDEX_TRAILER.len()] != INDEX_TRAILER.as_slice() {
            return Err(unsupported("index: trailer not found"));
        }
        let sz = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
        if sz as usize > MAX_USER_CHUNK_SIZE + SKIPPABLE_FRAME_HEADER {
            return Err(corrupt("index: size exceeds maximum"));
        }
        rs.seek(SeekFrom::End(-(sz as i64)))?;
        let mut buf = vec![0u8; sz as usize];
        rs.read_exact(&mut buf)?;
        self.load(&buf)?;
        Ok(())
    }

    /// Render this index as JSON.  Matches Go `Index.JSON`'s field order
    /// (`encoding/json` defaults with two-space indent).
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(64 + self.offsets.len() * 64);
        s.push_str("{\n");
        s.push_str(&format!(
            "  \"total_uncompressed\": {},\n",
            self.total_uncompressed
        ));
        s.push_str(&format!(
            "  \"total_compressed\": {},\n",
            self.total_compressed
        ));
        if self.offsets.is_empty() {
            s.push_str("  \"offsets\": null,\n");
        } else {
            s.push_str("  \"offsets\": [\n");
            for (i, o) in self.offsets.iter().enumerate() {
                s.push_str("    {\n");
                s.push_str(&format!("      \"compressed\": {},\n", o.compressed));
                s.push_str(&format!("      \"uncompressed\": {}\n", o.uncompressed));
                s.push_str(if i + 1 == self.offsets.len() {
                    "    }\n"
                } else {
                    "    },\n"
                });
            }
            s.push_str("  ],\n");
        }
        s.push_str(&format!(
            "  \"est_block_uncompressed\": {}\n",
            self.est_block_uncomp
        ));
        s.push('}');
        s
    }
}

/// Walk a MinLZ stream end-to-end and build an index from the chunk
/// headers without decoding any block payload.  The returned bytes are
/// the encoded 0x40 chunk (ready to be appended to the stream or stored
/// separately).
///
/// Matches Go `IndexStream`.
pub fn index_stream<R: Read>(mut r: R) -> io::Result<Vec<u8>> {
    let mut idx = Index {
        total_uncompressed: 0,
        total_compressed: 0,
        offsets: Vec::new(),
        est_block_uncomp: 0,
    };

    let mut hdr = [0u8; 4];
    let mut buf = vec![0u8; MAX_USER_CHUNK_SIZE];
    let mut read_header = false;

    loop {
        match read_exact_or_eof(&mut r, &mut hdr)? {
            ReadOutcome::Eof => {
                let mut out = Vec::new();
                idx.append_to(&mut out, idx.total_uncompressed, idx.total_compressed);
                return Ok(out);
            }
            ReadOutcome::Ok => {}
        }
        let start_chunk = idx.total_compressed;
        idx.total_compressed += 4;

        let chunk_type = hdr[0];
        if !read_header {
            if chunk_type != CHUNK_TYPE_STREAM_IDENTIFIER && chunk_type != CHUNK_TYPE_EOF {
                return Err(corrupt("index_stream: missing stream identifier"));
            }
            read_header = true;
        }
        let chunk_len = (hdr[1] as usize) | ((hdr[2] as usize) << 8) | ((hdr[3] as usize) << 16);
        idx.total_compressed += chunk_len as i64;

        let body = &mut buf[..chunk_len];
        r.read_exact(body)
            .map_err(|_| io::Error::from(io::ErrorKind::UnexpectedEof))?;

        match chunk_type {
            CHUNK_TYPE_LEGACY_COMPRESSED_DATA
            | CHUNK_TYPE_MINLZ_COMPRESSED_DATA
            | CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC => {
                if chunk_len < CHECKSUM_SIZE {
                    return Err(corrupt("index_stream: data chunk shorter than CRC"));
                }
                let d_len = crate::block::decoded_len_chunk_body(&body[CHECKSUM_SIZE..])
                    .map_err(|_| corrupt("index_stream: bad block size header"))?;
                if d_len > MAX_BLOCK_SIZE {
                    return Err(corrupt("index_stream: block exceeds max"));
                }
                if idx.est_block_uncomp == 0 {
                    idx.est_block_uncomp = d_len as i64;
                }
                idx.add(start_chunk, idx.total_uncompressed)?;
                idx.total_uncompressed += d_len as i64;
            }
            CHUNK_TYPE_UNCOMPRESSED_DATA => {
                if chunk_len < CHECKSUM_SIZE {
                    return Err(corrupt("index_stream: uncompressed chunk shorter than CRC"));
                }
                let n2 = chunk_len - CHECKSUM_SIZE;
                if n2 > MAX_BLOCK_SIZE {
                    return Err(corrupt("index_stream: uncompressed block exceeds max"));
                }
                if idx.est_block_uncomp == 0 {
                    idx.est_block_uncomp = n2 as i64;
                }
                idx.add(start_chunk, idx.total_uncompressed)?;
                idx.total_uncompressed += n2 as i64;
            }
            CHUNK_TYPE_STREAM_IDENTIFIER => {
                if chunk_len != MAGIC_BODY_LEN {
                    return Err(corrupt("index_stream: bad stream identifier length"));
                }
                let head5 = &body[..MAGIC_BODY_MINLZ.len()];
                if head5 != MAGIC_BODY_MINLZ && &body[..MAGIC_BODY_LEN] != MAGIC_BODY_S2 {
                    return Err(corrupt("index_stream: unknown stream magic"));
                }
            }
            CHUNK_TYPE_EOF => {}
            t if t <= MAX_NON_SKIPPABLE_CHUNK => {
                return Err(unsupported("index_stream: reserved unskippable chunk"));
            }
            _ => {} // skippable / padding / user chunks
        }
    }
}

/// Strip the 0x40 chunk envelope from an encoded index, leaving only the
/// payload.  Returns `None` on malformed input.
///
/// Matches Go `RemoveIndexHeaders`.
pub fn remove_index_headers(b: &[u8]) -> Option<&[u8]> {
    let save = 4 + INDEX_HEADER.len() + INDEX_TRAILER.len() + 4;
    if b.len() <= save {
        return None;
    }
    if b[0] != CHUNK_TYPE_INDEX {
        return None;
    }
    let chunk_len = (b[1] as usize) | ((b[2] as usize) << 8) | ((b[3] as usize) << 16);
    let b = &b[4..];
    if b.len() < chunk_len {
        return None;
    }
    let b = &b[..chunk_len];
    if &b[..INDEX_HEADER.len()] != INDEX_HEADER.as_slice() {
        return None;
    }
    let b = &b[INDEX_HEADER.len()..];
    if !b.ends_with(INDEX_TRAILER.as_slice()) {
        return None;
    }
    let b = &b[..b.len() - INDEX_TRAILER.len()];
    if b.len() < 4 {
        return None;
    }
    Some(&b[..b.len() - 4])
}

/// Re-wrap a payload produced by [`remove_index_headers`] back into a
/// valid 0x40 chunk.  Matches Go `RestoreIndexHeaders`.
pub fn restore_index_headers(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut b = Vec::with_capacity(4 + INDEX_HEADER.len() + input.len() + INDEX_TRAILER.len() + 4);
    b.extend_from_slice(&[CHUNK_TYPE_INDEX, 0, 0, 0]);
    b.extend_from_slice(INDEX_HEADER);
    b.extend_from_slice(input);
    let size = (b.len() + 4 + INDEX_TRAILER.len()) as u32;
    b.extend_from_slice(&size.to_le_bytes());
    b.extend_from_slice(INDEX_TRAILER);
    let chunk_len = (b.len() - SKIPPABLE_FRAME_HEADER) as u32;
    b[1] = chunk_len as u8;
    b[2] = (chunk_len >> 8) as u8;
    b[3] = (chunk_len >> 16) as u8;
    b
}

// -------- internal helpers --------

fn corrupt(msg: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn unsupported(msg: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, msg)
}

enum ReadOutcome {
    Ok,
    Eof,
}

fn read_exact_or_eof<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<ReadOutcome> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => {
                if n == 0 {
                    return Ok(ReadOutcome::Eof);
                }
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
            }
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(ReadOutcome::Ok)
}

fn put_uvarint(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

fn put_varint(buf: &mut Vec<u8>, v: i64) {
    let zz = ((v as u64) << 1) ^ ((v >> 63) as u64);
    put_uvarint(buf, zz)
}

fn read_uvarint(b: &[u8]) -> io::Result<(u64, usize)> {
    let mut x = 0u64;
    let mut s = 0u32;
    for (i, &byte) in b.iter().enumerate() {
        if i == MAX_VARINT_LEN_64 {
            return Err(corrupt("index: varint overflow"));
        }
        if byte < 0x80 {
            if i == MAX_VARINT_LEN_64 - 1 && byte > 1 {
                return Err(corrupt("index: varint overflow"));
            }
            return Ok((x | (u64::from(byte) << s), i + 1));
        }
        x |= u64::from(byte & 0x7f) << s;
        s += 7;
    }
    Err(io::Error::from(io::ErrorKind::UnexpectedEof))
}

fn read_varint(b: &[u8]) -> io::Result<(i64, usize)> {
    let (u, n) = read_uvarint(b)?;
    let v = ((u >> 1) as i64) ^ -((u & 1) as i64);
    Ok((v, n))
}

#[cfg(test)]
mod tests;
