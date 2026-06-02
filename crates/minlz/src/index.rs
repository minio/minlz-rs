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
//! Mirrors `index.go` from the Go reference.  Public offsets are `u64`
//! (they are byte positions, never negative); the wire format still
//! delta-encodes them as signed zigzag varints, so signed arithmetic is
//! confined to the codec internals here.
//!
//! Wire format: `SPEC.md` §4.12.

use core::fmt;
use std::io::{self, Read, Seek, SeekFrom};

use crate::stream::{MAX_BLOCK_SIZE, MAX_USER_CHUNK_SIZE};

/// Errors from index parsing, construction, and seeking.
///
/// Carries a description and maps onto a sensible [`io::ErrorKind`] when
/// converted to [`io::Error`] (`Truncated` → `UnexpectedEof`, `BadFormat` →
/// `Unsupported`, `Invalid` → `InvalidData`).
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The index data ended mid-parse (truncated chunk, varint, or trailer).
    Truncated,
    /// Not a MinLZ index: wrong chunk type, header, or trailer.
    BadFormat(&'static str),
    /// A decoded field violated an invariant: out of range, non-monotonic,
    /// negative, or too many entries.
    Invalid(&'static str),
    /// An I/O error from the underlying reader/seeker (only from
    /// [`Index::load_stream`] and [`index_stream`]).
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Truncated => f.write_str("minlz index: truncated"),
            Error::BadFormat(m) => write!(f, "minlz index: {m}"),
            Error::Invalid(m) => write!(f, "minlz index: {m}"),
            Error::Io(e) => write!(f, "minlz index: I/O error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<Error> for io::Error {
    fn from(e: Error) -> io::Error {
        match e {
            Error::Io(inner) => inner,
            Error::Truncated => {
                io::Error::new(io::ErrorKind::UnexpectedEof, "minlz index: truncated")
            }
            Error::BadFormat(m) => {
                io::Error::new(io::ErrorKind::Unsupported, format!("minlz index: {m}"))
            }
            Error::Invalid(m) => {
                io::Error::new(io::ErrorKind::InvalidData, format!("minlz index: {m}"))
            }
        }
    }
}

/// Specialized [`Result`] for index operations.
pub type Result<T> = core::result::Result<T, Error>;

/// MinLZ index chunk ID.
pub const CHUNK_TYPE_INDEX: u8 = 0x40;
/// Legacy S2 / Snappy index chunk ID.  Accepted on [`Index::load`]
/// but never emitted.
pub const LEGACY_INDEX_CHUNK: u8 = 0x99;

const INDEX_HEADER: &[u8; 6] = b"s2idx\x00";
const INDEX_TRAILER: &[u8; 6] = b"\x00xdi2s";
const MAX_INDEX_ENTRIES: usize = 1 << 16;
const MIN_INDEX_DIST: u64 = 1 << 20;
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

/// Pair of `(compressed, uncompressed)` stream offsets, in bytes from the
/// start of the stream.  Sorted ascending by both fields inside an [`Index`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OffsetPair {
    /// Compressed stream offset, counted from the first byte of the stream.
    pub compressed: u64,
    /// Uncompressed offset corresponding to the same point in the stream.
    pub uncompressed: u64,
}

/// MinLZ stream index.
///
/// Built incrementally by the writer (one `(compressed, uncompressed)` pair
/// every ~`est_block_uncomp` bytes) or recovered from the end of an
/// existing stream via [`Index::load_stream`] / [`Index::load`].
///
/// The entries are kept sorted ascending by both offsets; the fields are
/// private to preserve that invariant ([`find`](Self::find) relies on it).
#[derive(Debug, Clone, Default)]
pub struct Index {
    /// Total uncompressed size, or `None` if unknown (still being built).
    total_uncompressed: Option<u64>,
    /// Total compressed size, or `None` if unknown (e.g. padded streams).
    total_compressed: Option<u64>,
    /// Index entries.  Sorted ascending by both offsets.
    offsets: Vec<OffsetPair>,
    est_block_uncomp: u64,
}

impl Index {
    /// Empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total uncompressed size, once known (after [`load`](Self::load) /
    /// [`load_stream`](Self::load_stream) or [`append_to`](Self::append_to)).
    pub fn total_uncompressed(&self) -> Option<u64> {
        self.total_uncompressed
    }

    /// Total compressed size, once known.  `None` for padded streams.
    pub fn total_compressed(&self) -> Option<u64> {
        self.total_compressed
    }

    /// The index entries, ascending by offset.
    pub fn offsets(&self) -> &[OffsetPair] {
        &self.offsets
    }

    /// Reset to an empty index ready to receive entries for a stream whose
    /// maximum block size is `max_block`.  Matches Go `Index.reset`.
    pub fn reset(&mut self, max_block: usize) {
        let mut mb = max_block as u64;
        while mb < MIN_INDEX_DIST {
            mb *= 2;
        }
        self.est_block_uncomp = mb;
        self.total_uncompressed = None;
        self.total_compressed = None;
        self.offsets.clear();
    }

    /// Estimated uncompressed block size used by the delta encoder. Internal
    /// heuristic; test-only accessor (tests read it to verify the spacing).
    #[cfg(test)]
    fn est_block_uncomp(&self) -> u64 {
        self.est_block_uncomp
    }

    /// Append an entry to the index.  Entries *must* arrive in increasing
    /// order.
    ///
    /// Matches Go `Index.add`:
    /// * If the uncompressed gap from the last entry is `< est_block_uncomp`
    ///   (including a non-increasing offset), the call is a no-op — used to
    ///   thin the index during writing.
    /// * Otherwise a *decreasing* compressed offset is rejected
    ///   ([`Error::Invalid`]).
    /// * After insertion, if the entry count would exceed 65 535, the
    ///   internal `reduce_light` shrinker doubles `est_block_uncomp` and
    ///   re-thins entries in place.
    pub fn add(&mut self, compressed: u64, uncompressed: u64) -> Result<()> {
        // The wire format encodes offsets as signed (zigzag) varints, so a
        // value past `i64::MAX` cannot round-trip. Reject it here rather than
        // silently wrapping at encode time.
        if compressed > i64::MAX as u64 || uncompressed > i64::MAX as u64 {
            return Err(Error::Invalid("offset exceeds i64 wire range"));
        }
        if let Some(&latest) = self.offsets.last() {
            // Non-increasing uncompressed offset → gap 0 → skipped below.
            let gap = uncompressed.saturating_sub(latest.uncompressed);
            if gap < self.est_block_uncomp {
                return Ok(());
            }
            if latest.compressed > compressed {
                return Err(Error::Invalid("compressed offset decreased"));
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

    /// The index entry at or before `uncompressed_offset` (the largest entry
    /// whose uncompressed offset is `<= offset`).  Decode from
    /// [`compressed`](OffsetPair::compressed), then discard
    /// `offset - uncompressed` bytes.  An empty index returns the zero pair;
    /// an offset past the last entry returns the last entry.  Maps Go
    /// `Index.Find` (callers wanting "distance from end" pass
    /// `total_uncompressed - dist`; bounds are enforced by the seek layer).
    pub fn find(&self, uncompressed_offset: u64) -> OffsetPair {
        if self.offsets.is_empty() {
            return OffsetPair::default();
        }
        if self.offsets.len() > 200 {
            let n = self
                .offsets
                .partition_point(|p| p.uncompressed <= uncompressed_offset);
            let i = if n == 0 { 0 } else { n - 1 };
            return self.offsets[i];
        }
        let mut best = OffsetPair::default();
        for &p in &self.offsets {
            if p.uncompressed > uncompressed_offset {
                break;
            }
            best = p;
        }
        best
    }

    /// Reduce entry count to ≤ 65 535 by dropping entries while widening
    /// the implicit block size.  Matches Go `Index.reduce`.
    fn reduce(&mut self) {
        if self.offsets.len() < MAX_INDEX_ENTRIES {
            return;
        }
        let mut remove_n = (self.offsets.len() + 1) / MAX_INDEX_ENTRIES;
        while self.est_block_uncomp * (remove_n as u64 + 1) < MIN_INDEX_DIST
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
        self.est_block_uncomp += self.est_block_uncomp * remove_n as u64;
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
    /// `total_uncomp` / `total_comp` are recorded in the index (and on the
    /// wire as `-1` when `None`, e.g. a padded compressed total).
    ///
    /// Matches Go `Index.appendTo`.
    ///
    /// # Errors
    /// [`Error::Invalid`] if either total exceeds `i64::MAX`; the signed wire
    /// format cannot represent it. (Stored offsets are already bounded by
    /// [`add`](Self::add).)
    pub fn append_to(
        &mut self,
        buf: &mut Vec<u8>,
        total_uncomp: Option<u64>,
        total_comp: Option<u64>,
    ) -> Result<()> {
        for t in [total_uncomp, total_comp].into_iter().flatten() {
            if t > i64::MAX as u64 {
                return Err(Error::Invalid("index total exceeds i64 wire range"));
            }
        }
        self.reduce();
        self.total_uncompressed = total_uncomp;
        self.total_compressed = total_comp;
        let init = buf.len();
        buf.extend_from_slice(&[CHUNK_TYPE_INDEX, 0, 0, 0]);
        buf.extend_from_slice(INDEX_HEADER);
        put_varint(buf, opt_to_wire(total_uncomp));
        put_varint(buf, opt_to_wire(total_comp));
        put_varint(buf, self.est_block_uncomp as i64);
        put_varint(buf, self.offsets.len() as i64);

        let est = self.est_block_uncomp;
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
            if info.uncompressed != prev + est {
                has_uncomp = 1;
                break;
            }
        }
        buf.push(has_uncomp);

        if has_uncomp == 1 {
            for (idx, info) in self.offsets.iter().enumerate() {
                let mut u = info.uncompressed as i64;
                if idx > 0 {
                    let prev = self.offsets[idx - 1].uncompressed as i64;
                    u -= prev + est as i64;
                }
                put_varint(buf, u);
            }
        }

        let mut c_predict = (est / 2) as i64;
        for (idx, info) in self.offsets.iter().enumerate() {
            let mut c = info.compressed as i64;
            if idx > 0 {
                let prev = self.offsets[idx - 1].compressed as i64;
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
        Ok(())
    }

    /// Decode an index from `bytes`.  Accepts both the canonical 0x40
    /// chunk and the legacy 0x99 chunk; matches Go `Index.Load`.
    ///
    /// Returns the unconsumed tail (bytes after the chunk's trailer).
    pub fn load<'a>(&mut self, bytes: &'a [u8]) -> Result<&'a [u8]> {
        if bytes.len() <= 4 + INDEX_HEADER.len() + INDEX_TRAILER.len() {
            return Err(Error::Truncated);
        }
        if bytes[0] != CHUNK_TYPE_INDEX && bytes[0] != LEGACY_INDEX_CHUNK {
            return Err(Error::BadFormat("bad chunk type"));
        }
        let chunk_len =
            (bytes[1] as usize) | ((bytes[2] as usize) << 8) | ((bytes[3] as usize) << 16);
        let mut b = &bytes[4..];
        if b.len() < chunk_len {
            return Err(Error::Truncated);
        }
        if &b[..INDEX_HEADER.len()] != INDEX_HEADER.as_slice() {
            return Err(Error::BadFormat("bad header"));
        }
        b = &b[INDEX_HEADER.len()..];

        let (v, n) = read_varint(b)?;
        if v < 0 {
            // The uncompressed total is always known in a valid index.
            return Err(Error::Invalid("negative total uncompressed"));
        }
        self.total_uncompressed = Some(v as u64);
        b = &b[n..];

        let (v, n) = read_varint(b)?;
        self.total_compressed = wire_to_opt(v);
        b = &b[n..];

        let (v, n) = read_varint(b)?;
        if v < 0 {
            return Err(Error::Invalid("negative est_block"));
        }
        self.est_block_uncomp = v as u64;
        b = &b[n..];

        let (v, n) = read_varint(b)?;
        if v < 0 || v > MAX_INDEX_ENTRIES as i64 {
            return Err(Error::Invalid("entry count out of range"));
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
            return Err(Error::Truncated);
        }
        let has_uncomp = b[0];
        b = &b[1..];
        if has_uncomp > 1 {
            return Err(Error::BadFormat("invalid has_uncompressed byte"));
        }

        let est = self.est_block_uncomp as i64;
        for idx in 0..entries {
            let mut u_off = 0i64;
            if has_uncomp != 0 {
                let (v, n) = read_varint(b)?;
                u_off = v;
                b = &b[n..];
            }
            if idx > 0 {
                let prev = self.offsets[idx - 1].uncompressed as i64;
                u_off = prev
                    .checked_add(est)
                    .and_then(|s| u_off.checked_add(s))
                    .ok_or(Error::Invalid("uncompressed offset overflow"))?;
                if u_off <= prev {
                    return Err(Error::Invalid("non-monotonic uncompressed offset"));
                }
            }
            if u_off < 0 {
                return Err(Error::Invalid("negative uncompressed offset"));
            }
            self.offsets[idx].uncompressed = u_off as u64;
        }

        let mut c_predict = est / 2;
        for idx in 0..entries {
            let (mut c_off, n) = read_varint(b)?;
            b = &b[n..];
            if idx > 0 {
                let predict_next = c_predict
                    .checked_add(c_off / 2)
                    .ok_or(Error::Invalid("compressed predictor overflow"))?;
                let prev = self.offsets[idx - 1].compressed as i64;
                c_off = prev
                    .checked_add(c_predict)
                    .and_then(|s| c_off.checked_add(s))
                    .ok_or(Error::Invalid("compressed offset overflow"))?;
                if c_off <= prev {
                    return Err(Error::Invalid("non-monotonic compressed offset"));
                }
                c_predict = predict_next;
            }
            if c_off < 0 {
                return Err(Error::Invalid("negative compressed offset"));
            }
            self.offsets[idx].compressed = c_off as u64;
        }

        if b.len() < 4 + INDEX_TRAILER.len() {
            return Err(Error::Truncated);
        }
        b = &b[4..]; // stored size, redundant with chunk header
        if &b[..INDEX_TRAILER.len()] != INDEX_TRAILER.as_slice() {
            return Err(Error::BadFormat("bad trailer"));
        }
        Ok(&b[INDEX_TRAILER.len()..])
    }

    /// Load an index by seeking to the end of `rs`, finding the trailer,
    /// then reading the index chunk.  Matches Go `Index.LoadStream`.
    pub fn load_stream<R: Read + Seek>(&mut self, rs: &mut R) -> Result<()> {
        rs.seek(SeekFrom::End(-10))?; // io::Error -> Error::Io
        let mut tail = [0u8; 10];
        rs.read_exact(&mut tail)?;
        if &tail[4..4 + INDEX_TRAILER.len()] != INDEX_TRAILER.as_slice() {
            return Err(Error::BadFormat("trailer not found"));
        }
        let sz = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
        if sz as usize > MAX_USER_CHUNK_SIZE + SKIPPABLE_FRAME_HEADER {
            return Err(Error::Invalid("size exceeds maximum"));
        }
        rs.seek(SeekFrom::End(-(sz as i64)))?;
        let mut buf = vec![0u8; sz as usize];
        rs.read_exact(&mut buf)?;
        self.load(&buf)?; // crate::Error -> io::Error via From
        Ok(())
    }

    /// Render this index as JSON.  Matches Go `Index.JSON`'s field order
    /// (`encoding/json` defaults with two-space indent); unknown totals
    /// render as `-1` to match the Go output.
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(64 + self.offsets.len() * 64);
        s.push_str("{\n");
        s.push_str(&format!(
            "  \"total_uncompressed\": {},\n",
            opt_to_wire(self.total_uncompressed)
        ));
        s.push_str(&format!(
            "  \"total_compressed\": {},\n",
            opt_to_wire(self.total_compressed)
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
pub fn index_stream<R: Read>(mut r: R) -> Result<Vec<u8>> {
    let mut idx = Index::new();
    let mut comp_pos: u64 = 0;
    let mut uncomp_pos: u64 = 0;

    let mut hdr = [0u8; 4];
    let mut buf = vec![0u8; MAX_USER_CHUNK_SIZE];
    let mut read_header = false;

    loop {
        match read_exact_or_eof(&mut r, &mut hdr)? {
            ReadOutcome::Eof => {
                let mut out = Vec::new();
                idx.append_to(&mut out, Some(uncomp_pos), Some(comp_pos))?;
                return Ok(out);
            }
            ReadOutcome::Ok => {}
        }
        let start_chunk = comp_pos;
        comp_pos += 4;

        let chunk_type = hdr[0];
        if !read_header {
            if chunk_type != CHUNK_TYPE_STREAM_IDENTIFIER && chunk_type != CHUNK_TYPE_EOF {
                return Err(Error::BadFormat("missing stream identifier"));
            }
            read_header = true;
        }
        let chunk_len = (hdr[1] as usize) | ((hdr[2] as usize) << 8) | ((hdr[3] as usize) << 16);
        comp_pos += chunk_len as u64;

        let body = &mut buf[..chunk_len];
        r.read_exact(body).map_err(|_| Error::Truncated)?;

        match chunk_type {
            CHUNK_TYPE_LEGACY_COMPRESSED_DATA
            | CHUNK_TYPE_MINLZ_COMPRESSED_DATA
            | CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC => {
                if chunk_len < CHECKSUM_SIZE {
                    return Err(Error::Invalid("data chunk shorter than CRC"));
                }
                let d_len = crate::block::decoded_len_chunk_body(&body[CHECKSUM_SIZE..])
                    .map_err(|_| Error::Invalid("bad block size header"))?;
                if d_len > MAX_BLOCK_SIZE {
                    return Err(Error::Invalid("block exceeds max size"));
                }
                if idx.est_block_uncomp == 0 {
                    idx.est_block_uncomp = d_len as u64;
                }
                idx.add(start_chunk, uncomp_pos)?;
                uncomp_pos += d_len as u64;
            }
            CHUNK_TYPE_UNCOMPRESSED_DATA => {
                if chunk_len < CHECKSUM_SIZE {
                    return Err(Error::Invalid("uncompressed chunk shorter than CRC"));
                }
                let n2 = chunk_len - CHECKSUM_SIZE;
                if n2 > MAX_BLOCK_SIZE {
                    return Err(Error::Invalid("uncompressed block exceeds max size"));
                }
                if idx.est_block_uncomp == 0 {
                    idx.est_block_uncomp = n2 as u64;
                }
                idx.add(start_chunk, uncomp_pos)?;
                uncomp_pos += n2 as u64;
            }
            CHUNK_TYPE_STREAM_IDENTIFIER => {
                if chunk_len != MAGIC_BODY_LEN {
                    return Err(Error::BadFormat("bad stream identifier length"));
                }
                let head5 = &body[..MAGIC_BODY_MINLZ.len()];
                if head5 != MAGIC_BODY_MINLZ && &body[..MAGIC_BODY_LEN] != MAGIC_BODY_S2 {
                    return Err(Error::BadFormat("unknown stream magic"));
                }
            }
            CHUNK_TYPE_EOF => {}
            t if t <= MAX_NON_SKIPPABLE_CHUNK => {
                return Err(Error::BadFormat("reserved unskippable chunk"));
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

/// Map an optional byte count to its wire encoding (`-1` == unknown).
///
/// The `Some` value must be `<= i64::MAX`; callers guarantee this by
/// bounding offsets in [`Index::add`] and totals in [`Index::append_to`],
/// and `load`/`load_stream` only ever store values decoded from `i64`.
fn opt_to_wire(v: Option<u64>) -> i64 {
    debug_assert!(v.is_none_or(|x| x <= i64::MAX as u64));
    v.map_or(-1, |x| x as i64)
}

/// Inverse of [`opt_to_wire`]: a negative wire value means "unknown".
fn wire_to_opt(v: i64) -> Option<u64> {
    if v < 0 { None } else { Some(v as u64) }
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

fn read_uvarint(b: &[u8]) -> Result<(u64, usize)> {
    let mut x = 0u64;
    let mut s = 0u32;
    for (i, &byte) in b.iter().enumerate() {
        if i == MAX_VARINT_LEN_64 {
            return Err(Error::Invalid("varint overflow"));
        }
        if byte < 0x80 {
            if i == MAX_VARINT_LEN_64 - 1 && byte > 1 {
                return Err(Error::Invalid("varint overflow"));
            }
            return Ok((x | (u64::from(byte) << s), i + 1));
        }
        x |= u64::from(byte & 0x7f) << s;
        s += 7;
    }
    Err(Error::Truncated) // ran out of bytes before the varint terminated
}

fn read_varint(b: &[u8]) -> Result<(i64, usize)> {
    let (u, n) = read_uvarint(b)?;
    let v = ((u >> 1) as i64) ^ -((u & 1) as i64);
    Ok((v, n))
}

#[cfg(test)]
mod tests;
