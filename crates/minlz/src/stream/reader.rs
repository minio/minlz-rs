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

//! Streaming reader.
//!
//! Mirrors the single-threaded path of Go's `reader.go:Reader.Read`.
//! Multi-threaded decode is available through
//! [`Reader::decode_concurrent`].  Random access via the stream index is
//! provided by [`super::ReadSeeker`].  Block search tables and the
//! Snappy/S2 fallback decoder paths are out of scope.

use std::io::{self, BufRead, Read};

use crate::block;

use super::crc::masked_crc32c;
use super::error::{Error, Result};
use super::format::{
    block_size_from_indicator, get_uvarint, read_chunk_len, read_u32_le, CHECKSUM_SIZE,
    CHUNK_HEADER_SIZE, CHUNK_TYPE_EOF, CHUNK_TYPE_MINLZ_COMPRESSED_DATA,
    CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC, CHUNK_TYPE_STREAM_IDENTIFIER,
    CHUNK_TYPE_UNCOMPRESSED_DATA, MAGIC_BODY, MAGIC_BODY_LEN, MAX_BLOCK_SIZE,
    MAX_NON_SKIPPABLE_CHUNK, MAX_USER_NON_SKIPPABLE_CHUNK, MAX_VARINT_LEN_64,
    MIN_USER_NON_SKIPPABLE_CHUNK, MIN_USER_SKIPPABLE_CHUNK,
};

/// Callback invoked when a user-defined chunk (`0x80..=0xfd`) is encountered.
///
/// Receives the chunk ID and a slice over the chunk's payload.  Returning an
/// error aborts decoding and surfaces as the next [`Reader::read`] error.
///
/// The slice is borrowed from the reader's internal buffer and must not be
/// retained beyond the callback invocation.
pub type UserChunkCb = Box<dyn FnMut(u8, &[u8]) -> io::Result<()>>;

const USER_CB_COUNT: usize = (MAX_USER_NON_SKIPPABLE_CHUNK - MIN_USER_SKIPPABLE_CHUNK + 1) as usize;

/// Builder for [`Reader`] options.  Use the chained methods to configure, then
/// pass an [`io::Read`] to [`ReaderBuilder::build`].
///
/// ```
/// # use std::io::Cursor;
/// # use minlz::stream::ReaderBuilder;
/// let src = Cursor::new(Vec::<u8>::new());
/// let mut reader = ReaderBuilder::new().ignore_crc().build(src);
/// # let _ = &mut reader;
/// ```
#[must_use]
pub struct ReaderBuilder {
    max_block_size: usize,
    ignore_stream_id: bool,
    ignore_crc: bool,
    callbacks: [Option<UserChunkCb>; USER_CB_COUNT],
}

impl Default for ReaderBuilder {
    fn default() -> Self {
        Self {
            max_block_size: MAX_BLOCK_SIZE,
            ignore_stream_id: false,
            ignore_crc: false,
            callbacks: std::array::from_fn(|_| None),
        }
    }
}

impl ReaderBuilder {
    /// Create a new builder with default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap the maximum uncompressed block size the reader will accept.
    ///
    /// Defaults to [`MAX_BLOCK_SIZE`].  Blocks larger than this (as declared
    /// by either the stream identifier or an individual chunk) cause the
    /// reader to return [`Error::TooLarge`].
    ///
    /// # Panics
    ///
    /// Panics if `n` is zero or exceeds [`MAX_BLOCK_SIZE`].
    pub fn max_block_size(mut self, n: usize) -> Self {
        assert!(n > 0 && n <= MAX_BLOCK_SIZE, "max_block_size out of range");
        self.max_block_size = n;
        self
    }

    /// Skip the stream-identifier requirement at the start of the stream.
    ///
    /// Mirrors Go's `ReaderIgnoreStreamIdentifier`.  Also disables EOF
    /// total-size validation.
    pub fn ignore_stream_id(mut self) -> Self {
        self.ignore_stream_id = true;
        self
    }

    /// Skip CRC32C validation on every chunk.  Faster, but invalid streams
    /// may decode to garbage.
    pub fn ignore_crc(mut self) -> Self {
        self.ignore_crc = true;
        self
    }

    /// Register a callback for chunks with ID `id`.  `id` must lie in
    /// `0x80..=0xfd`.  Only one callback per ID; later calls overwrite.
    ///
    /// The callback receives `(id, payload)` and returns an [`io::Result`].
    /// An `Err` aborts decoding and is surfaced via the next read.
    ///
    /// # Panics
    ///
    /// Panics if `id` is outside the user-defined chunk range.
    pub fn user_chunk_callback<F>(mut self, id: u8, cb: F) -> Self
    where
        F: FnMut(u8, &[u8]) -> io::Result<()> + 'static,
    {
        assert!(
            (MIN_USER_SKIPPABLE_CHUNK..=MAX_USER_NON_SKIPPABLE_CHUNK).contains(&id),
            "user-chunk callback id must be in 0x80..=0xfd, got 0x{id:02x}"
        );
        self.callbacks[(id - MIN_USER_SKIPPABLE_CHUNK) as usize] = Some(Box::new(cb));
        self
    }

    /// Consume the builder and wrap `r` in a [`Reader`].
    pub fn build<R: Read>(self, r: R) -> Reader<R> {
        Reader::with_builder(r, self)
    }
}

/// Streaming reader over a MinLZ-compressed [`Read`] source.
pub struct Reader<R> {
    pub(super) r: R,
    /// Decompressed payload of the current block.
    pub(super) decoded: Vec<u8>,
    /// Read position within `decoded`.
    pub(super) decoded_pos: usize,
    /// Raw chunk-body scratch.
    pub(super) buf: Vec<u8>,
    /// Scratch for skippable / user-chunk bodies (kept separate so user-chunk
    /// callbacks can see the raw bytes without disturbing `buf`).
    pub(super) cb_buf: Vec<u8>,
    /// Uncompressed offset at the start of the *current* `decoded` block.
    pub(super) block_start: u64,
    /// Configured ceiling — chunks declaring more uncompressed bytes error.
    pub(super) max_block_user: usize,
    /// Negotiated ceiling — `min(max_block_user, indicator-derived value)`.
    pub(super) max_block: usize,
    pub(super) header_read: bool,
    pub(super) expect_eof: bool,
    pub(super) ignore_stream_id: bool,
    pub(super) ignore_crc: bool,
    /// Sticky error.  Once set, every subsequent read returns it.
    pub(super) err: Option<io::Error>,
    pub(super) callbacks: [Option<UserChunkCb>; USER_CB_COUNT],
}

impl<R: Read> Reader<R> {
    /// Create a reader with default options.
    pub fn new(r: R) -> Self {
        ReaderBuilder::new().build(r)
    }

    fn with_builder(r: R, b: ReaderBuilder) -> Self {
        Self {
            r,
            decoded: Vec::new(),
            decoded_pos: 0,
            buf: Vec::new(),
            cb_buf: Vec::new(),
            block_start: 0,
            max_block_user: b.max_block_size,
            max_block: b.max_block_size,
            header_read: b.ignore_stream_id,
            expect_eof: false,
            ignore_stream_id: b.ignore_stream_id,
            ignore_crc: b.ignore_crc,
            err: None,
            callbacks: b.callbacks,
        }
    }

    /// Reset the reader to consume from `r`, preserving options and buffers.
    pub fn reset(&mut self, r: R) {
        self.r = r;
        self.decoded.clear();
        self.decoded_pos = 0;
        self.buf.clear();
        self.cb_buf.clear();
        self.block_start = 0;
        self.max_block = self.max_block_user;
        self.header_read = self.ignore_stream_id;
        self.expect_eof = false;
        self.err = None;
    }

    /// Uncompressed offset at the start of the block currently being drained.
    pub fn block_start(&self) -> u64 {
        self.block_start
    }

    /// Uncompressed offset of the *next* byte that [`Read::read`] will
    /// return.  Equal to `block_start() + decoded_pos`.  Useful for
    /// pairing with index lookups.  Mirrors Go's
    /// `r.blockStart + int64(r.i)` accounting.
    pub fn current_offset(&self) -> u64 {
        self.block_start + self.decoded_pos as u64
    }

    /// Skip forward `n` decoded bytes without producing them.
    ///
    /// Optimized to skip whole blocks without decoding when the block's
    /// uncompressed length fits entirely inside the remaining skip
    /// distance — matches Go `Reader.Skip` (`reader.go:1034`).  CRC is
    /// **not** checked on fully-skipped blocks (matches Go's documented
    /// "CRC is not checked on skipped blocks").  The partial block at
    /// the tail of the skip range still has its CRC verified unless
    /// [`ReaderBuilder::ignore_crc`] is set.
    ///
    /// Returns [`io::ErrorKind::UnexpectedEof`] if the stream ends
    /// before `n` bytes have been skipped.  Subsequent reads after a
    /// failed `skip` continue to surface the sticky error — matches
    /// Go's behavior.
    pub fn skip(&mut self, mut n: u64) -> io::Result<()> {
        if let Some(e) = &self.err {
            return Err(clone_io_err(e));
        }
        while n > 0 {
            // Step 1: drain in-memory buffer.
            let avail = (self.decoded.len() - self.decoded_pos) as u64;
            if avail >= n {
                self.decoded_pos += n as usize;
                return Ok(());
            }
            n -= avail;
            self.decoded_pos = self.decoded.len();

            // Step 2: read next chunk header.
            let mut hdr = [0u8; CHUNK_HEADER_SIZE];
            match read_full_or_eof(&mut self.r, &mut hdr, !self.expect_eof)? {
                ReadOutcome::Eof => {
                    let e = io::Error::from(io::ErrorKind::UnexpectedEof);
                    let cloned = clone_io_err(&e);
                    self.err = Some(e);
                    return Err(cloned);
                }
                ReadOutcome::Ok => {}
            }
            let chunk_type = hdr[0];
            let chunk_len = read_chunk_len(&hdr[1..]);

            if !self.header_read {
                if chunk_type == CHUNK_TYPE_STREAM_IDENTIFIER {
                    self.header_read = true;
                } else if chunk_type <= MAX_NON_SKIPPABLE_CHUNK && chunk_type != CHUNK_TYPE_EOF {
                    return self.set_err::<()>(Error::Corrupt);
                }
            }

            let res: Result<()> = match chunk_type {
                CHUNK_TYPE_MINLZ_COMPRESSED_DATA | CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC => {
                    self.skip_compressed_or_decode(chunk_type, chunk_len, &mut n)
                }
                CHUNK_TYPE_UNCOMPRESSED_DATA => self.skip_uncompressed_or_decode(chunk_len, &mut n),
                CHUNK_TYPE_EOF => {
                    // Tried to skip past EOF — record sticky error so
                    // subsequent reads surface the same EOF.
                    let r = self.handle_eof_chunk(chunk_len);
                    if let Err(e) = r {
                        return self.set_err::<()>(e);
                    }
                    let e = io::Error::from(io::ErrorKind::UnexpectedEof);
                    let cloned = clone_io_err(&e);
                    self.err = Some(e);
                    return Err(cloned);
                }
                CHUNK_TYPE_STREAM_IDENTIFIER => self.handle_stream_identifier(chunk_len),
                t if t <= MAX_NON_SKIPPABLE_CHUNK => Err(Error::Unsupported),
                _ => self.handle_skippable(chunk_type, chunk_len),
            };
            if let Err(e) = res {
                return self.set_err::<()>(e);
            }
        }
        Ok(())
    }

    /// Compressed data chunk dispatch for [`Reader::skip`].  If the
    /// chunk's uncompressed length is ≤ the remaining skip distance,
    /// the body is read and discarded (no decode, no CRC).  Otherwise
    /// the chunk is decoded into `self.decoded` and `decoded_pos` is
    /// set so the *next* [`Read`] returns bytes from `block_start + n`.
    fn skip_compressed_or_decode(
        &mut self,
        chunk_type: u8,
        chunk_len: usize,
        n: &mut u64,
    ) -> Result<()> {
        if chunk_len < CHECKSUM_SIZE {
            return Err(Error::Corrupt);
        }
        self.ensure_buf(chunk_len)?;
        read_exact_or_corrupt(&mut self.r, &mut self.buf[..chunk_len])?;
        let checksum = read_u32_le(&self.buf[..CHECKSUM_SIZE]);
        let body = &self.buf[CHECKSUM_SIZE..chunk_len];
        let dlen = block::decoded_len_chunk_body(body)?;
        if dlen > self.max_block {
            return Err(Error::TooLarge);
        }
        if (dlen as u64) > *n {
            // Need decode for the partial block.
            self.block_start += self.decoded.len() as u64;
            self.decoded.clear();
            block::append_decoded_chunk_body(&mut self.decoded, body)?;
            if !self.ignore_crc {
                let to_crc: &[u8] = if chunk_type == CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC {
                    strip_dlen_varint(body)?
                } else {
                    &self.decoded[..]
                };
                if masked_crc32c(to_crc) != checksum {
                    return Err(Error::Crc);
                }
            }
            self.decoded_pos = *n as usize;
            *n = 0;
        } else {
            // Whole block fits in the skip — drop without decoding.
            self.block_start += self.decoded.len() as u64 + dlen as u64;
            self.decoded.clear();
            self.decoded_pos = 0;
            *n -= dlen as u64;
        }
        Ok(())
    }

    /// Uncompressed data chunk dispatch for [`Reader::skip`].  Same
    /// shape as the compressed variant but with a 1:1 wire length.
    fn skip_uncompressed_or_decode(&mut self, chunk_len: usize, n: &mut u64) -> Result<()> {
        if chunk_len < CHECKSUM_SIZE {
            return Err(Error::Corrupt);
        }
        let n2 = chunk_len - CHECKSUM_SIZE;
        if n2 > self.max_block {
            return Err(Error::TooLarge);
        }
        if (n2 as u64) > *n {
            // Partial — read CRC + body into `decoded` and verify.
            self.block_start += self.decoded.len() as u64;
            self.ensure_buf(CHECKSUM_SIZE)?;
            read_exact_or_corrupt(&mut self.r, &mut self.buf[..CHECKSUM_SIZE])?;
            let checksum = read_u32_le(&self.buf[..CHECKSUM_SIZE]);
            self.decoded.clear();
            self.decoded.resize(n2, 0);
            read_exact_or_corrupt(&mut self.r, &mut self.decoded[..])?;
            if !self.ignore_crc && masked_crc32c(&self.decoded) != checksum {
                return Err(Error::Crc);
            }
            self.decoded_pos = *n as usize;
            *n = 0;
        } else {
            // Whole chunk is inside the skip — discard CRC + body without
            // verifying.
            self.block_start += self.decoded.len() as u64 + n2 as u64;
            self.decoded.clear();
            self.decoded_pos = 0;
            self.skip_n(chunk_len)?;
            *n -= n2 as u64;
        }
        Ok(())
    }

    /// Drain the rest of the stream into `w` using `threads` worker threads
    /// for parallel block decoding.  Returns `(bytes_written, w)`.
    ///
    /// Must not be mixed with [`Read::read`] / [`std::io::BufRead::fill_buf`]
    /// on the same reader — calling this after consuming bytes via the
    /// single-threaded path returns [`io::ErrorKind::InvalidInput`].
    ///
    /// `threads >= 1` is required; `1` still spins up the MT pipeline (one
    /// worker) — use [`Read::read`] for the lowest-overhead single-threaded
    /// path.
    ///
    /// `W: Send + 'static` is required because the writer thread spawned
    /// inside outlives any caller-stack borrow.  Borrowed sinks
    /// (`&mut Vec<u8>`) are not supported; wrap them in an owned adapter
    /// (e.g. pass `Vec::new()` and copy back from the returned `W`).
    pub fn decode_concurrent<W>(&mut self, w: W, threads: usize) -> io::Result<(u64, W)>
    where
        W: io::Write + Send + 'static,
    {
        if self.decoded_pos < self.decoded.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "decode_concurrent called after Read",
            ));
        }
        if let Some(err) = &self.err {
            return Err(clone_io_err(err));
        }
        super::mt_reader::decode_concurrent_into(self, w, threads)
    }
}

impl<R: Read> Reader<R> {
    /// Make sure `self.decoded` has unread bytes available, or report that
    /// the stream has ended gracefully.  Returns `Ok(true)` if data is now
    /// queued in `decoded`, `Ok(false)` on clean EOF, `Err` on failure.
    fn fill_decoded(&mut self) -> io::Result<bool> {
        loop {
            if self.decoded_pos < self.decoded.len() {
                return Ok(true);
            }
            let mut hdr = [0u8; CHUNK_HEADER_SIZE];
            match read_full_or_eof(&mut self.r, &mut hdr, !self.expect_eof)? {
                ReadOutcome::Eof => return Ok(false),
                ReadOutcome::Ok => {}
            }
            let chunk_type = hdr[0];
            let chunk_len = read_chunk_len(&hdr[1..]);

            if !self.header_read {
                if chunk_type == CHUNK_TYPE_STREAM_IDENTIFIER {
                    self.header_read = true;
                } else if chunk_type <= MAX_NON_SKIPPABLE_CHUNK && chunk_type != CHUNK_TYPE_EOF {
                    return self.set_err::<()>(Error::Corrupt).map(|_| false);
                }
            }

            let res: Result<()> = match chunk_type {
                CHUNK_TYPE_MINLZ_COMPRESSED_DATA | CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC => {
                    self.handle_compressed_chunk(chunk_type, chunk_len)
                }
                CHUNK_TYPE_UNCOMPRESSED_DATA => self.handle_uncompressed_chunk(chunk_len),
                CHUNK_TYPE_EOF => self.handle_eof_chunk(chunk_len),
                CHUNK_TYPE_STREAM_IDENTIFIER => self.handle_stream_identifier(chunk_len),
                t if t <= MAX_NON_SKIPPABLE_CHUNK => Err(Error::Unsupported),
                _ => self.handle_skippable(chunk_type, chunk_len),
            };
            if let Err(e) = res {
                return self.set_err::<()>(e).map(|_| false);
            }
            // Else loop: either decoded was populated (top-of-loop returns
            // Ok(true)) or this was a metadata chunk and we read another.
        }
    }
}

impl<R: Read> Read for Reader<R> {
    fn read(&mut self, p: &mut [u8]) -> io::Result<usize> {
        if let Some(err) = &self.err {
            return Err(clone_io_err(err));
        }
        if p.is_empty() {
            return Ok(0);
        }
        if !self.fill_decoded()? {
            return Ok(0);
        }
        let n = (self.decoded.len() - self.decoded_pos).min(p.len());
        p[..n].copy_from_slice(&self.decoded[self.decoded_pos..self.decoded_pos + n]);
        self.decoded_pos += n;
        Ok(n)
    }
}

impl<R: Read> BufRead for Reader<R> {
    /// Returns a borrowed slice over the next chunk of decoded bytes.  The
    /// slice can be longer than any per-call buffer the user might pass to
    /// [`Read::read`], so `io::copy` can stream a whole block (up to
    /// `block_size`) without an intermediate memcpy.
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if let Some(err) = &self.err {
            return Err(clone_io_err(err));
        }
        if !self.fill_decoded()? {
            return Ok(&[]);
        }
        Ok(&self.decoded[self.decoded_pos..])
    }

    fn consume(&mut self, amt: usize) {
        let available = self.decoded.len() - self.decoded_pos;
        let amt = amt.min(available);
        self.decoded_pos += amt;
    }
}

// -------------------- chunk handlers --------------------

impl<R: Read> Reader<R> {
    fn handle_compressed_chunk(&mut self, chunk_type: u8, chunk_len: usize) -> Result<()> {
        if chunk_len < CHECKSUM_SIZE {
            return Err(Error::Corrupt);
        }
        self.block_start += self.decoded.len() as u64;
        self.ensure_buf(chunk_len)?;
        read_exact_or_corrupt(&mut self.r, &mut self.buf[..chunk_len])?;
        let checksum = read_u32_le(&self.buf[..CHECKSUM_SIZE]);
        let body = &self.buf[CHECKSUM_SIZE..chunk_len];

        // Body wire format is `[varint(dlen), compressed]` — the leading
        // `0` marker is only present in the standalone block format.
        let dlen = block::decoded_len_chunk_body(body)?;
        if dlen > self.max_block {
            return Err(Error::TooLarge);
        }
        self.decoded.clear();
        block::append_decoded_chunk_body(&mut self.decoded, body)?;
        debug_assert_eq!(self.decoded.len(), dlen);

        if !self.ignore_crc {
            let to_crc: &[u8] = if chunk_type == CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC {
                // CRC for 0x03 chunks is over the compressed body, *after*
                // stripping the dlen varint.  Mirror `Go reader.go:343`.
                strip_dlen_varint(body)?
            } else {
                &self.decoded[..]
            };
            if masked_crc32c(to_crc) != checksum {
                return Err(Error::Crc);
            }
        }
        self.decoded_pos = 0;
        Ok(())
    }

    fn handle_uncompressed_chunk(&mut self, chunk_len: usize) -> Result<()> {
        if chunk_len < CHECKSUM_SIZE {
            return Err(Error::Corrupt);
        }
        self.block_start += self.decoded.len() as u64;
        // Read the 4-byte checksum first into `buf` so we can stream the
        // payload directly into `decoded` (Go does the same).
        let n = chunk_len - CHECKSUM_SIZE;
        if n > self.max_block {
            return Err(Error::TooLarge);
        }
        self.ensure_buf(CHECKSUM_SIZE)?;
        read_exact_or_corrupt(&mut self.r, &mut self.buf[..CHECKSUM_SIZE])?;
        let checksum = read_u32_le(&self.buf[..CHECKSUM_SIZE]);

        self.decoded.clear();
        self.decoded.resize(n, 0);
        read_exact_or_corrupt(&mut self.r, &mut self.decoded[..])?;

        if !self.ignore_crc && masked_crc32c(&self.decoded) != checksum {
            return Err(Error::Crc);
        }
        self.decoded_pos = 0;
        Ok(())
    }

    fn handle_eof_chunk(&mut self, chunk_len: usize) -> Result<()> {
        if chunk_len > MAX_VARINT_LEN_64 {
            return Err(Error::Corrupt);
        }
        if chunk_len != 0 {
            let mut tmp = [0u8; MAX_VARINT_LEN_64];
            read_exact_or_corrupt(&mut self.r, &mut tmp[..chunk_len])?;
            if !self.ignore_stream_id {
                let (want_size, n) = get_uvarint(&tmp[..chunk_len]).ok_or(Error::Corrupt)?;
                if n != chunk_len {
                    return Err(Error::Corrupt);
                }
                let actual = self.block_start + self.decoded.len() as u64;
                if want_size != actual {
                    return Err(Error::Corrupt);
                }
            }
        }
        self.expect_eof = false;
        self.header_read = false;
        Ok(())
    }

    fn handle_stream_identifier(&mut self, chunk_len: usize) -> Result<()> {
        if chunk_len != MAGIC_BODY_LEN {
            return Err(Error::Corrupt);
        }
        let mut tmp = [0u8; MAGIC_BODY_LEN];
        read_exact_or_corrupt(&mut self.r, &mut tmp)?;
        if &tmp[..MAGIC_BODY.len()] != MAGIC_BODY {
            // Snappy/S2 fallback is out of scope.
            return Err(Error::Unsupported);
        }
        let indicator = tmp[MAGIC_BODY_LEN - 1];
        if indicator & 0xc0 != 0 {
            return Err(Error::Corrupt);
        }
        let new_max = block_size_from_indicator(indicator).ok_or(Error::Corrupt)?;
        if new_max > self.max_block_user {
            return Err(Error::TooLarge);
        }
        self.max_block = new_max;
        self.block_start = 0;
        self.decoded.clear();
        self.decoded_pos = 0;
        self.expect_eof = true;
        Ok(())
    }

    fn handle_skippable(&mut self, chunk_type: u8, chunk_len: usize) -> Result<()> {
        let id = chunk_type;
        let in_user_range = (MIN_USER_SKIPPABLE_CHUNK..=MAX_USER_NON_SKIPPABLE_CHUNK).contains(&id);
        if in_user_range {
            let cb_idx = (id - MIN_USER_SKIPPABLE_CHUNK) as usize;
            let mut maybe_cb = self.callbacks[cb_idx].take();
            if let Some(ref mut cb) = maybe_cb {
                self.cb_buf.clear();
                self.cb_buf.resize(chunk_len, 0);
                let read_res = read_exact_or_corrupt(&mut self.r, &mut self.cb_buf[..])
                    .map_err(io::Error::from);
                let invoke_res = read_res.and_then(|()| cb(id, &self.cb_buf));
                self.callbacks[cb_idx] = maybe_cb;
                invoke_res?;
                return Ok(());
            }
            if (MIN_USER_NON_SKIPPABLE_CHUNK..=MAX_USER_NON_SKIPPABLE_CHUNK).contains(&id) {
                return Err(Error::Corrupt);
            }
        }
        // Silent skip (padding 0xfe, reserved skippables 0x40..=0x7f, or
        // user-skippable without a registered callback).
        self.skip_n(chunk_len)
    }

    fn skip_n(&mut self, mut n: usize) -> Result<()> {
        let mut tmp = [0u8; 4096];
        while n > 0 {
            let take = n.min(tmp.len());
            read_exact_or_corrupt(&mut self.r, &mut tmp[..take])?;
            n -= take;
        }
        Ok(())
    }

    fn ensure_buf(&mut self, n: usize) -> Result<()> {
        let cap = self.max_buf_size();
        if n > cap {
            return Err(Error::Corrupt);
        }
        if self.buf.len() < n {
            self.buf.resize(n, 0);
        }
        Ok(())
    }

    fn max_buf_size(&self) -> usize {
        // MaxEncodedLen(maxBlock) + checksum.  block::max_encoded_len returns
        // None only when n > MAX_BLOCK_SIZE; capped above.
        crate::block::max_encoded_len(self.max_block).unwrap_or(MAX_BLOCK_SIZE) + CHECKSUM_SIZE
    }

    fn set_err<T>(&mut self, e: Error) -> io::Result<T> {
        let io_err: io::Error = e.into();
        let cloned = clone_io_err(&io_err);
        self.err = Some(io_err);
        Err(cloned)
    }
}

// -------------------- helpers --------------------

enum ReadOutcome {
    Ok,
    Eof,
}

/// `read_exact` with explicit handling of zero-byte EOF when permitted.
///
/// When `allow_eof` is true and the underlying reader reports EOF *before*
/// any bytes have been read, returns `Ok(Eof)`.  Otherwise mirrors Go's
/// `io.ReadFull`: short reads become [`Error::Corrupt`].
fn read_full_or_eof<R: Read>(
    r: &mut R,
    buf: &mut [u8],
    allow_eof: bool,
) -> io::Result<ReadOutcome> {
    let mut total = 0;
    while total < buf.len() {
        match r.read(&mut buf[total..]) {
            Ok(0) => {
                if total == 0 && allow_eof {
                    return Ok(ReadOutcome::Eof);
                }
                let e: Error = Error::Corrupt;
                return Err(e.into());
            }
            Ok(n) => total += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(ReadOutcome::Ok)
}

/// Strip the dlen varint from a chunk body so that what remains is the raw
/// compressed bytes (used by the 0x03 chunk type's CRC).
fn strip_dlen_varint(body: &[u8]) -> Result<&[u8]> {
    let (_, n) = get_uvarint(body).ok_or(Error::Corrupt)?;
    Ok(&body[n..])
}

/// `io::Error` is not `Clone`; this duplicates the kind/message so we can
/// keep one copy as the sticky error and return another to the caller.
fn clone_io_err(e: &io::Error) -> io::Error {
    io::Error::new(e.kind(), e.to_string())
}

/// `Read::read_exact` mapped to stream errors: `UnexpectedEof` becomes
/// [`Error::Corrupt`], matching Go's `Reader.readFull`.
fn read_exact_or_corrupt<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<()> {
    match r.read_exact(buf) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Err(Error::Corrupt),
        Err(e) => Err(Error::Io(e)),
    }
}

#[cfg(test)]
mod tests;
