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

//! Streaming writer (single-threaded).
//!
//! Index generation and appending are handled here via
//! [`WriterBuilder::generate_index`] / [`WriterBuilder::append_index`]
//! and [`Writer::close_index`].  For the multi-threaded variant see
//! [`super::MtWriter`].

use std::io::{self, Write};

use crate::block;
use crate::index::Index;

use super::crc::masked_crc32c;
use super::error::Error;
use super::format::{
    make_stream_header, put_uvarint, CHECKSUM_SIZE, CHUNK_HEADER_SIZE, CHUNK_TYPE_EOF,
    CHUNK_TYPE_MINLZ_COMPRESSED_DATA, CHUNK_TYPE_PADDING, CHUNK_TYPE_UNCOMPRESSED_DATA,
    DEFAULT_BLOCK_SIZE, MAX_BLOCK_SIZE, MAX_USER_CHUNK_SIZE, MAX_USER_NON_SKIPPABLE_CHUNK,
    MAX_VARINT_LEN_64, MIN_BLOCK_SIZE, MIN_USER_SKIPPABLE_CHUNK,
};

/// Builder for [`Writer`] options.
///
/// ```
/// # use minlz::stream::WriterBuilder;
/// # use minlz::Level;
/// let mut buf = Vec::<u8>::new();
/// let mut writer = WriterBuilder::new().level(Level::Smallest).build(&mut buf);
/// # let _ = &mut writer;
/// ```
#[must_use]
pub struct WriterBuilder {
    block_size: usize,
    level: block::Level,
    uncompressed: bool,
    padding: u32,
    flush_on_write: bool,
    generate_index: bool,
    append_index: bool,
}

impl Default for WriterBuilder {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            level: block::Level::Balanced,
            uncompressed: false,
            padding: 0,
            flush_on_write: false,
            generate_index: true,
            append_index: false,
        }
    }
}

impl WriterBuilder {
    /// New builder with defaults matching Go's `NewWriter`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum uncompressed block size.  Must lie in
    /// `[MIN_BLOCK_SIZE, MAX_BLOCK_SIZE]`.
    ///
    /// # Panics
    /// Panics if `n` is outside the supported range.
    pub fn block_size(mut self, n: usize) -> Self {
        assert!(
            (MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE).contains(&n),
            "block_size must be in [{MIN_BLOCK_SIZE}, {MAX_BLOCK_SIZE}], got {n}"
        );
        self.block_size = n;
        self
    }

    /// Set the compression level.
    pub fn level(mut self, level: block::Level) -> Self {
        self.level = level;
        self.uncompressed = false;
        self
    }

    /// Disable compression entirely; all blocks are emitted as 0x01
    /// uncompressed-data chunks.  Mirrors Go's `WriterUncompressed`.
    pub fn uncompressed(mut self) -> Self {
        self.uncompressed = true;
        self
    }

    /// Pad the total written size to a multiple of `n` bytes by emitting
    /// padding chunks (`0xfe`) at [`Writer::finish`] time.  `n == 0` or
    /// `n == 1` disables padding.
    ///
    /// # Panics
    /// Panics if `n > MAX_BLOCK_SIZE`.
    pub fn padding(mut self, n: u32) -> Self {
        assert!(
            n as usize <= MAX_BLOCK_SIZE,
            "padding must be ≤ {MAX_BLOCK_SIZE}, got {n}"
        );
        self.padding = if n <= 1 { 0 } else { n };
        self
    }

    /// Emit a block on every call to [`Write::write`].  Mirrors Go's
    /// `WriterFlushOnWrite`.  Less efficient — block size depends on the
    /// write size.
    pub fn flush_on_write(mut self) -> Self {
        self.flush_on_write = true;
        self
    }

    /// Toggle in-memory index generation.  Default is `true` so that
    /// [`Writer::close_index`] can return an index after writing.  Set
    /// to `false` for streaming output where no index is ever needed
    /// (matches Go `WriterCreateIndex`).
    ///
    /// # Panics
    /// Panics if called with `false` after [`Self::append_index`] was
    /// requested — the two flags cannot disagree.
    pub fn generate_index(mut self, b: bool) -> Self {
        if !b && self.append_index {
            panic!("generate_index(false) conflicts with append_index()");
        }
        self.generate_index = b;
        self
    }

    /// Append the index chunk to the end of the stream when finishing.
    /// Requires index generation (which is the default).  Mirrors Go
    /// `WriterAddIndex(true)`.
    pub fn append_index(mut self) -> Self {
        if !self.generate_index {
            panic!("append_index requires generate_index(true)");
        }
        self.append_index = true;
        self
    }

    /// Consume the builder and wrap `w` in a [`Writer`].
    pub fn build<W: Write>(self, w: W) -> Writer<W> {
        Writer::with_builder(w, self)
    }
}

/// Streaming MinLZ writer.
///
/// Implements [`Write`] for arbitrary uncompressed input.  Call
/// [`Writer::finish`] to flush remaining data, emit the EOF chunk, and
/// recover the underlying sink.  Without `finish`, the stream is truncated
/// and the reader will fail with [`Error::Corrupt`].
pub struct Writer<W> {
    w: W,
    /// Buffered uncompressed input (filled up to `block_size`, then flushed).
    ibuf: Vec<u8>,
    /// Scratch for the encoded block body + chunk header + checksum.
    obuf: Vec<u8>,
    block_size: usize,
    level: block::Level,
    uncompressed: bool,
    padding: u32,
    flush_on_write: bool,
    append_index: bool,
    /// `Some` when `generate_index` is enabled.  Receives one entry per
    /// block emitted (subject to the in-place spacing inside `Index::add`).
    index: Option<Index>,
    wrote_header: bool,
    uncomp_written: u64,
    comp_written: u64,
    err: Option<io::Error>,
}

impl<W: Write> Writer<W> {
    /// Wrap `w` with default options.
    pub fn new(w: W) -> Self {
        WriterBuilder::new().build(w)
    }

    fn with_builder(w: W, b: WriterBuilder) -> Self {
        let ibuf = Vec::with_capacity(b.block_size);
        let mut index = if b.generate_index {
            let mut idx = Index::default();
            idx.reset(b.block_size);
            Some(idx)
        } else {
            None
        };
        // Newly created indices start with total_*=-1 (Go's reset);
        // `index.add` later promotes them, but for callers who close
        // immediately we want non-negative totals.
        if let Some(i) = index.as_mut() {
            i.total_uncompressed = 0;
            i.total_compressed = 0;
        }
        Self {
            w,
            ibuf,
            obuf: Vec::new(),
            block_size: b.block_size,
            level: b.level,
            uncompressed: b.uncompressed,
            padding: b.padding,
            flush_on_write: b.flush_on_write,
            append_index: b.append_index,
            index,
            wrote_header: false,
            uncomp_written: 0,
            comp_written: 0,
            err: None,
        }
    }

    /// Reset the writer to wrap `w`, preserving options and buffers.
    pub fn reset(&mut self, w: W) {
        self.w = w;
        self.ibuf.clear();
        self.obuf.clear();
        self.wrote_header = false;
        self.uncomp_written = 0;
        self.comp_written = 0;
        self.err = None;
        if let Some(idx) = self.index.as_mut() {
            idx.reset(self.block_size);
            idx.total_uncompressed = 0;
            idx.total_compressed = 0;
        }
    }

    /// Total uncompressed bytes accepted by [`Write::write`] and total
    /// compressed bytes written to the sink so far.
    pub fn written(&self) -> (u64, u64) {
        (self.uncomp_written, self.comp_written)
    }

    /// Add a user chunk (`id` in `0x80..=0xfd`) to the stream.  Any pending
    /// buffered input is flushed first so that user chunks appear in the
    /// stream at the position the caller invoked the method.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] if `id` is out of range or
    /// `data.len() > MAX_USER_CHUNK_SIZE`.  Other errors come from the
    /// underlying sink.
    pub fn add_user_chunk(&mut self, id: u8, data: &[u8]) -> io::Result<()> {
        if !(MIN_USER_SKIPPABLE_CHUNK..=MAX_USER_NON_SKIPPABLE_CHUNK).contains(&id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("user-chunk id 0x{id:02x} out of range"),
            ));
        }
        if data.len() > MAX_USER_CHUNK_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "user-chunk exceeds maximum size",
            ));
        }
        if let Some(e) = &self.err {
            return Err(clone_io_err(e));
        }
        self.flush_ibuf()?;
        self.ensure_header()?;
        let hdr = [
            id,
            data.len() as u8,
            (data.len() >> 8) as u8,
            (data.len() >> 16) as u8,
        ];
        self.write_all_tracked(&hdr)?;
        self.write_all_tracked(data)?;
        Ok(())
    }

    /// Flush remaining buffered data, emit the EOF marker (and any
    /// configured padding), and return the underlying sink.
    ///
    /// When [`WriterBuilder::append_index`] is set, the index chunk is
    /// emitted *after* padding (matches Go `writer.go:closeIndex`).
    pub fn finish(mut self) -> io::Result<W> {
        if let Some(e) = self.err.take() {
            return Err(e);
        }
        let _ = self.close_inner(self.append_index)?;
        Ok(self.w)
    }

    /// Like [`Self::finish`] but also returns the encoded index bytes.
    /// Requires `generate_index` (the default) to have been left enabled.
    /// Inner writer is dropped.
    pub fn close_index(mut self) -> io::Result<Vec<u8>> {
        if let Some(e) = self.err.take() {
            return Err(e);
        }
        if self.index.is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "minlz: close_index called but generate_index is false",
            ));
        }
        self.close_inner(true)
    }

    /// Shared close path used by both [`Self::finish`] and
    /// [`Self::close_index`].  `build_index` controls whether the
    /// in-memory index is materialized into bytes (and, when
    /// `append_index = true`, written to the stream after padding).
    fn close_inner(&mut self, build_index: bool) -> io::Result<Vec<u8>> {
        self.flush_ibuf()?;
        self.ensure_header()?;
        self.emit_eof()?;

        let mut index_bytes = Vec::new();
        if build_index {
            // SAFETY: `build_index` implies `generate_index` (set true
            // for finish() only via `append_index = true` and for
            // close_index() only after the `is_none()` guard above).
            let idx = self
                .index
                .as_mut()
                .expect("build_index implies generate_index");
            let comp_total = if self.padding <= 1 {
                self.comp_written as i64
            } else {
                -1
            };
            idx.append_to(&mut index_bytes, self.uncomp_written as i64, comp_total);
            if self.append_index {
                // Count the index toward written bytes so the padding
                // size calc below sees the right total.
                self.comp_written += index_bytes.len() as u64;
            }
        }

        self.emit_padding()?;

        if self.append_index && !index_bytes.is_empty() {
            // The +len() bump happened above; write the bytes here
            // without re-counting them.
            self.w.write_all(&index_bytes).map_err(|e| {
                let cloned = clone_io_err(&e);
                self.err = Some(e);
                cloned
            })?;
        }

        self.w.flush()?;
        Ok(index_bytes)
    }

    fn ensure_header(&mut self) -> io::Result<()> {
        if !self.wrote_header {
            let h = make_stream_header(self.block_size);
            self.write_all_tracked(&h)?;
            self.wrote_header = true;
        }
        Ok(())
    }

    /// Flush the contents of `ibuf` as one or more chunks.
    fn flush_ibuf(&mut self) -> io::Result<()> {
        if self.ibuf.is_empty() {
            return Ok(());
        }
        // Take ownership so we can borrow self mutably for write_block.
        let buf = std::mem::take(&mut self.ibuf);
        let res = self.write_blocks(&buf);
        // Restore the buffer (cleared) for reuse.
        let mut buf = buf;
        buf.clear();
        self.ibuf = buf;
        res
    }

    /// Emit `data` as one or more block chunks, chunking at `block_size`.
    fn write_blocks(&mut self, mut data: &[u8]) -> io::Result<()> {
        self.ensure_header()?;
        while !data.is_empty() {
            let n = data.len().min(self.block_size);
            let (block, rest) = data.split_at(n);
            self.emit_block(block)?;
            data = rest;
            self.uncomp_written += n as u64;
        }
        Ok(())
    }

    /// Emit a single block as either an 0x02 compressed-data chunk or a
    /// 0x01 uncompressed-data chunk, depending on whether compression wins.
    fn emit_block(&mut self, block: &[u8]) -> io::Result<()> {
        if let Some(idx) = self.index.as_mut() {
            idx.add(self.comp_written as i64, self.uncomp_written as i64)?;
        }
        let crc = masked_crc32c(block);
        // Try compression unless explicitly disabled.
        let compressed_ok = if self.uncompressed {
            false
        } else {
            self.obuf.clear();
            block::append_encoded_chunk_body(&mut self.obuf, block, self.level)
                .map_err(|e| io::Error::from(Error::Block(e)))?
        };
        if compressed_ok {
            let chunk_len = CHECKSUM_SIZE + self.obuf.len();
            let mut hdr = [0u8; CHUNK_HEADER_SIZE + CHECKSUM_SIZE];
            hdr[0] = CHUNK_TYPE_MINLZ_COMPRESSED_DATA;
            hdr[1] = chunk_len as u8;
            hdr[2] = (chunk_len >> 8) as u8;
            hdr[3] = (chunk_len >> 16) as u8;
            hdr[4..8].copy_from_slice(&crc.to_le_bytes());
            self.write_all_tracked(&hdr)?;
            // Borrow obuf separately so write_all_tracked can use self.w.
            let body = std::mem::take(&mut self.obuf);
            let res = self.write_all_tracked(&body);
            self.obuf = body;
            self.obuf.clear();
            res
        } else {
            let chunk_len = CHECKSUM_SIZE + block.len();
            let mut hdr = [0u8; CHUNK_HEADER_SIZE + CHECKSUM_SIZE];
            hdr[0] = CHUNK_TYPE_UNCOMPRESSED_DATA;
            hdr[1] = chunk_len as u8;
            hdr[2] = (chunk_len >> 8) as u8;
            hdr[3] = (chunk_len >> 16) as u8;
            hdr[4..8].copy_from_slice(&crc.to_le_bytes());
            self.write_all_tracked(&hdr)?;
            self.write_all_tracked(block)
        }
    }

    fn emit_eof(&mut self) -> io::Result<()> {
        let mut tmp = [0u8; CHUNK_HEADER_SIZE + MAX_VARINT_LEN_64];
        tmp[0] = CHUNK_TYPE_EOF;
        let n = put_uvarint(&mut tmp[CHUNK_HEADER_SIZE..], self.uncomp_written);
        tmp[1] = n as u8;
        tmp[2] = 0;
        tmp[3] = 0;
        self.write_all_tracked(&tmp[..CHUNK_HEADER_SIZE + n])
    }

    fn emit_padding(&mut self) -> io::Result<()> {
        if self.padding < 2 {
            return Ok(());
        }
        let multiple = self.padding as u64;
        let leftover = self.comp_written % multiple;
        if leftover == 0 {
            return Ok(());
        }
        let mut to_add = multiple - leftover;
        // Padding chunk has a 4-byte header; pad-target must accommodate it.
        while to_add < CHUNK_HEADER_SIZE as u64 {
            to_add += multiple;
        }
        if to_add as usize > MAX_BLOCK_SIZE + CHUNK_HEADER_SIZE {
            // Should never happen — `padding ≤ MAX_BLOCK_SIZE` by the
            // builder check.
            return Ok(());
        }
        let body_len = (to_add as usize) - CHUNK_HEADER_SIZE;
        let mut hdr = [0u8; CHUNK_HEADER_SIZE];
        hdr[0] = CHUNK_TYPE_PADDING;
        hdr[1] = body_len as u8;
        hdr[2] = (body_len >> 8) as u8;
        hdr[3] = (body_len >> 16) as u8;
        self.write_all_tracked(&hdr)?;
        // Stream zero bytes in fixed-size chunks rather than allocating a
        // body_len-sized Vec (body_len can be up to MAX_BLOCK_SIZE).
        let zeros = [0u8; 4096];
        let mut remaining = body_len;
        while remaining > 0 {
            let n = remaining.min(zeros.len());
            self.write_all_tracked(&zeros[..n])?;
            remaining -= n;
        }
        Ok(())
    }

    fn write_all_tracked(&mut self, p: &[u8]) -> io::Result<()> {
        match self.w.write_all(p) {
            Ok(()) => {
                self.comp_written += p.len() as u64;
                Ok(())
            }
            Err(e) => {
                let cloned = clone_io_err(&e);
                self.err = Some(e);
                Err(cloned)
            }
        }
    }
}

impl<W: Write> Write for Writer<W> {
    fn write(&mut self, p: &[u8]) -> io::Result<usize> {
        if let Some(e) = &self.err {
            return Err(clone_io_err(e));
        }
        if p.is_empty() {
            return Ok(0);
        }
        if self.flush_on_write {
            // Treat the whole call as a single block (chunked at block_size
            // if larger).
            self.write_blocks(p)?;
            return Ok(p.len());
        }
        let mut written = 0;
        let mut p = p;
        // Drain into ibuf until it's full, then emit; if ibuf is empty and
        // the input is at least block_size, bypass the copy.
        while !p.is_empty() {
            let free = self.block_size - self.ibuf.len();
            if self.ibuf.is_empty() && p.len() >= self.block_size {
                let n = p.len() - (p.len() % self.block_size);
                let n = n.min(p.len());
                // Emit whole blocks directly from `p`.
                self.write_blocks(&p[..n])?;
                written += n;
                p = &p[n..];
                continue;
            }
            let take = p.len().min(free);
            self.ibuf.extend_from_slice(&p[..take]);
            written += take;
            p = &p[take..];
            if self.ibuf.len() == self.block_size {
                self.flush_ibuf()?;
            }
        }
        Ok(written)
    }

    /// Flush buffered input as full-or-partial blocks.  Does **not** emit
    /// the EOF marker — call [`Writer::finish`] for that.
    fn flush(&mut self) -> io::Result<()> {
        if let Some(e) = &self.err {
            return Err(clone_io_err(e));
        }
        self.flush_ibuf()?;
        self.w.flush()
    }
}

fn clone_io_err(e: &io::Error) -> io::Error {
    io::Error::new(e.kind(), e.to_string())
}

#[cfg(test)]
mod tests;
