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

//! Random-access reader: `ReadSeeker<R>` for `R: Read + Seek`.
//!
//! Mirrors Go `reader.go:ReadSeeker`.  Uses the [`Index`](crate::index::Index)
//! to translate uncompressed offsets into compressed file offsets, then
//! drives the existing [`Reader`] over the seeked position.

use std::io::{self, BufRead, Read, Seek, SeekFrom};

use crate::index::{restore_index_headers, Index, CHUNK_TYPE_INDEX, LEGACY_INDEX_CHUNK};

use super::Reader;

/// Header expected at offset 4 of a fully-framed `0x40` / `0x99` index
/// chunk.  Used by [`looks_like_full_index_chunk`] to disambiguate
/// between the full chunk and the bare form from
/// [`crate::index::remove_index_headers`].
const INDEX_HEADER: &[u8; 6] = b"s2idx\x00";

/// Random-access wrapper around a [`Reader`].
///
/// Construct via [`ReadSeeker::new`].  The underlying input must implement
/// both [`Read`] and [`Seek`].  An index is required — either passed in as
/// bytes or loaded from the end of the stream on construction.
///
/// `ReadSeeker` implements [`Read`] (forwarded to the inner reader) and
/// [`Seek`] (interpreted as uncompressed offsets, never compressed).  See
/// [`ReadSeeker::read_at`] for the [`std::io::Read`] + offset hybrid.
pub struct ReadSeeker<R: Read + Seek> {
    reader: Reader<R>,
    index: Index,
}

impl<R: Read + Seek> ReadSeeker<R> {
    /// Wrap `reader` with seek support.
    ///
    /// `index_bytes` accepts three shapes:
    ///
    /// * The full `0x40` chunk (or legacy `0x99` chunk) as emitted by
    ///   [`crate::stream::Writer::close_index`] /
    ///   [`crate::stream::Writer::finish`] with `append_index()` /
    ///   [`crate::index::Index::append_to`].
    /// * The compact "bare" form returned by
    ///   [`crate::index::remove_index_headers`] — this is the same chunk
    ///   minus the 4-byte chunk header, the `"s2idx\x00"` header, the
    ///   stored 4-byte size, and the `"\x00xdi2s"` trailer (~20 bytes
    ///   smaller).  Useful for sidecar storage.
    /// * An empty slice — the index is loaded from the end of the
    ///   underlying source via [`Index::load_stream`].  The source's
    ///   seek position before the call is preserved.
    ///
    /// The shape is auto-detected by checking the chunk-type byte and
    /// the `s2idx\x00` header at offset 4.
    pub fn new(mut reader: Reader<R>, index_bytes: &[u8]) -> io::Result<Self> {
        let mut index = Index::default();
        if !index_bytes.is_empty() {
            if looks_like_full_index_chunk(index_bytes) {
                index.load(index_bytes)?;
            } else {
                // Bare form — wrap it back into a full chunk and parse.
                let restored = restore_index_headers(index_bytes);
                if restored.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "minlz: empty index payload",
                    ));
                }
                index.load(&restored)?;
            }
        } else {
            let pos = reader.r.stream_position()?;
            let res = index.load_stream(&mut reader.r);
            // Always restore the underlying position before propagating.
            let restore = reader.r.seek(SeekFrom::Start(pos));
            res?;
            restore?;
        }
        Ok(Self { reader, index })
    }

    /// Borrow the loaded index (e.g. for tooling / JSON dump).
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Seek by uncompressed offset.  Returns the absolute offset reached.
    ///
    /// * [`SeekFrom::Start`] — absolute uncompressed offset.
    /// * [`SeekFrom::Current`] — relative to the next byte that
    ///   [`Read::read`] would return.
    /// * [`SeekFrom::End`] — relative to `index.total_uncompressed`;
    ///   requires the latter to be known (not `-1`).
    ///
    /// Fast path: when the target lands inside the currently-decoded
    /// block, no underlying I/O is performed — the read position is
    /// adjusted in place.
    pub fn seek_uncompressed(&mut self, pos: SeekFrom) -> io::Result<u64> {
        if let Some(e) = self.reader.err.as_ref() {
            // Reset on EOF only — caller can re-seek past EOF and retry.
            if e.kind() != io::ErrorKind::UnexpectedEof {
                return Err(clone_io_err(e));
            }
            self.reader.err = None;
        }

        let abs_off = match pos {
            SeekFrom::Start(o) => o as i64,
            SeekFrom::Current(d) => {
                let cur = self.reader.current_offset() as i64;
                cur.checked_add(d)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "seek overflow"))?
            }
            SeekFrom::End(d) => {
                if self.index.total_uncompressed < 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "seek from end requires known total_uncompressed",
                    ));
                }
                self.index
                    .total_uncompressed
                    .checked_add(d)
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "seek overflow"))?
            }
        };
        if abs_off < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "seek before start of file",
            ));
        }

        // Ensure the reader has consumed the stream identifier so that
        // subsequent partial-block reads pass the header check.  See Go
        // `reader.go:1403-1413`.  Empty-buffer `Read::read` short-circuits
        // in the Rust impl, so use `BufRead::fill_buf` to actually drive
        // the chunk dispatch.
        if !self.reader.header_read {
            self.reader.r.seek(SeekFrom::Start(0))?;
            let _ = self.reader.fill_buf()?;
        }

        // Fast path: in current buffer.
        let block_start = self.reader.block_start as i64;
        let block_end = block_start + self.reader.decoded.len() as i64;
        if abs_off >= block_start && abs_off < block_end {
            self.reader.decoded_pos = (abs_off - block_start) as usize;
            return Ok(abs_off as u64);
        }

        // Slow path: index lookup.
        let (c_off, u_off) = self.index.find(abs_off)?;

        self.reader.r.seek(SeekFrom::Start(c_off as u64))?;
        // Hard-reset block state: clearing the decoded buffer makes the
        // next data chunk's `block_start += decoded.len()` a no-op, so
        // block_start at the new chunk equals u_off as desired.
        self.reader.decoded.clear();
        self.reader.decoded_pos = 0;
        self.reader.block_start = u_off as u64;

        if u_off < abs_off {
            self.reader.skip((abs_off - u_off) as u64)?;
        }
        debug_assert!(u_off <= abs_off);

        Ok(abs_off as u64)
    }

    /// Read up to `p.len()` bytes starting at uncompressed `offset`.
    ///
    /// Loops over internal `Read` calls until `p` is full, end-of-stream
    /// is reached, or an error occurs:
    ///
    /// * **Full fill:** returns `Ok(p.len())`.
    /// * **End of stream before `p` is full:** returns `Ok(n)` with
    ///   `n < p.len()`.  The first `n` bytes of `p` contain valid data;
    ///   the rest is unspecified.  Callers MUST treat `n < p.len()` as
    ///   EOF — matches `std::os::unix::fs::FileExt::read_at`'s
    ///   short-read contract.
    /// * **I/O / decode error:** returns `Err(e)`.  Any bytes already
    ///   written to `p` are lost.  Use [`Self::read_exact_at`] if you
    ///   need a fill-or-error contract that mirrors
    ///   [`Read::read_exact`].
    ///
    /// Side effect: after the call returns, the underlying [`Reader`]
    /// is positioned at `offset + n` — subsequent [`Read::read`] calls
    /// continue from there.  Mirrors Go's documented behavior, but
    /// Rust uses `&mut self` instead of Go's runtime mutex because the
    /// borrow checker provides the same serialization at compile time.
    /// Callers needing parallel random access open multiple
    /// `ReadSeeker`s over independent file handles.
    pub fn read_at(&mut self, p: &mut [u8], offset: u64) -> io::Result<usize> {
        self.seek_uncompressed(SeekFrom::Start(offset))?;
        let mut n = 0;
        while n < p.len() {
            match self.reader.read(&mut p[n..]) {
                Ok(0) => return Ok(n), // EOF — partial fill is the signal.
                Ok(k) => n += k,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(n)
    }

    /// Like [`Self::read_at`] but errors with
    /// [`io::ErrorKind::UnexpectedEof`] if the stream ends before `p`
    /// is filled.  Matches [`Read::read_exact`]'s contract — on error,
    /// the bytes already written to `p` are unspecified.
    pub fn read_exact_at(&mut self, p: &mut [u8], offset: u64) -> io::Result<()> {
        let n = self.read_at(p, offset)?;
        if n < p.len() {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
        }
        Ok(())
    }
}

impl<R: Read + Seek> Read for ReadSeeker<R> {
    fn read(&mut self, p: &mut [u8]) -> io::Result<usize> {
        self.reader.read(p)
    }
}

impl<R: Read + Seek> Seek for ReadSeeker<R> {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.seek_uncompressed(pos)
    }
}

fn clone_io_err(e: &io::Error) -> io::Error {
    io::Error::new(e.kind(), e.to_string())
}

/// Returns `true` if `b` carries the chunk-header byte (0x40 or 0x99)
/// and the canonical `s2idx\x00` header at offset 4 — i.e. it's the
/// fully-framed index chunk rather than the bare form.
fn looks_like_full_index_chunk(b: &[u8]) -> bool {
    if b.len() < 4 + INDEX_HEADER.len() {
        return false;
    }
    let chunk_type = b[0];
    if chunk_type != CHUNK_TYPE_INDEX && chunk_type != LEGACY_INDEX_CHUNK {
        return false;
    }
    &b[4..4 + INDEX_HEADER.len()] == INDEX_HEADER.as_slice()
}
