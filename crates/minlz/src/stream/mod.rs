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

//! MinLZ stream codec — `Reader<R>` / `Writer<W>` over the framing
//! format from `SPEC.md` §4.
//!
//! * [`Reader`] / [`Writer`] — single-threaded encode / decode over any
//!   [`Read`](std::io::Read) / [`Write`](std::io::Write).
//! * [`MtWriter`] / [`Reader::decode_concurrent`] — worker-pool variants
//!   that compress / decompress blocks in parallel.
//! * [`ReadSeeker`] — wraps `Reader<R: Read + Seek>` with random access
//!   driven by the stream index (see [`crate::index`]).
//!
//! Block search tables (`SPEC.md` §4.13) are not implemented in this
//! crate.
//!
//! # SIMD chunks
//!
//! At the SIMD [`Level`](crate::Level)s ([SIMD blocks](crate::block#simd-blocks)) the writers
//! store SIMD blocks as chunk type 0x10 (`SPEC.md` §4.5B). Instead of a CRC32C, the chunk is
//! checked with an XXH3-64 hash of the uncompressed data; [`ReaderBuilder::ignore_crc`] skips
//! both checks. A stream written at a SIMD level sets a flag in its stream identifier, so
//! decoders without SIMD support reject the stream at once, instead of failing at the first SIMD
//! chunk. [`Reader`] rejects SIMD chunks in streams without the flag. Blocks that don't compress
//! are stored uncompressed, as at the other levels.

mod crc;
mod error;
mod format;
mod mt_pool;
mod mt_reader;
mod mt_writer;
mod reader;
mod seek;
mod writer;

use std::num::NonZeroUsize;

/// `n` workers, or the host's available parallelism for `0`.
pub(crate) fn worker_count(n: usize) -> usize {
    if n > 0 {
        return n;
    }
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

#[cfg(test)]
#[test]
fn worker_count_zero_is_available_parallelism() {
    let available = std::thread::available_parallelism().map_or(1, NonZeroUsize::get);
    assert_eq!(worker_count(0), available);
    assert_eq!(worker_count(3), 3);
}

pub use error::Error;
pub use format::{
    CHUNK_TYPE_PADDING, CHUNK_TYPE_STREAM_IDENTIFIER, DEFAULT_BLOCK_SIZE, MAX_BLOCK_SIZE,
    MAX_USER_CHUNK_SIZE, MAX_USER_NON_SKIPPABLE_CHUNK, MAX_USER_SKIPPABLE_CHUNK, MIN_BLOCK_SIZE,
    MIN_USER_NON_SKIPPABLE_CHUNK, MIN_USER_SKIPPABLE_CHUNK,
};
pub use mt_reader::ConcurrentDecode;
pub use mt_writer::{MtWriter, MtWriterBuilder};
pub(crate) use reader::simd_chunk_header;
pub use reader::{Reader, ReaderBuilder, UserChunkCb};
pub use seek::ReadSeeker;
pub use writer::{Writer, WriterBuilder, Written};
