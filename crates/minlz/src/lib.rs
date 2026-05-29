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

//! MinLZ — Rust port of the Go reference (`github.com/minio/minlz`).
//!
//! Implements the MinLZ v1.0 specification.  See `SPEC.md` §1–§4 in the
//! upstream Go repository for the wire format.
//!
//! # What's here
//!
//! * Block codec — encode at three compression levels
//!   ([`Level::Fastest`], [`Level::Balanced`], [`Level::Smallest`]) and a
//!   safe decoder.  See [`mod@block`].
//! * Streaming codec — [`stream::Reader`] / [`stream::Writer`] over any
//!   [`Read`](std::io::Read) / [`Write`](std::io::Write), with a
//!   multi-threaded variant ([`stream::MtWriter`],
//!   [`stream::Reader::decode_concurrent`]).
//! * Stream index — [`Index`] for random access; built incrementally by
//!   the writer (`append_index()`) or recovered from the tail of an
//!   existing stream.  [`stream::ReadSeeker`] is the seekable reader.
//! * Free functions [`index_stream`], [`remove_index_headers`],
//!   [`restore_index_headers`] for tooling.
//!
//! # What's *not* here
//!
//! Dictionaries, `LevelSuperFast`, LZ4 conversion, and the Snappy/S2
//! fallback decoder paths are explicitly out of scope.  Block search
//! tables (`SPEC.md` §4.13) are not implemented in this crate.
//!
//! # Endianness
//!
//! MinLZ's wire format is little-endian.  The tuned path targets LE
//! hosts (x86_64, aarch64, …) where unaligned 64-bit loads compile to a
//! single `MOVQ` / `LDR`.  Big-endian targets compile and pass tests,
//! but pay a per-load byte-swap and are not in the perf-target matrix.

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(missing_docs)]

pub mod block;
mod error;
pub mod index;
pub mod stream;

pub use block::{
    append_decoded, append_encoded, decode, decoded_len, encode, is_minlz, max_encoded_len,
    try_encode, Level, MAX_BLOCK_SIZE,
};
pub use error::Error;
pub use index::{
    index_stream, remove_index_headers, restore_index_headers, Index, OffsetPair, CHUNK_TYPE_INDEX,
};

/// A specialized [`Result`] type for MinLZ operations.
pub type Result<T> = core::result::Result<T, Error>;
