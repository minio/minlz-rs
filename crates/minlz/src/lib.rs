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
//! Implements the MinLZ v1.1 specification: the v1.0 block and stream
//! formats plus SIMD blocks.  See [`SPEC.md`] §1–§4 and [`SPEC_SIMD.md`]
//! in the upstream Go repository for the wire format.
//!
//! # What's here
//!
//! * Block codec — encode at three classic levels ([`Level::Fastest`],
//!   [`Level::Balanced`], [`Level::Smallest`]) or as SIMD blocks at the
//!   `*Simd` levels, and a safe decoder for both.  See [`mod@block`], and
//!   [SIMD blocks](block#simd-blocks) for when the SIMD levels pay off and
//!   who can read them.
//! * SIMD decoding with AVX-512 or AVX2 on x86-64 and NEON on aarch64,
//!   picked at runtime by CPU feature detection, with portable code
//!   everywhere else.  The SIMD encoders write the same bytes as Go's.
//! * Streaming codec — [`stream::Reader`] / [`stream::Writer`] over any
//!   [`Read`](std::io::Read) / [`Write`](std::io::Write), with a
//!   multi-threaded variant ([`stream::MtWriter`],
//!   [`stream::Reader::decode_concurrent`]).  Each block is checked with
//!   CRC32C, or with XXH3-64 for SIMD blocks (see
//!   [SIMD chunks](stream#simd-chunks)).
//! * Stream index — [`Index`] for random access; built incrementally by
//!   the writer (`append_index()`) or recovered from the tail of an
//!   existing stream.  [`stream::ReadSeeker`] is the seekable reader.
//! * Free functions [`index_stream`], [`remove_index_headers`],
//!   [`restore_index_headers`] for tooling.
//!
//! # What's *not* here
//!
//! Dictionaries, Go's classic `LevelSuperFast`, LZ4 conversion, and the
//! Snappy/S2 fallback decoder paths are explicitly out of scope.  Block
//! search tables (`SPEC.md` §4.13) are not implemented in this crate.
//!
//! [`SPEC.md`]: https://github.com/minio/minlz/blob/main/SPEC.md
//! [`SPEC_SIMD.md`]: https://github.com/minio/minlz/blob/main/SPEC_SIMD.md
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
#[doc(hidden)]
pub mod cpu;
mod error;
#[cfg(test)]
mod fuzz_seeds;
#[cfg(test)]
mod guard;
pub mod index;
#[doc(hidden)]
pub mod pivco;
#[cfg(any(test, feature = "reference"))]
#[doc(hidden)]
pub mod reference;
pub mod stream;
#[cfg(test)]
mod unzip;
#[doc(hidden)]
pub mod xxh3;

pub use block::{
    Level, MAX_BLOCK_SIZE, append_decoded, append_encoded, decode, decoded_len, encode, is_minlz,
    max_encoded_len, try_encode,
};
pub use error::Error;
pub use index::{
    CHUNK_TYPE_INDEX, Index, OffsetPair, index_stream, remove_index_headers, restore_index_headers,
};

/// A specialized [`Result`] type for MinLZ operations.
pub type Result<T> = core::result::Result<T, Error>;
