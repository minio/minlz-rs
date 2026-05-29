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

mod crc;
mod error;
mod format;
mod mt_pool;
mod mt_reader;
mod mt_writer;
mod reader;
mod seek;
mod writer;

pub use error::{Error, Result};
pub use format::{
    CHUNK_TYPE_PADDING, CHUNK_TYPE_STREAM_IDENTIFIER, DEFAULT_BLOCK_SIZE, MAX_BLOCK_SIZE,
    MAX_USER_CHUNK_SIZE, MAX_USER_NON_SKIPPABLE_CHUNK, MAX_USER_SKIPPABLE_CHUNK, MIN_BLOCK_SIZE,
    MIN_USER_NON_SKIPPABLE_CHUNK, MIN_USER_SKIPPABLE_CHUNK,
};
pub use mt_writer::{MtWriter, MtWriterBuilder};
pub use reader::{Reader, ReaderBuilder, UserChunkCb};
pub use seek::ReadSeeker;
pub use writer::{Writer, WriterBuilder};
