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

//! Codes blocks of up to 65535 bytes with PivCo ("pivoted coding") Huffman.
//!
//! Instead of a sequential stream of codes, a block stores one bitmap per internal Huffman
//! tree node, holding the left/right decision of every symbol that passes through it.
//! Subtrees whose leaves all share one depth are packed as D-bit codes instead. Decoding
//! merges the children of each node bottom-up, which vectorizes well.
//!
//! This module is a port of Go's `internal/pivco`, itself a port of the PH block codec of
//! PivCo-Huffman by Marcin Zukowski, github.com/MarcinZukowski/pivco-huffman at commit
//! 07d9efc69eef1dfbb0c683d98259319c80135d79 (2026-09-17), which is licensed under the
//! Apache License, Version 2.0, as is this module. Only what MinLZ uses is ported: plain
//! Huffman lengths (upstream effort 0), the FlatVertical layout and raw bitmaps (no ANS),
//! without the pivcohuf container. Go's `Shaper` (effort 2) is not ported.
//!
//! The block format is defined in `SPEC_PIVCO.md` in the upstream Go repository. It is the
//! upstream PH block without its symbol count, its marker prefix and the marker byte before
//! each flat region; otherwise the regions are byte for byte those of upstream.

pub mod kernel;

mod decode;
mod encode;
mod huffman;
mod table;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod testutil;

pub use decode::Decoder;
pub use encode::Encoder;
pub use huffman::build_lengths;
pub use table::Table;

use core::fmt;

/// The largest number of symbols in one block; K_right headers are 16 bits.
pub const MAX_BLOCK_SIZE: usize = 65535;
/// The longest Huffman code length.
pub const MAX_CODE_LEN: usize = 11;

/// A pivco error. Errors are static: returning one never allocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// Malformed input; the detail names the check that failed.
    Corrupt(&'static str),
    /// A block of 0 or more than [`MAX_BLOCK_SIZE`] symbols.
    BlockSize,
    /// The input has a symbol without a code.
    SymbolNotInTable,
    /// A [`Table`] was not built successfully.
    InvalidTable,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Corrupt(s) => write!(f, "pivco: corrupt input: {s}"),
            Error::BlockSize => f.write_str("pivco: block size must be 1..65535"),
            Error::SymbolNotInTable => f.write_str("pivco: symbol not in table"),
            Error::InvalidTable => f.write_str("pivco: table not built"),
        }
    }
}

impl std::error::Error for Error {}

const ERR_CODE_LEN: Error = Error::Corrupt("code length above 11");
const ERR_NO_SYMBOLS: Error = Error::Corrupt("no code lengths");
const ERR_KRAFT: Error = Error::Corrupt("code lengths are not a complete prefix code");
const ERR_TRUNCATED: Error = Error::Corrupt("truncated block");
const ERR_SINGLE: Error = Error::Corrupt("set bits in a single-symbol bitmap");
const ERR_SPLIT: Error = Error::Corrupt("right count exceeds node count");
const ERR_PADDING: Error = Error::Corrupt("non-zero padding bits");
const ERR_MERGE: Error = Error::Corrupt("bitmap does not match its right count");
const ERR_GEOMETRY: Error = Error::Corrupt("buffer placement out of range");
const ERR_TRAILING: Error = Error::Corrupt("trailing bytes after block");
