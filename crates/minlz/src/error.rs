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

//! Error type for the block codec.

use core::fmt;

/// Errors returned by the MinLZ block codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Input bytes are not a valid MinLZ block (truncated, malformed, or an
    /// out-of-bounds offset/length). The payload is a short, stable reason
    /// for diagnostics — do not match on its exact text. Equivalent to Go's
    /// `ErrCorrupt`.
    Corrupt(&'static str),
    /// Declared or input length exceeds [`crate::MAX_BLOCK_SIZE`].
    ///
    /// Equivalent to Go's `ErrTooLarge`.
    TooLarge,
    /// Compression level outside the supported range ([`crate::Level`]).
    ///
    /// Equivalent to Go's `ErrInvalidLevel`.
    InvalidLevel,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Corrupt(reason) => write!(f, "minlz: corrupt input: {reason}"),
            Error::TooLarge => f.write_str("minlz: decoded block is too large"),
            Error::InvalidLevel => f.write_str("minlz: invalid compression level"),
        }
    }
}

impl std::error::Error for Error {}

impl From<Error> for std::io::Error {
    /// Wrap a block/index error as an [`std::io::Error`] (kind `InvalidData`)
    /// for the I/O-returning stream and index entry points.
    fn from(e: Error) -> std::io::Error {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    }
}
