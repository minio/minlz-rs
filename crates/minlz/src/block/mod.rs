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

//! MinLZ block codec — public API.
//!
//! Block format reference: [`SPEC.md`] §1–§2 and [`SPEC_SIMD.md`] in the upstream Go
//! repository. [`decode`](fn@decode) and [`append_decoded`] read classic and SIMD blocks.
//!
//! # SIMD blocks
//!
//! SIMD blocks are a block type added in v1.1 of the format. They store literals and operations
//! in separate Huffman-coded streams, laid out so they can be decoded with SIMD. The `*Simd`
//! [`Level`]s write them:
//!
//! ```
//! use minlz::{Level, decode, encode};
//!
//! let src = b"SIMD blocks decode with AVX-512, AVX2 or NEON. ".repeat(1000);
//! let mut block = Vec::new();
//! encode(&mut block, &src, Level::FastestSimd)?;
//! let mut out = Vec::new();
//! decode(&mut out, &block)?;
//! assert_eq!(out, src);
//! # Ok::<(), minlz::Error>(())
//! ```
//!
//! * Each block carries its own Huffman tables, so the SIMD levels pay off for blocks of 64 KiB
//!   and more, such as stream blocks at the default 2 MiB. On small blocks the classic levels
//!   compress faster, and usually decompress faster too.
//! * The delta levels ([`Level::SuperFastSimdDelta`], [`Level::FastestSimdDelta`],
//!   [`Level::BalancedSimdDelta`]) store literals as the difference to the bytes at the last match
//!   offset. That mainly helps binary data, like tables of numbers, but makes text larger and
//!   costs decompression speed. [`Level::SmallestSimd`] decides this itself.
//! * Decoders for format v1.0 cannot read SIMD blocks. Use the classic levels for data that every
//!   MinLZ decoder must read.
//! * The decoder picks its tier once: AVX-512 on x86-64 CPUs with AVX-512 (F, BW, VL, VBMI,
//!   VBMI2), POPCNT, AVX2 and BMI2; AVX2 on CPUs with AVX2 and BMI2; NEON on little-endian
//!   aarch64. Other targets, and Miri, run portable code. The encoders' output does not depend on
//!   the tier, and is the same as the Go encoders'.
//!
//! [`SPEC.md`]: https://github.com/minio/minlz/blob/main/SPEC.md
//! [`SPEC_SIMD.md`]: https://github.com/minio/minlz/blob/main/SPEC_SIMD.md

mod decode;
mod emit;
mod encode_l0;
mod encode_l1;
mod encode_l2;
mod encode_l3;
mod format;
mod hash;
mod load_store;
mod match_len;
pub(crate) mod simd;
mod table_pool;

pub use format::{MAX_BLOCK_SIZE, max_encoded_len};

use crate::Error;

/// Spare bytes the classic encoders get past [`max_encoded_len`]. Like Go's `encodeBlockBest`,
/// L3 checks its size limit before it writes a literal header and a copy, so its last emission
/// can end up to 9 bytes past that length; such a block is discarded as incompressible.
pub(crate) const ENCODE_SLACK: usize = 16;

/// Compression level for [`encode`].
///
/// The discriminants are the Go API's int levels (`LevelFastest = 1`, …), so `level as i32`
/// and [`i32::from`] give Go's level, and [`Level::try_from`] converts back.
///
/// The SIMD levels write SIMD blocks, which need a decoder for MinLZ format specification v1.1
/// or later; streams store them as SIMD chunks (`SPEC.md` 4.5B). SIMD blocks typically compress
/// better than classic ones, and decompress faster on platforms with SIMD when blocks are 64 KiB
/// or more. See [SIMD blocks](crate::block#simd-blocks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
#[non_exhaustive]
pub enum Level {
    /// The fastest compression level. It takes significant shortcuts and usually compresses
    /// much worse; use it only if [`Level::Fastest`] is confirmed to be too slow.
    /// Maps to Go's `LevelSuperFast` (-1) / `encodeFastBlockGo`.
    SuperFast = -1,
    /// Single-pass with a 15-bit (or 13-bit for ≤64 KiB inputs) hash table.
    /// Maps to Go's `LevelFastest` / `encodeBlockGo`.
    Fastest = 1,
    /// Two-table encoder.  Maps to Go's `LevelBalanced` / `encodeBlockBetterGo`.
    Balanced = 2,
    /// Multi-candidate, size-scored encoder.  Maps to Go's
    /// `LevelSmallest` / `encodeBlockBest`.
    Smallest = 3,
    /// The fastest SIMD compression level.
    /// It takes shortcuts that cost compression; use it only if [`Level::FastestSimd`] is
    /// confirmed to be too slow.
    /// Go's `LevelSuperFastSIMD` (4).
    SuperFastSimd = 4,
    /// Fast SIMD compression with a good ratio, the SIMD counterpart of [`Level::Fastest`].
    /// Go's `LevelFastestSIMD` (5).
    FastestSimd = 5,
    /// A balance between compression and speed, the SIMD counterpart of [`Level::Balanced`].
    /// This is targeted to be approximately half the speed of [`Level::FastestSimd`].
    /// Go's `LevelBalancedSIMD` (6).
    BalancedSimd = 6,
    /// Will attempt the best possible compression, the SIMD counterpart of [`Level::Smallest`].
    /// There is no speed target for this level. It always tries both plain and delta literals.
    /// Go's `LevelSmallestSIMD` (7).
    SmallestSimd = 7,
    /// [`Level::SuperFastSimd`] with delta literals, which are mainly useful for binary encoded
    /// data and have a small decode penalty. Go's `LevelSuperFastSIMD | LevelSIMDDelta` (260).
    SuperFastSimdDelta = 4 | 256,
    /// [`Level::FastestSimd`] with delta literals. Go's `LevelFastestSIMD | LevelSIMDDelta` (261).
    FastestSimdDelta = 5 | 256,
    /// [`Level::BalancedSimd`] with delta literals. Go's `LevelBalancedSIMD | LevelSIMDDelta` (262).
    BalancedSimdDelta = 6 | 256,
}

impl Level {
    /// Reports whether the level writes SIMD blocks, which need a decoder for MinLZ format
    /// specification v1.1 or later.
    pub const fn is_simd(self) -> bool {
        match self {
            Level::SuperFast | Level::Fastest | Level::Balanced | Level::Smallest => false,
            Level::SuperFastSimd
            | Level::FastestSimd
            | Level::BalancedSimd
            | Level::SmallestSimd
            | Level::SuperFastSimdDelta
            | Level::FastestSimdDelta
            | Level::BalancedSimdDelta => true,
        }
    }
}

impl TryFrom<i32> for Level {
    type Error = Error;

    /// Convert from Go's int levels (`-1`, `1`-`7`, `260`-`262`); any other value
    /// is [`Error::InvalidLevel`].
    fn try_from(v: i32) -> Result<Self, Error> {
        match v {
            -1 => Ok(Level::SuperFast),
            1 => Ok(Level::Fastest),
            2 => Ok(Level::Balanced),
            3 => Ok(Level::Smallest),
            4 => Ok(Level::SuperFastSimd),
            5 => Ok(Level::FastestSimd),
            6 => Ok(Level::BalancedSimd),
            7 => Ok(Level::SmallestSimd),
            260 => Ok(Level::SuperFastSimdDelta),
            261 => Ok(Level::FastestSimdDelta),
            262 => Ok(Level::BalancedSimdDelta),
            _ => Err(Error::InvalidLevel),
        }
    }
}

impl From<Level> for i32 {
    fn from(l: Level) -> i32 {
        l as i32
    }
}

/// Encode `src` as a MinLZ block, *replacing* the contents of `dst`.
///
/// Equivalent to Go's `Encode(dst, src, level)`.  `dst` is cleared and then
/// filled with the encoded block.
pub fn encode(dst: &mut Vec<u8>, src: &[u8], level: Level) -> Result<(), Error> {
    dst.clear();
    append_encoded(dst, src, level)
}

/// Encode `src` as a MinLZ block, *appending* to `dst`.
///
/// Equivalent to Go's `AppendEncoded(dst, src, level)`.
pub fn append_encoded(dst: &mut Vec<u8>, src: &[u8], level: Level) -> Result<(), Error> {
    let max = max_encoded_len(src.len()).ok_or(Error::TooLarge)?;
    reserve_room(dst, max);
    let start = dst.len();
    // Worst case: a single literal block (header + raw bytes).
    if src.len() < format::MIN_NON_LITERAL_BLOCK_SIZE {
        encode_uncompressed(dst, src);
        return Ok(());
    }

    // Layout: [tag=0][varint(len)][compressed body], or [tag=0][SIMD block].
    dst.push(0);
    // Reserve worst-case body length so the encoder can write via index.
    dst.resize(start + max + ENCODE_SLACK, 0);

    let n = encode_body(&mut dst[start + 1..], src, level);
    if n > 0 {
        dst.truncate(start + 1 + n);
        Ok(())
    } else {
        // Not compressible: fall back to an uncompressed block.
        dst.truncate(start);
        encode_uncompressed(dst, src);
        Ok(())
    }
}

/// Reserves the encoders' room after `dst`. An empty `dst`, such as one made with Go's idiom
/// `with_capacity(max_encoded_len(n))`, grows by the slack instead of doubling; appends keep
/// amortized growth.
fn reserve_room(dst: &mut Vec<u8>, max: usize) {
    if dst.is_empty() {
        dst.reserve_exact(max + ENCODE_SLACK);
    } else {
        dst.reserve(max + ENCODE_SLACK);
    }
}

/// Try to encode `src` at the given [`Level`]; on success, append the block to
/// `dst` and return `true`. Returns `false` (leaving `dst` unchanged) if the
/// input is too small or does not compress to fewer bytes than the input — the
/// give-up path, parity with Go `TryEncode` returning `nil`.
pub fn try_encode(dst: &mut Vec<u8>, src: &[u8], level: Level) -> Result<bool, Error> {
    let max = max_encoded_len(src.len()).ok_or(Error::TooLarge)?;
    if src.len() < format::MIN_NON_LITERAL_BLOCK_SIZE {
        return Ok(false);
    }
    reserve_room(dst, max);
    let start = dst.len();
    dst.push(0);
    dst.resize(start + max + ENCODE_SLACK, 0);

    let n = encode_body(&mut dst[start + 1..], src, level);
    // Go's TryEncode also rejects "compressed >= src.len()" (see encode.go).
    if n > 0 && 1 + n < src.len() {
        dst.truncate(start + 1 + n);
        Ok(true)
    } else {
        dst.truncate(start);
        Ok(false)
    }
}

/// Decode a MinLZ block into `dst`.
///
/// On success `dst` holds the decoded bytes; on error it is empty.
/// Equivalent to Go's `Decode(dst, src)` (without the Snappy/S2 fallback).
///
/// As Go's `Decode` reuses `dst`, a SIMD block decodes over the bytes `dst` holds and zero-fills
/// only the rest of its output first, so clearing a reused `Vec` costs a zero-fill.
pub fn decode(dst: &mut Vec<u8>, src: &[u8]) -> Result<(), Error> {
    decode_at(dst, 0, src)
}

/// Append the decoded form of `src` to `dst`. On error `dst` is unchanged.
///
/// Equivalent to Go's `AppendDecoded(dst, src)`. A SIMD block zero-fills the spare capacity it
/// decodes into.
pub fn append_decoded(dst: &mut Vec<u8>, src: &[u8]) -> Result<(), Error> {
    decode_at(dst, dst.len(), src)
}

/// Decodes `src` into `dst` from `start`, keeping `dst[..start]`, all `dst` keeps on error.
fn decode_at(dst: &mut Vec<u8>, start: usize, src: &[u8]) -> Result<(), Error> {
    let (body, dlen) = match parse_header(src).inspect_err(|_| dst.truncate(start))? {
        Header::Simd { block, size } => return simd::decode_at(dst, start, size, block),
        Header::Literal(lits) => {
            dst.truncate(start);
            dst.extend_from_slice(lits);
            return Ok(());
        }
        Header::Classic { body, dlen } => (body, dlen),
    };
    dst.truncate(start);
    // The decoder uses LZ4-style 16-byte overshoot stores in its hot path;
    // give it `OVERSHOOT_PAD` bytes of writable padding past `dlen`.
    dst.reserve(dlen + decode::OVERSHOOT_PAD);
    // SAFETY: we just reserved enough room; `dst.as_mut_ptr().add(start)`
    // points to at least `dlen + OVERSHOOT_PAD` writable bytes.  The
    // decoder writes `dlen` valid bytes on success; the padding region's
    // contents are undefined and will not be exposed via the returned Vec.
    unsafe {
        let dst_ptr = dst.as_mut_ptr().add(start);
        decode::minlz_decode(dst_ptr, dlen, body)?;
        dst.set_len(start + dlen);
    }
    Ok(())
}

/// Returns the uncompressed length of a MinLZ block.
///
/// Equivalent to Go's `DecodedLen(src)`.
pub fn decoded_len(src: &[u8]) -> Result<usize, Error> {
    Ok(parse_header(src)?.decoded_len())
}

/// Returns `Some(uncompressed_size)` if `src` looks like a MinLZ block (leading
/// 0 byte), or `None` if it does not (e.g. a Snappy/S2 stream, which is out of
/// scope). `Err` only on a malformed MinLZ header.
///
/// Replaces a bare `(bool, usize)`: the size is meaningful exactly when the
/// answer is "yes", which `Option` expresses directly. Full validation still
/// happens inside [`decode`].
pub fn is_minlz(src: &[u8]) -> Result<Option<usize>, Error> {
    if src.is_empty() {
        return Err(Error::Corrupt("empty input"));
    }
    if src[0] != 0 {
        // Snappy/S2 fallback is out of scope.
        return Ok(None);
    }
    Ok(Some(parse_header(src)?.decoded_len()))
}

/// A parsed block header.
enum Header<'a> {
    /// An uncompressed block: its bytes.
    Literal(&'a [u8]),
    /// A classic block: its body and decoded length.
    Classic { body: &'a [u8], dlen: usize },
    /// A SIMD block from its flags byte, and its decoded size.
    Simd { block: &'a [u8], size: usize },
}

impl Header<'_> {
    fn decoded_len(&self) -> usize {
        match *self {
            Header::Literal(lits) => lits.len(),
            Header::Classic { dlen, .. } => dlen,
            Header::Simd { size, .. } => size,
        }
    }
}

fn parse_header(src: &[u8]) -> Result<Header<'_>, Error> {
    if src.is_empty() {
        return Err(Error::Corrupt("empty input"));
    }
    if src.len() == 1 {
        // Single byte: must be 0 (0-byte block).
        if src[0] == 0 {
            return Ok(Header::Literal(&[]));
        }
        return Err(Error::Corrupt("invalid single-byte block"));
    }
    if src[0] != 0 {
        // Snappy/S2 fallback path — refused here.
        return Err(Error::Corrupt("snappy/s2 stream not supported"));
    }
    let after_magic = &src[1..];
    let (v, n) =
        format::get_uvarint(after_magic).ok_or(Error::Corrupt("bad decoded-length varint"))?;
    if v >= 1 << 24 {
        let (size, block) = simd::simd_header(after_magic)?;
        return Ok(Header::Simd { block, size });
    }
    if v > MAX_BLOCK_SIZE as u64 {
        return Err(Error::TooLarge);
    }
    let dlen = v as usize;
    let body = &after_magic[n..];
    // Order matches Go's isMinLZ: body-empty before the v==0 short-circuit.
    if body.is_empty() {
        return Err(Error::Corrupt("empty block body"));
    }
    if dlen == 0 {
        return Ok(Header::Literal(body));
    }
    if dlen < body.len() {
        // A compressed block may not be larger than the decompressed block.
        return Err(Error::Corrupt("compressed larger than decoded size"));
    }
    Ok(Header::Classic { body, dlen })
}

fn encode_uncompressed(dst: &mut Vec<u8>, src: &[u8]) {
    if src.is_empty() {
        dst.push(0);
        return;
    }
    dst.push(0);
    dst.push(0);
    dst.extend_from_slice(src);
}

// ----- Stream-codec helpers ----------------------------------------------
//
// The standalone block format produced by [`encode`] / [`append_encoded`]
// has a leading `0` marker byte that distinguishes it from a Snappy or S2
// block.  When a MinLZ block is embedded in a stream chunk (types 0x02 or
// 0x03), the chunk type itself identifies it, so the leading marker is
// omitted.  The three helpers below speak that "chunk body" form.

/// Encode `src` as a MinLZ stream-chunk body to `dst[start..]` and return its
/// size: `varint(dlen) + body`, no leading `0` marker, for a 0x02 chunk, or a
/// SIMD block, whose header holds its type, for a 0x10 chunk at the SIMD
/// levels.  Returns 0 if compression isn't worthwhile (caller should emit a
/// 0x01 chunk).
///
/// `dst` grows to [`max_encoded_len`] + [`ENCODE_SLACK`] bytes past `start`
/// and keeps its length: the bytes after the body are left from earlier
/// blocks.  The encoders only write their output, so a stream reuses `dst`
/// for every block without zeroing it again (Go's `Writer.encodeBlock` on its
/// `obuf`).
pub(crate) fn encode_chunk_body(
    dst: &mut Vec<u8>,
    start: usize,
    src: &[u8],
    level: Level,
) -> Result<usize, Error> {
    let max = max_encoded_len(src.len()).ok_or(Error::TooLarge)?;
    let end = start + max + ENCODE_SLACK;
    if dst.len() < end {
        dst.resize(end, 0);
    }
    let n = encode_body(&mut dst[start..end], src, level);
    Ok(if n < src.len() { n } else { 0 })
}

/// Writes `src` to `dst` as `varint(len) + body` at the classic levels, or as a SIMD block, and
/// returns its size, or 0 if it doesn't compress into `dst` less its last [`ENCODE_SLACK`] bytes.
fn encode_body(dst: &mut [u8], src: &[u8], level: Level) -> usize {
    let room = dst.len() - ENCODE_SLACK;
    let encode_block = match level {
        Level::SuperFast => encode_l0::encode_block,
        Level::Fastest => encode_l1::encode_block,
        Level::Balanced => encode_l2::encode_block,
        Level::Smallest => encode_l3::encode_block,
        Level::SuperFastSimd
        | Level::FastestSimd
        | Level::BalancedSimd
        | Level::SmallestSimd
        | Level::SuperFastSimdDelta
        | Level::FastestSimdDelta
        | Level::BalancedSimdDelta => {
            return simd::encode_block_simd(&mut dst[..room], src, level.into());
        }
    };
    let h = format::put_uvarint_at(dst, src.len() as u64);
    let n = encode_block(&mut dst[h..], src);
    if n > 0 && h + n <= room { h + n } else { 0 }
}

/// Decode a MinLZ stream-chunk body (`varint(dlen) + body`).  `dst` is
/// appended with the decoded bytes.
pub(crate) fn append_decoded_chunk_body(dst: &mut Vec<u8>, body: &[u8]) -> Result<(), Error> {
    let (dlen, compressed) = split_chunk_body(body)?;
    let start = dst.len();
    dst.reserve(dlen + decode::OVERSHOOT_PAD);
    // SAFETY: same invariants as `decode_at` — we just reserved
    // `dlen + OVERSHOOT_PAD` writable bytes past `start`.
    unsafe {
        let dst_ptr = dst.as_mut_ptr().add(start);
        decode::minlz_decode(dst_ptr, dlen, compressed)?;
        dst.set_len(start + dlen);
    }
    Ok(())
}

/// Return the uncompressed length of a MinLZ stream-chunk body.
pub(crate) fn decoded_len_chunk_body(body: &[u8]) -> Result<usize, Error> {
    split_chunk_body(body).map(|(dlen, _)| dlen)
}

/// Split a chunk body into `dlen` and the compressed bytes.  Unlike a
/// standalone block, a chunk has no `dlen == 0` literal form (incompressible
/// data goes in a 0x01 chunk), and its compressed bytes never exceed `dlen`
/// (SPEC §4.4/§4.5; Go's reader rejects both).
fn split_chunk_body(body: &[u8]) -> Result<(usize, &[u8]), Error> {
    let (v, n) = format::get_uvarint(body).ok_or(Error::Corrupt("bad chunk length varint"))?;
    if v >> 24 != 0 {
        return Err(Error::Corrupt("chunk decoded length exceeds 24 bits"));
    }
    if v > MAX_BLOCK_SIZE as u64 {
        return Err(Error::TooLarge);
    }
    let (dlen, compressed) = (v as usize, &body[n..]);
    if dlen == 0 {
        return Err(Error::Corrupt("zero decoded length in chunk"));
    }
    if dlen < compressed.len() {
        return Err(Error::Corrupt("compressed larger than decoded size"));
    }
    Ok((dlen, compressed))
}

#[cfg(test)]
mod tests;
