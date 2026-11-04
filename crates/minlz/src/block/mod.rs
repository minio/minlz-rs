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
//! Block format reference: SPEC.md §1–§2 in the upstream Go repository.

mod decode;
mod emit;
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

/// Compression level for [`encode`].
///
/// The levels map to the Go API's int levels (`LevelFastest = 1`, …) with [`i32::from`] and
/// [`Level::try_from`].
///
/// The SIMD levels write SIMD blocks (`SPEC_SIMD.md`), which need a decoder for MinLZ format
/// specification v1.1 or later. Streams store them as SIMD chunks (`SPEC.md` 4.5B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Single-pass with a 15-bit (or 13-bit for ≤64 KiB inputs) hash table.
    /// Maps to Go's `LevelFastest` / `encodeBlockGo`.
    Fastest,
    /// Two-table encoder.  Maps to Go's `LevelBalanced` / `encodeBlockBetterGo`.
    Balanced,
    /// Multi-candidate, size-scored encoder.  Maps to Go's
    /// `LevelSmallest` / `encodeBlockBest`.
    Smallest,
    /// The fastest SIMD compression level.
    /// This will take significant shortcuts and usually provide much worse compression.
    /// Use only if [`Level::FastestSimd`] is confirmed to be too slow.
    /// Go's `LevelSuperFastSIMD` (4).
    SuperFastSimd,
    /// The fastest compression level.
    /// SIMD blocks typically compress better, and decompress faster
    /// on platforms where SIMD is available when blocks are 64KB or more.
    /// Go's `LevelFastestSIMD` (5).
    FastestSimd,
    /// The balanced compression level.
    /// This is targeted to be approximately half the speed of [`Level::FastestSimd`].
    /// SIMD blocks typically compress better, and decompress faster
    /// on platforms where SIMD is available when blocks are 64KB or more.
    /// Go's `LevelBalancedSIMD` (6).
    BalancedSimd,
    /// Will attempt the best possible compression.
    /// There is no speed target for this level. It always tries delta literals.
    /// SIMD blocks typically compress better, and decompress faster
    /// on platforms where SIMD is available when blocks are 64KB or more.
    /// Go's `LevelSmallestSIMD` (7).
    SmallestSimd,
    /// [`Level::SuperFastSimd`] with delta literals, which are mainly useful for binary encoded
    /// data and have a small decode penalty. Go's `LevelSuperFastSIMD | LevelSIMDDelta` (260).
    SuperFastSimdDelta,
    /// [`Level::FastestSimd`] with delta literals. Go's `LevelFastestSIMD | LevelSIMDDelta` (261).
    FastestSimdDelta,
    /// [`Level::BalancedSimd`] with delta literals. Go's `LevelBalancedSIMD | LevelSIMDDelta` (262).
    BalancedSimdDelta,
}

impl TryFrom<i32> for Level {
    type Error = Error;

    /// Convert from Go's int levels (`1`-`7`, `260`-`262`); any other value
    /// is [`Error::InvalidLevel`].
    fn try_from(v: i32) -> Result<Self, Error> {
        match v {
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
        match l {
            Level::Fastest => 1,
            Level::Balanced => 2,
            Level::Smallest => 3,
            Level::SuperFastSimd => 4,
            Level::FastestSimd => 5,
            Level::BalancedSimd => 6,
            Level::SmallestSimd => 7,
            Level::SuperFastSimdDelta => 4 | 256,
            Level::FastestSimdDelta => 5 | 256,
            Level::BalancedSimdDelta => 6 | 256,
        }
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
    dst.reserve(max);
    let start = dst.len();
    // Worst case: a single literal block (header + raw bytes).
    if src.len() < format::MIN_NON_LITERAL_BLOCK_SIZE {
        encode_uncompressed(dst, src);
        return Ok(());
    }

    // Layout: [tag=0][varint(len)][compressed body].
    dst.push(0);
    format::put_uvarint(dst, src.len() as u64);
    let mut body_start = dst.len();
    // Reserve worst-case body length so the encoder can write via index.
    dst.resize(start + max, 0);

    let n = match level {
        Level::Fastest => encode_l1::encode_block(&mut dst[body_start..], src),
        Level::Balanced => encode_l2::encode_block(&mut dst[body_start..], src),
        Level::Smallest => encode_l3::encode_block(&mut dst[body_start..], src),
        _ => {
            // A SIMD block has its own header after the indicator byte.
            body_start = start + 1;
            simd::encode_block_simd(&mut dst[body_start..], src, level.into())
        }
    };

    if n > 0 {
        dst.truncate(body_start + n);
        Ok(())
    } else {
        // Not compressible: fall back to an uncompressed block.
        dst.truncate(start);
        encode_uncompressed(dst, src);
        Ok(())
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
    dst.reserve(max);
    let start = dst.len();
    dst.push(0);
    format::put_uvarint(dst, src.len() as u64);
    let mut body_start = dst.len();
    dst.resize(start + max, 0);

    let n = match level {
        Level::Fastest => encode_l1::encode_block(&mut dst[body_start..], src),
        Level::Balanced => encode_l2::encode_block(&mut dst[body_start..], src),
        Level::Smallest => encode_l3::encode_block(&mut dst[body_start..], src),
        _ => {
            body_start = start + 1;
            simd::encode_block_simd(&mut dst[body_start..], src, level.into())
        }
    };

    // Go's TryEncode also rejects "compressed >= src.len()" (see encode.go).
    if n > 0 && (body_start - start) + n < src.len() {
        dst.truncate(body_start + n);
        Ok(true)
    } else {
        dst.truncate(start);
        Ok(false)
    }
}

/// Decode a MinLZ block into `dst`.
///
/// `dst` is cleared.  On success the decoded bytes are appended.
/// Equivalent to Go's `Decode(dst, src)` (without the Snappy/S2 fallback).
pub fn decode(dst: &mut Vec<u8>, src: &[u8]) -> Result<(), Error> {
    dst.clear();
    append_decoded(dst, src)
}

/// Append the decoded form of `src` to `dst`.
///
/// Equivalent to Go's `AppendDecoded(dst, src)`.
pub fn append_decoded(dst: &mut Vec<u8>, src: &[u8]) -> Result<(), Error> {
    let (literals, simd, body, dlen) = parse_header(src)?;
    if literals {
        dst.extend_from_slice(body);
        return Ok(());
    }
    if simd {
        return simd::append_decoded(dst, dlen, body);
    }
    let start = dst.len();
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
    let (_, _, _, dlen) = parse_header(src)?;
    Ok(dlen)
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
    let (_, _, _, dlen) = parse_header(src)?;
    Ok(Some(dlen))
}

/// Parse the block header.  Returns `(is_literal_only, is_simd, body_slice, dlen)`.
/// SIMD blocks are returned from their flags byte.
fn parse_header(src: &[u8]) -> Result<(bool, bool, &[u8], usize), Error> {
    if src.is_empty() {
        return Err(Error::Corrupt("empty input"));
    }
    if src.len() == 1 {
        // Single byte: must be 0 (0-byte block).
        if src[0] == 0 {
            return Ok((true, false, &src[1..], 0));
        }
        return Err(Error::Corrupt("invalid single-byte block"));
    }
    if src[0] != 0 {
        // Snappy/S2 fallback path — refused here.
        return Err(Error::Corrupt("snappy/s2 stream not supported"));
    }
    let after_magic = &src[1..];
    if let Some((v, _)) = format::get_uvarint(after_magic)
        && v >= 1 << 24
    {
        let (size, block) = simd::simd_header(after_magic)?;
        return Ok((false, true, block, size));
    }
    let (v, n) =
        format::get_uvarint(after_magic).ok_or(Error::Corrupt("bad decoded-length varint"))?;
    // Match Go decodedLen: anything that wouldn't fit in uint32 is corrupt …
    if v > 0xffff_ffff {
        return Err(Error::Corrupt("decoded length exceeds u32"));
    }
    // … and anything between uint32 and MaxBlockSize is "too large".
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
        return Ok((true, false, body, body.len()));
    }
    if dlen < body.len() {
        // A compressed block may not be larger than the decompressed block.
        return Err(Error::Corrupt("compressed larger than decoded size"));
    }
    Ok((false, false, body, dlen))
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
/// `dst` grows to [`max_encoded_len`] bytes past `start` and keeps its length:
/// the bytes after the body are left from earlier blocks.  The encoders only
/// write their output, so a stream reuses `dst` for every block without
/// zeroing it again (Go's `Writer.encodeBlock` on its `obuf`).
pub(crate) fn encode_chunk_body(
    dst: &mut Vec<u8>,
    start: usize,
    src: &[u8],
    level: Level,
) -> Result<usize, Error> {
    let max = max_encoded_len(src.len()).ok_or(Error::TooLarge)?;
    if dst.len() < start + max {
        dst.resize(start + max, 0);
    }
    let dst = &mut dst[start..start + max];
    let encode_block = match level {
        Level::Fastest => encode_l1::encode_block,
        Level::Balanced => encode_l2::encode_block,
        Level::Smallest => encode_l3::encode_block,
        _ => return Ok(simd::encode_block_simd(dst, src, level.into())),
    };
    if src.len() < format::MIN_NON_LITERAL_BLOCK_SIZE {
        return Ok(0);
    }
    let h = format::put_uvarint_at(dst, src.len() as u64);
    let n = encode_block(&mut dst[h..], src);
    Ok(if n > 0 && h + n < src.len() { h + n } else { 0 })
}

/// Decode a MinLZ stream-chunk body (`varint(dlen) + body`).  `dst` is
/// appended with the decoded bytes.
pub(crate) fn append_decoded_chunk_body(dst: &mut Vec<u8>, body: &[u8]) -> Result<(), Error> {
    let (dlen, compressed) = checked_chunk_body(body)?;
    let start = dst.len();
    dst.reserve(dlen + decode::OVERSHOOT_PAD);
    // SAFETY: same invariants as `append_decoded` — we just reserved
    // `dlen + OVERSHOOT_PAD` writable bytes past `start`.
    unsafe {
        let dst_ptr = dst.as_mut_ptr().add(start);
        decode::minlz_decode(dst_ptr, dlen, compressed)?;
        dst.set_len(start + dlen);
    }
    Ok(())
}

/// Return the uncompressed length declared by a MinLZ stream-chunk body
/// (Go's `minLZChunkLen`).
pub(crate) fn decoded_len_chunk_body(body: &[u8]) -> Result<usize, Error> {
    Ok(split_chunk_body(body)?.0)
}

/// Split a chunk body for decoding. Chunks have no literal-only form, so a
/// decoded length of 0 is corrupt, as is one below the compressed size.
pub(crate) fn checked_chunk_body(body: &[u8]) -> Result<(usize, &[u8]), Error> {
    let (dlen, compressed) = split_chunk_body(body)?;
    if dlen == 0 || dlen < compressed.len() {
        return Err(Error::Corrupt("invalid chunk decoded length"));
    }
    Ok((dlen, compressed))
}

fn split_chunk_body(body: &[u8]) -> Result<(usize, &[u8]), Error> {
    let (v, n) = format::get_uvarint(body).ok_or(Error::Corrupt("bad chunk length varint"))?;
    if v >> 24 != 0 {
        return Err(Error::Corrupt("chunk decoded length exceeds 24 bits"));
    }
    if v > MAX_BLOCK_SIZE as u64 {
        return Err(Error::TooLarge);
    }
    Ok((v as usize, &body[n..]))
}

#[cfg(test)]
mod tests;
