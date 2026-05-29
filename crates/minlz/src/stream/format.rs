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

//! Wire-format constants and small helpers for the MinLZ stream codec.
//!
//! All values match the Go reference (`minlz.go`).  Where Go exports a name
//! we mirror it; where Go keeps a name lowercase (`chunkType…`) we keep it
//! `pub(crate)`.

pub use crate::block::MAX_BLOCK_SIZE;

/// Minimum block size accepted by [`crate::stream::WriterBuilder::block_size`].
pub const MIN_BLOCK_SIZE: usize = 4 << 10;

/// Default block size used by [`crate::stream::Writer::new`] (2 MiB).
pub const DEFAULT_BLOCK_SIZE: usize = 2 << 20;

/// Largest block-size indicator value the spec allows (block size `1 << 23`).
pub(crate) const MAX_BLOCK_LOG: u8 = 23;

/// Maximum byte length of a user chunk's payload.
pub const MAX_USER_CHUNK_SIZE: usize = (1 << 24) - 1;

/// Lowest user-defined skippable chunk ID (`0x80`).
pub const MIN_USER_SKIPPABLE_CHUNK: u8 = 0x80;
/// Highest user-defined skippable chunk ID (`0xbf`).
pub const MAX_USER_SKIPPABLE_CHUNK: u8 = 0xbf;
/// Lowest user-defined non-skippable chunk ID (`0xc0`).
pub const MIN_USER_NON_SKIPPABLE_CHUNK: u8 = 0xc0;
/// Highest user-defined non-skippable chunk ID (`0xfd`).
pub const MAX_USER_NON_SKIPPABLE_CHUNK: u8 = 0xfd;

/// Padding chunk ID.
pub const CHUNK_TYPE_PADDING: u8 = 0xfe;
/// Stream-identifier chunk ID.
pub const CHUNK_TYPE_STREAM_IDENTIFIER: u8 = 0xff;

pub(crate) const CHUNK_TYPE_UNCOMPRESSED_DATA: u8 = 0x01;
pub(crate) const CHUNK_TYPE_MINLZ_COMPRESSED_DATA: u8 = 0x02;
pub(crate) const CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC: u8 = 0x03;
pub(crate) const CHUNK_TYPE_EOF: u8 = 0x20;
pub(crate) const MAX_NON_SKIPPABLE_CHUNK: u8 = 0x3f;

pub(crate) const CHECKSUM_SIZE: usize = 4;
pub(crate) const CHUNK_HEADER_SIZE: usize = 4;

pub(crate) const MAGIC_BODY: &[u8] = b"MinLz";
/// Magic body + 1-byte block-size indicator.
pub(crate) const MAGIC_BODY_LEN: usize = MAGIC_BODY.len() + 1;
/// Full stream-identifier chunk length: 4-byte header + 6-byte body.
pub(crate) const STREAM_HEADER_LEN: usize = CHUNK_HEADER_SIZE + MAGIC_BODY_LEN;

/// Maximum encoded form of a uvarint(u64).
pub(crate) const MAX_VARINT_LEN_64: usize = 10;

/// Map a block size (a power of two between [`MIN_BLOCK_SIZE`] and
/// [`MAX_BLOCK_SIZE`]) to the indicator byte stored in the stream header.
///
/// Returns the indicator value: `bits::len(block_size - 1) - 10`.
pub(crate) fn block_size_indicator(block_size: usize) -> u8 {
    debug_assert!((MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE).contains(&block_size));
    // bits::Len(n) in Go = number of bits required to represent n, i.e.
    // 32 - leading_zeros(n) for non-zero n.
    let bl = 64 - ((block_size - 1) as u64).leading_zeros() as u8;
    bl - 10
}

/// Inverse of [`block_size_indicator`]: turn a header indicator byte into the
/// block size it encodes.  Returns `None` if the indicator is out of range.
pub(crate) fn block_size_from_indicator(indicator: u8) -> Option<usize> {
    // Reader masks with &15 then adds 10 (see Go `reader.go:minLzHeader`).
    let n = (indicator & 15) + 10;
    if n > MAX_BLOCK_LOG {
        return None;
    }
    Some(1usize << n)
}

/// Build the 10-byte stream-identifier chunk for `block_size`.
pub(crate) fn make_stream_header(block_size: usize) -> [u8; STREAM_HEADER_LEN] {
    // 4-byte chunk header: type=0xff, len=6 (little-endian 3 bytes).
    // 6-byte body: "MinLz" + block-size indicator.
    [
        CHUNK_TYPE_STREAM_IDENTIFIER,
        MAGIC_BODY_LEN as u8,
        0,
        0,
        MAGIC_BODY[0],
        MAGIC_BODY[1],
        MAGIC_BODY[2],
        MAGIC_BODY[3],
        MAGIC_BODY[4],
        block_size_indicator(block_size),
    ]
}

/// Decode a 3-byte little-endian chunk length.
#[inline]
pub(crate) fn read_chunk_len(b: &[u8]) -> usize {
    debug_assert!(b.len() >= 3);
    (b[0] as usize) | ((b[1] as usize) << 8) | ((b[2] as usize) << 16)
}

/// Decode a 4-byte little-endian u32 checksum.
#[inline]
pub(crate) fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Encode an unsigned varint into `dst`; returns bytes written.
pub(crate) fn put_uvarint(dst: &mut [u8], mut value: u64) -> usize {
    let mut i = 0;
    while value >= 0x80 {
        dst[i] = (value as u8) | 0x80;
        value >>= 7;
        i += 1;
    }
    dst[i] = value as u8;
    i + 1
}

/// Decode an unsigned varint from `src`; returns `(value, n_bytes_consumed)`
/// or `None` on overflow / truncation.  Mirrors Go's `binary.Uvarint`.
pub(crate) fn get_uvarint(src: &[u8]) -> Option<(u64, usize)> {
    let mut x: u64 = 0;
    let mut s: u32 = 0;
    for (i, &b) in src.iter().enumerate() {
        if i == MAX_VARINT_LEN_64 {
            return None;
        }
        if b < 0x80 {
            if i == MAX_VARINT_LEN_64 - 1 && b > 1 {
                return None;
            }
            return Some((x | (u64::from(b) << s), i + 1));
        }
        x |= u64::from(b & 0x7f) << s;
        s += 7;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indicator_round_trip() {
        let sizes = [
            MIN_BLOCK_SIZE,
            8 << 10,
            64 << 10,
            1 << 20,
            DEFAULT_BLOCK_SIZE,
            MAX_BLOCK_SIZE,
        ];
        for &sz in &sizes {
            let ind = block_size_indicator(sz);
            assert_eq!(block_size_from_indicator(ind), Some(sz), "size {sz}");
        }
    }

    #[test]
    fn stream_header_matches_go() {
        // makeHeader(2<<20) in Go writes: ff 06 00 00 'M' 'i' 'n' 'L' 'z' 0x0b
        let h = make_stream_header(DEFAULT_BLOCK_SIZE);
        assert_eq!(
            &h,
            &[0xff, 0x06, 0x00, 0x00, b'M', b'i', b'n', b'L', b'z', 0x0b]
        );
    }

    #[test]
    fn uvarint_round_trip() {
        let mut buf = [0u8; MAX_VARINT_LEN_64];
        for &v in &[0u64, 1, 127, 128, 16383, 16384, 1 << 32, u64::MAX] {
            let n = put_uvarint(&mut buf, v);
            let (got, n2) = get_uvarint(&buf[..n]).unwrap();
            assert_eq!(got, v);
            assert_eq!(n, n2);
        }
    }
}
