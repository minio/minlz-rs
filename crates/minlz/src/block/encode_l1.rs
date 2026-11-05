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

//! L1 / Fastest encoder.
//!
//! Ports `encode_l1.go:encodeBlockGo` (for inputs > 64 KiB) and
//! `encode_l1.go:encodeBlockGo64K` (for inputs ≤ 64 KiB).

use super::emit::{emit_copy, emit_copy_lits2, emit_copy_lits3, emit_literal, emit_repeat};
use super::format::{
    INPUT_MARGIN, MAX_COPY2_LITS, MAX_COPY3_LITS, MAX_COPY3_OFFSET, MIN_COPY2_OFFSET,
    MIN_NON_LITERAL_BLOCK_SIZE,
};
use super::hash::{hash5, hash6};
use super::load_store::{load32, load64};

/// Entry point: dispatches on `src.len()`.  Returns the number of bytes
/// written into `dst` or `0` if the block is incompressible.
pub(super) fn encode_block(dst: &mut [u8], src: &[u8]) -> usize {
    if src.len() < MIN_NON_LITERAL_BLOCK_SIZE {
        return 0;
    }
    if src.len() <= SMALL_TABLE_MAX_LEN {
        encode_block_64k::<SMALL_TABLE_SIZE>(dst, src)
    } else if src.len() <= 65536 {
        encode_block_64k::<{ 1 << 13 }>(dst, src)
    } else {
        encode_block_big(dst, src)
    }
}

/// L1 encoder for inputs > 64 KiB (port of `encodeBlockGo`).
fn encode_block_big(dst: &mut [u8], src: &[u8]) -> usize {
    const TABLE_BITS: u32 = 15;
    const TABLE_SIZE: usize = 1 << TABLE_BITS;
    const SKIP_LOG: u32 = 6;

    let mut table = vec![0u32; TABLE_SIZE].into_boxed_slice();
    let s_limit = src.len() - INPUT_MARGIN;
    let dst_limit = src.len() - (src.len() >> 5) - 6;
    let mut next_emit: usize = 0;
    let mut s: usize = 1;
    // SAFETY: s + 8 = 9 ≤ src.len() (caller ensures src.len() ≥ MIN_NON_LITERAL_BLOCK_SIZE = 16).
    let mut cv = unsafe { load64(src, s) };
    let mut repeat: usize = 1;
    let mut d: usize = 0;

    'outer: loop {
        // Inner search loop — find a 4-byte match.
        let candidate;
        loop {
            let next_s = s + ((s - next_emit) >> SKIP_LOG) + 4;
            if next_s > s_limit {
                break 'outer;
            }
            // The c1 path shifts `s += 1` and the c2 path shifts `s += 2`
            // *after* this check, so the effective repeat for those paths is
            // `(s+1)-c1` and `(s+2)-c2` respectively.  Tightening `min_src_pos`
            // by +2 prevents `repeat > MAX_COPY3_OFFSET`, which would cause
            // `encode_copy3` to wrap the 21-bit offset field and emit a
            // corrupt block (caught by `tests::regression_l1_offset_boundary`).
            let min_src_pos = (s + 2).saturating_sub(MAX_COPY3_OFFSET);
            let hash0 = hash6(cv, TABLE_BITS) as usize;
            let hash1 = hash6(cv >> 8, TABLE_BITS) as usize;
            let c0 = table[hash0] as usize;
            let c1 = table[hash1] as usize;
            table[hash0] = s as u32;
            table[hash1] = (s + 1) as u32;
            let hash2 = hash6(cv >> 16, TABLE_BITS) as usize;

            // Repeat check at offset +1.
            let prev = s.wrapping_sub(repeat).wrapping_add(1);
            if prev <= s {
                // SAFETY: prev + 4 ≤ s + 5 ≤ src.len() (we're well below s_limit + INPUT_MARGIN).
                if ((cv >> 8) as u32) == unsafe { load32(src, prev) } {
                    let mut base = s + 1;
                    // Extend backwards.  base ≥ repeat at this point because
                    // base = s + 1 and we maintain `s ≥ repeat` after the
                    // first emit (repeat is updated to base - candidate ≤ base).
                    let mut i = base - repeat;
                    while base > next_emit && i > 0 && src[i - 1] == src[base - 1] {
                        i -= 1;
                        base -= 1;
                    }
                    if d + (base - next_emit) > dst_limit {
                        return 0;
                    }
                    d += emit_literal(&mut dst[d..], &src[next_emit..base]);
                    let mut cand = s - repeat + 4 + 1;
                    s += 4 + 1;
                    while s <= s_limit {
                        // SAFETY: s + 8 ≤ src.len(), cand + 8 ≤ src.len() (cand < s).
                        let diff = unsafe { load64(src, s) ^ load64(src, cand) };
                        if diff != 0 {
                            s += (diff.trailing_zeros() as usize) >> 3;
                            break;
                        }
                        s += 8;
                        cand += 8;
                    }
                    d += emit_repeat(&mut dst[d..], s - base);
                    next_emit = s;
                    if s >= s_limit {
                        break 'outer;
                    }
                    // SAFETY: s ≤ s_limit ⇒ s + 8 ≤ src.len().
                    cv = unsafe { load64(src, s) };
                    continue;
                }
            }

            // SAFETY for the three load32s below: c{0,1,2} are previously
            // recorded `s` values (≤ current s) or 0; in either case
            // c + 4 ≤ s + 4 ≤ s_limit + 4 ≤ src.len().  c2 may equal `s`
            // or `s+1` after hash-collision with hash0/hash1 — that's a
            // legitimate self-referential RLE-style match, handled by the
            // decoder's forward-copy path.
            if c0 >= min_src_pos && (cv as u32) == unsafe { load32(src, c0) } {
                candidate = c0;
                break;
            }
            let c2 = table[hash2] as usize;
            if c1 >= min_src_pos && ((cv >> 8) as u32) == unsafe { load32(src, c1) } {
                table[hash2] = (s + 2) as u32;
                candidate = c1;
                s += 1;
                break;
            }
            table[hash2] = (s + 2) as u32;
            if c2 >= min_src_pos && ((cv >> 16) as u32) == unsafe { load32(src, c2) } {
                candidate = c2;
                s += 2;
                break;
            }
            // SAFETY: next_s ≤ s_limit ⇒ next_s + 8 ≤ src.len().
            cv = unsafe { load64(src, next_s) };
            s = next_s;
        }

        // Found a 4-byte match.  Extend backwards.
        let mut candidate_pos = candidate;
        while candidate_pos > 0 && s > next_emit && src[candidate_pos - 1] == src[s - 1] {
            candidate_pos -= 1;
            s -= 1;
        }
        let base = s;
        repeat = base - candidate_pos;

        // Extend forwards.
        let mut cand = candidate_pos + 4;
        s += 4;
        while s + 8 <= src.len() {
            // SAFETY: s + 8 ≤ src.len(); cand + 8 ≤ src.len() because cand < s.
            let diff = unsafe { load64(src, s) ^ load64(src, cand) };
            if diff != 0 {
                s += (diff.trailing_zeros() as usize) >> 3;
                break;
            }
            s += 8;
            cand += 8;
        }
        let length = s - base;

        if next_emit != base {
            let lit_count = base - next_emit;
            if lit_count > MAX_COPY3_LITS || repeat < MIN_COPY2_OFFSET {
                if d + (s - next_emit) > dst_limit {
                    return 0;
                }
                d += emit_literal(&mut dst[d..], &src[next_emit..base]);
                d += emit_copy(&mut dst[d..], repeat, length);
            } else if repeat <= super::format::MAX_COPY2_OFFSET {
                d += emit_copy_lits2(&mut dst[d..], &src[next_emit..base], repeat, length);
            } else {
                d += emit_copy_lits3(&mut dst[d..], &src[next_emit..base], repeat, length);
            }
        } else {
            d += emit_copy(&mut dst[d..], repeat, length);
        }

        // Inner immediate-match loop.
        loop {
            next_emit = s;
            if s >= s_limit {
                break 'outer;
            }
            // SAFETY: s ≥ 5 (post-main-emit), s + 6 ≤ s_limit + 6 ≤ src.len().
            let x = unsafe { load64(src, s - 2) };
            if d > dst_limit {
                return 0;
            }
            let m2_hash = hash6(x, TABLE_BITS) as usize;
            let x_top = x >> 16;
            let curr_hash = hash6(x_top, TABLE_BITS) as usize;
            let cand = table[curr_hash] as usize;
            table[m2_hash] = (s - 2) as u32;
            table[curr_hash] = s as u32;
            // SAFETY: cand was a previously recorded `s` value (< current s)
            // or 0; cand + 4 ≤ src.len().
            if s.saturating_sub(cand) > MAX_COPY3_OFFSET
                || (x_top as u32) != unsafe { load32(src, cand) }
            {
                // SAFETY: s < s_limit ⇒ s + 1 + 8 ≤ src.len().
                cv = unsafe { load64(src, s + 1) };
                s += 1;
                break;
            }
            repeat = s - cand;
            let base2 = s;
            let mut c2 = cand + 4;
            s += 4;
            while s + 8 <= src.len() {
                // SAFETY: s + 8 ≤ src.len(); c2 + 8 ≤ src.len() (c2 < s).
                let diff = unsafe { load64(src, s) ^ load64(src, c2) };
                if diff != 0 {
                    s += (diff.trailing_zeros() as usize) >> 3;
                    break;
                }
                s += 8;
                c2 += 8;
            }
            d += emit_copy(&mut dst[d..], repeat, s - base2);
        }
    }

    // emitRemainder
    if next_emit < src.len() {
        if d + src.len() - next_emit > dst_limit {
            return 0;
        }
        d += emit_literal(&mut dst[d..], &src[next_emit..]);
    }
    d
}

/// Inputs up to this length use a [`SMALL_TABLE_SIZE`]-entry hash table.
const SMALL_TABLE_MAX_LEN: usize = 2 << 10;
/// Hash table entries for inputs up to [`SMALL_TABLE_MAX_LEN`] (2 KiB of
/// `u16`s).  Zeroing the full 16 KiB table would dominate the encode time
/// for such inputs, and most of it would stay empty.
const SMALL_TABLE_SIZE: usize = 1 << 10;

/// L1 encoder for inputs ≤ 64 KiB (port of `encodeBlockGo64K`), with a
/// `TABLE_SIZE`-entry hash table.
///
/// Go's `encodeBlockGo64K` always uses 8192 entries; small inputs use
/// [`SMALL_TABLE_SIZE`] instead (Go's amd64 assembly encoders also size the
/// table to the input).  Each size is a separate instantiation, so the hash
/// shift is a constant and table indexing needs no bounds checks.
fn encode_block_64k<const TABLE_SIZE: usize>(dst: &mut [u8], src: &[u8]) -> usize {
    const { assert!(TABLE_SIZE.is_power_of_two()) };
    const SKIP_LOG: u32 = 5;

    let table_bits = TABLE_SIZE.trailing_zeros();
    let mut table = [0u16; TABLE_SIZE];
    let s_limit = src.len() - INPUT_MARGIN;
    let dst_limit = src.len() - (src.len() >> 5) - 6;
    let mut next_emit: usize = 0;
    let mut s: usize = 1;
    // SAFETY: src.len() ≥ MIN_NON_LITERAL_BLOCK_SIZE = 16 ⇒ s + 8 ≤ src.len().
    let mut cv = unsafe { load64(src, s) };
    let mut repeat: usize = 1;
    let mut d: usize = 0;

    'outer: loop {
        let candidate;
        loop {
            let next_s = s + ((s - next_emit) >> SKIP_LOG) + 4;
            if next_s > s_limit {
                break 'outer;
            }
            let hash0 = hash5(cv, table_bits) as usize;
            let hash1 = hash5(cv >> 8, table_bits) as usize;
            let c0 = table[hash0] as usize;
            let c1 = table[hash1] as usize;
            table[hash0] = s as u16;
            table[hash1] = (s + 1) as u16;
            let hash2 = hash5(cv >> 16, table_bits) as usize;

            // Repeat check at offset +1.
            let prev = s.wrapping_sub(repeat).wrapping_add(1);
            if prev <= s {
                // SAFETY: prev + 4 ≤ s + 5 ≤ src.len().
                if ((cv >> 8) as u32) == unsafe { load32(src, prev) } {
                    let mut base = s + 1;
                    let mut i = base - repeat;
                    while base > next_emit && i > 0 && src[i - 1] == src[base - 1] {
                        i -= 1;
                        base -= 1;
                    }
                    if d + (base - next_emit) > dst_limit {
                        return 0;
                    }
                    d += emit_literal(&mut dst[d..], &src[next_emit..base]);
                    let mut cand = s - repeat + 4 + 1;
                    s += 4 + 1;
                    while s <= s_limit {
                        let diff = unsafe { load64(src, s) ^ load64(src, cand) };
                        if diff != 0 {
                            s += (diff.trailing_zeros() as usize) >> 3;
                            break;
                        }
                        s += 8;
                        cand += 8;
                    }
                    d += emit_repeat(&mut dst[d..], s - base);
                    next_emit = s;
                    if s >= s_limit {
                        break 'outer;
                    }
                    cv = unsafe { load64(src, s) };
                    continue;
                }
            }

            if (cv as u32) == unsafe { load32(src, c0) } {
                candidate = c0;
                break;
            }
            let c2 = table[hash2] as usize;
            if ((cv >> 8) as u32) == unsafe { load32(src, c1) } {
                table[hash2] = (s + 2) as u16;
                s += 1;
                candidate = c1;
                break;
            }
            table[hash2] = (s + 2) as u16;
            if ((cv >> 16) as u32) == unsafe { load32(src, c2) } {
                s += 2;
                candidate = c2;
                break;
            }
            cv = unsafe { load64(src, next_s) };
            s = next_s;
        }

        let mut candidate_pos = candidate;
        while candidate_pos > 0 && s > next_emit && src[candidate_pos - 1] == src[s - 1] {
            candidate_pos -= 1;
            s -= 1;
        }
        let base = s;
        repeat = base - candidate_pos;
        let mut cand = candidate_pos + 4;
        s += 4;
        while s + 8 <= src.len() {
            let diff = unsafe { load64(src, s) ^ load64(src, cand) };
            if diff != 0 {
                s += (diff.trailing_zeros() as usize) >> 3;
                break;
            }
            s += 8;
            cand += 8;
        }
        let length = s - base;

        if next_emit != base {
            let lit_count = base - next_emit;
            if lit_count > MAX_COPY2_LITS || repeat < MIN_COPY2_OFFSET {
                if d + (s - next_emit) > dst_limit {
                    return 0;
                }
                d += emit_literal(&mut dst[d..], &src[next_emit..base]);
                d += emit_copy(&mut dst[d..], repeat, length);
            } else {
                d += emit_copy_lits2(&mut dst[d..], &src[next_emit..base], repeat, length);
            }
        } else {
            d += emit_copy(&mut dst[d..], repeat, length);
        }

        loop {
            next_emit = s;
            if s >= s_limit {
                break 'outer;
            }
            let x = unsafe { load64(src, s - 2) };
            if d > dst_limit {
                return 0;
            }
            let m2_hash = hash5(x, table_bits) as usize;
            let x_top = x >> 16;
            let curr_hash = hash5(x_top, table_bits) as usize;
            let cand = table[curr_hash] as usize;
            table[m2_hash] = (s - 2) as u16;
            table[curr_hash] = s as u16;
            if (x_top as u32) != unsafe { load32(src, cand) } {
                cv = unsafe { load64(src, s + 1) };
                s += 1;
                break;
            }
            repeat = s - cand;
            let base2 = s;
            let mut c2 = cand + 4;
            s += 4;
            while s + 8 <= src.len() {
                let diff = unsafe { load64(src, s) ^ load64(src, c2) };
                if diff != 0 {
                    s += (diff.trailing_zeros() as usize) >> 3;
                    break;
                }
                s += 8;
                c2 += 8;
            }
            d += emit_copy(&mut dst[d..], repeat, s - base2);
        }
    }

    if next_emit < src.len() {
        if d + src.len() - next_emit > dst_limit {
            return 0;
        }
        d += emit_literal(&mut dst[d..], &src[next_emit..]);
    }
    d
}

#[cfg(test)]
mod tests {
    use super::super::format::max_encoded_len;
    use super::super::{Level, decode, encode};
    use super::*;

    /// `n` bytes of compressible text: random words from a 256-word
    /// vocabulary of random letters.
    fn words(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let vocab: Vec<Vec<u8>> = (0..256)
            .map(|_| {
                let len = 2 + next() % 7;
                let mut w: Vec<u8> = (0..len).map(|_| b'a' + (next() % 26) as u8).collect();
                w.push(b' ');
                w
            })
            .collect();
        let mut src = Vec::with_capacity(n + 16);
        while src.len() < n {
            src.extend_from_slice(&vocab[(next() % 256) as usize]);
        }
        src.truncate(n);
        src
    }

    fn encode_with(src: &[u8], f: impl FnOnce(&mut [u8], &[u8]) -> usize) -> Vec<u8> {
        let mut dst = vec![0; max_encoded_len(src.len()).unwrap()];
        let n = f(&mut dst, src);
        dst.truncate(n);
        dst
    }

    /// Inputs up to `SMALL_TABLE_MAX_LEN` use the small table and longer ones
    /// the full table; both round-trip.
    #[test]
    fn small_table_threshold() {
        for n in [SMALL_TABLE_MAX_LEN, SMALL_TABLE_MAX_LEN + 1] {
            let src = words(n, 1);
            let small = encode_with(&src, encode_block_64k::<SMALL_TABLE_SIZE>);
            let full = encode_with(&src, encode_block_64k::<{ 1 << 13 }>);
            assert_ne!(small, full, "n={n}: input does not tell the tables apart");
            let want = if n <= SMALL_TABLE_MAX_LEN {
                small
            } else {
                full
            };
            assert_eq!(encode_with(&src, encode_block), want, "n={n}");

            let mut enc = Vec::new();
            encode(&mut enc, &src, Level::Fastest).unwrap();
            let mut dec = Vec::new();
            decode(&mut dec, &enc).unwrap();
            assert_eq!(dec, src, "n={n}");
        }
    }
}
