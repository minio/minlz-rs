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

//! The parser of LevelBalancedSIMD, a port of `(*simdEncoder).encodeBlockBetterGo` and
//! `encodeBlockBetterGo64K` of Go's `encode_l2.go`.

use super::MAX_LEN;
use super::encode::{Encoder, load32, load64, match_at};
use super::parse_block::Entry;
use crate::block::format::INPUT_MARGIN;
use crate::block::hash::{hash4, hash7};
use crate::block::table_pool::{self, U16_13, U16_16, U32_14, U32_17};

/// simdBetterMaxOff returns the largest offsets of new 4 and 5 byte matches that Balanced SIMD
/// takes in a block of n bytes, indexed by length&1. Farther short matches cost more than their
/// literals: up to 2 KB the operation streams are stored, so every token, literal length and
/// offset symbol costs 8 bits; in larger blocks the offset bits grow with the offset.
/// Between 2 and 8 KB fewer operations can make a coded stream go stored, so the bound is looser
/// there.
pub(super) fn better_max_off(n: usize) -> [usize; 2] {
    if n <= 2 << 10 {
        [127, 511]
    } else if n <= 8 << 10 {
        [2047, 16383]
    } else {
        [1023, 16383]
    }
}

impl<'a> Encoder<'a> {
    /// encodeBlockBetterGo parses src like the function encodeBlockBetterGo, without its size and
    /// offset limits.
    pub(super) fn encode_block_better_go(&mut self, src: &'a [u8]) {
        let mut l_table = table_pool::get(&U32_17);
        let mut s_table = table_pool::get(&U32_14);
        self.parse_better(src, &mut l_table, &mut s_table);
        table_pool::put(&U32_14, s_table);
        table_pool::put(&U32_17, l_table);
    }

    /// encodeBlockBetterGo64K is encodeBlockBetterGo for blocks of at most 64 KB, with fewer and
    /// 16 bit table entries.
    pub(super) fn encode_block_better_go_64k(&mut self, src: &'a [u8]) {
        let mut l_table = table_pool::get(&U16_16);
        let mut s_table = table_pool::get(&U16_13);
        self.parse_better(src, &mut l_table, &mut s_table);
        table_pool::put(&U16_13, s_table);
        table_pool::put(&U16_16, l_table);
    }

    /// The body of encodeBlockBetterGo and encodeBlockBetterGo64K, which differ only in their
    /// tables.
    #[inline(always)]
    fn parse_better<E: Entry, const L: usize, const S: usize>(
        &mut self,
        src: &'a [u8],
        l_table: &mut [E; L],
        s_table: &mut [E; S],
    ) {
        const CHECK_REP: usize = 1;
        const WANT_REPEAT_BYTES: usize = 4;
        const REPEAT_MASK: u64 = ((1 << (WANT_REPEAT_BYTES * 8)) - 1) << (8 * CHECK_REP);
        let (l_bits, s_bits) = (L.trailing_zeros(), S.trailing_zeros());
        let s_limit = src.len() - INPUT_MARGIN;
        let mut next_emit = 0;
        let mut s = 1;
        let mut cv = load64(src, s);
        let mut repeat = 1;

        let max_off = better_max_off(src.len());
        'outer: loop {
            let mut candidate_l;
            let mut next_s;
            let (mut base, mut offset, mut length, mut rep) = (0, 0, 0, false);
            loop {
                next_s = s + 100.min(((s - next_emit) >> 7) + 1);
                if next_s > s_limit {
                    break 'outer;
                }
                let mut hash_l = hash7(cv, l_bits) as usize;
                let hash_s = hash4(cv, s_bits) as usize;
                candidate_l = l_table[hash_l].get();
                let candidate_s = s_table[hash_s].get();
                l_table[hash_l] = E::new(s);
                s_table[hash_s] = E::new(s);

                let val_long = load64(src, candidate_l);
                let val_short = load64(src, candidate_s);
                if cv == val_long {
                    break;
                }

                if cv & REPEAT_MASK == load64(src, s - repeat) & REPEAT_MASK {
                    base = s + CHECK_REP;
                    let mut i = base - repeat;
                    while base > next_emit && i > 0 && src[i - 1] == src[base - 1] {
                        i -= 1;
                        base -= 1;
                    }
                    (offset, length, rep) = (repeat, match_at(src, base, base - repeat), true);
                    break;
                }

                if cv as u32 == val_long as u32 {
                    break;
                }
                if cv as u32 == val_short as u32 {
                    hash_l = hash7(cv >> 8, l_bits) as usize;
                    candidate_l = l_table[hash_l].get();
                    l_table[hash_l] = E::new(s + 1);
                    if (cv >> 8) as u32 == load32(src, candidate_l) {
                        s += 1;
                        break;
                    }
                    candidate_l = candidate_s;
                    break;
                }
                cv = load64(src, next_s);
                s = next_s;
            }

            if !rep {
                while candidate_l > 0 && s > next_emit && src[candidate_l - 1] == src[s - 1] {
                    candidate_l -= 1;
                    s -= 1;
                }
                base = s;
                offset = base - candidate_l;
                length = match_at(src, base, candidate_l);
                // Bail if the match is equal or worse to the encoding.
                if length <= 5 && offset > max_off[length & 1] && repeat != offset {
                    s = next_s + 1;
                    if s >= s_limit {
                        break 'outer;
                    }
                    cv = load64(src, s);
                    continue;
                }
            }

            // A match of MAX_LEN+1 bytes needs an escape, so leave its last byte as a literal.
            if length == MAX_LEN + 1 {
                length -= 1;
            }
            if !self.fast(src, next_emit, base, base + length, offset) {
                if rep {
                    self.emit_repeat_lits(&src[next_emit..base], length);
                } else {
                    self.emit_copy_lits(&src[next_emit..base], offset, length);
                }
            } else if rep {
                self.put_lits(src, next_emit, base - next_emit);
                let t = 1 | self.put_ll(base - next_emit);
                if self.put_match(t, offset, length) {
                    self.end_chunk();
                }
            } else {
                let b = self.put_offset(offset);
                self.put_bits(b);
                self.put_lits(src, next_emit, base - next_emit);
                let t = self.put_ll(base - next_emit);
                if self.put_match(t, offset, length) {
                    self.end_chunk();
                }
            }
            repeat = self.rep;
            s = base + length;
            next_emit = s;
            if s >= s_limit {
                break 'outer;
            }

            let (mut index0, mut index1) = (base + 1, s - 2);
            if rep {
                while index0 < index1 {
                    let cv0 = load64(src, index0);
                    let cv1 = load64(src, index1);
                    l_table[hash7(cv0, l_bits) as usize] = E::new(index0);
                    s_table[hash4(cv0 >> 8, s_bits) as usize] = E::new(index0 + 1);
                    l_table[hash7(cv1, l_bits) as usize] = E::new(index1);
                    s_table[hash4(cv1 >> 8, s_bits) as usize] = E::new(index1 + 1);
                    index0 += 2;
                    index1 -= 2;
                }
                cv = load64(src, s);
                continue;
            }

            let cv0 = load64(src, index0);
            let cv1 = load64(src, index1);
            l_table[hash7(cv0, l_bits) as usize] = E::new(index0);
            s_table[hash4(cv0 >> 8, s_bits) as usize] = E::new(index0 + 1);
            l_table[hash7(cv1, l_bits) as usize] = E::new(index1);
            s_table[hash4(cv1 >> 8, s_bits) as usize] = E::new(index1 + 1);
            index0 += 1;
            index1 -= 1;
            cv = load64(src, s);

            let mut index2 = (index0 + index1 + 1) >> 1;
            while index2 < index1 {
                l_table[hash7(load64(src, index0), l_bits) as usize] = E::new(index0);
                l_table[hash7(load64(src, index2), l_bits) as usize] = E::new(index2);
                index0 += 2;
                index2 += 2;
            }
        }

        if next_emit < src.len() {
            self.emit_literals(&src[next_emit..]);
        }
    }
}
