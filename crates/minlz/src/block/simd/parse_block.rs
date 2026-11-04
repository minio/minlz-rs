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

//! The parser of LevelFastestSIMD, a port of `(*simdEncoder).encodeBlockGo` and
//! `encodeBlockGo64K` of Go's `encode_l1.go`.

use super::encode::{Encoder, load32, load64};
use crate::block::format::INPUT_MARGIN;
use crate::block::hash::hash6;
use crate::block::table_pool::{self, U16_14, U32_15};

/// A hash table entry: a position in the block.
pub(super) trait Entry: Copy + Default {
    fn new(s: usize) -> Self;
    fn get(self) -> usize;
}

impl Entry for u32 {
    #[inline(always)]
    fn new(s: usize) -> Self {
        s as u32
    }
    #[inline(always)]
    fn get(self) -> usize {
        self as usize
    }
}

impl Entry for u16 {
    #[inline(always)]
    fn new(s: usize) -> Self {
        s as u16
    }
    #[inline(always)]
    fn get(self) -> usize {
        self as usize
    }
}

impl<'a> Encoder<'a> {
    /// encodeBlockGo parses src like the function encodeBlockGo, without its size and offset
    /// limits.
    pub(super) fn encode_block_go(&mut self, src: &'a [u8]) {
        let mut table = table_pool::get(&U32_15);
        self.parse_block(src, &mut table);
        table_pool::put(&U32_15, table);
    }

    /// encodeBlockGo64K is encodeBlockGo for blocks of at most 64 KB, with fewer and 16 bit table
    /// entries.
    pub(super) fn encode_block_go_64k(&mut self, src: &'a [u8]) {
        let mut table = table_pool::get(&U16_14);
        self.parse_block(src, &mut table);
        table_pool::put(&U16_14, table);
    }

    /// The body of encodeBlockGo and encodeBlockGo64K, which differ only in their tables.
    #[inline(always)]
    fn parse_block<E: Entry, const N: usize>(&mut self, src: &'a [u8], table: &mut [E; N]) {
        const SKIP_LOG: usize = 6;
        const CHECK_REP: usize = 1;
        let table_bits = N.trailing_zeros();
        let s_limit = src.len() - INPUT_MARGIN;
        let mut next_emit = 0;
        let mut s = 1;
        let mut cv = load64(src, s);
        let mut repeat = 1;
        'outer: loop {
            let mut candidate;
            loop {
                let next_s = s + ((s - next_emit) >> SKIP_LOG) + 4;
                if next_s > s_limit {
                    break 'outer;
                }
                let hash0 = hash6(cv, table_bits) as usize;
                let hash1 = hash6(cv >> 8, table_bits) as usize;
                candidate = table[hash0].get();
                let candidate2 = table[hash1].get();
                table[hash0] = E::new(s);
                table[hash1] = E::new(s + 1);
                let hash2 = hash6(cv >> 16, table_bits) as usize;

                if (cv >> (CHECK_REP * 8)) as u32 == load32(src, s - repeat + CHECK_REP) {
                    let mut base = s + CHECK_REP;
                    let mut i = base - repeat;
                    while base > next_emit && i > 0 && src[i - 1] == src[base - 1] {
                        i -= 1;
                        base -= 1;
                    }
                    let mut candidate = s - repeat + 4 + CHECK_REP;
                    s += 4 + CHECK_REP;
                    while s <= s_limit {
                        let diff = load64(src, s) ^ load64(src, candidate);
                        if diff != 0 {
                            s += diff.trailing_zeros() as usize >> 3;
                            break;
                        }
                        s += 8;
                        candidate += 8;
                    }
                    if self.fast(src, next_emit, base, s, repeat) {
                        self.put_lits(src, next_emit, base - next_emit);
                        let t = 1 | self.put_ll(base - next_emit);
                        if self.put_match(t, repeat, s - base) {
                            self.end_chunk();
                        }
                    } else {
                        self.emit_repeat_lits(&src[next_emit..base], s - base);
                        repeat = self.rep;
                    }
                    next_emit = s;
                    if s >= s_limit {
                        break 'outer;
                    }
                    cv = load64(src, s);
                    continue;
                }

                if cv as u32 == load32(src, candidate) {
                    break;
                }
                candidate = table[hash2].get();
                if (cv >> 8) as u32 == load32(src, candidate2) {
                    table[hash2] = E::new(s + 2);
                    candidate = candidate2;
                    s += 1;
                    break;
                }
                table[hash2] = E::new(s + 2);
                if (cv >> 16) as u32 == load32(src, candidate) {
                    s += 2;
                    break;
                }
                cv = load64(src, next_s);
                s = next_s;
            }

            while candidate > 0 && s > next_emit && src[candidate - 1] == src[s - 1] {
                candidate -= 1;
                s -= 1;
            }
            let mut base = s;
            repeat = base - candidate;
            s += 4;
            candidate += 4;
            while s + 8 <= src.len() {
                let diff = load64(src, s) ^ load64(src, candidate);
                if diff != 0 {
                    s += diff.trailing_zeros() as usize >> 3;
                    break;
                }
                s += 8;
                candidate += 8;
            }
            if self.fast(src, next_emit, base, s, repeat) {
                let b = self.put_offset(repeat);
                self.put_bits(b);
                self.put_lits(src, next_emit, base - next_emit);
                let t = self.put_ll(base - next_emit);
                if self.put_match(t, repeat, s - base) {
                    self.end_chunk();
                }
            } else {
                self.emit_copy_lits(&src[next_emit..base], repeat, s - base);
                repeat = self.rep;
            }

            loop {
                next_emit = s;
                if s >= s_limit {
                    break 'outer;
                }
                let mut x = load64(src, s - 2);
                let m2_hash = hash6(x, table_bits) as usize;
                x >>= 16;
                let curr_hash = hash6(x, table_bits) as usize;
                candidate = table[curr_hash].get();
                table[m2_hash] = E::new(s - 2);
                table[curr_hash] = E::new(s);
                if x as u32 != load32(src, candidate) {
                    cv = load64(src, s + 1);
                    s += 1;
                    break;
                }
                repeat = s - candidate;
                base = s;
                s += 4;
                candidate += 4;
                while s + 8 <= src.len() {
                    let diff = load64(src, s) ^ load64(src, candidate);
                    if diff != 0 {
                        s += diff.trailing_zeros() as usize >> 3;
                        break;
                    }
                    s += 8;
                    candidate += 8;
                }
                if self.one_op(repeat, s - base) {
                    let b = self.put_offset(repeat);
                    self.put_bits(b);
                    if self.put_match(0, repeat, s - base) {
                        self.end_chunk();
                    }
                } else {
                    self.emit_copy(repeat, s - base);
                    repeat = self.rep;
                }
            }
        }

        if next_emit < src.len() {
            self.emit_literals(&src[next_emit..]);
        }
    }
}
