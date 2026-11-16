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

//! The parser of LevelSuperFastSIMD, a port of `(*simdEncoder).encodeFastBlockGo` and
//! `encodeFastBlockGo64K` of Go's `encode_l0.go`.

use super::encode::{Encoder, match_back};
use super::parse_block::Entry;
use crate::block::format::INPUT_MARGIN;
use crate::block::hash::hash8;
use crate::block::load_store::{load32, load64};
use crate::block::table_pool::{self, U16_13, U32_13};

impl<'a> Encoder<'a> {
    /// Parses src like Go's function `encodeFastBlockGo`, without its size and offset limits.
    pub(super) fn encode_fast_block(&mut self, src: &'a [u8]) {
        let mut table = table_pool::get(&U32_13);
        self.parse_fast(src, &mut table);
        table_pool::put(&U32_13, table);
    }

    /// Parses src as [`Encoder::encode_fast_block`] does, for blocks of at most 64 KB, with 16 bit
    /// table entries.
    pub(super) fn encode_fast_block_64k(&mut self, src: &'a [u8]) {
        let mut table = table_pool::get(&U16_13);
        self.parse_fast(src, &mut table);
        table_pool::put(&U16_13, table);
    }

    /// The body of encode_fast_block and encode_fast_block_64k, which differ only in their tables.
    ///
    /// Its loads are unchecked, as Go's: src has more than MIN_SAVING bytes, s and the entries
    /// of the zeroed table stay at most s_limit, and candidates and repeat sources are below s.
    // Not inlined into the callers, whose unwinding drops the table: a function with a landing
    // pad keeps a frame pointer on Windows, a register the loop needs.
    #[inline(never)]
    fn parse_fast<E: Entry, const N: usize>(&mut self, src: &'a [u8], table: &mut [E; N]) {
        const SKIP_LOG: usize = 5;
        const CHECK_REP: usize = 1;
        let table_bits = N.trailing_zeros();
        let s_limit = src.len() - INPUT_MARGIN;
        let mut next_emit = 0;
        let mut s = 1;
        // SAFETY: src.len() > MIN_SAVING.
        let mut cv = unsafe { load64(src, s) };
        let mut repeat = 1;
        'outer: loop {
            let mut candidate;
            loop {
                let next_s = s + ((s - next_emit) >> SKIP_LOG) + 5;
                if next_s > s_limit {
                    break 'outer;
                }
                let hash0 = hash8(cv, table_bits) as usize;
                // SAFETY: s + 5 <= next_s <= s_limit.
                let cv1 = unsafe { load64(src, s + 1) };
                let hash1 = hash8(cv1, table_bits) as usize;
                candidate = table[hash0].get();
                let candidate2 = table[hash1].get();
                table[hash0] = E::new(s);
                table[hash1] = E::new(s + 1);
                // SAFETY: as cv1.
                let cv2 = unsafe { load64(src, s + 2) };
                let hash2 = hash8(cv2, table_bits) as usize;

                // SAFETY: as cv1; the repeat offset is at most the output, so at most s.
                if cv1 as u32 == unsafe { load32(src, s - repeat + CHECK_REP) } {
                    let mut base = s + CHECK_REP;
                    let i = base - repeat;
                    // SAFETY: i < base <= s + 1.
                    base -= unsafe { match_back(src, base, i, i.min(base - next_emit)) };
                    let mut candidate = s - repeat + 4 + CHECK_REP;
                    s += 4 + CHECK_REP;
                    while s <= s_limit {
                        // SAFETY: candidate < s <= s_limit.
                        let diff = unsafe { load64(src, s) ^ load64(src, candidate) };
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
                    // SAFETY: s < s_limit.
                    cv = unsafe { load64(src, s) };
                    continue;
                }

                // SAFETY: the table holds 0 or positions below s_limit.
                if cv == unsafe { load64(src, candidate) } {
                    break;
                }
                candidate = table[hash2].get();
                // SAFETY: as candidate.
                if cv1 == unsafe { load64(src, candidate2) } {
                    table[hash2] = E::new(s + 2);
                    candidate = candidate2;
                    s += 1;
                    break;
                }
                table[hash2] = E::new(s + 2);
                // SAFETY: as above.
                if cv2 == unsafe { load64(src, candidate) } {
                    s += 2;
                    break;
                }
                // SAFETY: next_s <= s_limit.
                cv = unsafe { load64(src, next_s) };
                s = next_s;
            }

            let mut base = s;
            repeat = base - candidate;
            s += 8;
            candidate += 8;
            while s + 8 <= src.len() {
                // SAFETY: candidate < s.
                let diff = unsafe { load64(src, s) ^ load64(src, candidate) };
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
                // SAFETY: 8 <= s < s_limit after a match; the table as above.
                let m2_hash = hash8(unsafe { load64(src, s - 2) }, table_bits) as usize;
                let x = unsafe { load64(src, s) };
                let curr_hash = hash8(x, table_bits) as usize;
                candidate = table[curr_hash].get();
                table[m2_hash] = E::new(s - 2);
                table[curr_hash] = E::new(s);
                if x != unsafe { load64(src, candidate) } {
                    cv = unsafe { load64(src, s + 1) };
                    s += 1;
                    break;
                }
                repeat = s - candidate;
                base = s;
                s += 8;
                candidate += 8;
                while s + 8 <= src.len() {
                    // SAFETY: candidate < s.
                    let diff = unsafe { load64(src, s) ^ load64(src, candidate) };
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
