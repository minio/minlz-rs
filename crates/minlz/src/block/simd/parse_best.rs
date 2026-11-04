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

//! The parser of LevelSmallestSIMD, a port of `(*simdEncoder).encodeBlockBest` of Go's
//! `encode_l3.go`.

use super::MAX_LEN;
use super::encode::{BLOCK_BITS, Encoder, TABLE_BITS, match_back};
use crate::block::hash::{hash4, hash8};
use crate::block::load_store::{load8, load32, load64};
use crate::block::table_pool::{self, U64_18, U64_20};

/// Long hash matches.
const L_TABLE_BITS: u32 = 20;
/// Short hash matches.
const S_TABLE_BITS: u32 = 18;
const INPUT_MARGIN: usize = 8 + 2;
/// Estimated sizes in 1/16 bits.
const ESCAPE_BITS: isize = 6 * 16;
const MAX_SKIP: usize = 64;

#[inline(always)]
fn get_cur(x: u64) -> usize {
    (x & 0xffff_ffff) as usize
}

#[inline(always)]
fn get_prev(x: u64) -> usize {
    (x >> 32) as usize
}

/// A match candidate: it starts at s and copies from offset.
#[derive(Clone, Copy, Default)]
struct Match {
    offset: usize,
    s: usize,
    length: usize,
    score: isize,
    rep: bool,
    nextrep: bool,
}

/// What Go's closures of encodeBlockBest capture: the costs, and the best match so far.
struct Scorer<'s> {
    src: &'s [u8],
    s_limit: usize,
    next_emit: usize,
    lit_bits: isize,
    token_bits: isize,
    symbol_bits: isize,
    ll_bits: isize,
    /// simdSplitCost && e.noOverlap, which the parse can't change, as Smallest has no budget.
    split_cost: bool,
    best: Match,
}

impl Scorer<'_> {
    /// extend updates best with the match at offset, whose first n bytes the caller has checked.
    /// The checks are left to the inlined callers, since they reject most candidates.
    ///
    /// # Safety
    /// `offset < s` and `s + n <= self.src.len()` must hold.
    unsafe fn extend(&mut self, offset: usize, s: usize, n: usize, rep: bool) {
        let src = self.src;
        debug_assert!(offset < s && s + n <= src.len());
        let (mut p, mut end) = (s + n, offset + n);
        // SAFETY: end < p in the loop.
        while p < src.len() {
            if src.len() - p < 8 {
                if unsafe { load8(src, p) == load8(src, end) } {
                    end += 1;
                    p += 1;
                    continue;
                }
                break;
            }
            let diff = unsafe { load64(src, p) ^ load64(src, end) };
            if diff != 0 {
                end += diff.trailing_zeros() as usize >> 3;
                break;
            }
            p += 8;
            end += 8;
        }
        // SAFETY: offset < s, and s is at least next_emit.
        let back = unsafe { match_back(src, s, offset, offset.min(s - self.next_emit)) };
        let (m_offset, m_s) = (offset - back, s - back);
        let length = end - m_offset;

        // Bigger score is better.
        // -lit_bits*m_s is the base cost of the literals before the match.
        let mut score = self.lit_bits * (length as isize - m_s as isize) - self.token_bits;
        if m_s - self.next_emit > MAX_LEN {
            score -= ESCAPE_BITS;
        }
        if m_s > self.next_emit {
            score -= self.ll_bits;
        }
        let dist = m_s - m_offset;
        if !rep {
            let raw = (usize::BITS - dist.leading_zeros()).saturating_sub(4);
            score -= self.symbol_bits + 16 * raw as isize;
        }
        if self.split_cost && dist < length && dist < MAX_LEN {
            // emitLong splits the match in pieces of the offset, which doubles while more than a
            // piece is left.
            let (mut o, mut c) = (dist, dist);
            while c < length && o < MAX_LEN {
                if length - c > o {
                    o *= 2;
                    score -= self.symbol_bits;
                }
                score -= self.token_bits;
                c += o;
            }
        } else if dist < MAX_LEN {
            score -= self.token_bits * ((length - 1) / MAX_LEN) as isize;
        } else if length > MAX_LEN {
            score -= ESCAPE_BITS;
        }
        if !rep && score <= -self.lit_bits * m_s as isize {
            // Eliminate if no savings, we might find a better one.
            return;
        }
        let mut nextrep = false;
        if m_s + length < self.s_limit {
            const CHECKOFF: usize = 1;
            // SAFETY: m_offset < m_s, and m_s + length + CHECKOFF + 4 < s_limit + INPUT_MARGIN.
            nextrep = unsafe {
                load32(src, m_s + length + CHECKOFF) == load32(src, m_offset + length + CHECKOFF)
            };
        }

        let b = &self.best;
        if b.length != 0 {
            let keep = if b.score != score {
                b.score > score
            } else if b.s != m_s {
                // Pick whichever starts the earliest,
                // we can probably find a match right away
                b.s < m_s
            } else if b.nextrep != nextrep {
                // If one is a good repeat candidate, pick it.
                b.nextrep
            } else {
                // Pick the smallest distance offset.
                b.offset > m_offset
            };
            if keep {
                return;
            }
        }
        self.best = Match {
            offset: m_offset,
            s: m_s,
            length,
            score,
            rep,
            nextrep,
        };
    }

    /// # Safety
    /// `s + 4 <= self.src.len()` must hold.
    #[inline(always)]
    unsafe fn match_at(&mut self, offset: usize, s: usize, first: u32) {
        // Don't retest if we have the same offset.
        let b = &self.best;
        // SAFETY: offset < s.
        if s > offset
            && (b.length == 0 || b.s - b.offset != s - offset)
            && unsafe { load32(self.src, offset) } == first
        {
            unsafe { self.extend(offset, s, 4, false) };
        }
    }

    /// # Safety
    /// `offset < s` and `s + 4 <= self.src.len()` must hold.
    #[inline(always)]
    unsafe fn match_at_repeat(&mut self, offset: usize, s: usize, first: u32) {
        // 2 gives close to no improvement,
        // since it may just give 'literal -> len 2 repeat -> literal' section.
        // which eats up the gains in overhead.
        // 3 gives pretty consistent improvement
        const CHECKBYTES: usize = 3;
        const MASK: u32 = (1 << (8 * CHECKBYTES)) - 1;
        // Don't retest if we already have a repeat
        // SAFETY: offset < s.
        if !self.best.rep && unsafe { load32(self.src, offset) } & MASK == first & MASK {
            unsafe { self.extend(offset, s, CHECKBYTES, true) };
        }
    }
}

impl<'a> Encoder<'a> {
    /// encodeBlockBest parses src like the function encodeBlockBest, without its size and offset
    /// limits, scoring matches by their estimated size in a SIMD block.
    pub(super) fn encode_block_best(&mut self, src: &'a [u8]) {
        let mut l_table = table_pool::get(&U64_20);
        let mut s_table = table_pool::get(&U64_18);
        self.parse_best(src, &mut l_table, &mut s_table);
        table_pool::put(&U64_20, l_table);
        table_pool::put(&U64_18, s_table);
    }

    /// The body of encode_block_best.
    ///
    /// Its loads are unchecked, as Go's: src has more than MIN_SAVING bytes, s stays below
    /// s_limit, the zeroed tables hold 0 or positions below s, and repeat sources are below s.
    /// The Scorer checks that candidates are below the position they match.
    // Not inlined, see parse_fast.
    #[inline(never)]
    fn parse_best(
        &mut self,
        src: &'a [u8],
        l_table: &mut [u64; 1 << L_TABLE_BITS],
        s_table: &mut [u64; 1 << S_TABLE_BITS],
    ) {
        let (mut token_bits, mut symbol_bits, mut ll_bits) = (4 * 16, 7 * 16, 0);
        // Up to 2 KB the operation streams are stored at 8 bits a value, and a match that splits
        // a literal run adds a literal length. Slightly lower costs than that measure best.
        if src.len() <= 2 << 10 {
            (token_bits, symbol_bits, ll_bits) = (7 * 16, 8 * 16, 6 * 16);
        }
        // Literals cost the mean code length of src, which fits the data better than a fixed
        // cost.
        let mut lit_bits = 8 * 16;
        let n = self.lit_bits(src);
        if n + BLOCK_BITS + TABLE_BITS < 8 * src.len() {
            lit_bits = (16 * n / src.len()) as isize;
        }

        let s_limit = src.len() - INPUT_MARGIN;
        let mut m = Scorer {
            src,
            s_limit,
            next_emit: 0,
            lit_bits,
            token_bits,
            symbol_bits,
            ll_bits,
            split_cost: self.knobs.split_cost && self.no_overlap,
            best: Match::default(),
        };
        let mut s = 1;
        let mut repeat = 1;
        // SAFETY: src.len() > MIN_SAVING.
        let mut cv = unsafe { load64(src, s) };

        'outer: loop {
            m.best = Match::default();
            loop {
                let mut next_s = ((s - m.next_emit) >> 8) + 1;
                if next_s > MAX_SKIP {
                    next_s = s + MAX_SKIP;
                } else {
                    next_s += s;
                }
                if next_s > s_limit {
                    break 'outer;
                }
                let hash_l = hash8(cv, L_TABLE_BITS) as usize;
                let hash_s = hash4(cv, S_TABLE_BITS) as usize;
                let candidate_l = l_table[hash_l];
                let candidate_s = s_table[hash_s];

                // SAFETY (to the loads and matches before the tables are updated): s < next_s <=
                // s_limit, so 8 bytes can be loaded at s + 2. The repeat offset is at most the
                // output, so at most s.
                unsafe {
                    m.match_at(get_cur(candidate_l), s, cv as u32);
                    m.match_at(get_prev(candidate_l), s, cv as u32);
                    m.match_at(get_cur(candidate_s), s, cv as u32);
                    m.match_at(get_prev(candidate_s), s, cv as u32);
                    m.match_at_repeat(s - repeat, s, cv as u32);
                    m.match_at_repeat(s - repeat + 1, s + 1, (cv >> 8) as u32);
                }

                if m.best.length > 0 {
                    let next_short = s_table[hash4(cv >> 8, S_TABLE_BITS) as usize];
                    let mut s_fwd = s + 1;
                    let mut cv = unsafe { load64(src, s_fwd) };
                    let next_long = l_table[hash8(cv, L_TABLE_BITS) as usize];
                    unsafe {
                        m.match_at(get_cur(next_short), s_fwd, cv as u32);
                        m.match_at(get_prev(next_short), s_fwd, cv as u32);
                        m.match_at(get_cur(next_long), s_fwd, cv as u32);
                        m.match_at(get_prev(next_long), s_fwd, cv as u32);
                    }

                    s_fwd += 1;
                    cv = unsafe { load64(src, s_fwd) };
                    let next_long = l_table[hash8(cv, L_TABLE_BITS) as usize];

                    unsafe { m.match_at_repeat(s_fwd - repeat, s_fwd, cv as u32) };
                    let next_short = s_table[hash4(cv, S_TABLE_BITS) as usize];
                    unsafe {
                        m.match_at(get_cur(next_short), s_fwd, cv as u32);
                        m.match_at(get_prev(next_short), s_fwd, cv as u32);
                        m.match_at(get_cur(next_long), s_fwd, cv as u32);
                        m.match_at(get_prev(next_long), s_fwd, cv as u32);
                    }

                    // Search for a match at best match end, see if that is better.
                    // Allow some bytes at the beginning to mismatch.
                    // Sweet spot is around 1-2 bytes, but depends on input.
                    // The skipped bytes are tested in Extend backwards,
                    // and still picked up as part of the match if they do.
                    const SKIP_BEGINNING: usize = 2;
                    const SKIP_END: usize = 1;
                    let s_at = m.best.s + m.best.length - SKIP_END;
                    if s_at < s_limit {
                        let s_back = m.best.s + SKIP_BEGINNING - SKIP_END;
                        let back_l = m.best.length - SKIP_BEGINNING;
                        // SAFETY: s_back < s_at < s_limit.
                        cv = unsafe { load64(src, s_back) };
                        let at = unsafe { load64(src, s_at) };

                        let next = l_table[hash8(at, L_TABLE_BITS) as usize];
                        // Go's checkAt > 0.
                        if get_cur(next) > back_l {
                            unsafe { m.match_at(get_cur(next) - back_l, s_back, cv as u32) };
                        }
                        if get_prev(next) > back_l {
                            unsafe { m.match_at(get_prev(next) - back_l, s_back, cv as u32) };
                        }
                        // Quite small gain, but generally a benefit on very compressible material.
                        let next = s_table[hash4(at, S_TABLE_BITS) as usize];
                        if get_cur(next) > back_l {
                            unsafe { m.match_at(get_cur(next) - back_l, s_back, cv as u32) };
                        }
                        if get_prev(next) > back_l {
                            unsafe { m.match_at(get_prev(next) - back_l, s_back, cv as u32) };
                        }
                    }
                }

                l_table[hash_l] = s as u64 | candidate_l << 32;
                s_table[hash_s] = s as u64 | candidate_s << 32;

                if m.best.length > 0 {
                    break;
                }

                // SAFETY: next_s <= s_limit.
                cv = unsafe { load64(src, next_s) };
                s = next_s;
            }
            let start_idx = s + 1;
            let best = m.best;
            s = best.s;
            let base = s;
            let mut offset = s - best.offset;
            s += best.length;
            let mut n = best.length;
            if offset < MAX_LEN && n > 2 * MAX_LEN {
                n = MAX_LEN;
            }
            if best.rep {
                self.emit_repeat_lits(&src[m.next_emit..base], n);
            } else {
                self.emit_copy_lits(&src[m.next_emit..base], offset, n);
            }
            if n < best.length {
                // Matches over 32 bytes need an offset of at least 32. A multiple of a shorter
                // period is one, so the rest takes operations of up to 288 bytes instead of 32.
                offset *= MAX_LEN.div_ceil(offset);
                self.emit_copy(offset, best.length - n);
            }
            repeat = self.rep;

            m.next_emit = s;
            if s >= s_limit {
                break 'outer;
            }
            for i in start_idx..s {
                // SAFETY: i < s < s_limit.
                let cv0 = unsafe { load64(src, i) };
                let long0 = hash8(cv0, L_TABLE_BITS) as usize;
                let short0 = hash4(cv0, S_TABLE_BITS) as usize;
                l_table[long0] = i as u64 | l_table[long0] << 32;
                s_table[short0] = i as u64 | s_table[short0] << 32;
            }
            // SAFETY: s < s_limit.
            cv = unsafe { load64(src, s) };
        }

        if m.next_emit < src.len() {
            self.emit_literals(&src[m.next_emit..]);
        }
    }
}
