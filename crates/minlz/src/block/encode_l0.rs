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

//! L0 / SuperFast encoder.
//!
//! Ports `encode_l0.go:encodeFastBlockGo` (for inputs > 64 KiB) and
//! `encode_l0.go:encodeFastBlockGo64K` (for inputs ≤ 64 KiB).

use super::emit::{emit_copy, emit_copy_lits2, emit_copy_lits3, emit_literal, emit_repeat};
use super::format::{
    INPUT_MARGIN, MAX_COPY2_LITS, MAX_COPY2_OFFSET, MAX_COPY3_LITS, MAX_COPY3_OFFSET,
    MIN_COPY2_OFFSET, MIN_NON_LITERAL_BLOCK_SIZE,
};
use super::hash::hash8;
use super::load_store::{load32, load64};
use super::table_pool::{self, U32_13};

/// Entry point: dispatches on `src.len()`.  Returns the number of bytes
/// written into `dst` or `0` if the block is incompressible.
pub(super) fn encode_block(dst: &mut [u8], src: &[u8]) -> usize {
    if src.len() < MIN_NON_LITERAL_BLOCK_SIZE {
        0
    } else if src.len() <= 65536 {
        encode_block_64k(dst, src)
    } else {
        encode_block_big(dst, src)
    }
}

/// SuperFast encoder for inputs > 64 KiB (port of `encodeFastBlockGo`).
fn encode_block_big(dst: &mut [u8], src: &[u8]) -> usize {
    let mut table = table_pool::get(&U32_13);
    let n = encode_block_big_with(dst, src, &mut table);
    table_pool::put(&U32_13, table);
    n
}

fn encode_block_big_with(dst: &mut [u8], src: &[u8], table: &mut [u32; 1 << 13]) -> usize {
    const TABLE_BITS: u32 = 13;
    const SKIP_LOG: u32 = 5;

    let s_limit = src.len() - INPUT_MARGIN;
    let dst_limit = src.len() - (src.len() >> 3) - 6;
    let mut next_emit = 0;
    let mut s = 1;
    // SAFETY: src.len() > 64 KiB.
    let mut cv = unsafe { load64(src, s) };
    let mut repeat = 1;
    let mut d = 0;

    'outer: loop {
        let mut candidate;
        loop {
            let next_s = s + ((s - next_emit) >> SKIP_LOG) + 5;
            if next_s > s_limit {
                break 'outer;
            }
            let min_src_pos = s as isize - MAX_COPY3_OFFSET as isize;
            let hash0 = hash8(cv, TABLE_BITS) as usize;
            // SAFETY: s + 5 ≤ next_s ≤ s_limit.
            let cv1 = unsafe { load64(src, s + 1) };
            let hash1 = hash8(cv1, TABLE_BITS) as usize;
            candidate = table[hash0] as usize;
            let candidate2 = table[hash1] as usize;
            table[hash0] = s as u32;
            table[hash1] = (s + 1) as u32;
            // SAFETY: as cv1.
            let cv2 = unsafe { load64(src, s + 2) };
            let hash2 = hash8(cv2, TABLE_BITS) as usize;

            // SAFETY: as cv1; repeat is a past offset, so at most s.
            if cv1 as u32 == unsafe { load32(src, s - repeat + 1) } {
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
                    // SAFETY: cand < s ≤ s_limit.
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
                // SAFETY: s < s_limit.
                cv = unsafe { load64(src, s) };
                continue;
            }

            // SAFETY for the loads below: the table holds 0 or positions ≤ s + 2 < s_limit.
            if candidate as isize >= min_src_pos && cv == unsafe { load64(src, candidate) } {
                break;
            }
            candidate = table[hash2] as usize;
            if candidate2 as isize > min_src_pos && cv1 == unsafe { load64(src, candidate2) } {
                table[hash2] = (s + 2) as u32;
                candidate = candidate2;
                s += 1;
                break;
            }
            table[hash2] = (s + 2) as u32;
            if candidate as isize >= min_src_pos + 2 && cv2 == unsafe { load64(src, candidate) } {
                s += 2;
                break;
            }
            // SAFETY: next_s ≤ s_limit.
            cv = unsafe { load64(src, next_s) };
            s = next_s;
        }

        // No backward extension: Go's loop for it is disabled (`for false && ...`).
        let base = s;
        repeat = base - candidate;
        s += 8;
        candidate += 8;
        while s + 8 <= src.len() {
            // SAFETY: candidate < s.
            let diff = unsafe { load64(src, s) ^ load64(src, candidate) };
            if diff != 0 {
                s += (diff.trailing_zeros() as usize) >> 3;
                break;
            }
            s += 8;
            candidate += 8;
        }
        let length = s - base;

        if next_emit != base {
            if base - next_emit > MAX_COPY3_LITS || repeat < MIN_COPY2_OFFSET {
                if d + (base - next_emit) > dst_limit {
                    return 0;
                }
                d += emit_literal(&mut dst[d..], &src[next_emit..base]);
                d += emit_copy(&mut dst[d..], repeat, length);
            } else if repeat <= MAX_COPY2_OFFSET {
                d += emit_copy_lits2(&mut dst[d..], &src[next_emit..base], repeat, length);
            } else {
                d += emit_copy_lits3(&mut dst[d..], &src[next_emit..base], repeat, length);
            }
        } else {
            d += emit_copy(&mut dst[d..], repeat, length);
        }

        loop {
            next_emit = s;
            if s >= s_limit {
                break 'outer;
            }
            if d > dst_limit {
                return 0;
            }
            // SAFETY: 9 ≤ s < s_limit after a match.
            let m2_hash = hash8(unsafe { load64(src, s - 2) }, TABLE_BITS) as usize;
            let x = unsafe { load64(src, s) };
            let curr_hash = hash8(x, TABLE_BITS) as usize;
            candidate = table[curr_hash] as usize;
            table[m2_hash] = (s - 2) as u32;
            table[curr_hash] = s as u32;
            // SAFETY: the table holds 0 or positions below s.
            if s - candidate > MAX_COPY3_OFFSET || x != unsafe { load64(src, candidate) } {
                // SAFETY: s < s_limit.
                cv = unsafe { load64(src, s + 1) };
                s += 1;
                break;
            }
            repeat = s - candidate;
            let base = s;
            s += 8;
            candidate += 8;
            while s + 8 <= src.len() {
                // SAFETY: candidate < s.
                let diff = unsafe { load64(src, s) ^ load64(src, candidate) };
                if diff != 0 {
                    s += (diff.trailing_zeros() as usize) >> 3;
                    break;
                }
                s += 8;
                candidate += 8;
            }
            d += emit_copy(&mut dst[d..], repeat, s - base);
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

/// SuperFast encoder for inputs ≤ 64 KiB (port of `encodeFastBlockGo64K`).
fn encode_block_64k(dst: &mut [u8], src: &[u8]) -> usize {
    const TABLE_BITS: u32 = 12;
    const SKIP_LOG: u32 = 4;

    // Go's dstLimit is negative below 34 bytes. Every path then fails a dstLimit check: the
    // first copy has literals before it and an offset below MIN_COPY2_OFFSET.
    let Some(dst_limit) = (src.len() - (src.len() >> 4)).checked_sub(32) else {
        return 0;
    };
    let mut table = [0u16; 1 << TABLE_BITS];
    let s_limit = src.len() - INPUT_MARGIN;
    let mut next_emit = 0;
    let mut s = 1;
    // SAFETY: src.len() ≥ MIN_NON_LITERAL_BLOCK_SIZE = 16 ⇒ s + 8 ≤ src.len().
    let mut cv = unsafe { load64(src, s) };
    let mut repeat = 1;
    let mut d = 0;

    'outer: loop {
        let mut candidate;
        loop {
            let next_s = s + ((s - next_emit) >> SKIP_LOG) + 4;
            if next_s > s_limit {
                break 'outer;
            }
            let hash0 = hash8(cv, TABLE_BITS) as usize;
            // SAFETY: s + 4 ≤ next_s ≤ s_limit.
            let cv1 = unsafe { load64(src, s + 1) };
            let hash1 = hash8(cv1, TABLE_BITS) as usize;
            candidate = table[hash0] as usize;
            let candidate2 = table[hash1] as usize;
            table[hash0] = s as u16;
            table[hash1] = (s + 1) as u16;
            // SAFETY: as cv1.
            let cv2 = unsafe { load64(src, s + 2) };
            let hash2 = hash8(cv2, TABLE_BITS) as usize;

            // SAFETY: as cv1; repeat is a past offset, so at most s.
            if (cv >> 8) as u32 == unsafe { load32(src, s - repeat + 1) } {
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
                    // SAFETY: cand < s ≤ s_limit.
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
                // SAFETY: s < s_limit.
                cv = unsafe { load64(src, s) };
                continue;
            }

            // SAFETY for the loads below: the table holds 0 or positions ≤ s + 2 < s_limit.
            if cv == unsafe { load64(src, candidate) } {
                break;
            }
            candidate = table[hash2] as usize;
            if cv1 == unsafe { load64(src, candidate2) } {
                table[hash2] = (s + 2) as u16;
                candidate = candidate2;
                s += 1;
                break;
            }
            table[hash2] = (s + 2) as u16;
            if cv2 == unsafe { load64(src, candidate) } {
                s += 2;
                break;
            }
            // SAFETY: next_s ≤ s_limit.
            cv = unsafe { load64(src, next_s) };
            s = next_s;
        }

        let base = s;
        repeat = base - candidate;
        s += 8;
        candidate += 8;
        while s + 8 <= src.len() {
            // SAFETY: candidate < s.
            let diff = unsafe { load64(src, s) ^ load64(src, candidate) };
            if diff != 0 {
                s += (diff.trailing_zeros() as usize) >> 3;
                break;
            }
            s += 8;
            candidate += 8;
        }
        let length = s - base;

        if next_emit != base {
            if base - next_emit > MAX_COPY2_LITS || repeat < MIN_COPY2_OFFSET {
                if d + (base - next_emit) > dst_limit {
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
            if d > dst_limit {
                return 0;
            }
            // SAFETY: 9 ≤ s < s_limit after a match.
            let m2_hash = hash8(unsafe { load64(src, s - 2) }, TABLE_BITS) as usize;
            let x = unsafe { load64(src, s) };
            let curr_hash = hash8(x, TABLE_BITS) as usize;
            candidate = table[curr_hash] as usize;
            table[m2_hash] = (s - 2) as u16;
            table[curr_hash] = s as u16;
            // SAFETY: the table holds 0 or positions below s.
            if x != unsafe { load64(src, candidate) } {
                // SAFETY: s < s_limit.
                cv = unsafe { load64(src, s + 1) };
                s += 1;
                break;
            }
            repeat = s - candidate;
            let base = s;
            s += 8;
            candidate += 8;
            while s + 8 <= src.len() {
                // SAFETY: candidate < s.
                let diff = unsafe { load64(src, s) ^ load64(src, candidate) };
                if diff != 0 {
                    s += (diff.trailing_zeros() as usize) >> 3;
                    break;
                }
                s += 8;
                candidate += 8;
            }
            d += emit_copy(&mut dst[d..], repeat, s - base);
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
