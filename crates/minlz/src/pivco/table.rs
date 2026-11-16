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

use super::kernel::{
    KIND_BOTH, KIND_FLAT, KIND_FULL, KIND_LEAF_LEFT, R2S_LEN, Rec, load64, store64,
};
use super::{ERR_CODE_LEN, ERR_KRAFT, ERR_NO_SYMBOLS, Error, MAX_CODE_LEN};

/// Code lengths and the coding tree derived from them. The tree is a pure function of the
/// lengths, so encoder and decoder agree on it.
///
/// The default value is invalid. A built Table may be cloned and shared by many Encoders
/// and Decoders.
#[derive(Clone, Debug)]
pub struct Table {
    pub(super) lens: [u8; 256],
    /// Symbol to rank; 0xFF for unused symbols when nranks < 256.
    pub(super) s2r: [u8; 256],
    /// Rank to symbol, zero padded.
    pub(super) r2s: [u8; R2S_LEN],
    /// Pre-order schedule.
    pub(super) sched: [Rec; 255],
    /// 0 for an invalid table.
    pub(super) nsched: u16,
    /// Leaves; 2 for a single-symbol table.
    pub(super) nranks: u16,
}

impl Default for Table {
    fn default() -> Self {
        Table {
            lens: [0; 256],
            s2r: [0; 256],
            r2s: [0; R2S_LEN],
            sched: [Rec::default(); 255],
            nsched: 0,
            nranks: 0,
        }
    }
}

/// 2^bit same-length symbols under one subtree root at depth.
#[derive(Clone, Copy, Default)]
struct Chunk {
    depth: u8,
    bit: u8,
    idx: u8,
}

/// An internal node of the build walk, waiting for its right child.
#[derive(Clone, Copy, Default)]
struct Open {
    this: usize,
    lrank: usize,
    rrec: usize,
    rrank: usize,
    depth: u8,
    right: bool,
}

impl Table {
    /// Builds the table from code lengths, which must be 0..=[`MAX_CODE_LEN`] and form a
    /// complete prefix code. A lone used symbol may have any length 1..=MAX_CODE_LEN.
    /// On error the table is invalid.
    pub fn set_lengths(&mut self, lens: &[u8; 256]) -> Result<(), Error> {
        self.nsched = 0;
        // build reads lengths as 4 bits.
        let or = lens
            .as_chunks::<8>()
            .0
            .iter()
            .fold(0, |a, c| a | load64(c, 0));
        if or & 0xF0F0F0F0F0F0F0F0 != 0 {
            return Err(ERR_CODE_LEN);
        }
        self.lens = *lens;
        self.build()
    }

    /// Reports whether the table is built.
    #[inline]
    pub fn valid(&self) -> bool {
        self.nsched != 0
    }

    /// Returns the code lengths.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    pub fn lens(&self) -> &[u8; 256] {
        &self.lens
    }

    /// Returns the schedule records.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    pub fn recs(&self) -> &[Rec] {
        &self.sched[..usize::from(self.nsched)]
    }

    /// Returns the zero padded rank-to-symbol table.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    pub fn r2s(&self) -> &[u8; R2S_LEN] {
        &self.r2s
    }

    /// Returns the symbol-to-rank table; 0xFF for unused symbols when there are fewer
    /// than 256 ranks.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    pub fn s2r(&self) -> &[u8; 256] {
        &self.s2r
    }

    /// Returns the number of ranks (leaves); 2 for a single-symbol table.
    #[cfg_attr(not(feature = "unstable-internals"), allow(dead_code))]
    pub fn nranks(&self) -> usize {
        usize::from(self.nranks)
    }

    /// Returns the encoded size of a block with `h`, the counts of symbols 0 to h.len()-1,
    /// at most 256, which must sum to 1..=[`MAX_BLOCK_SIZE`](super::MAX_BLOCK_SIZE).
    /// It returns `None` if the table is not built or a counted symbol has no code.
    pub fn block_size(&self, h: &[u64]) -> Option<usize> {
        if !self.valid() {
            return None;
        }
        // Every record codes a range of ranks, so prefix sums of the rank counts give the
        // symbol count of each node.
        let mut cum = [0u64; 257];
        let (lens, s2r) = (&self.lens[..h.len()], &self.s2r[..h.len()]);
        for (s, &f) in h.iter().enumerate() {
            if f != 0 {
                if lens[s] == 0 {
                    return None;
                }
                cum[usize::from(s2r[s]) + 1] += f;
            }
        }
        let c = &mut cum[..usize::from(self.nranks) + 1];
        for r in 1..c.len() {
            c[r] += c[r - 1];
        }
        Some(self.node_size(&cum, 0, 0, usize::from(self.nranks)))
    }

    /// Returns the size of the regions the encoder walk writes for record rec, which codes
    /// the ranks lo..hi-1.
    fn node_size(&self, cum: &[u64; 257], rec: usize, lo: usize, hi: usize) -> usize {
        let n = (cum[hi] - cum[lo]) as usize;
        if n == 0 {
            return 0;
        }
        let r = self.sched[rec];
        let bm = (n + 7) >> 3;
        match r.kind() {
            KIND_FLAT => (n * usize::from(r.flat_depth()) + 7) >> 3,
            KIND_BOTH => bm,
            KIND_LEAF_LEFT => 2 + bm + self.node_size(cum, rec + 1, lo + 1, hi),
            _ => {
                let mid = usize::from(r.param) + 1;
                2 + bm
                    + self.node_size(cum, rec + 1, lo, mid)
                    + self.node_size(cum, rec + usize::from(r.right), mid, hi)
            }
        }
    }

    /// Ports upstream `pivco_build_decode_core` and the tail of `pivco_build_codec_table`
    /// (`src/huffman_table.c`) on the lengths, which must all be below 16.
    fn build(&mut self) -> Result<(), Error> {
        // Symbols are counted and placed in 8 interleaved segments of the symbol range (C
        // uses 4 quarters): neighboring symbols often share a length, and every counter or
        // cursor update waits for the previous one of its segment and length.
        const Q: usize = 32;
        let l = &self.lens;
        let mut h = [[0u8; 16]; 8];
        for i in 0..Q {
            h[0][usize::from(l[i] & 15)] += 1;
            h[1][usize::from(l[Q + i] & 15)] += 1;
            h[2][usize::from(l[2 * Q + i] & 15)] += 1;
            h[3][usize::from(l[3 * Q + i] & 15)] += 1;
            h[4][usize::from(l[4 * Q + i] & 15)] += 1;
            h[5][usize::from(l[5 * Q + i] & 15)] += 1;
            h[6][usize::from(l[6 * Q + i] & 15)] += 1;
            h[7][usize::from(l[7 * Q + i] & 15)] += 1;
        }
        let mut count = [0u16; MAX_CODE_LEN + 1];
        let mut base = [0u8; 16]; // first item of each length
        let (mut n, mut kraft) = (0usize, 0usize);
        for ln in 1..=MAX_CODE_LEN {
            let c: usize = h.iter().map(|s| usize::from(s[ln])).sum();
            base[ln] = n as u8;
            count[ln] = c as u16;
            n += c;
            kraft += c << (MAX_CODE_LEN - ln);
        }
        let zeros: usize = h.iter().map(|s| usize::from(s[0])).sum();
        if n + zeros != 256 {
            return Err(ERR_CODE_LEN); // lengths 12..15
        }
        if n == 0 {
            return Err(ERR_NO_SYMBOLS);
        }
        // Cursors into items: per length in ascending symbol order, unused symbols last.
        // Segment k's cursors are base plus the counts of segments 0..k-1, added bytewise. A
        // cursor only reaches 256, and carries into the next length's, when no symbol from
        // segment k on has its length and none is longer, so only unused cursors go wrong.
        // Lane 0 sums zero counts only (at most 224); the zero cursors are set apart.
        let mut cur = [[0u8; 16]; 8];
        let (mut b0, mut b1, mut z) = (load64(&base, 0), load64(&base, 8), n as u8);
        for (c, hk) in cur.iter_mut().zip(&h) {
            store64(c, 0, b0);
            store64(c, 8, b1);
            c[0] = z;
            z = z.wrapping_add(hk[0]);
            b0 = b0.wrapping_add(load64(hk, 0));
            b1 = b1.wrapping_add(load64(hk, 8));
        }
        let mut items = [0u8; 256];
        for i in 0..Q {
            let ln = |s: usize| usize::from(l[s * Q + i] & 15);
            let (a, b, c, d) = (ln(0), ln(1), ln(2), ln(3));
            // One load per cursor: the compiler would reload it after the items store.
            let (pa, pb, pc, pd) = (cur[0][a], cur[1][b], cur[2][c], cur[3][d]);
            items[usize::from(pa)] = i as u8;
            items[usize::from(pb)] = (Q + i) as u8;
            items[usize::from(pc)] = (2 * Q + i) as u8;
            items[usize::from(pd)] = (3 * Q + i) as u8;
            cur[0][a] = pa.wrapping_add(1);
            cur[1][b] = pb.wrapping_add(1);
            cur[2][c] = pc.wrapping_add(1);
            cur[3][d] = pd.wrapping_add(1);
            let (a, b, c, d) = (ln(4), ln(5), ln(6), ln(7));
            let (pa, pb, pc, pd) = (cur[4][a], cur[5][b], cur[6][c], cur[7][d]);
            items[usize::from(pa)] = (4 * Q + i) as u8;
            items[usize::from(pb)] = (5 * Q + i) as u8;
            items[usize::from(pc)] = (6 * Q + i) as u8;
            items[usize::from(pd)] = (7 * Q + i) as u8;
            cur[4][a] = pa.wrapping_add(1);
            cur[5][b] = pb.wrapping_add(1);
            cur[6][c] = pc.wrapping_add(1);
            cur[7][d] = pd.wrapping_add(1);
        }

        self.s2r = [0xFF; 256];
        if n == 1 {
            let s = items[0];
            self.sched[0] = Rec {
                op: KIND_BOTH,
                param: 0,
                right: 0,
            };
            self.r2s = [0; R2S_LEN];
            self.r2s[..2].fill(s);
            self.s2r[usize::from(s)] = 0;
            (self.nranks, self.nsched) = (2, 1);
            return Ok(());
        }
        if kraft != 1 << MAX_CODE_LEN {
            return Err(ERR_KRAFT);
        }

        // Each length splits into chunks by the set bits of its count, largest first,
        // taking that length's symbols in order. A count of at most 256 has at most 8 set bits.
        let mut chunks = [Chunk::default(); MAX_CODE_LEN * 8];
        let (mut nc, mut idx) = (0, 0usize);
        for ln in 1..=MAX_CODE_LEN as u8 {
            let mut c = count[usize::from(ln)];
            while c != 0 {
                let b = (15 - c.leading_zeros()) as u8;
                let d = if b > 0 { ln - b } else { ln };
                chunks[nc] = Chunk {
                    depth: d,
                    bit: b,
                    idx: idx as u8,
                };
                nc += 1;
                idx += 1 << b;
                c &= !(1 << b);
            }
        }
        // Stable sort by depth; generation order is already nearly sorted.
        for i in 1..nc {
            let x = chunks[i];
            let mut j = i;
            while j > 0 && chunks[j - 1].depth > x.depth {
                chunks[j] = chunks[j - 1];
                j -= 1;
            }
            chunks[j] = x;
        }

        // C dc_gen as a loop: a pre-order walk over the depth-sorted chunks, where each
        // open internal node waits on the stack for its right child. A complete code
        // guarantees that the next chunk is never shallower than the current node.
        let mut stack = [Open::default(); MAX_CODE_LEN];
        let (mut sp, mut nrec, mut rank, mut depth) = (0, 0, 0usize, 0u8);
        for c in &chunks[..nc] {
            while depth < c.depth {
                stack[sp] = Open {
                    this: nrec,
                    lrank: rank,
                    depth,
                    ..Open::default()
                };
                sp += 1;
                nrec += 1;
                depth += 1;
            }
            if c.bit == 0 {
                let s = items[usize::from(c.idx)]; // a leaf has no record
                self.r2s[rank] = s;
                self.s2r[usize::from(s)] = rank as u8;
                rank += 1;
            } else {
                let op = if c.bit > 1 {
                    KIND_FLAT | c.bit << 2
                } else {
                    KIND_BOTH
                };
                self.sched[nrec] = Rec {
                    op,
                    param: rank as u8,
                    right: 0,
                };
                nrec += 1;
                let syms = &items[usize::from(c.idx)..][..1 << c.bit];
                self.r2s[rank..rank + syms.len()].copy_from_slice(syms);
                for (i, &s) in syms.iter().enumerate() {
                    self.s2r[usize::from(s)] = (rank + i) as u8;
                }
                rank += syms.len();
            }
            while sp > 0 {
                let f = &mut stack[sp - 1];
                if !f.right {
                    (f.right, f.rrec, f.rrank) = (true, nrec, rank);
                    depth = f.depth + 1;
                    break;
                }
                // Only a left child can be a lone leaf: depth d's only singleton chunk is
                // length d's bit-0 chunk, which sorts first among depth-d chunks.
                let op = if f.rrec == f.this + 1 && f.rrank == f.lrank + 1 {
                    KIND_LEAF_LEFT
                } else {
                    KIND_FULL
                };
                self.sched[f.this] = Rec {
                    op,
                    param: (f.rrank - 1) as u8,
                    right: (f.rrec - f.this) as u8,
                };
                sp -= 1;
            }
        }
        self.r2s[rank..].fill(0);
        (self.nranks, self.nsched) = (rank as u16, nrec as u16);
        Ok(())
    }
}
