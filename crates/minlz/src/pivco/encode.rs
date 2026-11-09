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

use super::kernel::{HIST_SCRATCH, KIND_BOTH, KIND_FLAT, KIND_LEAF_LEFT, RANK_SLACK, Tier};
use super::{Error, MAX_BLOCK_SIZE, MAX_CODE_LEN, Table};

/// One bitmap staging slot per depth of internal nodes, 0..MAX_CODE_LEN.
const STAGE_SLOTS: usize = MAX_CODE_LEN;

/// The shortest input that [`Encoder::histogram`] counts with a kernel: a plain loop beats
/// the kernels' fixed cost below it.
pub(super) const HIST_DIRECT: usize = 512;

// Right ranks stack up along a path through the nodes that write them, FULL and LEAF_LEFT,
// which are at most MAX_CODE_LEN-2 deep: at most n per node.
const RS: usize = MAX_BLOCK_SIZE + RANK_SLACK;
const TS: usize = (MAX_CODE_LEN - 1) * MAX_BLOCK_SIZE + RANK_SLACK;
const BS: usize = MAX_BLOCK_SIZE.div_ceil(8) + RANK_SLACK;

/// The alignment of the coders' scratch, as Go's page-aligned allocations: the vector
/// kernels' 32-byte accesses otherwise split cache lines, which made AVX2 blocks up to
/// 1.5x slower.
pub(super) const SCRATCH_ALIGN: usize = 64;

/// Codes blocks. The default value is ready to use and allocates its scratch, sized for
/// [`MAX_BLOCK_SIZE`], on first use.
#[derive(Default)]
pub struct Encoder {
    pub(super) tier: Tier,
    /// Gather output and root partition input, then the right-child ranks of the walk,
    /// then the bitmap staging slots; empty until first use.
    slab: Vec<u8>,
    pub(super) hscratch: Vec<u8>,
}

impl Encoder {
    /// Returns an Encoder with its scratch allocated that runs the kernels of `tier`; for
    /// tests and benchmarks.
    pub fn with_tier(tier: Tier) -> Self {
        let mut e = Self::default();
        e.setup();
        e.tier = tier;
        e
    }

    fn setup(&mut self) {
        if !self.slab.is_empty() {
            return;
        }
        self.slab = vec![0; RS + TS + STAGE_SLOTS * BS + SCRATCH_ALIGN];
        self.tier = Tier::best();
    }

    /// Adds the byte counts of `src` to `hist`.
    pub fn histogram(&mut self, hist: &mut [u64; 256], src: &[u8]) {
        if src.len() < HIST_DIRECT {
            for &b in src {
                let h = &mut hist[usize::from(b)];
                *h = h.wrapping_add(1);
            }
            return;
        }
        self.setup();
        if self.hscratch.is_empty() && self.tier.hist_uses_scratch() {
            self.hscratch = vec![0; HIST_SCRATCH];
        }
        // The kernels count in u32.
        for s in src.chunks(u32::MAX as usize) {
            let mut c = [0u32; 256];
            self.tier.histogram(&mut c, s, &mut self.hscratch);
            for (h, &v) in hist.iter_mut().zip(&c) {
                *h = h.wrapping_add(u64::from(v));
            }
        }
    }

    /// Writes the block that codes `src` with `t` to `dst`, which must be exactly
    /// [`Table::block_size`] of the symbol counts of `src`; it panics if it is not.
    ///
    /// It returns [`Error::BlockSize`] unless 1 <= src.len() <= [`MAX_BLOCK_SIZE`],
    /// [`Error::InvalidTable`] if `t` is not built and [`Error::SymbolNotInTable`] if `src`
    /// has a symbol without a code.
    pub fn encode(&mut self, dst: &mut [u8], src: &[u8], t: &Table) -> Result<(), Error> {
        let n = src.len();
        if n == 0 || n > MAX_BLOCK_SIZE {
            return Err(Error::BlockSize);
        }
        if !t.valid() {
            return Err(Error::InvalidTable);
        }
        self.setup();
        let tier = self.tier;
        let off = self
            .slab
            .as_ptr()
            .align_offset(SCRATCH_ALIGN)
            .min(SCRATCH_ALIGN);
        let (ranks, rest) = self.slab[off..].split_at_mut(RS);
        let (tmp, stage) = rest.split_at_mut(TS);
        // Unused symbols map to rank 0xFF, which is a real rank only in a full table.
        if tier.gather(ranks, src, &t.s2r) == 0xFF && t.nranks < 256 {
            return Err(Error::SymbolNotInTable);
        }
        let mut w = EncWalk {
            tier,
            t,
            stage,
            out: dst,
            pos: 0,
        };
        w.node(0, &mut ranks[..n], tmp, 0);
        assert!(w.pos == w.out.len(), "pivco: dst is not the block size");
        Ok(())
    }
}

/// The state of one block's encode walk.
struct EncWalk<'a> {
    tier: Tier,
    t: &'a Table,
    stage: &'a mut [u8],
    /// Regions are written at out[pos..].
    out: &'a mut [u8],
    pos: usize,
}

impl EncWalk<'_> {
    /// Ports upstream `codec_encode_node` (`src/pivco_huffman_codec.c`): the K_right header
    /// in pre-order, the children larger first, the bitmap in post-order. `ranks` are the
    /// subtree's symbol ranks; `tmp` is scratch for the right ranks of this and all deeper
    /// nodes.
    fn node(&mut self, rec: usize, ranks: &mut [u8], tmp: &mut [u8], depth: usize) {
        let n = ranks.len();
        if n == 0 {
            return;
        }
        let (tier, r) = (self.tier, self.t.sched[rec]);
        let kind = r.kind();
        if kind == KIND_FLAT {
            let d = r.flat_depth();
            let tb = (n * usize::from(d)).div_ceil(8);
            tier.pack(&mut self.out[self.pos..self.pos + tb], ranks, d, r.param);
            self.pos += tb;
            return;
        }
        let bm = &mut self.stage[depth * BS..(depth + 1) * BS];
        let nr = match kind {
            KIND_BOTH => tier.part_none(ranks, bm, r.param),
            KIND_LEAF_LEFT => tier.part_right(ranks, tmp, bm, r.param),
            _ => tier.part_full(ranks, tmp, bm, r.param),
        };
        let nl = n - nr;
        if kind != KIND_BOTH {
            self.out[self.pos..self.pos + 2].copy_from_slice(&(nr as u16).to_le_bytes());
            self.pos += 2;
            // The right ranks are tmp[..nr]; deeper nodes stack theirs after them.
            let (rr, sub) = tmp.split_at_mut(nr);
            let right = rec + usize::from(r.right);
            if kind == KIND_LEAF_LEFT {
                self.node(rec + 1, rr, sub, depth + 1);
            } else if nr > nl {
                self.node(right, rr, sub, depth + 1);
                self.node(rec + 1, &mut ranks[..nl], sub, depth + 1);
            } else {
                self.node(rec + 1, &mut ranks[..nl], sub, depth + 1);
                self.node(right, rr, sub, depth + 1);
            }
        }
        let nb = n.div_ceil(8);
        self.out[self.pos..self.pos + nb].copy_from_slice(&self.stage[depth * BS..][..nb]);
        self.pos += nb;
    }
}
