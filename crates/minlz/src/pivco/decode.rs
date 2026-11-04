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

use super::encode::SCRATCH_ALIGN;
use super::kernel::{
    ARENA_SLACK, FLAG_DST, KIND_BOTH, KIND_FLAT, KIND_LEAF_LEFT, OP_CC, OP_COPY, OP_CV, OP_V512,
    OP_VV, Op, Program, Rec, Tier, load16, load64, vert_split,
};
use super::{
    ERR_GEOMETRY, ERR_MERGE, ERR_PADDING, ERR_SINGLE, ERR_SPLIT, ERR_TRAILING, ERR_TRUNCATED,
    Error, MAX_BLOCK_SIZE, Table,
};

/// Holds the children of any block: with K_right <= K checked, upstream's placement keeps
/// every buffer inside 2N.
const ARENA_SIZE: usize = 2 * MAX_BLOCK_SIZE + ARENA_SLACK;
/// One op per record plus two for each of at most 64 flats split by layout.
const MAX_OPS: usize = 384;

/// Decodes blocks. The default value is ready to use and allocates its scratch on first
/// use.
#[derive(Default)]
pub struct Decoder {
    pub(super) tier: Tier,
    /// Child buffers of one block; empty until first use.
    arena: Vec<u8>,
    ops: Vec<Op>,
}

impl Decoder {
    /// Returns a Decoder with its scratch allocated that runs the kernels of `tier`; for
    /// tests and benchmarks.
    pub fn with_tier(tier: Tier) -> Self {
        let mut d = Self::default();
        d.setup();
        d.tier = tier;
        d
    }

    fn setup(&mut self) {
        if self.arena.is_empty() {
            self.arena = vec![0; ARENA_SIZE + SCRATCH_ALIGN];
            self.ops = vec![Op::default(); MAX_OPS];
            self.tier = Tier::best();
        }
    }

    /// Decodes the block `src`, coded with `t`, to the dst.len() symbols of `dst`.
    /// `src` must be exactly the block.
    ///
    /// It returns [`Error::BlockSize`] unless 1 <= dst.len() <= [`MAX_BLOCK_SIZE`],
    /// [`Error::InvalidTable`] if `t` is not built, and [`Error::Corrupt`] if `src` is not
    /// exactly a block of dst.len() symbols as [`Encoder::encode`](super::Encoder::encode)
    /// writes it. On error, `dst` holds unspecified bytes.
    pub fn decode(&mut self, dst: &mut [u8], src: &[u8], t: &Table) -> Result<(), Error> {
        let n = dst.len();
        if n == 0 || n > MAX_BLOCK_SIZE {
            return Err(Error::BlockSize);
        }
        if !t.valid() {
            return Err(Error::InvalidTable);
        }
        self.setup();
        let mut pl = Planner {
            t,
            input: src,
            ops: &mut self.ops,
            nops: 0,
            pos: 0,
        };
        pl.node(0, n as u32, 0, 0, 0)?;
        if pl.pos != src.len() {
            return Err(ERR_TRAILING);
        }
        let nops = pl.nops;
        let off = self
            .arena
            .as_ptr()
            .align_offset(SCRATCH_ALIGN)
            .min(SCRATCH_ALIGN);
        let mut p = Program {
            ops: &self.ops[..nops],
            input: src,
            arena: &mut self.arena[off..off + ARENA_SIZE],
            dst,
            r2s: Some(&t.r2s),
        };
        self.tier.exec(&mut p).map_err(|_| ERR_MERGE)
    }
}

/// Parses and validates a block and turns it into kernel ops, in upstream's post-order and
/// with its buffer placement (`pivco_decode` and `codec_decode_subtree` in
/// `src/pivco_huffman_codec.c`). All ops run once the whole block is parsed.
struct Planner<'a> {
    t: &'a Table,
    /// The block.
    input: &'a [u8],
    ops: &'a mut [Op],
    nops: usize,
    /// Read cursor in input.
    pos: usize,
}

impl<'a> Planner<'a> {
    /// Plans the subtree of record rec at depth holding k symbols, decoded to out with tmp
    /// as the ping-pong partner. The root decodes to Dst and places its children at the
    /// start of the arena instead.
    fn node(&mut self, rec: usize, k: u32, out: u32, tmp: u32, depth: usize) -> Result<(), Error> {
        if k == 0 {
            return Ok(());
        }
        let r = self.t.sched[rec];
        let root = depth == 0;
        let fl = if root { FLAG_DST } else { 0 };
        let kind = r.kind();
        match kind {
            KIND_FLAT => return self.flat(r, k, out, fl),
            KIND_BOTH => {
                let (b, bm) = self.bitmap(k)?;
                let (p, q) = (
                    self.t.r2s[usize::from(r.param)],
                    self.t.r2s[usize::from(r.param) + 1],
                );
                // Equal leaves mean a single-symbol table, whose bitmap upstream writes as zeros.
                if p == q && ones(b) != 0 {
                    return Err(ERR_SINGLE);
                }
                *self.op()? = Op {
                    kind: OP_CC,
                    flags: fl,
                    p,
                    q,
                    k,
                    bm,
                    out,
                    ..Op::default()
                };
                return Ok(());
            }
            _ => {}
        }
        let kr = self.kr(k)?;
        let kl = k - kr;
        let (opk, p, l, rr) = if kind == KIND_LEAF_LEFT {
            // The right child decodes in place into the tail of out.
            let (rr, tmp) = if root { (0, kr) } else { (out + kl, tmp) };
            self.node(rec + 1, kr, rr, tmp, depth + 1)?;
            (OP_CV, self.t.r2s[usize::from(r.param)], 0, rr)
        } else {
            // The larger child (first on the wire, ties left) decodes in place at
            // out+ksmall, the smaller one into tmp with out as its partner.
            let right_first = kr > kl;
            let (mut first, mut second, mut kf, mut ks) =
                (rec + usize::from(r.right), rec + 1, kr, kl);
            if !right_first {
                (first, second, kf, ks) = (second, first, kl, kr);
            }
            let (fo, ft, so, st) = if root {
                (0, kf, kf, k)
            } else {
                (out + ks, tmp, tmp, out)
            };
            self.node(first, kf, fo, ft, depth + 1)?;
            self.node(second, ks, so, st, depth + 1)?;
            let (l, rr) = if right_first { (so, fo) } else { (fo, so) };
            (OP_VV, 0, l, rr)
        };
        let (_, bm) = self.bitmap(k)?;
        *self.op()? = Op {
            kind: opk,
            flags: fl,
            p,
            k,
            kr,
            bm,
            out,
            l,
            r: rr,
            ..Op::default()
        };
        Ok(())
    }

    /// Reads a K_right header.
    fn kr(&mut self, k: u32) -> Result<u32, Error> {
        if self.input.len() - self.pos < 2 {
            return Err(ERR_TRUNCATED);
        }
        let kr = u32::from(load16(self.input, self.pos));
        self.pos += 2;
        if kr > k {
            return Err(ERR_SPLIT);
        }
        Ok(kr)
    }

    /// Reads a k-bit bitmap and returns it with its offset. Its set bits are checked
    /// against K_right when the ops run.
    fn bitmap(&mut self, k: u32) -> Result<(&'a [u8], u32), Error> {
        let (input, nb) = (self.input, (k as usize).div_ceil(8));
        if input.len() - self.pos < nb {
            return Err(ERR_TRUNCATED);
        }
        let at = self.pos;
        self.pos += nb;
        padding(input[self.pos - 1], k)?;
        Ok((&input[at..self.pos], at as u32))
    }

    /// Reads a flat region and emits one op per layout part: 512-code and 128-code
    /// vertical blocks, then a natural tail; a depth-8 flat is the identity code.
    fn flat(&mut self, r: Rec, k: u32, mut out: u32, fl: u8) -> Result<(), Error> {
        let input = self.input;
        let d = u32::from(r.flat_depth());
        let tb = (k * d).div_ceil(8) as usize;
        if input.len() - self.pos < tb {
            return Err(ERR_TRUNCATED);
        }
        let mut bm = self.pos as u32;
        self.pos += tb;
        padding(input[self.pos - 1], k * d)?;
        if d == 8 {
            *self.op()? = Op {
                kind: OP_COPY,
                flags: fl,
                d: 8,
                p: r.param,
                k,
                bm,
                out,
                ..Op::default()
            };
            return Ok(());
        }
        let (n512, n128) = vert_split(k as usize, d as u8);
        // In op kind order: OP_V512, OP_V128, OP_NAT.
        let ns = [n512 as u32, n128 as u32, k - n512 as u32 - n128 as u32];
        for (i, &n) in ns.iter().enumerate() {
            if n == 0 {
                continue;
            }
            *self.op()? = Op {
                kind: OP_V512 + i as u8,
                flags: fl,
                d: d as u8,
                p: r.param,
                k: n,
                bm,
                out,
                ..Op::default()
            };
            bm += n * d / 8;
            out += n;
        }
        Ok(())
    }

    /// Appends an op slot, which the caller fills. Its geometry is not checked here: every
    /// tier's exec checks it.
    fn op(&mut self) -> Result<&mut Op, Error> {
        let n = self.nops;
        if n >= self.ops.len() {
            return Err(ERR_GEOMETRY);
        }
        self.nops = n + 1;
        Ok(&mut self.ops[n])
    }
}

/// Returns the number of set bits in b.
fn ones(b: &[u8]) -> u32 {
    let mut c = b.chunks_exact(8);
    let n: u32 = (&mut c).map(|w| load64(w, 0).count_ones()).sum();
    n + c.remainder().iter().map(|x| x.count_ones()).sum::<u32>()
}

/// Checks that the bits of the last byte of a bits-long region past its end are 0.
fn padding(last: u8, bits: u32) -> Result<(), Error> {
    let s = bits & 7;
    if s != 0 && last >> s != 0 {
        return Err(ERR_PADDING);
    }
    Ok(())
}
