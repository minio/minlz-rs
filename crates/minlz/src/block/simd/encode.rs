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

//! The SIMD block encoder, a port of Go's `encode-simd.go`. The parsers (`parse_*.rs`) emit
//! operations, which are coded a chunk at a time, then [`Encoder::finish`] codes the literals and
//! writes the block.
//!
//! Go's assembly parsers parse as its Go parsers, so only those are ported.

use std::cell::Cell;

use super::{
    BLOCK_TYPE, CLASS_LENS, CODE_BYTES, CODE_LENS, CURRENT, DELTA, DELTA_LENS, ESC, LAYOUTS, LITS,
    LL, MAX_CHUNKS, MAX_LEN, MIN_SAVING, OFFC, TOK,
};
use crate::block::format::put_uvarint;
use crate::block::load_store::{load8, load64};
use crate::pivco;

/// Go's `LevelSuperFastSIMD` and the other int levels of [`encode_block_simd`].
pub(crate) const LEVEL_SUPER_FAST: i32 = 4;
pub(crate) const LEVEL_FASTEST: i32 = 5;
pub(crate) const LEVEL_BALANCED: i32 = 6;
pub(crate) const LEVEL_SMALLEST: i32 = 7;
/// Go's `LevelSIMDDelta`, the flag of delta literals.
pub(crate) const LEVEL_DELTA: i32 = 256;

/// The chunk limit value of the header flags.
const CHUNK_LIMIT: u8 = 4;
pub(super) const CHUNK_OPS: usize = 1024 << CHUNK_LIMIT;
/// The longest literal run or match of an operation, with an escape.
pub(super) const MAX_LONG: usize = MAX_LEN + 1 + 255;
/// Estimates the overhead of a pivco block.
pub(super) const BLOCK_BITS: usize = 32 * 8;
/// The overhead SuperFast and Fastest assume. They store more small streams,
/// which encodes ~5% faster at 4 KB and costs at most 0.2% size.
const FAST_BLOCK_BITS: usize = 40 * 8;
/// The fewest values that, at one bit each, code smaller than stored.
const MIN_CODED: usize = BLOCK_BITS / 7 + 1;
/// The size of the largest chunk record: flags, 5 sizes and 4 tables.
const RECORD_MAX: usize = 1 + 5 * 3 + 68 + 17 + 88 + 128;
/// Added to the cost of a new table, since every table costs decode time.
pub(super) const TABLE_BITS: usize = 64 * 8;
/// The step of the literal block splits of LevelSmallestSIMD.
const LIT_SEG: usize = 8192;

/// The size of the class lengths of each layout.
const CLASS_BYTES: [usize; 6] = [11, 6, 12, 8, 7, 5];

/// log2 of the alphabet size of each stream, in 1/256 bits, rounded up: Go's
/// `math.Ceil(256 * math.Log2(float64(2*n)))` of [`CODE_BYTES`].
const MAX_ENTROPY: [usize; LITS + 1] = [1815, 1303, 1910, 2048, 2048];

/// log2(1+i/256) in 1/256 units, as Go's `math.Round(256 * math.Log2(1+float64(i)/256))`.
#[rustfmt::skip]
static LOG2_FRAC: [u8; 256] = [
    0, 1, 3, 4, 6, 7, 9, 10, 11, 13, 14, 16, 17, 18, 20, 21,
    22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42,
    44, 45, 46, 47, 49, 50, 51, 52, 54, 55, 56, 57, 59, 60, 61, 62,
    63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
    134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165, 166, 167, 168, 169, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 178,
    179, 180, 181, 182, 183, 184, 185, 185, 186, 187, 188, 189, 190, 191, 192, 192,
    193, 194, 195, 196, 197, 198, 198, 199, 200, 201, 202, 203, 203, 204, 205, 206,
    207, 208, 208, 209, 210, 211, 212, 212, 213, 214, 215, 216, 216, 217, 218, 219,
    220, 220, 221, 222, 223, 224, 224, 225, 226, 227, 228, 228, 229, 230, 231, 231,
    232, 233, 234, 234, 235, 236, 237, 238, 238, 239, 240, 241, 241, 242, 243, 244,
    244, 245, 246, 247, 247, 248, 249, 249, 250, 251, 252, 252, 253, 254, 255, 255,
];

/// Go's test knobs, which are false outside tests.
#[derive(Clone, Copy, Default)]
pub(crate) struct Knobs {
    /// `simdNoOverlap` sets no_overlap, without budget. LevelSuperFastSIMD has it with budget,
    /// except with delta literals: the no-overlap flag raises their references, so a chunk can't
    /// drop it.
    pub(crate) no_overlap: bool,
    /// `simdSplitCost` makes LevelSmallestSIMD price the splits of no_overlap.
    pub(crate) split_cost: bool,
    /// `simdLongRuns` makes the parsers continue matches longer than 64 bytes at offsets below 32
    /// at a multiple of the offset, as LevelSmallestSIMD does, instead of in operations of 32 bytes.
    pub(crate) long_runs: bool,
}

/// A table with its code lengths and its serialized form.
#[derive(Clone)]
pub(super) struct Table {
    t: pivco::Table,
    lens: [u8; 256],
    /// CODE_LENS, CLASS_LENS or DELTA_LENS.
    mode: usize,
    n: usize,
    b: [u8; 128],
}

impl Default for Table {
    fn default() -> Self {
        Table {
            t: pivco::Table::default(),
            lens: [0; 256],
            mode: 0,
            n: 0,
            b: [0; 128],
        }
    }
}

/// What put codes with.
struct Coder {
    pv: pivco::Encoder,
    cur: [Table; LITS + 1],
    have: [bool; LITS + 1],
    tmp: [Table; 2],
    hist: [u64; 256],
    sample: [u8; 4096],
}

/// The literal block splits: cuts, and for LevelSmallestSIMD the segment histograms and the
/// dynamic program.
#[derive(Default)]
struct LitSplit {
    cuts: Vec<usize>,
    dp: Vec<usize>,
    segs: Vec<[u32; 256]>,
}

/// The state of an [`Encoder`] that outlives a block, which Go pools with its encoder: the
/// streams of the current chunk, the buffers and the tables.
pub(super) struct Scratch {
    // The fast paths mask their indexes, so the compiler drops the bounds checks.
    tok: [u8; CHUNK_OPS],
    ll: [u8; CHUNK_OPS],
    offc: [u8; CHUNK_OPS],
    esc: [u8; 2 * CHUNK_OPS],
    /// Room for a 4 byte store at any u16 index. An offset has at most 20 raw bits.
    raw: [u8; (1 << 16) + 3],
    /// The literals of the block, then, with plain and delta literals, the delta literals.
    buf: Vec<u8>,
    /// The literal positions where plain and delta literals take turns, starting with plain.
    switches: Vec<usize>,
    lit_recs: Vec<u8>,
    chunk_recs: Vec<u8>,
    split: LitSplit,
    c: Coder,
}

impl Scratch {
    pub(super) fn new() -> Box<Self> {
        Box::new(Scratch {
            tok: [0; CHUNK_OPS],
            ll: [0; CHUNK_OPS],
            offc: [0; CHUNK_OPS],
            esc: [0; 2 * CHUNK_OPS],
            raw: [0; (1 << 16) + 3],
            buf: Vec::new(),
            switches: Vec::new(),
            lit_recs: Vec::new(),
            chunk_recs: Vec::new(),
            split: LitSplit::default(),
            c: Coder {
                pv: pivco::Encoder::default(),
                cur: Default::default(),
                have: [false; LITS + 1],
                tmp: Default::default(),
                hist: [0; 256],
                sample: [0; 4096],
            },
        })
    }
}

/// The block being written to dst, Go's `e.out`, and `limit`, the largest size of the encoded
/// block. Go's appends would grow a slice beyond dst; these fail the block instead.
struct Out<'a> {
    b: &'a mut [u8],
    n: usize,
    limit: isize,
}

impl Out<'_> {
    fn room(&self) -> usize {
        self.b.len() - self.n
    }

    /// Reports whether the block, with extra more bytes, is within the limit.
    fn fits(&self, extra: usize) -> bool {
        (self.n + extra) as isize <= self.limit
    }

    fn append(&mut self, v: &[u8]) {
        match self.b.get_mut(self.n..self.n + v.len()) {
            Some(d) => {
                d.copy_from_slice(v);
                self.n += v.len();
            }
            None => self.limit = -1,
        }
    }

    fn append_uvarint(&mut self, v: u64) {
        let (b, n) = uvarint(v);
        self.append(&b[..n]);
    }

    /// appendRvarint
    fn append_rvarint(&mut self, v: u64) {
        let (mut b, n) = uvarint(v);
        b[..n].reverse();
        self.append(&b[..n]);
    }
}

fn uvarint(mut v: u64) -> ([u8; 10], usize) {
    let (mut b, mut n) = ([0u8; 10], 0);
    while v >= 0x80 {
        b[n] = v as u8 | 0x80;
        v >>= 7;
        n += 1;
    }
    b[n] = v as u8;
    (b, n + 1)
}

/// Builds SIMD blocks, Go's `simdEncoder`. Parsers emit operations, which are coded a chunk at a
/// time, then finish codes the literals and writes the block.
pub(crate) struct Encoder<'a> {
    // Go's simdStreams: the stream sizes of the current chunk and the parser state.
    n_tok: usize,
    n_ll: usize,
    n_offc: usize,
    n_esc: usize,
    n_raw: usize,
    n_lits: usize,
    nacc: usize,
    acc: u64,
    pub(super) pos: usize,
    pub(super) rep: usize,
    /// A match of the chunk overlaps its output.
    overlap: bool,

    s: Box<Scratch>,
    src: &'a [u8],
    chunks: usize,
    /// The literals after the last operation, which finish adds to the literals once the block
    /// can fit.
    tail: &'a [u8],
    /// Where the delta literals start in s.buf, the plain ones at 0. With delta literals only,
    /// they are the same.
    dlits: usize,
    /// op_lits ends the literals of the operations, and chunk_lits starts those of the current chunk.
    op_lits: usize,
    chunk_lits: usize,
    /// Chooses the literal block splits by size.
    split_lits: bool,
    /// When set, makes put build one table at most: code lengths for at least fast_code values,
    /// else class lengths. Building tables dominates the encoding of small blocks.
    fast_code: usize,
    /// Delta literals must save 1/delta_div of a chunk; 0 disables them, -1 uses only them.
    delta_div: isize,
    /// no_overlap splits the matches that overlap their output, so every chunk has the no-overlap
    /// flag. With budget, a chunk stops splitting once that cost it more than split_budget, and
    /// overlaps. split_cost is what splitting cost the chunk: operations and offsets, against
    /// operations of 32 bytes.
    pub(super) no_overlap: bool,
    pub(super) budget: bool,
    split_cost: isize,
    /// Where the current chunk starts.
    chunk_pos: usize,
    /// Set when a delta literal of the chunk references an offset that the no-overlap flag would
    /// raise.
    delta_short: bool,
    out: Out<'a>,
    pub(crate) knobs: Knobs,
}

thread_local! {
    /// Go's `simdEncoders` pool.
    static ENCODERS: Cell<Option<Box<Scratch>>> = const { Cell::new(None) };
}

/// encodeBlockSIMD encodes src to dst as a SIMD block without its indicator byte, and returns its
/// size. It returns 0 if the block would not save the level's minimum, and at least
/// [`MIN_SAVING`] bytes.
pub(crate) fn encode_block_simd(dst: &mut [u8], src: &[u8], level: i32) -> usize {
    // No block this small saves MIN_SAVING bytes, and the parsers load 8 bytes at 1.
    if src.len() <= MIN_SAVING {
        return 0;
    }
    let mut e = Encoder::with_scratch(dst, ENCODERS.take().unwrap_or_else(Scratch::new));
    e.parse(src, level);
    let n = e.finish();
    ENCODERS.set(Some(e.s));
    n
}

/// matchAt returns how many bytes match at src[a..] and src[b..], like matchLen.
///
/// # Safety
/// `b < a <= src.len()` must hold.
#[inline(always)]
pub(super) unsafe fn match_at(src: &[u8], a: usize, b: usize) -> usize {
    debug_assert!(b < a && a <= src.len());
    let mut n = 0;
    while a + n + 8 <= src.len() {
        // SAFETY: b < a.
        let diff = unsafe { load64(src, a + n) ^ load64(src, b + n) };
        if diff != 0 {
            return n + (diff.trailing_zeros() >> 3) as usize;
        }
        n += 8;
    }
    // SAFETY: as above.
    while a + n < src.len() && unsafe { load8(src, a + n) == load8(src, b + n) } {
        n += 1;
    }
    n
}

/// Returns how many bytes match before src[a] and src[b], at most max: the back-extension of
/// the parsers, in a form that keeps LLVM from deriving an induction variable for every value
/// used after it.
///
/// # Safety
/// `max <= b < a <= src.len()` must hold.
#[inline(always)]
pub(super) unsafe fn match_back(src: &[u8], a: usize, b: usize, max: usize) -> usize {
    debug_assert!(max <= b && b < a && a <= src.len());
    let mut n = 0;
    // SAFETY: b - n - 1 < a - n - 1 < a, and n < max <= b.
    while n < max && unsafe { load8(src, a - n - 1) == load8(src, b - n - 1) } {
        n += 1;
    }
    n
}

/// split_budget returns the split cost a chunk of n operations allows with budget.
/// Data of short runs can otherwise take several times the operations, and fill more chunks than
/// a block has. Few chunks of real data reach it.
fn split_budget(n: usize) -> isize {
    (64 + n / 32) as isize
}

impl<'a> Encoder<'a> {
    /// An encoder that writes to out, as Go's `&simdEncoder{out: out[:0:len(out)]}`, with the
    /// state s.
    pub(super) fn with_scratch(out: &'a mut [u8], s: Box<Scratch>) -> Self {
        Encoder {
            n_tok: 0,
            n_ll: 0,
            n_offc: 0,
            n_esc: 0,
            n_raw: 0,
            n_lits: 0,
            nacc: 0,
            acc: 0,
            pos: 0,
            rep: 1,
            overlap: false,
            s,
            src: &[],
            chunks: 0,
            tail: &[],
            dlits: 0,
            op_lits: 0,
            chunk_lits: 0,
            split_lits: false,
            fast_code: 0,
            delta_div: 0,
            no_overlap: false,
            budget: false,
            split_cost: 0,
            chunk_pos: 0,
            delta_short: false,
            out: Out {
                b: out,
                n: 0,
                limit: 0,
            },
            knobs: Knobs::default(),
        }
    }

    /// parse codes the operations of src.
    pub(crate) fn parse(&mut self, src: &'a [u8], level: i32) {
        let delta = if level & LEVEL_DELTA != 0 { -1 } else { 0 };
        self.split_lits = level == LEVEL_SMALLEST;
        self.fast_code = 0;
        (self.no_overlap, self.budget) = (self.knobs.no_overlap, false);
        match level & !LEVEL_DELTA {
            LEVEL_SUPER_FAST => {
                self.fast_code = 1;
                if delta == 0 {
                    (self.no_overlap, self.budget) = (true, !self.knobs.no_overlap);
                }
                self.reset(src, delta, src.len() >> 3);
                if src.len() <= 64 << 10 {
                    self.encode_fast_block_go_64k(src);
                } else {
                    self.encode_fast_block_go(src);
                }
            }
            LEVEL_FASTEST => {
                self.fast_code = 4096;
                self.reset(src, delta, src.len() >> 5);
                if src.len() <= 64 << 10 {
                    self.encode_block_go_64k(src);
                } else {
                    self.encode_block_go(src);
                }
            }
            LEVEL_BALANCED => {
                self.reset(src, delta, src.len() >> 5);
                if src.len() <= 64 << 10 {
                    self.encode_block_better_go_64k(src);
                } else {
                    self.encode_block_better_go(src);
                }
            }
            _ => {
                self.reset(src, 256, 0);
                self.encode_block_best(src);
            }
        }
        if self.n_tok > 0 {
            self.end_chunk();
        }
        debug_assert_eq!(self.pos, src.len(), "operations do not cover the block");
    }

    /// reset starts a block for src, which must save at least save bytes. delta_div is as
    /// self.delta_div.
    pub(super) fn reset(&mut self, src: &'a [u8], delta_div: isize, save: usize) {
        let n = src.len();
        let size = if delta_div > 0 { 2 * n } else { n };
        let s = &mut *self.s;
        if s.buf.len() < size {
            s.buf.resize(size, 0);
        }
        self.dlits = if delta_div > 0 { n } else { 0 };
        self.out.n = 0;
        self.out.limit = n as isize - MIN_SAVING.max(save) as isize;
        self.out.append_uvarint(BLOCK_TYPE << 24 | n as u64);
        self.out.append(&[CHUNK_LIMIT]);
        // Operations are fewer than n/2, so this bounds the chunks and literal blocks.
        // Each chunk can switch between plain and delta literals, which starts a literal block.
        let chunks = n / 2 / CHUNK_OPS + 2;
        let parts = n / pivco::MAX_BLOCK_SIZE + chunks + 2;
        s.chunk_recs.clear();
        s.chunk_recs.reserve(chunks * RECORD_MAX);
        s.switches.clear();
        s.switches.reserve(chunks + 2);
        s.lit_recs.clear();
        s.lit_recs.reserve(parts * (3 + 128));
        (self.n_tok, self.n_ll, self.n_offc, self.n_esc) = (0, 0, 0, 0);
        (self.n_raw, self.n_lits, self.chunks) = (0, 0, 0);
        (self.op_lits, self.chunk_lits) = (0, 0);
        (self.acc, self.nacc, self.overlap, self.delta_short) = (0, 0, false, false);
        (self.split_cost, self.chunk_pos) = (0, 0);
        (self.src, self.tail, self.pos, self.rep) = (src, &[], 0, 1);
        self.delta_div = delta_div;
        s.c.have = [false; LITS + 1];
    }

    /// emitLiterals adds the literals after the last operation.
    pub(super) fn emit_literals(&mut self, lits: &'a [u8]) {
        self.tail = lits;
        self.pos += lits.len();
    }

    // The parsers add the common operations with fast, put_offset, put_bits, put_lits and
    // put_match, which must stay inlined to save the calls, and leave the others to the emitters.

    /// fast reports whether the plain literals src[from..base], and a match at offset that ends
    /// at s, fit put_lits and put_match. Delta literals are rare, so they are left to the emitters.
    #[inline(always)]
    pub(super) fn fast(
        &self,
        src: &[u8],
        from: usize,
        base: usize,
        s: usize,
        offset: usize,
    ) -> bool {
        self.delta_div == 0
            && base - from <= MAX_LONG
            && src.len() - from >= MAX_LEN
            && self.one_op(offset, s - base)
    }

    /// oneOp reports whether put_match can add a match of length bytes at offset.
    #[inline(always)]
    pub(super) fn one_op(&self, offset: usize, length: usize) -> bool {
        length <= MAX_LEN && (offset >= length || !self.no_overlap)
            || length <= MAX_LONG && offset >= MAX_LEN
    }

    /// putOffset adds the symbol of a match offset, and returns its raw bits and their count for
    /// put_bits.
    #[inline(always)]
    pub(super) fn put_offset(&mut self, offset: usize) -> (u64, usize) {
        // With the 8, offsets below 16 code as themselves without raw bits, and need no branch.
        let b = (usize::BITS - (offset | 8).leading_zeros()) as usize - 4;
        self.s.offc[self.n_offc & (CHUNK_OPS - 1)] = (8 * b + (offset >> b)) as u8;
        self.n_offc += 1;
        ((offset & ((1 << b) - 1)) as u64, b)
    }

    /// putBits adds the n raw bits of v.
    #[inline(always)]
    pub(super) fn put_bits(&mut self, (v, n): (u64, usize)) {
        // Store and advance without a branch, since the stores come at an irregular pace.
        // raw has room, as the offsets of a chunk have at most 20 raw bits each.
        let acc = self.acc | v << (self.nacc & 63);
        let n = n + self.nacc;
        let i = self.n_raw as u16 as usize;
        self.s.raw[i..i + 4].copy_from_slice(&(acc as u32).to_le_bytes());
        let k = n >> 5;
        self.n_raw += k << 2;
        (self.acc, self.nacc) = (acc >> ((k << 5) & 63), n & 31);
    }

    /// putLL adds the literal length ll, at most MAX_LONG, and returns the literal bit of the
    /// operation.
    #[inline(always)]
    pub(super) fn put_ll(&mut self, ll: usize) -> u8 {
        // Operations with and without literals alternate unpredictably, so don't branch on it.
        let n = ll.min(1);
        self.s.ll[self.n_ll & (CHUNK_OPS - 1)] = ll.min(MAX_LEN + 1) as u8;
        self.n_ll += n;
        if ll > MAX_LEN {
            self.s.esc[self.n_esc & (2 * CHUNK_OPS - 1)] = (ll - MAX_LEN - 1) as u8;
            self.n_esc += 1;
        }
        (n as u8) << 1
    }

    /// putLits adds the ll plain literals at src[from..], where ll is at most MAX_LONG.
    /// src has at least MAX_LEN bytes from from, so the literals, as long as src, have as much
    /// room after them.
    #[inline(always)]
    pub(super) fn put_lits(&mut self, src: &[u8], from: usize, ll: usize) {
        let n = self.n_lits;
        if ll > MAX_LEN {
            self.s.buf[n..n + ll].copy_from_slice(&src[from..from + ll]);
        } else {
            // An array copy inlines, where LLVM merges two copy_from_slice into one memcpy call.
            let d: &mut [u8; MAX_LEN] = (&mut self.s.buf[n..n + MAX_LEN]).try_into().unwrap();
            *d = src[from..from + MAX_LEN].try_into().unwrap();
        }
        self.n_lits += ll;
        self.pos += ll;
    }

    /// putMatch adds an operation with the literal and repeat bits t, and a match at offset that
    /// satisfies one_op. The offset becomes the repeat offset. It returns whether the chunk is
    /// full.
    #[inline(always)]
    pub(super) fn put_match(&mut self, mut t: u8, offset: usize, length: usize) -> bool {
        self.rep = offset;
        if length > MAX_LEN {
            self.s.esc[self.n_esc & (2 * CHUNK_OPS - 1)] = (length - MAX_LEN - 1) as u8;
            self.n_esc += 1;
            t |= ((MAX_LEN + 1) << 2) as u8;
        } else {
            t |= (length as u8) << 2;
            self.overlap = self.overlap || offset < length;
        }
        self.s.tok[self.n_tok & (CHUNK_OPS - 1)] = t;
        self.n_tok += 1;
        self.pos += length;
        self.n_tok == CHUNK_OPS
    }

    /// Adds plain literals, or with delta literals, both kinds for an operation with repeat
    /// offset rep.
    fn add_lits(&mut self, lits: &[u8], rep: usize) {
        if self.delta_div != 0 {
            self.delta_lits(lits, rep);
        } else {
            let n = self.n_lits;
            self.s.buf[n..n + lits.len()].copy_from_slice(lits);
            self.n_lits += lits.len();
        }
    }

    /// emitCopyLits adds lits, then a match of length bytes at offset.
    pub(super) fn emit_copy_lits(&mut self, mut lits: &[u8], offset: usize, length: usize) {
        while lits.len() > MAX_LONG {
            self.emit_long(&lits[..MAX_LONG], self.rep, 1, 0);
            lits = &lits[MAX_LONG..];
        }
        let prev = self.rep;
        self.rep = offset;
        let b = self.put_offset(offset);
        self.put_bits(b);
        if lits.len() > MAX_LEN || length > MAX_LEN || self.no_overlap && offset < length {
            self.emit_long(lits, prev, 0, length);
            return;
        }
        let mut t = 0;
        if !lits.is_empty() {
            t = 2;
            self.s.ll[self.n_ll] = lits.len() as u8;
            self.n_ll += 1;
            self.add_lits(lits, prev);
            self.pos += lits.len();
        }
        self.overlap = self.overlap || offset < length;
        self.s.tok[self.n_tok] = t | (length as u8) << 2;
        self.n_tok += 1;
        self.pos += length;
        if self.n_tok == CHUNK_OPS {
            self.end_chunk();
        }
    }

    /// emitRepeatLits adds lits, then a match of length bytes at the repeat offset.
    pub(super) fn emit_repeat_lits(&mut self, mut lits: &[u8], length: usize) {
        while lits.len() > MAX_LONG {
            self.emit_long(&lits[..MAX_LONG], self.rep, 1, 0);
            lits = &lits[MAX_LONG..];
        }
        if lits.len() > MAX_LEN || length > MAX_LEN || self.no_overlap && self.rep < length {
            self.emit_long(lits, self.rep, 1, length);
            return;
        }
        let mut t = 1;
        if !lits.is_empty() {
            t |= 2;
            self.s.ll[self.n_ll] = lits.len() as u8;
            self.n_ll += 1;
            self.add_lits(lits, self.rep);
            self.pos += lits.len();
        }
        self.overlap = self.overlap || self.rep < length;
        self.s.tok[self.n_tok] = t | (length as u8) << 2;
        self.n_tok += 1;
        self.pos += length;
        if self.n_tok == CHUNK_OPS {
            self.end_chunk();
        }
    }

    /// emitCopy adds a match of length bytes at offset, without literals.
    pub(super) fn emit_copy(&mut self, offset: usize, length: usize) {
        debug_assert!(self.pos > 0, "first operation without literals");
        self.rep = offset;
        let b = self.put_offset(offset);
        self.put_bits(b);
        self.emit_long(&[], 0, 0, length);
    }

    /// emitLong adds an operation with lits, at most MAX_LONG of them, and a match of length
    /// bytes at self.rep, split into more operations if needed. t is the repeat bit of the
    /// operation, and prev the repeat offset before it. The offset of the match must already be
    /// added.
    fn emit_long(&mut self, lits: &[u8], prev: usize, mut t: u8, mut length: usize) {
        if !lits.is_empty() {
            let mut n = lits.len();
            t |= 2;
            if n > MAX_LEN {
                self.s.esc[self.n_esc] = (n - MAX_LEN - 1) as u8;
                self.n_esc += 1;
                n = MAX_LEN + 1;
            }
            self.s.ll[self.n_ll] = n as u8;
            self.n_ll += 1;
            self.add_lits(lits, prev);
            self.pos += lits.len();
        }
        let (mut step, mut split, offset) = (MAX_LEN, false, self.rep);
        if self.rep >= MAX_LEN {
            step = MAX_LONG;
        } else if self.no_overlap && length != 0 {
            if self.budget && self.split_cost > split_budget(self.n_tok) {
                self.no_overlap = false;
            } else {
                (step, split) = (self.rep, true);
            }
        }
        let mut cost = -(length.div_ceil(MAX_LEN) as isize);
        loop {
            let n = length.min(step);
            if n > MAX_LEN {
                self.s.esc[self.n_esc] = (n - MAX_LEN - 1) as u8;
                self.n_esc += 1;
                t |= ((MAX_LEN + 1) << 2) as u8;
            } else {
                t |= (n as u8) << 2;
                self.overlap = self.overlap || self.rep < n;
            }
            self.s.tok[self.n_tok] = t;
            self.n_tok += 1;
            self.pos += n;
            length -= n;
            cost += 1;
            if self.n_tok == CHUNK_OPS {
                self.end_chunk();
            }
            if length == 0 {
                if split {
                    // The parsers continue from the offset of the match as the repeat offset, as
                    // without the split: they find most matches of short runs by it. An operation
                    // of no bytes sets it, while the chunk outputs more bytes than it has
                    // operations. Else the parsers take the multiple.
                    if self.rep != offset && self.pos - self.chunk_pos > self.n_tok {
                        let b = self.put_offset(offset);
                        self.put_bits(b);
                        self.rep = offset;
                        self.s.tok[self.n_tok] = 0;
                        self.n_tok += 1;
                        cost += 2;
                        if self.n_tok == CHUNK_OPS {
                            self.end_chunk();
                        }
                    }
                    self.split_cost += cost;
                }
                return;
            }
            t = 1;
            if split && step < MAX_LEN && step < length {
                // The output repeats every rep bytes, and now holds at least twice that, so twice
                // the offset matches too.
                self.rep *= 2;
                let b = self.put_offset(self.rep);
                self.put_bits(b);
                (t, step) = (0, self.rep);
                cost += 1;
                if step >= MAX_LEN {
                    step = MAX_LONG;
                }
            } else if step == MAX_LEN
                && (self.knobs.long_runs && length > 2 * MAX_LEN
                    || self.no_overlap && self.n_tok == 0)
            {
                // The output now holds 32 bytes of the period, so the first multiple of it from 32
                // matches too, in operations of up to 288 bytes. A chunk that the budget stopped
                // splitting ended in the match, and the next continues it so, as it splits.
                self.rep *= MAX_LEN.div_ceil(self.rep);
                let b = self.put_offset(self.rep);
                self.put_bits(b);
                (t, step, split) = (0, MAX_LONG, self.no_overlap);
            }
        }
    }

    /// deltaLits adds lits, and their delta literals (SPEC_SIMD.md 4.5), for an operation at
    /// self.pos with repeat offset rep.
    fn delta_lits(&mut self, lits: &[u8], rep: usize) {
        let n = self.n_lits;
        let s = &mut *self.s;
        s.buf[n..n + lits.len()].copy_from_slice(lits);
        self.n_lits += lits.len();
        self.op_lits = self.n_lits;
        let dl = &mut s.buf[self.dlits + n..self.dlits + self.n_lits];
        if self.delta_div > 0 {
            dl.copy_from_slice(lits);
        }
        // The no-overlap flag raises the references to 32 (SPEC_SIMD.md 4.5), so a chunk that
        // doesn't can't have it.
        let mut r = rep.min(self.pos);
        let raised = rep.max(MAX_LEN).min(self.pos);
        if self.no_overlap {
            r = raised;
        } else if r != raised {
            self.delta_short = true;
        }
        if r == 0 {
            return;
        }
        let (dl, rf) = (
            &mut dl[..lits.len().min(MAX_LEN)],
            &self.src[self.pos - r..],
        );
        if r >= MAX_LEN {
            for (d, &b) in dl.iter_mut().zip(rf) {
                *d = d.wrapping_sub(b);
            }
            return;
        }
        for (j, d) in dl.iter_mut().enumerate() {
            *d = d.wrapping_sub(rf[j % r]);
        }
    }

    /// endChunk codes the streams of the current chunk to out and its record to chunk_recs.
    pub(super) fn end_chunk(&mut self) {
        while self.nacc > 0 {
            self.s.raw[self.n_raw] = self.acc as u8;
            self.n_raw += 1;
            self.acc >>= 8;
            self.nacc = self.nacc.saturating_sub(8);
        }
        (self.acc, self.nacc) = (0, 0);
        // The parsers' operations average at least 4 bytes, so 8 MiB fits in MAX_CHUNKS chunks.
        // Fail the block if not.
        self.chunks += 1;
        if self.chunks > MAX_CHUNKS {
            self.out.limit = -1;
        }
        // A block that is already too large fails, so don't code more.
        if self.out.fits(self.s.chunk_recs.len()) {
            let mut flags = 0;
            if !self.overlap && (self.delta_div >= 0 || !self.delta_short) {
                flags = 2;
            }
            if self.delta_div < 0 {
                flags |= 1;
            }
            let s = &mut *self.s;
            let (rec, size) = (s.chunk_recs.len(), self.out.n + s.chunk_recs.len());
            let (out, recs, fc) = (&mut self.out, &mut s.chunk_recs, self.fast_code);
            recs.push(flags);
            s.c.put(out, recs, TOK, &s.tok[..self.n_tok], fc);
            s.c.put(out, recs, LL, &s.ll[..self.n_ll], fc);
            s.c.put(out, recs, OFFC, &s.offc[..self.n_offc], fc);
            s.c.put(out, recs, ESC, &s.esc[..self.n_esc], fc);
            if out.room() < self.n_raw {
                out.limit = -1;
            } else {
                out.append(&s.raw[..self.n_raw]);
                put_uvarint(recs, (self.n_raw as u64) << 3);
            }
            if self.delta_div > 0 && self.delta_chunk(self.out.n + self.s.chunk_recs.len() - size) {
                self.s.chunk_recs[rec] |= 1;
                if self.delta_short {
                    self.s.chunk_recs[rec] &= !2;
                }
            }
        }
        (self.n_tok, self.n_ll, self.n_offc, self.n_esc, self.n_raw) = (0, 0, 0, 0, 0);
        (self.overlap, self.delta_short) = (false, false);
        (self.split_cost, self.no_overlap) = (0, self.no_overlap || self.budget);
        self.chunk_pos = self.pos;
    }

    /// finish codes the literals and completes the block. It returns its size, or 0 if it is too
    /// large.
    pub(crate) fn finish(&mut self) -> usize {
        if !self.out.fits(self.s.chunk_recs.len()) {
            return 0;
        }
        // On incompressible data, coding the literals takes most of the time, only to fail the
        // block.
        let room = self.out.limit - (self.out.n + self.s.chunk_recs.len()) as isize;
        if self.delta_div <= 0 && !self.lits_fit(room) {
            return 0;
        }
        let (s, tail) = (&mut *self.s, self.tail);
        s.buf[self.n_lits..self.n_lits + tail.len()].copy_from_slice(tail);
        self.n_lits += tail.len();
        if !s.chunk_recs.is_empty() {
            self.out.append(&s.chunk_recs);
            self.out.append_rvarint(s.chunk_recs.len() as u64);
        }
        // The LZ section starts after the 4 byte size and the flags.
        let lz = self.out.n - 5;
        s.lit_recs.clear();
        // Plain and delta literals go in separate literal blocks. The literals after the
        // operations are plain.
        if s.switches.len() & 1 == 1 {
            s.switches.push(self.op_lits);
        }
        let mut start = 0;
        for i in 0..=s.switches.len() {
            let end = s.switches.get(i).copied().unwrap_or(self.n_lits);
            let (k, at) = if i & 1 == 1 || self.delta_div < 0 {
                (DELTA, self.dlits)
            } else {
                (LITS, 0)
            };
            let lits = &s.buf[at + start..at + end];
            start = end;
            if lits.is_empty() {
                continue;
            }
            let cuts = s.split.lit_blocks(self.split_lits, lits);
            for c in cuts.windows(2) {
                let v = &lits[c[0]..c[1]];
                s.c.put(&mut self.out, &mut s.lit_recs, k, v, self.fast_code);
                if !self.out.fits(s.lit_recs.len()) {
                    return 0;
                }
            }
        }
        self.out.append(&s.lit_recs);
        self.out.append_rvarint(s.lit_recs.len() as u64);
        self.out.append_rvarint(lz as u64);
        if self.out.n as isize > self.out.limit {
            return 0;
        }
        self.out.n
    }

    /// litBits returns the estimated size of lits in bits, for the parsers.
    pub(super) fn lit_bits(&mut self, lits: &[u8]) -> usize {
        self.s.c.lit_bits(lits)
    }

    /// litsFit reports whether the literals, with the tail, can fit in room bytes.
    fn lits_fit(&mut self, room: isize) -> bool {
        let need = (self.n_lits + self.tail.len()) as isize - room;
        if need <= 0 {
            return true;
        }
        let s = &mut *self.s;
        let need = s.c.sample_save(&s.buf[..self.n_lits], need);
        s.c.sample_save(self.tail, need) <= 0
    }

    /// deltaChunk reports whether the current chunk, with lz bytes of operations, takes delta
    /// literals: when they are estimated to save 1/delta_div of the chunk. It records where the
    /// literals switch.
    fn delta_chunk(&mut self, lz: usize) -> bool {
        let start = self.chunk_lits;
        self.chunk_lits = self.op_lits;
        if start == self.op_lits {
            return false;
        }
        let s = &mut *self.s;
        let raw = s.c.lit_bits(&s.buf[start..self.op_lits]) as isize;
        let delta =
            s.c.lit_bits(&s.buf[self.dlits + start..self.dlits + self.op_lits]) as isize;
        let d = raw - delta > (raw + 8 * lz as isize) / self.delta_div;
        if d != (s.switches.len() & 1 == 1) {
            s.switches.push(start);
        }
        d
    }
}

impl LitSplit {
    /// litBlocks returns the boundaries of the literal blocks of lits.
    /// With split_lits, they are on LIT_SEG boundaries, chosen for the smallest estimated size.
    /// Otherwise the blocks are equal.
    fn lit_blocks(&mut self, split_lits: bool, lits: &[u8]) -> &[usize] {
        self.cuts.clear();
        if !split_lits {
            let parts = lits.len().div_ceil(pivco::MAX_BLOCK_SIZE);
            let size = lits.len().div_ceil(parts);
            self.cuts.extend((0..lits.len()).step_by(size));
            self.cuts.push(lits.len());
            return &self.cuts;
        }
        let n = lits.len().div_ceil(LIT_SEG);
        self.segs.clear();
        self.segs.resize(n, [0; 256]);
        for (seg, l) in self.segs.iter_mut().zip(lits.chunks(LIT_SEG)) {
            for &b in l {
                seg[b as usize] += 1;
            }
        }
        // cost[i] is the smallest size of the first i segments, and from[i] where its last block
        // starts.
        self.dp.clear();
        self.dp.resize(2 * n + 2, 0);
        let (cost, from) = self.dp.split_at_mut(n + 1);
        cost[0] = 0;
        for i in 1..=n {
            cost[i] = usize::MAX;
            let mut h = [0u64; 256];
            for j in (i.saturating_sub(pivco::MAX_BLOCK_SIZE / LIT_SEG)..i).rev() {
                for (v, &f) in h.iter_mut().zip(&self.segs[j]) {
                    *v += f as u64;
                }
                // An entry is about 3 bytes.
                let c = cost[j] + 3 * 8 + lit_bits(&h, (i * LIT_SEG).min(lits.len()) - j * LIT_SEG);
                if c < cost[i] {
                    (cost[i], from[i]) = (c, j);
                }
            }
        }
        let mut i = n;
        while i > 0 {
            self.cuts.push((i * LIT_SEG).min(lits.len()));
            i = from[i];
        }
        self.cuts.push(0);
        self.cuts.reverse();
        &self.cuts
    }
}

/// simdLitBits estimates the size in bits of a literal block of n values with histogram h:
/// stored, or its entropy plus the overhead of a block with a new table of about 16 bytes.
/// The entropy is cheaper than building the tables, which would dominate the encoding of
/// literal-heavy data.
fn lit_bits(h: &[u64; 256], n: usize) -> usize {
    if n < MIN_CODED {
        return 8 * n;
    }
    // n*log2(n) - sum(f*log2(f)), in 1/256 bits.
    let mut e = (n * log2(n as u64)) as isize;
    for &f in h {
        if f > 1 {
            e -= (f as usize * log2(f)) as isize;
        }
    }
    (8 * n).min((e >> 8).max(0) as usize + BLOCK_BITS + TABLE_BITS + 16 * 8)
}

/// simdEntropyBits returns a lower bound of the entropy in bits of the counts h of n values.
/// log2 is within -1.88 and +0.5 of 256*log2, so n*log2(n) - sum(f*log2(f)) in 1/256 bits, as
/// lit_bits computes it, is at most 2.38n above 256 times the entropy.
fn entropy_bits(h: &[u64], n: usize) -> usize {
    let mut e = (n * log2(n as u64)) as isize - (5 * n).div_ceil(2) as isize;
    for &f in h {
        if f > 1 {
            e -= (f as usize * log2(f)) as isize;
        }
    }
    (e.max(0) >> 8) as usize
}

/// simdLog2 returns log2(x) for x > 0 in 1/256 units, from the 9 top bits of x.
fn log2(x: u64) -> usize {
    let b = 63 - x.leading_zeros() as usize;
    (b << 8) + LOG2_FRAC[((x << (63 - b)) >> 55 & 0xff) as usize] as usize
}

impl Coder {
    /// histogram sets hist[..syms] to the counts of v, whose values are below syms.
    fn histogram(&mut self, v: &[u8], syms: usize) {
        self.hist[..syms].fill(0);
        self.pv.histogram(&mut self.hist, v);
    }

    /// litBits returns the estimated size of lits in bits.
    fn lit_bits(&mut self, lits: &[u8]) -> usize {
        self.histogram(lits, 256);
        self.tmp[0].set_code(LITS, &self.hist);
        self.tmp[0].cost(&self.hist, 256).0
    }

    /// sampleSave returns need less the bytes that coding lits in blocks of up to
    /// pivco::MAX_BLOCK_SIZE saves, estimated from the entropy of 16 samples of 256 bytes spread
    /// over each block. It stops when need is met, as no block codes larger than stored. The
    /// samples' entropy is lower than the block's on average, so literals that coding would fit
    /// rarely fail.
    fn sample_save(&mut self, mut lits: &[u8], mut need: isize) -> isize {
        const RUNS: usize = 16;
        const RUN: usize = 4096 / RUNS;
        while !lits.is_empty() && need > 0 {
            let v = &lits[..lits.len().min(pivco::MAX_BLOCK_SIZE)];
            lits = &lits[v.len()..];
            let mut s = v;
            if v.len() > self.sample.len() {
                // One histogram of all samples is faster than one per sample.
                for i in 0..RUNS {
                    let at = i * (v.len() - RUN) / (RUNS - 1);
                    self.sample[i * RUN..(i + 1) * RUN].copy_from_slice(&v[at..at + RUN]);
                }
                s = &self.sample;
            }
            self.hist = [0; 256];
            self.pv.histogram(&mut self.hist, s);
            let bits = entropy_bits(&self.hist, s.len()) * v.len() / s.len();
            need -= v.len() as isize - (bits / 8) as isize;
        }
        need
    }

    /// put appends the values v of stream k to out, and their entry to recs. k is DELTA for delta
    /// literals. It takes the smallest of storing v, the current table and new tables for v.
    /// With fast_code, it takes the current table when that is smaller than storing,
    /// and otherwise only tries one new table.
    /// If v doesn't fit, the block fails.
    fn put(&mut self, out: &mut Out, recs: &mut Vec<u8>, k: usize, v: &[u8], fast_code: usize) {
        let layout = k;
        let k = k.min(LITS);
        // The values of the stream are below syms.
        let syms = 2 * CODE_BYTES[k];
        let (mut best, mut best_bits) = (None, 8 * v.len());
        let block_bits = if fast_code != 0 {
            FAST_BLOCK_BITS
        } else {
            BLOCK_BITS
        };
        // As MIN_CODED, for block_bits.
        if v.len() > block_bits / 7 {
            self.histogram(v, syms);
            if self.have[k] && take(&self.cur[k], &self.hist, syms, block_bits, &mut best_bits) {
                best = Some(Pick::Cur);
            }
            if fast_code == 0 || best.is_none() {
                // No prefix code codes v in fewer bits than its entropy, so new tables that can't
                // win with that aren't built. The entropy is at most MAX_ENTROPY per value, so it
                // is only computed when it can stop a table.
                let mut low = TABLE_BITS + block_bits;
                if low < best_bits
                    && best_bits
                        <= low + ((v.len() * MAX_ENTROPY[k] + 255) >> 8) + 8 * CODE_BYTES[k]
                {
                    low += entropy_bits(&self.hist[..syms], v.len());
                }
                if v.len() >= fast_code && low + 8 * CODE_BYTES[k] < best_bits {
                    let t = &mut self.tmp[0];
                    t.set_code(k, &self.hist);
                    let extra = TABLE_BITS + 8 * t.n + block_bits;
                    if take(t, &self.hist, syms, extra, &mut best_bits) {
                        best = Some(Pick::Tmp(0));
                    }
                }
                if (fast_code == 0 || v.len() < fast_code)
                    && low + 8 * CLASS_BYTES[layout] < best_bits
                {
                    let t = &mut self.tmp[1];
                    t.set_class(layout, &self.hist, syms);
                    let extra = TABLE_BITS + 8 * t.n + block_bits;
                    if take(t, &self.hist, syms, extra, &mut best_bits) {
                        best = Some(Pick::Tmp(1));
                    }
                }
            }
        }
        let room = out.room();
        if let Some(pick) = best {
            let t = match pick {
                Pick::Cur => &mut self.cur[k],
                Pick::Tmp(i) => &mut self.tmp[i],
            };
            let (mut mode, mut n) = (CURRENT, 0);
            if let Pick::Tmp(_) = pick {
                (mode, n) = (t.mode, t.n);
                // Tables are only built when used.
                t.t.set_lengths(&t.lens).expect("invalid table");
            }
            // Only tokens and literal blocks store their number of values (SPEC_SIMD.md 2.2).
            let hdr = if k == TOK || k == LITS { 2 } else { 0 };
            if let Some(size) = t.t.block_size(&self.hist[..syms]).map(|b| hdr + b)
                && size <= room
                && size + n < v.len()
            {
                let o = &mut out.b[out.n..out.n + size];
                if hdr > 0 {
                    o[..2].copy_from_slice(&(v.len() as u16).to_le_bytes());
                }
                if self.pv.encode(&mut o[hdr..], v, &t.t).is_ok() {
                    put_uvarint(recs, (size << 3 | mode) as u64);
                    out.n += size;
                    if let Pick::Tmp(i) = pick {
                        recs.extend_from_slice(&self.tmp[i].b[..n]);
                        self.cur[k].clone_from(&self.tmp[i]);
                        self.have[k] = true;
                    }
                    return;
                }
            }
        }
        if room < v.len() {
            out.limit = -1;
            return;
        }
        out.append(v);
        put_uvarint(recs, (v.len() as u64) << 3);
    }
}

/// The table put takes.
#[derive(Clone, Copy)]
enum Pick {
    Cur,
    Tmp(usize),
}

/// Go's try in put: reports whether t codes the histogram h[..syms], with extra bits, in fewer
/// bits than best_bits, which it then lowers to that.
fn take(t: &Table, h: &[u64; 256], syms: usize, extra: usize, best_bits: &mut usize) -> bool {
    match t.cost(h, syms) {
        (n, true) if n + extra < *best_bits => {
            *best_bits = n + extra;
            true
        }
        _ => false,
    }
}

impl Table {
    /// cost returns the size in bits of histogram h[..syms] coded with t, and false if a value
    /// has no code.
    fn cost(&self, h: &[u64; 256], syms: usize) -> (usize, bool) {
        let (mut n, mut ok) = (0u64, true);
        for (&f, &l) in h[..syms].iter().zip(&self.lens) {
            n += f * l as u64;
            ok = ok && (f == 0 || l != 0);
        }
        (n as usize, ok)
    }

    /// setCode sets t to the Huffman code of h, as code lengths of stream k. h must not be empty.
    fn set_code(&mut self, k: usize, h: &[u64; 256]) {
        let syms = 2 * CODE_BYTES[k];
        assert!(
            pivco::build_lengths(&h[..syms], &mut self.lens),
            "empty histogram"
        );
        self.mode = CODE_LENS;
        let lens = self.lens;
        self.set_nibbles(&lens[..syms]);
    }

    /// setClass sets t to the class lengths of h for layout k (SPEC_SIMD.md 2.3).
    /// The values of the stream are below syms.
    fn set_class(&mut self, k: usize, h: &[u64; 256], syms: usize) {
        let mut cl = [0u8; 24];
        let n;
        if k == TOK {
            n = tok_class_lens(h, &mut cl, &mut self.lens);
        } else {
            let lay = &LAYOUTS[k];
            let (mut w, mut lim) = ([0u64; 24], [0u8; 24]);
            for (v, &c) in lay.class[..syms].iter().enumerate() {
                if c != 0xff {
                    w[c as usize] += h[v];
                    lim[c as usize] = lim[c as usize].max(lay.inlen[v]);
                }
            }
            n = lay.n;
            for l in &mut lim[..n] {
                *l = pivco::MAX_CODE_LEN as u8 - *l;
            }
            limited_lens(&w[..n], &lim[..n], &mut cl[..n]);
            self.lens = [0; 256];
            for (v, &c) in lay.class[..syms].iter().enumerate() {
                if c != 0xff && cl[c as usize] != 0 {
                    self.lens[v] = cl[c as usize] + lay.inlen[v];
                }
            }
        }
        self.mode = if k == DELTA { DELTA_LENS } else { CLASS_LENS };
        self.set_nibbles(&cl[..n]);
    }

    fn set_nibbles(&mut self, l: &[u8]) {
        self.n = l.len().div_ceil(2);
        self.b[..self.n].fill(0);
        for (i, &v) in l.iter().enumerate() {
            self.b[i / 2] |= v << (4 * (i & 1));
        }
    }
}

/// simdTokClassLens sets cl to the token class lengths of h (SPEC_SIMD.md A.4),
/// and lens to the code lengths. It returns the number of class lengths.
fn tok_class_lens(h: &[u64; 256], cl: &mut [u8; 24], lens: &mut [u8; 256]) -> usize {
    const GROUPS: usize = 14;
    const MAX: u8 = pivco::MAX_CODE_LEN as u8;
    let grp = &LAYOUTS[TOK];
    let mut w = [0u64; GROUPS];
    let mut fw = [[0u64; 4]; 2];
    for v in 0..136 {
        w[grp.class[v >> 2] as usize] += h[v];
        fw[v >> 7][v & 3] += h[v];
    }
    // Flag codes for match length codes 0-31 and 32-33.
    let mut f = [[0u8; 4]; 2];
    let mut max_f = [0u8; 2];
    for i in 0..2 {
        limited_lens(&fw[i], &[3; 4], &mut f[i]);
        max_f[i] = *f[i].iter().max().unwrap();
    }
    let mut lim = [MAX; GROUPS];
    for ml in 0..34 {
        let c = grp.class[ml] as usize;
        lim[c] = lim[c].min(MAX - grp.inlen[ml] - max_f[ml >> 5]);
    }
    limited_lens(&w, &lim, &mut cl[..GROUPS]);
    cl[GROUPS..GROUPS + 4].copy_from_slice(&f[0]);
    cl[GROUPS + 4..GROUPS + 8].copy_from_slice(&f[1]);
    *lens = [0; 256];
    for v in 0..136 {
        let ml = v >> 2;
        let (g, fl) = (cl[grp.class[ml] as usize], f[ml >> 5][v & 3]);
        if g != 0 && fl != 0 {
            lens[v] = g + grp.inlen[ml] + fl;
        }
    }
    GROUPS + 8
}

/// simdLimitedLens sets l to the lengths of an optimal complete prefix code for the weights w,
/// with `l[i] <= lim[i]`, using package-merge. Weights of 0 get no code,
/// except that a code always has at least 2 lengths. w.len() must be at most 24.
fn limited_lens(w: &[u64], lim: &[u8], l: &mut [u8]) {
    l.fill(0);
    let mut sym = [0u8; 24];
    let (mut n, mut max_lim) = (0, 0u8);
    for (i, &x) in w.iter().enumerate() {
        if x > 0 {
            sym[n] = i as u8;
            n += 1;
            max_lim = max_lim.max(lim[i]);
        }
    }
    if n < 2 {
        for (li, &wi) in l.iter_mut().zip(w) {
            if wi > 0 || n < 2 {
                if wi == 0 {
                    n += 1;
                }
                *li = 1;
            }
        }
        return;
    }
    let s = &mut sym[..n];
    for i in 1..n {
        let mut j = i;
        while j > 0 && w[s[j] as usize] < w[s[j - 1] as usize] {
            s.swap(j, j - 1);
            j -= 1;
        }
    }
    // items holds the symbol of each item of a level, or -1 for a package of two items of the
    // level below. The selected items of a level are a prefix, and so are the items their
    // packages select below.
    let mut items = [[0i8; 48]; pivco::MAX_CODE_LEN + 1];
    let (mut prev, mut cur) = ([0u64; 48], [0u64; 48]);
    let mut np = 0;
    for lev in (1..=max_lim).rev() {
        let it = &mut items[lev as usize];
        let (mut nc, mut i, mut j) = (0, 0, 0);
        loop {
            while i < n && lim[s[i] as usize] < lev {
                i += 1;
            }
            if i < n && (j == np || w[s[i] as usize] <= prev[j]) {
                it[nc] = s[i] as i8;
                cur[nc] = w[s[i] as usize];
                nc += 1;
                i += 1;
            } else if j < np {
                it[nc] = -1;
                cur[nc] = prev[j];
                nc += 1;
                j += 1;
            } else {
                break;
            }
        }
        np = 0;
        for p in cur[..nc].as_chunks::<2>().0 {
            prev[np] = p[0] + p[1];
            np += 1;
        }
    }
    let (mut lev, mut k) = (1, 2 * n - 2);
    while k > 0 {
        let mut p = 0;
        for &x in &items[lev][..k] {
            if x < 0 {
                p += 1;
            } else {
                l[x as usize] += 1;
            }
        }
        k = 2 * p;
        lev += 1;
    }
}

#[cfg(test)]
mod tests;
