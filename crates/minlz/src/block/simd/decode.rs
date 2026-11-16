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

//! The SIMD block decoder, port of Go's `decode-simd.go` from `simdDecoder` on.
//! The operation loops are in [`super::exec`].

use core::cell::Cell;

use super::exec::{Tier, exec_exact, load64};
use super::{
    CLASS_LENS, CODE_BYTES, CODE_LENS, CURRENT, DELTA, DELTA_LENS, ESC, LAYOUTS, LITS, LL,
    MAX_CHUNK, MAX_CHUNKS, MAX_LEN, MAX_LIT_BLOCKS, OFF_TAB, OFFC, PAD, SLACK, STORED, TOK,
    rvarint,
};
use crate::Error;
use crate::block::format::get_uvarint;
use crate::pivco;
use crate::xxh3::Hasher;

/// The pivco decoder and the current table of each stream: what reading an entry needs.
#[derive(Default)]
pub(super) struct Tables {
    pv: pivco::Decoder,
    pub(super) tab: [pivco::Table; LITS + 1],
    have: [bool; LITS + 1],
}

/// The literals of a block decoded so far, `buf[..n]`. `buf` holds every literal of the block
/// and [`PAD`] bytes; the bytes after the literals are those of earlier blocks, except the
/// [`PAD`] bytes that [`Lits::fill`] zeroes.
#[derive(Default)]
pub(super) struct Lits {
    pub(super) buf: Vec<u8>,
    pub(super) n: usize,
    /// The literal blocks decoded so far.
    blocks: usize,
}

/// Decodes SIMD blocks (Go `simdDecoder`). It keeps its buffers between blocks: a thread
/// decodes with a cached one, so decoding does not allocate after warmup.
pub(super) struct Decoder {
    pub(super) tier: Tier,
    pub(super) t: Tables,
    pub(super) lits: Lits,
    tok: Box<[u8]>,
    ll: Box<[u8]>,
    offc: Box<[u8]>,
    esc: Box<[u8]>,
    h: Hasher,
}

/// The streams of a chunk (Go `simdChunk`). The stream slices stay readable past their values
/// (see the [memory rules](super::exec)): decoded streams have at least [`PAD`] bytes after
/// them, and stored streams run to the end of the block.
pub(super) struct Chunk<'a> {
    pub(super) tok: &'a [u8],
    pub(super) ll: &'a [u8],
    pub(super) offc: &'a [u8],
    pub(super) esc: &'a [u8],
    pub(super) raw: &'a [u8],
    pub(super) n_tok: usize,
    pub(super) n_ll: usize,
    pub(super) n_off: usize,
    pub(super) n_esc: usize,
    pub(super) bits: usize,
    pub(super) delta: bool,
    /// flat is set for chunks with the no-overlap flag. Their offsets are raised to the match
    /// length or 32, so copying 32 bytes at a time needs no overlap test. Longer matches can
    /// still overlap their output.
    pub(super) flat: bool,
}

/// Reads stream entries from recs, and their data from `body[pos..end]` (Go `simdReader`).
pub(super) struct Reader<'a> {
    pub(super) recs: &'a [u8],
    body: &'a [u8],
    pub(super) pos: usize,
    pub(super) end: usize,
}

thread_local! {
    static DECODERS: Cell<Option<Box<Decoder>>> = const { Cell::new(None) };
}

/// Decodes a SIMD block, from its flags byte, into dst, which must have the decoded size
/// (Go `decodeSIMD`). [`simd_header`](super::simd_header) returns both, and blocks must come
/// from it: only it checks the flags byte for reserved bits and chunk limit value 7.
pub(crate) fn decode_simd(dst: &mut [u8], block: &[u8]) -> Result<(), Error> {
    decode_simd_hash(dst, block, false).map(|_| ())
}

/// Is [`decode_simd`], which with hash also returns the XXH3 hash of dst (Go `decodeSIMDHash`).
/// It hashes the output of each chunk as soon as the chunk has run, while it is still in cache:
/// hashing a large block after decoding it reads it back from memory.
/// On error, dst may hold literals of earlier blocks, which the cached decoder keeps, so callers
/// must not expose it.
pub(crate) fn decode_simd_hash(dst: &mut [u8], block: &[u8], hash: bool) -> Result<u64, Error> {
    let mut s = DECODERS
        .try_with(Cell::take)
        .ok()
        .flatten()
        .unwrap_or_else(|| Box::new(Decoder::new()));
    s.tier = Tier::current();
    if hash {
        s.h.reset();
    }
    let ok = s.decode(dst, block, hash).is_some();
    let sum = if hash { s.h.sum64() } else { 0 };
    let _ = DECODERS.try_with(|c| c.set(Some(s)));
    if !ok {
        return Err(Error::Corrupt("invalid SIMD block"));
    }
    Ok(sum)
}

impl Decoder {
    pub(super) fn new() -> Self {
        let buf = |n| vec![0u8; n].into_boxed_slice();
        Decoder {
            tier: Tier::Go,
            t: Tables::default(),
            lits: Lits::default(),
            tok: buf(MAX_CHUNK + PAD),
            ll: buf(MAX_CHUNK + PAD),
            offc: buf(MAX_CHUNK + PAD),
            esc: buf(2 * MAX_CHUNK + PAD),
            h: Hasher::new(),
        }
    }

    /// Decodes block into dst, and adds dst to the hasher if hash.
    pub(super) fn decode(&mut self, dst: &mut [u8], block: &[u8], hash: bool) -> Option<()> {
        #[cfg(any(test, fuzzing))]
        if let Some(t) = super::exec::forced_pivco() {
            self.t.pv.set_tier(t);
        }
        let limit = (1024 << (block[0] & 7)).min(MAX_CHUNK);
        let body = &block[1..];
        let (lz, n) = rvarint(body)?;
        if lz > (body.len() - n) as u64 {
            return None;
        }
        let lz = lz as usize;
        let mut lr = self.literals(&body[lz..body.len() - n], dst.len())?;
        // The first operation has literals, so offset 0 only occurs as a delta reference at
        // position 0.
        dst[0] = 0;
        let mut st = [0, 0, 1, 0];
        if lz > 0 {
            self.ops(dst, body, lz, limit, &mut st, &mut lr, hash)?;
        }
        if !lr.recs.is_empty() {
            self.lits
                .fill(&mut self.t, &mut lr, 0, usize::MAX, dst.len())?;
        }
        let (d, l, n_lits) = (st[0], st[1], self.lits.n);
        if n_lits == 0 || lr.pos != lr.end || d + n_lits - l != dst.len() {
            return None;
        }
        dst[d..].copy_from_slice(&self.lits.buf[l..n_lits]);
        if hash {
            // ops added dst[..d].
            self.h.write(&dst[d..]);
        }
        Some(())
    }

    /// Starts the literal section sec of a block of size bytes, and returns the reader of its
    /// literal blocks.
    pub(super) fn literals<'a>(&mut self, sec: &'a [u8], size: usize) -> Option<Reader<'a>> {
        self.t.have = Default::default();
        let (t, n) = rvarint(sec)?;
        if t > (sec.len() - n) as u64 {
            return None;
        }
        let end = sec.len() - n - t as usize;
        if self.lits.buf.len() < size + PAD {
            self.lits.buf = vec![0; size + PAD];
        }
        (self.lits.n, self.lits.blocks) = (0, 0);
        Some(Reader {
            recs: &sec[end..sec.len() - n],
            body: sec,
            pos: 0,
            end,
        })
    }

    /// Runs the chunks of the LZ section `body[..end]`, with the literal blocks of lr.
    /// It adds the output of each chunk to the hasher if hash.
    #[allow(clippy::too_many_arguments)]
    fn ops(
        &mut self,
        dst: &mut [u8],
        body: &[u8],
        end: usize,
        limit: usize,
        st: &mut [usize; 4],
        lr: &mut Reader,
        hash: bool,
    ) -> Option<()> {
        let (t, n) = rvarint(&body[..end])?;
        if t > (end - n) as u64 {
            return None;
        }
        let t = t as usize;
        let mut r = Reader {
            recs: &body[end - n - t..end - n],
            body,
            pos: 0,
            end: end - n - t,
        };
        let tier = self.tier;
        let mut i = 0;
        while let Some((&flags, recs)) = r.recs.split_first() {
            r.recs = recs;
            if flags & 0xf0 != 0 || i == MAX_CHUNKS {
                return None;
            }
            let (n_tok, v) = self.t.stream(&mut r, TOK, 1, limit, &mut self.tok)?;
            let tok = v.unwrap_or(&self.tok[..]);
            let (mut out, n_lit, n_off, mut n_esc, ok) = tok_stats(tier, &tok[..n_tok]);
            if !ok || i == 0 && tok[0] & 2 == 0 {
                return None;
            }
            let (n_ll, v) = self.t.stream(&mut r, LL, n_lit, n_lit, &mut self.ll)?;
            let ll = v.unwrap_or(&self.ll[..]);
            let (sum_ll, n_ll_esc, ok) = ll_stats(tier, &ll[..n_lit]);
            if !ok {
                return None;
            }
            let (_, v) = self.t.stream(&mut r, OFFC, n_off, n_off, &mut self.offc)?;
            let offc = v.unwrap_or(&self.offc[..]);
            let (bits, ok) = off_stats(tier, &offc[..n_off]);
            if !ok {
                return None;
            }
            n_esc += n_ll_esc;
            let (_, v) = self.t.stream(&mut r, ESC, n_esc, n_esc, &mut self.esc)?;
            let esc = v.unwrap_or(&self.esc[..]);
            // Escapes may be any value, so only their sum is used.
            let (sum_esc, _, _) = ll_stats(tier, &esc[..n_esc]);
            out += sum_ll + sum_esc;
            let (e, m) = get_uvarint(r.recs)?;
            let nb = bits.div_ceil(8);
            if e != (nb as u64) << 3 || nb > r.end - r.pos || st[0] + out > dst.len() || out < n_tok
            {
                return None;
            }
            r.recs = &r.recs[m..];
            let c = Chunk {
                tok,
                ll,
                offc,
                esc,
                raw: &body[r.pos..],
                n_tok,
                n_ll,
                n_off,
                n_esc,
                bits,
                delta: flags & 1 != 0,
                flat: flags & 2 != 0,
            };
            r.pos += nb;
            debug_assert!(c.readable() && self.lits.buf.len() >= dst.len() + PAD);
            // Each literal escape adds at most 255; the escapes of long matches can add far more.
            let need = sum_ll + sum_esc.min(255 * n_ll_esc);
            if st[1] + need > self.lits.n {
                self.lits.fill(&mut self.t, lr, st[1], need, dst.len())?;
            }
            let d = st[0];
            // SAFETY: the tier comes from Tier::current, the stats above counted c's streams and
            // out, and the literal buffer has PAD bytes past dst.len(), which the literals used
            // never pass.
            unsafe { c.run(tier, dst, &self.lits.buf, out, st) };
            // Chunks only use more literals than are decoded once the literal blocks run out.
            if st[3] != 0 || st[1] > self.lits.n {
                return None;
            }
            if hash {
                // The output before st[0] is final: the loops only write past it.
                self.h.write(&dst[d..st[0]]);
            }
            i += 1;
        }
        (r.pos == r.end).then_some(())
    }
}

impl Lits {
    /// Decodes literal blocks of r until need literals follow position l, or none are left.
    /// Decoding them just before the chunk that reads them keeps them in cache.
    pub(super) fn fill(
        &mut self,
        t: &mut Tables,
        r: &mut Reader,
        l: usize,
        need: usize,
        size: usize,
    ) -> Option<()> {
        while !r.recs.is_empty() && self.n - l < need {
            if self.blocks == MAX_LIT_BLOCKS {
                return None;
            }
            self.blocks += 1;
            let n = self.n;
            let hi = pivco::MAX_BLOCK_SIZE.min(size - n);
            let (nv, v) = t.stream(r, LITS, 1, hi, &mut self.buf[n..])?;
            if let Some(v) = v {
                self.buf[n..n + nv].copy_from_slice(&v[..nv]);
            }
            self.n += nv;
        }
        // The loops read up to 32 bytes past the literals of a chunk.
        self.buf[self.n..self.n + PAD].fill(0);
        Some(())
    }
}

impl Tables {
    /// Reads the entry of stream k, which must have lo to hi values, and returns their number
    /// and, for a stored entry, the values: the data from the entry on, readable to the end of
    /// the section. Coded values are decoded into buf.
    pub(super) fn stream<'a>(
        &mut self,
        r: &mut Reader<'a>,
        k: usize,
        lo: usize,
        hi: usize,
        buf: &mut [u8],
    ) -> Option<(usize, Option<&'a [u8]>)> {
        let (e, m) = get_uvarint(r.recs)?;
        if e >> 3 > (r.end - r.pos) as u64 {
            return None;
        }
        r.recs = &r.recs[m..];
        let (size, mode) = ((e >> 3) as usize, (e & 7) as usize);
        let data = &r.body[r.pos..];
        r.pos += size;
        match mode {
            STORED => return (size >= lo && size <= hi).then_some((size, Some(data))),
            CURRENT if !self.have[k] => return None,
            CURRENT => {}
            CODE_LENS | CLASS_LENS | DELTA_LENS => {
                if mode == DELTA_LENS && k != LITS {
                    return None;
                }
                let t = self.set_table(k, mode, r.recs)?;
                r.recs = &r.recs[t..];
            }
            _ => return None,
        }
        // Only tokens and literal blocks store their number of values; lo == hi for the others.
        let (mut n, mut data) = (lo, &data[..size]);
        if k == TOK || k == LITS {
            if size < 2 {
                return None;
            }
            n = u16::from_le_bytes([data[0], data[1]]) as usize;
            data = &data[2..];
            if n < lo || n > hi {
                return None;
            }
        }
        self.pv.decode(&mut buf[..n], data, &self.tab[k]).ok()?;
        Some((n, None))
    }

    /// Reads a new table for stream k from b and returns its size.
    pub(super) fn set_table(&mut self, k: usize, mode: usize, b: &[u8]) -> Option<usize> {
        let mut lens = [0u8; 256];
        let n = if mode == CODE_LENS {
            let n = CODE_BYTES[k];
            unpack(&mut lens, b.get(..n)?);
            if (k == LL || k == OFFC) && lens[0] != 0 {
                return None;
            }
            n
        } else if k == TOK {
            let mut g = [0u8; 22];
            let n = g.len() / 2;
            unpack(&mut g, b.get(..n)?);
            if !complete(&g[..14]) || !complete(&g[14..18]) || !complete(&g[18..]) {
                return None;
            }
            let grp = &LAYOUTS[TOK];
            for (v, l) in lens[..136].iter_mut().enumerate() {
                let ml = v >> 2;
                let (gl, f) = (g[grp.class[ml] as usize], g[14 + 4 * (v >> 7) + (v & 3)]);
                *l = (gl + grp.inlen[ml] + f)
                    & ((gl + 15) >> 4).wrapping_neg()
                    & ((f + 15) >> 4).wrapping_neg();
            }
            n
        } else {
            let lay = &LAYOUTS[if mode == DELTA_LENS { DELTA } else { k }];
            // Class 0xff, outside the stream's range, has length 0.
            let mut cl = [0u8; 256];
            let n = lay.n.div_ceil(2);
            if b.len() < n || lay.n & 1 != 0 && b[n - 1] >> 4 != 0 {
                return None;
            }
            unpack(&mut cl, &b[..n]);
            // Branch-free: which classes have codes is data. x+15 has bit 4 set unless x is 0.
            for (v, &c) in lay.class[..2 * CODE_BYTES[k.min(LITS)]].iter().enumerate() {
                let x = cl[c as usize];
                lens[v] = (x + lay.inlen[v]) & ((x + 15) >> 4).wrapping_neg();
            }
            n
        };
        // Lengths are at most 22, a class length of 15 plus an in-class length of up to 7, so
        // adding 15 to a byte doesn't carry out of it, and the sum of 8 lengths fits a byte.
        // used misses lengths above 16, but set_lengths rejects any length above 11.
        let (mut used, mut sum) = (0, 0);
        for i in (0..2 * CODE_BYTES[k.min(LITS)]).step_by(8) {
            let x = load64(&lens, i);
            used += ((x + 0x0f0f0f0f0f0f0f0f) & 0x1010101010101010).count_ones();
            sum += x.wrapping_mul(0x0101010101010101) >> 56;
        }
        // pivco accepts any length for a lone value.
        if used == 1 && sum != 1 || self.tab[k].set_lengths(&lens).is_err() {
            return None;
        }
        self.have[k] = true;
        Some(n)
    }
}

/// Splits each byte of b into two nibbles of l, low nibble first.
fn unpack(l: &mut [u8], b: &[u8]) {
    for (i, &v) in b.iter().enumerate() {
        l[2 * i] = v & 15;
        l[2 * i + 1] = v >> 4;
    }
}

/// Reports whether the lengths l, up to 15, form a complete code (Go `simdComplete`).
pub(super) fn complete(l: &[u8]) -> bool {
    let k: usize = l.iter().filter(|&&v| v != 0).map(|&v| 1 << (15 - v)).sum();
    k == 1 << 15
}

const LO: u64 = 0x0101010101010101;
const LO7: u64 = 0x7f7f7f7f7f7f7f7f;
const HI: u64 = 0x8080808080808080;

/// Returns the summed match length codes of tok, and the number of operations with literals,
/// with new offsets and with escaped matches. ok is false for tokens above 135
/// (Go `simdTokStats`).
pub(super) fn tok_stats(tier: Tier, tok: &[u8]) -> (usize, usize, usize, usize, bool) {
    let (mut out, mut n_lit, mut n_rep, mut n_esc, mut i, ok) = tier.tok_stats_asm(tok);
    let mut bad = if ok { 0 } else { HI };
    while i + 8 <= tok.len() {
        let x = load64(tok, i);
        n_lit += (x & (LO << 1)).count_ones() as usize;
        n_rep += (x & LO).count_ones() as usize;
        // Match length code 33 is 132-135: bits 7 and 2.
        n_esc += (x & (x << 5) & HI).count_ones() as usize;
        bad |= x & ((x & 0x7878787878787878) + 0x7878787878787878);
        let y = (x >> 2) & 0x3f3f3f3f3f3f3f3f;
        let y = (y & 0x00ff00ff00ff00ff) + ((y >> 8) & 0x00ff00ff00ff00ff);
        out += (y.wrapping_mul(0x0001000100010001) >> 48) as usize;
        i += 8;
    }
    for &t in &tok[i..] {
        out += (t >> 2) as usize;
        n_lit += ((t >> 1) & 1) as usize;
        n_rep += (t & 1) as usize;
        n_esc += ((t >> 7) & (t >> 2) & 1) as usize;
        bad |= t as u64 & ((t as u64 & 0x78) + 0x78);
    }
    (out, n_lit, tok.len() - n_rep, n_esc, bad & HI == 0)
}

/// Returns the sum of the literal lengths ll, and the number of escapes.
/// ok is false for lengths outside 1-33 (Go `simdLLStats`).
pub(super) fn ll_stats(tier: Tier, ll: &[u8]) -> (usize, usize, bool) {
    let (mut sum, mut n_esc, mut i, ok) = tier.ll_stats_asm(ll);
    let mut bad = if ok { 0 } else { HI };
    while i + 8 <= ll.len() {
        let x = load64(ll, i);
        let z = x ^ 0x2121212121212121;
        n_esc += (!(((z & LO7) + LO7) | z) & HI).count_ones() as usize;
        bad |= !(((x & LO7) + LO7) | x) | ((x & LO7) + 0x5e5e5e5e5e5e5e5e) | x;
        let y = (x & 0x00ff00ff00ff00ff) + ((x >> 8) & 0x00ff00ff00ff00ff);
        sum += (y.wrapping_mul(0x0001000100010001) >> 48) as usize;
        i += 8;
    }
    for &v in &ll[i..] {
        sum += v as usize;
        if v as usize == MAX_LEN + 1 {
            n_esc += 1;
        }
        if v.wrapping_sub(1) as usize > MAX_LEN {
            bad = HI;
        }
    }
    (sum, n_esc, bad & HI == 0)
}

/// Returns the number of raw bits of the offset symbols offc.
/// ok is false for symbols outside 1-175 (Go `simdOffStats`).
pub(super) fn off_stats(tier: Tier, offc: &[u8]) -> (usize, bool) {
    let (mut n, mut i, ok) = tier.off_stats_asm(offc);
    let mut bad = if ok { 0 } else { HI };
    while i + 8 <= offc.len() {
        let x = load64(offc, i);
        // Zero bytes, and bytes of 176 and above.
        bad |= !(((x & LO7) + LO7) | x) | (x & ((x & LO7) + 0x5050505050505050));
        // The raw bits of symbol s are max(s>>3 - 1, 0).
        let y = (x >> 3) & 0x1f1f1f1f1f1f1f1f;
        n += (y.wrapping_mul(0x0101010101010101) >> 56) as usize;
        n -= ((((y & LO7) + LO7) | y) & HI).count_ones() as usize;
        i += 8;
    }
    for &s in &offc[i..] {
        n += (OFF_TAB[s as usize] >> 24) as usize;
        if s.wrapping_sub(1) >= 175 {
            bad = HI;
        }
    }
    (n, bad & HI == 0)
}

impl Chunk<'_> {
    /// Executes the operations of c, which output out bytes.
    ///
    /// # Safety
    ///
    /// tier comes from [`Tier::best`] or [`Tier::tiers`], c's counts and out are what its
    /// tokens read and output (as `tok_stats`, `ll_stats` and `off_stats` count them), and lits
    /// is readable [`SLACK`] bytes past the literals they use.
    pub(super) unsafe fn run(
        &self,
        tier: Tier,
        dst: &mut [u8],
        lits: &[u8],
        out: usize,
        st: &mut [usize; 4],
    ) {
        let (mut k, mut cur) = (self.n_tok, [self.n_ll, self.n_off, self.n_esc, self.bits]);
        let end = st[0] + out;
        if end + SLACK > dst.len() {
            (k, cur) = self.split(end, dst.len());
        }
        if !self.flat {
            if k > 0 {
                // SAFETY: the caller's (see # Safety); the split keeps rule 1.
                unsafe { tier.exec(dst, lits, self, k, st) };
            }
        } else if k > 0 {
            // Before position MAX_LEN, clamping can still make a match overlap, or a delta
            // reference repeat.
            let (mut k0, mut cur0) = (0, [0; 4]);
            if st[0] < MAX_LEN {
                (k0, cur0) = self.head(st[0], k);
                exec_exact(dst, lits, self, 0, k0, [0; 4], st);
            }
            if k > k0 {
                // SAFETY: as for tier.exec.
                unsafe { tier.exec_flat(dst, lits, self, k0, k, cur0, st) };
            }
        }
        debug_assert!(k == 0 || st[0] + SLACK <= dst.len());
        if k < self.n_tok {
            exec_exact(dst, lits, self, k, self.n_tok, cur, st);
        }
    }

    /// Returns how many of the first k operations start before position [`MAX_LEN`], with the
    /// chunk starting at output position d, and the stream positions after them.
    pub(super) fn head(&self, mut d: usize, k: usize) -> (usize, [usize; 4]) {
        let (mut li, mut oi, mut gi, mut pos, mut n) = (0, 0, 0, 0, 0);
        while n < k && d < MAX_LEN {
            let t = self.tok[n];
            if t & 2 != 0 {
                let mut ll = self.ll[li] as usize;
                li += 1;
                if ll > MAX_LEN {
                    ll += self.esc[gi] as usize;
                    gi += 1;
                }
                d += ll;
            }
            let mut ml = (t >> 2) as usize;
            if ml > MAX_LEN {
                ml += self.esc[gi] as usize;
                gi += 1;
            }
            if t & 1 == 0 {
                pos += (OFF_TAB[self.offc[oi] as usize] >> 24) as usize;
                oi += 1;
            }
            d += ml;
            n += 1;
        }
        (n, [li, oi, gi, pos])
    }

    /// Returns how many operations of c end at least [`SLACK`] bytes before limit, and the
    /// stream positions after them. end is the output position after c.
    pub(super) fn split(&self, mut end: usize, limit: usize) -> (usize, [usize; 4]) {
        let (mut li, mut oi, mut gi, mut pos) = (self.n_ll, self.n_off, self.n_esc, self.bits);
        let mut k = self.n_tok;
        while k > 0 && end + SLACK > limit {
            k -= 1;
            let t = self.tok[k];
            let mut n = (t >> 2) as usize;
            // The match escape follows the literal length escape.
            if n > MAX_LEN {
                gi -= 1;
                n += self.esc[gi] as usize;
            }
            if t & 2 != 0 {
                li -= 1;
                let mut ll = self.ll[li] as usize;
                if ll > MAX_LEN {
                    gi -= 1;
                    ll += self.esc[gi] as usize;
                }
                n += ll;
            }
            if t & 1 == 0 {
                oi -= 1;
                pos -= (OFF_TAB[self.offc[oi] as usize] >> 24) as usize;
            }
            end -= n;
        }
        (k, [li, oi, gi, pos])
    }
}
