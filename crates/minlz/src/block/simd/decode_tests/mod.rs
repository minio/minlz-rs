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

//! Port of Go's `decode-simd_test.go`: the test block builder, written from `SPEC_SIMD.md`
//! independently of the encoder, and the decoder tests (`block.rs`, `exec.rs`, `fuzz.rs`).

mod block;
mod exec;
mod fuzz;

use super::decode::{decode_simd, decode_simd_hash};
use super::{
    BLOCK_TYPE, CLASS_LENS, CODE_BYTES, CODE_LENS, CURRENT, DELTA, DELTA_LENS, ESC, LAYOUTS, LITS,
    LL, MAX_LEN, MIN_SAVING, OFFC, STORED, TOK, simd_header,
};
use crate::block::format::put_uvarint;
use crate::pivco;

/// SplitMix64. Go's tests use math/rand; the exact values don't matter.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn uint64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub(super) fn intn(&mut self, n: usize) -> usize {
        (self.uint64() % n as u64) as usize
    }

    pub(super) fn read(&mut self, b: &mut [u8]) {
        for c in b.chunks_mut(8) {
            let v = self.uint64().to_le_bytes();
            c.copy_from_slice(&v[..c.len()]);
        }
    }

    pub(super) fn perm(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            p.swap(i, self.intn(i + 1));
        }
        p
    }
}

/// Copies ll literals, then ml bytes from offset off back. off 0 repeats the offset.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct TestOp {
    pub(super) ll: usize,
    pub(super) ml: usize,
    pub(super) off: usize,
}

pub(super) const fn op(ll: usize, ml: usize, off: usize) -> TestOp {
    TestOp { ll, ml, off }
}

/// Executes ops as specified, and appends the literals left. The operations are in chunks of
/// chunk, and chunk i has the no-overlap flag if bit i%64 of flat is set.
pub(super) fn test_run(ops: &[TestOp], lits: &[u8], chunk: usize, flat: u64) -> Vec<u8> {
    let mut out = Vec::new();
    let (mut l, mut last) = (0, 1);
    for (i, o) in ops.iter().enumerate() {
        out.extend_from_slice(&lits[l..l + o.ll]);
        l += o.ll;
        if o.off != 0 {
            last = o.off;
        }
        let mut r = last.min(out.len());
        if (flat >> ((i / chunk) & 63)) & 1 != 0 {
            r = last.max(o.ml.min(MAX_LEN)).min(out.len());
        }
        for _ in 0..o.ml {
            // Only invalid blocks match at position 0.
            out.push(if r == 0 { 0 } else { out[out.len() - r] });
        }
    }
    out.extend_from_slice(&lits[l..]);
    out
}

/// Appends the reverse varint of v to b.
pub(super) fn test_rvarint(b: &mut Vec<u8>, v: u64) {
    let start = b.len();
    put_uvarint(b, v);
    b[start..].reverse();
}

pub(super) fn test_nibbles(l: &[u8]) -> Vec<u8> {
    let mut b = vec![0; l.len().div_ceil(2)];
    for (i, &v) in l.iter().enumerate() {
        b[i / 2] |= v << (4 * (i & 1));
    }
    b
}

#[derive(Clone, Debug, Default)]
pub(super) struct TestStream {
    pub(super) mode: usize,
    pub(super) table: Vec<u8>,
    pub(super) data: Vec<u8>,
}

impl TestStream {
    fn append_entry(&self, b: &mut Vec<u8>) {
        put_uvarint(b, (self.data.len() as u64) << 3 | self.mode as u64);
        b.extend_from_slice(&self.table);
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct TestChunk {
    pub(super) flags: u8,
    pub(super) s: [TestStream; 5],
    pub(super) ops: Vec<TestOp>,
    pub(super) vals: [Vec<u8>; 5],
}

/// A block in parts, so tests can change them.
#[derive(Clone, Debug, Default)]
pub(super) struct TestEnc {
    pub(super) typ: u64,
    pub(super) size: usize,
    pub(super) flags: u8,
    pub(super) lits: Vec<TestStream>,
    pub(super) chunks: Vec<TestChunk>,
    pub(super) no_lz: bool,
    /// Literal data without an entry.
    pub(super) lit_extra: Vec<u8>,
    pub(super) lz_extra: Vec<u8>,
    /// Chunk data bytes dropped from the end, entries unchanged.
    pub(super) lz_cut: usize,
    pub(super) lit_rec_extra: Vec<u8>,
    pub(super) chunk_rec_extra: Vec<u8>,
    /// Literals as stored.
    pub(super) stored: Vec<u8>,
}

impl TestEnc {
    pub(super) fn bytes(&self) -> Vec<u8> {
        let mut out = vec![0];
        put_uvarint(&mut out, self.typ << 24 | self.size as u64);
        out.push(self.flags);
        let start = out.len();
        let mut recs = Vec::new();
        if !self.no_lz {
            for c in &self.chunks {
                recs.push(c.flags);
                for s in &c.s {
                    out.extend_from_slice(&s.data);
                    s.append_entry(&mut recs);
                }
            }
            out.truncate(out.len() - self.lz_cut);
            out.extend_from_slice(&self.lz_extra);
            recs.extend_from_slice(&self.chunk_rec_extra);
            out.extend_from_slice(&recs);
            test_rvarint(&mut out, recs.len() as u64);
        }
        let lz = out.len() - start;
        recs.clear();
        for s in &self.lits {
            out.extend_from_slice(&s.data);
            s.append_entry(&mut recs);
        }
        out.extend_from_slice(&self.lit_extra);
        recs.extend_from_slice(&self.lit_rec_extra);
        out.extend_from_slice(&recs);
        test_rvarint(&mut out, recs.len() as u64);
        test_rvarint(&mut out, lz as u64);
        out
    }
}

/// Describes a block for [`TestBuilder::build`].
#[derive(Clone, Debug, Default)]
pub(super) struct TestBlock {
    pub(super) ops: Vec<TestOp>,
    /// Output literal values.
    pub(super) lits: Vec<u8>,
    /// Operations per chunk.
    pub(super) chunk: usize,
    /// Chunk limit value.
    pub(super) limit: usize,
    /// Chunk i has delta literals if bit i%64 is set.
    pub(super) delta: u64,
    /// Chunk i has the no-overlap flag if bit i%64 is set.
    pub(super) flat: u64,
    /// Literals per literal block.
    pub(super) lit_block: usize,
    /// Entry modes, cycled.
    pub(super) modes: Vec<usize>,
    pub(super) lit_modes: Vec<usize>,
}

#[derive(Default)]
pub(super) struct TestBuilder {
    enc: pivco::Encoder,
    cur: [Option<pivco::Table>; LITS + 1],
}

impl TestBuilder {
    /// Returns v coded with t as stored in an entry of stream k: a pivco block, after the number
    /// of values for tokens and literals (SPEC_SIMD.md 2.2).
    pub(super) fn code(
        &mut self,
        k: usize,
        v: &[u8],
        t: &pivco::Table,
    ) -> Result<Vec<u8>, pivco::Error> {
        let mut h = [0u64; 256];
        for &c in v {
            h[c as usize] += 1;
        }
        let size = t.block_size(&h).ok_or(pivco::Error::SymbolNotInTable)?;
        let mut d = Vec::new();
        if k == TOK || k == LITS {
            d.extend_from_slice(&(v.len() as u16).to_le_bytes());
        }
        let n = d.len();
        d.resize(n + size, 0);
        self.enc.encode(&mut d[n..], v, t)?;
        Ok(d)
    }

    /// Codes the values of stream k with mode, or stores them if that isn't possible.
    pub(super) fn stream(&mut self, k: usize, mode: usize, v: &[u8]) -> TestStream {
        let mut s = TestStream {
            mode,
            ..Default::default()
        };
        if v.is_empty() || mode == STORED || mode == DELTA_LENS && k != LITS {
            (s.mode, s.data) = (STORED, v.to_vec());
            return s;
        }
        if mode == CURRENT
            && let Some(t) = self.cur[k].clone()
            && let Ok(d) = self.code(k, v, &t)
        {
            s.data = d;
            return s;
        }
        let mut lens = [0u8; 256];
        if mode == CLASS_LENS || mode == DELTA_LENS {
            let cl;
            (cl, lens) = test_class_lens(k, mode);
            s.table = test_nibbles(&cl);
        } else {
            s.mode = CODE_LENS;
            let mut h = [0u64; 256];
            for &c in v {
                h[c as usize] += 1;
            }
            assert!(pivco::build_lengths(&h, &mut lens), "no values");
            s.table = test_nibbles(&lens[..2 * CODE_BYTES[k]]);
        }
        let mut t = pivco::Table::default();
        t.set_lengths(&lens).unwrap();
        s.data = self.code(k, v, &t).unwrap();
        self.cur[k] = Some(t);
        s
    }

    /// Returns the block parts and the decoded output.
    pub(super) fn build(&mut self, tb: &TestBlock) -> (TestEnc, Vec<u8>) {
        self.cur = Default::default();
        let chunk = if tb.chunk == 0 {
            1024 << tb.limit
        } else {
            tb.chunk
        };
        let out = test_run(&tb.ops, &tb.lits, chunk, tb.flat);
        let lit_block = if tb.lit_block == 0 {
            pivco::MAX_BLOCK_SIZE
        } else {
            tb.lit_block
        };
        let or_stored = |m: &[usize]| {
            if m.is_empty() {
                vec![STORED]
            } else {
                m.to_vec()
            }
        };
        let (modes, lit_modes) = (or_stored(&tb.modes), or_stored(&tb.lit_modes));
        let mut e = TestEnc {
            typ: BLOCK_TYPE,
            size: out.len(),
            flags: tb.limit as u8,
            stored: tb.lits.clone(),
            ..Default::default()
        };
        let (mut p, mut l, mut last) = (0, 0, 1);
        for (i, o) in tb.ops.iter().enumerate() {
            let ci = i / chunk;
            if tb.delta >> (ci & 63) & 1 != 0 {
                let r = if tb.flat >> (ci & 63) & 1 != 0 {
                    last.max(MAX_LEN).min(p)
                } else {
                    last.min(p)
                };
                for j in 0..o.ll.min(32) {
                    if r > 0 {
                        e.stored[l + j] = e.stored[l + j].wrapping_sub(out[p - r + j % r]);
                    }
                }
            }
            p += o.ll + o.ml;
            l += o.ll;
            if o.off != 0 {
                last = o.off;
            }
        }
        for i in (0..e.stored.len()).step_by(lit_block) {
            let v = e.stored[i..(i + lit_block).min(e.stored.len())].to_vec();
            let s = self.stream(LITS, lit_modes[e.lits.len() % lit_modes.len()], &v);
            e.lits.push(s);
        }
        for i in (0..tb.ops.len()).step_by(chunk) {
            let mut c = TestChunk {
                ops: tb.ops[i..(i + chunk).min(tb.ops.len())].to_vec(),
                ..Default::default()
            };
            let ci = e.chunks.len();
            if tb.delta >> (ci & 63) & 1 != 0 {
                c.flags = 1;
            }
            if tb.flat >> (ci & 63) & 1 != 0 {
                c.flags |= 2;
            }
            c.vals = test_values(&c.ops);
            for k in 0..4 {
                c.s[k] = self.stream(k, modes[(ci + k) % modes.len()], &c.vals[k]);
            }
            c.s[4].data = c.vals[4].clone();
            e.chunks.push(c);
        }
        (e, out)
    }
}

/// Returns class lengths that give every value of stream k a code.
pub(super) fn test_class_lens(k: usize, mode: usize) -> (Vec<u8>, [u8; 256]) {
    let trunc = |n: usize| -> Vec<u8> {
        let k = n.ilog2() as usize;
        (0..n)
            .map(|i| k as u8 + u8::from(i >= (1 << (k + 1)) - n))
            .collect()
    };
    let lay = &LAYOUTS[if mode == DELTA_LENS { DELTA } else { k }];
    let mut cl = trunc(lay.n);
    let mut lens = [0u8; 256];
    if k == TOK {
        cl.extend_from_slice(&[2; 8]);
        for (v, l) in lens[..136].iter_mut().enumerate() {
            *l = cl[lay.class[v >> 2] as usize] + lay.inlen[v >> 2] + 2;
        }
        return (cl, lens);
    }
    for (v, &c) in lay.class.iter().enumerate() {
        if c != 0xff {
            lens[v] = cl[c as usize] + lay.inlen[v];
        }
    }
    (cl, lens)
}

/// Returns the offset symbol of off, its raw bits and their number.
pub(super) fn test_offset(off: usize) -> (u8, u64, usize) {
    if off < 16 {
        return (off as u8, 0, 0);
    }
    let b = off.ilog2() as usize;
    (
        (8 * b - 16 + ((off >> (b - 3)) & 7)) as u8,
        off as u64 & ((1 << (b - 3)) - 1),
        b - 3,
    )
}

/// Returns the values of the streams of ops.
pub(super) fn test_values(ops: &[TestOp]) -> [Vec<u8>; 5] {
    let mut s: [Vec<u8>; 5] = Default::default();
    let (mut acc, mut nacc) = (0u64, 0);
    for o in ops {
        let (mut ll, mut ml) = (o.ll, o.ml);
        let mut t = 0u8;
        if ll > 0 {
            t |= 2;
            if ll > MAX_LEN {
                s[ESC].push((ll - MAX_LEN - 1) as u8);
                ll = MAX_LEN + 1;
            }
            s[LL].push(ll as u8);
        }
        if ml > MAX_LEN {
            s[ESC].push((ml - MAX_LEN - 1) as u8);
            ml = MAX_LEN + 1;
        }
        t |= (ml << 2) as u8;
        if o.off == 0 {
            t |= 1;
        } else {
            let (sym, raw, n) = test_offset(o.off);
            s[OFFC].push(sym);
            acc |= raw << nacc;
            nacc += n;
            while nacc >= 8 {
                s[4].push(acc as u8);
                acc >>= 8;
                nacc -= 8;
            }
        }
        s[TOK].push(t);
    }
    if nacc > 0 {
        s[4].push(acc as u8);
    }
    s
}

/// Returns random operations and their literals. Literals are drawn from alphabet, matches are
/// at most max_ml and offsets at most max_off.
pub(super) fn test_ops(
    rng: &mut Rng,
    n: usize,
    alphabet: &[u8],
    max_ml: usize,
    max_off: usize,
) -> (Vec<TestOp>, Vec<u8>) {
    let (mut ops, mut lits) = (Vec::new(), Vec::new());
    let (mut pos, mut last) = (0, 1);
    for i in 0..n {
        let mut o = TestOp::default();
        if i == 0 || rng.intn(3) > 0 {
            o.ll = 1 + rng.intn(8);
            if rng.intn(20) == 0 {
                o.ll = 1 + rng.intn(MAX_LEN + 256);
            }
        }
        for _ in 0..o.ll {
            lits.push(alphabet[rng.intn(alphabet.len())]);
        }
        pos += o.ll;
        o.ml = rng.intn(max_ml.min(MAX_LEN) + 1);
        if rng.intn(10) == 0 {
            o.ml = rng.intn(max_ml + 1);
        }
        if rng.intn(4) == 0 {
            // Repeat.
        } else if rng.intn(20) == 0 {
            // Clamped to the position.
            o.off = 1 + rng.intn(max_off.min((1 << 24) - 1));
        } else {
            o.off = 1 + rng.intn(pos.min(max_off).max(1));
        }
        let r = if o.off != 0 { o.off } else { last };
        if o.ml > MAX_LEN && r.min(pos) < MAX_LEN {
            o.ml = MAX_LEN;
        }
        if o.off != 0 {
            last = o.off;
        }
        pos += o.ml;
        ops.push(o);
    }
    for _ in 0..rng.intn(50) {
        lits.push(alphabet[rng.intn(alphabet.len())]);
    }
    (ops, lits)
}

/// Appends repeats until the block is 64 bytes smaller than its output.
pub(super) fn test_pad(b: &mut TestBuilder, tb: &mut TestBlock) -> (TestEnc, Vec<u8>) {
    loop {
        let (e, out) = b.build(tb);
        if e.bytes().len() - 1 + MIN_SAVING <= out.len() {
            return (e, out);
        }
        tb.ops.extend([op(0, MAX_LEN, 0); 32]);
    }
}

/// Decodes block through the block API, into a slice with guard bytes after it, and with its
/// hash, and checks the results are want.
#[track_caller]
pub(super) fn test_decode(block: &[u8], want: &[u8]) {
    let n = crate::decoded_len(block).unwrap();
    assert_eq!(n, want.len(), "decoded length");
    let mut got = Vec::new();
    crate::decode(&mut got, block).unwrap();
    if let Some(i) = got.iter().zip(want).position(|(g, w)| g != w) {
        panic!(
            "mismatch at {i} of {}: got {}, want {}",
            want.len(),
            got[i],
            want[i]
        );
    }
    let (_, body) = simd_header(&block[1..]).unwrap();
    let mut buf = vec![0xfe; n + 64];
    decode_simd(&mut buf[..n], body).unwrap();
    assert!(buf[..n] == *want, "slice output differs");
    assert!(buf[n..].iter().all(|&v| v == 0xfe), "wrote past dst");
    // Hashing chunk by chunk gives the hash of the output.
    let h = decode_simd_hash(&mut vec![0; n], body, true).unwrap();
    assert_eq!(h, crate::xxh3::hash(want), "hash");
}
