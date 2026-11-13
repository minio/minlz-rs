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

//! Test support for pivco: deterministic inputs as upstream's tests generate them (Go
//! `gen_test.go`), Go's `math/rand/v2` PCG, and the block checks of Go's `block_test.go`,
//! `table_test.go`, `corrupt_test.go` and `fuzz_test.go`.
//!
//! Besides the crate's tests, the cargo-fuzz targets, the allocation test and the
//! benchmarks include this file with `#[path]`; it reaches pivco through `super::` only
//! and uses only its public API, so their crate roots import the names it uses.

#![allow(dead_code)]

use super::kernel::{KIND_BOTH, KIND_FLAT, KIND_LEAF_LEFT, Tier};
use super::{Decoder, Encoder, Error, MAX_BLOCK_SIZE, MAX_CODE_LEN, Table, build_lengths};

/// Block sizes of the round-trip tests.
pub const BLOCK_SIZES: [usize; 22] = [
    1, 2, 7, 8, 9, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 640, 1000, 4096, 32768,
    65535,
];

/// The named distributions of [`dist`] that the tests use.
pub const TEST_DISTS: [&str; 22] = [
    "uniform",
    "english",
    "zipfian",
    "sparse_4",
    "sparse_16",
    "geometric",
    "two_sym_eq",
    "two_sym_skew",
    "odd_sigma",
    "proba80",
    "proba50",
    "proba14",
    "proba02",
    "flat_M3",
    "flat_M5",
    "flat_M6",
    "flat_M7",
    "single0",
    "single42",
    "single255",
    "two_equal",
    "sym0_255",
];

pub const GUARD_BYTE: u8 = 0xAA;

pub fn b2i(b: bool) -> usize {
    usize::from(b)
}

/// Upstream's PRNG (and Go's lensRand); `next` returns the new state.
#[derive(Clone, Copy)]
pub struct Xorshift64(pub u64);

impl Xorshift64 {
    pub fn next(&mut self) -> u64 {
        let mut v = self.0;
        v ^= v << 13;
        v ^= v >> 7;
        v ^= v << 17;
        self.0 = v;
        v
    }

    pub fn intn(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub fn perm(&mut self) -> [u8; 256] {
        let mut p: [u8; 256] = core::array::from_fn(|i| i as u8);
        for i in (1..256).rev() {
            let j = self.intn(i + 1);
            p.swap(i, j);
        }
        p
    }
}

/// Go's `math/rand/v2` PCG source with the `Rand` methods the tests use, so that inputs
/// match Go's for the same seeds.
pub struct Pcg {
    hi: u64,
    lo: u64,
}

impl Pcg {
    pub fn new(seed1: u64, seed2: u64) -> Self {
        Pcg {
            hi: seed1,
            lo: seed2,
        }
    }

    fn step(&mut self) -> (u64, u64) {
        const MUL_HI: u64 = 2549297995355413924;
        const MUL_LO: u64 = 4865540595714422341;
        const INC_HI: u64 = 6364136223846793005;
        const INC_LO: u64 = 1442695040888963407;
        let m = u128::from(self.lo) * u128::from(MUL_LO);
        let hi = ((m >> 64) as u64)
            .wrapping_add(self.hi.wrapping_mul(MUL_LO))
            .wrapping_add(self.lo.wrapping_mul(MUL_HI));
        let (lo, c) = (m as u64).overflowing_add(INC_LO);
        let hi = hi.wrapping_add(INC_HI).wrapping_add(u64::from(c));
        (self.hi, self.lo) = (hi, lo);
        (hi, lo)
    }

    pub fn uint64(&mut self) -> u64 {
        const CHEAP_MUL: u64 = 0xda942042e4dd58b5;
        let (mut hi, lo) = self.step();
        hi ^= hi >> 32;
        hi = hi.wrapping_mul(CHEAP_MUL);
        hi ^= hi >> 48;
        hi.wrapping_mul(lo | 1)
    }

    pub fn uint32(&mut self) -> u32 {
        (self.uint64() >> 32) as u32
    }

    fn uint64n(&mut self, n: u64) -> u64 {
        if n & (n - 1) == 0 {
            return self.uint64() & (n - 1);
        }
        let mul = |x: u64| {
            let m = u128::from(x) * u128::from(n);
            ((m >> 64) as u64, m as u64)
        };
        let (mut hi, mut lo) = mul(self.uint64());
        if lo < n {
            let thresh = n.wrapping_neg() % n;
            while lo < thresh {
                (hi, lo) = mul(self.uint64());
            }
        }
        hi
    }

    pub fn int_n(&mut self, n: usize) -> usize {
        self.uint64n(n as u64) as usize
    }

    pub fn float64(&mut self) -> f64 {
        (self.uint64() << 11 >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn perm(&mut self, n: usize) -> Vec<usize> {
        let mut p: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.uint64n(i as u64 + 1) as usize;
            p.swap(i, j);
        }
        p
    }
}

/// Go's kernel tests' `newRNG`.
pub fn new_rng() -> Pcg {
    Pcg::new(1, 2)
}

pub fn rand_bytes(rng: &mut Pcg, b: &mut [u8]) {
    for x in b {
        *x = rng.uint32() as u8;
    }
}

/// Returns a k-bit bitmap with zero padding bits where each bit is set with probability p,
/// and its popcount.
pub fn rand_bits(rng: &mut Pcg, k: usize, p: f64) -> (Vec<u8>, usize) {
    let mut bm = vec![0u8; k.div_ceil(8)];
    let mut pc = 0;
    for j in 0..k {
        if rng.float64() < p {
            bm[j / 8] |= 1 << (j % 8);
            pc += 1;
        }
    }
    (bm, pc)
}

pub fn fnv1a64(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &c in b {
        h = (h ^ u64::from(c)).wrapping_mul(0x100000001b3);
    }
    h
}

/// Returns a named histogram of upstream's test suite, or `None` if the name is unknown.
pub fn dist(name: &str) -> Option<[u64; 256]> {
    let mut f = [0u64; 256];
    match name {
        "uniform" => f.fill(100),
        "english" => {
            for (c, v) in [
                (b' ', 1830),
                (b'e', 1270),
                (b't', 910),
                (b'a', 820),
                (b'o', 750),
                (b'i', 700),
                (b'n', 670),
                (b's', 630),
                (b'h', 610),
                (b'r', 600),
                (b'd', 430),
                (b'l', 400),
                (b'c', 280),
                (b'u', 280),
                (b'm', 240),
                (b'w', 240),
                (b'f', 220),
                (b'g', 200),
                (b'y', 200),
                (b'p', 190),
                (b'b', 150),
                (b'v', 100),
                (b'k', 80),
                (b'j', 15),
                (b'x', 15),
                (b'q', 10),
                (b'z', 7),
                (b'.', 65),
                (b',', 61),
                (b'\n', 50),
            ] {
                f[usize::from(c)] = v;
            }
        }
        "zipfian" => {
            for (i, x) in f.iter_mut().enumerate() {
                *x = ((10000.0 / (i + 1) as f64) as u64).max(1);
            }
        }
        "sparse_4" => f[..4].fill(100),
        "sparse_16" => f[..16].fill(100),
        "geometric" => {
            for (i, x) in f.iter_mut().enumerate() {
                *x = 1 << 30usize.saturating_sub(i);
            }
        }
        "two_sym_eq" => (f[0], f[1]) = (500, 500),
        "two_sym_skew" => (f[0], f[1]) = (900, 100),
        "odd_sigma" => {
            for i in 0..71 {
                f[13 + 2 * i] = (100000 / (i + 3)) as u64;
            }
        }
        "proba80" => fse_proba(&mut f, 0.80),
        "proba50" => fse_proba(&mut f, 0.50),
        "proba14" => fse_proba(&mut f, 0.14),
        "proba02" => fse_proba(&mut f, 0.02),
        "flat_M3" | "flat_M5" | "flat_M6" | "flat_M7" => {
            f[..1 << (name.as_bytes()[6] - b'0')].fill(100)
        }
        "single0" => f[0] = 1,
        "single42" => f[42] = 1,
        "single255" => f[255] = 1,
        "two_equal" => (f[0], f[255]) = (1, 1),
        "sym0_255" => {
            for (i, x) in f.iter_mut().enumerate() {
                *x = i as u64 + 1;
            }
        }
        _ => return None,
    }
    Some(f)
}

/// Returns [`dist`] of a name the tests use.
pub fn dist_of(name: &str) -> [u64; 256] {
    dist(name).unwrap_or_else(|| panic!("unknown distribution {name}"))
}

/// bench_distributions.c make_fse_proba.
fn fse_proba(f: &mut [u64; 256], p: f64) {
    let mut table = [0usize; 2048];
    let (mut remaining, mut pos, mut s) = (2048usize, 0usize, 0usize);
    let p = if p < 0.01 { 0.005 } else { p.min(1.0) };
    while remaining > 0 {
        let n = ((remaining as f64 * p) as usize).max(1);
        let end = (pos + n).min(2048);
        table[pos..end].fill(s);
        pos = end.max(pos);
        s += 1;
        if s == 255 {
            s = 0;
        }
        remaining -= n;
    }
    for v in table {
        f[v] += 1;
    }
}

/// A random histogram, frequencies capped at 2^35.
pub fn rhist(seed: u64) -> [u64; 256] {
    const CAPF: u64 = 1 << 35;
    let mut f = [0u64; 256];
    let mut x = Xorshift64(seed);
    let mut perm: [u8; 256] = core::array::from_fn(|i| i as u8);
    for i in (1..256).rev() {
        let j = (x.next() % (i as u64 + 1)) as usize;
        perm.swap(i, j);
    }
    let nsym = 1 + (x.next() % 256) as usize;
    let mode = x.next() % 4;
    for (i, &p) in perm[..nsym].iter().enumerate() {
        let v = match mode {
            0 => 1 + x.next() % 1000,
            1 => {
                let sh = x.next() % 36;
                1 + x.next() % (1 << sh)
            }
            2 => 1 + (CAPF >> (i % 36)),
            _ => 1 << (x.next() % 3),
        };
        f[usize::from(p)] = v.min(CAPF);
    }
    f
}

/// Draws n IID symbols: r = next % total, first symbol whose cumulative frequency exceeds
/// r. Totals must not wrap.
pub fn sample(hist: &[u64; 256], n: usize, seed: u64) -> Vec<u8> {
    let mut cum = [0u64; 256];
    let mut total = 0u64;
    for (c, &v) in cum.iter_mut().zip(hist) {
        total += v;
        *c = total;
    }
    let mut x = Xorshift64(seed);
    (0..n)
        .map(|_| {
            let r = x.next() % total;
            let (mut lo, mut hi) = (0, 255);
            while lo < hi {
                let m = (lo + hi) / 2;
                if r < cum[m] {
                    hi = m;
                } else {
                    lo = m + 1;
                }
            }
            lo as u8
        })
        .collect()
}

/// Samples with weight 1<<(11-len) per used symbol.
pub fn sample_lens(lens: &[u8; 256], n: usize, seed: u64) -> Vec<u8> {
    let mut w = [0u64; 256];
    for (x, &l) in w.iter_mut().zip(lens) {
        if l > 0 {
            *x = 1 << (11 - l);
        }
    }
    sample(&w, n, seed)
}

pub fn kraft_sum(lens: &[u8; 256]) -> usize {
    lens.iter()
        .filter(|&&l| l != 0)
        .map(|&l| 1 << (MAX_CODE_LEN - usize::from(l)))
        .sum()
}

/// Returns the length of h up to its last used symbol.
pub fn used_syms(h: &[u64; 256]) -> usize {
    h.iter().rposition(|&f| f != 0).map_or(0, |i| i + 1)
}

pub fn check_lengths(h: &[u64; 256], lens: &[u8; 256]) {
    let mut used = 0;
    for (s, &l) in lens.iter().enumerate() {
        assert!(
            (h[s] != 0) == (l != 0) && usize::from(l) <= MAX_CODE_LEN,
            "sym {s}: freq {} len {l}",
            h[s]
        );
        used += b2i(l != 0);
    }
    if used > 1 {
        assert_eq!(kraft_sum(lens), 1 << MAX_CODE_LEN, "kraft, lens {lens:?}");
    }
}

pub fn lens_of(h: &[u64; 256]) -> [u8; 256] {
    let mut l = [0u8; 256];
    assert!(build_lengths(h, &mut l), "empty histogram");
    l
}

pub fn must_table(lens: &[u8; 256]) -> Table {
    let mut tb = Table::default();
    if let Err(e) = tb.set_lengths(lens) {
        panic!("set_lengths: {e}");
    }
    check_table(&tb);
    tb
}

/// Verifies the schedule semantically: walking it yields a tree whose leaf depths are the
/// code lengths, with ranks in order, so the code is prefix-free and decodable, and all
/// derived arrays agree.
pub fn check_table(tb: &Table) {
    assert!(tb.valid(), "invalid table");
    let recs = tb.recs();
    let mut leaf_depth = [0usize; 256];
    let mut nleaves = 0;
    let mut add_leaf = |rank: usize, depth: usize, nleaves: &mut usize| {
        assert_eq!(rank, *nleaves, "rank out of order");
        leaf_depth[rank] = depth;
        *nleaves += 1;
    };
    // visit walks the subtree at record i with root depth d and returns the next record.
    fn visit(
        recs: &[super::kernel::Rec],
        i: usize,
        d: usize,
        nl: &mut usize,
        add: &mut dyn FnMut(usize, usize, &mut usize),
    ) -> usize {
        assert!(
            i < recs.len() && d < MAX_CODE_LEN,
            "record {i} at depth {d} out of range"
        );
        let r = recs[i];
        let p = usize::from(r.param);
        match r.kind() {
            KIND_FLAT => {
                let fd = usize::from(r.flat_depth());
                assert!(
                    (2..=8).contains(&fd) && r.right == 0 && p == *nl,
                    "bad flat {r:?}"
                );
                for k in 0..1 << fd {
                    add(p + k, d + fd, nl);
                }
                i + 1
            }
            KIND_BOTH => {
                assert!(r.right == 0 && p == *nl, "bad both {r:?}");
                add(p, d + 1, nl);
                add(p + 1, d + 1, nl);
                i + 1
            }
            KIND_LEAF_LEFT => {
                assert!(r.right == 1 && p == *nl, "bad leaf-left {r:?}");
                add(p, d + 1, nl);
                visit(recs, i + 1, d + 1, nl, add)
            }
            _ => {
                let next = visit(recs, i + 1, d + 1, nl, add);
                assert!(
                    p + 1 == *nl && usize::from(r.right) == next - i,
                    "bad full {r:?} at {i}"
                );
                visit(recs, next, d + 1, nl, add)
            }
        }
    }
    let end = visit(recs, 0, 0, &mut nleaves, &mut add_leaf);
    assert!(
        end == recs.len() && nleaves == tb.nranks(),
        "schedule covers {nleaves} leaves, nranks {}",
        tb.nranks()
    );
    let (r2s, s2r, lens) = (tb.r2s(), tb.s2r(), tb.lens());
    for (i, &x) in r2s[tb.nranks()..].iter().enumerate() {
        assert_eq!(x, 0, "r2s[{}]: want zero padding", tb.nranks() + i);
    }
    if recs.len() == 1 && recs[0].kind() == KIND_BOTH && r2s[0] == r2s[1] {
        let s = usize::from(r2s[0]);
        for (i, &l) in lens.iter().enumerate() {
            let want = if i == s { 0 } else { 0xFF };
            assert!(
                (l != 0) == (i == s) && s2r[i] == want,
                "single symbol {s}: sym {i} len {l} s2r {}",
                s2r[i]
            );
        }
        return;
    }
    let mut seen = 0isize;
    for (r, &s) in r2s[..tb.nranks()].iter().enumerate() {
        let s = usize::from(s);
        assert!(
            usize::from(lens[s]) == leaf_depth[r] && usize::from(s2r[s]) == r,
            "rank {r} sym {s}: len {} depth {} s2r {}",
            lens[s],
            leaf_depth[r],
            s2r[s]
        );
        seen += 1;
    }
    for (s, &l) in lens.iter().enumerate() {
        assert!(
            l != 0 || s2r[s] == 0xFF,
            "unused sym {s} has rank {}",
            s2r[s]
        );
        seen -= (l != 0) as isize;
    }
    assert_eq!(seen, 0, "rank count != used symbols");
}

/// An Encoder and Decoder pinned to one kernel tier.
pub struct Coder {
    pub tier: Tier,
    pub enc: Encoder,
    pub dec: Decoder,
}

pub fn coders() -> Vec<Coder> {
    Tier::tiers()
        .into_iter()
        .map(|tier| Coder {
            tier,
            enc: Encoder::with_tier(tier),
            dec: Decoder::with_tier(tier),
        })
        .collect()
}

/// Returns n bytes after prefix guard bytes, followed by 64 more, as the whole buffer.
pub fn guarded(n: usize, prefix: usize) -> Vec<u8> {
    vec![GUARD_BYTE; prefix + n + 64]
}

pub fn intact(b: &[u8]) -> bool {
    b.iter().all(|&c| c == GUARD_BYTE)
}

pub fn block_size(c: &mut Coder, src: &[u8], tb: &Table) -> usize {
    let mut h = [0u64; 256];
    c.enc.histogram(&mut h, src);
    let size = tb.block_size(&h);
    let short = tb.block_size(&h[..used_syms(&h)]);
    match size {
        Some(s) if s > 0 && short == size => s,
        _ => panic!(
            "n={}: block_size {size:?}, {short:?} up to the last used symbol",
            src.len()
        ),
    }
}

/// Encodes and decodes src between guard bytes, checks that nothing outside the block and
/// the output is written, and returns the block.
pub fn round_trip(c: &mut Coder, src: &[u8], tb: &Table) -> Vec<u8> {
    let (n, tier) = (src.len(), c.tier);
    let size = block_size(c, src, tb);
    let mut all = guarded(size, 3);
    if let Err(e) = c.enc.encode(&mut all[3..3 + size], src, tb) {
        panic!("{tier} n={n}: {e}");
    }
    assert!(
        intact(&all[..3]) && intact(&all[3 + size..]),
        "{tier} n={n}: encoder wrote outside its output"
    );
    let blk = all[3..3 + size].to_vec();
    let mut again = vec![0; size];
    assert!(
        c.enc.encode(&mut again, src, tb).is_ok() && again == blk,
        "{tier} n={n}: encoding differs between calls"
    );
    let mut all = guarded(n, 5);
    if let Err(e) = c.dec.decode(&mut all[5..5 + n], &blk, tb) {
        panic!("{tier} n={n}: decode: {e}");
    }
    assert!(
        intact(&all[..5]) && intact(&all[5 + n..]),
        "{tier} n={n}: decoder wrote outside its output"
    );
    assert!(
        all[5..5 + n] == *src,
        "{tier} n={n}: decoded output differs"
    );
    check_fields(tb, &blk, n);
    blk
}

/// A field of a block found by [`walk_block`].
#[derive(Clone, Copy, Debug)]
pub struct BlockField {
    /// b'k' K_right, b'b' bitmap, b'f' flat codes.
    pub kind: u8,
    pub pos: usize,
    pub size: usize,
    pub bits: usize,
    /// Record and its symbol count.
    pub rec: usize,
    pub k: usize,
}

/// Parses a block of n symbols independently of the decoder and lists its fields in stream
/// order.
pub fn walk_block(tb: &Table, b: &[u8], n: usize) -> Vec<BlockField> {
    struct W<'a> {
        recs: &'a [super::kernel::Rec],
        b: &'a [u8],
        pos: usize,
        fs: Vec<BlockField>,
    }
    impl W<'_> {
        fn add(&mut self, kind: u8, size: usize, bits: usize, rec: usize, k: usize) -> usize {
            assert!(
                self.pos + size <= self.b.len(),
                "field {} of record {rec} ends past the block",
                kind as char
            );
            self.fs.push(BlockField {
                kind,
                pos: self.pos,
                size,
                bits,
                rec,
                k,
            });
            self.pos += size;
            self.pos - size
        }
        fn walk(&mut self, rec: usize, k: usize) {
            if k == 0 {
                return;
            }
            let r = self.recs[rec];
            match r.kind() {
                KIND_FLAT => {
                    let d = usize::from(r.flat_depth());
                    self.add(b'f', (k * d).div_ceil(8), k * d, rec, k);
                    return;
                }
                KIND_BOTH => {}
                kind => {
                    let at = self.add(b'k', 2, 16, rec, k);
                    let kr = usize::from(u16::from_le_bytes([self.b[at], self.b[at + 1]]));
                    assert!(kr <= k, "record {rec}: K_right {kr} > K {k}");
                    if kind == KIND_LEAF_LEFT {
                        self.walk(rec + 1, kr);
                    } else if kr > k - kr {
                        self.walk(rec + usize::from(r.right), kr);
                        self.walk(rec + 1, k - kr);
                    } else {
                        self.walk(rec + 1, k - kr);
                        self.walk(rec + usize::from(r.right), kr);
                    }
                }
            }
            self.add(b'b', k.div_ceil(8), k, rec, k);
        }
    }
    let mut w = W {
        recs: tb.recs(),
        b,
        pos: 0,
        fs: Vec::new(),
    };
    w.walk(0, n);
    assert_eq!(
        w.pos,
        b.len(),
        "block has {} bytes, fields end at {}",
        b.len(),
        w.pos
    );
    w.fs
}

/// Checks the canonical form of a block: zero padding bits.
pub fn check_fields(tb: &Table, b: &[u8], n: usize) {
    for f in walk_block(tb, b, n) {
        let last = b[f.pos + f.size - 1];
        assert!(
            f.bits % 8 == 0 || last >> (f.bits % 8) == 0,
            "field {} at {}: bad padding {last:#x}",
            f.kind as char,
            f.pos
        );
    }
}

/// Decodes n symbols of blk. An error must be [`Error::Corrupt`] and leave the guard bytes
/// around dst intact.
pub fn decode_err(c: &mut Coder, blk: &[u8], n: usize, tb: &Table) -> Result<Vec<u8>, Error> {
    let mut all = guarded(n, 4);
    let res = c.dec.decode(&mut all[4..4 + n], blk, tb);
    assert!(
        intact(&all[..4]) && intact(&all[4 + n..]),
        "{}: decoder wrote outside dst ({res:?})",
        c.tier
    );
    match res {
        Err(e @ Error::Corrupt(_)) => Err(e),
        Err(e) => panic!("{}: unexpected error {e}", c.tier),
        Ok(()) => Ok(all[4..4 + n].to_vec()),
    }
}

/// Fails unless blk, which decodes to src, is what the encoder writes for src.
pub fn canon(c: &mut Coder, tb: &Table, blk: &[u8], src: &[u8]) {
    let mut h = [0u64; 256];
    c.enc.histogram(&mut h, src);
    let size = tb.block_size(&h).expect("decoded a symbol without a code");
    let mut again = vec![0; size];
    assert!(
        size == blk.len() && c.enc.encode(&mut again, src, tb).is_ok() && again == blk,
        "accepted a block the encoder does not write for its output\nblock {blk:x?}\n  enc {again:x?}"
    );
}

/// Returns lengths with a depth-d flat subtree of 2^d symbols starting at symbol 16: alone
/// at the root ("root"), as the right child of the root ("sub"), or below three single
/// leaves ("deep").
pub fn flat_lens(d: usize, place: &str) -> [u8; 256] {
    let mut lens = [0u8; 256];
    let top = match place {
        "sub" => {
            lens[0] = 1;
            1
        }
        "deep" => {
            (lens[0], lens[1], lens[2]) = (1, 2, 3);
            3
        }
        _ => 0,
    };
    let first = if d == 8 { 0 } else { 16 };
    lens[first..first + (1 << d)].fill((top + d) as u8);
    lens
}

/// A valid block and the table it was coded with.
pub struct CorruptCase {
    pub name: String,
    pub tb: Table,
    pub src: Vec<u8>,
    pub blk: Vec<u8>,
}

/// Returns about 50 small valid blocks over tables of every record kind.
pub fn corrupt_cases() -> Vec<CorruptCase> {
    let mut c = coders().swap_remove(0);
    let mut cs = Vec::new();
    let mut add = |name: String, lens: [u8; 256], src: Vec<u8>| {
        let tb = must_table(&lens);
        let mut blk = vec![0; block_size(&mut c, &src, &tb)];
        c.enc.encode(&mut blk, &src, &tb).unwrap();
        cs.push(CorruptCase {
            name: format!("{name}/n{}", src.len()),
            tb,
            src,
            blk,
        });
    };
    for name in [
        "english",
        "zipfian",
        "sparse_16",
        "geometric",
        "two_sym_skew",
        "proba80",
        "single42",
        "odd_sigma",
    ] {
        let h = dist_of(name);
        for n in [1, 9, 100, 700, 3000] {
            add(name.into(), lens_of(&h), sample(&h, n, n as u64));
        }
    }
    let skews: [&[(u8, u64)]; 3] = [
        &[(b'a', 63), (b'b', 37)],
        &[(b'a', 50), (b'b', 32), (b'c', 18)],
        &[(b'a', 60), (b'b', 25), (b'c', 15)],
    ];
    for (i, sk) in skews.iter().enumerate() {
        let mut h = [0u64; 256];
        for &(s, f) in *sk {
            h[usize::from(s)] = f;
        }
        add(format!("skew{i}"), lens_of(&h), sample(&h, 3000, 9));
    }
    for d in [2, 5, 7] {
        let lens = flat_lens(d, "sub");
        let h: [u64; 256] = core::array::from_fn(|s| u64::from(lens[s] != 0));
        add(format!("flat{d}"), lens, sample(&h, 650, d as u64));
    }
    cs
}

/// Builds a table from 128 bytes of length nibbles if they are valid, else from the
/// histogram of data, so that most inputs reach the codec.
pub fn fuzz_table(lens: &[u8], data: &[u8]) -> Table {
    let mut l = [0u8; 256];
    for (i, &b) in lens.iter().enumerate() {
        (l[2 * i], l[2 * i + 1]) = (b & 15, b >> 4);
    }
    let mut tb = Table::default();
    if tb.set_lengths(&l).is_ok() {
        return tb;
    }
    let mut h = [0u64; 256];
    for &c in data {
        h[usize::from(c)] += 1;
    }
    h[0] += 1;
    build_lengths(&h, &mut l);
    tb.set_lengths(&l).expect("built lengths rejected");
    tb
}

/// Decodes n symbols of blk on every tier, which must agree, and checks that an accepted
/// block is what the encoder writes for its output.
pub fn fuzz_decode(cs: &mut [Coder], tb: &Table, n: usize, blk: &[u8]) {
    if n == 0 {
        return;
    }
    let (first, rest) = cs.split_first_mut().unwrap();
    let r = decode_err(first, blk, n, tb);
    for c in rest {
        let got = decode_err(c, blk, n, tb);
        assert!(
            got.as_ref().ok() == r.as_ref().ok() && got.is_ok() == r.is_ok(),
            "{}: ({got:?}) differs from {} ({r:?})",
            c.tier,
            first.tier
        );
    }
    if let Ok(src) = r {
        canon(first, tb, blk, &src);
    }
}

/// Go FuzzDecodeBlock: [128 bytes of length nibbles][symbol count, u16][block]. Invalid
/// lengths fall back to a table built from the block bytes.
pub fn fuzz_decode_block(cs: &mut [Coder], data: &[u8]) {
    if data.len() < 130 {
        return;
    }
    let blk = &data[130..];
    let n = usize::from(u16::from_le_bytes([data[128], data[129]]));
    fuzz_decode(cs, &fuzz_table(&data[..128], blk), n, blk);
}

/// The fixed tables of Go FuzzDecodeBlockFixed: those of [`corrupt_cases`], then those of
/// [`TEST_DISTS`].
pub fn fixed_tables() -> Vec<Table> {
    let mut tabs: Vec<Table> = corrupt_cases().into_iter().map(|c| c.tb).collect();
    tabs.extend(
        TEST_DISTS
            .iter()
            .map(|name| must_table(&lens_of(&dist_of(name)))),
    );
    tabs
}

/// Go FuzzDecodeBlockFixed: [table][symbol count, u16][block] over [`fixed_tables`], so
/// that mutating a block keeps its table.
pub fn fuzz_decode_block_fixed(cs: &mut [Coder], tabs: &[Table], data: &[u8]) {
    if data.len() < 3 {
        return;
    }
    let n = usize::from(u16::from_le_bytes([data[1], data[2]]));
    fuzz_decode(cs, &tabs[usize::from(data[0]) % tabs.len()], n, &data[3..]);
}

/// Go FuzzRoundTripBlock: a flags byte (ghost symbols in the table), then the data.
pub fn fuzz_round_trip_block(cs: &mut [Coder], data: &[u8]) {
    if data.len() < 2 {
        return;
    }
    let (flags, src) = (data[0], &data[1..data.len().min(MAX_BLOCK_SIZE + 1)]);
    let mut h = [0u64; 256];
    for &c in src {
        h[usize::from(c)] += 1;
    }
    if flags & 8 != 0 {
        for (i, x) in h.iter_mut().enumerate() {
            *x += i as u64 % 3;
        }
    }
    let mut l = [0u8; 256];
    build_lengths(&h, &mut l);
    let tb = must_table(&l);
    round_trip_all(cs, src, &tb);
}

/// Round-trips src on every coder; all tiers must write the same block.
pub fn round_trip_all(cs: &mut [Coder], src: &[u8], tb: &Table) {
    let (first, rest) = cs.split_first_mut().unwrap();
    let want = round_trip(first, src, tb);
    for c in rest {
        assert!(
            round_trip(c, src, tb) == want,
            "{} differs from {}",
            c.tier,
            first.tier
        );
    }
}

/// Go FuzzTable: 128 bytes of length nibbles (symbol 2i low, 2i+1 high), then a payload
/// mapped onto the used symbols, which must round-trip identically on every tier.
pub fn fuzz_table_lens(cs: &mut [Coder], data: &[u8]) {
    if data.len() < 128 {
        return;
    }
    let mut lens = [0u8; 256];
    for (i, &b) in data[..128].iter().enumerate() {
        (lens[2 * i], lens[2 * i + 1]) = (b & 15, b >> 4);
    }
    let mut tb = Table::default();
    if let Err(e) = tb.set_lengths(&lens) {
        assert!(
            matches!(e, Error::Corrupt(_)) && !tb.valid(),
            "err {e}, valid {}",
            tb.valid()
        );
        return;
    }
    check_table(&tb);
    let used: Vec<u8> = (0..=255u8).filter(|&s| lens[usize::from(s)] != 0).collect();
    let src: Vec<u8> = data[128..]
        .iter()
        .take(MAX_BLOCK_SIZE)
        .map(|&c| used[usize::from(c) % used.len()])
        .collect();
    if src.is_empty() {
        return;
    }
    round_trip_all(cs, &src, &tb);
}

/// Go FuzzLengths: data[0] picks a frequency scale, then (symbol, u16 count) triples.
pub fn fuzz_lengths(data: &[u8]) {
    if data.is_empty() {
        return;
    }
    let shift = [0, 20, 40, 48][usize::from(data[0] & 3)];
    let mut h = [0u64; 256];
    for p in data[1..].as_chunks::<3>().0 {
        h[usize::from(p[0])] = (u64::from(p[1]) | u64::from(p[2]) << 8) << shift;
    }
    let (mut lens, mut again) = ([0u8; 256], [0u8; 256]);
    if !build_lengths(&h, &mut lens) {
        assert!(h == [0; 256], "non-empty histogram rejected");
        return;
    }
    check_lengths(&h, &lens);
    build_lengths(&h, &mut again);
    assert!(again == lens, "not deterministic");
    must_table(&lens);
}
