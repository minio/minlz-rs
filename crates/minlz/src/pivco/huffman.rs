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

//! Code lengths from a histogram: a port of upstream `pivco_build_table`
//! (`src/huffman_table.c`) with `PIVCO_EFFORT_PLAIN`. Every tie rule below is load-bearing
//! for upstream's lengths.

use super::MAX_CODE_LEN;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Leaf {
    pub(super) freq: u64,
    pub(super) sym: u8,
}

/// Sets `lens` to the Huffman code lengths of `hist`, the counts of symbols 0 to
/// hist.len()-1, at most 256, limited to [`MAX_CODE_LEN`].
///
/// A lone used symbol gets length 1. It returns false, with `lens` all zero, if `hist` is
/// all zero. The lengths are upstream's wherever upstream is defined; it overruns an array
/// when the unlimited Huffman tree is deeper than 63.
pub fn build_lengths(hist: &[u64], lens: &mut [u8; 256]) -> bool {
    *lens = [0; 256];
    let mut leaves = [Leaf::default(); 256];
    let mut used = [0u8; 256]; // used symbols, ascending
    let mut n = 0usize;
    let mut bits = 0u64;
    for (s, &f) in hist.iter().enumerate() {
        // Branch-free append: the slot is overwritten unless f != 0.
        leaves[usize::from(n as u8)] = Leaf {
            freq: f,
            sym: s as u8,
        };
        used[usize::from(n as u8)] = s as u8;
        n += ((f | f.wrapping_neg()) >> 63) as usize;
        bits |= f;
    }
    match n {
        0 => return false,
        1 => {
            lens[usize::from(used[0])] = 1;
            return true;
        }
        _ => {}
    }
    let l = &mut leaves[..n];
    sort_leaves(l, bits);

    // Two-queue Huffman: leaves 0..n-1 in (freq, sym) order, internal nodes appended in
    // creation order. A leaf wins ties, reproducing a heap ordered by (freq, node index).
    let mut nfreq = [0u64; 511];
    let mut parent = [0u16; 511];
    for (f, x) in nfreq.iter_mut().zip(l.iter()) {
        *f = x.freq;
    }
    let (mut li, mut ih, mut ni) = (0, n, n);
    for _ in 0..n - 1 {
        let a = if li < n && (ih == ni || nfreq[li] <= nfreq[ih]) {
            li += 1;
            li - 1
        } else {
            ih += 1;
            ih - 1
        };
        let b = if li < n && (ih == ni || nfreq[li] <= nfreq[ih]) {
            li += 1;
            li - 1
        } else {
            ih += 1;
            ih - 1
        };
        // Wraps like C when the frequencies sum to 2^64 or more; the tree stays complete.
        nfreq[ni] = nfreq[a].wrapping_add(nfreq[b]);
        (parent[a], parent[b]) = (ni as u16, ni as u16);
        ni += 1;
    }
    let mut depth = [0u8; 511];
    let mut max_len = 0;
    for i in (0..ni - 1).rev() {
        depth[i] = depth[usize::from(parent[i])] + 1;
    }
    for (x, &d) in l.iter().zip(&depth) {
        lens[usize::from(x.sym)] = d;
        max_len = max_len.max(d);
    }
    if usize::from(max_len) > MAX_CODE_LEN {
        limit_lengths(lens, &used[..n]);
    }
    true
}

/// Sorts `l` stably by freq. `bits` is the OR of all frequencies.
/// Any stable sort gives the same order; this mirrors C's choice for speed.
pub(super) fn sort_leaves(l: &mut [Leaf], bits: u64) {
    if l.len() <= 40 {
        for i in 1..l.len() {
            let cur = l[i];
            let mut j = i;
            while j > 0 && l[j - 1].freq > cur.freq {
                l[j] = l[j - 1];
                j -= 1;
            }
            l[j] = cur;
        }
        return;
    }
    // LSD radix over the bytes present in bits. All planes are counted in one pass so
    // that runs into one bin of one plane overlap the other planes' work.
    let npass = (64 - bits.leading_zeros() as usize).div_ceil(8);
    // u8 counts as in C: only a bin holding all 256 leaves wraps (to 0), and then every
    // cursor is 0, which is still the right scatter.
    let mut cnt = [[0u8; 256]; 8];
    for x in l.iter() {
        for (p, c) in cnt[..npass].iter_mut().enumerate() {
            let b = &mut c[usize::from((x.freq >> (8 * p)) as u8)];
            *b = b.wrapping_add(1);
        }
    }
    let len = l.len();
    let mut tmp = [Leaf::default(); 256];
    let (mut src, mut dst): (&mut [Leaf], &mut [Leaf]) = (l, &mut tmp[..len]);
    for (p, cp) in cnt[..npass].iter().enumerate() {
        let mut pos = [0u16; 256];
        let (mut sum, mut dom, mut dom_count) = (0u16, 0u8, 0u8);
        for (k, &c) in cp.iter().enumerate() {
            pos[k] = sum;
            sum += u16::from(c);
            if c > dom_count {
                (dom, dom_count) = (k as u8, c);
            }
        }
        let sh = 8 * p;
        if usize::from(dom_count) >= len / 2 {
            // A bin with most leaves keeps its cursor in a register instead of chaining
            // through pos on every leaf.
            let mut d = pos[usize::from(dom)];
            for x in src.iter() {
                let k = (x.freq >> sh) as u8;
                if k == dom {
                    dst[usize::from(d)] = *x;
                    d += 1;
                } else {
                    let pk = &mut pos[usize::from(k)];
                    dst[usize::from(*pk)] = *x;
                    *pk += 1;
                }
            }
        } else {
            for x in src.iter() {
                let pk = &mut pos[usize::from((x.freq >> sh) as u8)];
                dst[usize::from(*pk)] = *x;
                *pk += 1;
            }
        }
        core::mem::swap(&mut src, &mut dst);
    }
    if npass % 2 == 1 {
        dst.copy_from_slice(src);
    }
}

/// C `limit_code_lengths` with max_len = [`MAX_CODE_LEN`]: DEFLATE-style count adjustment,
/// then relabeling by (clamped original length, symbol value).
/// `used` lists the symbols with a length, ascending.
fn limit_lengths(lens: &mut [u8; 256], used: &[u8]) {
    const M: usize = MAX_CODE_LEN;
    // Lengths above m are counted together; C counts them per length (in an array of 64,
    // which overflows for depths above 63) and then lumps them into m.
    let cl = |lens: &[u8; 256], s: u8| usize::from(lens[usize::from(s)].min(M as u8 + 1));
    let mut c4 = [[0u8; M + 2]; 4];
    let mut q = used.chunks_exact(4);
    for u in &mut q {
        c4[0][cl(lens, u[0])] += 1;
        c4[1][cl(lens, u[1])] += 1;
        c4[2][cl(lens, u[2])] += 1;
        c4[3][cl(lens, u[3])] += 1;
    }
    for &s in q.remainder() {
        c4[0][cl(lens, s)] += 1;
    }
    let mut count = [0usize; M + 2];
    for l in 1..=M + 1 {
        count[l.min(M)] += c4.iter().map(|c| usize::from(c[l])).sum::<usize>();
    }
    // Relabel order: by clamped original length, then symbol. Its bin sizes are the
    // counts before adjustment.
    let mut start = [0usize; M + 1];
    let mut sum = 0;
    for l in 1..=M {
        (start[l], sum) = (sum, sum + count[l]);
    }

    let mut kraft = 0;
    for (i, &c) in count.iter().enumerate().take(M + 1).skip(1) {
        kraft += c << (M - i);
    }
    const TARGET: usize = 1 << M;
    while kraft > TARGET {
        // Some length below m is in use: otherwise kraft = count[m] <= 256.
        let mut best = M - 1;
        while count[best] == 0 {
            best -= 1;
        }
        count[best] -= 1;
        count[best + 1] += 1;
        kraft -= 1 << (M - best - 1);
    }
    // C's search loop here always takes its first candidate (m -> m-1, +1).
    while kraft < TARGET && count[M] > 0 {
        count[M] -= 1;
        count[M - 1] += 1;
        kraft += 1;
        if kraft < TARGET {
            if count[M] < 2 {
                break;
            }
            count[M] -= 1;
            count[M - 1] += 1;
            kraft += 1;
        }
    }

    let mut order = [0u8; 256];
    for &s in used {
        let c = usize::from(lens[usize::from(s)]).min(M);
        order[start[c]] = s;
        start[c] += 1;
    }
    let mut si = 0;
    for (l, &c) in count.iter().enumerate().take(M + 1).skip(1) {
        for _ in 0..c {
            lens[usize::from(order[si])] = l as u8;
            si += 1;
        }
    }
}
