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

//! Ports of upstream `src/pivco_huffman_primitives_scalar.h` (gather, partitions) and
//! `src/pivco_huffman_hist_scalar.h`.

pub(super) fn gather_generic(ranks: &mut [u8], src: &[u8], s2r: &[u8; 256]) -> u8 {
    let ranks = &mut ranks[..src.len()];
    let mut mx = 0;
    for (r, &s) in ranks.iter_mut().zip(src) {
        let v = s2r[usize::from(s)];
        *r = v;
        mx = mx.max(v);
    }
    mx
}

// The partitions store every rank to both sides and advance one cursor, so they are
// branch-free. The left store lands at nl <= j, i.e. on an already read rank, which makes
// the in-place compaction safe and keeps it inside ranks[:n]; the dead right store lands
// in right[nr:n].

pub(super) fn part_full_generic(
    ranks: &mut [u8],
    n: usize,
    right: &mut [u8],
    bm: &mut [u8],
    thr: u8,
) -> usize {
    let (ranks, right, bm) = (&mut ranks[..n], &mut right[..n], &mut bm[..n.div_ceil(8)]);
    let (mut nl, mut nr) = (0, 0);
    for (i, b) in bm.iter_mut().enumerate() {
        let mut m = 0u8;
        for k in 0..(n - i * 8).min(8) {
            // SAFETY: nl + nr == j < n, the length of ranks and right.
            unsafe {
                let v = *ranks.get_unchecked(i * 8 + k);
                let g = usize::from(v > thr);
                m |= (g << k) as u8;
                *right.get_unchecked_mut(nr) = v;
                *ranks.get_unchecked_mut(nl) = v;
                nr += g;
                nl += g ^ 1;
            }
        }
        *b = m;
    }
    nr
}

pub(super) fn part_right_generic(
    ranks: &[u8],
    n: usize,
    right: &mut [u8],
    bm: &mut [u8],
    thr: u8,
) -> usize {
    let (ranks, right, bm) = (&ranks[..n], &mut right[..n], &mut bm[..n.div_ceil(8)]);
    let mut nr = 0;
    for (b, g8) in bm.iter_mut().zip(ranks.chunks(8)) {
        let mut m = 0u8;
        for (k, &v) in g8.iter().enumerate() {
            let g = usize::from(v > thr);
            m |= (g << k) as u8;
            // SAFETY: nr <= j < n, the length of right.
            unsafe { *right.get_unchecked_mut(nr) = v };
            nr += g;
        }
        *b = m;
    }
    nr
}

pub(super) fn part_none_generic(ranks: &[u8], n: usize, bm: &mut [u8], thr: u8) -> usize {
    let (ranks, bm) = (&ranks[..n], &mut bm[..n.div_ceil(8)]);
    let mut nr = 0;
    for (b, g8) in bm.iter_mut().zip(ranks.chunks(8)) {
        let mut m = 0u8;
        for (k, &v) in g8.iter().enumerate() {
            let g = usize::from(v > thr);
            m |= (g << k) as u8;
            nr += g;
        }
        *b = m;
    }
    nr
}

#[inline(always)]
fn bump(c: &mut [u32; 256], b: u8) {
    let x = &mut c[usize::from(b)];
    *x = x.wrapping_add(1);
}

/// Spreads consecutive bytes over 4 sub-histograms so that runs of one symbol do not
/// serialize on a single counter. Wrapping u32 counters are fine: h wraps the same way.
/// Byte loads, as in C: taking the bytes out of 64-bit loads costs a shift and a
/// zero-extension each.
pub(super) fn histogram_generic(h: &mut [u32; 256], src: &[u8]) {
    let mut c = [[0u32; 256]; 4];
    let (chunks, rest) = src.as_chunks::<16>();
    for s in chunks {
        bump(&mut c[0], s[0]);
        bump(&mut c[1], s[1]);
        bump(&mut c[2], s[2]);
        bump(&mut c[3], s[3]);
        bump(&mut c[0], s[4]);
        bump(&mut c[1], s[5]);
        bump(&mut c[2], s[6]);
        bump(&mut c[3], s[7]);
        bump(&mut c[0], s[8]);
        bump(&mut c[1], s[9]);
        bump(&mut c[2], s[10]);
        bump(&mut c[3], s[11]);
        bump(&mut c[0], s[12]);
        bump(&mut c[1], s[13]);
        bump(&mut c[2], s[14]);
        bump(&mut c[3], s[15]);
    }
    for &b in rest {
        bump(&mut c[0], b);
    }
    for (i, x) in h.iter_mut().enumerate() {
        *x = x
            .wrapping_add(c[0][i])
            .wrapping_add(c[1][i])
            .wrapping_add(c[2][i])
            .wrapping_add(c[3][i]);
    }
}
