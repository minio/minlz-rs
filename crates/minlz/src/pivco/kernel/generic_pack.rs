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

//! Ports of upstream `prim_enc_pack_dN` (`src/pivco_huffman_primitives_scalar.h`) and
//! `vert_pack_scalar`, `vert512_pack_scalar` (`src/pivco_huffman_vertical.h`).

use super::vert_split;

pub(super) fn pack_generic(dst: &mut [u8], ranks: &[u8], d: u8, base: u8) {
    let (n, du) = (ranks.len(), usize::from(d));
    let dst = &mut dst[..(n * du).div_ceil(8)];
    let (n512, n128) = vert_split(n, d);
    vert_pack::<64>(dst, &ranks[..n512], d, base);
    let (dst, ranks) = (&mut dst[n512 * du / 8..], &ranks[n512..]);
    vert_pack::<16>(dst, &ranks[..n128], d, base);
    nat_pack(&mut dst[n128 * du / 8..], &ranks[n128..], d, base);
}

/// Packs `ranks.len()`, a multiple of 8*LN, codes into LN-lane vertical blocks: code v of
/// a block is field v/LN of lane v%LN, whose d bytes sit at a stride of LN.
fn vert_pack<const LN: usize>(dst: &mut [u8], ranks: &[u8], d: u8, base: u8) {
    if ranks.is_empty() {
        return;
    }
    let mask = (1u64 << d) - 1;
    let du = usize::from(d);
    for (blk, src) in dst
        .chunks_exact_mut(LN * du)
        .zip(ranks.chunks_exact(8 * LN))
    {
        for l in 0..LN {
            let mut w = 0u64;
            for s in 0..8 {
                w |= (u64::from(src[s * LN + l].wrapping_sub(base)) & mask) << (s * du);
            }
            for j in 0..du {
                blk[j * LN + l] = (w >> (8 * j)) as u8;
            }
        }
    }
}

/// Packs codes as one LSB-first bit string; 8 codes fill exactly d bytes.
pub(super) fn nat_pack(dst: &mut [u8], ranks: &[u8], d: u8, base: u8) {
    if d == 8 {
        for (o, &r) in dst.iter_mut().zip(ranks) {
            *o = r.wrapping_sub(base);
        }
        return;
    }
    let mask = (1u64 << d) - 1;
    let du = usize::from(d);
    for (i, g) in ranks.chunks(8).enumerate() {
        let mut w = 0u64;
        for (k, &r) in g.iter().enumerate() {
            w |= (u64::from(r.wrapping_sub(base)) & mask) << (k * du);
        }
        let o = &mut dst[i * du..];
        if o.len() >= 8 {
            // The upper 8-d bytes are rewritten by the next group.
            o[..8].copy_from_slice(&w.to_le_bytes());
            continue;
        }
        for (j, x) in o[..(g.len() * du).div_ceil(8)].iter_mut().enumerate() {
            *x = (w >> (8 * j)) as u8;
        }
    }
}
