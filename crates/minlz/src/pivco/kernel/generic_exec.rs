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

//! Ports of the upstream merges (`merge_*_scalar` in
//! `src/pivco_huffman_primitives_scalar.h`) and flat unpacks
//! (`src/pivco_huffman_vertical.h`).
//!
//! The merges take raw pointers: in the in-place layout a merge's output overlaps one of
//! its inputs in the arena, which slices cannot express.

use super::{
    ARENA_SLACK, FLAG_DST, OP_CC, OP_COPY, OP_CV, OP_NAT, OP_V128, OP_V512, OP_VV, Op, Program,
    R2S_LEN, load64,
};

/// Checks every op's geometry against the slices before running it, so a malformed op
/// fails closed instead of panicking.
pub(super) fn exec_generic(p: &mut Program<'_>) -> Result<(), usize> {
    let ops = p.ops;
    for (i, op) in ops.iter().enumerate() {
        if !exec_op(p, op) {
            return Err(i);
        }
    }
    Ok(())
}

/// Reports whether `off..off+n` is inside a buffer of `len` bytes.
#[inline(always)]
fn fits(len: usize, off: u32, n: u64) -> bool {
    u64::from(off) + n <= len as u64
}

/// Returns `b[off..off+n]` if it is in range.
#[inline(always)]
fn span(b: &[u8], off: u32, n: u64) -> Option<&[u8]> {
    fits(b.len(), off, n).then(|| &b[off as usize..][..n as usize])
}

fn exec_op(p: &mut Program<'_>, op: &Op) -> bool {
    // Ops may address up to ARENA_SLACK bytes less than the arena, the over-read reserve
    // of the vector tiers.
    let alen = p.arena.len().saturating_sub(ARENA_SLACK);
    let (src, r2s) = (p.input, p.r2s);
    let to_dst = op.flags & FLAG_DST != 0;
    let (k, d) = (u64::from(op.k), u64::from(op.d));
    if !fits(if to_dst { p.dst.len() } else { alen }, op.out, k) {
        return false;
    }
    let (ku, out) = (op.k as usize, op.out as usize);
    match op.kind {
        OP_VV => {
            let Some(bm) = span(src, op.bm, k.div_ceil(8)) else {
                return false;
            };
            if !fits(alen, op.l, k) || !fits(alen, op.r, k) {
                return false;
            }
            let a = p.arena.as_mut_ptr();
            let o = if to_dst { p.dst.as_mut_ptr() } else { a };
            // SAFETY: out, l and r are in range for k bytes of their buffers (checked above),
            // bm has ceil(k/8) bytes, and the arena pointers all derive from a.
            let nr = unsafe {
                merge_vv(
                    o.add(out),
                    ku,
                    bm,
                    a.add(op.l as usize),
                    a.add(op.r as usize),
                )
            };
            nr == op.kr as usize
        }
        OP_CV => {
            let Some(bm) = span(src, op.bm, k.div_ceil(8)) else {
                return false;
            };
            if !fits(alen, op.r, k) {
                return false;
            }
            let a = p.arena.as_mut_ptr();
            let o = if to_dst { p.dst.as_mut_ptr() } else { a };
            // SAFETY: as for OP_VV.
            let nr = unsafe { merge_cv(o.add(out), ku, bm, a.add(op.r as usize), op.p) };
            nr == op.kr as usize
        }
        OP_CC => {
            let Some(bm) = span(src, op.bm, k.div_ceil(8)) else {
                return false;
            };
            merge_cc(out_slice(p, to_dst, out, ku), bm, op.p, op.q);
            true
        }
        OP_V512 | OP_V128 => {
            let ln: u64 = if op.kind == OP_V512 { 64 } else { 16 };
            let region = span(src, op.bm, k * d / 8);
            let lut = flat_lut(r2s, op);
            let (Some(region), Some(lut)) = (region, lut) else {
                return false;
            };
            if d > 7 || k % (8 * ln) != 0 {
                return false;
            }
            let o = out_slice(p, to_dst, out, ku);
            if ln == 64 {
                vert_unpack::<64>(o, region, op.d, lut);
            } else {
                vert_unpack::<16>(o, region, op.d, lut);
            }
            true
        }
        OP_NAT => {
            let region = span(src, op.bm, (k * d).div_ceil(8));
            let lut = flat_lut(r2s, op);
            let (Some(region), Some(lut)) = (region, lut) else {
                return false;
            };
            nat_unpack(out_slice(p, to_dst, out, ku), region, op.d, lut);
            true
        }
        OP_COPY => {
            let Some(region) = span(src, op.bm, k) else {
                return false;
            };
            if d != 8 {
                return false;
            }
            out_slice(p, to_dst, out, ku).copy_from_slice(region);
            true
        }
        _ => false,
    }
}

/// Returns an op's output, which the caller has range checked.
#[inline(always)]
fn out_slice<'p>(p: &'p mut Program<'_>, to_dst: bool, out: usize, k: usize) -> &'p mut [u8] {
    let b = if to_dst { &mut *p.dst } else { &mut *p.arena };
    &mut b[out..out + k]
}

/// Returns the 1<<D symbols of a flat subtree; D must be 2..8.
fn flat_lut<'a>(r2s: Option<&'a [u8; R2S_LEN]>, op: &Op) -> Option<&'a [u8]> {
    if op.d < 2 || op.d > 8 {
        return None;
    }
    let (p, n) = (usize::from(op.p), 1usize << op.d);
    let r2s = r2s?;
    (p + n <= R2S_LEN).then(|| &r2s[p..p + n])
}

/// # Safety
/// `p` must be valid for an 8-byte read.
#[inline(always)]
unsafe fn read64(p: *const u8) -> u64 {
    // SAFETY: the caller guarantees 8 readable bytes; no alignment is required.
    u64::from_le(unsafe { p.cast::<u64>().read_unaligned() })
}

/// # Safety
/// `p` must be valid for an 8-byte write.
#[inline(always)]
unsafe fn write64(p: *mut u8, v: u64) {
    // SAFETY: the caller guarantees 8 writable bytes; no alignment is required.
    unsafe { p.cast::<u64>().write_unaligned(v.to_le()) }
}

// The merges load both sources before storing each output byte (the 8-byte fast paths load
// the whole chunk first). In the in-place layout the larger child ends at out[K) and
// starts K_small bytes in; the write cursor j = l+r then never passes its read cursor, so
// no unread byte is overwritten. Sources are read only inside [0, K): l, r <= j.

/// Writes `out[j] = bit j ? r[nr++] : l[nl++]` for j < k and returns nr.
///
/// # Safety
/// `out` must be valid for k byte writes, `l` and `r` for k byte reads, and
/// `bm.len() == k.div_ceil(8)`. Pointers into one buffer must derive from one pointer.
unsafe fn merge_vv(out: *mut u8, k: usize, bm: &[u8], l: *const u8, r: *const u8) -> usize {
    let (mut nl, mut nr) = (0, 0);
    for (i, &b) in bm.iter().enumerate() {
        let (o, rem) = (i * 8, k - i * 8);
        if rem >= 8 {
            match b {
                0 => {
                    // SAFETY: nl + 8 <= o + 8 <= k.
                    unsafe { write64(out.add(o), read64(l.add(nl))) };
                    nl += 8;
                    continue;
                }
                0xff => {
                    // SAFETY: nr + 8 <= o + 8 <= k.
                    unsafe { write64(out.add(o), read64(r.add(nr))) };
                    nr += 8;
                    continue;
                }
                _ => {}
            }
        }
        for t in 0..rem.min(8) {
            let g = usize::from(b >> t & 1);
            // SAFETY: nl + nr == o + t < k.
            unsafe {
                let (x, y) = (l.add(nl).read(), r.add(nr).read());
                out.add(o + t).write(if g != 0 { y } else { x });
            }
            nr += g;
            nl += g ^ 1;
        }
    }
    nr
}

/// Writes `out[j] = bit j ? r[nr++] : c` for j < k and returns nr.
///
/// # Safety
/// As for [`merge_vv`].
unsafe fn merge_cv(out: *mut u8, k: usize, bm: &[u8], r: *const u8, c: u8) -> usize {
    let cc = u64::from(c) * 0x0101010101010101;
    let mut nr = 0;
    for (i, &b) in bm.iter().enumerate() {
        let (o, rem) = (i * 8, k - i * 8);
        if rem >= 8 {
            match b {
                0 => {
                    // SAFETY: o + 8 <= k.
                    unsafe { write64(out.add(o), cc) };
                    continue;
                }
                0xff => {
                    // SAFETY: nr + 8 <= o + 8 <= k.
                    unsafe { write64(out.add(o), read64(r.add(nr))) };
                    nr += 8;
                    continue;
                }
                _ => {}
            }
        }
        for t in 0..rem.min(8) {
            let g = usize::from(b >> t & 1);
            // SAFETY: nr <= o + t < k.
            unsafe {
                let x = r.add(nr).read();
                out.add(o + t).write(if g == 0 { c } else { x });
            }
            nr += g;
        }
    }
    nr
}

/// Writes `out[j] = bit j ? q : p`.
fn merge_cc(out: &mut [u8], bm: &[u8], p: u8, q: u8) {
    let (pp, qq) = (
        u64::from(p) * 0x0101010101010101,
        u64::from(q) * 0x0101010101010101,
    );
    for (o, &b) in out.chunks_mut(8).zip(bm) {
        if o.len() == 8 {
            // Spread bit k of b to all bits of byte k: isolate it in byte k, then carry
            // any non-zero byte into its bit 7.
            let t = (u64::from(b) * 0x0101010101010101) & 0x8040201008040201;
            let m = ((t + 0x7f7f7f7f7f7f7f7f) >> 7 & 0x0101010101010101) * 0xff;
            o.copy_from_slice(&(pp & !m | qq & m).to_le_bytes());
            continue;
        }
        for (t, x) in o.iter_mut().enumerate() {
            *x = if b >> t & 1 != 0 { q } else { p };
        }
    }
}

/// The inverse of `vert_pack`, mapping codes through lut.
fn vert_unpack<const LN: usize>(out: &mut [u8], src: &[u8], d: u8, lut: &[u8]) {
    let mask = (lut.len() - 1) as u64;
    let d = usize::from(d);
    for (o, blk) in out.chunks_exact_mut(8 * LN).zip(src.chunks_exact(LN * d)) {
        for l in 0..LN {
            let mut w = 0u64;
            for j in 0..d {
                // SAFETY: j < d and l < LN, so j*LN + l < LN*d == blk.len().
                w |= u64::from(unsafe { *blk.get_unchecked(j * LN + l) }) << (8 * j);
            }
            for s in 0..8 {
                // SAFETY: the index is masked to lut.len() - 1.
                o[s * LN + l] = unsafe { *lut.get_unchecked((w >> (s * d) & mask) as usize) };
            }
        }
    }
}

/// The inverse of `nat_pack`, mapping codes through lut.
fn nat_unpack(out: &mut [u8], src: &[u8], d: u8, lut: &[u8]) {
    let mask = (lut.len() - 1) as u64;
    let d = usize::from(d);
    for (i, o) in out.chunks_mut(8).enumerate() {
        let s = &src[i * d..];
        let w = if s.len() >= 8 {
            load64(s, 0)
        } else {
            s.iter()
                .enumerate()
                .fold(0, |w, (j, &c)| w | u64::from(c) << (8 * j))
        };
        for (k, x) in o.iter_mut().enumerate() {
            // SAFETY: the index is masked to lut.len() - 1.
            *x = unsafe { *lut.get_unchecked((w >> (k * d) & mask) as usize) };
        }
    }
}
