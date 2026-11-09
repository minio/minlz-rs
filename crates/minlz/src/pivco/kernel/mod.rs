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

//! The data-parallel primitives of the pivco codec (upstream
//! `src/pivco_huffman_primitives.h`), per CPU tier.
//!
//! The types in this module are the contract the tiers implement; the vector tiers rely on
//! the [`Op`] layout and the slack constants.
//!
//! Dispatch is a `match` on [`Tier`] with direct calls.

#[cfg(all(target_arch = "x86_64", not(miri)))]
mod avx2;
/// Without x86-64 (or under Miri) the AVX2 tier runs the generic kernels.
#[cfg(not(all(target_arch = "x86_64", not(miri))))]
mod avx2 {
    pub(super) use super::generic::{
        gather_generic as gather_avx2, part_full_generic as part_full_avx2,
        part_none_generic as part_none_avx2, part_right_generic as part_right_avx2,
    };
    pub(super) use super::generic_exec::exec_generic as exec_avx2;
    pub(super) use super::generic_pack::pack_generic as pack_avx2;
}
#[cfg(all(target_arch = "x86_64", not(miri)))]
mod avx512;
mod generic;
mod generic_exec;
mod generic_pack;
mod layout;
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
mod neon;

/// Without NEON, [`Tier::Neon`] runs the generic kernels (Go `neon_other.go`).
#[cfg(not(all(target_arch = "aarch64", target_endian = "little", not(miri))))]
mod neon {
    pub(super) use super::generic::{
        gather_generic as gather_neon, part_full_generic as part_full_neon,
        part_none_generic as part_none_neon, part_right_generic as part_right_neon,
    };
    pub(super) use super::generic_exec::exec_generic as exec_neon;
    pub(super) use super::generic_pack::pack_generic as pack_neon;
}

#[cfg(test)]
mod tests;

pub use layout::vert_split;

use core::fmt;

use crate::cpu;

/// One pre-order schedule record (upstream `pivco_sched_rec_t`).
///
/// * Kind = `op & 3`; flat depth D = `op >> 2` (2..8).
/// * `param`: FULL/LEAF_LEFT/BOTH: largest rank of the left subtree (ranks > `param` go
///   right); FLAT: first rank of the subtree.
/// * `right`: offset to the right child's record (FULL), 1 (LEAF_LEFT), 0 otherwise.
///
/// The left child, when it has a record, is at index+1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Rec {
    /// Kind and flat depth.
    pub op: u8,
    /// Split rank or first rank, by kind.
    pub param: u8,
    /// Offset to the right child's record.
    pub right: u8,
}

/// Record kind: internal node with two coded children.
pub const KIND_FULL: u8 = 0;
/// Record kind: subtree of 2^D leaves at one depth, coded as D-bit codes.
pub const KIND_FLAT: u8 = 1;
/// Record kind: node with two leaves.
pub const KIND_BOTH: u8 = 2;
/// Record kind: node whose left child is a leaf.
pub const KIND_LEAF_LEFT: u8 = 3;

impl Rec {
    /// Returns the record kind.
    #[inline(always)]
    pub const fn kind(self) -> u8 {
        self.op & 3
    }

    /// Returns D for flat records.
    #[inline(always)]
    pub const fn flat_depth(self) -> u8 {
        self.op >> 2
    }
}

// Decoder op kinds. All merges read bitmap bit j as (bm[j>>3]>>(j&7))&1.

/// `out[j] = bit ? R[r++] : L[l++]`
pub const OP_VV: u8 = 0;
/// `out[j] = bit ? R[r++] : P`
pub const OP_CV: u8 = 1;
/// `out[j] = bit ? Q : P`
pub const OP_CC: u8 = 2;
/// Flat, vertical layout, 64-lane blocks (K is a multiple of 512).
pub const OP_V512: u8 = 3;
/// Flat, vertical layout, 16-lane blocks (K is a multiple of 128).
pub const OP_V128: u8 = 4;
/// Flat, natural LSB-first packing.
pub const OP_NAT: u8 = 5;
/// Flat D=8: out = region (identity code to symbol map).
pub const OP_COPY: u8 = 6;

/// Makes `out` index [`Program::dst`] instead of [`Program::arena`].
pub const FLAG_DST: u8 = 1;

/// One decoder step. It is pointer-free and 32 bytes, laid out as Go's `kernel.Op`.
///
/// `l` and `r` index [`Program::arena`]. For flats, `p` is the first rank into
/// [`Program::r2s`] and `bm` the start of the packed region; codes are masked to D bits
/// before lookup.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Op {
    /// One of the `OP_*` kinds.
    pub kind: u8,
    /// [`FLAG_DST`] or 0.
    pub flags: u8,
    /// Flat depth.
    pub d: u8,
    /// Left constant (CV, CC) or first rank (flats).
    pub p: u8,
    /// Right constant (CC).
    pub q: u8,
    /// Zero; keeps the 32 bytes free of uninitialized padding.
    pub pad: [u8; 3],
    /// Number of output symbols.
    pub k: u32,
    /// Number of set bits required in bitmap[0:K) (VV, CV).
    pub kr: u32,
    /// Offset of the bitmap or flat region in [`Program::input`].
    pub bm: u32,
    /// Offset of the output.
    pub out: u32,
    /// Offset of the left child in the arena (VV).
    pub l: u32,
    /// Offset of the right child in the arena (VV, CV).
    pub r: u32,
}

/// The length of the padded rank-to-symbol table; vector kernels may load up to 64 bytes
/// at any first rank.
pub const R2S_LEN: usize = 256 + 64;

/// The list of decoder ops for one block.
///
/// [`Tier::exec`] runs the ops in order and checks each one against the slices before it
/// runs it, so a malformed op fails closed; beyond that, kernels only rely on the slack
/// documented on the constants.
pub struct Program<'a> {
    /// The ops, run in order.
    pub ops: &'a [Op],
    /// The block (Go: `In`); reads stay inside it (no slack), not always inside an op's
    /// region.
    pub input: &'a [u8],
    /// Child buffers; kernels may read up to [`ARENA_SLACK`] bytes past any cursor.
    pub arena: &'a mut [u8],
    /// Final output; writes are exact.
    pub dst: &'a mut [u8],
    /// Rank-to-symbol table; `None` fails every flat op.
    pub r2s: Option<&'a [u8; R2S_LEN]>,
}

/// Rank, right-output and bitmap staging slices passed to the encoder kernels must have at
/// least n+`RANK_SLACK` (bitmaps: ceil(n/8)+`RANK_SLACK`) bytes; kernels may store junk
/// there. Rank inputs and the in-place left outputs of [`Tier::part_full`] are exact.
pub const RANK_SLACK: usize = 64;
/// [`Program::arena`] must extend this far past the last byte any op can address.
pub const ARENA_SLACK: usize = 128;
/// The scratch size passed to [`Tier::histogram`].
pub const HIST_SCRATCH: usize = 4 * 16384 + 64;

/// Loads a little-endian value from `b[i..]`.
#[inline(always)]
pub(super) fn load16(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes(b[i..i + 2].try_into().unwrap())
}

/// Loads a little-endian value from `b[i..]`.
#[inline(always)]
pub(super) fn load64(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

/// Stores `v` little-endian to `b[i..]`.
#[inline(always)]
pub(super) fn store64(b: &mut [u8], i: usize, v: u64) {
    b[i..i + 8].copy_from_slice(&v.to_le_bytes());
}

/// A kernel implementation, lowest first per architecture.
///
/// Every method is safe to call with any tier: a tier this CPU does not support must run
/// the generic kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Portable Rust.
    #[default]
    Generic,
    /// x86-64 AVX2 + BMI2.
    Avx2,
    /// x86-64 AVX-512 (F, BW, VL, VBMI, VBMI2, VPOPCNTDQ) + BMI2.
    Avx512,
    /// arm64 NEON.
    Neon,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Tier::Generic => "generic",
            Tier::Avx2 => "avx2",
            Tier::Avx512 => "avx512",
            Tier::Neon => "neon",
        })
    }
}

impl Tier {
    /// Returns the fastest tier usable on this CPU.
    pub fn best() -> Tier {
        if cpu::has(cpu::PIVCO_AVX512) {
            return Tier::Avx512;
        }
        if cpu::has(cpu::PIVCO_AVX2) {
            return Tier::Avx2;
        }
        if cpu::has(cpu::NEON) {
            return Tier::Neon;
        }
        Tier::Generic
    }

    /// Returns all usable tiers, best first; [`Tier::Generic`] is always last.
    pub fn tiers() -> Vec<Tier> {
        let mut ts = Vec::new();
        if cpu::has(cpu::PIVCO_AVX512) {
            ts.push(Tier::Avx512);
        }
        if cpu::has(cpu::PIVCO_AVX2) {
            ts.push(Tier::Avx2);
        }
        if cpu::has(cpu::NEON) {
            ts.push(Tier::Neon);
        }
        ts.push(Tier::Generic);
        ts
    }

    /// Sets `ranks[i] = s2r[src[i]]` and returns the largest rank written.
    #[inline]
    pub fn gather(self, ranks: &mut [u8], src: &[u8], s2r: &[u8; 256]) -> u8 {
        match self {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            // SAFETY: the guard checked the CPU features.
            Tier::Avx512 if cpu::has(cpu::PIVCO_AVX512) => unsafe {
                avx512::gather(ranks, src, s2r)
            },
            Tier::Avx512 => generic::gather_generic(ranks, src, s2r),
            Tier::Avx2 => avx2::gather_avx2(ranks, src, s2r),
            Tier::Neon => neon::gather_neon(ranks, src, s2r),
            Tier::Generic => generic::gather_generic(ranks, src, s2r),
        }
    }

    /// Partitions `ranks[..n]` by rank > `thr`: writes the bitmap to `bm` (padding bits
    /// zero), moves right ranks in order to `right[..nr]`, compacts left ranks in order into
    /// `ranks[..n-nr]`, and returns nr.
    #[inline]
    pub fn part_full(
        self,
        ranks: &mut [u8],
        n: usize,
        right: &mut [u8],
        bm: &mut [u8],
        thr: u8,
    ) -> usize {
        match self {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            // SAFETY: the guard checked the CPU features.
            Tier::Avx512 if cpu::has(cpu::PIVCO_AVX512) => unsafe {
                avx512::part_full(ranks, n, right, bm, thr)
            },
            Tier::Avx512 => generic::part_full_generic(ranks, n, right, bm, thr),
            Tier::Avx2 => avx2::part_full_avx2(ranks, n, right, bm, thr),
            Tier::Neon => neon::part_full_neon(ranks, n, right, bm, thr),
            Tier::Generic => generic::part_full_generic(ranks, n, right, bm, thr),
        }
    }

    /// Is [`Tier::part_full`] without compacting the left ranks; `ranks` is read-only.
    #[inline]
    pub fn part_right(
        self,
        ranks: &[u8],
        n: usize,
        right: &mut [u8],
        bm: &mut [u8],
        thr: u8,
    ) -> usize {
        match self {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            // SAFETY: the guard checked the CPU features.
            Tier::Avx512 if cpu::has(cpu::PIVCO_AVX512) => unsafe {
                avx512::part_right(ranks, n, right, bm, thr)
            },
            Tier::Avx512 => generic::part_right_generic(ranks, n, right, bm, thr),
            Tier::Avx2 => avx2::part_right_avx2(ranks, n, right, bm, thr),
            Tier::Neon => neon::part_right_neon(ranks, n, right, bm, thr),
            Tier::Generic => generic::part_right_generic(ranks, n, right, bm, thr),
        }
    }

    /// Writes only the bitmap and returns the number of set bits.
    #[inline]
    pub fn part_none(self, ranks: &[u8], n: usize, bm: &mut [u8], thr: u8) -> usize {
        match self {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            // SAFETY: the guard checked the CPU features.
            Tier::Avx512 if cpu::has(cpu::PIVCO_AVX512) => unsafe {
                avx512::part_none(ranks, n, bm, thr)
            },
            Tier::Avx512 => generic::part_none_generic(ranks, n, bm, thr),
            Tier::Avx2 => avx2::part_none_avx2(ranks, n, bm, thr),
            Tier::Neon => neon::part_none_neon(ranks, n, bm, thr),
            Tier::Generic => generic::part_none_generic(ranks, n, bm, thr),
        }
    }

    /// Writes the codes `ranks[i]-base` (each < 1<<d) of a flat region in the vertical
    /// layout to exactly `dst[..(n*d+7)/8]`, with zero padding bits.
    #[inline]
    pub fn pack(self, dst: &mut [u8], ranks: &[u8], n: usize, d: u8, base: u8) {
        match self {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            // SAFETY: the guard checked the CPU features.
            Tier::Avx512 if cpu::has(cpu::PIVCO_AVX512) => unsafe {
                avx512::pack(dst, ranks, n, d, base)
            },
            Tier::Avx512 => generic_pack::pack_generic(dst, ranks, n, d, base),
            Tier::Avx2 => avx2::pack_avx2(dst, ranks, n, d, base),
            Tier::Neon => neon::pack_neon(dst, ranks, n, d, base),
            Tier::Generic => generic_pack::pack_generic(dst, ranks, n, d, base),
        }
    }

    /// Runs `p.ops`. It returns `Err` with the index of the first op that is malformed
    /// (unknown kind, D out of range, K not a multiple of its layout block, no R2S), out of
    /// range of the slices, or whose bitmap popcount does not match its KR.
    #[inline]
    pub fn exec(self, p: &mut Program<'_>) -> Result<(), usize> {
        match self {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            // SAFETY: the guard checked the CPU features.
            Tier::Avx512 if cpu::has(cpu::PIVCO_AVX512) => unsafe { avx512::exec(p) },
            Tier::Avx512 => generic_exec::exec_generic(p),
            Tier::Avx2 => avx2::exec_avx2(p),
            Tier::Neon => neon::exec_neon(p),
            Tier::Generic => generic_exec::exec_generic(p),
        }
    }

    /// Adds the byte counts of `src` to `h`. `scratch` must hold [`HIST_SCRATCH`] bytes if
    /// [`Tier::hist_uses_scratch`], else it may be empty. A shorter `scratch` runs the
    /// generic histogram, as the features may be re-enabled after the caller checked.
    #[inline]
    pub fn histogram(self, h: &mut [u32; 256], src: &[u8], scratch: &mut [u8]) {
        let _ = scratch; // only the AVX-512 histogram uses it
        match self {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            // SAFETY: hist_uses_scratch checked the CPU features.
            Tier::Avx512 if scratch.len() >= HIST_SCRATCH && self.hist_uses_scratch() => unsafe {
                avx512::histogram(h, src, scratch)
            },
            Tier::Avx512 => generic::histogram_generic(h, src),
            Tier::Avx2 => generic::histogram_generic(h, src),
            Tier::Neon => generic::histogram_generic(h, src),
            Tier::Generic => generic::histogram_generic(h, src),
        }
    }

    /// Reports whether [`Tier::histogram`] needs its scratch, so that callers allocate it
    /// only for tiers that do.
    #[inline]
    pub fn hist_uses_scratch(self) -> bool {
        match self {
            Tier::Avx512 => cpu::has(cpu::PIVCO_AVX512 | cpu::PIVCO_HIST_AVX512),
            Tier::Avx2 | Tier::Neon | Tier::Generic => false,
        }
    }
}
