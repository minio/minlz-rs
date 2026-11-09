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

//! The operation loops of SIMD blocks, Go's `simdExecGo`, `simdExecFlatGo`, `simdDeltaRef` and
//! `simdExecExact` (`decode-simd.go`), and their dispatch by CPU tier (`decode_amd64.go`,
//! `decode_arm64.go`, `decode_other.go`).
//!
//! # Memory rules
//!
//! The fast loops ([`Tier::exec`], [`Tier::exec_flat`]) copy literals, matches and delta references
//! in blocks of 32 bytes, so they read and write past the bytes an operation needs. The decoder
//! keeps these rules for every block, valid or not, and the vector loops may rely on them. For a
//! run of operations of a [`Chunk`] from stream positions `cur` (0 for [`Tier::exec`]):
//!
//! 1. **Output slack.** The operations end at least [`SLACK`](super::SLACK) bytes before `dst.len()`:
//!    [`Chunk::run`] splits a chunk so, and [`exec_exact`] runs the operations after the split.
//!    A fast loop writes at most `SLACK` bytes past the end of an operation, and nothing before
//!    `st[0]`.
//! 2. **Literals.** `lits` is readable `SLACK` bytes past the last literal the operations use.
//!    The decoder passes its whole literal buffer: the block's size plus [`PAD`](super::PAD) bytes, and the
//!    literals used never pass the output position.
//! 3. **Streams stay readable past their values.** `tok` is readable 7 tokens past the last
//!    operation run (Go's NEON batches read 8 tokens), `ll` and `offc` 8 bytes past the last value
//!    the operations use, and `raw` 8 bytes from byte `pos >> 3` of each bit position `pos` the
//!    run reaches, so `bits / 8 + 8` bytes in all. `esc` only needs its values.
//!    The decoder gives more: decoded streams sit in buffers with at least [`PAD`](super::PAD) bytes after
//!    their values, and stored streams are slices to the end of the block, which has at least
//!    9 bytes after the data of any chunk (its 6 or more record bytes, the reverse varint after
//!    the records, the literal section and the reverse varint that ends the block).
//!    The bytes past the values are arbitrary: they must not change the output. The Go loops
//!    read the literal length and offset symbol of every operation and mask them.
//! 4. **Offsets are clamped** to the output position, so matches and delta references never
//!    reach before `dst[0]`.
//! 5. **Output past the position** is initialized but arbitrary: past `st[0]`, `dst` holds what
//!    the caller's buffer held or zeroes ([`decode_at`](super::decode_at)), and the writes of
//!    rule 1. The fast loops may load those bytes, but only into output that later operations
//!    overwrite.
//!
//! The Go loops here index slices, so a broken rule panics instead of reaching out of bounds. The
//! tests run every tier on buffers that end where the rules say (Go's `TestSIMDExec`).
//!
//! # State
//!
//! `st` holds the output position, the literal position, the repeat offset and an error flag.
//! The loops set the flag for a match longer than [`MAX_LEN`] at an offset below `MAX_LEN`
//! (after clamping, and raising for the no-overlap flag), which `SPEC_SIMD.md` 4.3 forbids;
//! they still run such a match, within the rules above.

use super::decode::Chunk;
#[cfg(all(target_arch = "x86_64", not(miri)))]
use super::exec_avx2::{
    exec_avx2, exec_delta_avx2, exec_flat_avx2, exec_flat_delta_avx2, ll_stats_avx2,
    off_stats_avx2, tok_stats_avx2,
};
use super::{MAX_LEN, OFF_TAB};

/// The exec loops and stats of a CPU tier: Go's `cpu` flags that `simdExec`, `simdExecFlat`
/// and the stats dispatch on. A vector tier whose loops aren't built for the target runs the Go
/// loops, and [`Tier::usable`] reports it unusable so tests don't run them twice.
///
/// A vector tier's arms call `#[target_feature]` loops without checking the CPU again, so a
/// vector variant may only come from [`Tier::best`] or [`Tier::tiers`], which check it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tier {
    /// The Go loops and the SWAR stats, on every CPU.
    Go,
    /// AVX2 and BMI2 on x86-64 (Go `cpu.AVX2`).
    Avx2,
    /// AVX-512 on x86-64 (Go `cpu.AVX512`).
    Avx512,
    /// The per-operation NEON loops on aarch64 (Go `cpu.NEON` without `simdBatchNEON`).
    #[allow(dead_code, reason = "only tests run it, as Go's simdTiers")]
    Neon,
    /// The batched NEON loops on aarch64 (Go `simdBatchNEON`).
    NeonBatch,
}

#[cfg(test)]
thread_local! {
    static FORCED: core::cell::Cell<Option<Tier>> = const { core::cell::Cell::new(None) };
}

/// Runs f once per usable tier, with the decodes of this thread forced to it (Go `simdTiers`).
#[cfg(test)]
pub(crate) fn simd_tiers(mut f: impl FnMut(Tier)) {
    for t in Tier::tiers() {
        FORCED.set(Some(t));
        f(t);
    }
    FORCED.set(None);
}

/// The signature of the AVX2 and NEON exec loops, Go's
/// `simdExecAVX2(dst, lits, tok, ll, offc, raw, esc *byte, n int, st *[4]int)`: `exec_avx2`,
/// `exec_delta_avx2`, `exec_neon`, `exec_delta_neon`, `exec_batch_neon` and
/// `exec_batch_delta_neon`. They run the first n operations of a chunk, from the start of its
/// streams, as [`exec_go`] does (the delta variants for chunks with delta literals).
///
/// # Safety
///
/// The CPU has the features of the loop's tier, and the pointers come from `dst`, `lits` and the
/// streams of a [`Chunk`] that keep the [memory rules](self) for these operations.
pub(super) type ExecFn = unsafe fn(
    dst: *mut u8,
    lits: *const u8,
    tok: *const u8,
    ll: *const u8,
    offc: *const u8,
    raw: *const u8,
    esc: *const u8,
    n: usize,
    st: &mut [usize; 4],
);

/// [`ExecFn`] of the AVX-512 loops, Go's `simdExecAVX512`: `exec_avx512` and `exec_delta_avx512`.
/// `nraw` is the readable size of `raw`.
///
/// # Safety
///
/// As for [`ExecFn`].
pub(super) type ExecRawFn = unsafe fn(
    dst: *mut u8,
    lits: *const u8,
    tok: *const u8,
    ll: *const u8,
    offc: *const u8,
    raw: *const u8,
    nraw: usize,
    esc: *const u8,
    n: usize,
    st: &mut [usize; 4],
);

/// The signature of the AVX2 and NEON exec loops for chunks with the no-overlap flag, Go's
/// `simdExecFlatAVX2(dst, lits, tok, ll, offc, raw, esc *byte, n, pos int, st *[4]int)`:
/// `exec_flat_avx2`, `exec_flat_delta_avx2`, `exec_flat_neon`, `exec_flat_delta_neon`,
/// `exec_batch_flat_neon` and `exec_batch_flat_delta_neon`. They run n operations as
/// [`exec_flat_go`] does, from `tok`, `ll`, `offc` and `esc` at the operation's stream positions
/// and from bit `pos` of `raw`. The operations start at output position [`MAX_LEN`] or later.
///
/// # Safety
///
/// As for [`ExecFn`].
pub(super) type ExecFlatFn = unsafe fn(
    dst: *mut u8,
    lits: *const u8,
    tok: *const u8,
    ll: *const u8,
    offc: *const u8,
    raw: *const u8,
    esc: *const u8,
    n: usize,
    pos: usize,
    st: &mut [usize; 4],
);

/// [`ExecFlatFn`] of the AVX-512 loops, Go's `simdExecFlatAVX512`: `exec_flat_avx512` and
/// `exec_flat_delta_avx512`. `nraw` is the readable size of `raw` from its start, not from `pos`.
///
/// # Safety
///
/// As for [`ExecFn`].
pub(super) type ExecFlatRawFn = unsafe fn(
    dst: *mut u8,
    lits: *const u8,
    tok: *const u8,
    ll: *const u8,
    offc: *const u8,
    raw: *const u8,
    nraw: usize,
    esc: *const u8,
    n: usize,
    pos: usize,
    st: &mut [usize; 4],
);

/// The token stats of a vector tier over all of `b` (Go `simdTokStatsAVX2` and friends):
/// the summed match length codes, the operations with literals, the repeats and the escaped
/// matches, and whether every token is at most 135. Go's AVX2 and AVX-512 loops count literals
/// twice and escapes four times, which [`Tier::tok_stats_asm`] undoes.
///
/// # Safety
///
/// The CPU has the features of the tier. AVX2 takes a multiple of 32 bytes, NEON a multiple of
/// 16 bytes, AVX-512 any length above 0.
#[allow(dead_code, reason = "the signature of the vector tiers")]
pub(super) type TokStatsFn = unsafe fn(b: &[u8]) -> (usize, usize, usize, usize, bool);

/// The literal length stats of a vector tier over all of `b` (Go `simdLLStatsAVX2`): the sum of
/// the bytes, how many are 33, and whether every byte is 1 to 33. The sum must be exact for
/// any bytes, as escapes are summed with it. Lengths as for [`TokStatsFn`].
///
/// # Safety
///
/// As for [`TokStatsFn`].
#[allow(dead_code, reason = "the signature of the vector tiers")]
pub(super) type LLStatsFn = unsafe fn(b: &[u8]) -> (usize, usize, bool);

/// The offset symbol stats of a vector tier over all of `b`, and whether every byte is 1 to 175.
/// Go's AVX2 and AVX-512 loops return the sum of `b >> 3` and how often it is 0, which
/// [`Tier::off_stats_asm`] turns into raw bits; NEON returns the raw bits and 0. Lengths as for
/// [`TokStatsFn`].
///
/// # Safety
///
/// As for [`TokStatsFn`].
#[allow(dead_code, reason = "the signature of the vector tiers")]
pub(super) type OffStatsFn = unsafe fn(b: &[u8]) -> (usize, usize, bool);

impl Tier {
    /// Every usable tier, the Go loops first, as Go's `simdTiers` runs them.
    #[cfg_attr(not(test), allow(dead_code, reason = "for tests"))]
    pub(crate) fn tiers() -> Vec<Tier> {
        [
            Tier::Go,
            Tier::Avx2,
            Tier::Avx512,
            Tier::Neon,
            Tier::NeonBatch,
        ]
        .into_iter()
        .filter(|t| t.usable())
        .collect()
    }

    /// The tier Go dispatches to on this CPU.
    pub(crate) fn best() -> Tier {
        [Tier::Avx512, Tier::Avx2, Tier::NeonBatch]
            .into_iter()
            .find(|t| t.usable())
            .unwrap_or(Tier::Go)
    }

    /// The tier of the decodes of this thread: [`Tier::best`], or in tests the tier
    /// [`simd_tiers`] forces.
    pub(crate) fn current() -> Tier {
        #[cfg(test)]
        if let Some(t) = FORCED.get() {
            return t;
        }
        Tier::best()
    }

    /// Reports whether the loops of the tier exist and the CPU has their features. Miri gets
    /// the Go loops only.
    pub(crate) fn usable(self) -> bool {
        match self {
            Tier::Go => true,
            Tier::Avx2 => crate::cpu::has(crate::cpu::AVX2),
            Tier::Avx512 => crate::cpu::has(crate::cpu::AVX512),
            // cpu.rs detects NEON only where exec_neon is built.
            Tier::Neon | Tier::NeonBatch => crate::cpu::has(crate::cpu::NEON),
        }
    }

    /// Runs the first n operations of c (Go `simdExec`). See [`exec_go`].
    ///
    /// # Safety
    ///
    /// The tier comes from [`Tier::best`] or [`Tier::tiers`], c's counts are what its first n
    /// tokens read (as `tok_stats`, `ll_stats` and `off_stats` count them), and the operations
    /// keep [memory rules](self) 1 and 2. The vector loops trust all three; the Go loops panic.
    pub(super) unsafe fn exec(
        self,
        dst: &mut [u8],
        lits: &[u8],
        c: &Chunk,
        n: usize,
        st: &mut [usize; 4],
    ) {
        match self {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Tier::Avx512 => {
                use super::exec_avx512::{exec_avx512, exec_delta_avx512};
                let f: ExecRawFn = if c.delta {
                    exec_delta_avx512
                } else {
                    exec_avx512
                };
                // SAFETY: the caller keeps the CPU and memory rules (see # Safety).
                unsafe { exec_raw_vec(f, dst, lits, c, n, st) }
            }
            #[cfg(not(all(target_arch = "x86_64", not(miri))))]
            Tier::Avx512 => exec_go(dst, lits, c, n, st),
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Tier::Avx2 => {
                let f: ExecFn = if c.delta { exec_delta_avx2 } else { exec_avx2 };
                // SAFETY: as for Avx512.
                unsafe { exec_vec(f, dst, lits, c, n, st) }
            }
            #[cfg(not(all(target_arch = "x86_64", not(miri))))]
            Tier::Avx2 => exec_go(dst, lits, c, n, st),
            #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
            Tier::NeonBatch | Tier::Neon => {
                use super::exec_neon::*;
                let f: ExecFn = match (self, c.delta) {
                    (Tier::NeonBatch, true) => exec_batch_delta_neon,
                    (Tier::NeonBatch, false) => exec_batch_neon,
                    (_, true) => exec_delta_neon,
                    (_, false) => exec_neon,
                };
                // SAFETY: as for Avx512.
                unsafe { exec_vec(f, dst, lits, c, n, st) }
            }
            #[cfg(not(all(target_arch = "aarch64", target_endian = "little", not(miri))))]
            Tier::NeonBatch | Tier::Neon => exec_go(dst, lits, c, n, st),
            Tier::Go => exec_go(dst, lits, c, n, st),
        }
    }

    /// Runs operations k0 to k of c, which has the no-overlap flag, from stream positions cur
    /// (Go `simdExecFlat`). See [`exec_flat_go`].
    ///
    /// # Safety
    ///
    /// As for [`Tier::exec`], for operations k0 to k and stream positions cur.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn exec_flat(
        self,
        dst: &mut [u8],
        lits: &[u8],
        c: &Chunk,
        k0: usize,
        k: usize,
        cur: [usize; 4],
        st: &mut [usize; 4],
    ) {
        match self {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Tier::Avx512 => {
                use super::exec_avx512::{exec_flat_avx512, exec_flat_delta_avx512};
                let f: ExecFlatRawFn = if c.delta {
                    exec_flat_delta_avx512
                } else {
                    exec_flat_avx512
                };
                // SAFETY: the caller keeps the CPU and memory rules (see # Safety).
                unsafe { exec_flat_raw_vec(f, dst, lits, c, k0, k, cur, st) }
            }
            #[cfg(not(all(target_arch = "x86_64", not(miri))))]
            Tier::Avx512 => exec_flat_go(dst, lits, c, k0, k, cur, st),
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Tier::Avx2 => {
                let f: ExecFlatFn = if c.delta {
                    exec_flat_delta_avx2
                } else {
                    exec_flat_avx2
                };
                // SAFETY: as in exec.
                unsafe { exec_flat_vec(f, dst, lits, c, k0, k, cur, st) }
            }
            #[cfg(not(all(target_arch = "x86_64", not(miri))))]
            Tier::Avx2 => exec_flat_go(dst, lits, c, k0, k, cur, st),
            #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
            Tier::NeonBatch | Tier::Neon => {
                use super::exec_neon::*;
                let f: ExecFlatFn = match (self, c.delta) {
                    (Tier::NeonBatch, true) => exec_batch_flat_delta_neon,
                    (Tier::NeonBatch, false) => exec_batch_flat_neon,
                    (_, true) => exec_flat_delta_neon,
                    (_, false) => exec_flat_neon,
                };
                // SAFETY: as in exec.
                unsafe { exec_flat_vec(f, dst, lits, c, k0, k, cur, st) }
            }
            #[cfg(not(all(target_arch = "aarch64", target_endian = "little", not(miri))))]
            Tier::NeonBatch | Tier::Neon => exec_flat_go(dst, lits, c, k0, k, cur, st),
            Tier::Go => exec_flat_go(dst, lits, c, k0, k, cur, st),
        }
    }

    /// Returns the statistics of a prefix of tok and its length (Go `simdTokStatsAsm`):
    /// the summed match length codes, the operations with literals, the repeats and the escaped
    /// matches. ok is false if the prefix has a token above 135. See `tok_stats`.
    pub(super) fn tok_stats_asm(self, _tok: &[u8]) -> (usize, usize, usize, usize, usize, bool) {
        match self {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Tier::Avx512 if !_tok.is_empty() => {
                // SAFETY: the tier comes from Tier::best or Tier::tiers, which checked the CPU.
                let (out, lit, rep, esc, ok) =
                    unsafe { super::exec_avx512::tok_stats_avx512(_tok) };
                (out, lit >> 1, rep, esc >> 2, _tok.len(), ok)
            }
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Tier::Avx2 => {
                let n = _tok.len() & !31;
                if n == 0 {
                    return (0, 0, 0, 0, 0, true);
                }
                // SAFETY: the tier is usable, so the CPU has AVX2 and BMI2.
                let (out, lit, rep, esc, ok) = unsafe { tok_stats_avx2(&_tok[..n]) };
                (out, lit >> 1, rep, esc >> 2, n, ok)
            }
            #[cfg(not(all(target_arch = "x86_64", not(miri))))]
            Tier::Avx2 => (0, 0, 0, 0, 0, true),
            Tier::Avx512 => (0, 0, 0, 0, 0, true),
            #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
            Tier::NeonBatch | Tier::Neon => {
                let n = _tok.len() & !15;
                // SAFETY: the tier is usable.
                let (out, lit, rep, esc, ok) =
                    unsafe { super::exec_neon::tok_stats_neon(&_tok[..n]) };
                (out, lit, rep, esc, n, ok)
            }
            #[cfg(not(all(target_arch = "aarch64", target_endian = "little", not(miri))))]
            Tier::NeonBatch | Tier::Neon => (0, 0, 0, 0, 0, true),
            Tier::Go => (0, 0, 0, 0, 0, true),
        }
    }

    /// Returns the sum and escapes of a prefix of ll and its length (Go `simdLLStatsAsm`).
    /// ok is false if the prefix has a length outside 1-33; the sum is exact regardless.
    /// See `ll_stats`.
    pub(super) fn ll_stats_asm(self, _ll: &[u8]) -> (usize, usize, usize, bool) {
        match self {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Tier::Avx512 if !_ll.is_empty() => {
                // SAFETY: as in tok_stats_asm.
                let (sum, n_esc, ok) = unsafe { super::exec_avx512::ll_stats_avx512(_ll) };
                (sum, n_esc, _ll.len(), ok)
            }
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Tier::Avx2 => {
                let n = _ll.len() & !31;
                if n == 0 {
                    return (0, 0, 0, true);
                }
                // SAFETY: the tier is usable, so the CPU has AVX2 and BMI2.
                let (sum, n_esc, ok) = unsafe { ll_stats_avx2(&_ll[..n]) };
                (sum, n_esc, n, ok)
            }
            #[cfg(not(all(target_arch = "x86_64", not(miri))))]
            Tier::Avx2 => (0, 0, 0, true),
            Tier::Avx512 => (0, 0, 0, true),
            #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
            Tier::NeonBatch | Tier::Neon => {
                let n = _ll.len() & !15;
                // SAFETY: the tier is usable.
                let (sum, esc, ok) = unsafe { super::exec_neon::ll_stats_neon(&_ll[..n]) };
                (sum, esc, n, ok)
            }
            #[cfg(not(all(target_arch = "aarch64", target_endian = "little", not(miri))))]
            Tier::NeonBatch | Tier::Neon => (0, 0, 0, true),
            Tier::Go => (0, 0, 0, true),
        }
    }

    /// Returns the raw bits of a prefix of offc and its length (Go `simdOffStatsAsm`).
    /// ok is false if the prefix has a symbol outside 1-175. See `off_stats`.
    pub(super) fn off_stats_asm(self, _offc: &[u8]) -> (usize, usize, bool) {
        match self {
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Tier::Avx512 if !_offc.is_empty() => {
                // SAFETY: as in tok_stats_asm.
                let (sum, zeros, ok) = unsafe { super::exec_avx512::off_stats_avx512(_offc) };
                (sum + zeros - _offc.len(), _offc.len(), ok)
            }
            #[cfg(all(target_arch = "x86_64", not(miri)))]
            Tier::Avx2 => {
                let n = _offc.len() & !31;
                if n == 0 {
                    return (0, 0, true);
                }
                // SAFETY: the tier is usable, so the CPU has AVX2 and BMI2.
                let (sum, zeros, ok) = unsafe { off_stats_avx2(&_offc[..n]) };
                // Go's sum - n + zeros, ordered so it can't underflow: each byte adds to one.
                (sum + zeros - n, n, ok)
            }
            #[cfg(not(all(target_arch = "x86_64", not(miri))))]
            Tier::Avx2 => (0, 0, true),
            Tier::Avx512 => (0, 0, true),
            #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
            Tier::NeonBatch | Tier::Neon => {
                let n = _offc.len() & !15;
                // SAFETY: the tier is usable.
                let (bits, _, ok) = unsafe { super::exec_neon::off_stats_neon(&_offc[..n]) };
                (bits, n, ok)
            }
            #[cfg(not(all(target_arch = "aarch64", target_endian = "little", not(miri))))]
            Tier::NeonBatch | Tier::Neon => (0, 0, true),
            Tier::Go => (0, 0, true),
        }
    }
}

impl Chunk<'_> {
    /// Reports whether the streams are readable as the [memory rules](self) need for any run of
    /// the chunk's operations. The vector loops check it before they run.
    pub(super) fn readable(&self) -> bool {
        self.n_tok + 7 <= self.tok.len()
            && self.n_ll + 8 <= self.ll.len()
            && self.n_off + 8 <= self.offc.len()
            && self.n_esc <= self.esc.len()
            && self.bits / 8 + 8 <= self.raw.len()
    }
}

/// Runs the vector loop f on the first n operations of c, as [`Tier::exec`] does.
///
/// # Safety
///
/// The CPU has the features of f's tier, and the operations keep memory rules 1 and 2: they
/// end at least `SLACK` bytes before the end of dst, and lits is readable `SLACK` bytes past
/// their literals. [`Chunk::run`] splits chunks so, and the decoder's literal buffer is longer
/// than dst. c's counts are what its operations read. Rule 3 is checked here against those
/// counts; rule 4 is the loops' own.
#[allow(dead_code, reason = "called by the vector tiers")]
pub(super) unsafe fn exec_vec(
    f: ExecFn,
    dst: &mut [u8],
    lits: &[u8],
    c: &Chunk,
    n: usize,
    st: &mut [usize; 4],
) {
    assert!(n <= c.n_tok && c.readable() && st[1] <= lits.len() && st[0] <= dst.len());
    // SAFETY: the pointers cover their slices, which keep the memory rules: the streams are
    // checked above against c's counts, and the counts, dst and lits are the caller's (see
    // # Safety).
    unsafe {
        f(
            dst.as_mut_ptr(),
            lits.as_ptr(),
            c.tok.as_ptr(),
            c.ll.as_ptr(),
            c.offc.as_ptr(),
            c.raw.as_ptr(),
            c.esc.as_ptr(),
            n,
            st,
        )
    }
}

/// [`exec_vec`] for the AVX-512 loops.
///
/// # Safety
///
/// As for [`exec_vec`].
#[allow(dead_code, reason = "called by the vector tiers")]
pub(super) unsafe fn exec_raw_vec(
    f: ExecRawFn,
    dst: &mut [u8],
    lits: &[u8],
    c: &Chunk,
    n: usize,
    st: &mut [usize; 4],
) {
    assert!(n <= c.n_tok && c.readable() && st[1] <= lits.len() && st[0] <= dst.len());
    // SAFETY: as in exec_vec.
    unsafe {
        f(
            dst.as_mut_ptr(),
            lits.as_ptr(),
            c.tok.as_ptr(),
            c.ll.as_ptr(),
            c.offc.as_ptr(),
            c.raw.as_ptr(),
            c.raw.len(),
            c.esc.as_ptr(),
            n,
            st,
        )
    }
}

/// Runs the vector loop f on operations k0 to k of c from stream positions cur, as
/// [`Tier::exec_flat`] does.
///
/// # Safety
///
/// As for [`exec_vec`].
#[allow(dead_code, reason = "called by the vector tiers")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn exec_flat_vec(
    f: ExecFlatFn,
    dst: &mut [u8],
    lits: &[u8],
    c: &Chunk,
    k0: usize,
    k: usize,
    cur: [usize; 4],
    st: &mut [usize; 4],
) {
    assert!(k0 <= k && k <= c.n_tok && c.readable() && st[1] <= lits.len() && st[0] <= dst.len());
    assert!(cur[0] <= c.n_ll && cur[1] <= c.n_off && cur[2] <= c.n_esc && cur[3] <= c.bits);
    // SAFETY: as in exec_vec; the stream positions are inside the streams.
    unsafe {
        f(
            dst.as_mut_ptr(),
            lits.as_ptr(),
            c.tok[k0..].as_ptr(),
            c.ll[cur[0]..].as_ptr(),
            c.offc[cur[1]..].as_ptr(),
            c.raw.as_ptr(),
            c.esc[cur[2]..].as_ptr(),
            k - k0,
            cur[3],
            st,
        )
    }
}

/// [`exec_flat_vec`] for the AVX-512 loops.
///
/// # Safety
///
/// As for [`exec_vec`].
#[allow(dead_code, reason = "called by the vector tiers")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn exec_flat_raw_vec(
    f: ExecFlatRawFn,
    dst: &mut [u8],
    lits: &[u8],
    c: &Chunk,
    k0: usize,
    k: usize,
    cur: [usize; 4],
    st: &mut [usize; 4],
) {
    assert!(k0 <= k && k <= c.n_tok && c.readable() && st[1] <= lits.len() && st[0] <= dst.len());
    assert!(cur[0] <= c.n_ll && cur[1] <= c.n_off && cur[2] <= c.n_esc && cur[3] <= c.bits);
    // SAFETY: as in exec_flat_vec.
    unsafe {
        f(
            dst.as_mut_ptr(),
            lits.as_ptr(),
            c.tok[k0..].as_ptr(),
            c.ll[cur[0]..].as_ptr(),
            c.offc[cur[1]..].as_ptr(),
            c.raw.as_ptr(),
            c.raw.len(),
            c.esc[cur[2]..].as_ptr(),
            k - k0,
            cur[3],
            st,
        )
    }
}

#[inline(always)]
pub(super) fn load64(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

#[inline(always)]
fn load256(b: &[u8], i: usize) -> [u8; 32] {
    b[i..i + 32].try_into().unwrap()
}

#[inline(always)]
fn store256(b: &mut [u8], i: usize, v: [u8; 32]) {
    b[i..i + 32].copy_from_slice(&v);
}

/// Returns the nb raw bits of raw at bit pos.
#[inline(always)]
fn raw_bits(raw: &[u8], pos: usize, nb: usize) -> usize {
    ((load64(raw, pos >> 3) >> (pos & 7)) & ((1 << nb) - 1)) as usize
}

/// Runs the first n operations of c (Go `simdExecGo`).
/// They must end at least [`SLACK`](super::SLACK) bytes before the end of dst.
pub(super) fn exec_go(dst: &mut [u8], lits: &[u8], c: &Chunk, n: usize, st: &mut [usize; 4]) {
    let (mut d, mut l, mut last) = (st[0], st[1], st[2]);
    let (mut li, mut oi, mut gi, mut pos) = (0, 0, 0, 0);
    for &t in &c.tok[..n] {
        let lit = (t as usize >> 1) & 1;
        let nrep = (t as usize & 1) ^ 1;
        let mut ll = c.ll[li] as usize & lit.wrapping_neg();
        li += lit;
        let ot = OFF_TAB[c.offc[oi] as usize];
        oi += nrep;
        let nb = (ot >> 24) as usize;
        let o = (ot & 0xffffff) as usize + raw_bits(c.raw, pos, nb);
        pos += nb & nrep.wrapping_neg();
        let mut lv = load256(lits, l);
        if c.delta {
            delta_ref(&mut lv, dst, d, last.min(d));
        }
        store256(dst, d, lv);
        if ll > MAX_LEN {
            ll += c.esc[gi] as usize;
            gi += 1;
            for j in (32..ll).step_by(32) {
                store256(dst, d + j, load256(lits, l + j));
            }
        }
        d += ll;
        l += ll;
        if nrep != 0 {
            last = o;
        }
        let r = last.min(d);
        let mut ml = (t >> 2) as usize;
        if ml > MAX_LEN {
            ml += c.esc[gi] as usize;
            gi += 1;
            if r < MAX_LEN {
                st[3] = 1;
            }
        }
        if r < ml.min(MAX_LEN) {
            for j in 0..ml {
                dst[d + j] = dst[d - r + j];
            }
        } else {
            for j in (0..ml.max(1)).step_by(32) {
                store256(dst, d + j, load256(dst, d - r + j));
            }
        }
        d += ml;
    }
    st[..3].copy_from_slice(&[d, l, last]);
}

/// Runs operations k0 to k of c, which has the no-overlap flag, from stream positions cur
/// (Go `simdExecFlatGo`). They must end at least [`SLACK`](super::SLACK) bytes before the end of dst, and
/// start at [`MAX_LEN`] or later.
pub(super) fn exec_flat_go(
    dst: &mut [u8],
    lits: &[u8],
    c: &Chunk,
    k0: usize,
    k: usize,
    cur: [usize; 4],
    st: &mut [usize; 4],
) {
    let (mut d, mut l, mut last) = (st[0], st[1], st[2]);
    let [mut li, mut oi, mut gi, mut pos] = cur;
    for &t in &c.tok[k0..k] {
        let lit = (t as usize >> 1) & 1;
        let nrep = (t as usize & 1) ^ 1;
        let mut ll = c.ll[li] as usize & lit.wrapping_neg();
        li += lit;
        let ot = OFF_TAB[c.offc[oi] as usize];
        oi += nrep;
        let nb = (ot >> 24) as usize;
        let o = (ot & 0xffffff) as usize + raw_bits(c.raw, pos, nb);
        pos += nb & nrep.wrapping_neg();
        let mut lv = load256(lits, l);
        if c.delta {
            // The references are raised to 32 as well, and d is at least 32, so they don't repeat.
            let rf = load256(dst, d - last.max(MAX_LEN).min(d));
            for (v, r) in lv.iter_mut().zip(rf) {
                *v = v.wrapping_add(r);
            }
        }
        store256(dst, d, lv);
        if ll > MAX_LEN {
            ll += c.esc[gi] as usize;
            gi += 1;
            for j in (32..ll).step_by(32) {
                store256(dst, d + j, load256(lits, l + j));
            }
        }
        d += ll;
        l += ll;
        if nrep != 0 {
            last = o;
        }
        let mut ml = (t >> 2) as usize;
        // The raised offset is at least min(ml, 32), as d is.
        let r = last.max(ml.min(MAX_LEN)).min(d);
        if ml > MAX_LEN {
            ml += c.esc[gi] as usize;
            gi += 1;
        }
        for j in (0..ml.max(1)).step_by(32) {
            store256(dst, d + j, load256(dst, d - r + j));
        }
        d += ml;
    }
    st[..3].copy_from_slice(&[d, l, last]);
}

/// Adds the delta references of literals at d to lv, for repeat offset r (Go `simdDeltaRef`).
fn delta_ref(lv: &mut [u8; 32], dst: &[u8], d: usize, r: usize) {
    if r >= 32 {
        for (v, x) in lv.iter_mut().zip(load256(dst, d - r)) {
            *v = v.wrapping_add(x);
        }
        return;
    }
    // r is only 0 at position 0, where dst[0] is 0.
    let (rs, r) = (d - r, r.max(1));
    let mut k = 0;
    for v in lv.iter_mut() {
        *v = v.wrapping_add(dst[rs + k]);
        k += 1;
        if k == r {
            k = 0;
        }
    }
}

/// Runs operations k to end of c, from stream positions cur (Go `simdExecExact`).
/// It writes nothing past the operations.
pub(super) fn exec_exact(
    dst: &mut [u8],
    lits: &[u8],
    c: &Chunk,
    k: usize,
    end: usize,
    cur: [usize; 4],
    st: &mut [usize; 4],
) {
    let (mut d, mut l, mut last) = (st[0], st[1], st[2]);
    let [mut li, mut oi, mut gi, mut pos] = cur;
    for &t in &c.tok[k..end] {
        if t & 2 != 0 {
            let mut ll = c.ll[li] as usize;
            li += 1;
            if ll > MAX_LEN {
                ll += c.esc[gi] as usize;
                gi += 1;
            }
            dst[d..d + ll].copy_from_slice(&lits[l..l + ll]);
            let r = if c.flat {
                last.max(MAX_LEN).min(d)
            } else {
                last.min(d)
            };
            if c.delta && r > 0 {
                for j in 0..ll.min(32) {
                    dst[d + j] = dst[d + j].wrapping_add(dst[d - r + j % r]);
                }
            }
            d += ll;
            l += ll;
        }
        if t & 1 == 0 {
            let ot = OFF_TAB[c.offc[oi] as usize];
            oi += 1;
            let nb = (ot >> 24) as usize;
            last = (ot & 0xffffff) as usize + raw_bits(c.raw, pos, nb);
            pos += nb;
        }
        let mut ml = (t >> 2) as usize;
        let r = if c.flat {
            last.max(ml.min(MAX_LEN)).min(d)
        } else {
            last.min(d)
        };
        if ml > MAX_LEN {
            ml += c.esc[gi] as usize;
            gi += 1;
            if r < MAX_LEN {
                st[3] = 1;
            }
        }
        if r >= ml {
            dst.copy_within(d - r..d - r + ml, d);
        } else {
            for j in 0..ml {
                dst[d + j] = dst[d - r + j];
            }
        }
        d += ml;
    }
    st[..3].copy_from_slice(&[d, l, last]);
}
