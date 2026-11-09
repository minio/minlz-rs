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

//! The NEON operation loops and stream stats of SIMD blocks: Go's `decode-simd_arm64.s` (the
//! per-operation loops and the stats) and `decode-simd-batch_arm64.s` (the batched loops,
//! written by `_generate/simdneon`), dispatched by [`Tier`](super::exec::Tier) as
//! `decode_arm64.go` does. ASIMD is part of ARMv8; little-endian only, as lanes and integer
//! loads share one byte order.
//!
//! Literals and matches are copied 32 bytes at a time, as a pair of Q registers, so an operation
//! writes up to [`SLACK`](super::SLACK) bytes past its end, and streams are read past their
//! values: the [memory rules](super::exec) of the vector loops. [`Loop`] holds Go's registers
//! and its methods are Go's macros and out-of-line paths, which [`cold`] keeps out of line.
//!
//! The loops rely on the memory rules only, not on the stream checks of the decoder, so they
//! stay in bounds for any stream bytes. Go's loops take the escape path of a literal length
//! of exactly 33 and the 33 branch of `raise` for a match length code of exactly 33; here
//! both take every value from 33 up, which is the same for checked streams, and the pattern
//! table has a row for every match length code a token byte can hold. Each operation then
//! uses the escapes [`exec_go`](super::exec::exec_go) uses, and ends no later than there.
//!
//! Other differences: the batched loops take Go's ADDS and CSEL from an `asm!` block, as LLVM
//! adds and compares with 0; the flat loops' `raise` clamps to d itself instead of going to
//! `clampcheck`; the stats sum with UADALP (see [`sums`]) and test the largest byte with UMAXV,
//! which Go's assembler lacks.

use core::arch::aarch64::*;

use super::OFF_TAB;
use super::exec::{LLStatsFn, OffStatsFn, TokStatsFn};

/// [`OFF_TAB`] laid out for the loops (Go `simdOffTabNEON`): the mask of the raw bits in the
/// low 32 bits, the number of raw bits in bits 32-39 and the base offset in the top 24 bits.
/// Symbol 0 decodes to offset 0 with no raw bits: the batched loops give it to repeats, which
/// keep the last offset. The per-operation loops ignore the symbol of a repeat.
static OFF_TAB_NEON: [u64; 256] = {
    let mut t = [0u64; 256];
    let mut s = 1;
    while s < 256 {
        let (v, nb) = (OFF_TAB[s] as u64, OFF_TAB[s] as u64 >> 24);
        t[s] = ((1 << nb) - 1) | nb << 32 | (v & 0xffffff) << 40;
        s += 1;
    }
    t
};

#[repr(align(64))]
struct PermTab([[u8; 32]; 64]);

/// TBL indexes, indexed by min(offset, 32) (Go `simdPermTabNEON`): byte i of row r is i % r,
/// and i in row 32. Row 0 is row 1: a delta reference at position 0 repeats `dst[0]`, which
/// is 0. Go has 33 rows; the rows above 32, as row 32, are only read for token bytes above 135.
static PERM_TAB: PermTab = {
    let mut t = [[0u8; 32]; 64];
    let mut r = 2;
    while r < 64 {
        let mut i = 0;
        while i < 32 {
            t[r][i] = if r < 32 { (i % r) as u8 } else { i as u8 };
            i += 1;
        }
        r += 1;
    }
    PermTab(t)
};

const ONES: u64 = 0x0101010101010101;

/// Returns v, which LLVM can't see through. With ONES in sight it turns f * ONES - f into a
/// second multiply and builds the constant again for each batch; POS it adds to in the next
/// operation, which keeps the table entry live.
#[inline(always)]
fn reg(mut v: u64) -> u64 {
    // SAFETY: the template is empty; it neither reads nor writes memory.
    unsafe {
        core::arch::asm!("/* {0} */", inout(reg) v, options(pure, nomem, nostack, preserves_flags))
    };
    v
}

/// Marks the paths Go writes out of line: rustc gives the branches into a block that calls a
/// cold function their weights, so LLVM lays out the operations as Go does, and the empty asm
/// keeps LLVM from turning small blocks into selects on the hot path.
#[cold]
#[inline(always)]
fn cold() {
    // SAFETY: the template is empty; it neither reads nor writes memory.
    unsafe { core::arch::asm!("", options(nomem, nostack, preserves_flags)) };
}

#[inline(always)]
unsafe fn ld32(p: *const u8) -> uint8x16x2_t {
    // SAFETY: the caller guarantees 32 readable bytes at p.
    unsafe { uint8x16x2_t(vld1q_u8(p), vld1q_u8(p.add(16))) }
}

#[inline(always)]
unsafe fn st32(p: *mut u8, v: uint8x16x2_t) {
    // SAFETY: the caller guarantees 32 writable bytes at p.
    unsafe {
        vst1q_u8(p, v.0);
        vst1q_u8(p.add(16), v.1);
    }
}

/// Repeats the first r bytes of v, r up to 63, with row r of [`PERM_TAB`].
#[inline(always)]
unsafe fn perm(v: uint8x16x2_t, r: usize) -> uint8x16x2_t {
    // SAFETY: the table has 64 rows.
    unsafe {
        let p = ld32(PERM_TAB.0.as_ptr().add(r).cast());
        uint8x16x2_t(vqtbl2q_u8(v, p.0), vqtbl2q_u8(v, p.1))
    }
}

/// The registers of an operation loop: Go's DST, LITS, RAW, ESCP, DP, LITP, LAST, POS and ST.
struct Loop<'a> {
    dst: *mut u8,
    lits: *const u8,
    raw: *const u8,
    esc: *const u8,
    dp: *mut u8,
    litp: *const u8,
    last: usize,
    pos: usize,
    st: &'a mut [usize; 4],
}

impl<'a> Loop<'a> {
    /// OPS_START, with the stream pointers left to the loop.
    #[inline(always)]
    unsafe fn new(
        dst: *mut u8,
        lits: *const u8,
        raw: *const u8,
        esc: *const u8,
        pos: usize,
        st: &'a mut [usize; 4],
    ) -> Self {
        // SAFETY: st[0] and st[1] are positions in dst and lits.
        let (dp, litp) = unsafe { (dst.add(st[0]), lits.add(st[1])) };
        let last = st[2];
        Loop {
            dst,
            lits,
            raw,
            esc,
            dp,
            litp,
            last,
            pos,
            st,
        }
    }

    /// Writes the state back (Go `done`).
    #[inline(always)]
    fn done(self) {
        self.st[0] = self.dp.addr() - self.dst.addr();
        self.st[1] = self.litp.addr() - self.lits.addr();
        self.st[2] = self.last;
    }

    /// Stores the first 32 literals of an operation. With DELTA they are added to their
    /// references, from r = min(LAST, d) back before the operation, or with FLAT
    /// min(max(LAST, 32), d), which never repeats as d is at least 32.
    #[inline(always)]
    unsafe fn lits<const DELTA: bool, const FLAT: bool>(&mut self) {
        // SAFETY: rule 2 covers the literals, rule 1 the store; r <= d keeps the reference in
        // dst, and its 32 bytes end at most at dp + 32.
        unsafe {
            let mut v = ld32(self.litp);
            if DELTA {
                let d = self.dp.addr() - self.dst.addr();
                let rv = if FLAT {
                    let r = if self.last >= 32 { self.last } else { 32 };
                    ld32(self.dp.sub(if r > d { d } else { r }))
                } else {
                    let r = if self.last < d { self.last } else { d };
                    let rv = ld32(self.dp.sub(r));
                    if r < 32 {
                        // dpattern
                        cold();
                        perm(rv, r)
                    } else {
                        rv
                    }
                };
                v = uint8x16x2_t(vaddq_u8(v.0, rv.0), vaddq_u8(v.1, rv.1));
            }
            st32(self.dp, v);
        }
    }

    /// Moves past len literals, whose first 32 are stored; from 33 on the length is 33 plus
    /// an escape, and litlong copies the rest.
    #[inline(always)]
    unsafe fn lit_len(&mut self, mut len: usize) {
        // SAFETY: the literals and their 32 bytes of slack are in lits and dst (rules 1, 2).
        unsafe {
            if len >= 33 {
                // litlong: 32-byte steps after the first, (len-32+31)>>5 = (escape+32)>>5.
                cold();
                let e = usize::from(*self.esc);
                self.esc = self.esc.add(1);
                len = 33 + e;
                let (mut s, mut d) = (self.litp.add(32), self.dp.add(32));
                for _ in 0..(e + 32) >> 5 {
                    st32(d, ld32(s));
                    (s, d) = (s.add(32), d.add(32));
                }
            }
            self.dp = self.dp.add(len);
            self.litp = self.litp.add(len);
        }
    }

    /// Returns the raw bits at POS of the symbol with table entry e.
    #[inline(always)]
    unsafe fn raw_bits(&self, e: u64) -> u64 {
        // SAFETY: rule 3: raw is readable 8 bytes from byte pos >> 3.
        let r = unsafe { self.raw.add(self.pos >> 3).cast::<u64>().read_unaligned() };
        u64::from((r >> (self.pos & 7)) as u32 & e as u32)
    }

    /// Reports whether LAST reaches before dst (Go's SUBS, CCMP and BLO to clamp).
    #[inline(always)]
    fn clamp(&self) -> bool {
        let (s, borrow) = self.dp.addr().overflowing_sub(self.last);
        borrow | (s < self.dst.addr())
    }

    /// Copies the match with length code ml from r = min(LAST, d) back, r below ml selecting
    /// a pattern row (Go's clamp, pattern and check), or with FLAT from
    /// r = min(max(LAST, min(ml, 32)), d) back, which never overlaps (raise, clampcheck, clamp
    /// and copy), then moves past it (long, next). Codes from 33 on are escapes: Go tests the
    /// token, the batch takes the code alone out of the token word.
    #[inline(always)]
    unsafe fn copy<const FLAT: bool>(&mut self, ml: usize) {
        // SAFETY: r <= d keeps the source in dst, and its 32 bytes end at most at dp + 32;
        // rule 1 covers the stores. A raised r is at most 32, and FLAT operations start at 32
        // or later.
        unsafe {
            let src: *const u8;
            if FLAT {
                src = if self.last < ml {
                    // raise: to min(ml, 32), which d covers; an escape keeps a LAST from 32 on,
                    // which Go's raise sends to clampcheck.
                    cold();
                    let d = self.dp.addr() - self.dst.addr();
                    self.dp.sub(self.last.max(ml.min(32)).min(d))
                } else if self.clamp() {
                    // clamp
                    cold();
                    self.dst
                } else {
                    self.dp.sub(self.last)
                };
                st32(self.dp, ld32(src));
            } else if self.clamp() {
                // clamp
                cold();
                src = self.dst;
                let d = self.dp.addr() - self.dst.addr();
                let v = ld32(src);
                st32(self.dp, if d < ml { perm(v, d) } else { v });
            } else {
                src = self.dp.sub(self.last);
                if self.last < ml {
                    // pattern
                    cold();
                    st32(self.dp, perm(ld32(src), self.last));
                } else {
                    st32(self.dp, ld32(src));
                }
            }
            let mut ml = ml;
            if ml >= 33 {
                // long
                cold();
                if !FLAT && self.dp.addr() - src.addr() < 32 {
                    self.st[3] = 1;
                }
                let e = usize::from(*self.esc);
                self.esc = self.esc.add(1);
                ml = 33 + e;
                let (mut s, mut d) = (src.add(32), self.dp.add(32));
                for _ in 0..(e + 32) >> 5 {
                    st32(d, ld32(s));
                    (s, d) = (s.add(32), d.add(32));
                }
            }
            self.dp = self.dp.add(ml);
        }
    }

    /// Runs an operation of a batch with match length code ml, literal length len (0 without
    /// literals) and offset symbol sym (0 for a repeat).
    #[inline(always)]
    unsafe fn op<const DELTA: bool, const FLAT: bool>(
        &mut self,
        ml: usize,
        len: usize,
        sym: usize,
    ) {
        // SAFETY: as for the loops.
        unsafe {
            self.lits::<DELTA, FLAT>();
            self.lit_len(len);
            let e = OFF_TAB_NEON[sym];
            let r = self.raw_bits(e);
            self.pos = reg((self.pos + (e >> 32) as u8 as usize) as u64) as usize;
            // Only symbol 0 decodes to offset 0.
            // SAFETY: the instructions only use registers.
            core::arch::asm!(
                "adds {r}, {r}, {e}, lsr #40",
                "csel {last}, {r}, {last}, ne",
                r = inout(reg) r => _,
                e = in(reg) e,
                last = inout(reg) self.last,
                options(pure, nomem, nostack),
            );
            self.copy::<FLAT>(ml);
        }
    }
}

/// The per-operation loops: OPS_START and the loop of `simdExecNEON` and its variants.
///
/// # Safety
///
/// As for [`ExecFlatFn`](super::exec::ExecFlatFn); pos is 0 without FLAT.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn ops<const DELTA: bool, const FLAT: bool>(
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
) {
    // SAFETY: rule 3 covers the stream reads: ll and offc are read at the next value of each
    // operation, used or not.
    unsafe {
        let mut s = Loop::new(dst, lits, raw, esc, pos, st);
        let (mut tokp, end, mut llp, mut offp) = (tok, tok.add(n), ll, offc);
        while tokp < end {
            let t = usize::from(*tokp);
            tokp = tokp.add(1);
            s.lits::<DELTA, FLAT>();
            let (lit, len) = (t & 2 != 0, usize::from(*llp));
            llp = llp.add(usize::from(lit));
            s.lit_len(if lit { len } else { 0 });
            let e = OFF_TAB_NEON[usize::from(*offp)];
            let o = (e >> 40) as usize + s.raw_bits(e) as usize;
            let new = t & 1 == 0;
            if new {
                (s.last, s.pos) = (o, s.pos + (e >> 32) as u8 as usize);
            }
            offp = offp.add(usize::from(new));
            s.copy::<FLAT>(t >> 2);
        }
        s.done();
    }
}

/// Expands the values at p of the operations whose bytes of f are 1 into one byte per
/// operation, 0 for the others, and moves p past them. c = f * ONES counts them in bytes 0 to
/// i, and c - f gives the index of each value. Bytes without a value get index 0x80, for which
/// TBL gives 0.
///
/// # Safety
///
/// p is readable for 8 bytes.
#[inline(always)]
unsafe fn expand(f: u64, ones: u64, p: &mut *const u8) -> uint8x8_t {
    let c = f.wrapping_mul(ones);
    let idx = c.wrapping_sub(f) | (f ^ ones) << 7;
    // SAFETY: the caller guarantees the load; the indexes reach byte 7 at most.
    unsafe {
        let v = vcombine_u8(vld1_u8(*p), vdup_n_u8(0));
        *p = p.add((c >> 56) as usize);
        vqtbl1_u8(v, vcreate_u8(idx))
    }
}

/// The 8 bytes of v as a word.
#[inline(always)]
unsafe fn word(v: uint8x8_t) -> u64 {
    // SAFETY: the CPU has NEON.
    unsafe { vget_lane_u64::<0>(vreinterpret_u64_u8(v)) }
}

/// Expands the literal lengths and offset symbols of the 8 operations at tokp (Go `expand`).
/// A repeat has no offset symbol, and gets symbol 0.
///
/// # Safety
///
/// tokp, ll and offc are readable for 8 bytes.
#[inline(always)]
unsafe fn expand2(
    tokp: *const u8,
    ones: u64,
    ll: &mut *const u8,
    offc: &mut *const u8,
) -> (uint8x8_t, uint8x8_t) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let t = tokp.cast::<u64>().read_unaligned();
        (
            expand(t >> 1 & ones, ones, ll),
            expand(!t & ones, ones, offc),
        )
    }
}

/// Runs $body for $k = 0..8 as constants: a batch is unrolled, as Go's generator writes it.
macro_rules! unroll8 {
    ($k:ident => $body:expr) => {{
        let $k = 0;
        $body;
        let $k = 1;
        $body;
        let $k = 2;
        $body;
        let $k = 3;
        $body;
        let $k = 4;
        $body;
        let $k = 5;
        $body;
        let $k = 6;
        $body;
        let $k = 7;
        $body;
    }};
}

/// The batched loops (`_generate/simdneon`): each batch of 8 operations takes the words
/// expanded before it, and expands the next batch while it runs, if there is one. The words
/// of a partial last batch hold values past n, which are not used.
///
/// # Safety
///
/// As for [`ExecFlatFn`](super::exec::ExecFlatFn); pos is 0 without FLAT.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn batch<const DELTA: bool, const FLAT: bool>(
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
) {
    // SAFETY: rule 3: tok is readable 7 tokens past the last operation, so 8 bytes from any
    // operation; ll and offc 8 bytes past the values the operations before a batch use.
    unsafe {
        let mut s = Loop::new(dst, lits, raw, esc, pos, st);
        let (mut tokp, mut llp, mut offp, mut left) = (tok, ll, offc, n);
        if left == 0 {
            return s.done();
        }
        let ones = reg(ONES);
        let (mut lwn, mut swn) = expand2(tokp, ones, &mut llp, &mut offp);
        // left is the number of operations left, from tokp on.
        while left >= 8 {
            let tw = tokp.cast::<u64>().read_unaligned();
            tokp = tokp.add(8);
            let (lw, sw) = (word(lwn), word(swn));
            left -= 8;
            if left != 0 {
                (lwn, swn) = expand2(tokp, ones, &mut llp, &mut offp);
            }
            let b = |w: u64, k: usize| (w >> (8 * k)) as u8 as usize;
            unroll8!(k => s.op::<DELTA, FLAT>(b(tw, k) >> 2, b(lw, k), b(sw, k)));
        }
        if left != 0 {
            // The tail takes byte 0 and shifts the words down.
            let (mut tw, mut lw, mut sw) =
                (tokp.cast::<u64>().read_unaligned(), word(lwn), word(swn));
            loop {
                s.op::<DELTA, FLAT>(tw as u8 as usize >> 2, lw as u8 as usize, sw as u8 as usize);
                (tw, lw, sw) = (tw >> 8, lw >> 8, sw >> 8);
                left -= 1;
                if left == 0 {
                    break;
                }
            }
        }
        s.done();
    }
}

/// Writes an [`ExecFn`](super::exec::ExecFn) loop of the NEON tier.
macro_rules! exec_fn {
    ($(#[$doc:meta])* $name:ident = $body:ident::<$delta:literal>) => {
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// As for [`ExecFn`](super::exec::ExecFn).
        #[target_feature(enable = "neon")]
        #[allow(clippy::too_many_arguments)]
        pub(super) unsafe fn $name(
            dst: *mut u8,
            lits: *const u8,
            tok: *const u8,
            ll: *const u8,
            offc: *const u8,
            raw: *const u8,
            esc: *const u8,
            n: usize,
            st: &mut [usize; 4],
        ) {
            // SAFETY: as the caller guarantees.
            unsafe { $body::<$delta, false>(dst, lits, tok, ll, offc, raw, esc, n, 0, st) }
        }
    };
}

/// Writes an [`ExecFlatFn`](super::exec::ExecFlatFn) loop of the NEON tier.
macro_rules! exec_flat_fn {
    ($(#[$doc:meta])* $name:ident = $body:ident::<$delta:literal>) => {
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// As for [`ExecFlatFn`](super::exec::ExecFlatFn).
        #[target_feature(enable = "neon")]
        #[allow(clippy::too_many_arguments)]
        pub(super) unsafe fn $name(
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
        ) {
            // SAFETY: as the caller guarantees.
            unsafe { $body::<$delta, true>(dst, lits, tok, ll, offc, raw, esc, n, pos, st) }
        }
    };
}

exec_fn!(
    /// Runs the first n operations of a chunk (Go `simdExecNEON`). st holds the output
    /// position, the literal position, the repeat offset and an error flag.
    exec_neon = ops::<false>
);
exec_fn!(
    /// [`exec_neon`] for chunks with delta literals (Go `simdExecDeltaNEON`).
    exec_delta_neon = ops::<true>
);
exec_flat_fn!(
    /// [`exec_neon`] for chunks with the no-overlap flag, from bit pos of raw
    /// (Go `simdExecFlatNEON`). Their matches must start at position 32 or later.
    exec_flat_neon = ops::<false>
);
exec_flat_fn!(
    /// [`exec_flat_neon`] for chunks with delta literals (Go `simdExecFlatDeltaNEON`).
    exec_flat_delta_neon = ops::<true>
);
exec_fn!(
    /// [`exec_neon`] with the batched loop (Go `simdExecBatchNEON`), which expands the
    /// literal lengths and offset symbols of 8 operations at a time.
    exec_batch_neon = batch::<false>
);
exec_fn!(
    /// [`exec_delta_neon`] with the batched loop (Go `simdExecBatchDeltaNEON`).
    exec_batch_delta_neon = batch::<true>
);
exec_flat_fn!(
    /// [`exec_flat_neon`] with the batched loop (Go `simdExecBatchFlatNEON`).
    exec_batch_flat_neon = batch::<false>
);
exec_flat_fn!(
    /// [`exec_flat_delta_neon`] with the batched loop (Go `simdExecBatchFlatDeltaNEON`).
    exec_batch_flat_delta_neon = batch::<true>
);

// The stream checks of chunks take 16 bytes at a time. They keep the largest byte, or
// byte - 1, in a vector, and sum bytes into u16 lanes with UADALP. Go sums each vector with
// UADDLV into a D lane, which LLVM moves to a general register and back.

const _: (TokStatsFn, LLStatsFn, OffStatsFn) = (tok_stats_neon, ll_stats_neon, off_stats_neon);

/// Returns the largest byte of the first vector f makes of each 16 bytes of b, and the sums of
/// the others. b.len() is a multiple of 16.
#[inline(always)]
unsafe fn sums<const N: usize>(
    b: &[u8],
    f: impl Fn(uint8x16_t) -> (uint8x16_t, [uint8x16_t; N]),
) -> (u8, [usize; N]) {
    // SAFETY: the CPU has NEON; the loads read b.
    unsafe {
        let mut mx = vdupq_n_u8(0);
        let mut tot = [vdupq_n_u32(0); N];
        // A u16 lane takes 128 pair sums up to 510.
        for run in b.chunks(16 * 128) {
            let mut acc = [vdupq_n_u16(0); N];
            for c in run.as_chunks::<16>().0 {
                let (m, v) = f(vld1q_u8(c.as_ptr()));
                mx = vmaxq_u8(m, mx);
                for (a, v) in acc.iter_mut().zip(v) {
                    *a = vpadalq_u8(*a, v);
                }
            }
            for (t, a) in tot.iter_mut().zip(acc) {
                *t = vpadalq_u16(*t, a);
            }
        }
        (vmaxvq_u8(mx), tot.map(|t| vaddlvq_u32(t) as usize))
    }
}

/// Returns the token stats of b, whose length is a multiple of 16 (Go
/// `simdTokStatsNEON`): the summed match length codes, the operations with literals, the
/// repeats and the escaped matches, and whether every token is at most 135.
///
/// # Safety
///
/// The CPU has NEON.
#[target_feature(enable = "neon")]
pub(super) unsafe fn tok_stats_neon(b: &[u8]) -> (usize, usize, usize, usize, bool) {
    // SAFETY: the CPU has NEON.
    unsafe {
        let (c1, c2, c4) = (vdupq_n_u8(1), vdupq_n_u8(2), vdupq_n_u8(4));
        let (mx, [out, lit, rep, esc]) = sums(b, |v| {
            // Match length code 33 has bits 7 and 2.
            let e = vandq_u8(vandq_u8(v, vshrq_n_u8::<5>(v)), c4);
            let (l, r) = (vandq_u8(v, c2), vandq_u8(v, c1));
            (v, [vshrq_n_u8::<2>(v), l, r, e])
        });
        (out, lit >> 1, rep, esc >> 2, mx <= 135)
    }
}

/// Returns the sum of the bytes of b, whose length is a multiple of 16, and how many
/// are 33 (Go `simdLLStatsNEON`). ok is false if a byte is outside 1-33.
///
/// # Safety
///
/// The CPU has NEON.
#[target_feature(enable = "neon")]
pub(super) unsafe fn ll_stats_neon(b: &[u8]) -> (usize, usize, bool) {
    // SAFETY: the CPU has NEON.
    unsafe {
        let (c1, c33) = (vdupq_n_u8(1), vdupq_n_u8(33));
        // 0 wraps to 255, so 1-33 is at most 32.
        let (mx, [sum, esc]) = sums(b, |v| {
            (vsubq_u8(v, c1), [v, vandq_u8(vceqq_u8(v, c33), c1)])
        });
        (sum, esc, mx <= 32)
    }
}

/// Returns the raw bits of the offset symbols of b, whose length is a multiple of 16
/// (Go `simdOffStatsNEON`), and 0, as [`OffStatsFn`](super::exec::OffStatsFn) has it.
/// ok is false if a byte is outside 1-175.
///
/// # Safety
///
/// The CPU has NEON.
#[target_feature(enable = "neon")]
pub(super) unsafe fn off_stats_neon(b: &[u8]) -> (usize, usize, bool) {
    // SAFETY: the CPU has NEON.
    unsafe {
        let c1 = vdupq_n_u8(1);
        // The raw bits of symbol s are max(s>>3, 1) - 1.
        let (mx, [sum]) = sums(b, |v| (vsubq_u8(v, c1), [vmaxq_u8(vshrq_n_u8::<3>(v), c1)]));
        (sum - b.len(), 0, mx <= 174)
    }
}

#[cfg(test)]
mod tests {
    use super::super::decode::{Chunk, ll_stats, off_stats, tok_stats};
    use super::super::decode_tests::Rng;
    use super::super::exec::{Tier, exec_go};
    use super::super::{MAX_LEN, SLACK};
    use super::*;
    use crate::cpu;
    use crate::guard::Buffer;

    /// The tiers Go's `simdTiers` runs on arm64: go, neon and neon-batch, and the batched
    /// loops by default (Go `simdBatchNEON`).
    #[test]
    fn simd_tiers_neon() {
        let _d = cpu::Disabled::new(0);
        assert_eq!(Tier::tiers(), [Tier::Go, Tier::Neon, Tier::NeonBatch]);
        assert_eq!(Tier::best(), Tier::NeonBatch);
        drop(_d);
        let _d = cpu::Disabled::new(cpu::NEON);
        assert_eq!((Tier::tiers(), Tier::best()), (vec![Tier::Go], Tier::Go));
    }

    #[test]
    fn simd_tables_neon() {
        assert_eq!(OFF_TAB_NEON[0], 0);
        for (s, &e) in OFF_TAB_NEON.iter().enumerate().skip(1) {
            let nb = OFF_TAB[s] >> 24;
            assert_eq!(
                (e >> 40, (e >> 32) as u8 as u32, e as u32),
                (u64::from(OFF_TAB[s] & 0xffffff), nb, (1 << nb) - 1),
                "symbol {s}"
            );
        }
        for (r, row) in PERM_TAB.0.iter().enumerate() {
            for (i, &v) in row.iter().enumerate() {
                assert_eq!(v as usize, if r < 32 { i % r.max(1) } else { i }, "row {r}");
            }
        }
    }

    /// Returns n bytes of g against a guard page, starting with b.
    fn at<'g>(g: &'g mut Buffer, b: &[u8], n: usize, at_end: bool) -> &'g mut [u8] {
        let s = g.slice(n, at_end);
        s.copy_from_slice(&b[..n]);
        s
    }

    /// The sums are exact over many runs of 128 vectors, at the largest bytes. The lengths
    /// are multiples of 16.
    #[test]
    fn simd_stats_neon_long() {
        let mut rng = Rng(10);
        for n in [2032, 2048, 2064, 4112, 1 << 16, 70000, 1 << 17] {
            for fill in [0xff, 135, 33, 175, 1, 0] {
                let mut b = vec![fill; n];
                if fill == 0 {
                    rng.read(&mut b);
                }
                // SAFETY: the module is only built for aarch64, where ASIMD is baseline.
                let (t, l, o) =
                    unsafe { (tok_stats_neon(&b), ll_stats_neon(&b), off_stats_neon(&b)) };
                let b = &b[..];
                let g = (
                    tok_stats(Tier::Go, b),
                    ll_stats(Tier::Go, b),
                    off_stats(Tier::Go, b),
                );
                let (sum, codes) = (
                    b.iter().map(|&v| usize::from(v)).sum(),
                    b.iter().map(|&v| usize::from(v >> 2)).sum(),
                );
                assert_eq!((l.0, t.0), (sum, codes), "length {n} fill {fill}");
                assert_eq!(
                    (t.4, l.2, o.2),
                    (g.0.4, g.1.2, g.2.1),
                    "length {n} fill {fill}"
                );
                if g.0.4 {
                    assert_eq!(
                        (t.1, b.len() - t.2, t.3),
                        (g.0.1, g.0.2, g.0.3),
                        "length {n}"
                    );
                }
                if g.1.2 {
                    assert_eq!(l.1, g.1.1, "length {n}");
                }
                if g.2.1 {
                    assert_eq!(o.0, g.2.0, "length {n}");
                }
            }
        }
    }

    /// The loops keep to the memory rules for any stream bytes, read as exec_go reads them
    /// (see the module docs): the buffers end at guard pages where exec_go's reads and writes
    /// end, and dst starts at one every other run.
    #[test]
    fn simd_exec_neon_any_bytes() {
        let mut rng = Rng(9);
        let mut g: [Buffer; 7] = core::array::from_fn(|_| Buffer::new(1 << 20).unwrap());
        let [g0, g1, g2, g3, g4, g5, g6] = &mut g;
        for i in 0..20000 {
            let (delta, flat, tier) = (
                i & 1 != 0,
                i & 2 != 0,
                [Tier::Neon, Tier::NeonBatch][i >> 2 & 1],
            );
            let n = rng.intn(40);
            let mut v = [
                vec![0u8; n + 7],
                vec![0; 1000],
                vec![0; 1000],
                vec![0; 1000],
            ];
            for s in &mut v {
                rng.read(s);
            }
            // Mostly valid tokens, so operations run past the first escapes.
            if i & 8 != 0 {
                v[0].iter_mut().for_each(|t| *t %= 136);
                v[1].iter_mut().for_each(|l| *l %= 40);
            }
            let [tok, ll, offc, esc] = &v;
            let (d0, l0) = (
                rng.intn(100) + if flat { MAX_LEN } else { 0 },
                rng.intn(100),
            );
            let (mut d, mut l, mut li, mut oi, mut gi, mut bits) = (d0, l0, 0, 0, 0, 0);
            for &t in &tok[..n] {
                let t = usize::from(t);
                if t & 2 != 0 {
                    let mut n = usize::from(ll[li]);
                    li += 1;
                    if n > MAX_LEN {
                        n += usize::from(esc[gi]);
                        gi += 1;
                    }
                    (d, l) = (d + n, l + n);
                }
                if t & 1 == 0 {
                    bits += (OFF_TAB[usize::from(offc[oi])] >> 24) as usize;
                    oi += 1;
                }
                let mut ml = t >> 2;
                if ml > MAX_LEN {
                    ml += usize::from(esc[gi]);
                    gi += 1;
                }
                d += ml;
            }
            let mut raw = vec![0; bits / 8 + 8];
            rng.read(&mut raw);
            let (mut lv, mut dv) = (vec![0; l + SLACK], vec![0; d + SLACK]);
            rng.read(&mut lv);
            rng.read(&mut dv);
            let c = Chunk {
                tok: at(g0, tok, n + 7, true),
                ll: at(g1, ll, li + 8, true),
                offc: at(g2, offc, oi + 8, true),
                esc: at(g3, esc, gi, true),
                raw: at(g4, &raw, raw.len(), true),
                n_tok: n,
                n_ll: li,
                n_off: oi,
                n_esc: gi,
                bits,
                delta,
                flat,
            };
            let (dst, lits) = (
                at(g5, &dv, dv.len(), i & 16 != 0),
                at(g6, &lv, lv.len(), true),
            );
            let before = dst[..d0].to_vec();
            let mut st = [d0, l0, 1 + rng.intn(1 << 20), 0];
            let mut sg = st;
            if flat {
                // SAFETY: NEON is baseline on aarch64, the counts above are what the tokens
                // read, and dst and lits end SLACK bytes past what exec_go uses.
                unsafe { tier.exec_flat(dst, lits, &c, 0, n, [0; 4], &mut st) };
            } else {
                // SAFETY: as above.
                unsafe { tier.exec(dst, lits, &c, n, &mut st) };
                exec_go(dst, lits, &c, n, &mut sg);
            }
            assert!(
                st[0] <= d && st[1] <= l && dst[..d0] == before[..],
                "{tier:?} case {i} (delta {delta}, flat {flat}): state {st:?}, exec_go ends at {d}, {l}"
            );
        }
    }
}
