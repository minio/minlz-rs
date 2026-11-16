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

//! Ports of Go's `exec_test.go` and the tier-agnostic part of `exec_vec_test.go`.

use core::mem::{offset_of, size_of};

use super::*;
use crate::pivco::kernel::{
    ARENA_SLACK, FLAG_DST, OP_CC, OP_COPY, OP_CV, OP_NAT, OP_V128, OP_V512, OP_VV, Op, Program,
    R2S_LEN, vert_split,
};

/// Test-only node kind: any flat region.
const OP_FLAT: u8 = 255;

/// A Program's buffers, guarded. Arena and Dst start with junk, so kernels cannot rely on
/// zeroed memory.
#[derive(Clone)]
struct Env {
    ops: Vec<Op>,
    /// The block, followed by GUARD_LEN guard bytes the program does not see.
    input: Vec<u8>,
    in_len: usize,
    arena: Vec<u8>,
    /// Dst, followed by GUARD_LEN guard bytes the program does not see.
    dst_all: Vec<u8>,
    dst_len: usize,
    r2s: [u8; R2S_LEN],
    nil_r2s: bool,
}

impl Env {
    fn new(rng: &mut Pcg, arena_len: usize, dst_len: usize, input: &[u8]) -> Env {
        let arena = rand_vec(rng, arena_len + ARENA_SLACK);
        let dst_all = guarded(&rand_vec(rng, dst_len));
        let mut r2s = [0u8; R2S_LEN];
        rand_bytes(rng, &mut r2s);
        Env {
            ops: Vec::new(),
            input: guarded(input),
            in_len: input.len(),
            arena,
            dst_all,
            dst_len,
            r2s,
            nil_r2s: false,
        }
    }

    fn program(&mut self) -> Program<'_> {
        Program {
            ops: &self.ops,
            input: &self.input[..self.in_len],
            arena: &mut self.arena,
            dst: &mut self.dst_all[..self.dst_len],
            r2s: (!self.nil_r2s).then_some(&self.r2s),
        }
    }

    fn dst(&self) -> &[u8] {
        &self.dst_all[..self.dst_len]
    }

    /// Runs exec and checks that nothing changed except the outputs of ops[..upto] (the ops
    /// allowed to write).
    fn exec(&mut self, tier: Tier, upto: usize) -> Result<(), usize> {
        let (mut arena, mut dst) = (self.arena.clone(), self.dst_all.clone());
        let (input, r2s) = (self.input.clone(), self.r2s);
        let ret = tier.exec(&mut self.program());
        for op in &self.ops[..upto] {
            let (to, from) = if op.flags & FLAG_DST != 0 {
                (&mut dst, &self.dst_all)
            } else {
                (&mut arena, &self.arena)
            };
            let lo = (op.out as usize).min(from.len());
            let hi = (op.out as usize + op.k as usize).min(from.len());
            to[lo..hi].copy_from_slice(&from[lo..hi]);
        }
        assert!(
            arena == self.arena && dst == self.dst_all && input == self.input && r2s == self.r2s,
            "{tier}: exec wrote outside its outputs (ret {ret:?})"
        );
        ret
    }

    fn out(&self, op: &Op) -> &[u8] {
        let b = if op.flags & FLAG_DST != 0 {
            self.dst()
        } else {
            &self.arena[..]
        };
        &b[op.out as usize..][..op.k as usize]
    }
}

#[test]
fn exec_merge() {
    for tier in Tier::tiers() {
        let mut rng = new_rng();
        for k in test_sizes() {
            // Under Miri, the flat sizes add nothing to the merges.
            if cfg!(miri) && k > 200 {
                continue;
            }
            for density in [0.0, 1.0, 0.5, 0.05, 0.95] {
                test_merge(tier, &mut rng, k, density);
            }
        }
    }
}

fn test_merge(tier: Tier, rng: &mut Pcg, k: usize, density: f64) {
    let (bm, kr) = rand_bits(rng, k, density);
    let kl = k - kr;
    let (lv, rv) = (rand_vec(rng, kl), rand_vec(rng, kr));
    let (c0, c1) = (rng.uint32() as u8, rng.uint32() as u8);
    let (want_vv, _) = ref_merge(&bm, k, &lv, &rv);
    let (want_cv, _) = ref_merge(&bm, k, &vec![c0; k], &rv);
    let (want_cc, _) = ref_merge(&bm, k, &vec![c0; k], &vec![c1; k]);

    // The bitmap sits at the end of the input behind a junk prefix: reads must be exact.
    const PRE: u32 = 3;
    const OFF: u32 = 5;
    let mut src = rand_vec(rng, PRE as usize);
    src.extend_from_slice(&bm);
    let (ku, kru, klu) = (k as u32, kr as u32, kl as u32);
    let op = |kind, flags, p, q, out, l, r| Op {
        kind,
        flags,
        p,
        q,
        out,
        l,
        r,
        ..Op::default()
    };
    // Name, op, arena placements of the children, output.
    type Case<'a> = (&'a str, Op, Vec<(u32, &'a [u8])>, &'a [u8]);
    let cases: [Case; 9] = [
        (
            "vv",
            op(OP_VV, 0, 0, 0, OFF, OFF + ku, OFF + 2 * ku),
            vec![(OFF + ku, &lv), (OFF + 2 * ku, &rv)],
            &want_vv,
        ),
        (
            "vv/inplace-left",
            op(OP_VV, 0, 0, 0, OFF, OFF + kru, OFF + 2 * ku),
            vec![(OFF + kru, &lv), (OFF + 2 * ku, &rv)],
            &want_vv,
        ),
        (
            "vv/inplace-right",
            op(OP_VV, 0, 0, 0, OFF, OFF + 2 * ku, OFF + klu),
            vec![(OFF + 2 * ku, &lv), (OFF + klu, &rv)],
            &want_vv,
        ),
        (
            "vv/dst",
            op(OP_VV, FLAG_DST, 0, 0, 0, OFF, OFF + ku),
            vec![(OFF, &lv), (OFF + ku, &rv)],
            &want_vv,
        ),
        (
            "cv",
            op(OP_CV, 0, c0, 0, OFF, 0, OFF + ku),
            vec![(OFF + ku, &rv)],
            &want_cv,
        ),
        (
            "cv/inplace",
            op(OP_CV, 0, c0, 0, OFF, 0, OFF + klu),
            vec![(OFF + klu, &rv)],
            &want_cv,
        ),
        (
            "cv/dst",
            op(OP_CV, FLAG_DST, c0, 0, 0, 0, OFF),
            vec![(OFF, &rv)],
            &want_cv,
        ),
        ("cc", op(OP_CC, 0, c0, c1, OFF, 0, 0), vec![], &want_cc),
        (
            "cc/dst",
            op(OP_CC, FLAG_DST, c0, c1, 0, 0, 0),
            vec![],
            &want_cc,
        ),
    ];
    for (name, cop, places, want) in &cases {
        for kr_delta in [0isize, 1, -1] {
            let krd = kr as isize + kr_delta;
            if kr_delta != 0 && (cop.kind == OP_CC || krd < 0 || krd > k as isize) {
                continue;
            }
            let o = Op {
                k: ku,
                kr: krd as u32,
                bm: PRE,
                ..*cop
            };
            let mut e = Env::new(rng, OFF as usize + 3 * k, k, &src);
            for &(at, v) in places {
                e.arena[at as usize..][..v.len()].copy_from_slice(v);
            }
            e.ops = vec![o];
            let want_ret = if kr_delta != 0 { Err(0) } else { Ok(()) };
            let ret = e.exec(tier, 1);
            assert_eq!(
                ret, want_ret,
                "{tier} {name} k={k} kr={kr} (op KR {})",
                o.kr
            );
            if kr_delta == 0 {
                assert!(
                    e.out(&o) == *want,
                    "{tier} {name} k={k} kr={kr} density={density}: wrong output"
                );
            }
        }
    }
}

/// Splits an n-code flat region like the planner does.
fn flat_ops(n: usize, d: u8, p: u8, flags: u8, mut bm: u32, mut out: u32) -> Vec<Op> {
    if d == 8 {
        return vec![Op {
            kind: OP_COPY,
            flags,
            d: 8,
            k: n as u32,
            bm,
            out,
            ..Op::default()
        }];
    }
    let (n512, n128) = vert_split(n, d);
    let mut ops = Vec::new();
    for (kind, k) in [(OP_V512, n512), (OP_V128, n128), (OP_NAT, n - n512 - n128)] {
        if k == 0 && (n != 0 || kind != OP_NAT) {
            continue;
        }
        ops.push(Op {
            kind,
            flags,
            d,
            p,
            k: k as u32,
            bm,
            out,
            ..Op::default()
        });
        bm += (k * usize::from(d) / 8) as u32;
        out += k as u32;
    }
    ops
}

#[test]
fn exec_flat() {
    for tier in Tier::tiers() {
        let mut rng = new_rng();
        for n in test_sizes() {
            for d in 2..=8u8 {
                // Random regions: padding bits are the planner's business.
                let region = rand_vec(&mut rng, (n * usize::from(d)).div_ceil(8));
                let codes = ref_unpack(&region, n, usize::from(d));
                const PRE: u32 = 2;
                const OFF: u32 = 7;
                let mut src = rand_vec(&mut rng, PRE as usize);
                src.extend_from_slice(&region);
                let flags = (n % 2) as u8;
                let mut e = Env::new(&mut rng, OFF as usize + n, n, &src);
                let p = if d < 8 {
                    rng.int_n(257 - (1 << d)) as u8
                } else {
                    0
                };
                let out = if flags & FLAG_DST != 0 { 0 } else { OFF };
                e.ops = flat_ops(n, d, p, flags, PRE, out);
                let want: Vec<u8> = if d < 8 {
                    codes
                        .iter()
                        .map(|&c| e.r2s[usize::from(p) + usize::from(c)])
                        .collect()
                } else {
                    region
                };
                let nops = e.ops.len();
                assert_eq!(e.exec(tier, nops), Ok(()), "{tier} n={n} d={d}");
                let got = if flags & FLAG_DST != 0 {
                    e.dst()
                } else {
                    &e.arena[OFF as usize..][..n]
                };
                assert!(
                    got == want,
                    "{tier} n={n} d={d} flags={flags}: wrong output"
                );
            }
            // Natural D=8 through R2S (not emitted by the planner, but a valid op).
            let region = rand_vec(&mut rng, n);
            let mut e = Env::new(&mut rng, 0, n, &region);
            let p = rng.int_n(R2S_LEN - 256 + 1) as u8;
            e.ops = vec![Op {
                kind: OP_NAT,
                flags: FLAG_DST,
                d: 8,
                p,
                k: n as u32,
                ..Op::default()
            }];
            assert_eq!(e.exec(tier, 1), Ok(()), "{tier} nat8 n={n}");
            for (i, &c) in region.iter().enumerate() {
                assert_eq!(
                    e.dst()[i],
                    e.r2s[usize::from(p) + usize::from(c)],
                    "{tier} nat8 n={n}: wrong output at {i}"
                );
            }
        }
    }
}

/// A random decode tree whose ops are placed exactly like the upstream decoder
/// (`pivco_decode`, `codec_decode_subtree`), which exercises the nested in-place and
/// ping-pong aliasing.
#[derive(Default)]
struct TNode {
    kind: u8,
    d: u8,
    p: u8,
    q: u8,
    k: usize,
    kr: usize,
    bm: u32,
    l: Option<Box<TNode>>,
    r: Option<Box<TNode>>,
}

struct TreeGen<'a> {
    rng: &'a mut Pcg,
    input: Vec<u8>,
    r2s: &'a [u8; R2S_LEN],
}

impl TreeGen<'_> {
    /// Appends b to the input and returns its offset.
    fn store(&mut self, b: &[u8]) -> u32 {
        self.input.extend_from_slice(b);
        (self.input.len() - b.len()) as u32
    }

    /// Returns a subtree decoding to k values and those values.
    fn node(&mut self, k: usize, depth: usize) -> (Box<TNode>, Vec<u8>) {
        let mut n = Box::new(TNode {
            k,
            ..TNode::default()
        });
        if k == 0 {
            return (n, Vec::new());
        }
        let mut sel = self.rng.int_n(10);
        if depth >= 7 || k < 2 {
            sel = 7 + self.rng.int_n(3);
        }
        if sel >= 8 {
            (n.kind, n.d) = (OP_FLAT, (2 + self.rng.int_n(7)) as u8);
            let region = rand_vec(self.rng, (k * usize::from(n.d)).div_ceil(8));
            n.bm = self.store(&region);
            if n.d == 8 {
                return (n, region);
            }
            n.p = self.rng.int_n(257 - (1 << n.d)) as u8;
            let out = ref_unpack(&region, k, usize::from(n.d))
                .into_iter()
                .map(|c| self.r2s[usize::from(n.p) + usize::from(c)])
                .collect();
            return (n, out);
        }
        let densities = [0.0, 0.02, 0.3, 0.5, 0.7, 0.98, 1.0];
        let di = self.rng.int_n(densities.len());
        let (bm, kr) = rand_bits(self.rng, k, densities[di]);
        n.kr = kr;
        n.p = self.rng.uint32() as u8;
        n.q = self.rng.uint32() as u8;
        n.bm = self.store(&bm);
        let (lout, rout);
        if sel < 5 {
            n.kind = OP_VV;
            let (l, lo) = self.node(k - kr, depth + 1);
            let (r, ro) = self.node(kr, depth + 1);
            (n.l, n.r, lout, rout) = (Some(l), Some(r), lo, ro);
        } else if sel < 7 {
            n.kind = OP_CV;
            lout = vec![n.p; k];
            let (r, ro) = self.node(kr, depth + 1);
            (n.r, rout) = (Some(r), ro);
        } else {
            n.kind = OP_CC;
            (lout, rout) = (vec![n.p; k], vec![n.q; k]);
        }
        let (out, _) = ref_merge(&bm, k, &lout, &rout);
        (n, out)
    }
}

impl TNode {
    fn op(&self, out: u32, l: u32, r: u32) -> Vec<Op> {
        if self.kind == OP_FLAT {
            return flat_ops(self.k, self.d, self.p, 0, self.bm, out);
        }
        let (kind, p, q, k, kr, bm) = (
            self.kind,
            self.p,
            self.q,
            self.k as u32,
            self.kr as u32,
            self.bm,
        );
        vec![Op {
            kind,
            p,
            q,
            k,
            kr,
            bm,
            out,
            l,
            r,
            ..Op::default()
        }]
    }

    fn left(&self) -> &TNode {
        self.l.as_deref().unwrap()
    }

    fn right(&self) -> &TNode {
        self.r.as_deref().unwrap()
    }

    /// Mirrors the upstream subtree placement: out[0:K), scratch at tmp.
    fn sub(&self, ops: &mut Vec<Op>, out: u32, tmp: u32) {
        if self.k == 0 {
            return;
        }
        let (kl, kr) = ((self.k - self.kr) as u32, self.kr as u32);
        match self.kind {
            OP_CV => {
                let rb = out + kl;
                self.right().sub(ops, rb, tmp);
                ops.extend(self.op(out, 0, rb));
            }
            OP_VV => {
                let (lb, rb);
                if kr > kl {
                    (rb, lb) = (out + kl, tmp);
                    self.right().sub(ops, rb, tmp);
                    self.left().sub(ops, lb, out);
                } else {
                    (lb, rb) = (out + kr, tmp);
                    self.left().sub(ops, lb, tmp);
                    self.right().sub(ops, rb, out);
                }
                ops.extend(self.op(out, lb, rb));
            }
            _ => ops.extend(self.op(out, 0, 0)),
        }
    }

    /// Mirrors the upstream root placement; the root op writes Dst.
    fn root(&self) -> Vec<Op> {
        let mut ops = Vec::new();
        let (kl, kr, k) = ((self.k - self.kr) as u32, self.kr as u32, self.k as u32);
        let (mut l, mut r) = (0, 0);
        match self.kind {
            OP_CV => self.right().sub(&mut ops, 0, kr),
            OP_VV => {
                if kr > kl {
                    self.right().sub(&mut ops, 0, kr);
                    self.left().sub(&mut ops, kr, k);
                    (l, r) = (kr, 0);
                } else {
                    self.left().sub(&mut ops, 0, kl);
                    self.right().sub(&mut ops, kl, k);
                    (l, r) = (0, kl);
                }
            }
            _ => {}
        }
        let m = ops.len();
        ops.extend(self.op(0, l, r));
        for o in &mut ops[m..] {
            o.flags |= FLAG_DST;
        }
        ops
    }
}

/// Returns a random program over n symbols, its env and its expected output.
fn random_tree(rng: &mut Pcg, n: usize) -> (Env, Vec<u8>) {
    let mut r2s = [0u8; R2S_LEN];
    rand_bytes(rng, &mut r2s);
    let mut g = TreeGen {
        rng,
        input: Vec::new(),
        r2s: &r2s,
    };
    let (root, want) = g.node(n, 0);
    let input = g.input;
    let mut e = Env::new(rng, 2 * n, n, &input);
    e.r2s = r2s;
    e.ops = root.root();
    (e, want)
}

#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn exec_tree() {
    let sizes = test_sizes();
    for tier in Tier::tiers() {
        let mut rng = new_rng();
        for iter in 0..400 {
            let mut n = sizes[rng.int_n(sizes.len())];
            if n == 0 || iter % 50 == 0 {
                n = 65535;
            }
            let (mut e, want) = random_tree(&mut rng, n);
            let nops = e.ops.len();
            if let Err(i) = e.exec(tier, nops) {
                panic!("{tier} iter {iter} n={n}: op {i} ({:?}) failed", e.ops[i]);
            }
            assert!(
                e.dst() == want,
                "{tier} iter {iter} n={n}: wrong output ({nops} ops)"
            );
        }
    }
}

#[test]
fn exec_fail_closed() {
    const ARENA_LEN: usize = 1024;
    const DST_LEN: u32 = 64;
    const IN_LEN: u32 = 256;
    const LIM: u32 = ARENA_LEN as u32; // last addressable arena byte + 1
    let valid = |out: u32| Op {
        kind: OP_CC,
        p: 1,
        q: 2,
        k: 16,
        bm: 8,
        out,
        ..Op::default()
    };
    let o = |kind: u8, d: u8, k: u32| Op {
        kind,
        d,
        k,
        out: 200,
        ..Op::default()
    };
    let bad: Vec<(&str, Op, bool)> = vec![
        ("kind7", o(7, 0, 8), false),
        ("kind8", o(8, 0, 8), false),
        ("kind255", o(255, 0, 8), false),
        ("v512/d0", o(OP_V512, 0, 512), false),
        ("v512/d1", o(OP_V512, 1, 512), false),
        ("v512/d8", o(OP_V512, 8, 0), false),
        ("v128/d8", o(OP_V128, 8, 128), false),
        ("v128/d9", o(OP_V128, 9, 128), false),
        ("nat/d0", o(OP_NAT, 0, 8), false),
        ("nat/d1", o(OP_NAT, 1, 8), false),
        ("nat/d9", o(OP_NAT, 9, 8), false),
        ("copy/d7", o(OP_COPY, 7, 8), false),
        ("copy/d0", o(OP_COPY, 0, 8), false),
        ("v512/k", o(OP_V512, 2, 128), false),
        ("v128/k", o(OP_V128, 2, 64), false),
        (
            "r2s",
            Op {
                p: 255,
                ..o(OP_NAT, 7, 8)
            },
            false,
        ),
        (
            "r2s/d8",
            Op {
                p: 65,
                ..o(OP_NAT, 8, 8)
            },
            false,
        ),
        (
            "out",
            Op {
                out: LIM - 7,
                ..o(OP_CC, 0, 8)
            },
            false,
        ),
        (
            "out/huge",
            Op {
                out: 0xffffffff,
                ..o(OP_CC, 0, 0xffffffff)
            },
            false,
        ),
        (
            "out/dst",
            Op {
                flags: FLAG_DST,
                out: DST_LEN - 7,
                ..o(OP_CC, 0, 8)
            },
            false,
        ),
        (
            "bm",
            Op {
                bm: IN_LEN - 7,
                ..o(OP_CC, 0, 64)
            },
            false,
        ),
        (
            "region",
            Op {
                bm: IN_LEN - 2,
                ..o(OP_NAT, 3, 8)
            },
            false,
        ),
        (
            "region/copy",
            Op {
                bm: IN_LEN - 7,
                ..o(OP_COPY, 8, 8)
            },
            false,
        ),
        (
            "region/huge",
            Op {
                bm: 0,
                ..o(OP_NAT, 8, 0xffffffff)
            },
            false,
        ),
        (
            "vv/l",
            Op {
                l: LIM - 7,
                r: 300,
                ..o(OP_VV, 0, 8)
            },
            false,
        ),
        (
            "vv/r",
            Op {
                l: 300,
                r: LIM - 7,
                ..o(OP_VV, 0, 8)
            },
            false,
        ),
        (
            "cv/r",
            Op {
                r: 0xfffffff0,
                ..o(OP_CV, 0, 8)
            },
            false,
        ),
        (
            "vv/kr+1",
            Op {
                kr: 33,
                l: 300,
                r: 400,
                ..o(OP_VV, 0, 64)
            },
            true,
        ),
        (
            "vv/kr-1",
            Op {
                kr: 31,
                l: 300,
                r: 400,
                ..o(OP_VV, 0, 64)
            },
            true,
        ),
        (
            "vv/kr-huge",
            Op {
                kr: 0xffffffff,
                l: 300,
                r: 400,
                ..o(OP_VV, 0, 64)
            },
            true,
        ),
        (
            "cv/kr+1",
            Op {
                kr: 33,
                r: 400,
                ..o(OP_CV, 0, 64)
            },
            true,
        ),
        (
            "cv/kr0",
            Op {
                kr: 0,
                r: 400,
                ..o(OP_CV, 0, 64)
            },
            true,
        ),
    ];
    for tier in Tier::tiers() {
        let mut rng = new_rng();
        for &(name, op, written) in &bad {
            let mut input = rand_vec(&mut rng, IN_LEN as usize);
            input[..8].fill(0x55); // 32 of 64 bits set at bm=0
            input[8..10].fill(0); // the valid ops' bitmap
            let mut e = Env::new(&mut rng, ARENA_LEN, DST_LEN as usize, &input);
            e.ops = vec![valid(0), valid(100), op, valid(150)];
            let upto = if written { 3 } else { 2 };
            assert_eq!(e.exec(tier, upto), Err(2), "{tier} {name}");
            assert!(
                e.out(&valid(100)) == [1; 16],
                "{tier} {name}: op before the failure did not run"
            );
        }
        // A missing R2S fails flat lookups instead of crashing.
        let mut e = Env::new(&mut rng, ARENA_LEN, DST_LEN as usize, &[0; 8]);
        e.nil_r2s = true;
        e.ops = vec![Op {
            kind: OP_NAT,
            d: 4,
            k: 8,
            ..Op::default()
        }];
        assert_eq!(e.exec(tier, 0), Err(0), "{tier} nil R2S");
    }
}

/// Runs an op of each kind after a valid op of every kind and before another valid op: at
/// the limit of each of its checks, where it must run, and one past it, where exec must
/// return its index and the op after it must not run.
#[test]
fn exec_bounds() {
    const IN_LEN: u32 = 1024;
    const ALIM: u32 = 4096;
    const DST_LEN: u32 = 1024;
    let mut rng = new_rng();
    let input = rand_vec(&mut rng, IN_LEN as usize);
    let with_kr = |mut op: Op| {
        if (op.kind == OP_VV || op.kind == OP_CV)
            && op.bm as usize + (op.k as usize).div_ceil(8) <= IN_LEN as usize
        {
            op.kr = dec_popcnt(&input[op.bm as usize..], op.k as usize);
        }
        op
    };
    let mk = |kind, flags, d, p, q, k, bm, out, l, r| Op {
        kind,
        flags,
        d,
        p,
        q,
        k,
        bm,
        out,
        l,
        r,
        ..Op::default()
    };
    // The valid ops write the arena below 1416 and read children from [2048, 2477).
    let valid = [
        with_kr(mk(OP_VV, 0, 0, 0, 0, 100, 0, 0, 2048, 2200)),
        with_kr(mk(OP_CV, 0, 0, 3, 0, 77, 20, 128, 0, 2400)),
        mk(OP_CC, 0, 0, 1, 2, 45, 40, 256, 0, 0),
        mk(OP_V512, 0, 3, 9, 0, 512, 100, 512, 0, 0),
        mk(OP_V128, 0, 7, 192, 0, 128, 300, 1024, 0, 0),
        mk(OP_NAT, 0, 5, 200, 0, 45, 500, 1200, 0, 0),
        mk(OP_COPY, 0, 8, 0, 0, 37, 600, 1300, 0, 0),
    ];
    let last = mk(OP_CC, 0, 0, 5, 6, 16, 0, 1400, 0, 0);

    struct BCase {
        name: String,
        op: Op,
        runs: bool,
        /// Fails after writing its output (KR).
        written: bool,
        nil_r2s: bool,
    }
    let mut cases: Vec<BCase> = Vec::new();
    let mut add = |name: String, op: Op, runs: bool| {
        let name = format!("{}/{name}", op.kind);
        cases.push(BCase {
            name,
            op: with_kr(op),
            runs,
            written: false,
            nil_r2s: false,
        });
    };
    let need = |op: &Op| -> u32 {
        let (k, d) = (op.k, u32::from(op.d));
        match op.kind {
            OP_VV | OP_CV | OP_CC => k.div_ceil(8),
            OP_V512 | OP_V128 => k * d / 8,
            OP_NAT => (k * d).div_ceil(8),
            _ => op.k,
        }
    };
    let mut extra: Vec<BCase> = Vec::new();
    // Outputs go to Dst unless the case is about the arena's; children are at [3072, 3377).
    for op in [
        mk(OP_VV, FLAG_DST, 0, 0, 0, 77, 700, 0, 3072, 3300),
        mk(OP_CV, FLAG_DST, 0, 4, 0, 77, 700, 0, 0, 3300),
        mk(OP_CC, FLAG_DST, 0, 4, 8, 77, 700, 0, 0, 0),
        mk(OP_V512, FLAG_DST, 2, 50, 0, 512, 700, 0, 0, 0),
        mk(OP_V512, FLAG_DST, 7, 50, 0, 512, 500, 0, 0, 0),
        mk(OP_V128, FLAG_DST, 4, 50, 0, 128, 700, 0, 0, 0),
        mk(OP_V128, FLAG_DST, 7, 50, 0, 128, 700, 0, 0, 0),
        mk(OP_NAT, FLAG_DST, 3, 50, 0, 45, 700, 0, 0, 0),
        mk(OP_NAT, FLAG_DST, 7, 50, 0, 45, 700, 0, 0, 0),
        mk(OP_NAT, FLAG_DST, 8, 50, 0, 45, 700, 0, 0, 0),
        mk(OP_COPY, FLAG_DST, 8, 0, 0, 37, 700, 0, 0, 0),
    ] {
        // limit adds op with set(op, 0) at the limit and set(op, 1) past it.
        let mut limit = |name: &str, set: &dyn Fn(&mut Op, u32)| {
            for d in 0..2 {
                let mut o = op;
                set(&mut o, d);
                add(format!("{name}+{d}"), o, d == 0);
            }
        };
        limit("out", &|o, d| (o.flags, o.out) = (0, ALIM - o.k + d));
        limit("dst", &|o, d| o.out = DST_LEN - o.k + d);
        limit("in", &|o, d| o.bm = IN_LEN - need(o) + d);
        let lut = (OP_V512..=OP_NAT).contains(&op.kind);
        if lut && op.d >= 7 {
            limit(&format!("r2s/d{}", op.d), &|o, d| {
                o.p = (R2S_LEN as u32 - (1 << o.d) + d) as u8
            });
        }
        if op.kind == OP_VV {
            limit("l", &|o, d| o.l = ALIM - o.k + d);
        }
        if op.kind == OP_VV || op.kind == OP_CV {
            limit("r", &|o, d| o.r = ALIM - o.k + d);
        }
        extra.push(BCase {
            name: format!("{}/nilR2S", op.kind),
            op: with_kr(op),
            runs: !lut,
            written: false,
            nil_r2s: true,
        });
        if op.kind == OP_VV || op.kind == OP_CV {
            for dk in [1u32, u32::MAX] {
                let mut o = with_kr(op);
                o.kr = o.kr.wrapping_add(dk);
                extra.push(BCase {
                    name: format!("{}/kr{:+}", op.kind, dk as i32),
                    op: o,
                    runs: false,
                    written: true,
                    nil_r2s: false,
                });
            }
        }
        add("k0".into(), Op { k: 0, ..op }, true);
    }
    // D=64: the assembly's shifts by D wrap mod 64, so 1<<D passes its R2S check and only
    // the D range rejects it.
    for d in [0u8, 1, 2, 7, 8, 9, 64, 255] {
        let ok = (2..=7).contains(&d);
        add(
            format!("d{d}"),
            mk(OP_V512, 0, d, 0, 0, 512, 0, 0, 0, 0),
            ok,
        );
        add(
            format!("d{d}"),
            mk(OP_V128, 0, d, 0, 0, 128, 0, 0, 0, 0),
            ok,
        );
        add(
            format!("d{d}"),
            mk(OP_NAT, 0, d, 0, 0, 64, 0, 0, 0, 0),
            (2..=8).contains(&d),
        );
        add(
            format!("d{d}"),
            mk(OP_COPY, 0, d, 0, 0, 64, 0, 0, 0, 0),
            d == 8,
        );
    }
    add(
        "k640".into(),
        mk(OP_V512, 0, 2, 0, 0, 640, 0, 0, 0, 0),
        false,
    );
    add(
        "k192".into(),
        mk(OP_V128, 0, 2, 0, 0, 192, 0, 0, 0, 0),
        false,
    );
    for kind in [7u8, 8, 255] {
        add("kind".into(), mk(kind, 0, 8, 0, 0, 8, 0, 0, 0, 0), false);
    }
    cases.extend(extra);

    let base = Env::new(&mut rng, ALIM as usize, DST_LEN as usize, &input);
    for c in &cases {
        let mut ops: Vec<Op> = if c.nil_r2s {
            vec![valid[0], valid[1], valid[2], valid[6]]
        } else {
            valid.to_vec()
        };
        ops.extend([c.op, last]);
        let (mut want, mut upto) = (Ok(()), ops.len());
        if !c.runs {
            (want, upto) = (Err(ops.len() - 2), ops.len() - 2);
            if c.written {
                upto += 1;
            }
        }
        let prog = || {
            let mut e = base.clone();
            e.ops = ops.clone();
            e.nil_r2s = c.nil_r2s;
            e
        };
        for tier in Tier::tiers() {
            assert_eq!(
                prog().exec(tier, upto),
                want,
                "{tier} {}: {:?}",
                c.name,
                c.op
            );
            if tier != Tier::Generic {
                let _ = dec_diff_exec(tier, &mut prog(), &c.name);
            }
        }
    }
}

/// Checks Op keeps Go's kernel.Op layout, and the constants the vector tiers rely on.
#[test]
fn exec_layout() {
    let got = [
        offset_of!(Op, kind),
        offset_of!(Op, flags),
        offset_of!(Op, d),
        offset_of!(Op, p),
        offset_of!(Op, q),
        offset_of!(Op, k),
        offset_of!(Op, kr),
        offset_of!(Op, bm),
        offset_of!(Op, out),
        offset_of!(Op, l),
        offset_of!(Op, r),
        size_of::<Op>(),
    ];
    assert_eq!(
        got,
        [0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32],
        "Op offsets"
    );
    let consts = [
        OP_VV, OP_CV, OP_CC, OP_V512, OP_V128, OP_NAT, OP_COPY, FLAG_DST,
    ];
    assert_eq!(consts, [0, 1, 2, 3, 4, 5, 6, 1]);
    assert_eq!((ARENA_SLACK, R2S_LEN), (128, 320));
}

/// Runs e on the generic tier (on a copy) and on tier, and requires the same result and
/// memory, untouched inputs and Dst guard bytes. Only a failing op's own output may differ:
/// VV/CV write before their KR check.
fn dec_diff_exec(tier: Tier, e: &mut Env, what: &str) -> Result<(), usize> {
    let mut q = e.clone();
    let want = Tier::Generic.exec(&mut q.program());
    let (dst_all, input, r2s) = (e.dst_all.clone(), e.input.clone(), e.r2s);
    let got = tier.exec(&mut e.program());
    assert_eq!(
        got,
        want,
        "{tier} {what}: exec differs from generic; op {:?}",
        e.ops.get(got.err().or(want.err()).unwrap_or(usize::MAX))
    );
    if let Err(i) = want {
        let op = e.ops[i];
        let (out, refb) = if op.flags & FLAG_DST != 0 {
            (&e.dst_all[..e.dst_len], &mut q.dst_all[..q.dst_len])
        } else {
            (&e.arena[..], &mut q.arena[..])
        };
        let lo = (op.out as usize).min(out.len());
        let hi = (op.out as usize + op.k as usize).min(out.len());
        refb[lo..hi].copy_from_slice(&out[lo..hi]);
    }
    assert!(
        q.arena == e.arena,
        "{tier} {what}: arena differs from generic (ret {got:?})"
    );
    assert!(
        q.dst() == e.dst(),
        "{tier} {what}: dst differs from generic (ret {got:?})"
    );
    assert!(
        dst_all[e.dst_len..] == e.dst_all[e.dst_len..] && input == e.input && r2s == e.r2s,
        "{tier} {what}: wrote outside its outputs (ret {got:?})"
    );
    got
}

/// Breaks one op of an upstream-placed program in a way that cannot create aliasing the
/// placement proof does not cover, so both tiers must still agree byte for byte.
fn dec_mutate(rng: &mut Pcg, e: &Env, ops: &mut [Op]) {
    let i = rng.int_n(ops.len());
    let op = &mut ops[i];
    let src_len = e.in_len as i64;
    let out_lim = if op.flags & FLAG_DST != 0 {
        e.dst_len
    } else {
        e.arena.len() - ARENA_SLACK
    } as i64;
    let alim = (e.arena.len() - ARENA_SLACK) as i64;
    let over = |rng: &mut Pcg, k: u32, lim: i64| {
        (lim - i64::from(k) + 1 + rng.int_n(3) as i64).max(0) as u32
    };
    match rng.int_n(10) {
        0 => {
            let v = [-1i64, 1, rng.int_n(65536) as i64];
            op.kr = op.kr.wrapping_add(v[rng.int_n(3)] as u32);
        }
        1 => {
            let v = [
                OP_CC,
                OP_V512,
                OP_V128,
                OP_NAT,
                OP_COPY,
                7,
                (7 + rng.int_n(249)) as u8,
            ];
            op.kind = v[rng.int_n(7)];
        }
        2 => {
            op.d = rng.int_n(11) as u8;
            if rng.int_n(8) == 0 {
                op.d = rng.uint32() as u8;
            }
        }
        3 => (op.p, op.q) = (rng.uint32() as u8, rng.uint32() as u8),
        4 => op.flags ^= 2, // not a flag: ignored
        5 => op.flags |= FLAG_DST,
        6 => op.out = over(rng, op.k, out_lim),
        7 => {
            if op.kind == OP_VV || op.kind == OP_CV {
                if rng.int_n(2) == 0 {
                    op.l = over(rng, op.k, alim);
                } else {
                    op.r = over(rng, op.k, alim);
                }
            } else {
                op.k = !(rng.int_n(4) as u32);
            }
        }
        8 => {
            let need = (i64::from(op.k) * i64::from(op.d.max(1)) + 7) / 8;
            op.bm = (src_len - need + 1 + rng.int_n(2) as i64).max(0) as u32;
        }
        _ => op.k = !(rng.int_n(1 << 20) as u32),
    }
}

/// Runs random upstream-placed programs (as exec_tree) on every vector tier and the generic
/// one, half of them with one broken op.
#[test]
fn exec_conformance() {
    let tiers = vec_tiers();
    if tiers.is_empty() {
        return;
    }
    let sizes = test_sizes();
    let mut rng = Pcg::new(3, 4);
    for iter in 0..3000 {
        let mut n = sizes[rng.int_n(sizes.len())];
        if n == 0 || iter % 97 == 0 {
            n = 65535;
        }
        let (mut e, want) = random_tree(&mut rng, n);
        let mut ops = core::mem::take(&mut e.ops);
        let broken = iter % 2 == 1 && !ops.is_empty();
        if broken {
            dec_mutate(&mut rng, &e, &mut ops);
        }
        e.ops = ops;
        for &tier in &tiers {
            let mut c = e.clone();
            let ret = dec_diff_exec(tier, &mut c, "tree");
            assert!(
                broken || ret.is_ok(),
                "{tier} iter {iter} n={n}: valid op {ret:?} failed"
            );
            assert!(
                broken || c.dst() == want,
                "{tier} iter {iter} n={n}: wrong output"
            );
        }
    }
}

/// Runs every kind and many D values with valid geometry on every vector tier.
#[test]
fn exec_kind_sweep() {
    let tiers = vec_tiers();
    if tiers.is_empty() {
        return;
    }
    let mut rng = new_rng();
    for kind in 0..=255u8 {
        for d in [0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 64, 128, 255] {
            for k in [0usize, 1, 128, 512, 700] {
                let input = rand_vec(&mut rng, k + 16);
                let mut e = Env::new(&mut rng, 3 * k + 64, k, &input);
                let ku = k as u32;
                // Out, L and R do not overlap; P is random, so some flats fail on R2S.
                let p = rng.int_n(256) as u8;
                e.ops = vec![Op {
                    kind,
                    d,
                    p,
                    q: 9,
                    k: ku,
                    kr: dec_popcnt(&input[3..], k),
                    bm: 3,
                    out: 0,
                    l: ku,
                    r: 2 * ku,
                    ..Op::default()
                }];
                for &tier in &tiers {
                    let _ = dec_diff_exec(tier, &mut e.clone(), "sweep");
                }
            }
        }
    }
}

/// An empty program succeeds without writing.
#[test]
fn exec_empty() {
    for tier in Tier::tiers() {
        let mut e = Env::new(&mut new_rng(), 64, 24, &[0xff]);
        assert_eq!(e.exec(tier, 0), Ok(()), "{tier}: empty program");
    }
}

/// Decodes up to four ops from the input, with the arena split into zones so that outputs
/// never alias merge inputs, and compares the tiers (Go FuzzExec).
fn fuzz_exec(data: &[u8]) {
    let tiers = vec_tiers();
    // Out+K <= zone+7 < L <= L+K < R: no output overlaps a merge input.
    const ZONE: u32 = 1 << 12;
    const HDR: usize = 12;
    let nops = (data.len() / HDR).min(4);
    let input = &data[nops * HDR..];
    let ops: Vec<Op> = (0..nops)
        .map(|i| {
            let h = &data[i * HDR..];
            let mut op = Op {
                kind: h[0] % 9,
                flags: h[1] & 3,
                d: h[2] % 10,
                p: h[3],
                q: h[4],
                k: u32::from(u16::from_le_bytes([h[5], h[6]])) % (ZONE + 1),
                bm: u32::from(h[9]) | u32::from(h[10] & 15) << 8,
                out: u32::from(h[11] & 7),
                ..Op::default()
            };
            (op.l, op.r) = (
                ZONE + 16 + u32::from(h[11] >> 3),
                2 * ZONE + 64 + u32::from(h[11] >> 5),
            );
            // Mostly the right KR, so that merges run to the end.
            if op.bm as usize + (op.k as usize).div_ceil(8) <= input.len() {
                op.kr = dec_popcnt(&input[op.bm as usize..], op.k as usize);
            }
            op.kr = op
                .kr
                .wrapping_add([0, 0, 1, u32::MAX][usize::from(h[7] & 3)]);
            op
        })
        .collect();
    let mut rng = Pcg::new(data.len() as u64, 7);
    let mut e = Env::new(&mut rng, 3 * ZONE as usize + 128, ZONE as usize + 8, input);
    e.ops = ops;
    for tier in tiers {
        let _ = dec_diff_exec(tier, &mut e.clone(), "fuzz");
    }
}

/// Go FuzzExec over its seeds and seeded random inputs.
#[test]
fn fuzz_exec_seeded() {
    if vec_tiers().is_empty() {
        return;
    }
    fuzz_exec(&[
        0, 0, 0, 0, 0, 64, 0, 32, 0, 0, 0, 0, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55,
    ]);
    fuzz_exec(&[3, 2, 3, 1, 0, 0, 2, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]);
    fuzz_exec(&[5, 1, 7, 9, 0, 77, 0, 0, 0, 5, 0, 0, 9, 8, 7, 6]);
    let mut rng = Pcg::new(0xFE, 0xED);
    for _ in 0..2000 {
        let n = rng.int_n(48) + rng.int_n(600);
        fuzz_exec(&rand_vec(&mut rng, n));
    }
}

/// Places every input flush against an inaccessible page, before and after it, so any
/// read or write outside the exact ranges faults; for every tier.
#[test]
#[cfg_attr(miri, ignore = "no page guards under Miri")]
fn exec_page_guard() {
    // 0..=80 reaches every branch of the exact 0..16-byte tail loads and stores.
    let mut sizes = vec![127, 128, 129, 200, 511, 512, 513, 1000, 1024, 4096];
    sizes.extend(0..=80);
    for tier in Tier::tiers() {
        let mut rng = new_rng();
        for at_end in [true, false] {
            for &k in &sizes {
                dec_guard_merges(tier, &mut rng, k, at_end);
                for d in 2..=8 {
                    dec_guard_flat(tier, &mut rng, k, d, at_end);
                }
            }
        }
    }
}

/// A Program's buffers: In = src, Dst = k bytes and R2S, each in its own guarded pages,
/// and an arena of arena+ARENA_SLACK bytes flush against its guard.
struct GuardedProgram {
    bufs: [crate::guard::Buffer; 4],
    lens: [usize; 4],
    at_end: bool,
}

impl GuardedProgram {
    fn new(rng: &mut Pcg, src: &[u8], k: usize, arena: usize, at_end: bool) -> Self {
        let lens = [src.len(), k, arena + ARENA_SLACK, R2S_LEN];
        let mut g = GuardedProgram {
            bufs: lens.map(guard_buf),
            lens,
            at_end,
        };
        g.buf(0).copy_from_slice(src);
        for i in 1..4 {
            rand_bytes(rng, g.buf(i));
        }
        g
    }

    fn buf(&mut self, i: usize) -> &mut [u8] {
        // The arena is always flush against the guard after it.
        self.bufs[i].slice(self.lens[i], self.at_end || i == 2)
    }

    fn exec(&mut self, tier: Tier, ops: &[Op]) -> Result<(), usize> {
        let [input, dst, arena, r2s] = &mut self.bufs;
        let (lens, at_end) = (self.lens, self.at_end);
        tier.exec(&mut Program {
            ops,
            input: input.slice(lens[0], at_end),
            arena: arena.slice(lens[2], true),
            dst: dst.slice(lens[1], at_end),
            r2s: Some((&*r2s.slice(R2S_LEN, at_end)).try_into().unwrap()),
        })
    }
}

fn dec_guard_merges(tier: Tier, rng: &mut Pcg, k: usize, at_end: bool) {
    let (bm, kr) = rand_bits(rng, k, 0.5);
    let (ku, kl, kru) = (k as u32, (k - kr) as u32, kr as u32);
    let (l, r) = (rand_vec(rng, k - kr), rand_vec(rng, kr));
    for op in [
        Op {
            kind: OP_VV,
            flags: FLAG_DST,
            k: ku,
            kr: kru,
            l: ku - kl,
            r: 2 * ku - kru,
            ..Op::default()
        },
        Op {
            kind: OP_VV,
            k: ku,
            kr: kru,
            l: ku,
            r: 2 * ku,
            ..Op::default()
        },
        Op {
            kind: OP_CV,
            flags: FLAG_DST,
            p: 7,
            k: ku,
            kr: kru,
            r: 2 * ku - kru,
            ..Op::default()
        },
        Op {
            kind: OP_CC,
            flags: FLAG_DST,
            p: 7,
            q: 9,
            k: ku,
            ..Op::default()
        },
    ] {
        let mut g = GuardedProgram::new(rng, &bm, k, 3 * k, at_end);
        g.buf(2)[op.l as usize..][..l.len()].copy_from_slice(&l);
        g.buf(2)[op.r as usize..][..r.len()].copy_from_slice(&r);
        let want = match op.kind {
            OP_VV => ref_merge(&bm, k, &l, &r).0,
            OP_CV => ref_merge(&bm, k, &vec![7; k], &r).0,
            _ => ref_merge(&bm, k, &vec![7; k], &vec![9; k]).0,
        };
        let ret = g.exec(tier, &[op]);
        assert_eq!(ret, Ok(()), "{tier} k={k} kind {}: exec", op.kind);
        let out = if op.flags & FLAG_DST != 0 {
            g.buf(1)
        } else {
            &mut g.buf(2)[..k]
        };
        assert!(
            *out == want,
            "{tier} k={k} kind {} at_end={at_end}: wrong output",
            op.kind
        );
    }
}

fn dec_guard_flat(tier: Tier, rng: &mut Pcg, k: usize, d: u8, at_end: bool) {
    let du = usize::from(d);
    let mut region = rand_vec(rng, (k * du).div_ceil(8));
    let s = k * du % 8;
    if s != 0 {
        *region.last_mut().unwrap() &= (1 << s) - 1;
    }
    let mut g = GuardedProgram::new(rng, &region, k, 0, at_end);
    // The table is as close to its guard page as P allows.
    let first = if d < 8 && at_end {
        (R2S_LEN - (1 << d)).min(255)
    } else {
        0
    };
    let ret = g.exec(tier, &flat_ops(k, d, first as u8, FLAG_DST, 0, 0));
    assert_eq!(ret, Ok(()), "{tier} k={k} d={d}: exec");
    let want = if d < 8 {
        let r2s = g.buf(3).to_vec();
        ref_unpack(&region, k, du)
            .iter()
            .map(|&c| r2s[first + usize::from(c)])
            .collect()
    } else {
        region
    };
    assert!(
        *g.buf(1) == want,
        "{tier} k={k} d={d} at_end={at_end}: wrong output"
    );
}

/// Runs random, often malformed ops whose slices end at their limits (Out+K at Dst or the
/// arena less its slack, children at it, regions at the end of In, tables at the end of
/// R2S) on guarded buffers: an op must stay inside the slices whether it passes its checks
/// or not, and every tier must fail the same op as generic.
#[test]
#[cfg_attr(miri, ignore = "no page guards under Miri")]
fn exec_page_guard_malformed() {
    let mut rng = Pcg::new(0x9A, 0x6E);
    // Ends at lim, or anywhere below it.
    let place = |rng: &mut Pcg, n: usize, lim: usize| {
        let hi = lim.saturating_sub(n);
        (if rng.int_n(2) == 0 {
            hi
        } else {
            rng.int_n(hi + 1)
        }) as u32
    };
    for iter in 0..4000 {
        let (in_len, dst_len, arena) = (rng.int_n(1500), rng.int_n(1200), rng.int_n(2500));
        let src = rand_vec(&mut rng, in_len);
        let mut ops = Vec::new();
        for _ in 0..1 + rng.int_n(4) {
            let (kind, d, flags) = (rng.int_n(8) as u8, rng.int_n(10) as u8, rng.int_n(2) as u8);
            let k = [rng.int_n(1100), 128 * rng.int_n(9), 512 * rng.int_n(3)][rng.int_n(3)];
            let need = match kind {
                OP_VV | OP_CV | OP_CC => k.div_ceil(8),
                OP_COPY => k,
                _ => (k * usize::from(d)).div_ceil(8),
            };
            let bm = place(&mut rng, need, in_len);
            let mut kr = if bm as usize + k.div_ceil(8) <= in_len {
                dec_popcnt(&src[bm as usize..], k)
            } else {
                0
            };
            kr = kr.wrapping_add([0, 0, 1, u32::MAX][rng.int_n(4)]);
            let p = if d <= 8 && rng.int_n(2) == 0 {
                (R2S_LEN - (1 << d)).min(255) as u8
            } else {
                rng.uint32() as u8
            };
            let lim = if flags & FLAG_DST != 0 {
                dst_len
            } else {
                arena
            };
            ops.push(Op {
                kind,
                flags,
                d,
                p,
                q: 9,
                k: k as u32,
                kr,
                bm,
                out: place(&mut rng, k, lim),
                l: place(&mut rng, k, arena),
                r: place(&mut rng, k, arena),
            });
        }
        let mut g = GuardedProgram::new(&mut rng, &src, dst_len, arena, iter % 2 == 0);
        let (mut dst, mut ar, r2s) = (g.buf(1).to_vec(), g.buf(2).to_vec(), g.buf(3).to_vec());
        let want = Tier::Generic.exec(&mut Program {
            ops: &ops,
            input: &src,
            arena: &mut ar,
            dst: &mut dst,
            r2s: Some(r2s[..].try_into().unwrap()),
        });
        for tier in Tier::tiers() {
            assert_eq!(g.exec(tier, &ops), want, "{tier} iter {iter}: {ops:?}");
        }
    }
}

// The NEON specific tests (Go neon_arm64_test.go), a child of this module for its helpers.
#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
#[path = "neon.rs"]
mod neon;
