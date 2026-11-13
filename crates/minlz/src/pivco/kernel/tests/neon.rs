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

//! Ports of Go's `neon_arm64_test.go`. The shared tests run [`Tier::Neon`] through
//! [`Tier::tiers`] and `vec_tiers`; these are NEON specific.

use super::*;
use crate::{cpu, guard};

#[test]
fn neon_selected() {
    let _d = cpu::Disabled::new(0);
    assert!(
        Tier::best() == Tier::Neon
            && Tier::tiers() == [Tier::Neon, Tier::Generic]
            && !Tier::Neon.hist_uses_scratch(),
        "best() = {}, tiers() = {:?}",
        Tier::best(),
        Tier::tiers()
    );
}

/// Decodes every flat size up to 1100 at every depth, split as the planner does and as one
/// natural op, from exact regions and with the largest first rank: the natural loops stop
/// on the end of the input, here the region's end, which every size reaches at a different
/// point.
#[test]
fn neon_flat_sizes() {
    let mut rng = new_rng();
    for n in 0..=1100 {
        for d in 2..=8u8 {
            let region = rand_vec(&mut rng, (n * usize::from(d)).div_ceil(8));
            let first = if d < 8 {
                (R2S_LEN - (1 << d)).min(255) as u8
            } else {
                0
            };
            let nat = Op {
                kind: OP_NAT,
                flags: FLAG_DST,
                d,
                p: first,
                k: n as u32,
                ..Op::default()
            };
            for ops in [flat_ops(n, d, first, FLAG_DST, 0, 0), vec![nat]] {
                let kind = ops[0].kind;
                let mut e = Env::new(&mut rng, 0, n, &region);
                e.ops = ops;
                let ret = dec_diff_exec(Tier::Neon, &mut e, "flat");
                assert_eq!(ret, Ok(()), "n={n} d={d} kind {kind}");
            }
        }
    }
}

/// Ends the arena's accessible memory 15 bytes past the K bytes of the last merge child
/// (the over-read ARENA_SLACK covers); the rest of the slack the arena claims is an
/// inaccessible page. All-left and all-right bitmaps keep a cursor at its output position,
/// where its loads reach farthest.
#[test]
fn neon_merge_guard() {
    if !guard::supported() {
        return;
    }
    let page = |n: usize| guard::Buffer::new(n).expect("page guards");
    let mut rng = new_rng();
    for k in 0..200usize {
        for density in [0.0, 0.5, 1.0] {
            let (bits, kr) = rand_bits(&mut rng, k, density);
            let mut gbm = page(bits.len());
            let bm = gbm.slice(bits.len(), true);
            bm.copy_from_slice(&bits);
            let (ku, kru, kl) = (k as u32, kr as u32, k - kr);
            let merge = |kind, p, l, r| Op {
                kind,
                flags: FLAG_DST,
                p,
                k: ku,
                kr: kru,
                l,
                r,
                ..Op::default()
            };
            for (op, alim) in [
                (merge(OP_VV, 0, 0, ku), 2 * k),
                (merge(OP_VV, 0, ku, 0), 2 * k),
                (merge(OP_CV, 7, 0, 0), k),
            ] {
                let mut gmem = page(alim + 15);
                let mem = gmem.slice(alim + 15, true);
                rand_bytes(&mut rng, mem);
                // SAFETY: deliberately not upheld: the slice claims ARENA_SLACK-15 bytes of the
                // inaccessible page after mem, as Go's test does, which from_raw_parts_mut
                // forbids. Only the kernel may touch them, and it faults if it reads past the
                // 15; mem is not used again.
                let arena = unsafe {
                    core::slice::from_raw_parts_mut(mem.as_mut_ptr(), alim + ARENA_SLACK)
                };
                let l = if op.kind == OP_CV {
                    vec![7; kl]
                } else {
                    arena[op.l as usize..][..kl].to_vec()
                };
                let (want, _) = ref_merge(bm, k, &l, &arena[op.r as usize..][..kr]);
                let mut gdst = page(k);
                let mut p = Program {
                    ops: &[op],
                    input: bm,
                    arena,
                    dst: gdst.slice(k, true),
                    r2s: None,
                };
                let ret = Tier::Neon.exec(&mut p);
                assert!(
                    ret == Ok(()) && *p.dst == want,
                    "density {density} {op:?}: exec = {ret:?}, out equal {}",
                    *p.dst == want
                );
            }
        }
    }
}
