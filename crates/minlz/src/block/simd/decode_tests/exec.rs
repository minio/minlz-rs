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

//! Exec loop and stats tests of Go's `decode-simd_test.go`, run on every tier.

use super::*;
use crate::block::simd::decode::{Chunk, ll_stats, off_stats, tok_stats};
use crate::block::simd::exec::{Tier, exec_exact, exec_flat_go, exec_go, simd_tiers};
use crate::block::simd::{OFF_TAB, PAD, SLACK};
use crate::guard::Buffer;

type ExecFn = fn(Tier, &mut [u8], &[u8], &Chunk, usize, &mut [usize; 4]);
type Check<'a> = &'a dyn Fn(&[u8]) -> bool;

/// The operation loops to compare (Go `simdTestExecs`). dispatch runs the loops of the tier.
/// The flat loops run chunks with the no-overlap flag.
const TEST_EXECS: [(&str, bool, ExecFn); 6] = [
    ("go", false, |_, dst, lits, c, n, st| {
        exec_go(dst, lits, c, n, st)
    }),
    // SAFETY: the tiers come from simd_tiers, and the tests build chunks whose counts are what
    // their tokens read, in buffers that keep the memory rules.
    ("dispatch", false, |t, dst, lits, c, n, st| unsafe {
        t.exec(dst, lits, c, n, st)
    }),
    ("exact", false, test_exact),
    ("flat-go", true, |_, dst, lits, c, n, st| {
        test_flat(dst, lits, c, n, st, exec_flat_go)
    }),
    ("flat-dispatch", true, |t, dst, lits, c, n, st| {
        test_flat(dst, lits, c, n, st, |dst, lits, c, k0, k, cur, st| {
            // SAFETY: as for dispatch.
            unsafe { t.exec_flat(dst, lits, c, k0, k, cur, st) }
        })
    }),
    ("flat-exact", true, test_exact),
];

fn test_exact(_: Tier, dst: &mut [u8], lits: &[u8], c: &Chunk, n: usize, st: &mut [usize; 4]) {
    exec_exact(dst, lits, c, 0, n, [0; 4], st);
}

/// Runs the first n operations of a chunk with the no-overlap flag as `Chunk::run` does, with
/// f after the operations that start their match before MAX_LEN (Go `simdTestFlat`).
fn test_flat(
    dst: &mut [u8],
    lits: &[u8],
    c: &Chunk,
    n: usize,
    st: &mut [usize; 4],
    f: impl Fn(&mut [u8], &[u8], &Chunk, usize, usize, [usize; 4], &mut [usize; 4]),
) {
    let (mut k0, mut cur) = (0, [0; 4]);
    if st[0] < MAX_LEN {
        (k0, cur) = c.head(st[0], n);
        exec_exact(dst, lits, c, 0, k0, [0; 4], st);
    }
    if n > k0 {
        f(dst, lits, c, k0, n, cur, st);
    }
}

/// Returns the last n bytes of g, starting with b and zero after it.
fn at<'g>(g: &'g mut Buffer, b: &[u8], n: usize) -> &'g mut [u8] {
    let s = g.slice(n, true);
    s[..b.len()].copy_from_slice(b);
    s[b.len()..].fill(0);
    s
}

#[test]
fn simd_exec() {
    let mut rng = Rng(2);
    let mut b = TestBuilder::default();
    // A match longer than 32 bytes has an offset below 32, without and with the no-overlap flag.
    let mut cases: Vec<(Vec<TestOp>, Vec<u8>, bool, bool)> = Vec::new();
    // Miri: only the values on either side of MAX_LEN, and a few short random cases.
    let pick = |v: usize, edges: &[usize]| !cfg!(miri) || edges.contains(&v);
    for d0 in (1..=40).step_by(3).filter(|&d| pick(d, &[1, 28])) {
        for off in (1..=40).filter(|&o| pick(o, &[1, 32])) {
            for ml in (0..=40).step_by(3).filter(|&m| pick(m, &[30, 33])) {
                let mut lits = vec![0; d0 + 40];
                rng.read(&mut lits);
                // The second match starts at d0+5. The no-overlap flag raises its offset to 32.
                let bad = ml > MAX_LEN && off.min(d0 + 5) < MAX_LEN;
                let bad_flat = ml > MAX_LEN && d0 + 5 < MAX_LEN;
                let ops = vec![op(d0, 3, 7), op(2, ml, off), op(35, 5, 0)];
                cases.push((ops, lits, bad, bad_flat));
            }
        }
    }
    let (n_random, max_ops) = if cfg!(miri) { (6, 40) } else { (200, 300) };
    for _ in 0..n_random {
        let n = 1 + rng.intn(max_ops);
        let max_off = 1 << rng.intn(16);
        let (ops, lits) = test_ops(&mut rng, n, b"abcab\x00", 288, max_off);
        cases.push((ops, lits, false, false));
    }
    // Buffers end at an inaccessible page, right after what exec_go reads or writes.
    let mut guards: [Buffer; 7] = core::array::from_fn(|_| Buffer::new(1 << 20).unwrap());
    let [g0, g1, g2, g3, g4, g5, g6] = &mut guards;
    simd_tiers(|tier| {
        for (name, flat, f) in TEST_EXECS {
            for (i, (ops, lits, bad, bad_flat)) in cases.iter().enumerate() {
                for delta in [false, true] {
                    let tb = TestBlock {
                        ops: ops.clone(),
                        lits: lits.clone(),
                        chunk: ops.len(),
                        delta: delta as u64,
                        flat: flat as u64,
                        ..Default::default()
                    };
                    let bad = if flat { *bad_flat } else { *bad };
                    let (e, want) = b.build(&tb);
                    let v = &e.chunks[0].vals;
                    let l: usize = ops.iter().map(|o| o.ll).sum();
                    let d = l + want.len() - e.stored.len();
                    let (bits, _) = off_stats(Tier::Go, &v[OFFC]);
                    // Batches read 8 tokens, and 8 bytes at the literal lengths and offset
                    // symbols they use. raw is read 8 bytes at each bit position.
                    let nt = v[TOK].len();
                    let c = Chunk {
                        tok: at(g0, &v[TOK], nt + 7),
                        ll: at(g1, &v[LL], v[LL].len() + 8),
                        offc: at(g2, &v[OFFC], v[OFFC].len() + 8),
                        esc: at(g3, &v[ESC], v[ESC].len()),
                        raw: at(g4, &v[4], bits / 8 + 8),
                        n_tok: nt,
                        n_ll: v[LL].len(),
                        n_off: v[OFFC].len(),
                        n_esc: v[ESC].len(),
                        bits,
                        delta,
                        flat,
                    };
                    let dst = at(g5, &[], d + SLACK);
                    let lits = at(g6, &e.stored[..l], l + SLACK);
                    let mut st = [0, 0, 1, 0];
                    f(tier, dst, lits, &c, nt, &mut st);
                    if bad {
                        assert!(
                            st[3] != 0,
                            "{tier:?} {name} case {i} delta {delta}: long match below offset 32 not flagged"
                        );
                        continue;
                    }
                    assert!(
                        st[3] == 0 && st[0] == d && st[1] == l && dst[..d] == want[..d],
                        "{tier:?} {name} case {i} delta {delta}: state {st:?}, output mismatch {}",
                        dst[..d] != want[..d]
                    );
                }
            }
        }
    });
}

/// Running no operations writes nothing and keeps the state.
#[test]
fn simd_exec_empty() {
    simd_tiers(|tier| {
        // 1 is a repeat with no match, a literal length and an offset symbol without raw bits.
        let s = [1u8; PAD];
        let lits = [0xff; 100];
        let st0 = [40, 10, 9, 0];
        for (name, flat, f) in TEST_EXECS {
            for delta in [false, true] {
                let c = Chunk {
                    tok: &s,
                    ll: &s,
                    offc: &s,
                    esc: &s,
                    raw: &s,
                    // 0, so the streams are readable past it as the vector loops check.
                    n_tok: 0,
                    n_ll: 0,
                    n_off: 0,
                    n_esc: 0,
                    bits: 0,
                    delta,
                    flat,
                };
                let mut dst = [0u8; 100];
                let mut st = st0;
                f(tier, &mut dst, &lits, &c, 0, &mut st);
                let written = dst.iter().any(|&v| v != 0);
                assert!(
                    st == st0 && !written,
                    "{tier:?} {name} delta {delta}: state {st:?}, output written {written}"
                );
            }
        }
    });
}

/// Compares Tier::exec with exec_go on random streams. Offsets use every symbol, and most chunks
/// start late enough in dst that even the largest are not clamped. Only a prefix of the
/// operations runs, as before the exact loop.
#[test]
fn simd_exec_random() {
    // Miri: a 64 KiB history, which clamps the far offsets, and fewer, shorter chunks.
    let (hist, room, max_ops, cases) = if cfg!(miri) {
        (1 << 16, 1 << 16, 40, 12)
    } else {
        (1 << 24, 1 << 20, 300, 3000)
    };
    let mut rng = Rng(8);
    let mut base = vec![0; hist + room];
    rng.read(&mut base);
    let mut lits = vec![0; (max_ops + 100) * (MAX_LEN + 256) + 1000 + PAD];
    simd_tiers(|tier| {
        let (mut want, mut got) = (base.clone(), base.clone());
        for i in 0..cases {
            let (delta, flat) = (i & 1 != 0, i % 5 == 2);
            // Mostly new offsets with 20 raw bits span the most raw bytes per 16 operations.
            let wide = i % 3 == 1;
            let (mut d0, l0, mut last) = (hist, rng.intn(1000), 1 + rng.intn(hist - 1));
            if i % 4 == 0 {
                d0 = 1 + rng.intn(100);
            }
            let st0 = [d0, l0, last, 0];
            let (mut tok, mut ll, mut offc, mut esc, mut raw) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let (mut acc, mut nacc, mut bits) = (0u64, 0, 0);
            let mut d = d0;
            for _ in 0..1 + rng.intn(max_ops) {
                let mut t = rng.intn(4) as u8;
                if wide && rng.intn(16) > 0 {
                    t &= !1;
                }
                if t & 2 != 0 {
                    let mut n = 1 + rng.intn(MAX_LEN + 1);
                    ll.push(n as u8);
                    if n > MAX_LEN {
                        let e = rng.intn(256);
                        esc.push(e as u8);
                        n += e;
                    }
                    d += n;
                }
                if t & 1 == 0 {
                    let sym = if wide {
                        168 + rng.intn(8)
                    } else {
                        1 + rng.intn(175)
                    };
                    offc.push(sym as u8);
                    let ot = OFF_TAB[sym];
                    let nb = (ot >> 24) as usize;
                    let r = rng.uint64() & ((1 << nb) - 1);
                    last = (ot & 0xffffff) as usize + r as usize;
                    acc |= r << nacc;
                    (nacc, bits) = (nacc + nb, bits + nb);
                    while nacc >= 8 {
                        raw.push(acc as u8);
                        acc >>= 8;
                        nacc -= 8;
                    }
                }
                let mut ml = rng.intn(MAX_LEN + 2);
                if ml > MAX_LEN {
                    let r = if flat {
                        last.max(MAX_LEN).min(d)
                    } else {
                        last.min(d)
                    };
                    if r < MAX_LEN {
                        ml = MAX_LEN;
                    } else {
                        let e = rng.intn(256);
                        esc.push(e as u8);
                        d += e;
                    }
                }
                d += ml;
                tok.push(t | (ml << 2) as u8);
            }
            raw.extend_from_slice(&[acc as u8, 0, 0, 0, 0, 0, 0, 0]);
            let n_tok = tok.len();
            let (n_ll, n_off, n_esc) = (ll.len(), offc.len(), esc.len());
            for s in [&mut tok, &mut ll, &mut offc, &mut esc] {
                s.resize(s.len() + PAD, 0);
            }
            rng.read(&mut lits);
            let c = Chunk {
                tok: &tok,
                ll: &ll,
                offc: &offc,
                esc: &esc,
                raw: &raw,
                n_tok,
                n_ll,
                n_off,
                n_esc,
                bits,
                delta,
                flat,
            };
            let n = 1 + rng.intn(n_tok);
            let (mut sw, mut sg) = (st0, st0);
            if flat {
                test_flat(&mut want, &lits, &c, n, &mut sw, exec_flat_go);
                test_flat(
                    &mut got,
                    &lits,
                    &c,
                    n,
                    &mut sg,
                    // SAFETY: tier comes from simd_tiers, the counts above are what the tokens
                    // read, and want, got and lits have room past the operations.
                    |dst, lits, c, k0, k, cur, st| unsafe {
                        tier.exec_flat(dst, lits, c, k0, k, cur, st)
                    },
                );
            } else {
                exec_go(&mut want, &lits, &c, n, &mut sw);
                // SAFETY: as above.
                unsafe { tier.exec(&mut got, &lits, &c, n, &mut sg) };
            }
            assert!(
                sw == sg && want[d0..sw[0]] == got[d0..sw[0]],
                "{tier:?} case {i} ({n} of {n_tok} operations, delta {delta}, flat {flat}, start {st0:?}): state {sg:?}, want {sw:?}"
            );
            let end = sw[0].max(sg[0]) + SLACK + 2 * MAX_LEN;
            want[d0..end].copy_from_slice(&base[d0..end]);
            got[d0..end].copy_from_slice(&base[d0..end]);
        }
    });
}

#[test]
fn simd_stats() {
    simd_tiers(|tier| {
        let mut rng = Rng(7);
        // Miri runs only the safe Go stats; 200 streams take each kind 50 times.
        for i in 0..if cfg!(miri) { 200 } else { 20000 } {
            let n = rng.intn(200);
            let b: Vec<u8> = (0..n)
                .map(|_| {
                    (match i % 4 {
                        0 => rng.intn(256),
                        1 => rng.intn(136),
                        2 => 1 + rng.intn(33),
                        _ => 1 + rng.intn(175),
                    }) as u8
                })
                .collect();
            let (mut out, mut n_lit, mut n_off, mut n_esc) = (0, 0, 0, 0);
            let (mut sum, mut n_ll_esc, mut nb) = (0, 0, 0);
            let (mut tok_ok, mut ll_ok, mut off_ok) = (true, true, true);
            for &v in &b {
                let v = v as usize;
                out += v >> 2;
                n_lit += (v >> 1) & 1;
                n_off += (v & 1) ^ 1;
                n_esc += usize::from(v >> 2 == 33);
                tok_ok &= v <= 135;
                sum += v;
                n_ll_esc += usize::from(v == 33);
                ll_ok &= (1..=33).contains(&v);
                nb += (OFF_TAB[v] >> 24) as usize;
                off_ok &= (1..=175).contains(&v);
            }
            let (o, l, f, e, ok) = tok_stats(tier, &b);
            assert!(
                ok == tok_ok && (!ok || (o, l, f, e) == (out, n_lit, n_off, n_esc)),
                "{b:x?}: tokens {o} {l} {f} {e} {ok}"
            );
            let (a, c, ok) = ll_stats(tier, &b);
            assert!(
                ok == ll_ok && (!ok || (a, c) == (sum, n_ll_esc)),
                "{b:x?}: literal lengths {a} {c} {ok}"
            );
            // The sum is exact for any bytes: escapes are summed with it.
            assert_eq!(a, sum, "{b:x?}: literal length sum");
            let (a, ok) = off_stats(tier, &b);
            assert!(
                ok == off_ok && (!ok || a == nb),
                "{b:x?}: offsets {a} {ok}, want {nb}"
            );
        }
    });
}

/// The stream checks ignore the bytes after a stream, and find a bad byte at every position.
#[test]
fn simd_stats_tail() {
    simd_tiers(|tier| {
        let tok = |b: &[u8]| tok_stats(tier, b).4;
        let ll = |b: &[u8]| ll_stats(tier, b).2;
        let off = |b: &[u8]| off_stats(tier, b).1;
        // 0xff is bad in every stream, and adds to every count. 1 is a repeat with no match,
        // a literal length and an offset symbol without raw bits.
        let mut buf = [0xffu8; 300];
        // Miri: the checks are quadratic in n; 40 bytes cross several 8-byte words.
        for n in 1..=if cfg!(miri) { 40 } else { 200 } {
            let b = &mut buf[..n];
            b.fill(1);
            let (out, n_lit, n_off, n_esc, tok_ok) = tok_stats(tier, b);
            let (sum, n_ll_esc, ll_ok) = ll_stats(tier, b);
            let (nb, off_ok) = off_stats(tier, b);
            assert!(
                tok_ok
                    && ll_ok
                    && off_ok
                    && out + n_lit + n_off + n_esc + n_ll_esc + nb == 0
                    && sum == n,
                "length {n}: tokens {out} {n_lit} {n_off} {n_esc} {tok_ok}, literal lengths {sum} {n_ll_esc} {ll_ok}, offsets {nb} {off_ok}"
            );
            for i in 0..n {
                b[i] = 0xff;
                assert!(
                    !tok(b) && !ll(b) && !off(b),
                    "length {n}: bad byte at {i} not found"
                );
                b[i] = 1;
            }
            // The edges of each range.
            let edges: [(&str, u8, u8, Check); 5] = [
                ("token", 135, 136, &tok),
                ("literal length", 33, 34, &ll),
                ("literal length", 1, 0, &ll),
                ("offset symbol", 175, 176, &off),
                ("offset symbol", 1, 0, &off),
            ];
            for (name, good, bad, ok) in edges {
                b.fill(good);
                assert!(ok(b), "length {n}: {name} {good} not accepted");
                for i in 0..n {
                    b[i] = bad;
                    assert!(!ok(b), "length {n}: {name} {bad} at {i} not found");
                    b[i] = good;
                }
            }
        }
    });
}
