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

//! Go's `FuzzDecodeSIMD` (`fuzz_test.go`) over seeds and seeded mutations of them; the cargo-fuzz
//! target `simd_decode` runs the same check on fuzzer input through the public API.

use super::block::go_fixtures;
use super::*;
use crate::block::simd::exec::simd_tiers;
use crate::guard::Buffer;
use crate::reference::decode_simd_block;

/// Larger blocks only cost memory: every check is per chunk or per literal block.
const MAX_N: usize = 1 << 20;

/// Decodes the SIMD block body, from its flags byte, with a valid header of `body.len()` plus
/// the 64 bytes the format requires plus extra, on every tier, and compares with the reference
/// decoder.
fn fuzz_decode_simd(extra: u32, body: &[u8]) {
    let n = body.len() + 4 + MIN_SAVING + extra as usize;
    if body.is_empty() || n > MAX_N {
        return;
    }
    let mut hdr = vec![0];
    put_uvarint(&mut hdr, BLOCK_TYPE << 24 | n as u64);
    let mut sg = Buffer::new(hdr.len() + body.len()).unwrap();
    let src = sg.slice(hdr.len() + body.len(), true);
    src[..hdr.len()].copy_from_slice(&hdr);
    src[hdr.len()..].copy_from_slice(body);
    let src = &*src;
    let want = decode_simd_block(src);
    // Output must not depend on what dst held before. A nonzero fill catches reads of unwritten
    // output unless it equals the wanted byte, so vary it between inputs.
    let fill = 0xfe ^ extra as u8;
    // Against the guard page after dst, writes past it fault. Against the one before, offsets
    // reaching back too far fault.
    let mut dg = Buffer::new(n).unwrap();
    let dst = dg.slice(n, body.len() & 1 == 0);
    simd_tiers(|tier, pv| {
        dst.fill(fill);
        let got = simd_header(&src[1..]).and_then(|(size, b)| {
            assert_eq!(size, n);
            decode_simd(dst, b)
        });
        assert_eq!(
            got.is_ok(),
            want.is_ok(),
            "{tier:?} pivco {pv:?}: err {got:?}, reference {want:?}"
        );
        if let Ok(w) = &want {
            assert!(
                dst[..] == w[..],
                "{tier:?} pivco {pv:?}: reference mismatch"
            );
        }
    });
}

/// Applies 1 to 4 random edits: bit flips, byte overwrites, cuts and short appends.
fn mutate(rng: &mut Rng, b: &[u8]) -> Vec<u8> {
    let mut d = b.to_vec();
    for _ in 0..1 + rng.intn(4) {
        if d.is_empty() {
            d.push(rng.uint64() as u8);
            continue;
        }
        let i = rng.intn(d.len());
        match rng.intn(4) {
            0 => d[i] ^= 1 << rng.intn(8),
            1 => d[i] = rng.uint64() as u8,
            2 => d.truncate(i),
            _ => d.extend((0..1 + rng.intn(8)).map(|_| rng.uint64() as u8)),
        }
    }
    d
}

/// A block of one chunk of the most operations, which the encoders don't write.
fn max_chunk_block() -> Vec<u8> {
    let mut ops = vec![op(1, 4, 0); MAX_CHUNK];
    ops[0].off = 4;
    let (e, _) = TestBuilder::default().build(&TestBlock {
        lits: b"abc".repeat(ops.len() / 3 + 1),
        ops,
        limit: 6,
        ..Default::default()
    });
    e.bytes()
}

/// Go's seeds are blocks of its encoder; the SIMD fixtures of the Go encoder stand in for them.
/// Go also adds [`max_chunk_block`].
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn fuzz_decode_simd_seeded() {
    let mut rng = Rng(0x5eed);
    let mutants = 30;
    let blocks = go_fixtures(MAX_N).into_iter().map(|(_, b)| b);
    for block in blocks.chain([max_chunk_block()]) {
        let (size, body) = simd_header(&block[1..]).unwrap();
        let extra = (size - body.len() - 4 - MIN_SAVING) as u32;
        fuzz_decode_simd(extra, body);
        for _ in 0..mutants {
            let extra = match rng.intn(4) {
                0 => rng.intn(256) as u32,
                _ => extra,
            };
            fuzz_decode_simd(extra, &mutate(&mut rng, body));
        }
    }
}

/// Adds the seed Go's `FuzzDecodeSIMD` builds in code, see [`crate::fuzz_seeds`].
#[test]
#[ignore = "writes fuzz/corpus, see fuzz/RUNBOOK.md §1"]
fn write_fuzz_seeds() {
    let seed = crate::fuzz_seeds::simd_seed(&max_chunk_block()).unwrap();
    crate::fuzz_seeds::save("simd_decode", [seed]);
}
