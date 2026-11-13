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

//! Go's `fuzz_test.go` decode fuzzers over their seeds and seeded random mutations of them;
//! the cargo-fuzz target `pivco_decode` runs the same checks on fuzzer input.
//! FuzzRoundTripBlock, FuzzTable and FuzzLengths are in block.rs, table.rs and huffman.rs.

use crate::pivco::testutil::*;

/// Applies 1..4 random edits: bit flips, a byte overwrite, a cut or a short append.
fn mutate(x: &mut Xorshift64, b: &[u8], keep: usize) -> Vec<u8> {
    let mut d = b.to_vec();
    for _ in 0..1 + x.intn(4) {
        if d.len() <= keep {
            d.push(x.next() as u8);
            continue;
        }
        let i = keep + x.intn(d.len() - keep);
        match x.intn(4) {
            0 => d[i] ^= 1 << x.intn(8),
            1 => d[i] = x.next() as u8,
            2 => d.truncate(i),
            _ => d.extend((0..1 + x.intn(8)).map(|_| x.next() as u8)),
        }
    }
    d
}

/// Go FuzzDecodeBlock's seeds: [128 bytes of length nibbles][symbol count, u16][block].
fn decode_block_seeds() -> Vec<Vec<u8>> {
    corrupt_cases()
        .iter()
        .map(|cc| {
            let mut seed = vec![0u8; 128];
            for (s, &v) in cc.tb.lens().iter().enumerate() {
                seed[s / 2] |= v << (4 * (s & 1));
            }
            seed.extend_from_slice(&(cc.src.len() as u16).to_le_bytes());
            seed.extend_from_slice(&cc.blk);
            seed
        })
        .collect()
}

/// Go FuzzDecodeBlockFixed's seeds: [table][symbol count, u16][block].
fn decode_block_fixed_seeds() -> Vec<Vec<u8>> {
    corrupt_cases()
        .iter()
        .enumerate()
        .map(|(i, cc)| {
            [
                &[i as u8][..],
                &(cc.src.len() as u16).to_le_bytes(),
                &cc.blk,
            ]
            .concat()
        })
        .collect()
}

/// Go FuzzDecodeBlock.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn fuzz_decode_block_seeded() {
    let mut cs = coders();
    let mut x = Xorshift64(0xDEC0);
    for seed in decode_block_seeds() {
        fuzz_decode_block(&mut cs, &seed);
        for _ in 0..20 {
            fuzz_decode_block(&mut cs, &mutate(&mut x, &seed, 0));
        }
    }
}

/// Go FuzzDecodeBlockFixed.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn fuzz_decode_block_fixed_seeded() {
    let mut cs = coders();
    let tabs = fixed_tables();
    let mut x = Xorshift64(0xF1ED);
    for seed in decode_block_fixed_seeds() {
        fuzz_decode_block_fixed(&mut cs, &tabs, &seed);
        for _ in 0..40 {
            fuzz_decode_block_fixed(&mut cs, &tabs, &mutate(&mut x, &seed, 1));
        }
    }
}

/// Adds the seeds of Go's pivco fuzzers, after the mode byte of the target, see
/// [`crate::fuzz_seeds`].
#[test]
#[ignore = "writes fuzz/corpus, see fuzz/RUNBOOK.md §1"]
fn write_fuzz_seeds() {
    use crate::fuzz_seeds::save;
    let mode = |m: u8, seeds: Vec<Vec<u8>>| seeds.into_iter().map(move |s| [vec![m], s].concat());
    let decode = mode(0, decode_block_seeds()).chain(mode(1, decode_block_fixed_seeds()));
    save("pivco_decode", decode);
    save("pivco_roundtrip", super::block::round_trip_block_seeds());
    let lengths = super::huffman::LENGTHS_SEEDS.map(<[u8]>::to_vec).to_vec();
    save(
        "pivco_table",
        mode(0, super::table::table_seeds()).chain(mode(1, lengths)),
    );
}
