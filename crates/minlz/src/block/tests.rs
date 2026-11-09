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

//! Block-level integration tests, ported from `minlz_test.go`.

use super::*;
use crate::Error;

/// Helper: encode + decode at a given level and compare.
fn roundtrip_at(input: &[u8], level: Level) -> Result<(), String> {
    let mut enc = Vec::new();
    encode(&mut enc, input, level).map_err(|e| format!("encode: {e:?}"))?;
    let mut dec = Vec::new();
    decode(&mut dec, &enc).map_err(|e| format!("decode: {e:?}"))?;
    if dec != input {
        return Err(format!(
            "mismatch: got {} bytes, want {} bytes",
            dec.len(),
            input.len()
        ));
    }
    Ok(())
}

/// Run a round-trip at levels 1-7, as Go's `roundtrip`, and at -1.
///
/// Under miri, L3 and the SIMD levels are skipped: L3's only unsafe code
/// (`load32`/`load64`) is shared with L1+L2, while the L3 entry zeros 1.25M
/// `u64`s of thread-local tables on every call — the dominant cost of miri
/// runs on this crate.  Native builds hit every level.
fn roundtrip(input: &[u8]) -> Result<(), String> {
    let levels: &[Level] = if cfg!(miri) {
        &[Level::SuperFast, Level::Fastest, Level::Balanced]
    } else {
        &[
            Level::SuperFast,
            Level::Fastest,
            Level::Balanced,
            Level::Smallest,
            Level::SuperFastSimd,
            Level::FastestSimd,
            Level::BalancedSimd,
            Level::SmallestSimd,
        ]
    };
    for &level in levels {
        roundtrip_at(input, level).map_err(|e| format!("lvl {level:?}: {e}"))?;
    }
    Ok(())
}

/// Ports `TestMaxEncodedLen` from `minlz_test.go:45`.
#[test]
fn max_encoded_len_matches_go() {
    assert_eq!(max_encoded_len(0), Some(1));
    assert_eq!(max_encoded_len(1 << 5), Some((1 << 5) + 2));
    assert_eq!(max_encoded_len(MAX_BLOCK_SIZE), Some(MAX_BLOCK_SIZE + 2));
    assert_eq!(max_encoded_len(u32::MAX as usize), None);
    let sweep = if cfg!(miri) { 64 } else { 4096 };
    for i in 1..=sweep {
        assert_eq!(max_encoded_len(i), Some(i + 2));
    }
}

#[test]
fn level_discriminants_match_go() {
    for v in [-1, 1, 2, 3, 4, 5, 6, 7, 260, 261, 262] {
        let l = Level::try_from(v).unwrap();
        assert_eq!(l as i32, v, "{l:?}");
        assert_eq!(i32::from(l), v, "{l:?}");
    }
}

#[test]
fn level_is_simd() {
    for level in ALL_LEVELS {
        assert_eq!(level.is_simd(), i32::from(level) >= 4, "{level:?}");
    }
}

/// Ports `TestEmpty` from `minlz_test.go:199`.
#[test]
fn empty() {
    roundtrip(&[]).expect("empty roundtrip");
}

/// Ports `TestSmallCopy` from `minlz_test.go:205`.
#[test]
fn small_copy() {
    for i in 0..32 {
        let mut s = String::from("aaaa");
        s.push_str(&"b".repeat(i));
        s.push_str("aaaabbbb");
        roundtrip(s.as_bytes()).unwrap_or_else(|e| panic!("i={i}: {e}"));
    }
}

/// Cap iteration count under miri so tests still finish in reasonable time.
/// (Miri is 50-100× slower than native.)
fn cap(n: usize) -> usize {
    if cfg!(miri) { n.min(512) } else { n }
}

/// Ports `TestSmallRand` from `minlz_test.go:218`.
#[test]
fn small_rand() {
    let mut rng = Xor64::new(1);
    let mut n = 1;
    while n < cap(20_000) {
        let buf: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
        roundtrip(&buf).unwrap_or_else(|e| panic!("n={n}: {e}"));
        n += 23;
    }
}

/// Ports `TestSmallRegular` from `minlz_test.go:231`.
#[test]
fn small_regular() {
    let mut n = 1;
    while n < cap(20_000) {
        let buf: Vec<u8> = (0..n).map(|i| ((i % 10) as u8) + b'a').collect();
        roundtrip(&buf).unwrap_or_else(|e| panic!("n={n}: {e}"));
        n += 23;
    }
}

/// Ports `TestSmallRepeat` from `minlz_test.go:243`.
#[test]
fn small_repeat() {
    let mut n = 1;
    while n < cap(20_000) {
        let mut buf = vec![0u8; n];
        let half = n / 2;
        for (i, b) in buf[..half].iter_mut().enumerate() {
            *b = ((i * 255) / n) as u8;
        }
        for (i, b) in buf[half..].iter_mut().enumerate() {
            *b = ((i % 10) as u8) + b'a';
        }
        roundtrip(&buf).unwrap_or_else(|e| panic!("n={n}: {e}"));
        n += 23;
    }
}

/// Ports `TestInvalidVarint` from `minlz_test.go:258`.
///
/// In the Rust port (no Snappy/S2 fallback), the first three inputs all have
/// non-zero first bytes and are rejected outright with `Corrupt`.  We add
/// equivalent inputs prefixed with `\x00` so the varint logic itself is also
/// exercised.
#[test]
fn invalid_varint() {
    let cases: &[&[u8]] = &[
        // Inputs from Go (no leading 0 — rejected as not-minlz).
        b"\xff",
        b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00",
        b"\x80\x80\x80\x80\x10",
        // MinLZ-prefixed equivalents.
        b"\x00\xff",
        b"\x00\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00",
        b"\x00\x80\x80\x80\x80\x10",
    ];
    for input in cases {
        let err = decoded_len(input).expect_err("expected an error");
        assert!(matches!(err, Error::Corrupt(_)), "input={input:?}");
    }
}

/// Decoded zero-byte block accepts `[0x00]` and refuses `[]`.
#[test]
fn decoded_len_minimal() {
    assert_eq!(decoded_len(&[0x00]).unwrap(), 0);
    assert!(matches!(decoded_len(&[]).unwrap_err(), Error::Corrupt(_)));
    // `[0x00, 0x00]` declares length 0 with no body — Go rejects this.
    assert!(matches!(
        decoded_len(&[0x00, 0x00]).unwrap_err(),
        Error::Corrupt(_)
    ));
    // `[0x00, 0x00, 'A']` decodes to a 1-byte literal block.
    assert_eq!(decoded_len(&[0x00, 0x00, b'A']).unwrap(), 1);
}

/// 0-byte and 1-byte inputs survive the round-trip at every level.
#[test]
fn tiny_roundtrips() {
    roundtrip(b"").unwrap();
    roundtrip(b"a").unwrap();
    roundtrip(b"ab").unwrap();
    roundtrip(b"abc").unwrap();
}

/// Mid-size deterministic round-trip exercising L1's 64K hash table.
#[test]
#[cfg_attr(miri, ignore = "32 KiB input × 3 levels is impractical under miri")]
fn medium_64k_roundtrip() {
    // 32 KiB of period-7 data — should compress heavily.
    let mut buf = Vec::with_capacity(32 << 10);
    while buf.len() < 32 << 10 {
        buf.extend_from_slice(b"foobar ");
    }
    roundtrip(&buf).unwrap();
}

/// Large round-trip exercising L1's >64K hash table.
#[test]
#[cfg_attr(miri, ignore = "128 KiB input × 3 levels is impractical under miri")]
fn large_big_roundtrip() {
    // 128 KiB of period-13 data.
    let mut buf = Vec::with_capacity(128 << 10);
    while buf.len() < 128 << 10 {
        buf.extend_from_slice(b"the quick brown fox ");
    }
    roundtrip(&buf).unwrap();
}

/// English pangram concatenated 100x — input that the Go noasm encoder
/// compresses heavily (≈99% reduction).  Used in two tests below.
const PANGRAM_PARTS: &[&[u8]] = &[
    b"The quick brown fox jumps over the lazy dog. ",
    b"Pack my box with five dozen liquor jugs. ",
    b"How quickly daft jumping zebras vex. ",
    b"Sphinx of black quartz, judge my vow. ",
    b"The five boxing wizards jump quickly.",
];

fn pangram_x(times: usize) -> Vec<u8> {
    let pat_len: usize = PANGRAM_PARTS.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(pat_len * times);
    for _ in 0..times {
        for p in PANGRAM_PARTS {
            out.extend_from_slice(p);
        }
    }
    out
}

/// Encoder must actually compress varied repetitive input (parity vs Go noasm).
///
/// Single-period input is covered by `l1_repeated_fragment_compresses`.
#[test]
#[cfg_attr(miri, ignore = "20 KiB input is impractical under miri")]
fn encoder_compresses_natural_repetition() {
    let buf = pangram_x(100);
    let mut enc = Vec::new();
    encode(&mut enc, &buf, Level::Fastest).unwrap();
    assert!(
        enc.len() < buf.len() / 10,
        "expected heavy compression, got enc={} src={}",
        enc.len(),
        buf.len()
    );
}

/// Differential: decode blocks produced by the Go `noasm` encoder
/// (`encode_l1.go:encodeBlockGo` / `encodeBlockGo64K`) and verify the
/// plaintext matches.  Fixtures captured with `go run -tags=noasm`.
#[test]
#[cfg_attr(miri, ignore = "200 KiB pangram fixture is impractical under miri")]
fn decode_go_noasm_fixtures() {
    let pangram_100 = pangram_x(100);
    let pangram_1000 = pangram_x(1000);
    let fixtures: &[(&[u8], &str)] = &[
        (
            &pangram_100,
            "00d89a01e89a54686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672e205061636b206d7920626f782077697468206669766520646f7a656e206c6971756f72206a7567732e20486f7720717569636b6c792064616674206a756d70696e67207a6562726173207665782e20537068696e78206f6620626c61636b2071756172747a2c206a75646765206d7920766f772e20546865206669766520626f78696e672077697a617264730529492ef386006c792ef4634c2869636b6c792e",
        ),
        (
            &pangram_1000,
            "00f08a0ce88354686520717569636b2062726f776e20666f78206a756d7073206f76657220746865206c617a7920646f672e205061636b206d7920626f782077697468206669766520646f7a656e206c6971756f72206a7567732e20486f7720717569636b6c792064616674206a756d70696e67207a6562726173207665782e20537068696e78206f6620626c61636b2071756172747a2c206a75646765206d7920766f772e200128b86669766520626f78696e672077697a61726473206a756d70d118e386002efc7b04032869636b6c792e",
        ),
    ];
    for &(plain, hex_enc) in fixtures {
        let enc = hex_decode(hex_enc);
        let mut dec = Vec::new();
        decode(&mut dec, &enc).expect("decode go fixture");
        assert_eq!(dec.len(), plain.len(), "decoded length mismatch");
        assert_eq!(&dec[..], plain, "decoded contents mismatch");
    }
}

/// Differential: decode blocks produced by the Go ASM encoder.  The ASM
/// path uses different heuristics than `encodeBlockGo`, so this exercises
/// the decoder against a *different* valid wire format than our own
/// encoder produces.
#[test]
#[cfg_attr(miri, ignore = "32 KiB foobar fixture is impractical under miri")]
fn decode_go_asm_fixtures() {
    let foobar = repeat_bytes(b"foobar ", 32 << 10);
    let abc = repeat_bytes(b"abc", 16 << 10);
    let alpha = repeat_bytes(b"abcdefghijklmnopqrstuvwxyz", 1024);
    let fixtures: &[(&[u8], &str)] = &[
        (&foobar, "0080800230666f6f62617220b901f4c97f"),
        (&abc, "0080800110616263b900f4cd3f"),
        (
            &alpha,
            "008008c86162636465666768696a6b6c6d6e6f707172737475767778797a7906f4b603",
        ),
    ];
    for &(plain, hex_enc) in fixtures {
        let enc = hex_decode(hex_enc);
        let mut dec = Vec::new();
        decode(&mut dec, &enc).expect("decode go fixture");
        assert_eq!(dec.len(), plain.len(), "decoded length mismatch");
        assert_eq!(&dec[..], plain, "decoded contents mismatch");
    }
}

fn repeat_bytes(pat: &[u8], n: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(n + pat.len());
    while v.len() < n {
        v.extend_from_slice(pat);
    }
    v.truncate(n);
    v
}

fn hex_decode(s: &str) -> Vec<u8> {
    assert_eq!(s.len() % 2, 0);
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex"))
        .collect()
}

/// Ports `TestDecodeBlockOverlapping` from `decode_test.go:91`.
/// Tests RLE-style patterns (overlapping forward copies).
#[test]
#[cfg_attr(
    miri,
    ignore = "1 KiB-pattern roundtrips × 5 × 3 levels too slow for miri"
)]
fn decode_block_overlapping() {
    let patterns: &[Vec<u8>] = &[
        repeat_bytes(b"a", 1000),
        repeat_bytes(b"ab", 1000),
        repeat_bytes(b"abc", 999),
        repeat_bytes(b"abcd", 1000),
        repeat_bytes(b"abcdefg", 1001),
    ];
    for (i, pat) in patterns.iter().enumerate() {
        roundtrip(pat).unwrap_or_else(|e| panic!("pattern {i}: {e}"));
    }
}

/// Ports `TestDecodeBlockLongOffsets` from `decode_test.go:123`.
/// Tests Copy2/Copy3 offsets.
#[test]
#[cfg_attr(miri, ignore = "200 KiB random input too slow for miri")]
fn decode_block_long_offsets() {
    let size = 200_000;
    let mut data = vec![0u8; size];
    let mut rng = Xor64::new(123);
    for b in data.iter_mut() {
        *b = rng.next() as u8;
    }
    let pattern = b"REPEATED_PATTERN_DATA";
    for &off in &[100usize, 1000, 10_000, 65_000, 100_000] {
        if off + pattern.len() < size {
            data[off..off + pattern.len()].copy_from_slice(pattern);
            if off * 2 + pattern.len() < size {
                data[off * 2..off * 2 + pattern.len()].copy_from_slice(pattern);
            }
        }
    }
    roundtrip(&data).unwrap();
}

/// Ports `TestEncodeNoiseThenRepeats` from `minlz_test.go:783`.
/// First half incompressible, second half compressible — total < 75 % of input.
#[test]
#[cfg_attr(miri, ignore = "256 KiB+ inputs too slow for miri")]
fn encode_noise_then_repeats() {
    for &orig_len in &[256 * 1024usize, 2048 * 1024] {
        let mut src = vec![0u8; orig_len];
        let mut rng = Xor64::new(1);
        let half = orig_len / 2;
        for b in src[..half].iter_mut() {
            *b = rng.next() as u8;
        }
        for (i, b) in src[half..].iter_mut().enumerate() {
            *b = (i >> 8) as u8;
        }
        let mut enc = Vec::new();
        encode(&mut enc, &src, Level::Fastest).unwrap();
        let want_less_than = orig_len * 3 / 4;
        assert!(
            enc.len() < want_less_than,
            "orig_len={orig_len}: got {} encoded bytes, want less than {want_less_than}",
            enc.len()
        );
    }
}

/// Ports `TestEncodeHuge` from `encode_test.go:26`.
/// Encode + decode at MAX_BLOCK_SIZE for every level.
#[test]
#[cfg_attr(miri, ignore = "8 MiB input × 4 levels is impractical under miri")]
fn encode_huge() {
    let data = vec![0u8; MAX_BLOCK_SIZE];
    for &level in &[
        Level::SuperFast,
        Level::Fastest,
        Level::Balanced,
        Level::Smallest,
    ] {
        let mut enc = Vec::new();
        encode(&mut enc, &data, level).unwrap();
        let mut dec = Vec::new();
        decode(&mut dec, &enc).unwrap();
        assert_eq!(dec.len(), data.len(), "level {level:?}");
        assert!(enc.len() <= max_encoded_len(data.len()).unwrap());
    }
}

/// Ports `TestSizes` from `encode_test.go:75`.
/// `emit_copy_size(offset, length)` must equal `emit_copy(...)`'s output length
/// for every length 4..MAX_BLOCK_SIZE at three representative offsets.  Same
/// for `emit_repeat_size`.
#[test]
#[cfg_attr(miri, ignore = "MAX_BLOCK_SIZE sweep is impractical under miri")]
fn emit_size_predictors() {
    // Loop ranges trimmed for speed; full sweep covered by TestSizes in CI.
    let offsets = [10usize, 4000, 70_000];
    let mut tmp = vec![0u8; 16];
    let lengths: Vec<usize> = (4..16).chain((16..MAX_BLOCK_SIZE).step_by(257)).collect();
    for &offset in &offsets {
        for &length in &lengths {
            // emit_copy actual size
            let n = super::emit::emit_copy(&mut tmp, offset, length);
            let predicted = super::emit::emit_copy_size(offset, length);
            assert_eq!(
                n, predicted,
                "emit_copy_size(offset={offset}, length={length}): got {predicted}, want {n}"
            );
        }
    }
    // emit_repeat
    for length in 1..16 {
        let n = super::emit::emit_repeat(&mut tmp, length);
        let predicted = super::emit::emit_repeat_size(length);
        assert_eq!(
            n, predicted,
            "emit_repeat_size({length}): got {predicted}, want {n}"
        );
    }
    for length in (16..MAX_BLOCK_SIZE).step_by(257) {
        let n = super::emit::emit_repeat(&mut tmp, length);
        let predicted = super::emit::emit_repeat_size(length);
        assert_eq!(
            n, predicted,
            "emit_repeat_size({length}): got {predicted}, want {n}"
        );
    }
}

/// Try to locate `testdata/<name>` in the upstream Go repo.  Honours the
/// `MINLZ_TESTDATA` env override so CI can point at a different path.
/// Returns `None` if not found (tests then skip with an `eprintln`).
fn locate_testdata(name: &str) -> Option<Vec<u8>> {
    let candidates: Vec<std::path::PathBuf> = {
        let mut v = Vec::new();
        if let Ok(d) = std::env::var("MINLZ_TESTDATA") {
            v.push(std::path::PathBuf::from(d).join(name));
        }
        // crates/minlz/  -> minlz-rs/  -> minlz/ (Go repo root)
        let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        v.push(manifest.join("../../../testdata").join(name));
        v.push(manifest.join("../../testdata").join(name));
        v
    };
    for p in &candidates {
        if let Ok(b) = std::fs::read(p) {
            return Some(b);
        }
    }
    None
}

/// Ports the block-decode portion of `TestDecodeGoldenInput`
/// (`minlz_test.go:635`).  Reads the checked-in `Mark.Twain-Tom.Sawyer.txt.mzb`
/// and decodes against the plaintext.
#[test]
#[cfg_attr(
    miri,
    ignore = "filesystem I/O + 14 KiB decode is impractical under miri"
)]
fn decode_golden_mzb_block() {
    let Some(plain) = locate_testdata("Mark.Twain-Tom.Sawyer.txt") else {
        eprintln!("skip: testdata Mark.Twain-Tom.Sawyer.txt not found");
        return;
    };
    let Some(mzb) = locate_testdata("Mark.Twain-Tom.Sawyer.txt.mzb") else {
        eprintln!("skip: testdata Mark.Twain-Tom.Sawyer.txt.mzb not found");
        return;
    };
    let mut dec = Vec::new();
    decode(&mut dec, &mzb).expect("decode mzb");
    assert_eq!(dec.len(), plain.len());
    assert_eq!(&dec[..], &plain[..]);
}

/// Block round-trip across all levels on a realistic English-text input
/// (the Twain corpus).  This is the block-only subset of Go's
/// `TestRoundtrips` (`minlz_test.go:1477`).
#[test]
#[cfg_attr(miri, ignore = "14 KiB Twain × 3 levels is impractical under miri")]
fn roundtrip_twain_corpus() {
    let Some(plain) = locate_testdata("Mark.Twain-Tom.Sawyer.txt") else {
        eprintln!("skip: testdata Mark.Twain-Tom.Sawyer.txt not found");
        return;
    };
    for &level in &[Level::Fastest, Level::Balanced, Level::Smallest] {
        let mut enc = Vec::new();
        encode(&mut enc, &plain, level).unwrap();
        assert!(
            enc.len() < plain.len() * 4 / 5,
            "level {level:?}: ratio {} too poor",
            enc.len() as f64 / plain.len() as f64
        );
        let mut dec = Vec::new();
        decode(&mut dec, &enc).unwrap();
        assert_eq!(dec, plain, "level {level:?}");
    }
}

/// Ports the `copy1` sub-test of `TestEmitters` (`encode_test.go:108`).
/// Sweep every (offset, length) for Copy1 and parse the output back.
#[test]
#[cfg_attr(miri, ignore = "281K iterations too slow for miri")]
fn emitters_copy1() {
    let mut tmp = [0u8; 16];
    for off in 1..=1024usize {
        for l in 4..=273usize {
            let n = super::emit::emit_copy(&mut tmp, off, l);
            let bytes = &tmp[..n];
            // Tag in low 2 bits.
            assert_eq!(bytes[0] & 3, 0b01, "tag at off={off} len={l}");
            let mut length = ((bytes[0] >> 2) & 15) as usize;
            let offset = ((u16::from_le_bytes([bytes[0], bytes[1]]) >> 6) as usize) + 1;
            let consumed = if length == 15 {
                length = bytes[2] as usize + 18;
                3
            } else {
                length += 4;
                2
            };
            assert_eq!(length, l, "length parse at off={off} expected={l}");
            assert_eq!(offset, off, "offset parse at off={off} length={l}");
            assert_eq!(consumed, n, "byte count at off={off} length={l}");
        }
    }
}

/// Ports the `copy2-lits` sub-test of `TestEmitters` (`encode_test.go:195`).
/// 1-4 fused literals at every copy length 4..=11.
#[test]
#[cfg_attr(miri, ignore = "large offset sweep too slow for miri")]
fn emitters_copy2_lits() {
    let mut tmp = [0u8; 16];
    let mut lits: Vec<u8> = vec![1];
    while lits.len() <= 4 {
        for off in (super::format::MIN_COPY2_OFFSET..=super::format::MAX_COPY2_OFFSET).step_by(997)
        {
            for l in 4..=11usize {
                let n = super::emit::emit_copy_lits2(&mut tmp, &lits, off, l);
                let bytes = &tmp[..n];
                assert_eq!(bytes[0] & 3, 0b11, "tag");
                assert_eq!(bytes[0] & 4, 0, "copy3 bit set unexpectedly");
                let value = bytes[0] >> 3;
                let offset = (u16::from_le_bytes([bytes[1], bytes[2]]) as usize)
                    + super::format::MIN_COPY2_OFFSET;
                let lit_len = ((value & 3) as usize) + 1;
                let copy_len = ((value >> 2) as usize) + 4;
                assert_eq!(copy_len, l, "copy_len at off={off} l={l}");
                assert_eq!(offset, off, "offset at off={off} l={l}");
                assert_eq!(lit_len, lits.len());
                assert_eq!(n, 3 + lits.len());
                assert_eq!(&bytes[3..], &lits[..]);
            }
        }
        lits.push((lits.len() + 1) as u8);
    }
}

/// Ports the `copy3` sub-test of `TestEmitters` (`encode_test.go:358`).
/// Tests `emit_copy_lits3` / `emit_copy` for Copy3 range.
#[test]
#[cfg_attr(miri, ignore = "large length sweep too slow for miri")]
fn emitters_copy3() {
    let mut tmp = [0u8; 16];
    let mut lits: Vec<u8> = Vec::new();
    while lits.len() <= 3 {
        let mut off = super::format::MAX_COPY2_OFFSET + 1;
        while off <= super::format::MAX_COPY3_OFFSET {
            // Length sweep with geometric growth.
            let mut l = 4usize;
            while l <= 1 << 16 {
                let n = if !lits.is_empty() {
                    super::emit::emit_copy_lits3(&mut tmp, &lits, off, l)
                } else {
                    super::emit::emit_copy(&mut tmp, off, l)
                };
                let bytes = &tmp[..n];
                assert_eq!(
                    bytes[0] & 7,
                    super::format::TAG_COPY3,
                    "tag at off={off} l={l}"
                );
                let val = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                let length_field = ((val >> 5) & 63) as usize;
                let offset = ((val >> 11) as usize) + super::format::MIN_COPY3_OFFSET;
                let (length, consumed) = match length_field {
                    0..=60 => (length_field + 4, 4),
                    61 => (bytes[4] as usize + 64, 5),
                    62 => (u16::from_le_bytes([bytes[4], bytes[5]]) as usize + 64, 6),
                    63 => (
                        ((bytes[4] as usize)
                            | ((bytes[5] as usize) << 8)
                            | ((bytes[6] as usize) << 16))
                            + 64,
                        7,
                    ),
                    _ => panic!("bad length field"),
                };
                let nlits = ((val >> 3) & 3) as usize;
                assert_eq!(nlits, lits.len(), "lit count at off={off} l={l}");
                if nlits > 0 {
                    assert_eq!(&bytes[consumed..consumed + nlits], &lits[..]);
                }
                assert_eq!(length, l, "length at off={off} l={l}");
                assert_eq!(offset, off, "offset at off={off} l={l}");
                assert_eq!(consumed + nlits, n, "total bytes at off={off} l={l}");
                l = ((l as f64) * 1.5) as usize + 1;
            }
            off = off.saturating_mul(2);
            if off == 0 {
                break;
            }
        }
        lits.push((lits.len() + 1) as u8);
    }
}

/// Smoke test: decode every file in the cargo-fuzz seed corpus directory
/// (if present).  Equivalent to running `decode_arbitrary` for one
/// iteration per seed — must not panic or read OOB on any input, and SIMD
/// blocks must decode as the reference decoder does.
#[test]
#[cfg_attr(
    miri,
    ignore = "thousands of seeds — fuzz the focused miri_decoder test instead"
)]
fn decode_arbitrary_corpus_smoke() {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest.join("fuzz/corpus/decode_arbitrary");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skip: corpus not extracted at {}", dir.display());
        eprintln!("  see crates/minlz/fuzz/RUNBOOK.md §1 to populate it");
        return;
    };
    let mut count = 0usize;
    let mut buf = Vec::with_capacity(1 << 20);
    for ent in entries.flatten() {
        let Ok(bytes) = std::fs::read(ent.path()) else {
            continue;
        };
        buf.clear();
        // Any error is acceptable; a panic is not.
        let res = decode(&mut buf, &bytes);
        let simd = bytes.first() == Some(&0)
            && format::get_uvarint(&bytes[1..]).is_some_and(|(v, _)| v >= 1 << 24);
        if simd {
            let want = crate::reference::decode_simd_block(&bytes);
            assert_eq!(res.is_ok(), want.is_ok(), "{:?}", ent.path());
            assert!(want.is_err() || buf == want.unwrap(), "{:?}", ent.path());
        }
        count += 1;
    }
    eprintln!("decode_arbitrary_corpus_smoke: scanned {count} seeds");
    assert!(
        count > 0,
        "corpus directory exists but had no readable files"
    );
}

/// Smoke test: round-trip every file in the cargo-fuzz roundtrip seed
/// corpus at every level.  Validates the encoder doesn't produce
/// uncompiable / un-decodable output for any seed.
#[test]
#[cfg_attr(
    miri,
    ignore = "thousands of seeds — fuzz the focused miri_decoder test instead"
)]
fn roundtrip_corpus_smoke() {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest.join("fuzz/corpus/roundtrip");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        eprintln!("skip: corpus not extracted at {}", dir.display());
        return;
    };
    let mut enc = Vec::new();
    let mut dec = Vec::new();
    let mut count = 0usize;
    for ent in entries.flatten() {
        let Ok(bytes) = std::fs::read(ent.path()) else {
            continue;
        };
        if bytes.len() > MAX_BLOCK_SIZE {
            continue;
        }
        for &level in &[
            Level::SuperFast,
            Level::Fastest,
            Level::Balanced,
            Level::Smallest,
        ] {
            enc.clear();
            encode(&mut enc, &bytes, level).expect("encode");
            dec.clear();
            decode(&mut dec, &enc).expect("decode");
            assert_eq!(dec, bytes, "level {level:?} seed {:?}", ent.path());
        }
        count += 1;
    }
    eprintln!("roundtrip_corpus_smoke: scanned {count} seeds across 4 levels");
    assert!(
        count > 0,
        "corpus directory exists but had no readable files"
    );
}

/// Focused miri target: exercise the decoder's unsafe paths on a small,
/// curated set of inputs that hits each tag class (literal, repeat,
/// copy1, copy2, copy2-fused, copy3) at least once.  Designed to run
/// fast enough under miri to give meaningful UB-detection coverage in
/// a few minutes.  Runs natively too — just much faster.
#[test]
fn miri_decoder_unsafe_paths() {
    // Round-trip each small input at L0 and L1 (exercises every encoder unsafe
    // load/store and every decoder tag class via emit_copy / emit_literal
    // / emit_repeat).
    let inputs: &[&[u8]] = &[
        // Empty + tiny.
        b"",
        b"a",
        b"ab",
        b"abcdefghijklmnop",            // 16 bytes — boundary of the LZ4 overshoot.
        b"abcdefghijklmnopq",           // 17 bytes — one past the boundary.
        // Repeated short pattern (exercises copy1 + small offsets).
        b"abcabcabcabcabcabcabcabcabcabc",
        // Mixed: prefix + RLE (exercises forward_copy_ptr — overlap path).
        b"prefix aaaaaaaaaaaaaaaa suffix",
        // Long-ish English (exercises copy2 + emit_literal extension).
        b"the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.",
    ];
    // L0 and L1 alone exercise every encoder unsafe load/store; L2/L3 add
    // table-zeroing cost (esp. L3's 1.25M `u64`s) without new UB coverage.
    let levels: &[Level] = if cfg!(miri) {
        &[Level::SuperFast, Level::Fastest]
    } else {
        &[
            Level::SuperFast,
            Level::Fastest,
            Level::Balanced,
            Level::Smallest,
        ]
    };
    for &input in inputs {
        for &level in levels {
            let mut enc = Vec::new();
            encode(&mut enc, input, level).expect("encode");
            let mut dec = Vec::new();
            decode(&mut dec, &enc).expect("decode");
            assert_eq!(
                dec.as_slice(),
                input,
                "level {level:?} on {} bytes",
                input.len()
            );
        }
    }

    // Decode-only: known Go-asm fixtures that hit copy3 + extended-length
    // tag paths the round-trip alone can miss.
    let alpha = repeat_bytes(b"abcdefghijklmnopqrstuvwxyz", 1024);
    let enc = hex_decode("008008c86162636465666768696a6b6c6d6e6f707172737475767778797a7906f4b603");
    let mut dec = Vec::new();
    decode(&mut dec, &enc).expect("decode alpha fixture");
    assert_eq!(dec.len(), alpha.len());
    assert_eq!(&dec[..], &alpha[..]);
}

/// Tiny RNG — same as Go's `math/rand` would do for these test sizes is
/// not required; reproducibility is what we want.
struct Xor64(u64);
impl Xor64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x as u32
    }
}

/// An 8-byte marker at `p = 9` and at `q = p + MAX_COPY3_OFFSET + delta`.
/// The filler has a match every 128 bytes, so the L1 skip step stays small,
/// and a match that ends `gap` bytes before `q`, so the L1 search finds the
/// marker through candidate c0, c1 or c2 (gap 0, 1, 2).  Filler windows
/// whose hash collides with the marker's are scrubbed, so the table still
/// holds `p` when the search reaches `q`.  Extends Go's
/// `buildMaxOffsetBoundary` (minio/minlz#75), which places the match at gap 2.
fn max_offset_boundary_gap(seed: u64, delta: usize, gap: usize) -> Option<Vec<u8>> {
    use super::hash::hash6;
    use crate::block::format::MAX_COPY3_OFFSET as MAX;
    let p = 9;
    let q = p + MAX + delta;
    let n = q + 4096;
    let mut rng = Xor64::new(seed);
    let mut src: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
    for i in (2000..n - 128).step_by(128) {
        src.copy_within(i + 64 - 1000..i + 128 - 1000, i + 64);
    }
    src.copy_within(q - 1200 - gap..q - 1000 - gap, q - 200 - gap);
    if gap > 0 {
        src[q - gap] = src[q - gap - 1000] ^ 0xff;
    }
    // Keep the match at q from extending backwards.
    if src[q - 1] == src[p - 1] {
        src[q - 1] ^= 0x33;
        if gap == 0 {
            src[q - 1001] = src[q - 1];
        }
    }
    let marker = b"\xfeSeNtInL";
    src[p..p + 8].copy_from_slice(marker);
    src[q..q + 8].copy_from_slice(marker);

    let load = |s: &[u8], i: usize| u64::from_le_bytes(s[i..i + 8].try_into().unwrap());
    let h = hash6(load(&src, p), 15);
    let keep = |j: usize| (p..p + 8).contains(&j) || (q - 1210 - gap..q + 8).contains(&j);
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..=n - 8 {
            if i == p || i == q || hash6(load(&src, i), 15) != h {
                continue;
            }
            let j = if keep(i + 5) { i } else { i + 5 };
            if keep(j) {
                return None;
            }
            src[j] ^= 0x5a;
            changed = true;
        }
    }
    Some(src)
}

/// L1 must not use a match more than `MAX_COPY3_OFFSET` back: the 21-bit
/// Copy3 offset field would wrap and produce a corrupt block.  Covers
/// offsets `MAX_COPY3_OFFSET + 0..=3` through each of the three search
/// candidates, which are matched at `s`, `s + 1` and `s + 2`.
#[test]
#[cfg_attr(miri, ignore = "2 MiB inputs are impractical under miri")]
fn l1_max_copy3_offset_boundary() {
    for gap in 0..3 {
        for delta in 0..4 {
            for seed in 0..4 {
                let src = max_offset_boundary_gap(seed, delta, gap).expect("filler collides");
                let levels: &[Level] = if seed == 0 {
                    &[Level::Fastest, Level::Balanced, Level::Smallest]
                } else {
                    &[Level::Fastest]
                };
                for &level in levels {
                    roundtrip_at(&src, level).unwrap_or_else(|e| {
                        panic!("gap={gap} delta={delta} seed={seed} {level:?}: {e}")
                    });
                }
            }
        }
    }
}

/// Ports `TestEncodeLongMatch` (`encode_test.go`): a block that is one long
/// match is not stored. Sizes cover the 64K and the general encoders.
#[test]
fn encode_long_match() {
    // Miri: smaller, but still one size for the 64 KiB encoders and one for the general ones.
    let sizes = if cfg!(miri) {
        [2000, 70000]
    } else {
        [60000, 1 << 20]
    };
    for size in sizes {
        let data = repeat_bytes(b"abcdefghij", size);
        // Miri also runs the SIMD parsers over 64 KiB here, as no other Miri test does.
        let simd: &[Level] = if cfg!(miri) && size > 64 << 10 {
            &[
                Level::SuperFastSimd,
                Level::FastestSimd,
                Level::BalancedSimd,
            ]
        } else {
            &[]
        };
        for &level in [
            Level::SuperFast,
            Level::Fastest,
            Level::Balanced,
            Level::Smallest,
        ]
        .iter()
        .chain(simd)
        {
            let mut comp = Vec::new();
            encode(&mut comp, &data, level).unwrap();
            assert!(
                comp.len() <= 1024,
                "level {level:?}, size {size}: not compressed, got {} bytes",
                comp.len()
            );
            let mut dec = Vec::new();
            decode(&mut dec, &comp).unwrap();
            assert!(dec == data, "level {level:?}, size {size}: decode mismatch");
        }
    }
}

/// Miri: the >64 KiB L0, L1 and L2 encoders, which are separate code from their 64 KiB versions, on
/// segments from 16 seeds that repeat a few KiB back, right after each other, or with a changed
/// byte (a repeat offset), and a last segment that repeats the first, over 64 KiB back.
/// encode_long_match's single match skips most of their loops, and Miri ignores
/// l1_max_copy3_offset_boundary. The SIMD parsers share their code with the 64 KiB
/// versions, which simd_encode_short runs. 27 s under Miri without L0, which adds 30%; each
/// SIMD level would add 17 s.
#[cfg(miri)]
#[test]
fn big_encoders_mixed_miri() {
    let seg = |seed: u64, n: usize| {
        let mut g = Xor64::new(seed);
        (0..n).map(move |_| g.next() as u8)
    };
    let mut rng = Xor64::new(9);
    let mut data: Vec<u8> = seg(1 << 40, 64).collect();
    while data.len() < 70_000 {
        let r = rng.next();
        let n = 4 + r as usize % 90;
        // A quarter of the segments are new.
        let seed = if r >> 8 & 3 == 0 {
            u64::from(r) << 8
        } else {
            u64::from(r >> 10 & 15)
        };
        let at = data.len();
        data.extend(seg(seed, n));
        if r >> 16 & 7 == 0 {
            data[at + n / 2] ^= 0x55;
        }
    }
    data.extend(seg(1 << 40, 64));
    for level in [Level::SuperFast, Level::Fastest, Level::Balanced] {
        roundtrip_at(&data, level).unwrap_or_else(|e| panic!("{level:?}: {e}"));
    }
}

/// Port of Go's `buildMaxOffsetBoundary`: a hash1/hash2 match at offset
/// `MAX_COPY3_OFFSET + delta` (minio/minlz#75). Random bytes come from
/// [`Xor64`], not Go's `math/rand`.
pub(super) fn build_max_offset_boundary(seed: u64, delta: usize) -> Option<Vec<u8>> {
    use super::format::MAX_COPY3_OFFSET as M;
    use super::hash::hash6;
    let load64 = |b: &[u8], i: usize| u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
    let p = 9;
    let q = p + M + delta;
    let n = q + 4096;
    let mut rng = Xor64::new(seed);
    let mut src: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
    // Regular short matches keep nextEmit close, so the skip step stays small.
    for i in (2000..).step_by(128).take_while(|i| i + 128 < n) {
        src.copy_within(i + 64 - 1000..i + 128 - 1000, i + 64);
    }
    src.copy_within(q - 1202..q - 1002, q - 202);
    let marker = b"\xfeSeNtInL";
    src[p..p + 8].copy_from_slice(marker);
    src[q..q + 8].copy_from_slice(marker);
    src[q - 2] = src[q - 1002] ^ 0xff;
    src[q - 1] = src[p - 1] ^ 0xff;
    // Keep the marker's hash bucket holding p: scrub colliding filler windows.
    let hm = hash6(load64(&src, p), 15);
    let protected = |j: usize| (p..p + 8).contains(&j) || (q - 1210..q + 8).contains(&j);
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..=n - 8 {
            if i == p || i == q || hash6(load64(&src, i), 15) != hm {
                continue;
            }
            let j = if protected(i + 5) { i } else { i + 5 };
            if protected(j) {
                return None;
            }
            src[j] ^= 0x5a;
            changed = true;
        }
    }
    Some(src)
}

/// Ports `TestMaxOffsetBoundary` (`encode_test.go`), with 3 seeds instead of
/// Go's 10 to keep debug test time down.
#[test]
#[cfg_attr(miri, ignore = "2 MiB inputs too slow for miri")]
fn max_offset_boundary() {
    for level in [
        Level::SuperFast,
        Level::Fastest,
        Level::Balanced,
        Level::Smallest,
    ] {
        for delta in 0..4 {
            for seed in 0..3 {
                let Some(src) = build_max_offset_boundary(seed, delta) else {
                    continue;
                };
                roundtrip_at(&src, level)
                    .unwrap_or_else(|e| panic!("level={level:?} delta={delta} seed={seed}: {e}"));
            }
        }
    }
}

/// A marker over zero filler and a 6-byte piece of it at offset k of a copy at
/// distance `MAX_COPY3_OFFSET + delta`, so that only L1's probe at s+k can find
/// the match: it must take it at delta 0 and reject it at delta 1
/// (minio/minlz#75). [`max_offset_boundary`]'s Xor64 inputs only reach the s+2
/// probe. The sizes are Go's (`-tags noasm`).
#[test]
#[cfg_attr(miri, ignore = "2 MiB inputs too slow for miri")]
fn max_offset_probes() {
    use super::format::MAX_COPY3_OFFSET as M;
    const X: usize = 1000;
    let mut rng = Xor64::new(7);
    let marker: Vec<u8> = (0..16).map(|_| rng.next() as u8 | 1).collect();
    let sizes = [55, 54, 53].map(|l1| [[l1, 63, 53], [57, 63, 53]]);
    for (k, sizes) in sizes.into_iter().enumerate() {
        for (delta, sizes) in sizes.into_iter().enumerate() {
            let what = format!("k {k} delta {delta}");
            let y = X + M + delta;
            let mut src = vec![0u8; y + 4096];
            src[X..X + 16].copy_from_slice(&marker);
            for (i, b) in src[y..y + 16].iter_mut().enumerate() {
                *b = marker[i] ^ if (k..k + 6).contains(&i) { 0 } else { 0x80 };
            }
            for (level, want) in [Level::Fastest, Level::Balanced, Level::Smallest]
                .into_iter()
                .zip(sizes)
            {
                let mut enc = Vec::new();
                encode(&mut enc, &src, level).unwrap();
                let mut dec = Vec::new();
                decode(&mut dec, &enc).unwrap_or_else(|e| panic!("{level:?} {what}: {e}"));
                assert!(dec == src, "{level:?} {what}: roundtrip mismatch");
                assert_eq!(enc.len(), want, "{level:?} {what}");
            }
        }
    }
}

/// [`max_offset_probes`] for SuperFast, whose probes compare 8 bytes: a copy
/// at distance `MAX_COPY3_OFFSET + delta` that only the probe at s+k finds
/// (k 0..3), or the probe right after a copy that ends at it (k 3). The sizes
/// are Go's (`-tags noasm`).
#[test]
#[cfg_attr(miri, ignore = "2 MiB inputs too slow for miri")]
fn l0_max_offset_probes() {
    use super::format::MAX_COPY3_OFFSET as M;
    const X: usize = 1000;
    let mut rng = Xor64::new(7);
    let marker: Vec<u8> = (0..24).map(|_| rng.next() as u8 | 1).collect();
    let tail: Vec<u8> = (0..16).map(|_| rng.next() as u8 | 1).collect();
    for (k, sizes) in [[68, 72], [67, 72], [66, 72], [101, 98]]
        .into_iter()
        .enumerate()
    {
        for (delta, want) in sizes.into_iter().enumerate() {
            let y = X + M + delta;
            let mut src = vec![0u8; y + 4096];
            src[X..X + 24].copy_from_slice(&marker);
            let at = if k == 3 {
                src[X + 100..X + 116].copy_from_slice(&tail);
                src[y - 16..y].copy_from_slice(&tail);
                0
            } else {
                k
            };
            for (i, b) in src[y..y + 24].iter_mut().enumerate() {
                *b = marker[i] ^ if (at..at + 8).contains(&i) { 0 } else { 0x80 };
            }
            let mut enc = Vec::new();
            encode(&mut enc, &src, Level::SuperFast).unwrap();
            let mut dec = Vec::new();
            decode(&mut dec, &enc).unwrap_or_else(|e| panic!("k {k} delta {delta}: {e}"));
            assert!(dec == src, "k {k} delta {delta}: roundtrip mismatch");
            assert_eq!(enc.len(), want, "k {k} delta {delta}");
        }
    }
}

/// Input with matches at the lengths and offsets where the encodings change
/// size, repeated offsets, and copies of the block start. A Go copy of it made
/// [`encode_matches_go`]'s golden values.
fn border_mix(seed: u64, n: usize) -> Vec<u8> {
    const LENS: [usize; 16] = [
        3, 4, 5, 11, 12, 18, 19, 20, 29, 30, 31, 64, 65, 66, 285, 286,
    ];
    const OFFS: [usize; 6] = [8, 64, 1024, 1100, 70000, 3 << 20];
    const RUNS: [usize; 3] = [4, 30, 300];
    let mut rng = Xor64::new(seed);
    let alpha = 2 + rng.next() as usize % 255;
    let mut b = Vec::with_capacity(n + 300);
    let mut last = 1;
    while b.len() < n {
        let r = rng.next() as usize;
        if b.is_empty() || r.is_multiple_of(3) {
            for _ in 0..1 + (r >> 2) % RUNS[(r >> 10) % 3] {
                b.push((rng.next() as usize % alpha) as u8);
            }
            continue;
        }
        let mut off = 1 + (r >> 8) % OFFS[(r >> 6) % OFFS.len()].min(b.len());
        match (r >> 12) % 8 {
            0 => off = b.len(),
            1 | 2 => off = last,
            _ => {}
        }
        last = off;
        for _ in 0..LENS[(r >> 2) % LENS.len()] {
            b.push(b[b.len() - off]);
        }
    }
    b.truncate(n);
    b
}

/// Pins L0-L3 output to Go's (`-tags noasm`). The L3 border re-matching
/// (minio/minlz#53) and candidate checks change the parse but not its
/// validity, so only exact output catches a drift. Each row: seed, size and
/// XXH3 of the input, then size and XXH3 of the L0, L1, L2 and L3 blocks.
/// The L0 column is `simdgo enc -level -1` (`_helpers/simdgo` built with
/// `-tags noasm`) of the same inputs.
#[test]
#[cfg_attr(miri, ignore = "L3 on 100 KB inputs too slow for miri")]
fn encode_matches_go() {
    type Golden = [(usize, u64); 4];
    let check = |what: &str, src: &[u8], want: Golden| {
        let levels = [
            Level::SuperFast,
            Level::Fastest,
            Level::Balanced,
            Level::Smallest,
        ];
        for (level, want) in levels.into_iter().zip(want) {
            let mut enc = Vec::new();
            encode(&mut enc, src, level).unwrap();
            assert_eq!(
                (enc.len(), crate::xxh3::hash(&enc)),
                want,
                "{what} {level:?}"
            );
        }
    };
    #[rustfmt::skip]
    let rows: [(u64, usize, u64, Golden); 6] = [
        (1, 3000, 0x697d_1b8e_435e_602f,
         [(2100, 0xe5e2_4e27_04e5_892b), (1900, 0x516e_5924_8f03_2488),
          (1841, 0xf45e_d3fc_1105_4f51), (1838, 0x2d1c_3b3d_381c_b9b6)]),
        (2, 5000, 0x9219_c182_3284_54e8,
         [(2987, 0xe70b_e55a_b5d4_795d), (2459, 0xe322_ab1c_ca55_f457),
          (2369, 0x4b0e_0a37_6c02_5342), (2348, 0x69b1_5c20_9c4b_be16)]),
        (4, 9000, 0xe50b_16b4_a15a_70c6,
         [(5594, 0xe9fb_3b57_3b02_986b), (4932, 0x58ee_ec80_2766_c5e6),
          (4800, 0xff67_6f4c_aaab_91ac), (4731, 0x23a2_4a19_43a7_caa6)]),
        (1102, 5000, 0x679b_ade3_4ce4_fadd,
         [(1780, 0xd28f_7747_3d6e_117f), (1657, 0x2008_3de1_0682_12b3),
          (1669, 0x2821_9ec3_57f5_b8de), (1621, 0x1c3d_2075_8d4a_0676)]),
        (0, 100_000, 0xd6d6_510d_1e34_5737,
         [(44093, 0x326f_2c96_acf0_264c), (41032, 0x892a_20fc_9488_d471),
          (40848, 0x78ae_eb66_eba1_edca), (39780, 0xf00d_14d4_3ed2_37b9)]),
        (3, 100_000, 0x9cc4_f824_4875_c653,
         [(41217, 0x3ec6_24f2_e7f3_d23d), (38375, 0x5e85_d7dc_d338_9385),
          (38132, 0x7a01_abdb_dd60_ac49), (37032, 0x9984_13ca_707d_1411)]),
    ];
    for (seed, n, src_hash, want) in rows {
        let src = border_mix(seed, n);
        assert_eq!(
            crate::xxh3::hash(&src),
            src_hash,
            "border_mix({seed}) differs from Go's"
        );
        check(&format!("border_mix({seed}, {n})"), &src, want);
    }
    if let Some(twain) = locate_testdata("Mark.Twain-Tom.Sawyer.txt") {
        let want = [
            (14170, 0x465a_b3f8_2fba_3339),
            (10337, 0xdeaf_18b8_501d_3beb),
            (9301, 0x7c75_f53d_b2a8_7f17),
            (8874, 0x0d8f_7ffb_1d66_e78c),
        ];
        check("twain", &twain, want);
    }
}

/// Size of the shortest header that can encode a literal (or repeat) of n
/// bytes. Port of Go's `literalHeaderLen`.
fn literal_header_len(n: usize) -> usize {
    match n {
        ..=29 => 1,
        30..=0x11d => 2,
        0x11e..=0x1001d => 3,
        _ => 4,
    }
}

/// Calls `f(header_len, length, repeat)` for every literal and repeat tag in
/// a block body. Port of Go's `walkLiteralHeaders`.
fn walk_literal_headers(b: &[u8], mut f: impl FnMut(usize, usize, bool)) {
    let mut s = 0;
    while s < b.len() {
        let tag = b[s];
        match tag & 3 {
            0 => {
                let (hdr, length) = match tag >> 3 {
                    x @ 0..29 => (1, x as usize + 1),
                    29 => (2, 30 + b[s + 1] as usize),
                    30 => (3, 30 + (b[s + 1] as usize | (b[s + 2] as usize) << 8)),
                    _ => (
                        4,
                        30 + (b[s + 1] as usize
                            | (b[s + 2] as usize) << 8
                            | (b[s + 3] as usize) << 16),
                    ),
                };
                let repeat = tag & 4 != 0;
                f(hdr, length, repeat);
                s += hdr;
                if !repeat {
                    s += length;
                }
            }
            1 => s += if (tag >> 2) & 15 == 15 { 3 } else { 2 },
            2 => {
                s += match tag >> 2 {
                    61 => 4,
                    62 => 5,
                    63 => 6,
                    _ => 3,
                }
            }
            _ => {
                let lits = (tag >> 3) as usize & 3;
                if tag & 4 == 0 {
                    // Copy2 with 1-4 fused literals.
                    s += 3 + lits + 1;
                    continue;
                }
                s += 4 + lits;
                s += match (tag >> 5) as usize | ((b[s - lits - 3] & 7) as usize) << 3 {
                    61 => 1,
                    62 => 2,
                    63 => 3,
                    _ => 0,
                };
            }
        }
    }
    assert_eq!(
        s,
        b.len(),
        "token walk ended at {s}, block is {} bytes",
        b.len()
    );
}

/// Ports `TestEncodeLiteralHeaderMinimal` (`encode_test.go`, minio/minlz#79):
/// every encoder writes literals with the shortest header, in every size
/// class. Go's `MaxBlockSize` row (non-short only) is left out.
#[test]
#[cfg_attr(miri, ignore = "multi-MiB inputs too slow for miri")]
fn encode_literal_header_minimal() {
    let mut rng = Xor64::new(1);
    for n in [1000, 4000, 16000, 60000, 500_000, 2_000_000] {
        // Keep the incompressible run small enough that every level still
        // compresses the block.
        let lit_lens = [
            29,
            30,
            285,
            286,
            287,
            n / 4,
            30 + 0xffff,
            31 + 0xffff,
            70000,
        ];
        for lit_len in lit_lens.into_iter().filter(|&l| l <= n / 4) {
            // Compressible filler around one run of random bytes.
            let mut src: Vec<u8> = b"minlz literal headers "
                .iter()
                .copied()
                .cycle()
                .take(n)
                .collect();
            let start = (n - lit_len) / 2;
            src[start..start + lit_len]
                .iter_mut()
                .for_each(|b| *b = rng.next() as u8);
            for level in [
                Level::SuperFast,
                Level::Fastest,
                Level::Balanced,
                Level::Smallest,
            ] {
                let what = format!("n={n} litLen={lit_len} level={level:?}");
                let mut enc = Vec::new();
                encode(&mut enc, &src, level).unwrap();
                let Ok(Header::Classic { body, .. }) = parse_header(&enc) else {
                    panic!("{what}: block was not compressed");
                };
                let mut longest = 0;
                walk_literal_headers(body, |hdr, length, repeat| {
                    if !repeat {
                        longest = longest.max(length);
                    }
                    let want = literal_header_len(length);
                    assert_eq!(hdr, want, "{what}: repeat={repeat} length {length}");
                });
                assert!(
                    longest + 16 >= lit_len,
                    "{what}: longest literal is {longest} bytes; test input is not doing its job"
                );
            }
        }
    }
}

/// Ports `TestEncodeRandomStored` (`encode_test.go`): every level stores
/// random input as an uncompressed block.
#[test]
#[cfg_attr(miri, ignore = "MiB inputs too slow for miri")]
fn encode_random_stored() {
    let mut rng = Xor64::new(0);
    for size in [100, 1000, 10000, 100_000, 1_000_000] {
        let src: Vec<u8> = (0..size).map(|_| rng.next() as u8).collect();
        let mut want = Vec::new();
        encode_uncompressed(&mut want, &src);
        for level in [
            Level::SuperFast,
            Level::Fastest,
            Level::Balanced,
            Level::Smallest,
        ] {
            let mut got = Vec::new();
            encode(&mut got, &src, level).unwrap();
            assert!(
                got == want,
                "{size}/{level:?}: encoded to {} bytes, want a {} byte stored block",
                got.len(),
                want.len()
            );
        }
    }
}

/// The classic levels round-trip at the sizes where their encoders change: the shortest
/// compressed block, the limit of the 64 KiB encoders, and past the copy3 offset limit, where
/// the end of the input repeats its start.
#[test]
#[cfg_attr(miri, ignore = "3 MiB inputs too slow for miri")]
fn classic_edge_sizes() {
    let mut rng = Xor64::new(5);
    for n in [15, 16, 17, 33, 34, 35, 65535, 65536, 65537, 3 << 20] {
        let noise: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
        let mut far = noise.clone();
        far.copy_within(..n / 4, n - n / 4);
        let inputs = [
            repeat_bytes(&noise[..n.min(100)], n),
            vec![0; n],
            noise,
            far,
        ];
        for (i, src) in inputs.iter().enumerate() {
            for level in [
                Level::SuperFast,
                Level::Fastest,
                Level::Balanced,
                Level::Smallest,
            ] {
                roundtrip_at(src, level)
                    .unwrap_or_else(|e| panic!("{level:?} n={n} input {i}: {e}"));
            }
        }
    }
}

/// Port of Go's `mixedRuns`: `total` bytes of alternating incompressible and
/// compressible runs of `run_len` each, from [`Xor64`].
fn mixed_runs(rng: &mut Xor64, total: usize, run_len: usize) -> Vec<u8> {
    const WORDS: [&[u8]; 6] = [
        b"lorem",
        b"ipsum",
        b"dolor",
        b"sit",
        b"amet",
        b"consectetur",
    ];
    let mut out = Vec::with_capacity(total + 2 * run_len);
    while out.len() < total {
        out.extend((0..run_len).map(|_| rng.next() as u8));
        let start = out.len();
        while out.len() - start < run_len {
            out.extend_from_slice(WORDS[rng.next() as usize % WORDS.len()]);
            out.push(b' ');
        }
        out.truncate(start + run_len);
    }
    out.truncate(total);
    out
}

/// Golden version of Go's `TestBetterSkipClampMatchesAsm`: Balanced above
/// 64 KiB clamps its adaptive skip at 100 (minio/minlz#61), so its output on
/// long noise runs matches Go's (`-tags noasm`) byte for byte. Without the
/// clamp the skip strides past the start of the next text run.
#[test]
#[cfg_attr(miri, ignore = "4 MiB inputs too slow for miri")]
fn better_skip_clamp_golden() {
    // (run length, Go output size, XXH3 of Go output)
    let golden = [
        (64 << 10, 2_430_986, 0x7314_5a0e_28bc_91cb_u64),
        (256 << 10, 2_430_468, 0x29ff_9690_6550_4525),
    ];
    for (run_len, want_len, want_hash) in golden {
        let src = mixed_runs(&mut Xor64::new(1), 4 << 20, run_len);
        let mut got = Vec::new();
        encode(&mut got, &src, Level::Balanced).unwrap();
        assert_eq!(got.len(), want_len, "run length {run_len}");
        assert_eq!(crate::xxh3::hash(&got), want_hash, "run length {run_len}");
    }
}

/// Port of Go's `overlapTailBlock`: a block ending in an overlapping copy,
/// with its exact expected output.
fn overlap_tail_block(offset: usize, length: usize, trailing: usize) -> (Vec<u8>, Vec<u8>) {
    // A preceding match keeps even short final copies smaller than raw input.
    const PREFIX_MATCH: usize = 64;
    let literal: Vec<u8> = (0..offset).map(|i| (i * 173 + i / 97) as u8).collect();
    let head = offset + PREFIX_MATCH + length;
    let mut want: Vec<u8> = (0..head).map(|i| literal[i % offset]).collect();
    want.extend((0..trailing).map(|i| (i * 29 + 7) as u8));
    let mut payload = vec![0u8; max_encoded_len(want.len()).unwrap() + 16];
    let mut n = emit::emit_literal(&mut payload, &literal);
    n += emit::emit_copy(&mut payload[n..], offset, PREFIX_MATCH);
    n += emit::emit_copy(&mut payload[n..], offset, length);
    if trailing != 0 {
        n += emit::emit_literal(&mut payload[n..], &want[head..]);
    }
    let mut encoded = vec![0];
    format::put_uvarint(&mut encoded, want.len() as u64);
    encoded.extend_from_slice(&payload[..n]);
    (encoded, want)
}

/// Adds the seeds Go's `FuzzDecodeBlock` and `FuzzEncodingBlocks` build in code, see
/// [`crate::fuzz_seeds`].
#[test]
#[ignore = "writes fuzz/corpus, see fuzz/RUNBOOK.md §1"]
fn write_fuzz_seeds() {
    let offsets = [1, 3, 16, 17, 63, 64, 65, 4096, 65536];
    let blocks = offsets
        .into_iter()
        .flat_map(|o| [127, 128, 4096].map(|l| overlap_tail_block(o, l, 0).0));
    crate::fuzz_seeds::save("decode_arbitrary", blocks);
    let inputs = (1..=2).filter_map(|delta| build_max_offset_boundary(0, delta));
    crate::fuzz_seeds::save("roundtrip", inputs);
}

/// Ports `TestDecodeOverlapTail` (`decode_test.go`): copy-width boundaries.
/// Go's check for writes outside the destination has no counterpart, as
/// `decode` owns its `Vec`.
#[test]
#[cfg_attr(miri, ignore = "thousands of decodes too slow for miri")]
fn decode_overlap_tail() {
    let offsets = [
        1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 1023, 1024,
        4095, 4096, 65535, 65536, 65537,
    ];
    let lengths = [
        4, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 4096,
    ];
    let mut got = Vec::new();
    for offset in offsets {
        for length in lengths.into_iter().chain([7 * offset + 1, 7 * offset + 3]) {
            for trailing in [0, 1, 32] {
                let what = format!("offset{offset}/length{length}/trailing{trailing}");
                let (encoded, want) = overlap_tail_block(offset, length, trailing);
                decode(&mut got, &encoded).unwrap_or_else(|e| panic!("{what}: {e}"));
                assert!(got == want, "{what}");
            }
        }
    }
}

/// Ports `TestDecodeOverlapTailCorrupt` (`decode_test.go`): length mismatches
/// and truncated matches.
#[test]
fn decode_overlap_tail_corrupt() {
    let mut dec = Vec::new();
    // Miri: 65536 builds and decodes 512 KiB five times; 2048 is a copy2 as well.
    let copy2 = if cfg!(miri) { 2048 } else { 65536 };
    for offset in [1, 3, 15, 16, 17, 63, 64, 65, 1024, copy2] {
        let (encoded, want) = overlap_tail_block(offset, 7 * offset + 1, 0);
        let (_, header) = format::get_uvarint(&encoded[1..]).unwrap();
        for dlen in [want.len() - 1, want.len() + 1] {
            let mut bad = vec![0];
            format::put_uvarint(&mut bad, dlen as u64);
            bad.extend_from_slice(&encoded[1 + header..]);
            let res = decode(&mut dec, &bad);
            assert!(
                matches!(res, Err(Error::Corrupt(_))),
                "offset={offset} dlen={dlen}"
            );
        }
        for cut in 1..=3 {
            let res = decode(&mut dec, &encoded[..encoded.len() - cut]);
            assert!(
                matches!(res, Err(Error::Corrupt(_))),
                "offset={offset} cut={cut}"
            );
        }
    }
}

/// Ports `TestDecodeOverlapTailLarge` (`decode_test.go`): overlapping copies
/// at the maximum block size.
#[test]
#[cfg_attr(miri, ignore = "8 MiB decodes too slow for miri")]
fn decode_overlap_tail_large() {
    let mut got = Vec::new();
    for offset in [1, 3, 16, 17, 63, 64, 65, 4096, 65536] {
        let (encoded, want) = overlap_tail_block(offset, MAX_BLOCK_SIZE - offset - 64, 0);
        decode(&mut got, &encoded).unwrap_or_else(|e| panic!("offset={offset}: {e}"));
        assert!(got == want, "offset={offset}");
    }
}

/// Text, LE counters for delta literals, and noise with repeats, each over and under 64 KiB.
fn mixed_inputs() -> Vec<Vec<u8>> {
    let mut rng = Xor64::new(7);
    let counters: Vec<u8> = (0..30_000u32).flat_map(|i| (i * 3).to_le_bytes()).collect();
    let mut noisy = Vec::new();
    while noisy.len() < 100_000 {
        let k = rng.next() as usize % 40 + 1;
        if rng.next().is_multiple_of(3) && noisy.len() > 1000 {
            let o = rng.next() as usize % 1000 + 1;
            for _ in 0..k {
                noisy.push(noisy[noisy.len() - o]);
            }
        } else {
            noisy.extend((0..k).map(|_| rng.next() as u8 & 0x3f));
        }
    }
    let mut v = vec![pangram_x(3000), counters, noisy];
    v.extend(v.clone().iter().map(|b| b[..20_000].to_vec()));
    v
}

const ALL_LEVELS: [Level; 11] = [
    Level::SuperFast,
    Level::Fastest,
    Level::Balanced,
    Level::Smallest,
    Level::SuperFastSimd,
    Level::FastestSimd,
    Level::BalancedSimd,
    Level::SmallestSimd,
    Level::SuperFastSimdDelta,
    Level::FastestSimdDelta,
    Level::BalancedSimdDelta,
];

/// Streams reuse the buffer of [`encode_chunk_body`] without zeroing it, so a dirty buffer must
/// give the same chunk body as a fresh one.
#[test]
#[cfg_attr(miri, ignore = "encodes 360 KiB at every level")]
fn encode_chunk_body_reuses_dirty_buffer() {
    let inputs = mixed_inputs();
    for level in ALL_LEVELS {
        let mut dirty = vec![0xa5; 7];
        for (i, src) in inputs.iter().enumerate() {
            let mut fresh = Vec::new();
            let n = encode_chunk_body(&mut fresh, 7, src, level).unwrap();
            dirty.iter_mut().for_each(|b| *b = !*b);
            let m = encode_chunk_body(&mut dirty, 7, src, level).unwrap();
            assert_eq!(fresh[7..7 + n], dirty[7..7 + m], "{level:?} input {i}");
            if n == 0 {
                continue;
            }
            let mut dec = Vec::new();
            if level.is_simd() {
                let mut blk = vec![0];
                blk.extend_from_slice(&fresh[7..7 + n]);
                decode(&mut dec, &blk).unwrap();
            } else {
                append_decoded_chunk_body(&mut dec, &fresh[7..7 + n]).unwrap();
            }
            assert!(dec == *src, "{level:?} input {i}: round trip");
        }
    }
}

/// Regression: the L1 encoders guarded the literal-then-copy emit with
/// `d + (s - next_emit) > dst_limit` after `s` had been extended to the end
/// of the match, so the match length was counted as literals.  When the
/// first match ran to near the end of the input the guard tripped and the
/// block was stored uncompressed.  Go's `encode_l1.go` guards with the
/// literals alone.  Covers `encode_block_64k` (64 KiB) and
/// `encode_block_big` (200 KiB), with fragments on both sides of
/// `MIN_COPY2_OFFSET`.
#[test]
#[cfg_attr(miri, ignore = "64 KiB and 200 KiB inputs are impractical under miri")]
fn l1_repeated_fragment_compresses() {
    let mut rng = Xor64::new(0x5eed);
    for &size in &[64 << 10, 200 << 10] {
        for &frag_len in &[16usize, 63, 64, 251, 4096] {
            let frag: Vec<u8> = (0..frag_len).map(|_| rng.next() as u8).collect();
            let src = repeat_bytes(&frag, size);
            let mut enc = Vec::new();
            encode(&mut enc, &src, Level::Fastest).unwrap();
            assert!(
                enc.len() < src.len() / 10,
                "size={size} frag_len={frag_len}: got {} encoded bytes",
                enc.len()
            );
            let mut dec = Vec::new();
            decode(&mut dec, &enc).unwrap();
            assert!(dec == src, "size={size} frag_len={frag_len}: mismatch");
        }
    }
}

/// `k` segments of 400 random bytes and `[a; 4]`, then 400 random bytes and
/// `t` bytes of `k + 1`. Each segment costs L3 a 3-byte literal header and a
/// repeat for its 404 bytes, so the output overtakes the input.
fn l3_overflow_input(k: usize, t: usize) -> Vec<u8> {
    let mut rng = Xor64::new(0);
    let mut rnd = |avoid: u8| loop {
        let b = (rng.next() >> 24) as u8;
        if b != avoid && b != avoid.wrapping_add(1) {
            return b;
        }
    };
    let mut v = Vec::new();
    for a in 1..=k as u8 {
        v.extend((0..400).map(|_| rnd(a)));
        v.extend_from_slice(&[a; 4]);
    }
    let last = k as u8 + 1;
    v.extend((0..400).map(|_| rnd(last)));
    v.extend(std::iter::repeat_n(last, t));
    v
}

/// Regression: L3 (like Go's `encodeBlockBest`) checks its size limit before
/// it writes a literal header and a repeat, so its last emission ended up to
/// 9 bytes past `max_encoded_len`: heap corruption, and a truncated block
/// from `encode`.
#[test]
#[cfg_attr(miri, ignore = "300 L3 encodes of 18 KB are impractical under miri")]
fn l3_last_emission_stays_in_bounds() {
    for k in [40, 43, 44, 45, 46] {
        for t in 40..60 {
            let src = l3_overflow_input(k, t);
            let mut enc = Vec::new();
            encode(&mut enc, &src, Level::Smallest).unwrap();
            let mut out = Vec::new();
            decode(&mut out, &enc).unwrap();
            assert!(out == src, "k={k} t={t}");
            let mut tried = Vec::new();
            if try_encode(&mut tried, &src, Level::Smallest).unwrap() {
                decode(&mut out, &tried).unwrap();
                assert!(out == src, "try_encode k={k} t={t}");
            }
            let max = max_encoded_len(src.len()).unwrap();
            let mut body = vec![0xaa; max + 64];
            body.truncate(max);
            let n = encode_chunk_body(&mut body, 0, &src, Level::Smallest).unwrap();
            assert!(n < src.len(), "k={k} t={t}");
            // SAFETY: `body` holds `max + 64` initialized bytes, and `encode_chunk_body`
            // only grows its length to `max + ENCODE_SLACK`, within that capacity.
            let tail = unsafe { std::slice::from_raw_parts(body.as_ptr().add(body.len()), 8) };
            assert_eq!(tail, [0xaa; 8], "chunk body overflow k={k} t={t}");
        }
    }
}

/// The emitters write through unchecked stores; debug builds catch a store
/// past `dst`, as they do for the loads.
#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "i + 4 <= b.len()")]
fn emit_copy_short_dst_panics_in_debug() {
    let mut buf = [0u8; 8];
    emit::emit_copy(&mut buf[..1], 204_800, 4);
}
