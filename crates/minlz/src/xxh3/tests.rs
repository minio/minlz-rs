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

use super::*;

/// Hashes of github.com/zeebo/xxh3, from _helpers/xxh3vec.
const VECTORS: &str = include_str!("vectors.txt");

/// gen of _helpers/xxh3vec: the little-endian bytes of successive xorshift64 states.
fn gen_input(n: usize) -> Vec<u8> {
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut b = Vec::with_capacity(n + 8);
    while b.len() < n {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        b.extend_from_slice(&x.to_le_bytes());
    }
    b.truncate(n);
    b
}

/// One-shot vectors (len, hash) and split vectors (len, hash, write sizes).
#[allow(clippy::type_complexity)]
fn vectors() -> (Vec<(usize, u64)>, Vec<(usize, u64, Vec<usize>)>) {
    let (mut one, mut split) = (Vec::new(), Vec::new());
    for line in VECTORS.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split(' ').collect();
        let n = f[1].parse().unwrap();
        let h = u64::from_str_radix(f[2], 16).unwrap();
        match f[0] {
            "h" => one.push((n, h)),
            "s" => split.push((n, h, f[3].split(',').map(|s| s.parse().unwrap()).collect())),
            _ => panic!("bad line {line:?}"),
        }
    }
    (one, split)
}

fn src() -> Vec<u8> {
    // Miri hashes a few KB a second, and its tests stay below two blocks.
    gen_input(if cfg!(miri) { 2113 } else { 3 << 20 })
}

/// Writes b in pieces of the given sizes, cycled, as the Go helper does.
fn write_split(h: &mut Hasher, b: &[u8], sizes: &[usize]) {
    let mut i = 0;
    for &s in sizes.iter().cycle() {
        if i == b.len() {
            break;
        }
        let s = s.min(b.len() - i);
        h.write(&b[i..i + s]);
        i += s;
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "parsing the vectors takes a minute; c_reference covers the scalar path"
)]
fn one_shot_matches_go() {
    let (one, _) = vectors();
    assert_eq!(one.len(), 4097 + 12);
    let src = src();
    for t in tiers() {
        for &(n, want) in &one {
            assert_eq!(hash_tier(t, &src[..n]), want, "{t:?} len {n}");
        }
    }
    for &(n, want) in &one {
        assert_eq!(hash(&src[..n]), want, "len {n}");
    }
}

#[test]
#[cfg_attr(
    miri,
    ignore = "parsing the vectors takes a minute; streaming_splits covers the scalar path"
)]
fn streaming_matches_go() {
    let (_, split) = vectors();
    assert!(split.len() > 200);
    let src = src();
    for t in tiers() {
        let mut h = Hasher::with_tier(t);
        for (n, want, sizes) in &split {
            h.reset();
            write_split(&mut h, &src[..*n], sizes);
            assert_eq!(h.sum64(), *want, "{t:?} len {n} sizes {sizes:?}");
        }
    }
}

/// Every write boundary around the block and buffer sizes, checking each prefix sum on the way.
#[test]
fn streaming_splits() {
    let src = gen_input(9000);
    let mut x: u64 = 1;
    let mut rnd = |m: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % m) as usize
    };
    // Miri: one block and a stripe, without the many short writes of round 0.
    let (ns, firsts, rounds): (&[usize], &[usize], _) = if cfg!(miri) {
        (&[0, 65, 1089], &[0, 64, 1088], 1..4)
    } else {
        (
            &[
                0, 1, 63, 64, 65, 240, 241, 1023, 1024, 1025, 1087, 1088, 1089, 2111, 2113, 9000,
            ],
            &[
                0, 1, 63, 64, 65, 960, 1023, 1024, 1025, 1087, 1088, 1089, 2112, 2113,
            ],
            0..4,
        )
    };
    for t in tiers() {
        let mut h = Hasher::with_tier(t);
        for &n in ns {
            for &first in firsts {
                for round in rounds.clone() {
                    h.reset();
                    let mut i = first.min(n);
                    h.write(&src[..i]);
                    while i < n {
                        let s = match round {
                            0 => rnd(70),
                            1 => rnd(1100),
                            2 => rnd(2300),
                            _ => [1, 64, 960, 1024, 1088, 1089][rnd(6)],
                        }
                        .min(n - i);
                        h.write(&src[i..i + s]);
                        i += s;
                        assert_eq!(
                            h.sum64(),
                            hash_tier(Tier::Scalar, &src[..i]),
                            "{t:?} at {i}"
                        );
                    }
                    assert_eq!(
                        h.sum64(),
                        hash_tier(Tier::Scalar, &src[..n]),
                        "{t:?} len {n}"
                    );
                }
            }
        }
    }
}

#[test]
fn tiers_agree_on_unaligned_input() {
    let src = src();
    let mut buf = vec![0u8; (1 << 20) + 64 + 64];
    // Miri has only the scalar tier, and hashes slowly.
    let (ns, offs): (&[usize], &[usize]) = if cfg!(miri) {
        (&[241, 1025], &[1, 63])
    } else {
        (
            &[241, 1000, 1024, 1025, 4096, 65543, 1 << 20],
            &[1, 3, 8, 13, 31, 32, 63],
        )
    };
    for &n in ns {
        let want = hash_tier(Tier::Scalar, &src[..n]);
        for &off in offs {
            buf[off..off + n].copy_from_slice(&src[..n]);
            let b = &buf[off..off + n];
            for t in tiers() {
                assert_eq!(hash_tier(t, b), want, "{t:?} len {n} off {off}");
                let mut h = Hasher::with_tier(t);
                for c in b.chunks(65536 + off) {
                    h.write(c);
                }
                assert_eq!(h.sum64(), want, "{t:?} streaming len {n} off {off}");
            }
        }
    }
}

/// XXH3_64bits of the C reference on bytes (i+1)%251, from zeebo/xxh3's compat_test.go.
#[test]
fn c_reference() {
    let b: Vec<u8> = (0..4096).map(|i| ((i + 1) % 251) as u8).collect();
    for (n, want) in [
        (0, 0x2d06800538d394c2),
        (3, 0xebce9b7632ae733b),
        (16, 0xeb5aeb9a32450f6a),
        (128, 0xce22cae9106851df),
        (240, 0xa5a910b2d7e065b0),
        (241, 0xb6515f490cdd4ce5),
        (1000, 0x30e4c5246f300d11),
        (1024, 0x546f61a5b0b850c1),
        (1025, 0xa58696e72de6df58),
        (4095, 0x268198759d7bdf74),
    ] {
        assert_eq!(hash(&b[..n]), want, "len {n}");
    }
}

#[test]
fn hasher_reuse() {
    let src = gen_input(5000);
    let mut h = Hasher::new();
    h.write(&src);
    let first = h.sum64();
    assert_eq!(first, h.sum64());
    h.reset();
    h.write(&src[..100]);
    h.reset();
    h.write(&src);
    assert_eq!(h.sum64(), first);
    assert_eq!(Hasher::default().sum64(), hash(&[]));
}
