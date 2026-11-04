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

//! SIMD blocks of the Go encoder, `tests/fixtures/simd`, written by `_helpers/simdgo` (see its
//! manifest). The inputs are regenerated here from their names, as `_helpers/simdgo/fixtures.go`
//! generates them.

use std::path::{Path, PathBuf};

/// Go's int levels of the fixtures: the SIMD levels, then the delta ones (`LevelSIMDDelta` = 256).
const LEVELS: [i32; 7] = [4, 5, 6, 7, 260, 261, 262];

/// An input and the block Go encoded it to at a level.
struct Fixture {
    name: String,
    level: i32,
    input: Vec<u8>,
    block: Vec<u8>,
    simd: bool,
}

fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn fixture_dir() -> PathBuf {
    manifest_dir().join("tests/fixtures/simd")
}

fn sawyer_text() -> Vec<u8> {
    std::fs::read(manifest_dir().join("../../testdata/Mark.Twain-Tom.Sawyer.txt"))
        .expect("read Tom Sawyer")
}

/// Reads the manifest, and regenerates and checks the inputs.
fn fixtures() -> Vec<Fixture> {
    let text = sawyer_text();
    let manifest =
        std::fs::read_to_string(fixture_dir().join("manifest.txt")).expect("read manifest");
    let mut out: Vec<Fixture> = Vec::new();
    for line in manifest.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let &[name, level, insize, encsize, kind, hash] = &f[..] else {
            panic!("manifest line {line:?}");
        };
        let input = match out.last() {
            Some(p) if p.name == name => p.input.clone(),
            _ => input(name, &text),
        };
        assert_eq!(
            (input.len().to_string(), format!("{:016x}", fnv1a(&input))),
            (insize.to_string(), hash.to_string()),
            "{name}: the input differs from Go's"
        );
        let simd = kind == "simd";
        let block = match kind {
            "simd" => std::fs::read(fixture_dir().join(format!("{name}.L{level}.mzb")))
                .expect("read block"),
            _ if input.is_empty() => vec![0],
            "stored" => [&[0, 0], &input[..]].concat(),
            _ => panic!("{name}: kind {kind}"),
        };
        assert_eq!(
            block.len().to_string(),
            encsize,
            "{name} L{level}: block size"
        );
        out.push(Fixture {
            name: name.into(),
            level: level.parse().unwrap(),
            input,
            block,
            simd,
        });
    }
    out
}

/// The Go helper `_helpers/simdgo` built as a binary, from env `MINLZ_SIMDGO`.
fn go_helper() -> Option<PathBuf> {
    std::env::var_os("MINLZ_SIMDGO").map(PathBuf::from)
}

/// Runs the Go helper's encoder in a scratch directory.
struct GoEncoder {
    go: PathBuf,
    dir: PathBuf,
}

impl GoEncoder {
    fn new(go: PathBuf, tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("minlz-simdgo-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        GoEncoder { go, dir }
    }

    /// Returns `minlz.Encode` of `src` at Go's int `level`.
    fn encode(&self, src: &[u8], level: i32) -> Vec<u8> {
        let (inp, outp) = (self.dir.join("in.bin"), self.dir.join("out.mzb"));
        std::fs::write(&inp, src).unwrap();
        let st = std::process::Command::new(&self.go)
            .args(["enc", "-level", &level.to_string()])
            .args([&inp, &outp])
            .status()
            .expect("run MINLZ_SIMDGO");
        assert!(st.success(), "simdgo enc -level {level}: {st}");
        std::fs::read(&outp).unwrap()
    }
}

impl Drop for GoEncoder {
    fn drop(&mut self) {
        for f in ["in.bin", "out.mzb"] {
            let _ = std::fs::remove_file(self.dir.join(f));
        }
        let _ = std::fs::remove_dir(&self.dir);
    }
}

/// The manifest matches the regenerated inputs and the files of the fixture directory.
#[test]
fn fixture_inputs() {
    let fx = fixtures();
    let names: std::collections::HashSet<String> = fx
        .iter()
        .filter(|f| f.simd)
        .map(|f| format!("{}.L{}.mzb", f.name, f.level))
        .collect();
    for e in std::fs::read_dir(fixture_dir()).unwrap() {
        let name = e.unwrap().file_name().into_string().unwrap();
        assert!(
            name == "manifest.txt" || names.contains(&name),
            "{name} is not in the manifest"
        );
    }
    assert!(
        fx.chunks(LEVELS.len())
            .all(|c| c.iter().map(|f| f.level).eq(LEVELS))
    );
    assert!(names.len() > 200, "{} SIMD blocks", names.len());
}

/// The fixtures are what the Go helper encodes, when `MINLZ_SIMDGO` is set.
#[test]
fn fixtures_are_current() {
    let Some(go) = go_helper() else {
        eprintln!("MINLZ_SIMDGO not set, skipping");
        return;
    };
    let enc = GoEncoder::new(go, "current");
    for f in fixtures() {
        assert!(
            enc.encode(&f.input, f.level) == f.block,
            "{} L{}: Go encodes it differently",
            f.name,
            f.level
        );
    }
}

/// Runs f at the best exec tier and the lower ones, as Go's simdTiers, with the features f gets
/// turned off. The lock keeps the tests of this file from changing the process-wide features
/// under each other.
fn each_tier(mut f: impl FnMut(u32)) {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for mask in [
        0,
        minlz::cpu::AVX512,
        minlz::cpu::AVX512 | minlz::cpu::AVX2 | minlz::cpu::NEON,
    ] {
        minlz::cpu::disable(mask);
        f(mask);
        minlz::cpu::enable_all();
    }
}

/// The decoder decodes every fixture to its input, through [`minlz::decode`] and
/// [`minlz::append_decoded`], and reports its size.
#[test]
fn decoder_decodes_fixtures() {
    let (fixtures, mut got) = (fixtures(), Vec::new());
    each_tier(|mask| {
        for f in &fixtures {
            let id = format!("{} L{} without {mask:#x}", f.name, f.level);
            assert_eq!(minlz::decoded_len(&f.block), Ok(f.input.len()), "{id}");
            assert_eq!(minlz::is_minlz(&f.block), Ok(Some(f.input.len())), "{id}");
            minlz::decode(&mut got, &f.block).unwrap_or_else(|e| panic!("{id}: {e}"));
            assert!(got == f.input, "{id}: output differs");
            got.clear();
            got.extend_from_slice(b"prefix");
            minlz::append_decoded(&mut got, &f.block).unwrap_or_else(|e| panic!("{id}: {e}"));
            assert!(
                got[..6] == *b"prefix" && got[6..] == f.input,
                "{id}: appended output differs"
            );
        }
    });
}

#[cfg(feature = "reference")]
mod reference {
    use super::*;
    use minlz::reference::{
        SimdWalker, decode_simd_block, decode_simd_block_strict, walk_simd_block,
    };

    /// What the blocks use, from [`SimdWalker`].
    #[derive(Default)]
    struct Coverage {
        /// Entries by stream and mode.
        modes: [[usize; 5]; 6],
        flags: [usize; 16],
        chunks: usize,
        lit_blocks: usize,
        escapes: usize,
        max_offset: usize,
        // Of the block being walked.
        block_chunks: usize,
        block_lit_blocks: usize,
    }

    impl SimdWalker for Coverage {
        fn entry(&mut self, stream: usize, mode: usize, values: usize, _: usize, _: usize) {
            self.modes[stream][mode] += 1;
            self.block_lit_blocks += usize::from(stream == 5);
            self.escapes += if stream == 3 { values } else { 0 };
        }
        fn chunk(&mut self, flags: u8, _: usize) {
            self.flags[flags as usize] += 1;
            self.block_chunks += 1;
        }
        fn op(&mut self, _: usize, _: usize, offset: usize, _: bool) {
            self.max_offset = self.max_offset.max(offset);
        }
    }

    /// The reference decoder decodes every SIMD fixture, also in strict mode, and the fixtures cover
    /// the features of the format.
    #[test]
    fn reference_decodes_fixtures() {
        let mut cov = Coverage::default();
        for f in fixtures().iter().filter(|f| f.simd) {
            let id = format!("{} L{}", f.name, f.level);
            assert_eq!(
                decode_simd_block_strict(&f.block).as_ref(),
                Ok(&f.input),
                "{id}"
            );
            (cov.block_chunks, cov.block_lit_blocks) = (0, 0);
            assert_eq!(
                walk_simd_block(&f.block, &mut cov).as_ref(),
                Ok(&f.input),
                "{id}"
            );
            cov.chunks = cov.chunks.max(cov.block_chunks);
            cov.lit_blocks = cov.lit_blocks.max(cov.block_lit_blocks);
        }
        let Coverage {
            modes,
            flags,
            chunks,
            lit_blocks,
            escapes,
            max_offset,
            ..
        } = cov;
        eprintln!(
            "modes {modes:?}\nchunk flags {flags:?}\nmost chunks {chunks}, most literal blocks {lit_blocks}"
        );
        eprintln!("escapes {escapes}, longest offset {max_offset}");
        // The chunk streams are stored, code with new tables of each kind and reuse them; offset bits
        // are only stored. The literals also use delta lengths, but no fixture reuses their table.
        for (s, m) in modes.iter().enumerate().take(4) {
            assert!(m[..4].iter().all(|&n| n > 0), "stream {s} modes {m:?}");
        }
        assert!(modes[4][0] > 0 && modes[4][1..].iter().all(|&n| n == 0));
        let lm = modes[5];
        assert!(
            lm[0] > 0 && lm[2] > 0 && lm[3] > 0 && lm[4] > 0,
            "literal modes {lm:?}"
        );
        assert!(
            flags[1] > 0 && flags[2] > 0 && flags[3] > 0,
            "chunk flags {flags:?}"
        );
        assert!(chunks > 1 && lit_blocks > 1 && escapes > 0 && max_offset > 1 << 20);
    }

    /// Mutated SIMD fixtures (bit flips, byte edits, deletions and cuts after the header) decode as
    /// the reference decoder decodes them: both reject, or both accept with the same output. The
    /// reference's default mode, not strict, has the decoder's rules: it raises the offsets of
    /// chunks with the no-overlap flag.
    #[test]
    fn mutants_match_reference() {
        let mut mutants = Vec::new();
        for f in fixtures().iter().filter(|f| f.simd) {
            let id = format!("{} L{}", f.name, f.level);
            let mut r = Rng(fnv1a(id.as_bytes()));
            // The header: the indicator byte and the varint.
            let h = 1 + f.block[1..].iter().position(|&c| c < 0x80).unwrap() + 1;
            for m in 0..if f.input.len() > 1 << 20 { 4 } else { 16 } {
                let mut b = f.block.clone();
                for _ in 0..1 + r.intn(3) {
                    if b.len() <= h {
                        break;
                    }
                    let p = h + r.intn(b.len() - h);
                    match r.intn(4) {
                        0 => b[p] ^= 1 << r.intn(8),
                        1 => b[p] = r.next() as u8,
                        2 => _ = b.remove(p),
                        _ => b.truncate(p),
                    }
                }
                mutants.push((format!("{id} mutant {m}"), decode_simd_block(&b), b));
            }
        }
        let ok = mutants.iter().filter(|m| m.1.is_ok()).count();
        eprintln!("{ok} of {} mutants decoded", mutants.len());
        assert!(ok > 0 && ok < mutants.len());
        let mut got = Vec::new();
        each_tier(|mask| {
            for (id, want, b) in &mutants {
                let res = minlz::decode(&mut got, b);
                assert_eq!(
                    res.is_ok(),
                    want.is_ok(),
                    "{id} without {mask:#x}: {res:?}, reference {want:?}"
                );
                if let Ok(w) = want {
                    assert!(got == *w, "{id} without {mask:#x}: output differs");
                }
            }
        });
    }

    /// Encodes a fresh random input with the Go helper at every level, when `MINLZ_SIMDGO` is set,
    /// and decodes it with the reference decoder and the decoder.
    #[test]
    fn go_encodes_random_input() {
        let Some(go) = go_helper() else {
            eprintln!("MINLZ_SIMDGO not set, skipping");
            return;
        };
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let (mut r, text) = (Rng(seed), sawyer_text());
        let size = r.intn(400 << 10);
        let mut src = Vec::new();
        while src.len() < size {
            let n = 1 + r.intn(64 << 10);
            let part = match r.intn(7) {
                0 => r.text_like(n),
                1 => r.words(n),
                2 => {
                    let p = 1 + r.intn(64);
                    r.period_runs(p, n)
                }
                3 => vec![0; n],
                4 => r.bytes(n.min(4 << 10)),
                5 => r.sawyer(&text, n),
                _ => {
                    let start = r.intn(src.len() + 1);
                    src[start..].iter().copied().cycle().take(n).collect()
                }
            };
            src.extend_from_slice(&part);
        }
        src.truncate(size);

        let enc = GoEncoder::new(go, "random");
        for level in LEVELS {
            let block = enc.encode(&src, level);
            let got = match block.get(1) {
                _ if src.is_empty() => Ok(block[1..].to_vec()),
                Some(0) => Ok(block[2..].to_vec()),
                _ => decode_simd_block_strict(&block),
            };
            assert!(
                got.as_ref() == Ok(&src),
                "seed {seed:#x} size {size} level {level}: {:?}",
                got.err()
            );
            let mut got = Vec::new();
            let res = minlz::decode(&mut got, &block);
            assert!(
                res.is_ok() && got == src,
                "seed {seed:#x} size {size} level {level}: decoder: {res:?}"
            );
        }
    }
}

fn fnv1a(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf29ce484222325, |h, &c| {
        (h ^ c as u64).wrapping_mul(0x100000001b3)
    })
}

/// Generates the input `name`, "kind-size", from an [`Rng`] seeded with the FNV-1a hash of the name.
fn input(name: &str, text: &[u8]) -> Vec<u8> {
    let (kind, size) = name.rsplit_once('-').expect(name);
    let size: usize = size.parse().expect(name);
    let mut r = Rng(fnv1a(name.as_bytes()));
    match kind {
        "zeros" => vec![0; size],
        "random" => r.bytes(size),
        "textlike" => r.text_like(size),
        "sparse" => r.sparse(size),
        "sawyer" => r.sawyer(text, size),
        "mixed" => r.mixed(text, size),
        _ => r.period_runs(
            kind.strip_prefix("period")
                .and_then(|p| p.parse().ok())
                .expect(name),
            size,
        ),
    }
}

const WORDS: [&str; 16] = [
    "the", "and", "of", "to", "a", "in", "is", "it", "you", "that", "he", "was", "for", "on",
    "are", "with",
];

/// SplitMix64, and the generators of `_helpers/simdgo/fixtures.go`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn intn(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }

    /// Appends a random word of a small vocabulary and a separator to `b`.
    fn word(&mut self, b: &mut Vec<u8>) {
        b.extend_from_slice(WORDS[self.intn(WORDS.len())].as_bytes());
        b.push(if self.intn(8) == 0 { b'\n' } else { b' ' });
    }

    /// Random words.
    fn words(&mut self, size: usize) -> Vec<u8> {
        let mut b = Vec::new();
        while b.len() < size {
            self.word(&mut b);
        }
        b.truncate(size);
        b
    }

    /// Phrases of 1 to 8 random words, each repeated 1 to 8 times.
    fn text_like(&mut self, size: usize) -> Vec<u8> {
        let mut b = Vec::new();
        while b.len() < size {
            let mut p = Vec::new();
            for _ in 0..1 + self.intn(8) {
                self.word(&mut p);
            }
            for _ in 0..1 + self.intn(8) {
                b.extend_from_slice(&p);
            }
        }
        b.truncate(size);
        b
    }

    /// A random pattern of 7 bytes, each time followed by one of 2 random bytes. It codes many
    /// operations cheaply, so the block has several chunks, which can reuse tables.
    fn sparse(&mut self, size: usize) -> Vec<u8> {
        let (pat, alpha) = (self.bytes(7), self.bytes(2));
        let mut b = Vec::new();
        while b.len() < size {
            b.extend_from_slice(&pat);
            b.push(alpha[self.intn(2)]);
        }
        b.truncate(size);
        b
    }

    /// Runs of 64 to 4159 bytes, each repeating a random pattern of `period` bytes.
    fn period_runs(&mut self, period: usize, size: usize) -> Vec<u8> {
        let mut b = Vec::new();
        while b.len() < size {
            let pat = self.bytes(period);
            for i in 0..64 + self.intn(4096) {
                b.push(pat[i % period]);
            }
        }
        b.truncate(size);
        b
    }

    /// `text`, then runs of 1 to 8 lines of it, a quarter of the runs with a random lowercase letter.
    fn sawyer(&mut self, text: &[u8], size: usize) -> Vec<u8> {
        let lines: Vec<&[u8]> = text.split_inclusive(|&c| c == b'\n').collect();
        let mut b = text.to_vec();
        while b.len() < size {
            let (start, first) = (b.len(), self.intn(lines.len()));
            for l in &lines[first..(first + 1 + self.intn(8)).min(lines.len())] {
                b.extend_from_slice(l);
            }
            if self.intn(4) == 0 {
                let i = start + self.intn(b.len() - start);
                b[i] = b'a' + self.intn(26) as u8;
            }
        }
        b.truncate(size);
        b
    }

    /// The kinds of data a large block meets: text, literals without matches, short periods, zeros,
    /// random data and many short matches of words. Then copies of earlier data at offsets of every
    /// octave, each followed by a random byte.
    fn mixed(&mut self, text: &[u8], size: usize) -> Vec<u8> {
        let mut b = self.sawyer(text, 128 << 10);
        for _ in 0..(48 << 10) {
            b.push(text[self.intn(text.len())]);
        }
        for p in [1, 2, 3, 4, 7, 8, 16, 31, 32, 33, 64] {
            b.extend_from_slice(&self.period_runs(p, 8 << 10));
        }
        b.resize(b.len() + (128 << 10), 0);
        b.extend_from_slice(&self.bytes(4 << 10));
        b.extend_from_slice(&self.words(96 << 10));
        while b.len() < size {
            let n = 256 + self.intn(8192);
            let o = self.intn(b.len());
            let off = 1 + (o >> self.intn(20));
            for _ in 0..n {
                b.push(b[b.len() - off]);
            }
            b.push(self.next() as u8);
        }
        b.truncate(size);
        b
    }
}

/// Go's SIMD int levels.
const RUST_LEVELS: [i32; 7] = [4, 5, 6, 7, 260, 261, 262];

fn rust_encode(src: &[u8], level: i32) -> Vec<u8> {
    let mut out = Vec::new();
    minlz::encode(&mut out, src, minlz::Level::try_from(level).unwrap()).unwrap();
    out
}

/// Describes where blocks a and b first differ: the byte, and with the reference walker, the
/// first entry, chunk or operation.
fn block_diff(a: &[u8], b: &[u8]) -> String {
    let at = a.iter().zip(b).position(|(x, y)| x != y);
    let d = format!(
        "{} and {} bytes, first difference at byte {at:?}",
        a.len(),
        b.len()
    );
    #[cfg(feature = "reference")]
    let d = d + &walk::walk_diff(a, b);
    d
}

/// The Rust encoder writes the fixtures' blocks byte for byte.
#[test]
fn rust_encodes_fixtures() {
    let mut n = 0;
    for f in fixtures().iter().filter(|f| RUST_LEVELS.contains(&f.level)) {
        let got = rust_encode(&f.input, f.level);
        assert!(
            got == f.block,
            "{} L{}: {}",
            f.name,
            f.level,
            block_diff(&got, &f.block)
        );
        n += 1;
    }
    assert!(n > 50, "{n} fixtures");
}

/// Inputs for the Go cross check: the kinds of the fixtures and random mixtures of them, of sizes
/// up to 2 MiB, and the benchmark files of `testdata/bench` (or `$MINLZ_TESTDATA/bench`).
fn cross_inputs() -> Vec<(String, Vec<u8>)> {
    let text = sawyer_text();
    let mut v = Vec::new();
    let sizes = [
        0,
        15,
        16,
        64,
        65,
        66,
        100,
        127,
        300,
        1000,
        4095,
        4097,
        20000,
        65535,
        65536,
        65537,
        200_000,
        1 << 20,
        2 << 20,
    ];
    for kind in [
        "zeros",
        "random",
        "textlike",
        "sparse",
        "sawyer",
        "period1",
        "period3",
        "period17",
        "period31",
        "period32",
        "period33",
        "period100",
    ] {
        for size in sizes {
            let name = format!("{kind}-{size}");
            v.push((name.clone(), input(&name, &text)));
        }
    }
    v.push(("mixed-2097152".into(), input("mixed-2097152", &text)));
    for seed in 0..24u64 {
        let mut r = Rng(seed.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x5eed);
        let size = r.intn(if seed % 3 == 0 { 2 << 20 } else { 200 << 10 });
        v.push((format!("mixture-{seed}"), r.mixture(&text, size)));
    }
    let dir = std::env::var_os("MINLZ_TESTDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir().join("../../testdata"))
        .join("bench");
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let name = e.file_name().into_string().unwrap();
            if !name.ends_with(".partial") {
                v.push((name, std::fs::read(e.path()).unwrap()));
            }
        }
    }
    v
}

/// The Rust encoder writes the blocks of the Go helper, when `MINLZ_SIMDGO` is set, for the
/// inputs of [`cross_inputs`].
#[test]
fn rust_encodes_as_go() {
    let Some(go) = go_helper() else {
        eprintln!("MINLZ_SIMDGO not set, skipping");
        return;
    };
    let enc = GoEncoder::new(go, "rustenc");
    let mut bad = Vec::new();
    let inputs = cross_inputs();
    for (name, src) in &inputs {
        for level in RUST_LEVELS {
            let (got, want) = (rust_encode(src, level), enc.encode(src, level));
            if got != want {
                bad.push(format!("{name} L{level}: {}", block_diff(&got, &want)));
            }
        }
    }
    eprintln!("{} inputs", inputs.len());
    assert!(
        bad.is_empty(),
        "{} differences:\n{}",
        bad.len(),
        bad.join("\n")
    );
}

#[cfg(feature = "reference")]
mod walk {
    use minlz::reference::{SimdWalker, walk_simd_block};

    /// Records the entries, chunks and operations of a block.
    #[derive(Default)]
    struct Events {
        ev: Vec<String>,
        chunks: usize,
        ops: usize,
    }

    impl SimdWalker for Events {
        fn entry(&mut self, stream: usize, mode: usize, values: usize, data: usize, table: usize) {
            self.ev.push(format!(
                "entry: stream {stream} mode {mode}, {values} values, {data} bytes, table {table}"
            ));
        }
        fn chunk(&mut self, flags: u8, ops: usize) {
            self.ev.push(format!(
                "chunk {}: flags {flags:#x}, {ops} ops",
                self.chunks
            ));
            self.chunks += 1;
        }
        fn op(&mut self, lits: usize, ml: usize, offset: usize, repeat: bool) {
            self.ev.push(format!(
                "op {}: {lits} literals, match {ml} at {offset}, repeat {repeat}",
                self.ops
            ));
            self.ops += 1;
        }
    }

    /// The first event where the walks of blocks a and b differ.
    pub(super) fn walk_diff(a: &[u8], b: &[u8]) -> String {
        let (mut ea, mut eb) = (Events::default(), Events::default());
        let (ra, rb) = (walk_simd_block(a, &mut ea), walk_simd_block(b, &mut eb));
        let i = ea.ev.iter().zip(&eb.ev).take_while(|(x, y)| x == y).count();
        let none = String::from("end");
        format!(
            "\n  first differing event {i}: {}\n  vs {}\n  decodes {:?} / {:?}",
            ea.ev.get(i).unwrap_or(&none),
            eb.ev.get(i).unwrap_or(&none),
            ra.map(|v| v.len()),
            rb.map(|v| v.len())
        )
    }
}

impl Rng {
    /// A mixture of the kinds of data of the fixtures, and copies of earlier data.
    fn mixture(&mut self, text: &[u8], size: usize) -> Vec<u8> {
        let mut src = Vec::new();
        while src.len() < size {
            let n = 1 + self.intn(64 << 10);
            let part = match self.intn(8) {
                0 => self.text_like(n),
                1 => self.words(n),
                2 => {
                    let p = 1 + self.intn(64);
                    self.period_runs(p, n)
                }
                3 => vec![0; n],
                4 => self.bytes(n.min(4 << 10)),
                5 => self.sawyer(text, n),
                6 => self.sparse(n),
                _ => {
                    let start = self.intn(src.len() + 1);
                    src[start..].iter().copied().cycle().take(n).collect()
                }
            };
            src.extend_from_slice(&part);
        }
        src.truncate(size);
        src
    }
}
