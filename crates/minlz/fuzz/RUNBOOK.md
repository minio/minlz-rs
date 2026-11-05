# Fuzzing runbook

The corpus-smoke unit tests under `cargo test --lib` already feed
~3500 seeds from the Go repo through the decoder (SIMD blocks against
the reference decoder) and the round-trip path in deterministic order,
and the `*_seeded` tests run the SIMD and pivco targets' checks over
their seeds and mutations of them.  This document is for actual
fuzzing — i.e. coverage-guided random mutation on top of those seeds.

## 0. One-time setup

```bash
# 1. Install nightly toolchain (libfuzzer ships only on nightly).
rustup toolchain install nightly
rustup component add llvm-tools-preview --toolchain nightly

# 2. Install cargo-fuzz.
cargo install cargo-fuzz

# 3. Sanity check.
cd minlz-rs/crates/minlz/fuzz
cargo +nightly fuzz list
# Expected output:
#   decode_arbitrary
#   roundtrip
#   stream_decode_arbitrary
#   stream_roundtrip
#   index_load
#   pivco_decode
#   pivco_roundtrip
#   pivco_table
#   simd_decode
```

## 1. Seed the corpus

Run once.  Two sources fill `crates/minlz/fuzz/corpus/<target>/`;
subsequent `cargo fuzz run` invocations will accumulate beyond the
seed.

### Generated seeds (every target)

The ignored `write_fuzz_seeds` tests (`src/fuzz_seeds.rs`,
`src/block/tests.rs`, `src/pivco/tests/fuzz.rs`) add the seeds Go's
fuzzers add with `f.Add`, in each target's input format, named by
content hash: Go's seed zips encoded at the levels Go's fuzzers use
(SIMD blocks for `decode_arbitrary` and `simd_decode`, SIMD streams for
`stream_decode_arbitrary`), raw inputs for `roundtrip` and
`stream_roundtrip`, and the pivco cases.  The raw inputs are Tom Sawyer
and the SIMD fixtures' inputs, plus, with `MINLZ_TESTDATA` set to Go's
`testdata`, `enc_regressions.zip`, `fuzz/block-corpus-{raw,enc}.zip`
and `fuzz/FuzzEncodingBlocks/`; `decode_arbitrary` also gets
`dec-block-regressions.zip` and `fuzz/block-corpus-dec.zip`.  The
pivco and `simd_decode` targets have no other seeds.

```bash
cd minlz-rs
# Absolute: the tests run in crates/minlz.
MINLZ_TESTDATA="$PWD/../minlz/testdata" \
  cargo test -p minlz --release --lib write_fuzz_seeds -- --ignored
```

After the build it takes seconds.  With Go's zips it writes about 12k
seeds (300 MB); CI runs it without them.

### Upstream tarball (every target)

Downloads the upstream seed tarball and extracts the minimized corpora
of all nine targets.

Requires `zstd` on `PATH` (tar shells out to it):

```bash
# Debian/Ubuntu:  sudo apt install zstd
# Fedora/RHEL:    sudo dnf install zstd
# Arch:           sudo pacman -S zstd
# macOS:          brew install zstd
zstd --version  # sanity check
```

Then seed:

```bash
cd minlz-rs/crates/minlz/fuzz
curl -fsSL https://klauspost.com/download/rust-fuzz-corpus.tar.zst \
  | tar --zstd -xf - --no-same-owner --no-same-permissions
# corpus/<target>/ are now populated.  Verify:
for t in $(cargo fuzz list); do
    printf '%-25s %s seeds\n' "$t" "$(ls corpus/$t 2>/dev/null | wc -l)"
done
```

To re-seed a single target without touching the others, name it on the
tar command line:

```bash
curl -fsSL https://klauspost.com/download/rust-fuzz-corpus.tar.zst \
  | tar --zstd -xf - --no-same-owner --no-same-permissions corpus/index_load
```

The same tarball backs the GitHub Actions fuzz workflow (one fetch per
CI run, shared with the matrix jobs via a workflow artifact) — see
`.github/workflows/fuzz.yml`.

## 2. Run a target

`cargo fuzz run` blocks until interrupted (or until `-max_total_time`
elapses).  20 minutes per target gets you the 1-hour total budget the
stage plan asks for.

```bash
cd minlz-rs/crates/minlz/fuzz

# 20 min each — `-jobs=N -workers=N` parallelises across cores.
cargo +nightly fuzz run decode_arbitrary        -- -max_total_time=1200 -jobs=8 -workers=8
cargo +nightly fuzz run roundtrip               -- -max_total_time=1200 -jobs=8 -workers=8
cargo +nightly fuzz run stream_decode_arbitrary -- -max_total_time=1200 -jobs=8 -workers=8
cargo +nightly fuzz run stream_roundtrip        -- -max_total_time=1200 -jobs=8 -workers=8
cargo +nightly fuzz run index_load              -- -max_total_time=1200 -jobs=8 -workers=8
cargo +nightly fuzz run pivco_decode            -- -max_total_time=1200 -jobs=8 -workers=8
cargo +nightly fuzz run pivco_roundtrip         -- -max_total_time=1200 -jobs=8 -workers=8
cargo +nightly fuzz run pivco_table             -- -max_total_time=1200 -jobs=8 -workers=8
cargo +nightly fuzz run simd_decode             -- -max_total_time=1200 -jobs=8 -workers=8
```

`bash run.sh` runs every target in turn (`MAX_SECS`, `JOBS`, `WORKERS`;
from `/mnt` on WSL it mirrors the tree to `WORK_ROOT` on ext4).

Every `-jobs` process first replays the whole corpus. With the full seed
set that is 20 minutes for `roundtrip` at 24 jobs (10 levels per input,
up to 2.5 MB inputs), so the whole budget goes to replay and nothing gets
mutated, and each job grows to 1.1 GB RSS. `stream_roundtrip`'s corpus is
twice that size. Run those two in libFuzzer's fork mode, which
replays the corpus once and gives each child a subset:
`FORK=24 bash run.sh roundtrip stream_roundtrip` (`-fork=24
-ignore_crashes=1`). It prints one status line per finished child job;
crashes still land in `artifacts/<target>/`, but every crashing mutation
does, so one bug can leave hundreds. The one replay still takes
11 of 15 minutes for `roundtrip` and 7 for `stream_roundtrip`;
`MAX_LEN=262144` truncates the seeds as they load (8 and 3 minutes) and
gave 4.5x and 5x the executions, but loses what only longer inputs reach
(offsets past 256 KiB, several 2 MiB blocks per stream): `ft` after the
replay fell from 79.5k to 75.0k and from 76.8k to 72.0k.

Live output looks like this:
```
INFO: seed corpus: files: 1798 min: 1b max: 1048576b total: 12345678b ...
#10240  pulse  cov: 1234 ft: 5678 corp: 1234/56789b lim: 4096 exec/s: 102400 ...
```

`cov` (coverage) and `ft` (feature) numbers climb fast at first, then
plateau — that means the fuzzer has explored most of the reachable
code paths.

### Useful flags

| Flag                             | Effect                                                                  |
|----------------------------------|-------------------------------------------------------------------------|
| `-max_total_time=N`              | Run for `N` seconds, then exit cleanly.                                 |
| `-runs=N`                        | Stop after `N` total executions.                                        |
| `-jobs=N -workers=N`             | Spawn `N` parallel fuzzer processes.  Use ≤ # physical cores.            |
| `-max_len=N`                     | Cap input length (default 4 KiB; bump to 1 MiB for full coverage).      |
| `-dict=path/to/dict`             | Load a dictionary of interesting byte sequences.                        |
| `-only_ascii=1`                  | Restrict mutations to ASCII.  Sometimes helpful for text-heavy formats. |

Recommended for MinLZ:
```bash
cargo +nightly fuzz run decode_arbitrary -- \
  -max_total_time=1200 -jobs=8 -workers=8 -max_len=1048576
```

## 3. Triage a finding

If `libFuzzer` finds a panic/OOB/timeout, it writes the offending input
to `fuzz/artifacts/<target>/crash-<hash>`.

Reproduce + minimise:
```bash
# Reproduce one specific crash (single execution, full backtrace).
cargo +nightly fuzz run decode_arbitrary fuzz/artifacts/decode_arbitrary/crash-deadbeef

# Shrink it to a minimal counterexample.
cargo +nightly fuzz tmin decode_arbitrary fuzz/artifacts/decode_arbitrary/crash-deadbeef
```

Once minimised, add the bytes as a permanent regression seed under
`corpus/decode_arbitrary/` and commit them — the next CI run will replay
the case as part of the corpus-smoke unit tests, so the bug stays fixed.

## 4. Coverage report (optional)

```bash
cargo +nightly fuzz coverage decode_arbitrary
cargo +nightly cov -- show \
  target/x86_64-unknown-linux-gnu/coverage/x86_64-unknown-linux-gnu/release/decode_arbitrary \
  -instr-profile=fuzz/coverage/decode_arbitrary/coverage.profdata \
  -ignore-filename-regex='/rustc/|/.cargo/' \
  -format=html -output-dir=fuzz/coverage/decode_arbitrary/html
```

Open `fuzz/coverage/decode_arbitrary/html/index.html` in a browser to
see which branches the fuzzer reached.  Should be > 90 % for the
decoder; the few unreachable lines are inside `unsafe` SAFETY-guarded
branches that are never executable on valid corpus.

## 5. Stage A "DONE" target

The plan asks for **≥ 1 hour total wall-clock with no findings**, split
roughly equally across the five targets.  Concretely:

```bash
cd minlz-rs/crates/minlz/fuzz
# Seed corpus first (§1) if you haven't already.
# Note: stream targets benefit from a higher -max_len so the 9 MiB cap
# inside each target is actually reached.
for tgt in decode_arbitrary roundtrip stream_decode_arbitrary stream_roundtrip index_load; do
    cargo +nightly fuzz run "$tgt" -- -max_total_time=900 -jobs=8 -workers=8 -max_len=9437184
done
```

If any target finds a crash, add the minimised seed to
`corpus/<target>/`, fix the underlying bug, and rerun until clean.

## What the targets check

| Target                     | Invariant                                                            |
|----------------------------|----------------------------------------------------------------------|
| `decode_arbitrary`         | Block decoder: any byte slice → no panic / OOB; a dst holding garbage gives the same output; `decoded_len` agrees.  SIMD blocks: same result as the reference decoder, on every exec tier. |
| `roundtrip`                | Block codec: `decode(encode(x, level)) == x` at every classic and SIMD level, SIMD blocks on every exec tier and with the strict reference decoder; `len ≤ max_encoded_len`. |
| `simd_decode`              | SIMD body + `extra` under a valid header (≤ 1 MiB): every exec tier decodes as the reference decoder does. |
| `stream_decode_arbitrary`  | Stream decoder (cap 9 MiB): random bytes → ST `Reader::read_to_end` + MT `Reader::decode_concurrent`.  Either path must return `Ok`/`Err`, not panic / deadlock; they agree, and `skip` matches on valid streams.  Mode byte picks threads / ignore CRC / 1 MiB max block size. |
| `stream_roundtrip`         | Stream codec (cap 9 MiB): payload → ST `Writer` + MT `MtWriter`, decode through matching reader, assert byte-equality and written counts.  Mode byte picks concurrency / write method / block size / padding + index; the second byte the level. |
| `index_load`               | `Index::load` on any bytes → no panic; `append_to` → `load` round-trips. |
| `pivco_*`                  | See [`README.md`](README.md). |

Block fuzzers and stream fuzzers are kept separate because the chunk-
header layout and codec invariants differ.  Within stream fuzzers, the
ST and MT paths share the same target (and corpus) so any bug-finding
seed exercises both modes in one iteration.
