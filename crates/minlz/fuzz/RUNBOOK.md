# Fuzzing runbook

The corpus-smoke unit tests under `cargo test --lib` already feed
~3500 seeds from the Go repo through the decoder and the round-trip
path in deterministic order.  This document is for actual fuzzing —
i.e. coverage-guided random mutation on top of those seeds.

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
```

## 1. Seed the corpus

Run once.  Downloads the upstream seed tarball and extracts every
target's corpus into `crates/minlz/fuzz/corpus/<target>/`.  Subsequent
`cargo fuzz run` invocations will accumulate beyond the seed.

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
curl -fsSL https://download.klauspost.com/rust-fuzz-corpus.tar.zst \
  | tar --zstd -xf - --no-same-owner --no-same-permissions
# corpus/{decode_arbitrary,roundtrip,stream_decode_arbitrary,stream_roundtrip,index_load}/
# are now populated.  Verify:
for t in decode_arbitrary roundtrip stream_decode_arbitrary stream_roundtrip index_load; do
    printf '%-25s %s seeds\n' "$t" "$(ls corpus/$t 2>/dev/null | wc -l)"
done
```

To re-seed a single target without touching the others, name it on the
tar command line:

```bash
curl -fsSL https://download.klauspost.com/rust-fuzz-corpus.tar.zst \
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
```

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
| `decode_arbitrary`         | Block decoder: any byte slice → no panic / OOB.                      |
| `roundtrip`                | Block codec: `decode(encode(x, level)) == x` at every level.         |
| `stream_decode_arbitrary`  | Stream decoder (cap 9 MiB): random bytes → ST `Reader::read_to_end` + MT `Reader::decode_concurrent`.  Either path must return `Ok`/`Err`, not panic / deadlock. |
| `stream_roundtrip`         | Stream codec (cap 9 MiB): payload → ST `Writer` + MT `MtWriter`, decode through matching reader, assert byte-equality.  Mode byte picks concurrency / level / block size. |

Block fuzzers and stream fuzzers are kept separate because the chunk-
header layout and codec invariants differ.  Within stream fuzzers, the
ST and MT paths share the same target (and corpus) so any bug-finding
seed exercises both modes in one iteration.
