# minlz fuzz harness

Nine [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) targets: ports of
the Go fuzzers (`fuzz_test.go`, `internal/pivco`) plus an index-level
robustness target. Go's search fuzzers have no counterpart (no block search
in Rust).

| Target                     | Go fuzzer                | What it checks                                                                |
|----------------------------|--------------------------|-------------------------------------------------------------------------------|
| `decode_arbitrary`         | `FuzzDecodeBlock`        | Block decoder never panics or reads OOB; garbage in dst does not change the output; `decoded_len` agrees; SIMD blocks decode as the reference decoder does, and on every exec tier. |
| `roundtrip`                | `FuzzEncodingBlocks`     | Block encode at every classic and SIMD level (≤ 8 MiB) stays within `max_encoded_len` and decodes back, SIMD blocks on every exec tier and with the strict reference decoder. |
| `simd_decode`              | `FuzzDecodeSIMD`         | SIMD block bodies with a valid header decode on every exec tier as the reference decoder does. |
| `stream_decode_arbitrary`  | `FuzzStreamDecode`       | Single- and multi-threaded stream decoders never panic / OOB / deadlock, agree, and `skip` matches. |
| `stream_roundtrip`         | `FuzzStreamEncode`       | Stream `Writer` / `MtWriter` round-trip across concurrency × level × block size × write method, with padding and index. |
| `index_load`               | —                        | `Index::load` parses arbitrary bytes safely; `append_to` → `load` round-trips. |
| `pivco_decode`             | `FuzzDecodeBlock{,Fixed}` | pivco block decode on every kernel tier: tiers agree, errors fail closed, accepted blocks are canonical. |
| `pivco_roundtrip`          | `FuzzRoundTripBlock`     | pivco encode + decode on every kernel tier with a table built from the data; tiers write identical blocks. |
| `pivco_table`              | `FuzzTable`, `FuzzLengths` | pivco `Table::set_lengths` / `build_lengths` on arbitrary lengths and histograms, round-tripping a payload. |

ASan's redzones around libFuzzer's exact-size input and exact-capacity
outputs stand in for the guard pages of Go's fuzzers.

## Running

```
cargo install cargo-fuzz       # one-time, needs nightly toolchain
cd crates/minlz/fuzz
cargo +nightly fuzz run decode_arbitrary -- -max_total_time=1200
MAX_SECS=1200 bash run.sh      # every target in turn
```

See [`RUNBOOK.md`](RUNBOOK.md) for triage, coverage collection, and the
`run.sh` driver that mirrors the corpus from `/mnt` to an ext4 path on
WSL.

## Seed corpora

The original five targets have seed corpora in a single tarball hosted at
<https://download.klauspost.com/rust-fuzz-corpus.tar.zst>. Seeds for every
target, derived as Go's fuzzers derive theirs (SIMD fixtures, pivco cases,
and Go's seed zips when `MINLZ_TESTDATA` points at Go's `testdata`), come
from the ignored `write_fuzz_seeds` tests. See [`RUNBOOK.md` §1](RUNBOOK.md)
for both.

CI (`.github/workflows/fuzz.yml`) fetches the tarball and generates the
seeds without Go's zips once per run, and shares both with the matrix jobs
via workflow artifacts.
