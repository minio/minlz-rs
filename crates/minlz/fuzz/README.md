# minlz fuzz harness

Five [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) targets — three
block-level (ports of the Go reference fuzzers) plus stream-level and
index-level robustness targets.

| Target                     | What it checks                                                                |
|----------------------------|-------------------------------------------------------------------------------|
| `decode_arbitrary`         | Block decoder never panics or reads OOB on any byte slice.                    |
| `roundtrip`                | Block encode + decode at every level reproduces the input (≤ 8 MiB).          |
| `stream_decode_arbitrary`  | Single- and multi-threaded stream decoders never panic / OOB / deadlock.      |
| `stream_roundtrip`         | Stream `Writer` / `MtWriter` round-trip across concurrency × level × block size. |
| `index_load`               | `Index::load` parses arbitrary bytes safely; `append_to` → `load` round-trips. |

## Running

```
cargo install cargo-fuzz       # one-time, needs nightly toolchain
cd crates/minlz/fuzz
cargo +nightly fuzz run decode_arbitrary        -- -max_total_time=1200
cargo +nightly fuzz run roundtrip               -- -max_total_time=1200
cargo +nightly fuzz run stream_decode_arbitrary -- -max_total_time=1200
cargo +nightly fuzz run stream_roundtrip        -- -max_total_time=1200
cargo +nightly fuzz run index_load              -- -max_total_time=1200
```

See [`RUNBOOK.md`](RUNBOOK.md) for triage, coverage collection, and the
`run.sh` driver that mirrors the corpus from `/mnt` to an ext4 path on
WSL.

## Seed corpora

Seed corpora for all five targets are bundled in a single tarball
hosted at <https://download.klauspost.com/rust-fuzz-corpus.tar.zst>.
See [`RUNBOOK.md` §1](RUNBOOK.md) for the one-liner that downloads
and extracts it into `corpus/<target>/`.

The same tarball is fetched once per CI run by
`.github/workflows/fuzz.yml` and shared with the matrix jobs via a
workflow artifact.
