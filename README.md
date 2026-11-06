# MinLZ — Rust

[![Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue)](#license)

A fast LZ77-style block / streaming compressor — Rust port of
[`github.com/minio/minlz`](https://github.com/minio/minlz).

This implements the **MinLZ specification v1.0** (block + stream codec,
including the optional stream index for random access), plus **SIMD
blocks** from v1.1.  The format is specified in the Go repository's
[SPEC.md] and [SPEC_SIMD.md].  Block search tables (`SPEC.md` §4.13)
are **not** implemented.

[SPEC.md]: https://github.com/minio/minlz/blob/main/SPEC.md
[SPEC_SIMD.md]: https://github.com/minio/minlz/blob/main/SPEC_SIMD.md

## What you get

- **Eight compression levels** — the classic `Level::SuperFast` (-1),
  `Level::Fastest` (1), `Level::Balanced` (2, default), `Level::Smallest`
  (3), and the SIMD levels `Level::SuperFastSimd` (4), `FastestSimd`
  (5), `BalancedSimd` (6), `SmallestSimd` (7), the first three also with
  delta literals.  See [SIMD blocks](#simd-blocks).
- **SIMD decoding** — AVX-512 or AVX2 on x86-64 and NEON on aarch64,
  picked at runtime by CPU feature detection, with portable code on
  every other target.  The SIMD encoders write the same bytes as the Go
  encoders on every CPU.
- **Block codec** — `encode` / `decode` / `try_encode` / `append_encoded`
  / `append_decoded` / `decoded_len` / `is_minlz` / `max_encoded_len`.
  Blocks are capped at 8 MiB per the spec.  The decoders read classic
  and SIMD blocks.
- **Stream codec** — `stream::Reader<R: Read>` / `stream::Writer<W: Write>`,
  multi-block framing with CRC32C integrity per block, or XXH3-64 per
  SIMD block.
- **Multi-threaded streaming** — `stream::MtWriter<W>` parallelises
  encoding across worker threads; `Reader::decode_concurrent` parallelises
  decoding.
- **Random access** — `index::Index` plus `stream::ReadSeeker` give you
  uncompressed-offset seeks against any `Read + Seek` source.  The index
  can be appended to the stream, stored as a sidecar, or recovered
  post-hoc with `index::index_stream` over a sequential reader.
- **User-defined chunks** — embed metadata inline in a stream
  (chunk IDs 0x80–0xFD, skippable and non-skippable).
- **CRC32C and XXH3** — CRC32C is hardware-accelerated on x86_64
  (SSE4.2) and aarch64 (CRC32 extension), with a portable table
  fallback.  XXH3-64 is built in (AVX-512, AVX2 or SSE2 on x86_64,
  NEON on aarch64), so the crate has no dependencies.
- **Safe public API** — no public function is `unsafe`, but the hot
  paths use `unsafe` inside: the classic block decoder and encoders, the
  SIMD encoders' parsers (unchecked loads with `debug_assert!`ed
  bounds), the SIMD operation loops and entropy-coding kernels
  (`core::arch` intrinsics), and the CRC32C and XXH3 kernels.  The crate
  is `#![deny(missing_docs)]` and CI runs Clippy, Miri and cargo-fuzz.

## Quick start

Add the dependency:

```toml
[dependencies]
minlz = { git = "https://github.com/minio/minlz-rs" }
```

### Compress / decompress a block

Blocks are best for small payloads (≤ 8 MiB).  They carry no
integrity check — wrap them in a stream if corruption matters.

```rust
use minlz::{encode, decode, Level};

fn roundtrip(src: &[u8]) -> Result<(), minlz::Error> {
    let mut compressed = Vec::new();
    encode(&mut compressed, src, Level::Balanced)?;

    let mut decoded = Vec::new();
    decode(&mut decoded, &compressed)?;
    assert_eq!(decoded, src);
    Ok(())
}
```

### Compress / decompress a stream

Streams are independent blocks with framing, a checksum per block
(CRC32C, or XXH3 for SIMD blocks), an EOF marker, and (optionally) an
appended index.  Both `Writer` and `Reader` are
`std::io::Write` / `std::io::Read`, so they plug into anything in the
ecosystem.

```rust
use std::io::{Read, Write};
use minlz::stream::{Reader, Writer};

fn roundtrip(src: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut compressed: Vec<u8> = Vec::new();
    let mut w = Writer::new(&mut compressed);
    w.write_all(src)?;
    w.finish()?;                              // flush + EOF

    let mut out = Vec::new();
    Reader::new(&compressed[..]).read_to_end(&mut out)?;
    Ok(out)
}
```

`WriterBuilder` exposes `.level()`, `.block_size()`, `.padding()`,
`.append_index()`, `.uncompressed()`, etc.  `ReaderBuilder` lets you
cap `max_block_size`, opt out of CRC checks for speed, and register
callbacks for user chunks.

### Parallel encode / decode

`MtWriter` and `Reader::decode_concurrent` saturate all cores with a
single worker pool.  `MtWriter` requires `W: Send + 'static` because it
moves the sink into the writer thread and returns it via `finish`.

```rust
use std::io::Write;
use minlz::stream::{MtWriter, Reader};

fn parallel_roundtrip(payload: &[u8]) -> std::io::Result<()> {
    let mut w = MtWriter::new(Vec::<u8>::new());
    w.write_all(payload)?;
    let compressed = w.finish()?;

    let threads = std::thread::available_parallelism()?.get();
    let mut reader = Reader::new(&compressed[..]);
    let (n, _sink) = reader.decode_concurrent(Vec::<u8>::new(), threads)?;
    println!("decoded {n} bytes");
    Ok(())
}
```

### Random access by uncompressed offset

```rust
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use minlz::stream::{ReadSeeker, Reader, WriterBuilder};

fn random_read(payload: &[u8], offset: u64) -> std::io::Result<Vec<u8>> {
    let mut compressed: Vec<u8> = Vec::new();
    let mut w = WriterBuilder::new()
        .append_index()                  // index chunk after EOF
        .build(&mut compressed);
    w.write_all(payload)?;
    w.finish()?;

    // Empty slice → ReadSeeker loads the index from the tail itself.
    let reader = Reader::new(Cursor::new(compressed));
    let mut rs = ReadSeeker::new(reader, &[])?;
    rs.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0u8; 4096];
    rs.read_exact(&mut buf)?;
    Ok(buf)
}
```

Working examples for these patterns live under
[`crates/minlz/examples/`](crates/minlz/examples/) — see
`index_random_access.rs` and `index_sidecar.rs` for the full plumbing.

## Compression levels

| Level                         | Use when…                                             | Notes                                                                                   |
|-------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `Level::SuperFast` (-1)       | `Level::Fastest` is confirmed to be too slow.         | New matches need 8 bytes and are not extended back; usually much worse ratio than L1.   |
| `Level::Fastest`  (1)         | Throughput is the constraint.                         | Fast classic encode; modest ratio.                                                      |
| `Level::Balanced` (2)         | The default — what you want unless you have a reason. | 0.6–0.7× L1's encode speed; meaningfully better ratio.                                  |
| `Level::Smallest` (3)         | Archival; output is read many times.                  | Encode is roughly an order of magnitude slower than L2.  Decode speed is similar to L2. |
| `Level::SuperFastSimd` (4)    | SIMD encode speed matters most.                       | 1.3–1.4× L5's encode speed for 9–11 % larger output.  The fastest decode.               |
| `Level::FastestSimd` (5)      | As L1, when all readers support format v1.1.          | About L1's encode speed; 18–21 % smaller output and 2.0–2.4× L1's decode speed.         |
| `Level::BalancedSimd` (6)     | As L2, when all readers support format v1.1.          | 0.6–0.7× L5's encode speed.  The `mz` default.                                          |
| `Level::SmallestSimd` (7)     | As L3, when all readers support format v1.1.          | 1.5× L3's encode speed.  Picks plain or delta literals per chunk.                       |
| `Level::*SimdDelta` (260–262) | Binary data, such as tables of numbers.               | L4–L6 with delta literals.  Text gets larger and decodes slower.                        |

The SIMD notes are from the single-threaded stream benchmarks below
(2 MiB blocks).

A complementary knob is `WriterBuilder::block_size` (8 KiB–8 MiB,
default 2 MiB).  Smaller blocks trade ratio for lower memory and
slightly higher per-block overhead; larger blocks give the L2/L3
encoders more matching range.  The SIMD levels gain the most from
large blocks.

## SIMD blocks

SIMD blocks are a block type added in v1.1 of the format, see
[SPEC_SIMD.md].  They store literals and operations in separate
Huffman-coded streams, laid out so they can be decoded with SIMD.  The
SIMD levels write them; `decode`, `append_decoded` and the stream
readers read both block types.

```rust
use minlz::{decode, encode, Level};

fn roundtrip(src: &[u8]) -> Result<(), minlz::Error> {
    let mut compressed = Vec::new();
    encode(&mut compressed, src, Level::FastestSimd)?;

    // decode handles both block types.
    let mut decoded = Vec::new();
    decode(&mut decoded, &compressed)?;
    assert_eq!(decoded, src);
    Ok(())
}
```

- **When they pay off.** Each block carries its own Huffman tables, so
  SIMD levels pay off for blocks of 64 KiB and more (per the Go
  package's block benchmarks), such as streams with the default 2 MiB
  blocks.  For small blocks the classic levels compress faster, and
  usually decompress faster too.
- **Delta literals.** `Level::SuperFastSimdDelta`, `FastestSimdDelta`
  and `BalancedSimdDelta` store literals as the difference to the bytes
  at the last match offset.  This mainly helps binary data, like tables
  of numbers, but makes text larger and costs decompression speed.
  `Level::SmallestSimd` decides this itself.
- **Compatibility.** Decoders for format v1.0, such as the Go package
  before v1.3.0, cannot read SIMD blocks, nor streams written at SIMD
  levels.  Use the classic levels for data that every MinLZ decoder must
  read.
- **Streams.** SIMD blocks are stored as chunk type 0x10.  Instead of a
  CRC32C they are checked with a 64-bit XXH3 hash of the uncompressed
  data; `ReaderBuilder::ignore_crc` skips both checks.  The stream
  identifier of a stream written at a SIMD level has a flag (bit 6)
  set, so decoders without SIMD support reject the stream at once
  instead of failing at the first SIMD block.  Readers reject SIMD
  chunks in streams without the flag.  Blocks that don't compress are
  stored uncompressed, as with the other levels.
- **CPU dispatch.** The decoder picks its tier once: AVX-512 on x86-64
  CPUs with AVX-512 (F, BW, VL, VBMI, VBMI2), POPCNT, AVX2 and BMI2;
  AVX2 on CPUs with AVX2 and BMI2; NEON on little-endian aarch64.
  Other targets, and Miri, use portable Rust.  The entropy coder that
  the SIMD encoders and decoder share has AVX-512, AVX2 and NEON
  kernels, picked the same way; on x86-64 they also need POPCNT, and
  AVX-512 needs VPOPCNTDQ.  Output does not depend on the tier.

## Performance

`minlz-1` to `minlz-3` are the classic levels (`Level::Fastest`,
`Balanced`, `Smallest`).  The `minlz-simd-*` rows are the SIMD levels,
numbered like the Go README's "MinLZ SIMD" rows and the `mz` flags:
`xfast` is `Level::SuperFastSimd`, 1 to 3 are `FastestSimd`,
`BalancedSimd` and `SmallestSimd`.  SIMD streams need a decoder for
format specification v1.1; the classic levels can be read by every
MinLZ decoder.

#### Snappy benchmark set, geomean across 11 files

| Codec                | Avg ratio | Encode MB/s | Decode MB/s |
|----------------------|----------:|------------:|------------:|
| **minlz-1**          | **2.78x** |     **960** |   **4 423** |
| **minlz-2**          | **3.12x** |     **670** |   **4 743** |
| **minlz-3**          | **3.40x** |      **62** |   **3 892** |
| **minlz-simd-xfast** | **3.07x** |   **1 257** |   **9 725** |
| **minlz-simd-1**     | **3.41x** |     **923** |   **8 793** |
| **minlz-simd-2**     | **3.62x** |     **609** |   **7 935** |
| **minlz-simd-3**     | **3.95x** |      **92** |   **6 888** |
| snappy               |     2.17x |       1 007 |       3 007 |
| lz4_flex             |     2.23x |         835 |       1 583 |
| gzip-1               |     2.50x |         288 |         531 |
| zstd-1               |     3.49x |         559 |       1 742 |

#### Real-world workloads, first 1 GiB of each (geomean across 5 files)

CockroachDB log, GitHub events JSON, GitHub ranks binary, NYC taxi CSV,
VM Image.

| Codec                         | Avg ratio | Encode MB/s | Decode MB/s |
|-------------------------------|----------:|------------:|------------:|
| **minlz-1, MT × 32**          | **4.45x** |  **12 089** |  **25 557** |
| **minlz-2, MT × 32**          | **4.89x** |   **9 393** |  **26 715** |
| **minlz-3, MT × 32**          | **5.54x** |     **844** |  **26 344** |
| **minlz-simd-xfast, MT × 32** | **5.15x** |  **10 621** |  **27 440** |
| **minlz-simd-1, MT × 32**     | **5.60x** |  **10 614** |  **27 525** |
| **minlz-simd-2, MT × 32**     | **5.98x** |   **8 259** |  **28 634** |
| **minlz-simd-3, MT × 32**     | **6.94x** |     **952** |  **28 091** |
| **minlz-1**                   | **4.45x** |   **1 538** |   **3 770** |
| **minlz-2**                   | **4.89x** |     **894** |   **3 389** |
| **minlz-3**                   | **5.54x** |     **105** |   **3 771** |
| **minlz-simd-xfast**          | **5.15x** |   **2 013** |  **10 208** |
| **minlz-simd-1**              | **5.60x** |   **1 549** |   **8 978** |
| **minlz-simd-2**              | **5.98x** |     **885** |   **8 694** |
| **minlz-simd-3**              | **6.94x** |     **155** |   **8 616** |
| snappy                        |     3.31x |       1 322 |       2 758 |
| lz4_flex                      |     3.64x |       1 348 |       6 411 |
| gzip-1                        |     4.02x |         463 |         732 |
| zstd-1                        |     5.83x |       1 027 |       2 623 |

The MT rows use `stream::MtWriter` for encode and
`Reader::decode_concurrent` for decode at 32 worker threads.  Encode
scales 5–11× over the single-threaded baseline; decode sustains
~25–29 GB/s into an `io::sink()` target at every level.  MT rows swing
up to ±15% between runs (1 GiB decodes in 30–65 ms).

Against the classic level of the same number, SIMD streams are 18–21%
smaller on the real-world files (14–19% on the small Snappy-set files)
and decode 2.3–2.6× faster single-threaded (1.7–2.0× on the Snappy
set).  Single-threaded encode runs at 0.91–1.01× the classic speed at
levels 1 and 2, and 1.5× faster at level 3.  `minlz-simd-xfast`
encodes 1.3× faster than `minlz-1` and decodes 2.7× faster, at a
ratio between `minlz-2` and `minlz-3` on the real-world files.  With
32 threads SIMD encode runs at 0.88× the classic speed at levels 1
and 2 (1.13× at level 3), and decode is capped near 25–29 GB/s for all
levels, so SIMD gains only ~7% there.  Delta literals (`+delta`, in
the per-file tables) made the output smaller only on geo.protodata
(1–2%) and paper-100k.pdf (40 bytes), and cost 11–17% encode and 8–12%
decode speed on the real-world files.

Headlines: at **Fastest**, MinLZ matches Snappy/LZ4 encode speed with
substantially better ratio (2.78× vs 2.17× on the Snappy set; 4.45×
vs 3.31× on real workloads) and decodes 1.4–1.5× faster than
Snappy.  Every SIMD level decodes faster than LZ4 (8.6–10.2
GB/s vs 6.4 GB/s on real workloads).  On real workloads **SIMD Fastest**
compresses within 4% of zstd-1 at 1.5× its encode speed and 3.4× its
decode speed; **SIMD Balanced** and **SIMD Smallest** compress smaller than
zstd-1 on both sets.  Classic **Smallest** trades encoder throughput
for ratios near zstd-1 while keeping decode close to MinLZ-1.

<details>
<summary><b>Individual Results</b></summary>


### alice29.txt (0.15 MiB)

| codec                  | enc size | ratio | enc MB/s | dec MB/s |
|------------------------|---------:|------:|---------:|---------:|
| minlz-1                |   81,529 | 1.87x |    489.5 |   1480.9 |
| minlz-2                |   69,003 | 2.20x |    340.3 |   2481.1 |
| minlz-3                |   62,522 | 2.43x |     34.5 |   2493.3 |
| minlz-simd-xfast       |   71,348 | 2.13x |    985.7 |   8842.4 |
| minlz-simd-1           |   60,524 | 2.51x |    536.8 |   6527.4 |
| minlz-simd-2           |   56,857 | 2.67x |    313.4 |   2239.9 |
| minlz-simd-3           |   52,734 | 2.88x |     43.7 |   2437.3 |
| minlz-simd-xfast+delta |   97,504 | 1.56x |    759.3 |   2577.8 |
| minlz-simd-1+delta     |   69,192 | 2.20x |    442.0 |   2556.1 |
| minlz-simd-2+delta     |   61,622 | 2.47x |    281.6 |   2307.9 |
| snappy                 |   88,074 | 1.73x |    532.9 |   1649.6 |
| lz4_flex               |   88,702 | 1.71x |    533.8 |   1836.8 |
| gzip-1                 |   76,497 | 1.99x |    181.4 |    323.0 |
| zstd-1                 |   61,069 | 2.49x |    316.7 |   1063.6 |

### asyoulik.txt (0.12 MiB)

| codec                  | enc size | ratio | enc MB/s | dec MB/s |
|------------------------|---------:|------:|---------:|---------:|
| minlz-1                |   75,380 | 1.66x |    490.1 |   1442.2 |
| minlz-2                |   62,955 | 1.99x |    332.4 |   2326.7 |
| minlz-3                |   57,949 | 2.16x |     26.8 |   2597.1 |
| minlz-simd-xfast       |   64,428 | 1.94x |    852.1 |   8515.6 |
| minlz-simd-1           |   53,664 | 2.33x |    465.3 |   6419.4 |
| minlz-simd-2           |   50,917 | 2.46x |    283.9 |   6047.3 |
| minlz-simd-3           |   47,951 | 2.61x |     38.4 |   5442.6 |
| minlz-simd-xfast+delta |   88,506 | 1.41x |    781.4 |   2305.3 |
| minlz-simd-1+delta     |   64,070 | 1.95x |    434.8 |   5538.9 |
| minlz-simd-2+delta     |   56,693 | 2.21x |    245.2 |   5027.3 |
| snappy                 |   77,532 | 1.61x |    500.3 |   1638.5 |
| lz4_flex               |   79,831 | 1.57x |    531.8 |   1783.2 |
| gzip-1                 |   64,955 | 1.93x |    168.9 |    317.6 |
| zstd-1                 |   54,665 | 2.29x |    318.8 |   1082.9 |

### fireworks.jpeg (0.12 MiB)

| codec                  | enc size | ratio | enc MB/s | dec MB/s |
|------------------------|---------:|------:|---------:|---------:|
| minlz-1                |  123,118 | 1.00x |   2072.3 |  26190.0 |
| minlz-2                |  123,118 | 1.00x |   1917.3 |  26190.0 |
| minlz-3                |  123,046 | 1.00x |    258.7 |   2093.4 |
| minlz-simd-xfast       |  123,118 | 1.00x |   2097.0 |  26190.0 |
| minlz-simd-1           |  123,118 | 1.00x |   1960.1 |  26190.0 |
| minlz-simd-2           |  123,118 | 1.00x |   1609.1 |  26759.3 |
| minlz-simd-3           |  122,920 | 1.00x |    362.7 |  19538.6 |
| minlz-simd-xfast+delta |  123,118 | 1.00x |   1905.5 |  26190.0 |
| minlz-simd-1+delta     |  123,118 | 1.00x |   2038.0 |  26190.0 |
| minlz-simd-2+delta     |  123,118 | 1.00x |   1873.6 |  25644.4 |
| snappy                 |  123,119 | 1.00x |   2699.4 |  11612.5 |
| lz4_flex               |  123,108 | 1.00x |   1235.9 |   4734.3 |
| gzip-1                 |  123,043 | 1.00x |    352.9 |    559.0 |
| zstd-1                 |  123,102 | 1.00x |    611.5 |   3116.3 |

### geo.protodata (0.11 MiB)

| codec                  | enc size | ratio | enc MB/s | dec MB/s |
|------------------------|---------:|------:|---------:|---------:|
| minlz-1                |   17,503 | 6.78x |   1894.4 |  11187.5 |
| minlz-2                |   16,369 | 7.24x |   1273.8 |  10588.2 |
| minlz-3                |   14,788 | 8.02x |     81.4 |  10494.5 |
| minlz-simd-xfast       |   17,799 | 6.66x |   1677.3 |  14462.0 |
| minlz-simd-1           |   16,288 | 7.28x |   1486.1 |  13031.6 |
| minlz-simd-2           |   15,083 | 7.86x |    986.6 |  12100.8 |
| minlz-simd-3           |   13,462 | 8.81x |    132.9 |  12615.7 |
| minlz-simd-xfast+delta |   17,577 | 6.75x |   1451.5 |  12751.4 |
| minlz-simd-1+delta     |   15,976 | 7.42x |   1460.4 |  12615.7 |
| minlz-simd-2+delta     |   14,908 | 7.95x |    920.7 |  10980.4 |
| snappy                 |   23,364 | 5.08x |   1602.5 |   4982.7 |
| lz4_flex               |   19,472 | 6.09x |   1178.8 |   2348.3 |
| gzip-1                 |   21,196 | 5.59x |    440.5 |   1304.6 |
| zstd-1                 |   14,571 | 8.14x |    813.4 |   1864.6 |

### html (0.10 MiB)

| codec                  | enc size | ratio | enc MB/s | dec MB/s |
|------------------------|---------:|------:|---------:|---------:|
| minlz-1                |   19,873 | 5.15x |   1289.7 |   7314.3 |
| minlz-2                |   17,855 | 5.74x |    805.0 |   7757.6 |
| minlz-3                |   16,044 | 6.38x |     66.5 |   8192.0 |
| minlz-simd-xfast       |   17,828 | 5.74x |   1458.7 |  11636.4 |
| minlz-simd-1           |   16,202 | 6.32x |   1181.1 |  10343.4 |
| minlz-simd-2           |   15,164 | 6.75x |    707.7 |   9660.4 |
| minlz-simd-3           |   13,429 | 7.63x |    101.9 |  10778.9 |
| minlz-simd-xfast+delta |   20,632 | 4.96x |   1302.8 |  10039.2 |
| minlz-simd-1+delta     |   17,922 | 5.71x |   1099.9 |   8982.5 |
| minlz-simd-2+delta     |   16,341 | 6.27x |    656.4 |   8462.8 |
| snappy                 |   22,872 | 4.48x |   1335.1 |   4145.7 |
| lz4_flex               |   21,341 | 4.80x |   1023.0 |   2068.7 |
| gzip-1                 |   18,762 | 5.46x |    409.4 |   1061.1 |
| zstd-1                 |   15,408 | 6.65x |    690.0 |   1570.6 |

### html_x_4 (0.39 MiB)

| codec                  | enc size |  ratio | enc MB/s | dec MB/s |
|------------------------|---------:|-------:|---------:|---------:|
| minlz-1                |   19,876 | 20.61x |   2426.5 |  17808.7 |
| minlz-2                |   17,862 | 22.93x |   1905.1 |  16856.0 |
| minlz-3                |   16,051 | 25.52x |    250.1 |  17731.6 |
| minlz-simd-xfast       |   18,448 | 22.20x |   2512.9 |  24674.7 |
| minlz-simd-1           |   16,674 | 24.57x |   2170.6 |  23814.0 |
| minlz-simd-2           |   15,778 | 25.96x |   1690.5 |  22505.5 |
| minlz-simd-3           |   14,024 | 29.21x |    365.4 |   5793.5 |
| minlz-simd-xfast+delta |   21,255 | 19.27x |   2328.6 |   5611.0 |
| minlz-simd-1+delta     |   18,390 | 22.27x |   1922.1 |   5769.0 |
| minlz-simd-2+delta     |   16,955 | 24.16x |   1563.4 |   5736.7 |
| snappy                 |   92,318 |  4.44x |   1465.0 |   3720.3 |
| lz4_flex               |   83,835 |  4.89x |   1244.6 |    896.7 |
| gzip-1                 |   74,838 |  5.47x |    508.9 |    909.4 |
| zstd-1                 |   15,458 | 26.50x |   1989.3 |   3657.1 |

### kppkn.gtb (0.18 MiB)

| codec                  | enc size | ratio | enc MB/s | dec MB/s |
|------------------------|---------:|------:|---------:|---------:|
| minlz-1                |   62,111 | 2.97x |    643.6 |   1697.2 |
| minlz-2                |   52,776 | 3.49x |    485.7 |   1626.8 |
| minlz-3                |   46,024 | 4.00x |     48.9 |   1709.8 |
| minlz-simd-xfast       |   41,369 | 4.46x |    753.6 |   3432.4 |
| minlz-simd-1           |   45,312 | 4.07x |    635.4 |   2648.3 |
| minlz-simd-2           |   40,689 | 4.53x |    433.7 |   2797.0 |
| minlz-simd-3           |   33,953 | 5.43x |     54.6 |   2982.5 |
| minlz-simd-xfast+delta |   45,610 | 4.04x |    657.8 |   3031.6 |
| minlz-simd-1+delta     |   46,224 | 3.99x |    581.3 |   2809.8 |
| minlz-simd-2+delta     |   41,321 | 4.46x |    409.5 |   2827.0 |
| snappy                 |   69,566 | 2.65x |    910.7 |   2014.4 |
| lz4_flex               |   73,070 | 2.52x |    670.0 |   1743.8 |
| gzip-1                 |   57,660 | 3.20x |    291.2 |    428.9 |
| zstd-1                 |   39,334 | 4.69x |    450.2 |   1123.9 |

### lcet10.txt (0.41 MiB)

| codec                  | enc size | ratio | enc MB/s | dec MB/s |
|------------------------|---------:|------:|---------:|---------:|
| minlz-1                |  202,580 | 2.11x |    498.0 |   1555.2 |
| minlz-2                |  173,653 | 2.46x |    382.4 |   1598.9 |
| minlz-3                |  154,551 | 2.76x |     41.8 |   1652.2 |
| minlz-simd-xfast       |  187,488 | 2.28x |    864.8 |   9317.8 |
| minlz-simd-1           |  155,763 | 2.74x |    505.9 |   6752.4 |
| minlz-simd-2           |  143,630 | 2.97x |    358.5 |   6350.5 |
| minlz-simd-3           |  130,530 | 3.27x |     59.2 |   5578.5 |
| minlz-simd-xfast+delta |  251,689 | 1.70x |    767.1 |   2891.3 |
| minlz-simd-1+delta     |  171,716 | 2.49x |    457.9 |   6079.1 |
| minlz-simd-2+delta     |  152,420 | 2.80x |    324.2 |   5894.4 |
| snappy                 |  234,745 | 1.82x |    572.8 |   1527.9 |
| lz4_flex               |  233,299 | 1.83x |    533.7 |    822.7 |
| gzip-1                 |  208,361 | 2.05x |    195.4 |    327.1 |
| zstd-1                 |  158,517 | 2.69x |    398.3 |   1190.1 |

### paper-100k.pdf (0.10 MiB)

| codec                  | enc size | ratio | enc MB/s | dec MB/s |
|------------------------|---------:|------:|---------:|---------:|
| minlz-1                |   84,061 | 1.22x |   1799.6 |  18963.0 |
| minlz-2                |   82,726 | 1.24x |    900.6 |  15515.2 |
| minlz-3                |   82,295 | 1.24x |     35.1 |  16254.0 |
| minlz-simd-xfast       |   85,281 | 1.20x |   1508.1 |  15058.8 |
| minlz-simd-1           |   84,118 | 1.22x |   1928.4 |  13653.3 |
| minlz-simd-2           |   83,517 | 1.23x |   1051.3 |  15283.6 |
| minlz-simd-3           |   81,903 | 1.25x |     81.7 |  14027.4 |
| minlz-simd-xfast+delta |   85,241 | 1.20x |   1896.3 |  14840.6 |
| minlz-simd-1+delta     |   84,118 | 1.22x |   1928.4 |  13473.7 |
| minlz-simd-2+delta     |   83,517 | 1.23x |    864.1 |  14840.6 |
| snappy                 |   85,327 | 1.20x |   2337.9 |   9570.1 |
| lz4_flex               |   83,541 | 1.23x |   1657.0 |   1681.4 |
| gzip-1                 |   82,476 | 1.24x |    368.1 |    636.8 |
| zstd-1                 |   83,768 | 1.22x |    618.4 |   2316.7 |

### plrabn12.txt (0.46 MiB)

| codec                  | enc size | ratio | enc MB/s | dec MB/s |
|------------------------|---------:|------:|---------:|---------:|
| minlz-1                |  287,399 | 1.68x |    480.2 |   1295.0 |
| minlz-2                |  242,611 | 1.99x |    338.3 |   1550.4 |
| minlz-3                |  218,616 | 2.20x |     37.5 |   1510.1 |
| minlz-simd-xfast       |  254,492 | 1.89x |   1108.2 |   3434.5 |
| minlz-simd-1           |  208,331 | 2.31x |    471.2 |   4360.7 |
| minlz-simd-2           |  194,598 | 2.48x |    301.1 |   5862.1 |
| minlz-simd-3           |  180,171 | 2.67x |     53.9 |   5941.6 |
| minlz-simd-xfast+delta |  351,604 | 1.37x |    912.8 |   2511.0 |
| minlz-simd-1+delta     |  236,506 | 2.04x |    423.7 |   5209.3 |
| minlz-simd-2+delta     |  208,926 | 2.31x |    267.1 |   4804.2 |
| snappy                 |  319,362 | 1.51x |    479.1 |   1245.8 |
| lz4_flex               |  325,580 | 1.48x |    533.9 |    880.0 |
| gzip-1                 |  264,498 | 1.82x |    170.7 |    294.5 |
| zstd-1                 |  217,838 | 2.21x |    382.6 |   1824.5 |

### urls.10K (0.67 MiB)

| codec                  | enc size | ratio | enc MB/s | dec MB/s |
|------------------------|---------:|------:|---------:|---------:|
| minlz-1                |  260,961 | 2.69x |    780.4 |   2399.5 |
| minlz-2                |  230,304 | 3.05x |    510.4 |   2088.3 |
| minlz-3                |  207,314 | 3.39x |     56.3 |   2165.6 |
| minlz-simd-xfast       |  246,901 | 2.84x |   1052.0 |   5434.1 |
| minlz-simd-1           |  201,340 | 3.49x |    756.2 |   6476.8 |
| minlz-simd-2           |  187,425 | 3.75x |    510.2 |   5173.8 |
| minlz-simd-3           |  169,148 | 4.15x |     90.9 |   5845.9 |
| minlz-simd-xfast+delta |  297,649 | 2.36x |    909.8 |   3188.4 |
| minlz-simd-1+delta     |  224,528 | 3.13x |    677.0 |   4579.8 |
| minlz-simd-2+delta     |  204,332 | 3.44x |    439.8 |   4649.6 |
| snappy                 |  335,620 | 2.09x |    817.9 |   2052.3 |
| lz4_flex               |  336,026 | 2.09x |    821.9 |   1092.7 |
| gzip-1                 |  283,675 | 2.47x |    316.7 |    495.6 |
| zstd-1                 |  208,206 | 3.37x |    574.4 |   2068.6 |

### cockroach.node1.log (1024.00 MiB)

| codec                       |    enc size |  ratio | enc MB/s | dec MB/s |
|-----------------------------|------------:|-------:|---------:|---------:|
| minlz-1 (MT)                |  87,251,010 | 12.31x |  14993.0 |  34713.4 |
| minlz-2 (MT)                |  81,602,796 | 13.16x |  11996.0 |  35954.9 |
| minlz-3 (MT)                |  66,336,592 | 16.19x |   2060.1 |  36218.8 |
| minlz-simd-xfast (MT)       |  69,535,562 | 15.44x |  13697.0 |  33608.5 |
| minlz-simd-1 (MT)           |  67,647,001 | 15.87x |  13323.5 |  26809.2 |
| minlz-simd-2 (MT)           |  64,752,248 | 16.58x |  10732.0 |  34352.6 |
| minlz-simd-3 (MT)           |  49,306,242 | 21.78x |   2272.2 |  33790.2 |
| minlz-simd-xfast+delta (MT) |  73,486,274 | 14.61x |  13414.9 |  32848.8 |
| minlz-simd-1+delta (MT)     |  69,994,416 | 15.34x |  12705.2 |  34450.0 |
| minlz-simd-2+delta (MT)     |  66,453,023 | 16.16x |  10714.3 |  32038.4 |
| minlz-1                     |  87,251,010 | 12.31x |   2892.4 |   5812.3 |
| minlz-2                     |  81,602,796 | 13.16x |   1766.5 |   5708.9 |
| minlz-3                     |  66,336,592 | 16.19x |    202.9 |   6822.1 |
| minlz-simd-xfast            |  69,535,562 | 15.44x |   3240.8 |  13973.8 |
| minlz-simd-1                |  67,647,001 | 15.87x |   2978.0 |  12867.0 |
| minlz-simd-2                |  64,752,248 | 16.58x |   1726.6 |  12970.0 |
| minlz-simd-3                |  49,306,242 | 21.78x |    273.3 |  14113.7 |
| minlz-simd-xfast+delta      |  73,486,274 | 14.61x |   2801.7 |  12713.0 |
| minlz-simd-1+delta          |  69,994,416 | 15.34x |   2680.5 |  12368.9 |
| minlz-simd-2+delta          |  66,453,023 | 16.16x |   1581.8 |  12548.7 |
| snappy                      | 153,202,881 |  7.01x |   2287.9 |   4073.6 |
| lz4_flex                    | 116,439,097 |  9.22x |   2147.8 |   9018.2 |
| gzip-1                      | 111,321,553 |  9.65x |    930.2 |   1472.7 |
| zstd-1                      |  57,692,789 | 18.61x |   1948.5 |   4509.6 |

### github-june-2days-2019.json (1024.00 MiB)

| codec                       |    enc size | ratio | enc MB/s | dec MB/s |
|-----------------------------|------------:|------:|---------:|---------:|
| minlz-1 (MT)                | 161,415,423 | 6.65x |  13571.6 |  27357.7 |
| minlz-2 (MT)                | 153,662,320 | 6.99x |  11462.2 |  30792.1 |
| minlz-3 (MT)                | 133,007,996 | 8.07x |   1175.3 |  31525.1 |
| minlz-simd-xfast (MT)       | 148,151,060 | 7.25x |  12250.6 |  30543.5 |
| minlz-simd-1 (MT)           | 130,000,666 | 8.26x |  12116.8 |  33756.0 |
| minlz-simd-2 (MT)           | 127,144,724 | 8.45x |   9911.4 |  34876.6 |
| minlz-simd-3 (MT)           | 109,816,945 | 9.78x |   1370.3 |  33230.9 |
| minlz-simd-xfast+delta (MT) | 167,789,709 | 6.40x |  11760.1 |  30657.0 |
| minlz-simd-1+delta (MT)     | 140,968,767 | 7.62x |  11550.6 |  33334.3 |
| minlz-simd-2+delta (MT)     | 136,200,334 | 7.88x |  10123.6 |  30624.5 |
| minlz-1                     | 161,415,423 | 6.65x |   1924.3 |   4551.7 |
| minlz-2                     | 153,662,320 | 6.99x |   1102.1 |   4143.8 |
| minlz-3                     | 133,007,996 | 8.07x |    129.8 |   5094.8 |
| minlz-simd-xfast            | 148,151,060 | 7.25x |   2319.0 |  10872.0 |
| minlz-simd-1                | 130,000,666 | 8.26x |   1961.3 |   9931.5 |
| minlz-simd-2                | 127,144,724 | 8.45x |   1098.5 |   9849.0 |
| minlz-simd-3                | 109,816,945 | 9.78x |    196.6 |  10439.4 |
| minlz-simd-xfast+delta      | 167,789,709 | 6.40x |   2036.1 |   9771.5 |
| minlz-simd-1+delta          | 140,968,767 | 7.62x |   1766.3 |   9541.7 |
| minlz-simd-2+delta          | 136,200,334 | 7.88x |   1000.9 |   9353.9 |
| snappy                      | 259,082,296 | 4.14x |   1424.3 |   2897.6 |
| lz4_flex                    | 227,398,199 | 4.72x |   1369.9 |   6235.0 |
| gzip-1                      | 230,645,392 | 4.66x |    489.5 |    776.0 |
| zstd-1                      | 129,752,397 | 8.28x |   1261.5 |   3293.2 |

### github-ranks-backup.bin (1024.00 MiB)

| codec                       |    enc size | ratio | enc MB/s | dec MB/s |
|-----------------------------|------------:|------:|---------:|---------:|
| minlz-1 (MT)                | 350,786,224 | 3.06x |  10961.9 |  22386.4 |
| minlz-2 (MT)                | 306,255,841 | 3.51x |   8293.0 |  23803.7 |
| minlz-3 (MT)                | 285,008,213 | 3.77x |    557.6 |  24078.2 |
| minlz-simd-xfast (MT)       | 276,767,613 | 3.88x |  10262.2 |  29710.9 |
| minlz-simd-1 (MT)           | 275,910,099 | 3.89x |  10012.6 |  30085.7 |
| minlz-simd-2 (MT)           | 251,096,916 | 4.28x |   7526.4 |  27122.2 |
| minlz-simd-3 (MT)           | 224,220,947 | 4.79x |    654.5 |  27355.6 |
| minlz-simd-xfast+delta (MT) | 302,303,358 | 3.55x |   9988.3 |  26059.4 |
| minlz-simd-1+delta (MT)     | 291,415,266 | 3.68x |   9629.5 |  28030.8 |
| minlz-simd-2+delta (MT)     | 263,555,273 | 4.07x |   6827.0 |  27123.0 |
| minlz-1                     | 350,786,224 | 3.06x |   1438.6 |   3631.1 |
| minlz-2                     | 306,255,841 | 3.51x |    722.0 |   2853.0 |
| minlz-3                     | 285,008,213 | 3.77x |     80.7 |   2869.2 |
| minlz-simd-xfast            | 276,767,613 | 3.88x |   1916.4 |  10013.6 |
| minlz-simd-1                | 275,910,099 | 3.89x |   1407.0 |   7964.4 |
| minlz-simd-2                | 251,096,916 | 4.28x |    706.4 |   7307.9 |
| minlz-simd-3                | 224,220,947 | 4.79x |    129.1 |   6846.6 |
| minlz-simd-xfast+delta      | 302,303,358 | 3.55x |   1512.8 |   8830.9 |
| minlz-simd-1+delta          | 291,415,266 | 3.68x |   1149.9 |   7490.7 |
| minlz-simd-2+delta          | 263,555,273 | 4.07x |    613.1 |   6837.1 |
| snappy                      | 352,063,664 | 3.05x |   1343.8 |   2859.6 |
| lz4_flex                    | 365,716,326 | 2.94x |   1422.9 |   6154.0 |
| gzip-1                      | 287,570,030 | 3.73x |    485.2 |    804.0 |
| zstd-1                      | 273,678,613 | 3.92x |    919.6 |   2311.7 |

### nyc-taxi-data-10M.csv (1024.00 MiB)

| codec                       |    enc size | ratio | enc MB/s | dec MB/s |
|-----------------------------|------------:|------:|---------:|---------:|
| minlz-1 (MT)                | 299,283,039 | 3.59x |  11417.4 |  24724.2 |
| minlz-2 (MT)                | 255,081,991 | 4.21x |   8804.6 |  26169.3 |
| minlz-3 (MT)                | 219,671,113 | 4.89x |    756.4 |  27812.0 |
| minlz-simd-xfast (MT)       | 260,904,606 | 4.12x |  10003.6 |  30124.6 |
| minlz-simd-1 (MT)           | 221,359,786 | 4.85x |  10720.1 |  32926.1 |
| minlz-simd-2 (MT)           | 201,090,838 | 5.34x |   8072.7 |  32309.7 |
| minlz-simd-3 (MT)           | 174,560,119 | 6.15x |    790.4 |  31215.0 |
| minlz-simd-xfast+delta (MT) | 283,405,138 | 3.79x |   9699.4 |  27155.4 |
| minlz-simd-1+delta (MT)     | 227,626,008 | 4.72x |  10303.7 |  30133.8 |
| minlz-simd-2+delta (MT)     | 204,196,556 | 5.26x |   7263.1 |  30334.7 |
| minlz-1                     | 299,283,039 | 3.59x |    836.3 |   2425.8 |
| minlz-2                     | 255,081,991 | 4.21x |    630.7 |   2531.2 |
| minlz-3                     | 219,671,113 | 4.89x |     82.0 |   2821.4 |
| minlz-simd-xfast            | 260,904,606 | 4.12x |   1269.2 |   8758.3 |
| minlz-simd-1                | 221,359,786 | 4.85x |    856.8 |   7636.5 |
| minlz-simd-2                | 201,090,838 | 5.34x |    625.2 |   7751.1 |
| minlz-simd-3                | 174,560,119 | 6.15x |    112.8 |   8373.7 |
| minlz-simd-xfast+delta      | 283,405,138 | 3.79x |    999.8 |   7612.0 |
| minlz-simd-1+delta          | 227,626,008 | 4.72x |    759.1 |   6628.9 |
| minlz-simd-2+delta          | 204,196,556 | 5.26x |    566.8 |   6873.9 |
| snappy                      | 428,270,599 | 2.51x |    782.7 |   1803.2 |
| lz4_flex                    | 392,333,490 | 2.74x |    844.0 |   5670.0 |
| gzip-1                      | 363,226,950 | 2.96x |    281.7 |    434.3 |
| zstd-1                      | 222,461,279 | 4.83x |    658.6 |   1921.8 |

### rawstudio-mint14.tar (1024.00 MiB)

| codec                       |    enc size | ratio | enc MB/s | dec MB/s |
|-----------------------------|------------:|------:|---------:|---------:|
| minlz-1 (MT)                | 556,167,710 | 1.93x |  10137.6 |  20741.9 |
| minlz-2 (MT)                | 520,141,476 | 2.06x |   7284.3 |  19730.6 |
| minlz-3 (MT)                | 494,618,847 | 2.17x |    419.9 |  16594.6 |
| minlz-simd-xfast (MT)       | 531,320,146 | 2.02x |   7846.9 |  16932.9 |
| minlz-simd-1 (MT)           | 481,107,715 | 2.23x |   7772.2 |  17625.2 |
| minlz-simd-2 (MT)           | 449,611,081 | 2.39x |   5947.6 |  18333.1 |
| minlz-simd-3 (MT)           | 417,998,073 | 2.57x |    484.7 |  18243.5 |
| minlz-simd-xfast+delta (MT) | 546,332,174 | 1.97x |   7478.4 |  17286.8 |
| minlz-simd-1+delta (MT)     | 488,937,855 | 2.20x |   7638.1 |  17541.2 |
| minlz-simd-2+delta (MT)     | 456,403,262 | 2.35x |   5314.0 |  17627.5 |
| minlz-1                     | 556,167,710 | 1.93x |   1283.2 |   3266.1 |
| minlz-2                     | 520,141,476 | 2.06x |    642.8 |   2618.4 |
| minlz-3                     | 494,618,847 | 2.17x |     72.6 |   2709.2 |
| minlz-simd-xfast            | 531,320,146 | 2.02x |   1808.6 |   8319.2 |
| minlz-simd-1                | 481,107,715 | 2.23x |   1265.6 |   7506.1 |
| minlz-simd-2                | 449,611,081 | 2.39x |    646.9 |   6863.3 |
| minlz-simd-3                | 417,998,073 | 2.57x |    115.1 |   5622.5 |
| minlz-simd-xfast+delta      | 546,332,174 | 1.97x |   1489.2 |   7209.4 |
| minlz-simd-1+delta          | 488,937,855 | 2.20x |   1061.6 |   6385.7 |
| minlz-simd-2+delta          | 456,403,262 | 2.35x |    559.2 |   5626.7 |
| snappy                      | 604,501,920 | 1.78x |   1179.0 |   2624.1 |
| lz4_flex                    | 584,342,866 | 1.84x |   1259.6 |   5518.9 |
| gzip-1                      | 504,957,412 | 2.13x |    343.6 |    526.2 |
| zstd-1                      | 465,496,019 | 2.31x |    766.4 |   1880.4 |

</details>


<details>
<summary><b>Methodology</b></summary>

In-memory only; each input loaded once into a `Vec<u8>`, then encode
through each codec's streaming / frame API (so framing overhead and
checksums are included).  Decode timing targets `io::sink()` — for MinLZ
ST that is a `BufRead::fill_buf` loop, for MinLZ MT it is
`decode_concurrent(io::sink(), 32)`, for the rest it is
`io::copy(decoder, &mut io::sink())`.  This mirrors what
[`mz bench`](bin/mz/src/bench.rs) reports (`dec(st)` / `dec(mt)`).
Each variant is verified once via a full decode-to-`Vec` byte compare
against the source; the verify pass is excluded from the timed loop.
Best of 3 iterations per (codec, input).  MinLZ encodes with
`stream::Writer` (ST) or `stream::MtWriter` at 32 threads.  Each codec
uses its default block / frame size: MinLZ 2 MiB, Snappy 64 KiB
(spec-mandated), `lz4_flex` 64 KiB, gzip 32 KiB DEFLATE window, zstd's
per-level default window (128 KiB at L1).  SIMD blocks decode with
AVX-512 on this CPU.  Single-threaded runs are pinned to one CPU
(`start /affinity 10`); MT runs are not pinned.
Build: `cargo build --release` in the harness directory (`lto = "thin"`,
`codegen-units = 1`), Rust 1.92.

The Snappy-set files decode as one block per stream, and each decode
builds a new reader.  Whether its fresh buffers page-fault on first
touch depends on the Windows heap state, so per-file decode speeds on
that set swing 2–4× between runs, for every codec; over three runs the
geomeans moved by up to 17%.

Hardware: AMD Ryzen 9 9950X (16C / 32T, Zen 5), 64 GiB DDR5, Windows 11.

Corpus: 11 files from the [Snappy benchmark set] plus the first 1 GiB
of each of five real-world inputs.  Full per-file breakdowns live in
[`_helpers/cross_codec/results.md`](_helpers/cross_codec/results.md);
the harness itself is at
[`_helpers/cross_codec/`](_helpers/cross_codec/) and rebuilds with
`cargo build --release` from that directory.  Measuring appends CSV
rows; `--report` prints the tables:

```sh
./target/release/cross_codec --csv snappy.csv SNAPPY_DIR    # pinned to one CPU
./target/release/cross_codec --csv real.csv FILE...         # pinned to one CPU
./target/release/cross_codec --mt-only --mt-threads 32 --csv real.csv FILE...
./target/release/cross_codec --report snappy.csv real.csv
```

[Snappy benchmark set]: https://github.com/google/snappy/tree/main/testdata

</details>

#### Rust vs Go

Speed of this crate relative to the Go package (Rust MB/s / Go MB/s)
on the same buffers: the Go benchmark loads the crate as a `cdylib`
built with the release profile and alternates the two in one process.
Single-threaded runs are pinned to one core; each figure is the best of
7 samples of at least 100 ms.  Inputs are text, protobuf and CSV, plus
JSON for streams.  Ryzen 9 9950X, Windows 11.

| Workload                                            |               Rust / Go |
|-----------------------------------------------------|------------------------:|
| SIMD block decode, AVX-512                          |       1.07× (1.00–1.14) |
| SIMD block decode, AVX2                             |       1.02× (0.97–1.05) |
| SIMD block encode, AVX-512 and AVX2, vs Go parsers  |       1.07× (1.01–1.17) |
| SIMD block encode, SuperFast and Fastest, vs Go asm |              0.92–0.96× |
| Stream decode, 1 thread                             |              1.01–1.18× |
| Stream encode, 1 thread, vs Go parsers / Go asm     | 1.03–1.10× / 0.93–0.98× |
| Stream decode, 32 threads                           |              1.09–1.34× |
| Stream encode, 32 threads, vs Go parsers / Go asm   | 0.88–1.15× / 0.90–1.04× |
| XXH3 vs `zeebo/xxh3`, 4 KB / 64 KB / 4 MB           |   1.09× / 1.24× / 1.08× |
| XXH3 vs `zeebo/xxh3` with #35, same sizes           |   0.92× / 1.00× / 1.08× |

Single figures are geomeans, with the lowest and highest cell in
parentheses.  Blocks are 1 and 4 MB at levels 4–7 and 260–262.
Single-threaded streams use 4 MB blocks at levels 5 and 6;
multi-threaded streams use 2 MB blocks, ~1 GiB inputs and levels 2, 5
and 6.  "Go asm" is Go's default amd64 build, whose assembly parsers
are not ported; "Go parsers" are its portable parsers, which write the
same output and which Go uses in `noasm` / `purego` builds.  An
`MtWriter` keeps its worker threads and buffers for about 10 s after
`finish`, for the next one with the same concurrency and block size;
the 32-thread figures include that reuse.  Starting them fresh costs
35–67K page faults per 1 GiB stream on Windows, and makes 32-thread
encode 6–20% slower.  Multi-threaded results vary by up to ±10%
between runs.  XXH3 is compared with `zeebo/xxh3` master (86009ce)
and with its pending [PR #35](https://github.com/zeebo/xxh3/pull/35),
which speeds up its AVX2 and AVX-512 loops.  The XXH3 figures are
AVX-512; on AVX2, Rust is 0.97–0.99× of master and 0.91–0.96× of #35.

## CLI: `mz`

The `mz` binary at `bin/mz/` is a Rust port of the Go `cmd/mz` tool.

Build or install from a checkout of this repo:

```sh
cargo install --path bin/mz            # installs to ~/.cargo/bin/mz
cargo build --release -p mz            # or just build; binary at target/release/mz
```

```
mz c -2 input.json                     # compress to input.json.mz
mz c --classic -2 input.json           # classic blocks, for v1.0 decoders
mz d input.json.mz                     # decompress back to input.json
mz cat input.json.mz                   # decompress to stdout
mz bench -3 input.json                 # encode + decode N=5 times, print MB/s
mz d --offset=1G+nl input.json.mz      # seek to 1 GiB, advance to newline
mz verify input.json.mz                # validate without writing output
```

As in Go's `cmd/mz`, compression writes SIMD blocks by default:
`-xfast`, `-1`, `-2` (the default) and `-3` select
`Level::SuperFastSimd`, `FastestSimd`, `BalancedSimd` and
`SmallestSimd`.  `--classic` writes classic blocks, which every MinLZ
decoder can read: `Level::SuperFast` at `-xfast`, and `Fastest` to
`Smallest` at `-1` to `-3`.
`--simd.delta` adds delta literals at `-xfast`, `-1` and `-2`.  The
default block size is 8 MiB, as in Go.  `mz --help` lists every flag.

## Spec compliance

This crate implements the **MinLZ specification v1.0**, plus SIMD
blocks from v1.1: block type 1 ([SPEC_SIMD.md]), stream chunk type 0x10
with its XXH3 check, and the SIMD stream flag.  Features not
implemented (versus the Go reference):

- Snappy / S2 fallback decoding for streams whose first byte indicates a
  legacy magic.
- Block search tables (`SPEC.md` §4.13), and with them `mz search`.
- The `mz stats` and `mz d -block-debug` diagnostics.

## Development

```sh
cargo test --workspace --features minlz/reference  # 370+ unit, integration and doc tests
cargo doc --no-deps                  # rustdoc
cargo bench -p minlz --bench block   # criterion: block codec, classic and SIMD
cargo bench -p minlz --bench stream  # criterion: stream codec (incl. MT and seek)
cargo bench -p minlz --bench pivco   # criterion: SIMD entropy-coding kernels
cargo bench -p minlz --bench xxh3    # criterion: XXH3
```

The `reference` feature builds a clean-room reference decoder that the
tests and fuzzers use as an oracle.  The SIMD encoders are checked
against blocks written by the Go encoder, in
[`crates/minlz/tests/fixtures/simd.zip`](crates/minlz/tests/fixtures/simd.zip).
Further cross checks against the Go package run when these are set,
and are skipped otherwise:

- `MINLZ_SIMDGO` — the Go helper in [`_helpers/simdgo/`](_helpers/simdgo/)
  (`go build` there; it expects the Go repository next to this one, as
  `../minlz`).  `cargo test --release -p minlz --test simd_fixtures`
  then also encodes fresh inputs with Go at every SIMD level and
  compares the blocks byte for byte.
- `MINLZ_GO_CMD` — a Go `mz` binary (`go build ./cmd/mz` in the Go
  repository), for the stream cross checks in
  `cargo test -p minlz --features reference --lib stream` and the CLI
  ones in `cargo test -p mz`.
- `MINLZ_TESTDATA` — the Go repository's `testdata` directory: more
  inputs for the encoder tests, and Go's fuzz seeds.

`bash _helpers/arm64-test.sh [filter]` runs the library tests on
aarch64 (the NEON tiers) under QEMU in Docker, from Linux or Git Bash.

Fuzzing lives at [`crates/minlz/fuzz/`](crates/minlz/fuzz/) with nine
targets: block (`roundtrip`, `decode_arbitrary`, `simd_decode`), stream
(`stream_roundtrip`, `stream_decode_arbitrary`, `index_load`) and pivco
(`pivco_decode`, `pivco_roundtrip`, `pivco_table`).  Seeds come from the ignored
`write_fuzz_seeds` tests.  See its [`README.md`](crates/minlz/fuzz/README.md)
and [`RUNBOOK.md`](crates/minlz/fuzz/RUNBOOK.md) for setup.

MSRV: **Rust 1.89**.

## License

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) —
same as the upstream Go implementation.
