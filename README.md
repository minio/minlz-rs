# MinLZ — Rust

[![Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue)](#license)

A fast LZ77-style block / streaming compressor — Rust port of
[`github.com/minio/minlz`](https://github.com/minio/minlz).

This implements the **MinLZ specification v1.0** (block + stream codec,
including the optional stream index for random access).  Block search
tables (`SPEC.md` §4.13) are **not** implemented.

## What you get

- **Three compression levels** — `Level::Fastest` (1), `Level::Balanced`
  (2, default), `Level::Smallest` (3).
- **Block codec** — `encode` / `decode` / `try_encode` / `append_encoded`
  / `append_decoded` / `decoded_len` / `is_minlz` / `max_encoded_len`.
  Blocks are capped at 8 MiB per the spec.
- **Stream codec** — `stream::Reader<R: Read>` / `stream::Writer<W: Write>`,
  multi-block framing with CRC32C integrity per block.
- **Multi-threaded streaming** — `stream::MtWriter<W>` parallelises
  encoding across worker threads; `Reader::decode_concurrent` parallelises
  decoding.
- **Random access** — `index::Index` plus `stream::ReadSeeker` give you
  uncompressed-offset seeks against any `Read + Seek` source.  The index
  can be appended to the stream, stored as a sidecar, or recovered
  post-hoc with `index::index_stream` over a sequential reader.
- **User-defined chunks** — embed metadata inline in a stream
  (chunk IDs 0x80–0xFD, skippable and non-skippable).
- **CRC32C** — hardware-accelerated on x86_64 (SSE4.2) and aarch64
  (CRC32 extension), with a portable table fallback.
- **Safe public API** — `unsafe` is confined to the hot path of the
  block decoder and a few load/store helpers; the crate itself is
  `#![deny(missing_docs)]` and CI runs Clippy + Miri + cargo-fuzz.

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

Streams are independent blocks with framing, CRC32C, an EOF marker, and
(optionally) an appended index.  Both `Writer` and `Reader` are
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

| Level                 | Use when…                                             | Notes                                                                                   |
|-----------------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `Level::Fastest`  (1) | Throughput is the constraint.                         | Highest encode speed; modest ratio.                                                     |
| `Level::Balanced` (2) | The default — what you want unless you have a reason. | ~50 % of L1's encode speed; meaningfully better ratio.                                  |
| `Level::Smallest` (3) | Archival; output is read many times.                  | Encode is roughly an order of magnitude slower than L2.  Decode speed is similar to L2. |

A complementary knob is `WriterBuilder::block_size` (8 KiB–8 MiB,
default 2 MiB).  Smaller blocks trade ratio for lower memory and
slightly higher per-block overhead; larger blocks give the L2/L3
encoders more matching range.

## Performance

#### Snappy benchmark set, geomean across 11 files

| Codec       | Avg ratio | Encode MB/s | Decode MB/s |
|-------------|----------:|------------:|------------:|
| **minlz-1** | **2.78x** |     **740** |   **3 305** |
| **minlz-2** | **3.12x** |     **631** |   **3 773** |
| **minlz-3** | **3.40x** |      **62** |   **3 178** |
| snappy      |     2.17x |         901 |       1 789 |
| lz4_flex    |     2.23x |         816 |       1 476 |
| gzip-1      |     2.50x |         271 |         538 |
| zstd-1      |     3.49x |         543 |       1 722 |

#### Real-world workloads, first 1 GiB of each (geomean across 5 files)

CockroachDB log, GitHub events JSON, GitHub ranks binary, NYC taxi CSV,
VM Image.

| Codec                | Avg ratio | Encode MB/s | Decode MB/s |
|----------------------|----------:|------------:|------------:|
| **minlz-1, MT × 32** | **4.45x** |  **10 190** |  **27 181** |
| **minlz-2, MT × 32** | **4.89x** |   **7 647** |  **28 543** |
| **minlz-3, MT × 32** | **5.54x** |     **805** |  **29 375** |
| **minlz-1**          | **4.45x** |   **1 399** |   **3 661** |
| **minlz-2**          | **4.89x** |     **835** |   **3 289** |
| **minlz-3**          | **5.54x** |      **97** |   **3 667** |
| snappy               |     3.31x |       1 325 |       2 637 |
| lz4_flex             |     3.65x |       1 322 |       6 499 |
| gzip-1               |     4.03x |         447 |         715 |
| zstd-1               |     5.83x |       1 020 |       2 631 |

The MT rows use `stream::MtWriter` for encode and
`Reader::decode_concurrent` for decode at 32 worker threads.  Encode
scales 7–9× over the single-threaded baseline; decode scales 7–8× and
sustains ~27–29 GB/s into an `io::sink()` target.

Headlines: at **Fastest**, MinLZ matches Snappy/LZ4 encode speed with
substantially better ratio (2.78× → 2.17× on the Snappy set; 4.45× →
3.31× on real workloads) and roughly doubles their decode speed against
Snappy.  LZ4 wins raw decode throughput on large workloads; everything
else trails MinLZ.  **Balanced** lands near zstd-1 on speed but with
mid-range LZ ratio.  **Smallest** trades encoder throughput for ratios
near zstd-1 while keeping decode close to MinLZ-1.

<details>
<summary><b>Individual Results</b></summary>


### alice29.txt (0.15 MiB)

| codec      |    enc size | ratio  | enc MB/s | dec MB/s |
|------------|------------:|-------:|---------:|---------:|
| minlz-1    |      81,529 |  1.87x |    444.6 |   1791.4 |
| minlz-2    |      69,003 |  2.20x |    334.0 |   1736.2 |
| minlz-3    |      62,524 |  2.43x |     30.2 |   1889.3 |
| snappy     |      88,074 |  1.73x |    514.7 |   1208.0 |
| lz4_flex   |      88,702 |  1.71x |    505.3 |   1817.1 |
| gzip-1     |      76,497 |  1.99x |    169.3 |    305.9 |
| zstd-1     |      61,069 |  2.49x |    316.5 |   1841.3 |

### asyoulik.txt (0.12 MiB)

| codec      |    enc size | ratio  | enc MB/s | dec MB/s |
|------------|------------:|-------:|---------:|---------:|
| minlz-1    |      75,380 |  1.66x |    411.8 |   1412.9 |
| minlz-2    |      62,955 |  1.99x |    311.9 |   1551.2 |
| minlz-3    |      57,951 |  2.16x |     26.6 |   1710.1 |
| snappy     |      77,532 |  1.61x |    461.2 |   1122.7 |
| lz4_flex   |      79,831 |  1.57x |    558.6 |   1526.6 |
| gzip-1     |      64,955 |  1.93x |    158.6 |    280.8 |
| zstd-1     |      54,665 |  2.29x |    282.6 |   1940.8 |

### fireworks.jpeg (0.12 MiB)

| codec      |    enc size | ratio  | enc MB/s | dec MB/s |
|------------|------------:|-------:|---------:|---------:|
| minlz-1    |     123,118 |  1.00x |   1197.4 |   4049.1 |
| minlz-2    |     123,118 |  1.00x |   1778.8 |  26759.3 |
| minlz-3    |     123,046 |  1.00x |    266.4 |   2097.0 |
| snappy     |     123,119 |  1.00x |   1497.5 |   2766.1 |
| lz4_flex   |     123,108 |  1.00x |   1204.4 |   3588.7 |
| gzip-1     |     123,043 |  1.00x |    330.1 |    559.3 |
| zstd-1     |     123,102 |  1.00x |    552.2 |   2404.2 |

### geo.protodata (0.11 MiB)

| codec      |    enc size | ratio  | enc MB/s | dec MB/s |
|------------|------------:|-------:|---------:|---------:|
| minlz-1    |      17,503 |  6.78x |   1098.0 |  12890.0 |
| minlz-2    |      16,369 |  7.24x |   1030.3 |  12225.6 |
| minlz-3    |      14,790 |  8.02x |     83.8 |  12890.0 |
| snappy     |      23,364 |  5.08x |   1726.2 |   1940.9 |
| lz4_flex   |      19,472 |  6.09x |   1057.9 |   1983.1 |
| gzip-1     |      21,196 |  5.59x |    419.9 |   1638.0 |
| zstd-1     |      14,571 |  8.14x |    745.4 |   1728.7 |

### html (0.10 MiB)

| codec      |    enc size | ratio  | enc MB/s | dec MB/s |
|------------|------------:|-------:|---------:|---------:|
| minlz-1    |      19,873 |  5.15x |    925.9 |   9309.1 |
| minlz-2    |      17,855 |  5.74x |    706.2 |   9061.9 |
| minlz-3    |      16,049 |  6.38x |     66.3 |   9941.7 |
| snappy     |      22,872 |  4.48x |   1331.6 |   1822.1 |
| lz4_flex   |      21,341 |  4.80x |    977.1 |   1802.8 |
| gzip-1     |      18,762 |  5.46x |    365.6 |   1241.2 |
| zstd-1     |      15,408 |  6.65x |    577.6 |   1508.1 |

### html_x_4 (0.39 MiB)

| codec      |    enc size | ratio  | enc MB/s | dec MB/s |
|------------|------------:|-------:|---------:|---------:|
| minlz-1    |      19,876 | 20.61x |   2019.7 |  19692.3 |
| minlz-2    |      17,862 | 22.93x |   1962.6 |  17964.9 |
| minlz-3    |      16,056 | 25.51x |    281.0 |  19412.3 |
| snappy     |      92,318 |  4.44x |   1399.9 |   3875.1 |
| lz4_flex   |      83,835 |  4.89x |   1458.7 |    884.3 |
| gzip-1     |      74,838 |  5.47x |    489.6 |    916.3 |
| zstd-1     |      15,458 | 26.50x |   2244.4 |   3624.8 |

### kppkn.gtb (0.18 MiB)

| codec      |    enc size | ratio  | enc MB/s | dec MB/s |
|------------|------------:|-------:|---------:|---------:|
| minlz-1    |      62,111 |  2.97x |    560.4 |   2500.9 |
| minlz-2    |      52,776 |  3.49x |    474.4 |   2679.1 |
| minlz-3    |      46,026 |  4.00x |     48.1 |   2884.5 |
| snappy     |      69,566 |  2.65x |    784.7 |   1472.2 |
| lz4_flex   |      73,070 |  2.52x |    680.4 |   1793.0 |
| gzip-1     |      57,660 |  3.20x |    271.3 |    416.5 |
| zstd-1     |      39,334 |  4.69x |    434.7 |   1111.0 |

### lcet10.txt (0.41 MiB)

| codec      |    enc size | ratio  | enc MB/s | dec MB/s |
|------------|------------:|-------:|---------:|---------:|
| minlz-1    |     202,580 |  2.11x |    467.7 |   1411.2 |
| minlz-2    |     173,653 |  2.46x |    377.1 |   1445.2 |
| minlz-3    |     154,555 |  2.76x |     41.7 |   1500.0 |
| snappy     |     234,745 |  1.82x |    571.5 |   1544.5 |
| lz4_flex   |     233,299 |  1.83x |    540.0 |    822.3 |
| gzip-1     |     208,361 |  2.05x |    188.8 |    321.5 |
| zstd-1     |     158,517 |  2.69x |    404.4 |   1144.7 |

### paper-100k.pdf (0.10 MiB)

| codec      |    enc size | ratio  | enc MB/s | dec MB/s |
|------------|------------:|-------:|---------:|---------:|
| minlz-1    |      84,062 |  1.22x |    972.5 |   2694.7 |
| minlz-2    |      82,749 |  1.24x |    779.9 |   2137.8 |
| minlz-3    |      82,295 |  1.24x |     35.6 |   2133.3 |
| snappy     |      85,327 |  1.20x |   1486.2 |   2178.7 |
| lz4_flex   |      83,541 |  1.23x |   1265.8 |   1503.7 |
| gzip-1     |      82,476 |  1.24x |    348.5 |    638.0 |
| zstd-1     |      83,768 |  1.22x |    590.2 |   1932.1 |

### plrabn12.txt (0.46 MiB)

| codec      |    enc size | ratio  | enc MB/s | dec MB/s |
|------------|------------:|-------:|---------:|---------:|
| minlz-1    |     287,399 |  1.68x |    440.1 |   1254.5 |
| minlz-2    |     242,611 |  1.99x |    323.3 |   1145.9 |
| minlz-3    |     218,619 |  2.20x |     38.1 |   1254.5 |
| snappy     |     319,362 |  1.51x |    473.3 |   1235.5 |
| lz4_flex   |     325,580 |  1.48x |    536.1 |    884.5 |
| gzip-1     |     264,498 |  1.82x |    164.0 |    291.0 |
| zstd-1     |     217,838 |  2.21x |    392.3 |   1180.5 |

### urls.10K (0.67 MiB)

| codec      |    enc size | ratio  | enc MB/s | dec MB/s |
|------------|------------:|-------:|---------:|---------:|
| minlz-1    |     260,961 |  2.69x |    723.4 |   1779.7 |
| minlz-2    |     230,304 |  3.05x |    527.0 |   1619.2 |
| minlz-3    |     207,327 |  3.39x |     53.2 |   1708.7 |
| snappy     |     335,620 |  2.09x |    878.0 |   1906.8 |
| lz4_flex   |     336,026 |  2.09x |    834.6 |   1176.6 |
| gzip-1     |     283,675 |  2.47x |    302.9 |    492.8 |
| zstd-1     |     208,206 |  3.37x |    622.1 |   1673.2 |

### cockroach.node1.log (1024.00 MiB)

| codec        |    enc size |  ratio | enc MB/s | dec MB/s |
|--------------|------------:|-------:|---------:|---------:|
| minlz-1 (MT) |  87,251,010 | 12.31x |  12751.3 |  34120.3 |
| minlz-2 (MT) |  81,602,796 | 13.16x |  10482.3 |  40264.8 |
| minlz-3 (MT) |  66,351,239 | 16.18x |   2030.1 |  43143.3 |
| minlz-1      |  87,251,010 | 12.31x |   2632.8 |   5652.8 |
| minlz-2      |  81,602,796 | 13.16x |   1591.7 |   5453.1 |
| minlz-3      |  66,351,239 | 16.18x |    192.8 |   6634.5 |
| snappy       | 153,202,881 |  7.01x |   2295.9 |   3836.9 |
| lz4_flex     | 116,439,097 |  9.22x |   2090.2 |   9272.9 |
| gzip-1       | 111,321,553 |  9.65x |    886.2 |   1441.0 |
| zstd-1       |  57,692,789 | 18.61x |   1899.0 |   4554.3 |

### github-june-2days-2019.json (1024.00 MiB)

| codec        |    enc size | ratio | enc MB/s | dec MB/s |
|--------------|------------:|------:|---------:|---------:|
| minlz-1 (MT) | 161,415,425 | 6.65x |  12000.7 |  34678.3 |
| minlz-2 (MT) | 153,662,320 | 6.99x |   8781.6 |  35534.6 |
| minlz-3 (MT) | 133,119,515 | 8.07x |   1106.3 |  34744.4 |
| minlz-1      | 161,415,425 | 6.65x |   1750.3 |   4361.5 |
| minlz-2      | 153,662,320 | 6.99x |   1023.4 |   3985.1 |
| minlz-3      | 133,119,515 | 8.07x |    121.2 |   4929.8 |
| snappy       | 259,082,296 | 4.14x |   1430.6 |   2747.3 |
| lz4_flex     | 227,398,199 | 4.72x |   1341.3 |   6375.6 |
| gzip-1       | 230,645,392 | 4.66x |    470.9 |    753.8 |
| zstd-1       | 129,752,397 | 8.28x |   1251.1 |   3286.1 |

### github-ranks-backup.bin (1024.00 MiB)

| codec        |    enc size | ratio | enc MB/s | dec MB/s |
|--------------|------------:|------:|---------:|---------:|
| minlz-1 (MT) | 350,786,224 | 3.06x |   9937.1 |  25762.2 |
| minlz-2 (MT) | 306,255,841 | 3.51x |   6604.8 |  26770.5 |
| minlz-3 (MT) | 285,433,540 | 3.76x |    527.8 |  26953.1 |
| minlz-1      | 350,786,224 | 3.06x |   1293.5 |   3557.8 |
| minlz-2      | 306,255,841 | 3.51x |    675.7 |   2775.0 |
| minlz-3      | 285,433,540 | 3.76x |     72.5 |   2788.8 |
| snappy       | 352,063,664 | 3.05x |   1356.1 |   2777.1 |
| lz4_flex     | 365,716,326 | 2.94x |   1407.0 |   6253.7 |
| gzip-1       | 287,570,030 | 3.73x |    470.9 |    794.9 |
| zstd-1       | 273,678,613 | 3.92x |    924.2 |   2309.6 |

### nyc-taxi-data-10M.csv (1024.00 MiB)

| codec        |    enc size | ratio | enc MB/s | dec MB/s |
|--------------|------------:|------:|---------:|---------:|
| minlz-1 (MT) | 299,283,039 | 3.59x |   8565.8 |  27149.5 |
| minlz-2 (MT) | 255,081,991 | 4.21x |   7141.0 |  27931.8 |
| minlz-3 (MT) | 219,709,751 | 4.89x |    709.6 |  29522.7 |
| minlz-1      | 299,283,039 | 3.59x |    772.7 |   2365.3 |
| minlz-2      | 255,081,991 | 4.21x |    599.7 |   2481.2 |
| minlz-3      | 219,709,751 | 4.89x |     79.3 |   2727.2 |
| snappy       | 428,270,599 | 2.51x |    774.2 |   1730.6 |
| lz4_flex     | 392,333,490 | 2.74x |    838.2 |   5620.8 |
| gzip-1       | 363,226,950 | 2.96x |    270.0 |    420.7 |
| zstd-1       | 222,461,279 | 4.83x |    655.1 |   1938.9 |

### rawstudio-mint14.tar (1024.00 MiB)

| codec        |    enc size | ratio | enc MB/s | dec MB/s |
|--------------|------------:|------:|---------:|---------:|
| minlz-1 (MT) | 556,167,773 | 1.93x |   8434.6 |  17928.1 |
| minlz-2 (MT) | 520,983,163 | 2.06x |   6023.8 |  17707.1 |
| minlz-3 (MT) | 494,650,520 | 2.17x |    401.8 |  18336.0 |
| minlz-1      | 556,167,773 | 1.93x |   1165.2 |   3169.1 |
| minlz-2      | 520,983,163 | 2.06x |    614.4 |   2572.6 |
| minlz-3      | 494,650,520 | 2.17x |     63.9 |   2665.3 |
| snappy       | 604,501,920 | 1.78x |   1185.6 |   2514.7 |
| lz4_flex     | 584,342,866 | 1.84x |   1221.8 |   5579.3 |
| gzip-1       | 504,957,412 | 2.13x |    334.9 |    515.7 |
| zstd-1       | 465,496,019 | 2.31x |    767.7 |   1880.3 |

</details>


<details>
<summary><b>Methodology</b></summary>

In-memory only; each input loaded once into a `Vec<u8>`, then encode
through each codec's streaming / frame API (so framing overhead and CRC
checks are included).  Decode timing targets `io::sink()` — for MinLZ
ST that is a `BufRead::fill_buf` loop, for MinLZ MT it is
`decode_concurrent(io::sink(), 32)`, for the rest it is
`io::copy(decoder, &mut io::sink())`.  This mirrors what
[`mz bench`](bin/mz/src/bench.rs) reports (`dec(st)` / `dec(mt)`).
Each variant is verified once via a full decode-to-`Vec` byte compare
against the source; the verify pass is excluded from the timed loop.
Best of 3 iterations per (codec, input).  The MinLZ stream writer at
concurrency = 1 / 32.  Each codec uses its default block / frame size:
MinLZ 2 MiB, Snappy 64 KiB (spec-mandated), `lz4_flex` 64 KiB, gzip
32 KiB DEFLATE window, zstd's per-level default window (128 KiB at L1).
Build: `cargo --release` with workspace LTO (`lto = "thin"`,
`codegen-units = 1`).


Hardware: AMD Ryzen 9 9950X (16C / 32T, Zen 5), 64 GiB DDR5, Windows 11.

Corpus: 11 files from the [Snappy benchmark set] plus the first 1 GiB
of each of five real-world inputs.  Full per-file breakdowns live in
[`_helpers/cross_codec/results.md`](_helpers/cross_codec/results.md);
the harness itself is at
[`_helpers/cross_codec/`](_helpers/cross_codec/) and rebuilds with
`cargo build --release` from that directory.  Re-run MT with
`./target/release/cross_codec --mt-only --mt-threads 32 --md PATH...`.

[Snappy benchmark set]: https://github.com/google/snappy/tree/main/testdata

</details>

## CLI: `mz`

The `mz` binary at `bin/mz/` is a Rust port of the Go `cmd/mz` tool.

Build or install from a checkout of this repo:

```sh
cargo install --path bin/mz            # installs to ~/.cargo/bin/mz
cargo build --release -p mz            # or just build; binary at target/release/mz
```

```
mz c -2 input.json                     # compress to input.json.mz
mz d input.json.mz                     # decompress back to input.json
mz cat input.json.mz                   # decompress to stdout
mz bench -3 input.json                 # encode + decode N=5 times, print MB/s
mz d --offset=1G+nl input.json.mz      # seek to 1 GiB, advance to newline
mz verify input.json.mz                # validate without writing output
```

`mz --help` lists every flag.

## Spec compliance

This crate implements the **MinLZ specification v1.0**.  Subset
features not implemented (versus the Go reference):

- Snappy / S2 fallback decoding for streams whose first byte indicates a
  legacy magic.
- Block search tables (`SPEC.md` §4.13).

## Development

```sh
cargo test --workspace               # 200+ unit + integration tests
cargo doc --no-deps                  # rustdoc
cargo bench -p minlz --bench block   # criterion: block codec
cargo bench -p minlz --bench stream  # criterion: stream codec (incl. MT and seek)
```

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
