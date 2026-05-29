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

//! Bench subcommand: repeat compress + decompress on `<input>` and print
//! throughput.  Output format mirrors Go's `cmd/mz c --bench N` so a quick
//! `diff` against the Go binary is possible.

use std::io::{self, BufRead, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use minlz::stream::{MtWriterBuilder, Reader, WriterBuilder};
use minlz::Level;

use crate::args::Options;
use crate::io_util::{mb_per_sec, open_input, resolve_threads};

pub fn run(opts: Options, input: Option<PathBuf>) -> io::Result<()> {
    run_inner(&opts, input.as_deref(), BenchMode::Both)
}

/// Compress-only bench, invoked by `mz c --bench N`.
pub fn run_compress(opts: &Options, input: &std::path::Path) -> io::Result<()> {
    run_inner(opts, Some(input), BenchMode::Compress)
}

/// Decompress-only bench, invoked by `mz d --bench N`.
pub fn run_decompress(opts: &Options, input: &std::path::Path) -> io::Result<()> {
    run_inner(opts, Some(input), BenchMode::Decompress)
}

enum BenchMode {
    Compress,
    Decompress,
    Both,
}

fn run_inner(opts: &Options, input: Option<&std::path::Path>, mode: BenchMode) -> io::Result<()> {
    let input = input.ok_or_else(|| io::Error::other("no input file given"))?;
    let n = opts.bench_n.unwrap_or(5);
    let level = if opts.uncompressed {
        Level::Fastest
    } else {
        opts.level.unwrap_or(Level::Balanced)
    };
    let (mut src, _) = open_input(input)?;
    let mut data = Vec::new();
    src.read_to_end(&mut data)?;
    if !opts.quiet {
        eprintln!("Reading {}... ({} bytes)", input.display(), data.len());
    }

    let need_decode_loop = matches!(mode, BenchMode::Decompress | BenchMode::Both) || opts.verify;
    let need_encode_loop = matches!(mode, BenchMode::Compress | BenchMode::Both);
    let threads = resolve_threads(opts.threads).get();
    let block_size = opts
        .block_size
        .unwrap_or(crate::compress::CLI_DEFAULT_BLOCK_SIZE);

    // For decompress-only mode, the input IS the compressed stream — skip
    // the pre-encode + verify-roundtrip step entirely.
    let mut verify_buf: Vec<u8>;
    if matches!(mode, BenchMode::Decompress) {
        verify_buf = data.clone();
    } else {
        // Pre-encode once for the decode loop to verify byte-equality
        // without burning the entire iter budget on encode/decode
        // interleaving.
        verify_buf = Vec::with_capacity(data.len());
        if threads <= 1 {
            let wb = WriterBuilder::new().level(level).block_size(block_size);
            let mut w = wb.build(&mut verify_buf);
            w.write_all(&data)?;
            let _ = w.finish()?;
        } else {
            let wb = MtWriterBuilder::new()
                .level(level)
                .block_size(block_size)
                .concurrency(std::num::NonZeroUsize::new(threads).unwrap());
            let mut w = wb.build(std::mem::take(&mut verify_buf));
            w.write_all(&data)?;
            verify_buf = w.finish()?;
        }
        if opts.verify {
            let mut dec_verify = Vec::with_capacity(data.len());
            Reader::new(&verify_buf[..]).read_to_end(&mut dec_verify)?;
            if dec_verify != data {
                return Err(io::Error::other("bench: decoded content mismatch"));
            }
        }
    }
    let encoded_size = verify_buf.len() as u64;
    let uncompressed_size = if matches!(mode, BenchMode::Decompress) {
        // For decompress, the input bytes are compressed; the "data size"
        // for throughput is the decoded output size.  We don't know it
        // until we decode at least once.
        let mut probe = Vec::with_capacity(data.len() * 2);
        Reader::new(&verify_buf[..]).read_to_end(&mut probe)?;
        probe.len() as u64
    } else {
        data.len() as u64
    };

    let mut best_enc_mbps: f64 = 0.0;
    let mut best_dec_mbps: f64 = 0.0;
    let mut last_enc_size: u64 = encoded_size;
    for i in 0..n {
        let mut enc_mbps: Option<f64> = None;
        if need_encode_loop {
            // Encode: write into an owned buffer.  In ST mode we use the
            // borrowed-writer API; in MT mode the Writer owns the buffer.
            let mut buf;
            let start = Instant::now();
            if threads <= 1 {
                buf = Vec::with_capacity(verify_buf.len());
                let wb = WriterBuilder::new().level(level).block_size(block_size);
                let mut w = wb.build(&mut buf);
                w.write_all(&data)?;
                let _ = w.finish()?;
            } else {
                let wb = MtWriterBuilder::new()
                    .level(level)
                    .block_size(block_size)
                    .concurrency(std::num::NonZeroUsize::new(threads).unwrap());
                let mut w = wb.build(Vec::with_capacity(verify_buf.len()));
                w.write_all(&data)?;
                buf = w.finish()?;
            }
            let enc_elapsed = start.elapsed();
            last_enc_size = buf.len() as u64;
            let mbps = mb_per_sec(data.len() as u64, enc_elapsed);
            if mbps > best_enc_mbps {
                best_enc_mbps = mbps;
            }
            enc_mbps = Some(mbps);
            if opts.verify {
                // Re-decode and compare; matches Go's `c --bench --verify`.
                let mut dec = Vec::with_capacity(data.len());
                Reader::new(&buf[..]).read_to_end(&mut dec)?;
                if dec != data {
                    return Err(io::Error::other("bench: verify mismatch"));
                }
            }
        }

        let (dec_a_mbps, dec_b_mbps) = if need_decode_loop {
            // Decode (a): single-threaded io::copy → io::sink().
            let start = Instant::now();
            let mut sink = io::sink();
            let n_dec_a = io::copy(&mut Reader::new(&verify_buf[..]), &mut sink)?;
            let dec_a_elapsed = start.elapsed();
            if n_dec_a != uncompressed_size {
                return Err(io::Error::other("bench: decoded length mismatch"));
            }
            let dec_a_mbps = mb_per_sec(n_dec_a, dec_a_elapsed);

            // Decode (b): in MT mode, decode_concurrent → sink; in ST
            // mode, BufRead::fill_buf loop (skip the read→buf memcpy).
            let start = Instant::now();
            let n_dec_b: u64 = if threads > 1 {
                let mut reader = Reader::new(&verify_buf[..]);
                let (n, _w) = reader.decode_concurrent(io::sink(), threads)?;
                n
            } else {
                let mut reader = Reader::new(&verify_buf[..]);
                let mut acc: u64 = 0;
                loop {
                    let chunk_len = {
                        let buf = reader.fill_buf()?;
                        if buf.is_empty() {
                            break;
                        }
                        buf.len()
                    };
                    acc += chunk_len as u64;
                    reader.consume(chunk_len);
                }
                acc
            };
            let dec_b_elapsed = start.elapsed();
            if n_dec_b != uncompressed_size {
                return Err(io::Error::other("bench: alt-decode length mismatch"));
            }
            let dec_b_mbps = mb_per_sec(n_dec_b, dec_b_elapsed);
            let dec_mbps = dec_a_mbps.max(dec_b_mbps);
            if dec_mbps > best_dec_mbps {
                best_dec_mbps = dec_mbps;
            }
            (Some(dec_a_mbps), Some(dec_b_mbps))
        } else {
            (None, None)
        };

        if !opts.quiet {
            let pct = 100.0 * last_enc_size as f64 / uncompressed_size as f64;
            let ratio = uncompressed_size as f64 / last_enc_size as f64;
            let enc_str = enc_mbps
                .map(|v| format!("enc {v:.01}MB/s, "))
                .unwrap_or_default();
            let dec_str = match (dec_a_mbps, dec_b_mbps) {
                (Some(a), Some(b)) if threads > 1 => {
                    format!("dec(st) {a:.01}MB/s, dec(mt) {b:.01}MB/s")
                }
                (Some(a), Some(b)) => format!("dec(read) {a:.01}MB/s, dec(bufread) {b:.01}MB/s"),
                _ => String::new(),
            };
            eprintln!(
                "iter {}/{}: {} -> {} [{:.02}% {:.03}:1]; {enc_str}{dec_str}",
                i + 1,
                n,
                uncompressed_size,
                last_enc_size,
                pct,
                ratio,
            );
        }
    }
    if !opts.quiet {
        let mut parts = Vec::<String>::new();
        if need_encode_loop {
            parts.push(format!("enc {best_enc_mbps:.01}MB/s"));
        }
        if need_decode_loop {
            parts.push(format!("dec {best_dec_mbps:.01}MB/s"));
        }
        parts.push(format!(
            "ratio {:.03}:1",
            uncompressed_size as f64 / last_enc_size as f64
        ));
        eprintln!("best: {}", parts.join(", "));
    }
    Ok(())
}
