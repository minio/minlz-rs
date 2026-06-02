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

//! Compress subcommand.

use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use minlz::Level;
use minlz::stream::{MAX_BLOCK_SIZE, MtWriterBuilder, WriterBuilder};

use crate::args::Options;
use crate::io_util::{
    CountingReader, CountingWriter, mb_per_sec, open_input, open_output, open_output_send,
    resolve_threads,
};

const EXT_STREAM: &str = "mz";
const EXT_BLOCK: &str = "mzb";

/// CLI default block size — matches Go `cmd/mz`'s `-bs 8M` default.
/// (Library default in `stream::DEFAULT_BLOCK_SIZE` is 2 MiB.)
pub(crate) const CLI_DEFAULT_BLOCK_SIZE: usize = 8 << 20;

pub fn run(opts: Options, input: Option<PathBuf>) -> io::Result<()> {
    let input = input.ok_or_else(|| io::Error::other("no input file given (use - for stdin)"))?;

    // `c --bench N` runs the throughput bench instead of writing a file —
    // matches Go's `cmd/mz/compress.go::runBenchStream` behaviour.
    if opts.bench_n.is_some() {
        return crate::bench::run_compress(&opts, &input);
    }

    if opts.block {
        compress_block(&input, &opts)
    } else {
        compress_stream(&input, &opts)
    }
}

fn compress_stream(input: &Path, opts: &Options) -> io::Result<()> {
    let threads = resolve_threads(opts.threads).get();
    if threads <= 1 {
        compress_stream_st(input, opts)
    } else {
        compress_stream_mt(input, opts, threads)
    }
}

fn compress_stream_st(input: &Path, opts: &Options) -> io::Result<()> {
    let (mut src, _src_size) = open_input(input)?;
    let dst_path = dest_path(input, opts, EXT_STREAM);
    let dst = open_output(&dst_path)?;

    let mut counted_src = CountingReader::new(&mut src);
    let mut counted_dst = CountingWriter::new(dst);

    let mut builder = WriterBuilder::new();
    if opts.uncompressed {
        builder = builder.uncompressed();
    } else if let Some(lvl) = opts.level {
        builder = builder.level(lvl);
    }
    builder = builder.block_size(opts.block_size.unwrap_or(CLI_DEFAULT_BLOCK_SIZE));
    if let Some(pad) = opts.padding {
        builder = builder.padding(pad);
    }
    if opts.index {
        builder = builder.append_index();
    } else {
        builder = builder.generate_index(false);
    }

    if !opts.quiet {
        eprint!("Compressing {} -> {}", input.display(), dst_path.display());
    }
    let start = Instant::now();
    let mut w = builder.build(&mut counted_dst)?;
    io::copy(&mut counted_src, &mut w)?;
    let _ = w.finish()?;
    counted_dst.flush()?;
    let elapsed = start.elapsed();
    print_compress_summary(opts, counted_src.bytes, counted_dst.bytes, elapsed);
    if opts.remove && input != Path::new("-") {
        drop(src);
        std::fs::remove_file(input)?;
        if !opts.quiet {
            eprintln!("Removing {}", input.display());
        }
    }
    Ok(())
}

fn compress_stream_mt(input: &Path, opts: &Options, threads: usize) -> io::Result<()> {
    let (mut src, _src_size) = open_input(input)?;
    let dst_path = dest_path(input, opts, EXT_STREAM);
    let dst = open_output_send(&dst_path)?;

    let mut builder = MtWriterBuilder::new();
    if opts.uncompressed {
        builder = builder.uncompressed();
    } else if let Some(lvl) = opts.level {
        builder = builder.level(lvl);
    }
    builder = builder
        .block_size(opts.block_size.unwrap_or(CLI_DEFAULT_BLOCK_SIZE))
        .concurrency(threads);
    if let Some(pad) = opts.padding {
        builder = builder.padding(pad);
    }
    if opts.index {
        builder = builder.append_index();
    } else {
        builder = builder.generate_index(false);
    }

    if !opts.quiet {
        eprint!("Compressing {} -> {}", input.display(), dst_path.display());
    }
    let start = Instant::now();
    let mut w = builder.build(dst)?;
    // We can't wrap the underlying writer in a counter (it's owned by
    // the MT writer thread); track input bytes via the reader-side.
    let mut counted_src = CountingReader::new(&mut src);
    io::copy(&mut counted_src, &mut w)?;
    let mut out = w.finish()?;
    out.flush()?;
    let elapsed = start.elapsed();
    // For output size, ask the filesystem (we wrote to a real file).
    let out_size = if dst_path == Path::new("-") {
        // stdout: best-effort approximation (unknown).
        0
    } else {
        std::fs::metadata(&dst_path).map(|m| m.len()).unwrap_or(0)
    };
    print_compress_summary(opts, counted_src.bytes, out_size, elapsed);
    if opts.remove && input != Path::new("-") {
        drop(src);
        std::fs::remove_file(input)?;
        if !opts.quiet {
            eprintln!("Removing {}", input.display());
        }
    }
    Ok(())
}

fn compress_block(input: &Path, opts: &Options) -> io::Result<()> {
    let (mut src, src_size) = open_input(input)?;
    if let Some(sz) = src_size {
        if sz > MAX_BLOCK_SIZE as u64 {
            return Err(io::Error::other("maximum block size of 8 MiB exceeded"));
        }
    }
    let mut buf = Vec::with_capacity(src_size.unwrap_or(0) as usize);
    src.read_to_end(&mut buf)?;
    if buf.len() > MAX_BLOCK_SIZE {
        return Err(io::Error::other("maximum block size of 8 MiB exceeded"));
    }
    let dst_path = dest_path(input, opts, EXT_BLOCK);
    let mut dst = open_output(&dst_path)?;
    let level = if opts.uncompressed {
        Level::Fastest // block::encode has no uncompressed mode; the encoder
    // will fall back to the inline literal block if the
    // data doesn't compress, which is the closest analog.
    } else {
        opts.level.unwrap_or(Level::Balanced)
    };
    if !opts.quiet {
        eprint!("Compressing {} -> {}", input.display(), dst_path.display());
    }
    let start = Instant::now();
    let mut enc = Vec::with_capacity(minlz::max_encoded_len(buf.len()).unwrap_or(buf.len() + 16));
    minlz::encode(&mut enc, &buf, level).map_err(io::Error::other)?;
    dst.write_all(&enc)?;
    dst.flush()?;
    let elapsed = start.elapsed();
    print_compress_summary(opts, buf.len() as u64, enc.len() as u64, elapsed);
    if opts.verify {
        let mut dec = Vec::with_capacity(buf.len());
        minlz::decode(&mut dec, &enc).map_err(io::Error::other)?;
        if dec != buf {
            return Err(io::Error::other("verify: decoded content mismatch"));
        }
        if !opts.quiet {
            eprintln!("... Verified ok.");
        }
    }
    if opts.remove && input != Path::new("-") {
        std::fs::remove_file(input)?;
        if !opts.quiet {
            eprintln!("Removing {}", input.display());
        }
    }
    Ok(())
}

/// Decide the destination path given `input`, `--output`, `--stdout` and the
/// target extension.
pub fn dest_path(input: &Path, opts: &Options, ext: &str) -> PathBuf {
    if opts.stdout {
        return PathBuf::from("-");
    }
    if let Some(out) = opts.output.as_ref() {
        return out.clone();
    }
    if input == Path::new("-") {
        return PathBuf::from("-");
    }
    let mut p = input.as_os_str().to_owned();
    p.push(".");
    p.push(ext);
    PathBuf::from(p)
}

fn print_compress_summary(opts: &Options, input: u64, output: u64, elapsed: std::time::Duration) {
    if opts.quiet {
        return;
    }
    if input == 0 {
        eprintln!(" {input} -> {output}; 0.0MB/s");
        return;
    }
    let pct = 100.0 * output as f64 / input as f64;
    let ratio = input as f64 / output as f64;
    let mbps = mb_per_sec(input, elapsed);
    eprintln!(" {input} -> {output} [{pct:.02}% {ratio:.03}:1]; {mbps:.01}MB/s");
}
