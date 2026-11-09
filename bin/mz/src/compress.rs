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
use std::thread::{self, JoinHandle};
use std::time::Instant;

use minlz::stream::{MAX_BLOCK_SIZE, MtWriterBuilder, Reader, WriterBuilder};

use crate::args::Options;
use crate::io_util::{
    CountingReader, CountingWriter, check_not_input, mb_per_sec, open_input, open_output,
    open_output_send, resolve_threads,
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
    let dst_path = dest_path(input, opts, EXT_STREAM)?;
    let (dst, verifier) = verify_to(open_output(&dst_path)?, opts.verify)?;

    let mut counted_src = CountingReader::new(&mut src);
    let mut counted_dst = CountingWriter::new(dst);

    let mut builder = WriterBuilder::new();
    builder = match opts.level {
        Some(lvl) => builder.level(lvl),
        None => builder.uncompressed(),
    };
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
    drop(counted_dst);
    check_verify(verifier, opts)?;
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
    let dst_path = dest_path(input, opts, EXT_STREAM)?;
    let (dst, verifier) = verify_to(open_output_send(&dst_path)?, opts.verify)?;

    let mut builder = MtWriterBuilder::new();
    builder = match opts.level {
        Some(lvl) => builder.level(lvl),
        None => builder.uncompressed(),
    };
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
    let mut w = builder.build(CountingWriter::new(dst))?;
    let mut counted_src = CountingReader::new(&mut src);
    io::copy(&mut counted_src, &mut w)?;
    let mut out = w.finish()?;
    out.flush()?;
    let elapsed = start.elapsed();
    print_compress_summary(opts, counted_src.bytes, out.bytes, elapsed);
    drop(out);
    check_verify(verifier, opts)?;
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
    let (src, src_size) = open_input(input)?;
    if let Some(sz) = src_size
        && sz > MAX_BLOCK_SIZE as u64
    {
        return Err(io::Error::other("maximum block size of 8 MiB exceeded"));
    }
    let mut buf = Vec::with_capacity(src_size.unwrap_or(0) as usize);
    // Stdin has no size to check up front; one byte past the limit is enough to reject it.
    src.take(MAX_BLOCK_SIZE as u64 + 1).read_to_end(&mut buf)?;
    if buf.len() > MAX_BLOCK_SIZE {
        return Err(io::Error::other("maximum block size of 8 MiB exceeded"));
    }
    let dst_path = dest_path(input, opts, EXT_BLOCK)?;
    let mut dst = open_output(&dst_path)?;
    if !opts.quiet {
        eprint!("Compressing {} -> {}", input.display(), dst_path.display());
    }
    let start = Instant::now();
    let mut enc = Vec::with_capacity(minlz::max_encoded_len(buf.len()).unwrap_or(buf.len() + 16));
    match opts.level {
        Some(level) => minlz::encode(&mut enc, &buf, level).map_err(io::Error::other)?,
        // Go's encodeUncompressed; `Level` has no uncompressed variant.
        None if buf.is_empty() => enc.push(0),
        None => {
            enc.extend_from_slice(&[0, 0]);
            enc.extend_from_slice(&buf);
        }
    }
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

type Verifier = Option<JoinHandle<io::Result<u64>>>;

/// Go's `verifyTo`: decodes the output on another thread while it is written. Every chunk
/// carries a checksum of its input, so a clean decode verifies it.
fn verify_to<W: Write>(out: W, verify: bool) -> io::Result<(Tee<W>, Verifier)> {
    if !verify {
        return Ok((Tee { out, pipe: None }, None));
    }
    let (mut r, w) = io::pipe()?;
    let h = thread::spawn(move || {
        let res = io::copy(&mut Reader::new(&mut r), &mut io::sink());
        // Keep draining so a decode error is reported by check_verify, not as a broken pipe
        // in the writer.
        io::copy(&mut r, &mut io::sink())?;
        res
    });
    Ok((Tee { out, pipe: Some(w) }, Some(h)))
}

/// Waits for the decoder of [`verify_to`]. The output must be dropped first, to close the
/// pipe.
fn check_verify(verifier: Verifier, opts: &Options) -> io::Result<()> {
    let Some(h) = verifier else { return Ok(()) };
    h.join()
        .unwrap_or_else(|e| std::panic::resume_unwind(e))
        .map_err(|e| io::Error::other(format!("verify: {e}")))?;
    if !opts.quiet {
        eprintln!("... Verified ok.");
    }
    Ok(())
}

/// Copies what is written to `out` into the verify pipe, if any.
struct Tee<W> {
    out: W,
    pipe: Option<io::PipeWriter>,
}

impl<W: Write> Write for Tee<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.out.write(buf)?;
        if let Some(p) = &mut self.pipe {
            p.write_all(&buf[..n])?;
        }
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

/// Decide the destination path given `input`, `--output`, `--stdout` and the
/// target extension.
pub fn dest_path(input: &Path, opts: &Options, ext: &str) -> io::Result<PathBuf> {
    if opts.stdout {
        return Ok(PathBuf::from("-"));
    }
    if let Some(out) = opts.output.as_ref() {
        check_not_input(input, out)?;
        return Ok(out.clone());
    }
    if input == Path::new("-") {
        return Ok(PathBuf::from("-"));
    }
    let mut p = input.as_os_str().to_owned();
    p.push(".");
    p.push(ext);
    Ok(PathBuf::from(p))
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
