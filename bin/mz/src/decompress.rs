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

//! Decompress subcommand.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use minlz::stream::{ConcurrentDecode, ReadSeeker, Reader};

use crate::args::Options;
use crate::io_util::{
    CountingReader, CountingWriter, mb_per_sec, open_input, open_output, open_output_send,
    resolve_threads,
};

pub fn run(opts: Options, input: Option<PathBuf>) -> io::Result<()> {
    let input = input.ok_or_else(|| io::Error::other("no input file given (use - for stdin)"))?;
    if opts.bench_n.is_some() {
        return crate::bench::run_decompress(&opts, &input);
    }
    let is_block = opts.block || extension_matches(&input, "mzb");
    if is_block {
        decompress_block(&input, &opts)
    } else {
        decompress_stream(&input, &opts)
    }
}

fn decompress_stream(input: &Path, opts: &Options) -> io::Result<()> {
    let threads = resolve_threads(opts.threads).get();
    if opts.follow {
        return decompress_stream_follow(input, opts);
    }
    // Random-access (offset/tail) requires the seekable ST path.
    if opts.offset.is_some() || opts.tail.is_some() {
        return decompress_stream_seek(input, opts);
    }
    // Verify mode is single-threaded — we just want to drain into sink.
    if threads <= 1 || opts.verify {
        decompress_stream_st(input, opts)
    } else {
        decompress_stream_mt(input, opts, threads)
    }
}

/// `--follow` path.  Wraps the input file in a [`FollowingReader`] that
/// blocks instead of returning EOF, sleeping 1 s and re-opening each
/// time the underlying file hits its end.  Output goes to stdout (or
/// the user-supplied destination) and the call never returns under
/// normal operation — the user terminates with Ctrl-C.
///
/// `--offset` is supported (seek before following); `--tail` is
/// rejected at parse time.
fn decompress_stream_follow(input: &Path, opts: &Options) -> io::Result<()> {
    let dst_path = dest_path(input, opts);
    if !opts.quiet {
        eprintln!("Following {} -> {}", input.display(), dst_path.display());
    }
    let follower = FollowingReader::open(input)?;
    let reader = Reader::new(follower);
    let dst = open_output(&dst_path)?;
    let mut counted_dst = CountingWriter::new(dst);

    if let Some(off) = opts.offset {
        let mut rs = ReadSeeker::new(reader, &[])
            .map_err(|e| io::Error::other(format!("--follow --offset requires an index: {e}")))?;
        rs.seek(SeekFrom::Start(off))?;
        if opts.tail_next_nl {
            advance_past_newline(&mut rs)?;
        }
        io::copy(&mut rs, &mut counted_dst)?;
    } else {
        let mut reader = reader;
        io::copy(&mut reader, &mut counted_dst)?;
    }
    counted_dst.flush()?;
    Ok(())
}

fn decompress_stream_seek(input: &Path, opts: &Options) -> io::Result<()> {
    if input == Path::new("-") {
        return Err(io::Error::other(
            "--offset / --tail require a seekable input (not stdin)",
        ));
    }
    let file = File::open(input)?;
    let src = BufReader::with_capacity(256 * 1024, file);
    let reader = Reader::new(src);
    let mut rs = ReadSeeker::new(reader, &[]).map_err(|e| {
        io::Error::other(format!(
            "--offset / --tail require an index in the stream: {e}"
        ))
    })?;

    let dst_path = dest_path(input, opts);
    if !opts.quiet {
        eprint!(
            "Decompressing {} -> {}",
            input.display(),
            dst_path.display()
        );
    }

    // Resolve the source seek + how many bytes to emit.
    let total = rs.index().total_uncompressed().ok_or_else(|| {
        io::Error::other("--offset / --tail require an index with known total size")
    })?;
    let (start, limit) = match (opts.offset, opts.tail) {
        (Some(off), None) => (off, total.saturating_sub(off)),
        (None, Some(t)) => {
            let t = t.min(total);
            (total - t, t)
        }
        (Some(off), Some(t)) => {
            let end = off.saturating_add(t).min(total);
            (off, end.saturating_sub(off))
        }
        (None, None) => unreachable!("decompress_stream_seek called without offset/tail"),
    };
    rs.seek(SeekFrom::Start(start))?;
    if opts.tail_next_nl {
        advance_past_newline(&mut rs)?;
    }

    let start_t = Instant::now();
    if opts.verify {
        let mut limited = (&mut rs).take(limit);
        io::copy(&mut limited, &mut io::sink())?;
        let elapsed = start_t.elapsed();
        print_decompress_summary(opts, total, limit, elapsed);
        if !opts.quiet {
            eprintln!("... Verified ok.");
        }
        return Ok(());
    }
    let dst = open_output(&dst_path)?;
    let mut counted_dst = CountingWriter::new(dst);
    let mut limited = (&mut rs).take(limit);
    io::copy(&mut limited, &mut counted_dst)?;
    counted_dst.flush()?;
    let elapsed = start_t.elapsed();
    print_decompress_summary(opts, total, counted_dst.bytes, elapsed);
    Ok(())
}

fn decompress_stream_st(input: &Path, opts: &Options) -> io::Result<()> {
    let (src, _) = open_input(input)?;
    let dst_path = dest_path(input, opts);
    let mut counted_src = CountingReader::new(src);

    if !opts.quiet {
        eprint!(
            "Decompressing {} -> {}",
            input.display(),
            dst_path.display()
        );
    }

    let mut reader = Reader::new(&mut counted_src);
    let start = Instant::now();
    if opts.verify {
        let mut sink = io::sink();
        io::copy(&mut reader, &mut sink)?;
        let elapsed = start.elapsed();
        print_decompress_summary(opts, counted_src.bytes, 0, elapsed);
        if !opts.quiet {
            eprintln!("... Verified ok.");
        }
        return Ok(());
    }
    let dst = open_output(&dst_path)?;
    let mut counted_dst = CountingWriter::new(dst);
    io::copy(&mut reader, &mut counted_dst)?;
    counted_dst.flush()?;
    let elapsed = start.elapsed();
    print_decompress_summary(opts, counted_src.bytes, counted_dst.bytes, elapsed);
    if opts.remove && input != Path::new("-") {
        std::fs::remove_file(input)?;
        if !opts.quiet {
            eprintln!("Removing {}", input.display());
        }
    }
    Ok(())
}

fn decompress_stream_mt(input: &Path, opts: &Options, threads: usize) -> io::Result<()> {
    let (src, _) = open_input(input)?;
    let dst_path = dest_path(input, opts);
    let mut counted_src = CountingReader::new(src);
    let dst = open_output_send(&dst_path)?;

    if !opts.quiet {
        eprint!(
            "Decompressing {} -> {}",
            input.display(),
            dst_path.display()
        );
    }
    let start = Instant::now();
    let mut reader = Reader::new(&mut counted_src);
    let ConcurrentDecode {
        bytes_written: decoded_bytes,
        writer: _w,
    } = reader.decode_concurrent(dst, threads)?;
    let elapsed = start.elapsed();
    print_decompress_summary(opts, counted_src.bytes, decoded_bytes, elapsed);
    if opts.remove && input != Path::new("-") {
        std::fs::remove_file(input)?;
        if !opts.quiet {
            eprintln!("Removing {}", input.display());
        }
    }
    Ok(())
}

fn decompress_block(input: &Path, opts: &Options) -> io::Result<()> {
    let (mut src, _) = open_input(input)?;
    let mut buf = Vec::new();
    src.read_to_end(&mut buf)?;
    let dst_path = dest_path(input, opts);
    if !opts.quiet {
        eprint!(
            "Decompressing {} -> {}",
            input.display(),
            dst_path.display()
        );
    }
    let start = Instant::now();
    let mut dec = Vec::with_capacity(buf.len() * 2);
    minlz::decode(&mut dec, &buf).map_err(io::Error::other)?;
    let elapsed = start.elapsed();
    if !opts.verify {
        let mut dst = open_output(&dst_path)?;
        dst.write_all(&dec)?;
        dst.flush()?;
    }
    print_decompress_summary(opts, buf.len() as u64, dec.len() as u64, elapsed);
    if opts.verify && !opts.quiet {
        eprintln!("... Verified ok.");
    }
    if opts.remove && input != Path::new("-") && !opts.verify {
        std::fs::remove_file(input)?;
        if !opts.quiet {
            eprintln!("Removing {}", input.display());
        }
    }
    Ok(())
}

fn extension_matches(p: &Path, ext: &str) -> bool {
    p.extension().and_then(|s| s.to_str()) == Some(ext)
}

fn dest_path(input: &Path, opts: &Options) -> PathBuf {
    if opts.verify {
        return PathBuf::from("(verify)");
    }
    if opts.stdout {
        return PathBuf::from("-");
    }
    if let Some(out) = opts.output.as_ref() {
        return out.clone();
    }
    if input == Path::new("-") {
        return PathBuf::from("-");
    }
    // Strip a recognized compression extension to get the destination name.
    let ext = input.extension().and_then(|s| s.to_str()).unwrap_or("");
    if matches!(ext, "mz" | "mzb") {
        input.with_extension("")
    } else {
        // Unrecognised extension — write to `<input>.out` to avoid clobbering.
        let mut p = input.as_os_str().to_owned();
        p.push(".out");
        PathBuf::from(p)
    }
}

/// Consume bytes from `r` until just past the next `\n`, so that
/// subsequent reads start at a line boundary.  Used by the `+nl` suffix
/// on `--tail` / `--offset` (matches Go `cmd/mz/decompress.go:392-398`
/// and `:423-437`).
fn advance_past_newline<R: Read>(r: &mut R) -> io::Result<()> {
    let mut byte = [0u8; 1];
    loop {
        match r.read(&mut byte)? {
            0 => return Ok(()), // EOF: nothing more to align.
            _ => {
                if byte[0] == b'\n' {
                    return Ok(());
                }
            }
        }
    }
}

/// `tail -f` style follower.  Wraps an `std::fs::File`; on read returning
/// 0 bytes (EOF) it sleeps 1 s, re-opens the path, seeks to the last
/// position it had reached, and retries — effectively never returning
/// EOF.  The caller terminates with Ctrl-C.
///
/// Mirrors Go `cmd/mz/decompress.go:followingReader` (line ~990).
struct FollowingReader {
    path: std::path::PathBuf,
    file: std::fs::File,
    offset: u64,
}

impl FollowingReader {
    fn open(path: &Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
            offset: 0,
        })
    }
}

impl Read for FollowingReader {
    fn read(&mut self, p: &mut [u8]) -> io::Result<usize> {
        loop {
            match self.file.read(p)? {
                0 => {
                    // EOF on the current file handle.  Drop it, sleep,
                    // re-open, and seek to `self.offset`.  Go uses a
                    // 1-second pause; we match.
                    std::thread::sleep(std::time::Duration::from_secs(1));
                    let mut new_file = std::fs::File::open(&self.path)?;
                    new_file.seek(SeekFrom::Start(self.offset))?;
                    self.file = new_file;
                }
                n => {
                    self.offset += n as u64;
                    return Ok(n);
                }
            }
        }
    }
}

impl Seek for FollowingReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        let new = self.file.seek(pos)?;
        self.offset = new;
        Ok(new)
    }
}

fn print_decompress_summary(opts: &Options, input: u64, output: u64, elapsed: std::time::Duration) {
    if opts.quiet {
        return;
    }
    if input == 0 {
        eprintln!(" {input} -> {output}; 0.0MB/s");
        return;
    }
    let pct = 100.0 * output as f64 / input as f64;
    let ratio = output as f64 / input as f64;
    let mbps = mb_per_sec(output, elapsed);
    eprintln!(" {input} -> {output} [{pct:.02}% 1:{ratio:.03}]; {mbps:.01}MB/s");
}
