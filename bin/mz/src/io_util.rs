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

//! Tiny I/O helpers: open input / output as buffered streams, plus a
//! counter wrapper for progress reporting.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::num::NonZeroUsize;
use std::path::Path;

/// Resolve the worker-thread count from `Options::threads`, falling
/// back to `available_parallelism()` and finally to 1.
pub fn resolve_threads(opt: Option<NonZeroUsize>) -> NonZeroUsize {
    opt.unwrap_or_else(|| std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
}

const BUF_SIZE: usize = 256 * 1024;

/// Open `path` for reading.  `-` means stdin.  Returns the boxed reader and
/// the input size in bytes (`None` if unknown, e.g. stdin or a pipe).
pub fn open_input(path: &Path) -> io::Result<(Box<dyn Read>, Option<u64>)> {
    if path == Path::new("-") {
        let stdin = io::stdin();
        // Stdin is buffered already on most platforms; we still wrap to get
        // a larger buffer for throughput.
        return Ok((Box::new(BufReader::with_capacity(BUF_SIZE, stdin)), None));
    }
    let file = File::open(path)?;
    let size = file.metadata().ok().map(|m| m.len());
    Ok((Box::new(BufReader::with_capacity(BUF_SIZE, file)), size))
}

/// Open `path` for writing, truncating any existing file.  `-` means stdout.
pub fn open_output(path: &Path) -> io::Result<Box<dyn Write>> {
    if path == Path::new("-") {
        return Ok(Box::new(BufWriter::with_capacity(BUF_SIZE, io::stdout())));
    }
    let file = File::create(path)?;
    Ok(Box::new(BufWriter::with_capacity(BUF_SIZE, file)))
}

/// Like [`open_output`], but `Send + 'static` so the value can be moved
/// into the MT writer thread.
pub fn open_output_send(path: &Path) -> io::Result<Box<dyn Write + Send + 'static>> {
    if path == Path::new("-") {
        return Ok(Box::new(BufWriter::with_capacity(BUF_SIZE, io::stdout())));
    }
    let file = File::create(path)?;
    Ok(Box::new(BufWriter::with_capacity(BUF_SIZE, file)))
}

/// `Read` adapter that counts the bytes read for progress reporting.
pub struct CountingReader<R> {
    inner: R,
    pub bytes: u64,
}

impl<R: Read> CountingReader<R> {
    pub fn new(r: R) -> Self {
        Self { inner: r, bytes: 0 }
    }
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.bytes += n as u64;
        Ok(n)
    }
}

/// `Write` adapter that counts the bytes written for progress reporting.
pub struct CountingWriter<W> {
    inner: W,
    pub bytes: u64,
}

impl<W: Write> CountingWriter<W> {
    pub fn new(w: W) -> Self {
        Self { inner: w, bytes: 0 }
    }
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.bytes += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Format throughput in MB/s, mirroring Go's `%.1fMB/s`.
pub fn mb_per_sec(bytes: u64, elapsed: std::time::Duration) -> f64 {
    let s = elapsed.as_secs_f64();
    if s <= 0.0 {
        return 0.0;
    }
    (bytes as f64 / 1e6) / s
}
