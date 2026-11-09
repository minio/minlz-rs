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
use std::path::{Path, PathBuf};
use std::sync::Mutex;

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
    Ok(Box::new(BufWriter::with_capacity(BUF_SIZE, create(path)?)))
}

/// The output file being written, which [`remove_output`] removes when the run fails, so a
/// failed run leaves no partial output.
static OUTPUT: Mutex<Option<PathBuf>> = Mutex::new(None);

fn create(path: &Path) -> io::Result<File> {
    let file = File::create(path)?;
    // Only a regular file is removed, never a device or a symlink given as output.
    if std::fs::symlink_metadata(path).is_ok_and(|m| m.is_file()) {
        *OUTPUT.lock().unwrap() = Some(path.to_owned());
    }
    Ok(file)
}

/// Removes the output file of a failed run, if [`open_output`] created one.
pub fn remove_output() {
    if let Some(p) = OUTPUT.lock().unwrap().take() {
        let _ = std::fs::remove_file(p);
    }
}

/// Rejects an output that is the input file: [`open_output`] would
/// truncate it before the first read. This compares file identity, so
/// path aliases, symlinks, hard links and stdin redirected from the
/// output are all caught.
pub fn check_not_input(input: &Path, output: &Path) -> io::Result<()> {
    if output != Path::new("-") && is_file(output) && is_input(input, output) {
        return Err(io::Error::other("input and output are the same file"));
    }
    Ok(())
}

// Only regular files are compared: `same_file` opens both paths, and opening a FIFO blocks
// until the other end appears. An output it can't open for reading can't be the input.
fn is_input(input: &Path, output: &Path) -> bool {
    let same = if input == Path::new("-") {
        same_file::Handle::stdin().and_then(|h| Ok(h == same_file::Handle::from_path(output)?))
    } else if is_file(input) {
        same_file::is_same_file(input, output)
    } else {
        return false;
    };
    matches!(same, Ok(true))
}

/// Syncs the output to disk before `--rm` deletes the input, so a crash can't lose both.
/// Stdout and other outputs that aren't regular files are skipped: pipes and terminals
/// can't be synced, and opening a FIFO would block.
pub fn sync_output(path: &Path) -> io::Result<()> {
    if path != Path::new("-") && is_file(path) {
        // Reopened because the writers are boxed. Windows syncs only through a handle with
        // write access.
        File::options().write(true).open(path)?.sync_all()?;
    }
    // The output is complete, and will be the only copy once the input is removed.
    OUTPUT.lock().unwrap().take();
    Ok(())
}

fn is_file(p: &Path) -> bool {
    std::fs::metadata(p).is_ok_and(|m| m.is_file())
}

/// Like [`open_output`], but `Send + 'static` so the value can be moved
/// into the MT writer thread.
pub fn open_output_send(path: &Path) -> io::Result<Box<dyn Write + Send + 'static>> {
    if path == Path::new("-") {
        return Ok(Box::new(BufWriter::with_capacity(BUF_SIZE, io::stdout())));
    }
    Ok(Box::new(BufWriter::with_capacity(BUF_SIZE, create(path)?)))
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
