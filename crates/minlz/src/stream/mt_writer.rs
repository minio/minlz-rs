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

//! Multi-threaded streaming writer.
//!
//! A worker pool compresses blocks in parallel while a single writer
//! thread drains ordered per-block result channels and writes to the
//! underlying sink.
//!
//! `W` is moved into the writer thread on construction; [`finish`]
//! returns it back via a one-shot channel.  That ownership round-trip
//! requires `W: Send + 'static` — borrowed writers (`&mut Vec<u8>`)
//! must be wrapped in an owned adapter or use the single-threaded
//! [`Writer`](super::Writer).
//!
//! [`finish`]: MtWriter::finish

use std::io::{self, Write};
use std::marker::PhantomData;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::block;
use crate::index::Index;

use super::Concurrency;
use super::crc::masked_crc32c;
use super::error::Error;
use super::format::{
    CHECKSUM_SIZE, CHUNK_HEADER_SIZE, CHUNK_TYPE_EOF, CHUNK_TYPE_MINLZ_COMPRESSED_DATA,
    CHUNK_TYPE_PADDING, CHUNK_TYPE_UNCOMPRESSED_DATA, DEFAULT_BLOCK_SIZE, MAX_BLOCK_SIZE,
    MAX_USER_CHUNK_SIZE, MAX_USER_NON_SKIPPABLE_CHUNK, MAX_VARINT_LEN_64, MIN_BLOCK_SIZE,
    MIN_USER_SKIPPABLE_CHUNK, make_stream_header, put_uvarint,
};
use super::mt_pool::BufferPool;

/// One item in the per-stream order queue.  The writer thread drains
/// this in submission order and writes the bytes to `W`.
enum OrderItem {
    /// A future block — wait on this channel for the worker's encoded body.
    Block(Receiver<WorkerOutput>),
    /// Pre-formatted bytes — header / EOF / padding / user-chunk.  Written
    /// directly without waiting on a worker.
    Raw(Vec<u8>),
}

/// What a worker sends back: (header, body) so the writer thread can
/// emit the two with separate `write_all` calls without an extra copy.
/// `input_return` is the input buffer the worker is done with — the
/// writer thread releases it back to the input pool after writing, so
/// the buffer round-trips on a single ownership chain without crossing
/// the pool mutex twice per block.
struct WorkerOutput {
    /// 4-byte chunk header + 4-byte CRC.
    hdr: [u8; CHUNK_HEADER_SIZE + CHECKSUM_SIZE],
    /// Encoded body (or raw uncompressed body for 0x01 chunks).
    body: Vec<u8>,
    /// `true` if `body` aliases the input (incompressible 0x01 path).
    /// In that case `input_return` is empty; only `body` needs returning
    /// (to the input pool, since that's where it came from).
    body_is_input: bool,
    /// Input buffer to return to `input_pool` once the writer is done.
    /// May be empty if we couldn't reuse it (e.g. compressed path moved
    /// `body` out of the obuf pool, leaving the input intact here).
    input_return: Vec<u8>,
    /// Uncompressed length of this block.  Used by the writer thread to
    /// maintain its `uncomp_written` counter for the index.
    uncomp_len: u32,
}

struct BlockJob {
    /// Owned input bytes to compress.  Pulled from `input_pool` by
    /// the dispatcher and returned by the writer thread via
    /// `WorkerOutput::input_return`.
    uncompressed: Vec<u8>,
    level: block::Level,
    uncompressed_mode: bool,
    /// Single-slot channel the worker writes its result into.
    result_tx: SyncSender<WorkerOutput>,
    /// Pool to acquire the output buffer from + return it through the
    /// writer thread.
    obuf_pool: Arc<BufferPool>,
}

/// Builder for [`MtWriter`].  All knobs match [`super::WriterBuilder`]
/// except `concurrency`, which selects the worker count.
#[must_use]
pub struct MtWriterBuilder {
    block_size: usize,
    level: block::Level,
    uncompressed: bool,
    padding: u32,
    /// Requested worker count (raw); validated into a [`Concurrency`] at
    /// [`build`](MtWriterBuilder::build). `0` is rejected there.
    concurrency: usize,
    generate_index: bool,
    append_index: bool,
}

impl Default for MtWriterBuilder {
    fn default() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            level: block::Level::Balanced,
            uncompressed: false,
            padding: 0,
            concurrency: Concurrency::available().get(),
            generate_index: true,
            append_index: false,
        }
    }
}

impl MtWriterBuilder {
    /// New builder with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum uncompressed block size.  Must lie in
    /// `[MIN_BLOCK_SIZE, MAX_BLOCK_SIZE]`; validated by [`build`](Self::build).
    pub fn block_size(mut self, n: usize) -> Self {
        self.block_size = n;
        self
    }

    /// Set the compression level.
    pub fn level(mut self, level: block::Level) -> Self {
        self.level = level;
        self.uncompressed = false;
        self
    }

    /// Bypass compression — emit only 0x01 (uncompressed) chunks.
    pub fn uncompressed(mut self) -> Self {
        self.uncompressed = true;
        self
    }

    /// Pad total output to a multiple of `n` at finish time.  `0`/`1`
    /// disables padding.  `n > MAX_BLOCK_SIZE` is rejected by
    /// [`build`](Self::build).
    pub fn padding(mut self, n: u32) -> Self {
        self.padding = if n <= 1 { 0 } else { n };
        self
    }

    /// Number of worker threads to spin up.  `0` is rejected by
    /// [`build`](Self::build); `1` still uses the MT pipeline (one worker) —
    /// for the lowest-overhead path use [`super::Writer`].
    pub fn concurrency(mut self, n: usize) -> Self {
        self.concurrency = n;
        self
    }

    /// Toggle in-memory index generation.  Default `true`.  Set to
    /// `false` for streaming output where no index is ever needed.
    /// Matches [`super::WriterBuilder::generate_index`]. If
    /// [`append_index`](Self::append_index) is also requested, the index is
    /// generated regardless of this flag.
    pub fn generate_index(mut self, b: bool) -> Self {
        self.generate_index = b;
        self
    }

    /// Append the index chunk to the end of the stream when finishing.
    /// Implies index generation.  Matches
    /// [`super::WriterBuilder::append_index`].
    pub fn append_index(mut self) -> Self {
        self.generate_index = true;
        self.append_index = true;
        self
    }

    /// Consume the builder and wrap `w`.
    ///
    /// # Errors
    /// [`Error::Config`] if `block_size` is outside `[MIN_BLOCK_SIZE,
    /// MAX_BLOCK_SIZE]`, `padding` exceeds `MAX_BLOCK_SIZE`, or `concurrency`
    /// was set to `0`.
    pub fn build<W: Write + Send + 'static>(self, w: W) -> Result<MtWriter<W>, Error> {
        if !(MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE).contains(&self.block_size) {
            return Err(Error::Config(
                "block_size must be in [MIN_BLOCK_SIZE, MAX_BLOCK_SIZE]",
            ));
        }
        if self.padding as usize > MAX_BLOCK_SIZE {
            return Err(Error::Config("padding must be ≤ MAX_BLOCK_SIZE"));
        }
        let concurrency = Concurrency::new(self.concurrency)
            .ok_or(Error::Config("concurrency must be ≥ 1"))?
            .get();
        Ok(MtWriter::with_builder(w, self, concurrency))
    }
}

/// Multi-threaded MinLZ stream writer.
///
/// Workers compress blocks in parallel.  A single writer thread drains
/// per-block result channels in submission order and writes to `W`.
/// `W` is moved into the writer thread on construction and returned via
/// [`finish`].  See module-level docs for the rationale.
///
/// [`finish`]: MtWriter::finish
pub struct MtWriter<W: Write + Send + 'static> {
    block_size: usize,
    level: block::Level,
    uncompressed: bool,
    /// Padding multiple; emitted by the writer thread on channel close.
    /// `0` or `1` disables padding (matches Go).
    #[allow(dead_code)]
    padding: u32,
    ibuf: Vec<u8>,
    /// One submit channel per worker — the dispatcher picks via
    /// round-robin so workers never contend on a shared receiver.
    submit_txs: Vec<SyncSender<BlockJob>>,
    /// Round-robin cursor for `submit_txs`.
    next_worker: usize,
    order_tx: Option<SyncSender<OrderItem>>,
    finish_rx: Receiver<io::Result<W>>,
    workers: Vec<JoinHandle<()>>,
    writer_thread: Option<JoinHandle<()>>,
    /// Shared error slot — set by the writer thread if a downstream
    /// write fails.  Read by [`MtWriter::write`] and [`finish`].
    err: Arc<Mutex<Option<io::Error>>>,
    /// Pool of 1-block input buffers.  Round-trips through the worker
    /// + writer thread back to here on every block.
    input_pool: Arc<BufferPool>,
    /// Pool of compressed-output buffers (worker writes obuf, writer
    /// thread returns).
    obuf_pool: Arc<BufferPool>,
    uncomp_written: u64,
    wrote_header: bool,
    _w: PhantomData<fn() -> W>,
}

impl<W: Write + Send + 'static> MtWriter<W> {
    /// Wrap `w` with default options + `available_parallelism()` workers.
    pub fn new(w: W) -> Self {
        MtWriterBuilder::new()
            .build(w)
            .expect("default MtWriterBuilder options are always valid")
    }

    fn with_builder(w: W, b: MtWriterBuilder, concurrency: usize) -> Self {
        // One bounded submit channel per worker — capacity 1 means
        // backpressure kicks in immediately once a worker is busy, but
        // the dispatcher can fan out to another worker without blocking.
        let mut submit_txs = Vec::with_capacity(concurrency);
        let mut submit_rxs = Vec::with_capacity(concurrency);
        for _ in 0..concurrency {
            let (tx, rx) = mpsc::sync_channel::<BlockJob>(1);
            submit_txs.push(tx);
            submit_rxs.push(rx);
        }
        let (order_tx, order_rx) = mpsc::sync_channel::<OrderItem>(concurrency + 1);
        let (finish_tx, finish_rx) = mpsc::sync_channel::<io::Result<W>>(1);
        let err: Arc<Mutex<Option<io::Error>>> = Arc::new(Mutex::new(None));
        let input_pool = Arc::new(BufferPool::new(concurrency + 1, b.block_size));
        // Output buffer needs MaxEncodedLen(block_size) worst-case.
        let obuf_cap = block::max_encoded_len(b.block_size).unwrap_or(b.block_size + 16);
        let obuf_pool = Arc::new(BufferPool::new(concurrency + 1, obuf_cap));

        // Spawn workers.
        let mut workers = Vec::with_capacity(concurrency);
        for rx in submit_rxs {
            workers.push(std::thread::spawn(move || worker_loop(rx)));
        }

        // Spawn writer thread.
        let err_writer = err.clone();
        let input_pool_w = input_pool.clone();
        let obuf_pool_w = obuf_pool.clone();
        let writer_index = if b.generate_index || b.append_index {
            let mut idx = Index::default();
            idx.reset(b.block_size);
            // Totals stay unknown (None) until `append_to` records them at close.
            Some(idx)
        } else {
            None
        };
        let block_size_w = b.block_size;
        let append_index_w = b.append_index;
        let padding_w = b.padding;
        let writer_thread = std::thread::spawn(move || {
            writer_loop(
                w,
                order_rx,
                finish_tx,
                err_writer,
                input_pool_w,
                obuf_pool_w,
                writer_index,
                append_index_w,
                block_size_w,
                padding_w,
            );
        });

        Self {
            block_size: b.block_size,
            level: b.level,
            uncompressed: b.uncompressed,
            padding: b.padding,
            ibuf: Vec::with_capacity(b.block_size),
            submit_txs,
            next_worker: 0,
            order_tx: Some(order_tx),
            finish_rx,
            workers,
            writer_thread: Some(writer_thread),
            err,
            input_pool,
            obuf_pool,
            uncomp_written: 0,
            wrote_header: false,
            _w: PhantomData,
        }
    }

    /// Total uncompressed bytes accepted.
    pub fn uncompressed_written(&self) -> u64 {
        self.uncomp_written
    }

    /// Encode a buffer in one shot.  Takes ownership so the worker
    /// thread can borrow it for the whole encode without copies.
    /// Equivalent to Go's `Writer.EncodeBuffer` but always copies into
    /// per-block jobs (the worker owns each block's input).
    pub fn encode_buffer(&mut self, buf: &[u8]) -> io::Result<()> {
        self.flush_ibuf()?;
        self.write_blocks(buf)?;
        Ok(())
    }

    /// Add a user chunk (`id` in `0x80..=0xfd`) at this stream position.
    /// Pending buffered input is flushed first so the chunk appears
    /// after any data the user has already written.
    pub fn add_user_chunk(&mut self, id: u8, data: &[u8]) -> io::Result<()> {
        if !(MIN_USER_SKIPPABLE_CHUNK..=MAX_USER_NON_SKIPPABLE_CHUNK).contains(&id) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("user-chunk id 0x{id:02x} out of range"),
            ));
        }
        if data.len() > MAX_USER_CHUNK_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "user-chunk exceeds maximum size",
            ));
        }
        self.check_err()?;
        self.flush_ibuf()?;
        self.ensure_header()?;
        let mut raw = Vec::with_capacity(CHUNK_HEADER_SIZE + data.len());
        raw.push(id);
        raw.push(data.len() as u8);
        raw.push((data.len() >> 8) as u8);
        raw.push((data.len() >> 16) as u8);
        raw.extend_from_slice(data);
        self.submit_raw(raw)
    }

    /// Flush pending input, emit the EOF chunk + any padding, join all
    /// threads, and return the underlying writer.
    pub fn finish(mut self) -> io::Result<W> {
        self.check_err()?;
        self.flush_ibuf()?;
        self.ensure_header()?;
        // EOF chunk.
        let mut tmp = [0u8; CHUNK_HEADER_SIZE + MAX_VARINT_LEN_64];
        tmp[0] = CHUNK_TYPE_EOF;
        let n = put_uvarint(&mut tmp[CHUNK_HEADER_SIZE..], self.uncomp_written);
        tmp[1] = n as u8;
        self.submit_raw(tmp[..CHUNK_HEADER_SIZE + n].to_vec())?;

        // Signal end of stream.  The writer thread emits padding (if
        // configured) and the index chunk (if `append_index`) on
        // channel close — see `writer_loop`.
        drop(self.order_tx.take());
        self.submit_txs.clear();
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
        if let Some(h) = self.writer_thread.take() {
            let _ = h.join();
        }
        match self.finish_rx.recv() {
            Ok(Ok(w)) => Ok(w),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(io::Error::other("writer thread did not return W")),
        }
    }

    fn check_err(&self) -> io::Result<()> {
        // Lock poison only panics if a worker thread already panicked while
        // holding this mutex — it propagates that panic rather than masking
        // it. Same rationale applies to every `lock().unwrap()` below.
        if let Some(e) = self.err.lock().unwrap().as_ref() {
            return Err(clone_io_err(e));
        }
        Ok(())
    }

    fn ensure_header(&mut self) -> io::Result<()> {
        if self.wrote_header {
            return Ok(());
        }
        self.wrote_header = true;
        let hdr = make_stream_header(self.block_size);
        self.submit_raw(hdr.to_vec())
    }

    fn submit_raw(&mut self, bytes: Vec<u8>) -> io::Result<()> {
        // `order_tx` is `Some` for the writer's entire usable life; it is
        // only `.take()`n in `finish()` (which consumes `self`) and `Drop`
        // (terminal), so no submit path can observe `None` here.
        match self.order_tx.as_ref().unwrap().send(OrderItem::Raw(bytes)) {
            Ok(()) => Ok(()),
            Err(_) => Err(self.collect_pipeline_error()),
        }
    }

    /// Acquire a pooled input buffer, copy `block` into it, and submit
    /// to a worker.  The buffer round-trips back to the pool via the
    /// writer thread.
    fn submit_block(&mut self, block: &[u8]) -> io::Result<()> {
        self.uncomp_written += block.len() as u64;
        let mut input = self.input_pool.acquire();
        input.extend_from_slice(block);
        let (result_tx, result_rx) = mpsc::sync_channel::<WorkerOutput>(1);
        // Push order slot first so the writer thread can drain in order.
        // `order_tx` is `Some` until `finish()`/`Drop` (see `submit_raw`).
        if self
            .order_tx
            .as_ref()
            .unwrap()
            .send(OrderItem::Block(result_rx))
            .is_err()
        {
            self.input_pool.release(input);
            return Err(self.collect_pipeline_error());
        }
        let job = BlockJob {
            uncompressed: input,
            level: self.level,
            uncompressed_mode: self.uncompressed,
            result_tx,
            obuf_pool: self.obuf_pool.clone(),
        };
        // Round-robin pick across worker submit channels — no
        // Mutex<Receiver> contention, just one `send` per block.
        let n = self.submit_txs.len();
        let idx = self.next_worker % n;
        self.next_worker = (self.next_worker + 1) % n;
        match self.submit_txs[idx].try_send(job) {
            Ok(()) => Ok(()),
            Err(mpsc::TrySendError::Full(j)) => self.send_block_fallback(j, idx + 1, 1),
            Err(mpsc::TrySendError::Disconnected(_)) => Err(self.collect_pipeline_error()),
        }
    }

    /// Helper that continues the round-robin search after a `try_send`
    /// returned `Full`.  Recovers the job and tries remaining workers.
    fn send_block_fallback(
        &mut self,
        mut job: BlockJob,
        mut idx: usize,
        mut tried: usize,
    ) -> io::Result<()> {
        let n = self.submit_txs.len();
        loop {
            let slot = idx % n;
            idx = (idx + 1) % n;
            match self.submit_txs[slot].try_send(job) {
                Ok(()) => {
                    self.next_worker = idx;
                    return Ok(());
                }
                Err(mpsc::TrySendError::Full(j)) => {
                    tried += 1;
                    job = j;
                    if tried >= n {
                        // Block on the next one.
                        return self.submit_txs[slot]
                            .send(job)
                            .map_err(|_| self.collect_pipeline_error());
                    }
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    return Err(self.collect_pipeline_error());
                }
            }
        }
    }

    fn collect_pipeline_error(&self) -> io::Error {
        match self.err.lock().unwrap().as_ref() {
            Some(e) => clone_io_err(e),
            None => io::Error::other("MinLZ MT pipeline closed unexpectedly"),
        }
    }

    fn write_blocks(&mut self, mut data: &[u8]) -> io::Result<()> {
        self.ensure_header()?;
        while !data.is_empty() {
            self.check_err()?;
            let n = data.len().min(self.block_size);
            let (block, rest) = data.split_at(n);
            self.submit_block(block)?;
            data = rest;
        }
        Ok(())
    }

    fn flush_ibuf(&mut self) -> io::Result<()> {
        if self.ibuf.is_empty() {
            return Ok(());
        }
        let buf = std::mem::take(&mut self.ibuf);
        let res = self.write_blocks(&buf);
        let mut buf = buf;
        buf.clear();
        self.ibuf = buf;
        res
    }
}

impl<W: Write + Send + 'static> Write for MtWriter<W> {
    fn write(&mut self, p: &[u8]) -> io::Result<usize> {
        self.check_err()?;
        if p.is_empty() {
            return Ok(0);
        }
        let mut written = 0;
        let mut p = p;
        while !p.is_empty() {
            let free = self.block_size - self.ibuf.len();
            if self.ibuf.is_empty() && p.len() >= self.block_size {
                let n = p.len() - (p.len() % self.block_size);
                let n = n.min(p.len());
                self.write_blocks(&p[..n])?;
                written += n;
                p = &p[n..];
                continue;
            }
            let take = p.len().min(free);
            self.ibuf.extend_from_slice(&p[..take]);
            written += take;
            p = &p[take..];
            if self.ibuf.len() == self.block_size {
                self.flush_ibuf()?;
            }
        }
        Ok(written)
    }

    /// Push any pending input into the encode pipeline, but do not emit
    /// EOF.  The writer thread keeps draining.
    fn flush(&mut self) -> io::Result<()> {
        self.check_err()?;
        self.flush_ibuf()?;
        Ok(())
    }
}

impl<W: Write + Send + 'static> Drop for MtWriter<W> {
    fn drop(&mut self) {
        // Best-effort cleanup if the user forgot finish(): drop the
        // submit + order channels so workers/writer exit, then join.
        // Pending data is silently lost — matches Go.
        self.submit_txs.clear();
        drop(self.order_tx.take());
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
        if let Some(h) = self.writer_thread.take() {
            let _ = h.join();
        }
    }
}

// -------------------- worker / writer thread bodies --------------------

fn worker_loop(submit_rx: Receiver<BlockJob>) {
    while let Ok(job) = submit_rx.recv() {
        let BlockJob {
            uncompressed,
            level,
            uncompressed_mode,
            result_tx,
            obuf_pool,
        } = job;
        let uncomp_len = uncompressed.len() as u32;
        let crc = masked_crc32c(&uncompressed);
        let mut obuf = obuf_pool.acquire();
        let compressed_ok = if uncompressed_mode {
            false
        } else {
            match block::append_encoded_chunk_body(&mut obuf, &uncompressed, level) {
                Ok(ok) => ok,
                Err(e) => {
                    obuf_pool.release(obuf);
                    let _ = result_tx.send(WorkerOutput {
                        hdr: error_marker(),
                        body: format!("encode: {e}").into_bytes(),
                        body_is_input: false,
                        input_return: uncompressed,
                        uncomp_len,
                    });
                    continue;
                }
            }
        };
        // For the 0x01 (uncompressed) path we send the input buffer
        // through as the body — saves a copy and lets the writer thread
        // release it as the body afterwards.  `input_return` is left
        // empty so the writer doesn't double-release.
        let (chunk_type, chunk_len, body, body_is_input, input_return) = if compressed_ok {
            let chunk_len = CHECKSUM_SIZE + obuf.len();
            (
                CHUNK_TYPE_MINLZ_COMPRESSED_DATA,
                chunk_len,
                obuf,
                false,
                uncompressed,
            )
        } else {
            // Incompressible: discard the (empty/wasted) obuf, send the
            // input as the body.
            obuf_pool.release(obuf);
            let chunk_len = CHECKSUM_SIZE + uncompressed.len();
            (
                CHUNK_TYPE_UNCOMPRESSED_DATA,
                chunk_len,
                uncompressed,
                true,
                Vec::new(),
            )
        };
        let mut hdr = [0u8; CHUNK_HEADER_SIZE + CHECKSUM_SIZE];
        hdr[0] = chunk_type;
        hdr[1] = chunk_len as u8;
        hdr[2] = (chunk_len >> 8) as u8;
        hdr[3] = (chunk_len >> 16) as u8;
        hdr[4..8].copy_from_slice(&crc.to_le_bytes());
        let _ = result_tx.send(WorkerOutput {
            hdr,
            body,
            body_is_input,
            input_return,
            uncomp_len,
        });
    }
}

/// A header value the writer thread treats as "the worker hit an
/// error, the body is the message".  Type byte `0xff` is the stream
/// identifier (never produced by a worker), so this is unambiguous.
fn error_marker() -> [u8; CHUNK_HEADER_SIZE + CHECKSUM_SIZE] {
    [0xff, 0, 0, 0, 0, 0, 0, 0]
}

fn is_error_marker(hdr: &[u8; CHUNK_HEADER_SIZE + CHECKSUM_SIZE]) -> bool {
    hdr[0] == 0xff && hdr[1] == 0 && hdr[2] == 0 && hdr[3] == 0
}

#[allow(clippy::too_many_arguments)]
fn writer_loop<W: Write + Send + 'static>(
    mut w: W,
    order_rx: Receiver<OrderItem>,
    finish_tx: SyncSender<io::Result<W>>,
    err_slot: Arc<Mutex<Option<io::Error>>>,
    input_pool: Arc<BufferPool>,
    obuf_pool: Arc<BufferPool>,
    mut index: Option<Index>,
    append_index: bool,
    _block_size: usize,
    padding: u32,
) {
    let mut comp_written: u64 = 0;
    let mut uncomp_written: u64 = 0;

    let res = (|| -> io::Result<()> {
        while let Ok(item) = order_rx.recv() {
            match item {
                OrderItem::Raw(bytes) => {
                    w.write_all(&bytes).map_err(|e| record(&err_slot, e))?;
                    comp_written += bytes.len() as u64;
                }
                OrderItem::Block(rx) => match rx.recv() {
                    Ok(out) => {
                        if is_error_marker(&out.hdr) {
                            let msg = String::from_utf8_lossy(&out.body).into_owned();
                            let e = io::Error::new(io::ErrorKind::InvalidData, msg);
                            return Err(record(&err_slot, e));
                        }
                        let WorkerOutput {
                            hdr,
                            body,
                            body_is_input,
                            input_return,
                            uncomp_len,
                        } = out;
                        // Record the (comp_offset, uncomp_offset) pair
                        // for this block *before* incrementing — these
                        // are offsets at the start of the chunk.
                        if let Some(idx) = index.as_mut() {
                            idx.add(comp_written, uncomp_written)
                                .map_err(|e| record(&err_slot, e.into()))?;
                        }
                        let body_len = body.len();
                        w.write_all(&hdr).map_err(|e| record(&err_slot, e))?;
                        let write_res = w.write_all(&body).map_err(|e| record(&err_slot, e));
                        // Return buffers to their pools.  `body` came
                        // from input_pool if `body_is_input`, otherwise
                        // from obuf_pool.  `input_return` (if non-empty)
                        // is always from input_pool.
                        if body_is_input {
                            input_pool.release(body);
                        } else {
                            obuf_pool.release(body);
                        }
                        if !input_return.is_empty() || input_return.capacity() > 0 {
                            input_pool.release(input_return);
                        }
                        write_res?;
                        comp_written += (hdr.len() + body_len) as u64;
                        uncomp_written += uncomp_len as u64;
                    }
                    Err(_) => {
                        let e = io::Error::other("MinLZ worker exited without producing output");
                        return Err(record(&err_slot, e));
                    }
                },
            }
        }
        // Channel closed (MtWriter::finish dropped order_tx after queuing
        // EOF).  Wire order from here matches Go's `writer.go:closeIndex`:
        // data → EOF → padding → index.  Padding is emitted *before*
        // the index but its size includes the index length, so the
        // total compressed size is aligned to `padding`.
        let mut idx_bytes = Vec::new();
        if let Some(idx) = index.as_mut() {
            // `total_compressed` is bytes-before-index; if padding will
            // be added, store -1 (unknown) so seekers don't rely on it.
            let comp_total = if padding <= 1 {
                Some(comp_written)
            } else {
                None
            };
            idx.append_to(&mut idx_bytes, Some(uncomp_written), comp_total)
                .map_err(|e| record(&err_slot, e.into()))?;
            if append_index {
                comp_written += idx_bytes.len() as u64;
            } else {
                idx_bytes.clear();
            }
        }

        if padding > 1 {
            emit_mt_padding(&mut w, &mut comp_written, padding, &err_slot)?;
        }

        if !idx_bytes.is_empty() {
            w.write_all(&idx_bytes).map_err(|e| record(&err_slot, e))?;
        }
        w.flush().map_err(|e| record(&err_slot, e))?;
        Ok(())
    })();
    let payload = res.map(|_| w);
    let _ = finish_tx.send(payload);
}

/// Emit a single padding (`0xfe`) chunk so that the total bytes
/// written (`*comp_written`, including the upcoming index chunk for
/// `append_index = true`) is aligned to `multiple`.  Zero body — the
/// decoder skips the chunk regardless of payload.  Mirrors the ST
/// writer's `emit_padding` (`stream/writer.rs`).
fn emit_mt_padding<W: Write>(
    w: &mut W,
    comp_written: &mut u64,
    multiple: u32,
    err_slot: &Arc<Mutex<Option<io::Error>>>,
) -> io::Result<()> {
    let multiple = multiple as u64;
    let leftover = *comp_written % multiple;
    if leftover == 0 {
        return Ok(());
    }
    let mut to_add = multiple - leftover;
    while to_add < CHUNK_HEADER_SIZE as u64 {
        to_add += multiple;
    }
    if to_add as usize > MAX_BLOCK_SIZE + CHUNK_HEADER_SIZE {
        return Ok(());
    }
    let body_len = (to_add as usize) - CHUNK_HEADER_SIZE;
    let mut hdr = [0u8; CHUNK_HEADER_SIZE];
    hdr[0] = CHUNK_TYPE_PADDING;
    hdr[1] = body_len as u8;
    hdr[2] = (body_len >> 8) as u8;
    hdr[3] = (body_len >> 16) as u8;
    w.write_all(&hdr).map_err(|e| record(err_slot, e))?;
    let zeros = [0u8; 4096];
    let mut remaining = body_len;
    while remaining > 0 {
        let n = remaining.min(zeros.len());
        w.write_all(&zeros[..n]).map_err(|e| record(err_slot, e))?;
        remaining -= n;
    }
    *comp_written += to_add;
    Ok(())
}

fn record(slot: &Arc<Mutex<Option<io::Error>>>, e: io::Error) -> io::Error {
    let cloned = clone_io_err(&e);
    // Lock poison only propagates a prior worker panic (see `check_err`).
    let mut guard = slot.lock().unwrap();
    if guard.is_none() {
        *guard = Some(e);
    }
    cloned
}

fn clone_io_err(e: &io::Error) -> io::Error {
    io::Error::new(e.kind(), e.to_string())
}

// Silence unused-import warning if Error import is needed elsewhere.
#[allow(dead_code)]
fn _ensure_error_import_used() -> Error {
    Error::Corrupt
}

#[cfg(test)]
mod tests;
