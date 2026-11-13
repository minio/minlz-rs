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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::block;
use crate::index::Index;

use super::error::Error;
use super::format::{
    CHECKSUM_SIZE, CHUNK_HEADER_SIZE, CHUNK_TYPE_EOF, CHUNK_TYPE_PADDING, DEFAULT_BLOCK_SIZE,
    MAX_BLOCK_SIZE, MAX_USER_CHUNK_SIZE, MAX_USER_NON_SKIPPABLE_CHUNK, MAX_VARINT_LEN_64,
    MIN_BLOCK_SIZE, MIN_USER_SKIPPABLE_CHUNK, SIMD_HASH_SIZE, make_stream_header, put_uvarint,
};
use super::mt_pool::BufferPool;
use super::writer::{Written, encode_block, put_uncompressed_header, simd_stream};

/// One item in the per-stream order queue.  The writer thread drains
/// this in submission order and writes the bytes to `W`.
enum OrderItem {
    /// A future block — wait on this channel for the worker's encoded body.
    Block(Receiver<io::Result<Encoded>>),
    /// Pre-formatted bytes — header / EOF / padding / user-chunk.  Written
    /// directly without waiting on a worker.
    Raw(Vec<u8>),
    /// Flush `W` and answer once everything queued before it is written.
    Flush(SyncSender<()>),
    /// End of stream, sent by `finish` after the EOF chunk: write padding and
    /// the index.  A close without it (a dropped writer) writes neither.
    /// `true` serializes the index even without `append_index`, for
    /// `close_index`.
    Finish(bool),
}

/// A worker's encoded block.  `input` is the block's uncompressed bytes; the
/// writer thread returns it to the input pool, so the buffer round-trips
/// without crossing the pool mutex twice per block.
enum Encoded {
    /// `buf[..len]` is the whole chunk; `buf` goes back to the output pool (a
    /// pooled buffer is longer than its chunk, see [`encode_block`]).
    Chunk {
        buf: Vec<u8>,
        len: usize,
        input: Vec<u8>,
    },
    /// A 0x01 chunk: `hdr` then `input`, written without a copy.
    Stored {
        hdr: [u8; CHUNK_HEADER_SIZE + CHECKSUM_SIZE],
        input: Vec<u8>,
    },
}

struct BlockJob {
    /// Owned input bytes to compress.  Pulled from `input_pool` by
    /// the dispatcher and returned by the writer thread via [`Encoded`].
    uncompressed: Vec<u8>,
    level: block::Level,
    uncompressed_mode: bool,
    /// Single-slot channel the worker writes its result into.
    result_tx: SyncSender<io::Result<Encoded>>,
    /// Pool to acquire the output buffer from + return it through the
    /// writer thread.
    obuf_pool: Arc<BufferPool>,
}

/// How long a parked [`Engine`] waits for the next [`MtWriter`] of its shape before it exits.
const ENGINE_IDLE: Duration = if cfg!(test) {
    Duration::from_secs(2)
} else {
    Duration::from_secs(10)
};

/// Parked engines, most recently parked last.
static IDLE: Mutex<Vec<Engine>> = Mutex::new(Vec::new());

static NEXT_ENGINE_ID: AtomicU64 = AtomicU64::new(0);

/// One stream's `writer_loop`, boxed so an engine serves any `W`.
type Run = Box<dyn FnOnce() + Send>;

/// The threads and buffers of one [`MtWriter`], parked after a successful
/// [`finish`](MtWriter::finish) for the next writer of the same shape: the encoders keep their
/// scratch in thread locals, which fresh threads must allocate (and fault in) again.
struct Engine {
    id: u64,
    /// A `fork()`ed child inherits parked engines but not their threads.
    pid: u32,
    /// `(concurrency, block_size)`.
    key: (usize, usize),
    /// One submit channel per worker — the dispatcher picks via
    /// round-robin so workers never contend on a shared receiver.
    submit_txs: Vec<SyncSender<BlockJob>>,
    workers: Vec<JoinHandle<()>>,
    /// `Some` until dropped; closing it stops the runner.
    run_tx: Option<Sender<Run>>,
    /// Runs one stream's `writer_loop` at a time.
    runner: Option<JoinHandle<()>>,
    /// Pool of 1-block input buffers.  Round-trips through the worker
    /// + writer thread back to the dispatcher on every block.
    input_pool: Arc<BufferPool>,
    /// Pool of compressed-output buffers (worker writes obuf, writer
    /// thread returns).
    obuf_pool: Arc<BufferPool>,
}

impl Engine {
    /// The most recently parked engine of this shape, or a new one.
    fn take(concurrency: usize, block_size: usize) -> Self {
        let key = (concurrency, block_size);
        let pid = std::process::id();
        let parked = {
            let mut idle = IDLE.lock().unwrap();
            // Dropping would join threads that do not exist in this process.
            idle.extract_if(.., |e| e.pid != pid)
                .for_each(std::mem::forget);
            idle.iter()
                .rposition(|e| e.key == key)
                .map(|i| idle.remove(i))
        };
        parked.unwrap_or_else(|| Self::new(key))
    }

    fn new(key: (usize, usize)) -> Self {
        let (concurrency, block_size) = key;
        // Capacity 1 per worker: backpressure kicks in as soon as a worker
        // is busy, but the dispatcher can fan out to another worker without blocking.
        let (submit_txs, workers) = (0..concurrency)
            .map(|_| {
                let (tx, rx) = mpsc::sync_channel::<BlockJob>(1);
                (tx, std::thread::spawn(move || worker_loop(rx)))
            })
            .unzip();
        let id = NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed);
        let (run_tx, run_rx) = mpsc::channel::<Run>();
        let runner = std::thread::spawn(move || runner_loop(run_rx, id));
        // encode_block needs the chunk header, the checksum or hash, and MaxEncodedLen(block_size)
        // plus the encoders' slack; a smaller buffer would double on its first full block.
        let obuf_cap = CHUNK_HEADER_SIZE
            + SIMD_HASH_SIZE
            + block::max_encoded_len(block_size).unwrap_or(block_size + 16)
            + block::ENCODE_SLACK;
        // Keep every buffer a stream has in flight, so the next stream allocates none: up to
        // concurrency + 3 inputs (the order queue, the block being written, the one being
        // submitted) plus `ibuf`, and concurrency + 2 outputs.
        Self {
            id,
            pid: std::process::id(),
            key,
            submit_txs,
            workers,
            run_tx: Some(run_tx),
            runner: Some(runner),
            input_pool: Arc::new(BufferPool::new(concurrency + 4, block_size)),
            obuf_pool: Arc::new(BufferPool::new(concurrency + 2, obuf_cap)),
        }
    }

    fn park(self) {
        // Under Miri no thread may outlive its test, so the engine drops here.
        if !cfg!(miri) {
            IDLE.lock().unwrap().push(self);
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.submit_txs.clear();
        drop(self.run_tx.take());
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
        // An expiring runner drops its own engine.
        let me = std::thread::current().id();
        if let Some(h) = self.runner.take().filter(|h| h.thread().id() != me) {
            let _ = h.join();
        }
    }
}

/// Runs streams until the engine drops, or until it has sat parked for [`ENGINE_IDLE`].
fn runner_loop(run_rx: Receiver<Run>, id: u64) {
    loop {
        match run_rx.recv_timeout(ENGINE_IDLE) {
            Ok(run) => run(),
            Err(RecvTimeoutError::Disconnected) => return,
            Err(RecvTimeoutError::Timeout) => {
                let mut idle = IDLE.lock().unwrap();
                // Not parked: a writer just took it and is sending its stream.
                let Some(i) = idle.iter().position(|e| e.id == id) else {
                    continue;
                };
                let engine = idle.remove(i);
                drop(idle);
                drop(engine);
                return;
            }
        }
    }
}

/// Builder for [`MtWriter`].  All knobs match [`super::WriterBuilder`]
/// except `concurrency`, which selects the worker count.
#[must_use]
pub struct MtWriterBuilder {
    block_size: usize,
    level: block::Level,
    uncompressed: bool,
    padding: u32,
    /// Requested worker count; `0` is the host's available parallelism.
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
            concurrency: 0,
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
    /// The SIMD levels write SIMD chunks, which need a decoder for MinLZ format specification
    /// v1.1 or later.
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

    /// Number of worker threads to spin up; `0` (the default) uses the host's available
    /// parallelism. `1` still uses the MT pipeline (one worker); for the lowest-overhead path use
    /// [`super::Writer`].
    pub fn concurrency(mut self, n: usize) -> Self {
        self.concurrency = n;
        self
    }

    /// Toggle in-memory index generation.  Default `true` so that
    /// [`MtWriter::close_index`] can return an index.  Set to `false` for
    /// streaming output where no index is ever needed.
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
    /// [`io::ErrorKind::InvalidInput`] carrying an [`Error::Config`] if `block_size` is outside
    /// `[MIN_BLOCK_SIZE, MAX_BLOCK_SIZE]` or `padding` exceeds `MAX_BLOCK_SIZE`.
    pub fn build<W: Write + Send + 'static>(self, w: W) -> io::Result<MtWriter<W>> {
        if !(MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE).contains(&self.block_size) {
            return Err(
                Error::Config("block_size must be in [MIN_BLOCK_SIZE, MAX_BLOCK_SIZE]").into(),
            );
        }
        if self.padding as usize > MAX_BLOCK_SIZE {
            return Err(Error::Config("padding must be ≤ MAX_BLOCK_SIZE").into());
        }
        let concurrency = super::worker_count(self.concurrency);
        Ok(MtWriter::with_builder(w, self, concurrency))
    }
}

/// Multi-threaded MinLZ stream writer.
///
/// Workers compress blocks in parallel.  A single writer thread drains
/// per-block result channels in submission order and writes to `W`.
/// `W` is moved into the writer thread on construction and returned via
/// [`finish`] or [`close_index`].  See module-level docs for the rationale.
///
/// After a successful [`finish`] or [`close_index`] the worker threads, the
/// writer thread and the block buffers stay alive for about 10 seconds, for the
/// next `MtWriter` with the same concurrency and block size to reuse; unused,
/// they exit.
///
/// [`finish`]: MtWriter::finish
/// [`close_index`]: MtWriter::close_index
pub struct MtWriter<W: Write + Send + 'static> {
    block_size: usize,
    level: block::Level,
    uncompressed: bool,
    ibuf: Vec<u8>,
    /// Round-robin cursor for the engine's `submit_txs`.
    next_worker: usize,
    order_tx: Option<SyncSender<OrderItem>>,
    /// `W` and the serialized index (empty unless requested).
    finish_rx: Receiver<io::Result<(W, Vec<u8>)>>,
    has_index: bool,
    /// `Some` until a successful [`finish`] parks it; otherwise dropped with
    /// the writer.
    engine: Option<Engine>,
    /// Shared error slot — set by the writer thread if a downstream
    /// write fails.  Read by [`MtWriter::write`] and [`finish`].
    err: Arc<Mutex<Option<io::Error>>>,
    uncomp_written: u64,
    /// Compressed bytes written to `W`, updated by the writer thread.
    comp_written: Arc<AtomicU64>,
    wrote_header: bool,
    _w: PhantomData<fn() -> W>,
}

impl<W: Write + Send + 'static> MtWriter<W> {
    /// Wrap `w` with default options and the host's available parallelism.
    pub fn new(w: W) -> Self {
        MtWriterBuilder::new()
            .build(w)
            .expect("default MtWriterBuilder options are always valid")
    }

    fn with_builder(w: W, b: MtWriterBuilder, concurrency: usize) -> Self {
        let engine = Engine::take(concurrency, b.block_size);
        let (order_tx, order_rx) = mpsc::sync_channel::<OrderItem>(concurrency + 1);
        let (finish_tx, finish_rx) = mpsc::sync_channel::<io::Result<(W, Vec<u8>)>>(1);
        let err: Arc<Mutex<Option<io::Error>>> = Arc::new(Mutex::new(None));

        let err_writer = err.clone();
        let comp_written = Arc::new(AtomicU64::new(0));
        let comp_counter = comp_written.clone();
        let input_pool_w = engine.input_pool.clone();
        let obuf_pool_w = engine.obuf_pool.clone();
        let writer_index = if b.generate_index || b.append_index {
            let mut idx = Index::default();
            idx.reset(b.block_size);
            // Totals stay unknown (None) until `append_to` records them at close.
            Some(idx)
        } else {
            None
        };
        let append_index_w = b.append_index;
        let padding_w = b.padding;
        // The runner is alive: a failed send would drop `finish_tx`, which `finish` reports.
        let _ = engine.run_tx.as_ref().unwrap().send(Box::new(move || {
            writer_loop(
                w,
                order_rx,
                finish_tx,
                err_writer,
                comp_counter,
                input_pool_w,
                obuf_pool_w,
                writer_index,
                append_index_w,
                padding_w,
            );
        }));

        Self {
            block_size: b.block_size,
            level: b.level,
            uncompressed: b.uncompressed,
            ibuf: engine.input_pool.acquire(),
            next_worker: 0,
            order_tx: Some(order_tx),
            finish_rx,
            has_index: b.generate_index || b.append_index,
            engine: Some(engine),
            err,
            uncomp_written: 0,
            comp_written,
            wrote_header: false,
            _w: PhantomData,
        }
    }

    /// Total uncompressed bytes accepted, and compressed bytes the writer thread has written to
    /// the sink. The compressed count lags behind the input until [`Write::flush`].
    pub fn written(&self) -> Written {
        Written {
            uncompressed: self.uncomp_written,
            compressed: self.comp_written.load(Ordering::Relaxed),
        }
    }

    /// Flush pending input, then encode `buf` as whole blocks.  Like Go's
    /// `Writer.EncodeBuffer`, but each block is copied for its worker, so
    /// `buf` may be reused as soon as this returns.
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

    /// Flush pending input, emit the EOF chunk + any padding, wait for the
    /// writer thread, and return the underlying writer.
    pub fn finish(mut self) -> io::Result<W> {
        self.close(false).map(|(w, _)| w)
    }

    /// Like [`Self::finish`] but also returns the encoded index, as
    /// [`Writer::close_index`](super::Writer::close_index) does.
    /// Requires `generate_index` (the default) to have been left enabled.
    pub fn close_index(mut self) -> io::Result<(W, Vec<u8>)> {
        if !self.has_index {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "minlz: close_index called but generate_index is false",
            ));
        }
        self.close(true)
    }

    fn close(&mut self, build_index: bool) -> io::Result<(W, Vec<u8>)> {
        self.check_err()?;
        self.flush_ibuf()?;
        // EOF chunk; an empty stream is this chunk alone, as Go writes it.
        let mut tmp = [0u8; CHUNK_HEADER_SIZE + MAX_VARINT_LEN_64];
        tmp[0] = CHUNK_TYPE_EOF;
        let n = put_uvarint(&mut tmp[CHUNK_HEADER_SIZE..], self.uncomp_written);
        tmp[1] = n as u8;
        self.submit_raw(tmp[..CHUNK_HEADER_SIZE + n].to_vec())?;

        // A failed send means the writer thread already stopped on an error,
        // which `finish_rx` reports.
        let _ = self
            .order_tx
            .take()
            .unwrap()
            .send(OrderItem::Finish(build_index));
        let res = match self.finish_rx.recv() {
            Ok(res) => res,
            Err(_) => Err(io::Error::other("writer thread did not return W")),
        };
        // After an error a worker or the runner may be gone, so only a clean
        // stream parks the engine.
        if res.is_ok() {
            let engine = self.engine.take().unwrap();
            engine.input_pool.release(std::mem::take(&mut self.ibuf));
            engine.park();
        }
        res
    }

    fn engine(&self) -> &Engine {
        // `Some` until `finish` parks it.
        self.engine.as_ref().unwrap()
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
        let hdr = make_stream_header(self.block_size, simd_stream(self.level, self.uncompressed));
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
        let mut input = self.engine().input_pool.acquire();
        input.extend_from_slice(block);
        let (result_tx, result_rx) = mpsc::sync_channel::<io::Result<Encoded>>(1);
        // Push order slot first so the writer thread can drain in order.
        // `order_tx` is `Some` until `finish()`/`Drop` (see `submit_raw`).
        if self
            .order_tx
            .as_ref()
            .unwrap()
            .send(OrderItem::Block(result_rx))
            .is_err()
        {
            self.engine().input_pool.release(input);
            return Err(self.collect_pipeline_error());
        }
        let job = BlockJob {
            uncompressed: input,
            level: self.level,
            uncompressed_mode: self.uncompressed,
            result_tx,
            obuf_pool: self.engine().obuf_pool.clone(),
        };
        // Round-robin pick across worker submit channels — no
        // Mutex<Receiver> contention, just one `send` per block.
        let n = self.engine().submit_txs.len();
        let idx = self.next_worker % n;
        self.next_worker = (self.next_worker + 1) % n;
        match self.engine().submit_txs[idx].try_send(job) {
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
        let n = self.engine().submit_txs.len();
        loop {
            let slot = idx % n;
            idx = (idx + 1) % n;
            match self.engine().submit_txs[slot].try_send(job) {
                Ok(()) => {
                    self.next_worker = idx;
                    return Ok(());
                }
                Err(mpsc::TrySendError::Full(j)) => {
                    tried += 1;
                    job = j;
                    if tried >= n {
                        // Block on the next one.
                        return self.engine().submit_txs[slot]
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

    /// Write all pending input to `W` and flush it, but do not emit EOF.
    fn flush(&mut self) -> io::Result<()> {
        self.check_err()?;
        self.flush_ibuf()?;
        let (done_tx, done_rx) = mpsc::sync_channel(1);
        // `order_tx` is `Some` until `finish()`/`Drop` (see `submit_raw`).
        let sent = self
            .order_tx
            .as_ref()
            .unwrap()
            .send(OrderItem::Flush(done_tx));
        // A failed writer thread drops `done_tx` unanswered.
        if sent.is_err() || done_rx.recv().is_err() {
            return Err(self.collect_pipeline_error());
        }
        Ok(())
    }
}

impl<W: Write + Send + 'static> Drop for MtWriter<W> {
    fn drop(&mut self) {
        // Best-effort cleanup if the user forgot finish(): close the order
        // channel and wait for the writer thread, so `W` is dropped before we
        // return.  Queued blocks are written; buffered input, the EOF chunk,
        // padding and index are not, as in Go without Close.  The engine
        // drops after this and joins its threads.
        if let Some(tx) = self.order_tx.take() {
            drop(tx);
            let _ = self.finish_rx.recv();
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
        let mut obuf = obuf_pool.reuse();
        let n = if uncompressed_mode {
            0
        } else {
            match encode_block(&mut obuf, &uncompressed, level) {
                Ok(n) => n,
                Err(e) => {
                    obuf_pool.release(obuf);
                    let e = io::Error::new(io::ErrorKind::InvalidData, format!("encode: {e}"));
                    let _ = result_tx.send(Err(e));
                    continue;
                }
            }
        };
        let out = if n > 0 {
            Encoded::Chunk {
                buf: obuf,
                len: n,
                input: uncompressed,
            }
        } else {
            obuf_pool.release(obuf);
            let mut hdr = [0u8; CHUNK_HEADER_SIZE + CHECKSUM_SIZE];
            put_uncompressed_header(&mut hdr, &uncompressed);
            Encoded::Stored {
                hdr,
                input: uncompressed,
            }
        };
        let _ = result_tx.send(Ok(out));
    }
}

#[allow(clippy::too_many_arguments)]
fn writer_loop<W: Write + Send + 'static>(
    mut w: W,
    order_rx: Receiver<OrderItem>,
    finish_tx: SyncSender<io::Result<(W, Vec<u8>)>>,
    err_slot: Arc<Mutex<Option<io::Error>>>,
    comp_counter: Arc<AtomicU64>,
    input_pool: Arc<BufferPool>,
    obuf_pool: Arc<BufferPool>,
    mut index: Option<Index>,
    append_index: bool,
    padding: u32,
) {
    let mut comp_written: u64 = 0;
    let mut uncomp_written: u64 = 0;

    let res = (|| -> io::Result<Vec<u8>> {
        let build_index = loop {
            // Closed without `Finish`: the MtWriter was dropped.
            let Ok(item) = order_rx.recv() else {
                return Ok(Vec::new());
            };
            match item {
                OrderItem::Raw(bytes) => {
                    w.write_all(&bytes).map_err(|e| record(&err_slot, e))?;
                    comp_written += bytes.len() as u64;
                    comp_counter.store(comp_written, Ordering::Relaxed);
                }
                OrderItem::Flush(done) => {
                    w.flush().map_err(|e| record(&err_slot, e))?;
                    let _ = done.send(());
                }
                OrderItem::Finish(build_index) => break build_index,
                OrderItem::Block(rx) => {
                    let out = rx
                        .recv()
                        .unwrap_or_else(|_| {
                            Err(io::Error::other(
                                "MinLZ worker exited without producing output",
                            ))
                        })
                        .map_err(|e| record(&err_slot, e))?;
                    // Index entries are the offsets at the start of the chunk.
                    if let Some(idx) = index.as_mut() {
                        idx.add(comp_written, uncomp_written)
                            .map_err(|e| record(&err_slot, e))?;
                    }
                    let (res, chunk_len, input) = match out {
                        Encoded::Chunk { buf, len, input } => {
                            let res = w.write_all(&buf[..len]);
                            obuf_pool.release(buf);
                            (res, len, input)
                        }
                        Encoded::Stored { hdr, input } => {
                            let res = w.write_all(&hdr).and_then(|()| w.write_all(&input));
                            (res, hdr.len() + input.len(), input)
                        }
                    };
                    uncomp_written += input.len() as u64;
                    input_pool.release(input);
                    res.map_err(|e| record(&err_slot, e))?;
                    comp_written += chunk_len as u64;
                    comp_counter.store(comp_written, Ordering::Relaxed);
                }
            }
        };
        // Wire order from here matches Go's `writer.go:closeIndex`:
        // data → EOF → padding → index.  Padding is emitted *before*
        // the index but its size includes the index length, so the
        // total compressed size is aligned to `padding`.
        let mut idx_bytes = Vec::new();
        if (build_index || append_index)
            && let Some(idx) = index.as_mut()
        {
            // `total_compressed` is bytes-before-index; if padding will
            // be added, store -1 (unknown) so seekers don't rely on it.
            let comp_total = if padding <= 1 {
                Some(comp_written)
            } else {
                None
            };
            idx.append_to(&mut idx_bytes, Some(uncomp_written), comp_total)
                .map_err(|e| record(&err_slot, e))?;
            if append_index {
                comp_written += idx_bytes.len() as u64;
            }
        }

        if padding > 1 {
            emit_mt_padding(&mut w, &mut comp_written, padding, &err_slot)?;
        }

        if append_index {
            w.write_all(&idx_bytes).map_err(|e| record(&err_slot, e))?;
        }
        w.flush().map_err(|e| record(&err_slot, e))?;
        Ok(idx_bytes)
    })();
    let payload = res.map(|idx_bytes| (w, idx_bytes));
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

#[cfg(test)]
mod tests;
