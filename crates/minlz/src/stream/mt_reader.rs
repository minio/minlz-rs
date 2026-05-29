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

//! Parallel-decode helper invoked by [`super::Reader::decode_concurrent`].
//!
//! The dispatcher (caller's thread) reads chunk headers serially.  Data
//! chunks (0x01 / 0x02 / 0x03) are handed to a worker pool for parallel
//! decoding.  Metadata chunks (stream id, EOF, padding, user-chunk,
//! skippables) are handled inline.  A dedicated writer thread drains
//! the per-block result receivers in submission order so the decoded
//! output is byte-identical to a single-threaded decode.
//!
//! Mirrors Go's `reader.go::DecodeConcurrent`.

use std::io::{self, Read, Write};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

use crate::block;

use super::crc::masked_crc32c;
use super::error::Error;
use super::format::{
    block_size_from_indicator, get_uvarint, read_chunk_len, read_u32_le, CHECKSUM_SIZE,
    CHUNK_HEADER_SIZE, CHUNK_TYPE_EOF, CHUNK_TYPE_MINLZ_COMPRESSED_DATA,
    CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC, CHUNK_TYPE_STREAM_IDENTIFIER,
    CHUNK_TYPE_UNCOMPRESSED_DATA, MAGIC_BODY, MAGIC_BODY_LEN, MAX_BLOCK_SIZE,
    MAX_NON_SKIPPABLE_CHUNK, MAX_USER_NON_SKIPPABLE_CHUNK, MAX_VARINT_LEN_64,
    MIN_USER_NON_SKIPPABLE_CHUNK, MIN_USER_SKIPPABLE_CHUNK,
};
use super::mt_pool::BufferPool;
use super::reader::Reader;

/// Worker job: decode one compressed/uncompressed chunk body.
struct DecodeJob {
    chunk_type: u8,
    body: Vec<u8>,
    checksum: u32,
    ignore_crc: bool,
    max_block: usize,
    result_tx: SyncSender<DecodeOutput>,
    /// Pool to return the (input) compressed buffer to after decode.
    compressed_pool: Arc<BufferPool>,
    /// Pool to acquire the decoded output buffer from.
    decoded_pool: Arc<BufferPool>,
}

/// Worker output sent to the writer thread: (decoded body, ok flag).
/// The writer thread releases `body` back to its pool after writing.
struct DecodeOutput {
    result: Result<Vec<u8>, Error>,
}

pub(super) fn decode_concurrent_into<R: Read, W: Write + Send + 'static>(
    reader: &mut Reader<R>,
    w: W,
    threads: usize,
) -> io::Result<(u64, W)> {
    let threads = threads.max(1);
    // One submit channel per worker (capacity 1) for round-robin
    // dispatch — no Mutex<Receiver> contention.
    let mut submit_txs = Vec::with_capacity(threads);
    let mut submit_rxs = Vec::with_capacity(threads);
    for _ in 0..threads {
        let (tx, rx) = mpsc::sync_channel::<DecodeJob>(1);
        submit_txs.push(tx);
        submit_rxs.push(rx);
    }
    let (order_tx, order_rx) = mpsc::sync_channel::<Receiver<DecodeOutput>>(threads + 1);
    let (finish_tx, finish_rx) = mpsc::sync_channel::<io::Result<(u64, W)>>(1);
    let err: Arc<Mutex<Option<io::Error>>> = Arc::new(Mutex::new(None));
    let pool_slots = threads + 1;
    let compressed_pool = Arc::new(BufferPool::new(pool_slots, reader.max_block / 8));
    let decoded_pool = Arc::new(BufferPool::new(pool_slots, reader.max_block));

    let mut workers = Vec::with_capacity(threads);
    for rx in submit_rxs {
        workers.push(std::thread::spawn(move || worker_loop(rx)));
    }
    let err_writer = err.clone();
    let decoded_pool_w = decoded_pool.clone();
    let writer_handle = std::thread::spawn(move || {
        writer_loop(w, order_rx, finish_tx, err_writer, decoded_pool_w);
    });

    let dispatch_res = dispatch_loop(
        reader,
        &mut submit_txs,
        &order_tx,
        &err,
        &compressed_pool,
        &decoded_pool,
    );

    submit_txs.clear();
    drop(order_tx);
    for h in workers {
        let _ = h.join();
    }
    let _ = writer_handle.join();

    let pipeline_res = finish_rx
        .recv()
        .unwrap_or_else(|_| Err(io::Error::other("MinLZ writer thread exited unexpectedly")));

    match (dispatch_res, pipeline_res) {
        (Ok(()), Ok(payload)) => Ok(payload),
        (Err(e), _) => Err(e),
        (_, Err(e)) => Err(e),
    }
}

fn dispatch_loop<R: Read>(
    reader: &mut Reader<R>,
    submit_txs: &mut [SyncSender<DecodeJob>],
    order_tx: &SyncSender<Receiver<DecodeOutput>>,
    err: &Arc<Mutex<Option<io::Error>>>,
    compressed_pool: &Arc<BufferPool>,
    decoded_pool: &Arc<BufferPool>,
) -> io::Result<()> {
    let mut uncomp_emitted: u64 = 0;
    let mut next_worker: usize = 0;
    loop {
        if let Some(e) = err.lock().unwrap().as_ref() {
            return Err(clone_io_err(e));
        }
        let mut hdr = [0u8; CHUNK_HEADER_SIZE];
        match read_full_or_eof(&mut reader.r, &mut hdr, !reader.expect_eof)? {
            ReadOutcome::Eof => return Ok(()),
            ReadOutcome::Ok => {}
        }
        let chunk_type = hdr[0];
        let chunk_len = read_chunk_len(&hdr[1..]);

        if !reader.header_read {
            if chunk_type == CHUNK_TYPE_STREAM_IDENTIFIER {
                reader.header_read = true;
            } else if chunk_type <= MAX_NON_SKIPPABLE_CHUNK && chunk_type != CHUNK_TYPE_EOF {
                return Err(io::Error::from(Error::Corrupt));
            }
        }

        match chunk_type {
            CHUNK_TYPE_MINLZ_COMPRESSED_DATA | CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC
                if chunk_len >= CHECKSUM_SIZE =>
            {
                if chunk_len > max_buf_size(reader.max_block) {
                    return Err(io::Error::from(Error::Corrupt));
                }
                // Acquire a pooled buffer, sized for the full chunk
                // (incl. checksum prefix).  We strip the checksum below
                // before the worker sees it.
                let mut buf = compressed_pool.acquire();
                buf.resize(chunk_len, 0);
                read_exact_or_corrupt(&mut reader.r, &mut buf)?;
                let checksum = read_u32_le(&buf[..CHECKSUM_SIZE]);
                // Shift body to the start, truncate — keeps the same
                // allocation but removes the checksum prefix.  Avoids
                // `split_off`'s extra alloc.
                buf.copy_within(CHECKSUM_SIZE.., 0);
                buf.truncate(chunk_len - CHECKSUM_SIZE);
                let body = buf;
                let dlen = block::decoded_len_chunk_body(&body)
                    .map_err(|e| io::Error::from(Error::Block(e)))?;
                uncomp_emitted += dlen as u64;
                submit_compressed(
                    submit_txs,
                    &mut next_worker,
                    order_tx,
                    chunk_type,
                    body,
                    checksum,
                    reader.ignore_crc,
                    reader.max_block,
                    compressed_pool,
                    decoded_pool,
                )?;
            }
            CHUNK_TYPE_UNCOMPRESSED_DATA if chunk_len >= CHECKSUM_SIZE => {
                if chunk_len > max_buf_size(reader.max_block) {
                    return Err(io::Error::from(Error::Corrupt));
                }
                let n = chunk_len - CHECKSUM_SIZE;
                if n > reader.max_block {
                    return Err(io::Error::from(Error::TooLarge));
                }
                let mut cbuf = [0u8; CHECKSUM_SIZE];
                read_exact_or_corrupt(&mut reader.r, &mut cbuf)?;
                let checksum = read_u32_le(&cbuf);
                let mut body = decoded_pool.acquire();
                body.resize(n, 0);
                read_exact_or_corrupt(&mut reader.r, &mut body)?;
                if !reader.ignore_crc && masked_crc32c(&body) != checksum {
                    decoded_pool.release(body);
                    return Err(io::Error::from(Error::Crc));
                }
                uncomp_emitted += n as u64;
                let (tx, rx) = mpsc::sync_channel::<DecodeOutput>(1);
                if order_tx.send(rx).is_err() {
                    decoded_pool.release(body);
                    return Err(io::Error::other("MinLZ writer thread closed"));
                }
                let _ = tx.send(DecodeOutput { result: Ok(body) });
            }
            CHUNK_TYPE_EOF => {
                if chunk_len > MAX_VARINT_LEN_64 {
                    return Err(io::Error::from(Error::Corrupt));
                }
                if chunk_len > 0 {
                    let mut tmp = [0u8; MAX_VARINT_LEN_64];
                    read_exact_or_corrupt(&mut reader.r, &mut tmp[..chunk_len])?;
                    if !reader.ignore_stream_id {
                        let (want, n) = get_uvarint(&tmp[..chunk_len]).ok_or(Error::Corrupt)?;
                        if n != chunk_len || want != uncomp_emitted {
                            return Err(io::Error::from(Error::Corrupt));
                        }
                    }
                }
                reader.expect_eof = false;
                reader.header_read = false;
            }
            CHUNK_TYPE_STREAM_IDENTIFIER => {
                if chunk_len != MAGIC_BODY_LEN {
                    return Err(io::Error::from(Error::Corrupt));
                }
                let mut tmp = [0u8; MAGIC_BODY_LEN];
                read_exact_or_corrupt(&mut reader.r, &mut tmp)?;
                if &tmp[..MAGIC_BODY.len()] != MAGIC_BODY {
                    return Err(io::Error::from(Error::Unsupported));
                }
                let indicator = tmp[MAGIC_BODY_LEN - 1];
                if indicator & 0xc0 != 0 {
                    return Err(io::Error::from(Error::Corrupt));
                }
                let new_max = block_size_from_indicator(indicator).ok_or(Error::Corrupt)?;
                if new_max > reader.max_block_user {
                    return Err(io::Error::from(Error::TooLarge));
                }
                reader.max_block = new_max;
                uncomp_emitted = 0;
                reader.expect_eof = true;
            }
            t if t <= MAX_NON_SKIPPABLE_CHUNK => {
                return Err(io::Error::from(Error::Unsupported));
            }
            _ => {
                handle_skippable_dispatch(reader, chunk_type, chunk_len)?;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn submit_compressed(
    submit_txs: &mut [SyncSender<DecodeJob>],
    next_worker: &mut usize,
    order_tx: &SyncSender<Receiver<DecodeOutput>>,
    chunk_type: u8,
    body: Vec<u8>,
    checksum: u32,
    ignore_crc: bool,
    max_block: usize,
    compressed_pool: &Arc<BufferPool>,
    decoded_pool: &Arc<BufferPool>,
) -> io::Result<()> {
    let (tx, rx) = mpsc::sync_channel::<DecodeOutput>(1);
    if order_tx.send(rx).is_err() {
        compressed_pool.release(body);
        return Err(io::Error::other("MinLZ writer thread closed"));
    }
    let mut job = DecodeJob {
        chunk_type,
        body,
        checksum,
        ignore_crc,
        max_block,
        result_tx: tx,
        compressed_pool: compressed_pool.clone(),
        decoded_pool: decoded_pool.clone(),
    };
    // Round-robin across worker submit channels.
    let n = submit_txs.len();
    let mut tried = 0;
    loop {
        let idx = *next_worker % n;
        *next_worker = (*next_worker + 1) % n;
        match submit_txs[idx].try_send(job) {
            Ok(()) => return Ok(()),
            Err(mpsc::TrySendError::Full(j)) => {
                tried += 1;
                job = j;
                if tried >= n {
                    // All workers busy — block on this one.
                    return submit_txs[idx]
                        .send(job)
                        .map_err(|_| io::Error::other("MinLZ worker pool closed"));
                }
            }
            Err(mpsc::TrySendError::Disconnected(_)) => {
                return Err(io::Error::other("MinLZ worker pool closed"));
            }
        }
    }
}

fn handle_skippable_dispatch<R: Read>(
    reader: &mut Reader<R>,
    chunk_type: u8,
    chunk_len: usize,
) -> io::Result<()> {
    let id = chunk_type;
    if (MIN_USER_SKIPPABLE_CHUNK..=MAX_USER_NON_SKIPPABLE_CHUNK).contains(&id) {
        let cb_idx = (id - MIN_USER_SKIPPABLE_CHUNK) as usize;
        let mut maybe_cb = reader.callbacks[cb_idx].take();
        if let Some(ref mut cb) = maybe_cb {
            let mut buf = vec![0u8; chunk_len];
            let read_res = read_exact_or_corrupt(&mut reader.r, &mut buf);
            let invoke_res = read_res.and_then(|()| cb(id, &buf));
            reader.callbacks[cb_idx] = maybe_cb;
            invoke_res?;
            return Ok(());
        }
        if (MIN_USER_NON_SKIPPABLE_CHUNK..=MAX_USER_NON_SKIPPABLE_CHUNK).contains(&id) {
            return Err(io::Error::from(Error::Corrupt));
        }
    }
    skip_n(&mut reader.r, chunk_len)
}

fn skip_n<R: Read>(r: &mut R, mut n: usize) -> io::Result<()> {
    let mut tmp = [0u8; 4096];
    while n > 0 {
        let take = n.min(tmp.len());
        read_exact_or_corrupt(r, &mut tmp[..take])?;
        n -= take;
    }
    Ok(())
}

fn max_buf_size(max_block: usize) -> usize {
    block::max_encoded_len(max_block).unwrap_or(MAX_BLOCK_SIZE) + CHECKSUM_SIZE
}

// -------------------- worker / writer threads --------------------

fn worker_loop(submit_rx: Receiver<DecodeJob>) {
    while let Ok(job) = submit_rx.recv() {
        let DecodeJob {
            chunk_type,
            body,
            checksum,
            ignore_crc,
            max_block,
            result_tx,
            compressed_pool,
            decoded_pool,
        } = job;
        let mut decoded = decoded_pool.acquire();
        let res: Result<Vec<u8>, Error> = (|| {
            let dlen = block::decoded_len_chunk_body(&body)?;
            if dlen > max_block {
                return Err(Error::TooLarge);
            }
            block::append_decoded_chunk_body(&mut decoded, &body)?;
            if !ignore_crc {
                let to_crc: &[u8] = if chunk_type == CHUNK_TYPE_MINLZ_COMPRESSED_DATA_COMP_CRC {
                    let (_, n) = get_uvarint(&body).ok_or(Error::Corrupt)?;
                    &body[n..]
                } else {
                    &decoded[..]
                };
                if masked_crc32c(to_crc) != checksum {
                    return Err(Error::Crc);
                }
            }
            Ok(std::mem::take(&mut decoded))
        })();
        // body is the compressed buffer; return to its pool regardless.
        compressed_pool.release(body);
        // If we couldn't produce decoded output (or it was moved out
        // already), return whatever decoded buffer we still have.
        if !decoded.is_empty() || decoded.capacity() > 0 {
            decoded_pool.release(decoded);
        }
        let _ = result_tx.send(DecodeOutput { result: res });
    }
}

fn writer_loop<W: Write + Send + 'static>(
    mut w: W,
    order_rx: Receiver<Receiver<DecodeOutput>>,
    finish_tx: SyncSender<io::Result<(u64, W)>>,
    err_slot: Arc<Mutex<Option<io::Error>>>,
    decoded_pool: Arc<BufferPool>,
) {
    let mut written: u64 = 0;
    let res = (|| -> io::Result<()> {
        while let Ok(rx) = order_rx.recv() {
            match rx.recv() {
                Ok(DecodeOutput { result: Ok(buf) }) => {
                    let write_res = w.write_all(&buf).map_err(|e| record(&err_slot, e));
                    written += buf.len() as u64;
                    decoded_pool.release(buf);
                    write_res?;
                }
                Ok(DecodeOutput { result: Err(e) }) => return Err(record(&err_slot, e.into())),
                Err(_) => {
                    return Err(record(
                        &err_slot,
                        io::Error::other("MinLZ worker exited without producing output"),
                    ));
                }
            }
        }
        w.flush().map_err(|e| record(&err_slot, e))?;
        Ok(())
    })();
    let payload = res.map(|()| (written, w));
    let _ = finish_tx.send(payload);
}

fn record(slot: &Arc<Mutex<Option<io::Error>>>, e: io::Error) -> io::Error {
    let cloned = clone_io_err(&e);
    let mut guard = slot.lock().unwrap();
    if guard.is_none() {
        *guard = Some(e);
    }
    cloned
}

fn clone_io_err(e: &io::Error) -> io::Error {
    io::Error::new(e.kind(), e.to_string())
}

// -------------------- I/O helpers --------------------

enum ReadOutcome {
    Ok,
    Eof,
}

fn read_full_or_eof<R: Read>(
    r: &mut R,
    buf: &mut [u8],
    allow_eof: bool,
) -> io::Result<ReadOutcome> {
    let mut total = 0;
    while total < buf.len() {
        match r.read(&mut buf[total..]) {
            Ok(0) => {
                if total == 0 && allow_eof {
                    return Ok(ReadOutcome::Eof);
                }
                return Err(io::Error::from(Error::Corrupt));
            }
            Ok(n) => total += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(ReadOutcome::Ok)
}

fn read_exact_or_corrupt<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<()> {
    match r.read_exact(buf) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Err(io::Error::from(Error::Corrupt)),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests;
