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

//! Tiny buffer pool used by [`super::MtWriter`] and the parallel
//! decoder.  Mirrors Go's `sync.Pool` of `[]byte` on `writer.go`.
//!
//! The pool holds up to `max_items` empty-but-allocated `Vec<u8>`.
//! `acquire` either pops one or allocates a fresh `Vec` with the
//! requested capacity hint.  `release` clears the vec and pushes it
//! back if there's room, otherwise drops it.
//!
//! Bounded to keep memory steady; one pool is sized to
//! `concurrency + 1` per the MT plan ("one extra in-flight block").

use std::sync::Mutex;

pub(super) struct BufferPool {
    inner: Mutex<Vec<Vec<u8>>>,
    max_items: usize,
    cap_hint: usize,
}

impl BufferPool {
    pub(super) fn new(max_items: usize, cap_hint: usize) -> Self {
        Self {
            inner: Mutex::new(Vec::with_capacity(max_items)),
            max_items,
            cap_hint,
        }
    }

    /// Pop a recycled buffer or allocate a fresh one of at least
    /// `cap_hint` bytes capacity.  Returns an empty `Vec`.
    pub(super) fn acquire(&self) -> Vec<u8> {
        // `lock().unwrap()` only panics if another thread panicked while
        // holding this lock; that propagates a pre-existing worker panic
        // rather than introducing a new failure mode.
        if let Some(buf) = self.inner.lock().unwrap().pop() {
            return buf;
        }
        Vec::with_capacity(self.cap_hint)
    }

    /// Return a buffer to the pool (cleared first).  If the pool is
    /// full, the buffer is dropped.
    pub(super) fn release(&self, mut buf: Vec<u8>) {
        buf.clear();
        // Lock poison only propagates a prior worker panic (see `acquire`).
        let mut guard = self.inner.lock().unwrap();
        if guard.len() < self.max_items {
            guard.push(buf);
        }
    }
}
