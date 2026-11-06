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

//! The hash tables of the encoders by entry size and bits, Go's `tablePool`s (`encode.go`).
//! Pooled tables keep the encoders from allocating per block, and large tables off the stack.

use std::cell::Cell;
use std::thread::LocalKey;

/// A per-thread pool of one table.
pub(super) type Pool<T> = LocalKey<Cell<Option<Box<T>>>>;

thread_local! {
    pub(super) static U32_13: Cell<Option<Box<[u32; 1 << 13]>>> = const { Cell::new(None) };
    pub(super) static U32_15: Cell<Option<Box<[u32; 1 << 15]>>> = const { Cell::new(None) };
    pub(super) static U16_13: Cell<Option<Box<[u16; 1 << 13]>>> = const { Cell::new(None) };
    pub(super) static U16_14: Cell<Option<Box<[u16; 1 << 14]>>> = const { Cell::new(None) };
    pub(super) static U32_14: Cell<Option<Box<[u32; 1 << 14]>>> = const { Cell::new(None) };
    /// Go's `encLPool`.
    pub(super) static U32_17: Cell<Option<Box<[u32; 1 << 17]>>> = const { Cell::new(None) };
    /// Go's `encLPool64K`.
    pub(super) static U16_15: Cell<Option<Box<[u16; 1 << 15]>>> = const { Cell::new(None) };
    /// Go's `encLPoolSIMD64K`.
    pub(super) static U16_16: Cell<Option<Box<[u16; 1 << 16]>>> = const { Cell::new(None) };
    /// Go's `encBestLPool`.
    pub(super) static U64_20: Cell<Option<Box<[u64; 1 << 20]>>> = const { Cell::new(None) };
    /// Go's `encBestSPool`.
    pub(super) static U64_18: Cell<Option<Box<[u64; 1 << 18]>>> = const { Cell::new(None) };
}

/// Returns a zeroed table from p.
pub(super) fn get<E: Copy + Default, const N: usize>(p: &'static Pool<[E; N]>) -> Box<[E; N]> {
    match p.take() {
        Some(mut t) => {
            t.fill(E::default());
            t
        }
        None => match vec![E::default(); N].into_boxed_slice().try_into() {
            Ok(t) => t,
            Err(_) => unreachable!("the vector has N entries"),
        },
    }
}

/// Returns t to p.
pub(super) fn put<T>(p: &'static Pool<T>, t: Box<T>) {
    p.set(Some(t));
}
