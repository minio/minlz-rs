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

//! Test memory between inaccessible pages, so accesses outside a buffer fault.
//! Port of Go's `internal/guard`.

/// Memory between two inaccessible pages. Where page protection is unsupported (other
/// platforms, Miri), the memory is a plain allocation and not guarded.
pub(crate) struct Buffer {
    body: *mut u8,
    len: usize,
    #[cfg_attr(not(any(windows, unix)), allow(dead_code))]
    map: (*mut u8, usize),
    // Owns the allocation `body` points into where pages are not guarded.
    _fallback: Vec<u8>,
}

/// Reports whether [`Buffer`]s are guarded here.
pub(crate) const fn supported() -> bool {
    cfg!(all(
        any(windows, target_os = "linux", target_os = "macos"),
        not(miri)
    ))
}

impl Buffer {
    /// Returns a Buffer with room for at least `size` bytes, or `None` if mapping failed.
    pub(crate) fn new(size: usize) -> Option<Buffer> {
        if !supported() {
            let mut fallback = vec![0u8; size.max(1)];
            return Some(Buffer {
                body: fallback.as_mut_ptr(),
                len: size,
                map: (core::ptr::null_mut(), 0),
                _fallback: fallback,
            });
        }
        let pg = sys::page_size();
        let body = size.div_ceil(pg).max(1) * pg;
        let total = body + 2 * pg;
        let base = sys::map(total)?;
        // SAFETY: base maps `total` bytes; both guard pages lie inside it.
        let ok = unsafe { sys::protect(base, pg) && sys::protect(base.add(pg + body), pg) };
        let buf = Buffer {
            // SAFETY: pg < total.
            body: unsafe { base.add(pg) },
            len: body,
            map: (base, total),
            _fallback: Vec::new(),
        };
        ok.then_some(buf)
    }

    /// Returns `n` bytes flush against the inaccessible page after them if `at_end`,
    /// else against the one before them.
    pub(crate) fn slice(&mut self, n: usize, at_end: bool) -> &mut [u8] {
        let len = self.len;
        assert!(n <= len, "guard: {n} bytes do not fit {len}");
        let off = if at_end { len - n } else { 0 };
        // SAFETY: [off, off+n) lies inside the mapped (or allocated) body, which self owns.
        unsafe { core::slice::from_raw_parts_mut(self.body.add(off), n) }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        #[cfg(any(windows, target_os = "linux", target_os = "macos"))]
        if !self.map.0.is_null() {
            // SAFETY: map is the region sys::map returned, unmapped once.
            unsafe { sys::unmap(self.map.0, self.map.1) };
        }
    }
}

#[cfg(windows)]
mod sys {
    use core::ffi::c_void;

    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_NOACCESS: u32 = 0x01;
    const PAGE_READWRITE: u32 = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(addr: *mut c_void, size: usize, typ: u32, prot: u32) -> *mut c_void;
        fn VirtualProtect(addr: *mut c_void, size: usize, prot: u32, old: *mut u32) -> i32;
        fn VirtualFree(addr: *mut c_void, size: usize, typ: u32) -> i32;
        fn SetErrorMode(mode: u32) -> u32;
    }

    pub(super) fn page_size() -> usize {
        4096
    }

    pub(super) fn map(total: usize) -> Option<*mut u8> {
        // SAFETY: plain allocation call; a null result is handled.
        let p = unsafe {
            VirtualAlloc(
                core::ptr::null_mut(),
                total,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        (!p.is_null()).then_some(p.cast())
    }

    /// # Safety
    /// `p..p+n` must be inside a region returned by [`map`].
    pub(super) unsafe fn protect(p: *mut u8, n: usize) -> bool {
        let mut old = 0u32;
        // SAFETY: caller guarantees the range is mapped.
        unsafe { VirtualProtect(p.cast(), n, PAGE_NOACCESS, &mut old) != 0 }
    }

    /// # Safety
    /// `p` must be a region returned by [`map`], not yet unmapped.
    pub(super) unsafe fn unmap(p: *mut u8, _total: usize) {
        // SAFETY: caller guarantees p came from VirtualAlloc.
        unsafe { VirtualFree(p.cast(), 0, MEM_RELEASE) };
    }

    /// Ends the process at a fault without Windows Error Reporting's dialog.
    pub(super) fn no_fault_dialog() {
        const SEM_NOGPFAULTERRORBOX: u32 = 0x0002;
        // SAFETY: only sets the process error mode.
        unsafe { SetErrorMode(SEM_NOGPFAULTERRORBOX) };
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod sys {
    use core::ffi::{c_int, c_long, c_void};

    const PROT_NONE: c_int = 0;
    const PROT_RW: c_int = 1 | 2;
    const MAP_PRIVATE: c_int = 2;
    #[cfg(target_os = "linux")]
    const MAP_ANON: c_int = 0x20;
    #[cfg(target_os = "macos")]
    const MAP_ANON: c_int = 0x1000;
    #[cfg(target_os = "linux")]
    const SC_PAGESIZE: c_int = 30;
    #[cfg(target_os = "macos")]
    const SC_PAGESIZE: c_int = 29;

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            off: i64,
        ) -> *mut c_void;
        fn mprotect(addr: *mut c_void, len: usize, prot: c_int) -> c_int;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
        fn sysconf(name: c_int) -> c_long;
    }

    pub(super) fn page_size() -> usize {
        // SAFETY: sysconf has no preconditions.
        let n = unsafe { sysconf(SC_PAGESIZE) };
        if n > 0 { n as usize } else { 4096 }
    }

    pub(super) fn map(total: usize) -> Option<*mut u8> {
        // SAFETY: anonymous private mapping; failure (MAP_FAILED = -1) is handled.
        let p = unsafe {
            mmap(
                core::ptr::null_mut(),
                total,
                PROT_RW,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            )
        };
        (p as isize != -1 && !p.is_null()).then_some(p.cast())
    }

    /// # Safety
    /// `p..p+n` must be a page-aligned range inside a region returned by [`map`].
    pub(super) unsafe fn protect(p: *mut u8, n: usize) -> bool {
        // SAFETY: caller guarantees the range is mapped and page aligned.
        unsafe { mprotect(p.cast(), n, PROT_NONE) == 0 }
    }

    /// # Safety
    /// `p` must be a region of `total` bytes returned by [`map`], not yet unmapped.
    pub(super) unsafe fn unmap(p: *mut u8, total: usize) {
        // SAFETY: caller guarantees p/total came from mmap.
        unsafe { munmap(p.cast(), total) };
    }
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
mod sys {
    pub(super) fn page_size() -> usize {
        4096
    }
    pub(super) fn map(_total: usize) -> Option<*mut u8> {
        None
    }
    pub(super) unsafe fn protect(_p: *mut u8, _n: usize) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slices_are_flush() {
        let mut g = Buffer::new(100).unwrap();
        let s = g.slice(100, true);
        s.fill(7);
        assert_eq!(s.len(), 100);
        let s = g.slice(10, false);
        s.fill(1);
        assert!(s.iter().all(|&b| b == 1));
    }

    #[test]
    #[should_panic(expected = "do not fit")]
    fn oversized_slice_panics() {
        Buffer::new(16).unwrap().slice(1 << 20, true);
    }

    #[test]
    fn at_end_is_flush_with_body() {
        let mut g = Buffer::new(16).unwrap();
        let end = g.body.wrapping_add(g.len);
        assert_eq!(g.slice(4, true).as_mut_ptr_range().end, end);
    }

    /// Go's TestFault: the byte just outside a slice, at either end, faults. A fault ends the
    /// process, so this test runs itself as a child for each access.
    #[test]
    fn faults() {
        if !supported() {
            return;
        }
        if let Ok(c) = std::env::var("MINLZ_GUARD_FAULT") {
            let (n, at_end) = c.split_once(',').unwrap();
            let (n, at_end): (usize, bool) = (n.parse().unwrap(), at_end.parse().unwrap());
            #[cfg(windows)]
            sys::no_fault_dialog();
            let mut g = Buffer::new(10000).unwrap();
            let b = g.slice(n, at_end);
            assert_eq!(b.len(), n);
            b.iter_mut().enumerate().for_each(|(i, v)| *v = i as u8);
            let p = b
                .as_mut_ptr()
                .wrapping_offset(if at_end { n as isize } else { -1 });
            // SAFETY: deliberately not upheld: p is in a guard page, and the fault ends this child.
            unsafe { p.write_volatile(1) };
            return;
        }
        for n in [1, 100, 10000] {
            for at_end in [false, true] {
                let out = std::process::Command::new(std::env::current_exe().unwrap())
                    .args(["guard::tests::faults", "--exact", "--test-threads=1"])
                    .env("MINLZ_GUARD_FAULT", format!("{n},{at_end}"))
                    .output()
                    .unwrap();
                assert!(
                    faulted(out.status),
                    "n {n} at_end {at_end}: no fault: {out:?}"
                );
            }
        }
    }

    #[cfg(windows)]
    fn faulted(s: std::process::ExitStatus) -> bool {
        const STATUS_ACCESS_VIOLATION: u32 = 0xC000_0005;
        s.code() == Some(STATUS_ACCESS_VIOLATION as i32)
    }

    #[cfg(unix)]
    fn faulted(s: std::process::ExitStatus) -> bool {
        use std::os::unix::process::ExitStatusExt;
        // SIGSEGV, or SIGBUS (7 on Linux, 10 on macOS).
        matches!(s.signal(), Some(11 | 7 | 10))
    }
}
