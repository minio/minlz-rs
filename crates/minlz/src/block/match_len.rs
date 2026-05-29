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

//! Byte-wise prefix match length.
//!
//! Direct port of `asm_none.go:matchLen`.

use super::load_store::load64;

/// Returns how many leading bytes of `a` and `b` match.
///
/// Requires `a.len() <= b.len()`.
#[inline]
#[allow(dead_code)] // used by L2/L3 in future stages
pub(super) fn match_len(a: &[u8], b: &[u8]) -> usize {
    debug_assert!(a.len() <= b.len());
    let mut checked = 0;
    while a.len() - checked >= 8 {
        // SAFETY: `checked + 8 <= a.len() <= b.len()`.
        let diff = unsafe { load64(a, checked) ^ load64(b, checked) };
        if diff != 0 {
            return checked + (diff.trailing_zeros() as usize >> 3);
        }
        checked += 8;
    }
    let tail = &a[checked..];
    for (i, (&x, &y)) in tail.iter().zip(b[checked..].iter()).enumerate() {
        if x != y {
            return checked + i;
        }
    }
    a.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference impl: byte-by-byte.
    fn ref_match_len(a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    /// Ports `TestMatchLen` from `minlz_test.go:1607`.
    #[test]
    fn matches_reference() {
        let nums = [
            0_usize, 1, 2, 7, 8, 9, 16, 20, 29, 30, 31, 32, 33, 34, 38, 39, 40,
        ];
        for y_index in (31..=40).rev() {
            let mut xxx = [b'x'; 40];
            if y_index < xxx.len() {
                xxx[y_index] = b'y';
            }
            for &i in &nums {
                for &j in &nums {
                    if i >= j {
                        continue;
                    }
                    let got = match_len(&xxx[j..], &xxx[i..]);
                    let want = ref_match_len(&xxx[j..], &xxx[i..]);
                    assert!(
                        got <= want,
                        "y_index={y_index} i={i} j={j} got={got} want={want}"
                    );
                    assert_eq!(
                        got, want,
                        "y_index={y_index} i={i} j={j} got={got} want={want}"
                    );
                }
            }
        }
    }
}
