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

//! Port of Go's `golden_test.go` (without TestJointGolden: the Shaper is not ported).
//!
//! The goldens in testdata were produced by the upstream C library at the commit in the
//! module doc: `tables.txt.gz` holds its code lengths and schedules, `blocks.txt.gz` its PH
//! blocks with the FlatVertical layout, converted to this module's form by removing the
//! symbol count, the marker prefix byte and the marker byte of each flat region. Digests
//! are FNV-1a 64.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};

use crate::pivco::testutil::*;
use crate::pivco::{Table, build_lengths};

/// One manifest line: its kind (first token) and key=value fields.
struct Record {
    kind: String,
    line: usize,
    f: HashMap<String, String>,
}

/// Parses a gzipped golden manifest, skipping '#' lines.
fn read_manifest(gz: &[u8]) -> Vec<Record> {
    let rd = BufReader::new(flate2::read::GzDecoder::new(gz));
    let mut recs = Vec::new();
    for (i, s) in rd.lines().enumerate() {
        let s = s.expect("manifest");
        let s = s.trim_end_matches('\r');
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let mut tok = s.split_whitespace();
        let kind = tok.next().unwrap().to_string();
        let f = tok
            .map(|x| {
                let (k, v) = x.split_once('=').unwrap_or((x, ""));
                (k.to_string(), v.to_string())
            })
            .collect();
        recs.push(Record {
            kind,
            line: i + 1,
            f,
        });
    }
    recs
}

/// Parses an unsigned integer with Go's base-0 prefix rules for hex.
fn parse_u64(v: &str, base: u32) -> Option<u64> {
    match (base, v.strip_prefix("0x").or_else(|| v.strip_prefix("0X"))) {
        (0, Some(h)) => u64::from_str_radix(h, 16).ok(),
        (0, None) => v.parse().ok(),
        _ => u64::from_str_radix(v, base).ok(),
    }
}

impl Record {
    fn get(&self, key: &str) -> &str {
        self.f.get(key).map_or("", String::as_str)
    }

    fn uint(&self, key: &str, base: u32) -> u64 {
        parse_u64(self.get(key), base)
            .unwrap_or_else(|| panic!("line {}: {key}={:?}", self.line, self.get(key)))
    }

    /// Returns a len:digest field.
    fn len_digest(&self, key: &str) -> (usize, u64) {
        let (l, d) = self
            .get(key)
            .split_once(':')
            .unwrap_or_else(|| panic!("line {}: {key}", self.line));
        (l.parse().unwrap(), u64::from_str_radix(d, 16).unwrap())
    }

    /// Returns a lengths field: one hex nibble per symbol 0..255.
    fn lens(&self, key: &str) -> [u8; 256] {
        let v = self.get(key).as_bytes();
        assert_eq!(v.len(), 256, "line {}: length nibbles", self.line);
        core::array::from_fn(|i| (v[i] as char).to_digit(16).expect("hex nibble") as u8)
    }

    /// Returns the histogram of a gen=dist:<name> or gen=rhist:<seed> table.
    fn hist(&self) -> [u64; 256] {
        let (k, v) = self.get("gen").split_once(':').unwrap_or(("", ""));
        let h = match k {
            "dist" => dist(v),
            "rhist" => parse_u64(v, 0).map(rhist),
            _ => None,
        };
        h.unwrap_or_else(|| panic!("line {}: bad histogram {:?}", self.line, self.get("gen")))
    }
}

/// Returns the tables of tables.txt.gz by name.
fn golden_tables() -> HashMap<String, Record> {
    read_manifest(include_bytes!("../testdata/tables.txt.gz"))
        .into_iter()
        .map(|r| {
            assert_eq!(r.kind, "T", "line {}", r.line);
            (r.get("name").to_string(), r)
        })
        .collect()
}

/// Checks build_lengths and the schedules against the C library.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn table_golden() {
    let tabs = golden_tables();
    assert!(!tabs.is_empty(), "no tables");
    for (name, r) in &tabs {
        let want = r.lens("lens");
        if !r.get("gen").is_empty() {
            let h = r.hist();
            let raw: Vec<u8> = h.iter().flat_map(|f| f.to_le_bytes()).collect();
            assert_eq!(
                fnv1a64(&raw),
                r.uint("hist", 16),
                "{name}: histogram generator differs from C"
            );
            let mut got = [0u8; 256];
            build_lengths(&h, &mut got);
            assert!(
                got == want,
                "{name}: lengths\n got {got:x?}\nwant {want:x?}"
            );
        }
        let tb = must_table(&want);
        let sched: Vec<u8> = tb
            .recs()
            .iter()
            .flat_map(|x| [x.op, x.param, x.right])
            .collect();
        for (key, b) in [
            ("sched", &sched[..]),
            ("r2s", &tb.r2s[..usize::from(tb.nranks)]),
            ("s2r", &tb.s2r[..]),
        ] {
            assert_eq!(fnv1a64(b), r.uint(key, 16), "{name}: {key} digest mismatch");
        }
        assert!(
            r.uint("nsched", 10) == u64::from(tb.nsched)
                && r.uint("nranks", 10) == u64::from(tb.nranks),
            "{name}: nsched {}, nranks {}",
            tb.nsched,
            tb.nranks
        );
    }
}

/// Regenerates the input of a block record: "s:<seed>" samples the table's histogram,
/// "l:<seed>" its lengths, "hex:<bytes>" is the input.
fn golden_input(tabs: &HashMap<String, Record>, r: &Record) -> (Vec<u8>, [u8; 256]) {
    let tr = tabs
        .get(r.get("table"))
        .unwrap_or_else(|| panic!("line {}: unknown table {:?}", r.line, r.get("table")));
    let lens = tr.lens("lens");
    let n = r.uint("n", 10) as usize;
    let (k, v) = r.get("gen").split_once(':').unwrap_or(("", ""));
    let input = match k {
        "s" => sample(&tr.hist(), n, parse_u64(v, 0).unwrap()),
        "l" => sample_lens(&lens, n, parse_u64(v, 0).unwrap()),
        "hex" => hex(v),
        _ => panic!("line {}: bad generator {:?}", r.line, r.get("gen")),
    };
    assert!(
        input.len() == n && fnv1a64(&input) == r.uint("in", 16),
        "line {}: input generator differs from C",
        r.line
    );
    (input, lens)
}

fn hex(v: &str) -> Vec<u8> {
    (0..v.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&v[i..i + 2], 16).unwrap())
        .collect()
}

/// Encodes every block of blocks.txt.gz on every tier, requires C's regions byte for byte
/// and block_size to predict their size, then decodes them.
#[test]
#[cfg_attr(miri, ignore)] // slow under Miri
fn block_golden() {
    let tabs = golden_tables();
    let recs = read_manifest(include_bytes!("../testdata/blocks.txt.gz"));
    assert!(!recs.is_empty(), "no blocks");
    for c in &mut coders() {
        for r in &recs {
            let case = r.get("case");
            let (input, lens) = golden_input(&tabs, r);
            let mut tb = Table::default();
            tb.set_lengths(&lens).unwrap();
            let (n, digest) = r.len_digest("out");
            let mut h = [0u64; 256];
            c.enc.histogram(&mut h, &input);
            assert_eq!(tb.block_size(&h), Some(n), "{case}: block_size");
            let mut out = vec![0; n];
            c.enc.encode(&mut out, &input, &tb).unwrap();
            assert_eq!(
                fnv1a64(&out),
                digest,
                "{case} (table {} n {}): differs from C",
                r.get("table"),
                input.len()
            );
            if let Some(x) = r.f.get("hex") {
                assert_eq!(out, hex(x), "{case}");
            }
            let mut got = vec![0; input.len()];
            assert!(
                c.dec.decode(&mut got, &out, &tb).is_ok() && got == input,
                "{case}: decode"
            );
        }
    }
}
