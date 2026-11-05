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

use std::sync::LazyLock;

use super::pivco::{PivcoTable, new_pivco_table};
use super::{MAX_BLOCK_SIZE, MAX_CODE_LEN, MAX_VARINT_LEN64, uvarint};

// Streams of a chunk, in order, then the literals.
const STREAM_TOKENS: usize = 0;
const STREAM_LIT_LENS: usize = 1;
const STREAM_OFFSETS: usize = 2;
const STREAM_ESCAPES: usize = 3;
const STREAM_OFFSET_BITS: usize = 4;
const STREAM_LITERALS: usize = 5;

// Entry modes.
const MODE_STORED: usize = 0;
const MODE_CURRENT: usize = 1;
const MODE_CODE_LENGTHS: usize = 2;
// 3 is class lengths, the default of read_table.
const MODE_DELTA_LENGTHS: usize = 4;

/// The most chunks of a block.
const MAX_CHUNKS: usize = 128;

/// The values of each class, in ascending order.
struct ClassLayout(Vec<Vec<usize>>);

struct SimdStream {
    name: &'static str,
    lo: usize,
    hi: usize,
    classes: ClassLayout,
}

// For tokens, the classes are the match length code groups.
static SIMD_STREAMS: LazyLock<[SimdStream; STREAM_LITERALS + 1]> = LazyLock::new(|| {
    let s = |name, lo, hi, classes| SimdStream {
        name,
        lo,
        hi,
        classes,
    };
    [
        s(
            "tokens",
            0,
            135,
            last_values(0, &[0, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 31, 32, 33]),
        ),
        s(
            "literal lengths",
            1,
            33,
            last_values(1, &[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 33]),
        ),
        s("offset symbols", 1, 175, offset_classes()),
        s(
            "escapes",
            0,
            255,
            last_values(
                0,
                &[
                    0, 1, 2, 4, 6, 10, 14, 22, 30, 46, 62, 94, 126, 190, 254, 255,
                ],
            ),
        ),
        s("offset bits", 0, 255, ClassLayout(Vec::new())),
        s("literals", 0, 255, classes_of(0, 255, literal_class)),
    ]
});

static DELTA_CLASSES: LazyLock<ClassLayout> = LazyLock::new(|| {
    classes_of(0, 255, |b| {
        (usize::BITS - b.min(256 - b).leading_zeros()) as usize
    })
});

/// Returns the classes of the values from `first` on, given the last value of each class.
fn last_values(mut first: usize, last: &[usize]) -> ClassLayout {
    let mut l = Vec::new();
    for &end in last {
        let mut c = Vec::new();
        while first <= end {
            c.push(first);
            first += 1;
        }
        l.push(c);
    }
    ClassLayout(l)
}

/// Returns the classes of the values `lo` to `hi`, given the class of each value.
fn classes_of(lo: usize, hi: usize, class: impl Fn(usize) -> usize) -> ClassLayout {
    let mut l: Vec<Vec<usize>> = Vec::new();
    for v in lo..=hi {
        let c = class(v);
        if l.len() <= c {
            l.resize(c + 1, Vec::new());
        }
        l[c].push(v);
    }
    ClassLayout(l)
}

fn offset_classes() -> ClassLayout {
    let mut last = vec![1, 3, 7, 15];
    last.extend((4..=23).map(|c| 8 * c - 9));
    last_values(1, &last)
}

fn literal_class(b: usize) -> usize {
    let b = b as u8;
    if b == 0x00 {
        0
    } else if matches!(b, 0x09 | 0x0a | 0x0d) {
        1
    } else if b < 0x20 || b == 0x7f {
        2
    } else if b == b' ' {
        3
    } else if b.is_ascii_digit() {
        4
    } else if b"aeinorst".contains(&b) {
        5
    } else if b.is_ascii_lowercase() {
        6
    } else if b.is_ascii_uppercase() {
        7
    } else if b"\",-./:_".contains(&b) {
        8
    } else if b <= 0x7e {
        9
    } else if b <= 0xbf {
        10
    } else if (0xc2..=0xf4).contains(&b) {
        11
    } else {
        12
    }
}

/// Returns the code lengths inside a class of `n` values.
fn in_class_lengths(n: usize) -> Vec<usize> {
    let k = n.ilog2() as usize;
    (0..n)
        .map(|i| k + usize::from(i >= (1 << (k + 1)) - n))
        .collect()
}

impl ClassLayout {
    /// Returns the code length of each value, given the length of each class.
    fn lengths(&self, cl: &[usize]) -> [usize; 256] {
        let mut lens = [0; 256];
        for (c, vals) in self.0.iter().enumerate() {
            if cl[c] == 0 {
                continue;
            }
            for (i, inl) in in_class_lengths(vals.len()).into_iter().enumerate() {
                lens[vals[i]] = cl[c] + inl;
            }
        }
        lens
    }
}

/// Reads the reverse varint at the end of `b`, and returns it and the bytes before it.
fn rvarint(b: &[u8]) -> Result<(u64, &[u8]), String> {
    let rev: Vec<u8> = b.iter().rev().take(MAX_VARINT_LEN64).copied().collect();
    let (v, n) = uvarint(&rev).ok_or("invalid reverse varint")?;
    Ok((v, &b[..b.len() - n]))
}

/// Reads `n` lengths of 4 bits from `b`, and returns them and their size in bytes.
fn nibbles(b: &[u8], n: usize) -> Result<(Vec<usize>, usize), String> {
    let size = n.div_ceil(2);
    if b.len() < size {
        return Err("table runs past its records".into());
    }
    let l: Vec<usize> = b[..size]
        .iter()
        .flat_map(|&c| [(c & 15) as usize, (c >> 4) as usize])
        .collect();
    if l.len() > n && l[n] != 0 {
        return Err("table pad nibble is not 0".into());
    }
    Ok((l[..n].to_vec(), size))
}

/// Reports whether the lengths `l` form a complete code.
fn complete(l: &[usize]) -> bool {
    l.iter()
        .filter(|&&v| v != 0)
        .map(|&v| 1 << (15 - v))
        .sum::<usize>()
        == 1 << 15
}

/// Reads a table of stream `k` from `tab`, and returns it and its size.
fn read_table(k: usize, mode: usize, tab: &[u8]) -> Result<(PivcoTable, usize), String> {
    let s = &SIMD_STREAMS[k];
    let named = |e: String| format!("{}: {e}", s.name);
    let (lens, n) = if mode == MODE_CODE_LENGTHS {
        let (l, n) = nibbles(tab, s.hi + 1).map_err(named)?;
        let mut lens = [0; 256];
        lens[..l.len()].copy_from_slice(&l);
        (lens, n)
    } else if k == STREAM_TOKENS {
        let (l, n) = nibbles(tab, 22).map_err(named)?;
        if !complete(&l[..14]) || !complete(&l[14..18]) || !complete(&l[18..]) {
            return Err(named(
                "token group lengths or flag codes are not complete codes".into(),
            ));
        }
        let ml = s.classes.lengths(&l[..14]);
        let mut lens = [0; 256];
        for t in 0..136 {
            let f = if t >> 2 >= 32 {
                l[18 + (t & 3)]
            } else {
                l[14 + (t & 3)]
            };
            if ml[t >> 2] != 0 && f != 0 {
                lens[t] = ml[t >> 2] + f;
            }
        }
        (lens, n)
    } else {
        let classes = if mode == MODE_DELTA_LENGTHS {
            &*DELTA_CLASSES
        } else {
            &s.classes
        };
        let (l, n) = nibbles(tab, classes.0.len()).map_err(named)?;
        (classes.lengths(&l), n)
    };
    let (mut used, mut kraft) = (0, 0);
    let mut lens8 = [0u8; 256];
    for (v, &ln) in lens.iter().enumerate() {
        if ln == 0 {
            continue;
        }
        if ln > MAX_CODE_LEN {
            return Err(format!(
                "{}: code length {ln} of value {v} above {MAX_CODE_LEN}",
                s.name
            ));
        }
        if v < s.lo || v > s.hi {
            return Err(format!(
                "{}: code for value {v} outside the stream's range",
                s.name
            ));
        }
        used += 1;
        kraft += 1 << (MAX_CODE_LEN - ln);
        lens8[v] = ln as u8;
    }
    if kraft != 1 << MAX_CODE_LEN && (used != 1 || kraft != 1 << (MAX_CODE_LEN - 1)) {
        return Err(format!("{}: table is not a complete code", s.name));
    }
    Ok((new_pivco_table(&lens8).map_err(named)?, n))
}

/// Reads entries from records, and their data from the data before the records.
struct EntryReader<'a> {
    recs: &'a [u8],
    data: &'a [u8],
}

struct SimdDecoder<'w> {
    size: usize,
    tables: [Option<PivcoTable>; STREAM_LITERALS + 1],
    lits: Vec<u8>,
    /// Literals used.
    l: usize,
    out: Vec<u8>,
    repeat: usize,
    ops: usize,
    strict: bool,
    w: &'w mut dyn SimdWalker,
}

/// Receives the contents of a SIMD block from [`walk_simd_block`]. The methods do nothing by default.
pub trait SimdWalker {
    /// Gets each stream entry in block order: the stream (0 tokens, 1 literal lengths, 2 offset
    /// symbols, 3 escapes, 4 offset bits, 5 literals), its mode, its number of values, and the bytes
    /// of its data and new table.
    fn entry(&mut self, _stream: usize, _mode: usize, _values: usize, _data: usize, _table: usize) {
    }
    /// Gets the flags and number of operations of each chunk, after its entries.
    fn chunk(&mut self, _flags: u8, _ops: usize) {}
    /// Gets each operation: its number of literals, match length and offset, and whether the
    /// offset is repeated.
    fn op(&mut self, _lits: usize, _ml: usize, _offset: usize, _repeat: bool) {}
}

struct NoWalker;

impl SimdWalker for NoWalker {}

/// A reference decoder of SIMD blocks, as specified in SPEC_SIMD.md. `src` is the block, including
/// its indicator byte. It is not optimized for speed, but for readability.
pub fn decode_simd_block(src: &[u8]) -> Result<Vec<u8>, String> {
    decode(src, false, &mut NoWalker)
}

/// [`decode_simd_block`], but also rejects matches in chunks with the no-overlap flag that rely on
/// their offset being raised, which SPEC_SIMD.md allows. It checks encoders.
pub fn decode_simd_block_strict(src: &[u8]) -> Result<Vec<u8>, String> {
    decode(src, true, &mut NoWalker)
}

/// [`decode_simd_block`], and reports the contents of the block to `w`.
pub fn walk_simd_block(src: &[u8], w: &mut dyn SimdWalker) -> Result<Vec<u8>, String> {
    decode(src, false, w)
}

fn decode(src: &[u8], strict: bool, w: &mut dyn SimdWalker) -> Result<Vec<u8>, String> {
    let Some((0, src)) = src.split_first() else {
        return Err("indicator byte is not 0".into());
    };
    let (v, n) = uvarint(src).ok_or("invalid header varint")?;
    if v >= 1 << 32 {
        return Err("header varint above 32 bits".into());
    }
    if v >> 24 != 1 {
        return Err(format!("block type {} is not a SIMD block", v >> 24));
    }
    let size = (v & ((1 << 24) - 1)) as usize;
    if size > MAX_BLOCK_SIZE {
        return Err(format!("decoded size {size} above 8 MiB"));
    }
    if src.len() + 64 > size {
        return Err(format!(
            "block of {} bytes is not 64 bytes smaller than its decoded size {size}",
            src.len()
        ));
    }
    if n == src.len() {
        return Err("no flags byte".into());
    }
    let flags = src[n];
    if flags & 0xe0 != 0 || flags & 7 > 5 {
        return Err(format!("invalid block flags {flags:#x}"));
    }
    let (lz_size, body) = rvarint(&src[n + 1..]).map_err(|e| format!("LZ section size: {e}"))?;
    if lz_size > body.len() as u64 {
        return Err(format!("LZ section size {lz_size} larger than the block"));
    }
    let lz = lz_size as usize;
    let mut d = SimdDecoder {
        size,
        tables: Default::default(),
        lits: Vec::new(),
        l: 0,
        out: Vec::with_capacity(size),
        repeat: 1,
        ops: 0,
        strict,
        w,
    };
    d.literals(&body[lz..])?;
    if lz > 0 {
        d.chunks(&body[..lz], 1024 << (flags & 7))?;
    }
    d.out.extend_from_slice(&d.lits[d.l..]);
    if d.out.len() != size {
        return Err(format!("decoded {} bytes, want {size}", d.out.len()));
    }
    Ok(d.out)
}

/// Splits a section into its records, after their reverse varint size, and the data before them.
fn entries<'a>(sec: &'a [u8], what: &str) -> Result<EntryReader<'a>, String> {
    let (t, sec) = rvarint(sec).map_err(|e| format!("{what} records size: {e}"))?;
    if t > sec.len() as u64 {
        return Err(format!("{what} records size {t} larger than its section"));
    }
    let (data, recs) = sec.split_at(sec.len() - t as usize);
    Ok(EntryReader { recs, data })
}

/// Pops the first value of `s`, the stream `name`, at operation `ops`.
fn next(s: &mut &[u8], name: &str, ops: usize) -> Result<usize, String> {
    let (&v, rest) = s
        .split_first()
        .ok_or_else(|| format!("operation {ops}: {name} missing"))?;
    *s = rest;
    Ok(v as usize)
}

/// Reads a length code and its escape.
fn length(code: usize, esc: &mut &[u8], ops: usize) -> Result<usize, String> {
    if code < 33 {
        return Ok(code);
    }
    Ok(code + next(esc, "escape", ops)?)
}

impl SimdDecoder<'_> {
    /// Decodes the literal section.
    fn literals(&mut self, sec: &[u8]) -> Result<(), String> {
        let mut r = entries(sec, "literal")?;
        while !r.recs.is_empty() {
            let v = self.stream(&mut r, STREAM_LITERALS, -1)?;
            if v.is_empty() || v.len() > 65535 {
                return Err(format!("literal block of {} literals", v.len()));
            }
            if self.lits.len() + v.len() > self.size {
                return Err("more literals than the decoded size".into());
            }
            self.lits.extend_from_slice(&v);
        }
        if self.lits.is_empty() {
            return Err("no literal blocks".into());
        }
        if !r.data.is_empty() {
            return Err(format!(
                "{} bytes of literal data without an entry",
                r.data.len()
            ));
        }
        Ok(())
    }

    /// Runs the operations of the LZ section.
    fn chunks(&mut self, sec: &[u8], limit: usize) -> Result<(), String> {
        let mut r = entries(sec, "chunk")?;
        let mut i = 0;
        while let Some((&flags, recs)) = r.recs.split_first() {
            r.recs = recs;
            if flags & 0xf0 != 0 {
                return Err(format!("invalid chunk flags {flags:#x}"));
            }
            if i == MAX_CHUNKS {
                return Err(format!("more than {MAX_CHUNKS} chunks"));
            }
            // The tokens give the number of values of the next streams, and the literal lengths
            // add escapes (SPEC_SIMD.md 4.1). -1 is a number stored with the values.
            let mut c: [Vec<u8>; STREAM_OFFSET_BITS + 1] = Default::default();
            let mut counts: [isize; STREAM_OFFSET_BITS + 1] = [-1, 0, 0, 0, -1];
            for (k, ck) in c.iter_mut().enumerate() {
                *ck = self.stream(&mut r, k, counts[k])?;
                for &v in ck.iter() {
                    match k {
                        STREAM_TOKENS => {
                            counts[STREAM_LIT_LENS] += ((v >> 1) & 1) as isize;
                            counts[STREAM_OFFSETS] += ((v & 1) ^ 1) as isize;
                            if v >> 2 == 33 {
                                counts[STREAM_ESCAPES] += 1;
                            }
                        }
                        STREAM_LIT_LENS if v == 33 => counts[STREAM_ESCAPES] += 1,
                        _ => {}
                    }
                }
            }
            self.chunk(&c, flags, limit)?;
            i += 1;
        }
        if !r.data.is_empty() {
            return Err(format!(
                "{} bytes of chunk data without an entry",
                r.data.len()
            ));
        }
        Ok(())
    }

    /// Reads the next entry of `r` for stream `k`, and returns the stream's values.
    /// A pivco block holds `count` values, or stores their number if `count` is -1.
    fn stream(&mut self, r: &mut EntryReader, k: usize, count: isize) -> Result<Vec<u8>, String> {
        let s = &SIMD_STREAMS[k];
        let (e, n) = uvarint(r.recs).ok_or_else(|| format!("{}: invalid entry", s.name))?;
        r.recs = &r.recs[n..];
        let (size, mode) = (e >> 3, (e & 7) as usize);
        if size > r.data.len() as u64 {
            return Err(format!("{}: stored size {size} past the data", s.name));
        }
        let (mut vals, data) = r.data.split_at(size as usize);
        r.data = data;
        let mut tab = 0;
        match mode {
            MODE_STORED => {}
            _ if mode > MODE_DELTA_LENGTHS => {
                return Err(format!("{}: reserved mode {mode}", s.name));
            }
            _ if k == STREAM_OFFSET_BITS => return Err(format!("offset bits in mode {mode}")),
            MODE_DELTA_LENGTHS if k != STREAM_LITERALS => {
                return Err(format!("{}: delta lengths outside the literals", s.name));
            }
            MODE_CURRENT => {
                if self.tables[k].is_none() {
                    return Err(format!("{}: no current table", s.name));
                }
            }
            _ => {
                let (t, tn) = read_table(k, mode, r.recs)?;
                r.recs = &r.recs[tn..];
                self.tables[k] = Some(t);
                tab = tn;
            }
        }
        let vals = if mode == MODE_STORED {
            vals.to_vec()
        } else {
            let mut count = count;
            if count < 0 {
                if vals.len() < 2 {
                    return Err(format!("{}: number of values past the data", s.name));
                }
                count = u16::from_le_bytes([vals[0], vals[1]]) as isize;
                vals = &vals[2..];
            }
            let t = self.tables[k]
                .as_ref()
                .expect("table read or checked above");
            t.decode(vals, count as usize)
                .map_err(|e| format!("{}: {e}", s.name))?
        };
        if let Some(v) = vals
            .iter()
            .find(|&&v| (v as usize) < s.lo || v as usize > s.hi)
        {
            return Err(format!("{}: value {v} outside the stream's range", s.name));
        }
        self.w.entry(k, mode, vals.len(), size as usize, tab);
        Ok(vals)
    }

    /// Runs the operations of a chunk, given its streams.
    fn chunk(
        &mut self,
        c: &[Vec<u8>; STREAM_OFFSET_BITS + 1],
        flags: u8,
        limit: usize,
    ) -> Result<(), String> {
        let delta = flags & 1 != 0;
        let (tok, raw) = (&c[STREAM_TOKENS], &c[STREAM_OFFSET_BITS]);
        let (mut ll, mut offc, mut esc) = (
            &c[STREAM_LIT_LENS][..],
            &c[STREAM_OFFSETS][..],
            &c[STREAM_ESCAPES][..],
        );
        if tok.is_empty() || tok.len() > limit {
            return Err(format!("chunk of {} operations, limit {limit}", tok.len()));
        }
        self.w.chunk(flags, tok.len());
        let (mut nbits, start) = (0, self.out.len());
        for &t in tok {
            if self.ops == 0 && t & 2 == 0 {
                return Err("first operation without literals".into());
            }
            let mut lits = 0;
            if t & 2 != 0 {
                let n = next(&mut ll, "literal length", self.ops)?;
                let n = length(n, &mut esc, self.ops)?;
                if self.l + n > self.lits.len() {
                    return Err(format!(
                        "operation {}: reads more literals than the literal section holds",
                        self.ops
                    ));
                }
                if self.out.len() + n > self.size {
                    return Err(format!(
                        "operation {}: output past the decoded size",
                        self.ops
                    ));
                }
                let p = self.out.len();
                let mut r = self.repeat.min(p);
                if flags & 2 != 0 {
                    // The no-overlap flag raises delta references to 32 first.
                    r = self.repeat.max(32).min(p);
                }
                for j in 0..n {
                    let mut b = self.lits[self.l + j];
                    if delta && j < 32 && r > 0 {
                        b = b.wrapping_add(self.out[p - r + j % r]);
                    }
                    self.out.push(b);
                }
                self.l += n;
                lits = n;
            }
            if t & 1 == 0 {
                let sym = next(&mut offc, "offset symbol", self.ops)?;
                let (mut off, mut nb) = (sym, 0);
                if sym >= 16 {
                    let (b, m) = ((sym + 16) >> 3, (sym + 16) & 7);
                    (off, nb) = ((8 + m) << (b - 3), b - 3);
                }
                for i in 0..nb {
                    if nbits >= 8 * raw.len() {
                        return Err(format!("operation {}: offset bits missing", self.ops));
                    }
                    off += (((raw[nbits >> 3] >> (nbits & 7)) & 1) as usize) << i;
                    nbits += 1;
                }
                self.repeat = off;
            }
            let ml = length((t >> 2) as usize, &mut esc, self.ops)?;
            let mut r = self.repeat.min(self.out.len());
            if flags & 2 != 0 {
                if self.strict && r < ml.min(32) {
                    return Err(format!(
                        "operation {}: match of {ml} bytes at offset {r} in a chunk without overlaps",
                        self.ops
                    ));
                }
                r = self.repeat.max(ml.min(32)).min(self.out.len());
            }
            if ml > 32 && r < 32 {
                return Err(format!(
                    "operation {}: match of {ml} bytes at offset {r}",
                    self.ops
                ));
            }
            if self.out.len() + ml > self.size {
                return Err(format!(
                    "operation {}: output past the decoded size",
                    self.ops
                ));
            }
            for _ in 0..ml {
                self.out.push(self.out[self.out.len() - r]);
            }
            self.w.op(lits, ml, r, t & 1 != 0);
            self.ops += 1;
        }
        if self.out.len() - start < tok.len() {
            return Err(format!(
                "chunk outputs {} bytes with {} operations",
                self.out.len() - start,
                tok.len()
            ));
        }
        if !ll.is_empty() || !offc.is_empty() || !esc.is_empty() {
            return Err(format!(
                "chunk with {} literal lengths, {} offset symbols and {} escapes unused",
                ll.len(),
                offc.len(),
                esc.len()
            ));
        }
        if raw.len() != nbits.div_ceil(8) {
            return Err(format!(
                "chunk with {} bytes of offset bits, reads {nbits} bits",
                raw.len()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compares the class layouts with the arrays of SPEC_SIMD.md Appendix A: the copy in testdata,
    /// and the Go repository's spec when it is checked out next to this one.
    #[test]
    fn simd_layouts_spec() {
        check_layouts(include_str!("testdata/SPEC_SIMD_appendix_a.md"));
        let go = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../minlz/SPEC_SIMD.md");
        // Miri isolates the file system.
        if !cfg!(miri)
            && let Ok(spec) = std::fs::read_to_string(go)
        {
            check_layouts(&spec);
        }
    }

    /// Parses an array header, "Class, values 1-33:" or "In-class length, values 1-33:".
    fn array_header(line: &str) -> Option<(&str, usize, usize)> {
        let (what, rest) = line.split_once(", ")?;
        if what != "Class" && what != "In-class length" {
            return None;
        }
        let (lo, hi) = rest
            .strip_suffix(':')?
            .rsplit(' ')
            .next()?
            .split_once('-')?;
        Some((what, lo.parse().ok()?, hi.parse().ok()?))
    }

    fn check_layouts(spec: &str) {
        let streams = &*SIMD_STREAMS;
        let layouts: [(&str, &ClassLayout); 6] = [
            ("1", &streams[STREAM_LIT_LENS].classes),
            ("2", &streams[STREAM_ESCAPES].classes),
            ("3", &streams[STREAM_OFFSETS].classes),
            ("4", &streams[STREAM_TOKENS].classes),
            ("5", &streams[STREAM_LITERALS].classes),
            ("6", &DELTA_CLASSES),
        ];
        let spec = spec.replace('\r', "");
        let mut found = 0;
        for sec in spec.split("\n## A.").skip(1) {
            let Some((a, l)) = layouts.iter().find(|(a, _)| sec.starts_with(a)) else {
                continue;
            };
            let (mut class, mut inl) = ([0; 256], [0; 256]);
            for (c, vals) in l.0.iter().enumerate() {
                for (i, n) in in_class_lengths(vals.len()).into_iter().enumerate() {
                    (class[vals[i]], inl[vals[i]]) = (c, n);
                }
            }
            let lines: Vec<&str> = sec.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                let Some((what, lo, hi)) = array_header(line) else {
                    continue;
                };
                let want = if what == "Class" { &class } else { &inl };
                let mut v = lo;
                let numbers = lines[i + 1..]
                    .iter()
                    .take_while(|l| {
                        l.bytes()
                            .all(|c| c.is_ascii_digit() || c == b',' || c == b' ')
                    })
                    .flat_map(|l| l.split([',', ' ']))
                    .filter(|f| !f.is_empty());
                for f in numbers {
                    let n: usize = f.parse().unwrap();
                    assert!(v <= hi, "A.{a} {what}: more than {} values", hi + 1 - lo);
                    assert_eq!(n, want[v], "A.{a} {what} of {v}: spec {n}, got {}", want[v]);
                    v += 1;
                }
                assert_eq!(v, hi + 1, "A.{a} {what}: {v} values");
                found += 1;
            }
        }
        assert_eq!(found, 2 * layouts.len(), "found {found} arrays");
    }

    fn append_uvarint(b: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            b.push(v as u8 | 0x80);
            v >>= 7;
        }
        b.push(v as u8);
    }

    /// Decodes the example of SPEC_SIMD.md 4.5.
    #[test]
    fn simd_delta_example() {
        let rv = |b: &mut Vec<u8>, v: usize| {
            let mut r = Vec::new();
            append_uvarint(&mut r, v as u64);
            r.reverse();
            b.extend_from_slice(&r);
        };
        // Literals 41 42 43 07 00 41 42 43 with offset 5, then 07 01 41 42 44 07 and a repeat of 4,
        // then 3 repeats of 32 to pass the 64-byte rule.
        let lits = [
            0x41, 0x42, 0x43, 0x07, 0x00, 0x41, 0x42, 0x43, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00,
        ];
        let streams: [&[u8]; 5] = [
            &[2, (4 << 2) | 3, (32 << 2) | 1, (32 << 2) | 1, (32 << 2) | 1],
            &[8, 6],
            &[5],
            &[],
            &[],
        ];
        let mut want = vec![
            0x41, 0x42, 0x43, 0x07, 0x00, 0x41, 0x42, 0x43, 0x07, 0x01, 0x41, 0x42, 0x44, 0x07,
        ];
        while want.len() < 14 + 4 + 96 {
            want.push(want[want.len() - 5]);
        }
        let mut b = vec![0];
        append_uvarint(&mut b, (1 << 24) | want.len() as u64);
        b.push(0);
        let start = b.len();
        let mut tab = vec![1u8];
        for s in streams {
            b.extend_from_slice(s);
            tab.push((s.len() << 3) as u8);
        }
        b.extend_from_slice(&tab);
        rv(&mut b, tab.len());
        let lz = b.len() - start;
        b.extend_from_slice(&lits);
        b.push((lits.len() << 3) as u8);
        rv(&mut b, 1);
        rv(&mut b, lz);
        assert_eq!(decode_simd_block(&b), Ok(want));
    }
}
