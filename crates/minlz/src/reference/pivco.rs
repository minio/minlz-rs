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

use super::MAX_CODE_LEN;

// Node kinds of the coding tree (SPEC_PIVCO.md 2.3). Groups are PIVCO_BOTH and up.
const PIVCO_FULL: u8 = 0;
const PIVCO_LEAF_LEFT: u8 = 1;
const PIVCO_BOTH: u8 = 2;
const PIVCO_FLAT: u8 = 3;
const PIVCO_LEAF: u8 = 4;

#[derive(Default)]
struct PivcoNode {
    kind: u8,
    left: Option<Box<PivcoNode>>,
    right: Option<Box<PivcoNode>>,
    // Of a group: its first rank, and log2 of its number of values.
    first: usize,
    d: usize,
}

/// The coding tree of a Huffman table (SPEC_PIVCO.md 2).
pub(super) struct PivcoTable {
    root: Box<PivcoNode>,
    /// The value of each rank.
    ranks: Vec<u8>,
    /// The table codes one value (2.5).
    single: bool,
}

const ERR_TREE: &str = "table paths do not form a tree";

/// Builds the coding tree of the code lengths `lens` (SPEC_PIVCO.md 1.2, 2).
pub(super) fn new_pivco_table(lens: &[u8; 256]) -> Result<PivcoTable, String> {
    let mut coded = Vec::new();
    let mut kraft = 0;
    for (v, &l) in lens.iter().enumerate() {
        let l = l as usize;
        if l > MAX_CODE_LEN {
            return Err(format!("code length {l} of value {v} above {MAX_CODE_LEN}"));
        }
        if l > 0 {
            coded.push(v as u8);
            kraft += 1 << (MAX_CODE_LEN - l);
        }
    }
    match coded.len() {
        0 => return Err("table codes no value".into()),
        1 => {
            let v = coded[0];
            let root = Box::new(PivcoNode {
                kind: PIVCO_BOTH,
                d: 1,
                ..Default::default()
            });
            return Ok(PivcoTable {
                root,
                ranks: vec![v, v],
                single: true,
            });
        }
        _ if kraft != 1 << MAX_CODE_LEN => return Err("table is not a complete code".into()),
        _ => {}
    }

    struct Group {
        depth: usize,
        length: usize,
        values: Vec<u8>,
    }
    let mut groups = Vec::new();
    for l in 1..=MAX_CODE_LEN {
        let mut vals: Vec<u8> = coded
            .iter()
            .copied()
            .filter(|&v| lens[v as usize] as usize == l)
            .collect();
        for b in (0..=8).rev() {
            if (vals.len() >> b) & 1 == 1 {
                let rest = vals.split_off(1 << b);
                groups.push(Group {
                    depth: l - b,
                    length: l,
                    values: vals,
                });
                vals = rest;
            }
        }
    }
    groups.sort_by_key(|g| (g.depth, g.length));

    let mut root: Option<Box<PivcoNode>> = None;
    let mut ranks = Vec::new();
    let mut path = 0usize;
    for (i, g) in groups.iter().enumerate() {
        if i > 0 {
            path = (path + 1) << (g.depth - groups[i - 1].depth);
        }
        if path >= 1 << g.depth {
            return Err(ERR_TREE.into());
        }
        let mut p = &mut root;
        for bit in (0..g.depth).rev() {
            let nd = p.get_or_insert_with(Box::default);
            if nd.kind != PIVCO_FULL {
                return Err(ERR_TREE.into());
            }
            p = if (path >> bit) & 1 == 0 {
                &mut nd.left
            } else {
                &mut nd.right
            };
        }
        if p.is_some() {
            return Err(ERR_TREE.into());
        }
        let d = g.values.len().ilog2() as usize;
        let kind = match d {
            0 => PIVCO_LEAF,
            1 => PIVCO_BOTH,
            _ => PIVCO_FLAT,
        };
        *p = Some(Box::new(PivcoNode {
            kind,
            first: ranks.len(),
            d,
            ..Default::default()
        }));
        ranks.extend_from_slice(&g.values);
    }
    fn kinds(nd: &mut PivcoNode) -> Result<(), String> {
        if nd.kind != PIVCO_FULL {
            return Ok(());
        }
        // 2.3: the right child of an internal node is never a leaf.
        let (Some(left), Some(right)) = (&mut nd.left, &mut nd.right) else {
            return Err(ERR_TREE.into());
        };
        if right.kind == PIVCO_LEAF {
            return Err(ERR_TREE.into());
        }
        if left.kind == PIVCO_LEAF {
            nd.kind = PIVCO_LEAF_LEFT;
        }
        kinds(left)?;
        kinds(right)
    }
    let mut root = root.ok_or(ERR_TREE)?;
    kinds(&mut root)?;
    Ok(PivcoTable {
        root,
        ranks,
        single: false,
    })
}

impl PivcoTable {
    /// Decodes the pivco block `src` of `n` values (SPEC_PIVCO.md 5, 6).
    pub(super) fn decode(&self, src: &[u8], n: usize) -> Result<Vec<u8>, String> {
        if !(1..=65535).contains(&n) {
            return Err(format!("pivco block of {n} values"));
        }
        let mut r = PivcoReader { t: self, src };
        let v = r.node(&self.root, n)?;
        if !r.src.is_empty() {
            return Err(format!("{} bytes after the pivco block", r.src.len()));
        }
        Ok(v)
    }
}

/// Reads the regions of a pivco block in order.
struct PivcoReader<'a> {
    t: &'a PivcoTable,
    src: &'a [u8],
}

impl<'a> PivcoReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.src.len() {
            return Err(format!(
                "pivco region of {n} bytes runs past the end of the block"
            ));
        }
        let (b, rest) = self.src.split_at(n);
        self.src = rest;
        Ok(b)
    }

    /// Reads the bitmap of a node holding `k` values (SPEC_PIVCO.md 3.2).
    fn bitmap(&mut self, k: usize) -> Result<&'a [u8], String> {
        let b = self.take(k.div_ceil(8))?;
        if k & 7 != 0 && b[b.len() - 1] >> (k & 7) != 0 {
            return Err("pivco bitmap has an unused bit set".into());
        }
        Ok(b)
    }

    /// Decodes the subtree of `nd`, which holds `k` values.
    fn node(&mut self, nd: &PivcoNode, k: usize) -> Result<Vec<u8>, String> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let t = self.t;
        let ranks = &t.ranks;
        match nd.kind {
            PIVCO_BOTH => {
                let bm = self.bitmap(k)?;
                return (0..k)
                    .map(|j| {
                        let bit = ((bm[j >> 3] >> (j & 7)) & 1) as usize;
                        if t.single && bit != 0 {
                            return Err("pivco bitmap bit set with one coded value".to_string());
                        }
                        Ok(ranks[nd.first + bit])
                    })
                    .collect();
            }
            PIVCO_FLAT => {
                let b = self.take((k * nd.d).div_ceil(8))?;
                return Ok(flat_codes(b, k, nd.d)?
                    .into_iter()
                    .map(|x| ranks[nd.first + x])
                    .collect());
            }
            _ => {}
        }

        let rc = self.take(2)?;
        let kr = u16::from_le_bytes([rc[0], rc[1]]) as usize;
        if kr > k {
            return Err(format!("pivco right count {kr} above {k} values"));
        }
        let (Some(l), Some(r)) = (nd.left.as_deref(), nd.right.as_deref()) else {
            unreachable!("leaves are only left children");
        };
        let (left, right) = if nd.kind == PIVCO_LEAF_LEFT {
            (vec![ranks[l.first]; k - kr], self.node(r, kr)?)
        } else if k - kr >= kr {
            let left = self.node(l, k - kr)?;
            (left, self.node(r, kr)?)
        } else {
            let right = self.node(r, kr)?;
            (self.node(l, k - kr)?, right)
        };
        let bm = self.bitmap(k)?;
        let set: u32 = bm.iter().map(|b| b.count_ones()).sum();
        if set as usize != kr {
            return Err(format!("pivco bitmap has {set} bits set, right count {kr}"));
        }
        let (mut left, mut right) = (left.into_iter(), right.into_iter());
        Ok((0..k)
            .map(|j| {
                if (bm[j >> 3] >> (j & 7)) & 1 == 1 {
                    right.next()
                } else {
                    left.next()
                }
            })
            .map(|v| v.expect("bitmap counts checked"))
            .collect())
    }
}

/// Reads the `k` flat codes of `d` bits in `b` (SPEC_PIVCO.md 3.3).
fn flat_codes(mut b: &[u8], k: usize, d: usize) -> Result<Vec<usize>, String> {
    let (mut n512, mut n128) = (0, 0);
    if d < 8 {
        n512 = 512 * (k / 512);
        n128 = 128 * ((k - n512) / 128);
    }
    let mut c = Vec::with_capacity(k);
    while c.len() < n512 + n128 {
        let w = if c.len() < n512 { 64 } else { 16 };
        let mut blk = vec![0; 8 * w];
        for l in 0..w {
            let mut lane = 0u64;
            for i in 0..d {
                lane |= (b[i * w + l] as u64) << (8 * i);
            }
            for m in 0..8 {
                blk[m * w + l] = ((lane >> (m * d)) & ((1 << d) - 1)) as usize;
            }
        }
        c.extend_from_slice(&blk);
        b = &b[w * d..];
    }
    let rest = k - c.len();
    for i in 0..rest {
        let mut x = 0;
        for m in 0..d {
            let bit = i * d + m;
            x |= (((b[bit >> 3] >> (bit & 7)) & 1) as usize) << m;
        }
        c.push(x);
    }
    let end = rest * d;
    if end & 7 != 0 && b[end >> 3] >> (end & 7) != 0 {
        return Err("pivco flat codes have an unused bit set".into());
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(lens: &[(u8, u8)]) -> PivcoTable {
        let mut l = [0u8; 256];
        for &(v, n) in lens {
            l[v as usize] = n;
        }
        new_pivco_table(&l).unwrap()
    }

    /// Decodes the examples of SPEC_PIVCO.md appendix A, and changed copies of them.
    #[test]
    fn pivco_examples() {
        let a1 = table(&[
            (b'a', 3),
            (b'b', 3),
            (b'c', 3),
            (b'd', 3),
            (b'e', 2),
            (b'f', 3),
            (b'g', 3),
        ]);
        assert_eq!(a1.ranks, b"abcdefg", "A.1 ranks");
        let a5 = table(&(b'a'..=b'h').map(|v| (v, 3)).collect::<Vec<_>>());
        let mut a5vals: Vec<u8> = (b'a'..=b'h').flat_map(|v| [v; 16]).collect();
        a5vals.extend_from_slice(b"aa");
        let a5blk = [&[0x88; 16][..], &[0xc6; 16], &[0xfa; 16], &[0]].concat();

        let tests: [(&str, &PivcoTable, &[u8], &[u8]); 5] = [
            (
                "A.2",
                &a1,
                &[0x08, 0x00, 0x03, 0x00, 0x00, 0x85, 0x1b, 0x07, 0xc3, 0x39],
                b"fedcbafeedbeef",
            ),
            (
                "A.3 eee",
                &a1,
                &[0x03, 0x00, 0x00, 0x00, 0x00, 0x07],
                b"eee",
            ),
            ("A.3 abba", &a1, &[0x00, 0x00, 0x14, 0x00], b"abba"),
            ("A.4", &table(&[(b'x', 1)]), &[0, 0], b"xxxxxxxxxxx"),
            ("A.5", &a5, &a5blk, &a5vals),
        ];
        for (name, t, blk, vals) in tests {
            let got = t.decode(blk, vals.len());
            assert_eq!(got.as_deref(), Ok(vals), "{name}");
            // Every changed bit or size must fail or decode to other values (6).
            for i in 0..blk.len() {
                for b in 0..8 {
                    let mut c = blk.to_vec();
                    c[i] ^= 1 << b;
                    let got = t.decode(&c, vals.len());
                    assert_ne!(
                        got.as_deref(),
                        Ok(vals),
                        "{name}: byte {i} bit {b} changed decodes the same"
                    );
                }
            }
            assert!(
                t.decode(&blk[..blk.len() - 1], vals.len()).is_err(),
                "{name}: truncated block decodes"
            );
            assert!(
                t.decode(&[blk, &[0]].concat(), vals.len()).is_err(),
                "{name}: extended block decodes"
            );
        }
    }
}
