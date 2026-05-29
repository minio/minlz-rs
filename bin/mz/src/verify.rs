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

//! Verify subcommand: read + validate, no output.

use std::io::{self, Read};
use std::path::PathBuf;

use minlz::stream::Reader;

use crate::args::Options;
use crate::io_util::open_input;

pub fn run(opts: Options, input: Option<PathBuf>) -> io::Result<()> {
    let input = input.ok_or_else(|| io::Error::other("no input file given"))?;
    let (mut src, _) = open_input(&input)?;
    let is_block = opts.block || input.extension().and_then(|s| s.to_str()) == Some("mzb");
    if is_block {
        let mut buf = Vec::new();
        src.read_to_end(&mut buf)?;
        let mut dec = Vec::new();
        minlz::decode(&mut dec, &buf).map_err(io::Error::other)?;
        if !opts.quiet {
            eprintln!("{} ok ({} bytes)", input.display(), dec.len());
        }
        return Ok(());
    }
    let mut reader = Reader::new(&mut src);
    let mut sink = io::sink();
    let n = io::copy(&mut reader, &mut sink)?;
    if !opts.quiet {
        eprintln!("{} ok ({} bytes)", input.display(), n);
    }
    Ok(())
}
