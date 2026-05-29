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

//! `mz` — MinLZ command-line tool.
//!
//! Compresses and decompresses MinLZ streams (`.mz`) and single
//! blocks (`.mzb`); also supports verify, tail, seek by offset, and a
//! throughput benchmark mode.  Run `mz --help` for the full flag list.

mod args;
mod bench;
mod compress;
mod decompress;
mod io_util;
mod verify;

use std::process::ExitCode;

use args::{Cli, Subcommand};

const USAGE: &str = "\
MinLZ compression tool (Rust port).

Usage:
  mz c [options] <input>       Compress one file (or - for stdin).
  mz d [options] <input>       Decompress one file (or - for stdin).
  mz cat [options] <input>     Alias for `d -c` (decompress to stdout).
  mz tail [options] <input>    Alias for `d -c --tail=1K+nl`; with --follow,
                               behaves like `tail -f`.
  mz verify <input>            Read and validate; no output.
  mz bench [options] <input>   Repeat compress/decompress and report throughput.

If no subcommand is given, the input filename's extension decides:
  *.mz / *.mzb  -> decompress
  anything else -> compress

Options (compress):
  -1, -2, -3          Level shortcut (fastest / balanced / smallest).
  --level N           Same, explicit (1, 2 or 3).
  -0                  Uncompressed; emit 0x01 chunks only.
  --block-size N      Max block size (default 8M).  Units: K, M, G accepted.
  --pad N             Pad total output to a multiple of N bytes.
  --block             Single-block mode (.mzb).  Loads entire input into memory.
  --index, --no-index Toggle seek-index appending (default: on).

Options (decompress):
  --offset SIZE       Seek to uncompressed offset before reading.  Requires
                      an index in the stream.  Append `+nl` to advance to
                      the next newline (e.g. `--offset=4K+nl`).
  --tail SIZE         Emit only the last SIZE uncompressed bytes.  Same
                      `+nl` suffix supported.
  --follow            Re-open the input file when EOF is reached (like
                      `tail -f`).  Sleeps 1 s between retries.  Cannot
                      be combined with --tail / --bench / stdin.

Common options:
  -c, --stdout        Write all output to stdout.
  -o, --output FILE   Write output to FILE.
  --rm                Delete source after success.
  -q, --quiet         Don't print progress.
  --verify            Read + validate without producing output (decompress).
                      Or, for compress, re-decode in parallel and check.
  --bench N           Repeat operation N times; print best/avg throughput.
  --threads N, --cpu N
                      Worker-thread count (default: available_parallelism()).
                      `1` forces the single-threaded path.
  -h, --help          Print this help and exit.
";

fn main() -> ExitCode {
    let argv: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let cli = match args::parse(argv) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\nERROR: {e}");
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    if cli.options.help {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let result = run(cli);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("\nERROR: {e}");
            ExitCode::from(2)
        }
    }
}

fn run(cli: Cli) -> std::io::Result<()> {
    match cli.subcommand {
        Subcommand::Compress => compress::run(cli.options, cli.input),
        Subcommand::Decompress => decompress::run(cli.options, cli.input),
        Subcommand::Verify => verify::run(cli.options, cli.input),
        Subcommand::Bench => bench::run(cli.options, cli.input),
    }
}
