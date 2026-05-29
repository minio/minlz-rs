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

//! Hand-rolled argument parser for the `mz` CLI.

use std::ffi::OsString;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use minlz::Level;

#[derive(Debug)]
pub struct Cli {
    pub subcommand: Subcommand,
    pub options: Options,
    pub input: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subcommand {
    Compress,
    Decompress,
    Verify,
    Bench,
}

#[derive(Debug)]
pub struct Options {
    pub level: Option<Level>,
    pub uncompressed: bool,
    pub block_size: Option<usize>,
    pub padding: Option<u32>,
    pub block: bool,
    pub stdout: bool,
    pub output: Option<PathBuf>,
    pub remove: bool,
    pub quiet: bool,
    pub verify: bool,
    pub bench_n: Option<u32>,
    /// Worker-thread count.  `Some(NonZeroUsize::new(1))` ⇒ single-threaded
    /// path (stage-C `Writer` / `Read::read`); `Some(n>1)` ⇒ MT path
    /// (`MtWriter` / `Reader::decode_concurrent`); `None` ⇒ default
    /// (`available_parallelism`).
    pub threads: Option<NonZeroUsize>,
    /// Append an index chunk on compress.  Defaults to `true` for the
    /// CLI, matching Go `cmd/mz`'s `-index` default.
    pub index: bool,
    /// Decompress: seek to this uncompressed offset before reading.
    /// Requires the stream to carry an index.
    pub offset: Option<u64>,
    /// Decompress: only emit the last N uncompressed bytes (`-tail`).
    /// Requires an index.
    pub tail: Option<u64>,
    /// After seeking via `--tail` or `--offset`, skip bytes until the
    /// next `\n` so output starts at a line boundary.  Toggled by the
    /// `+nl` suffix on the flag value (e.g. `--tail=1K+nl`).
    pub tail_next_nl: bool,
    /// `--follow` / `-follow`: re-open the input file when EOF is
    /// reached, like `tail -f`.  Sleeps 1 s between retries.
    pub follow: bool,
    pub help: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            level: None,
            uncompressed: false,
            block_size: None,
            padding: None,
            block: false,
            stdout: false,
            output: None,
            remove: false,
            quiet: false,
            verify: false,
            bench_n: None,
            threads: None,
            index: true,
            offset: None,
            tail: None,
            tail_next_nl: false,
            follow: false,
            help: false,
        }
    }
}

#[derive(Debug)]
pub enum ParseError {
    UnknownFlag(String),
    MissingValue(String),
    InvalidValue(String, String),
    NoSubcommand,
    ConflictingLevels,
    TooManyInputs,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownFlag(s) => write!(f, "unknown flag: {s}"),
            Self::MissingValue(s) => write!(f, "missing value for {s}"),
            Self::InvalidValue(s, v) => write!(f, "invalid value for {s}: {v}"),
            Self::NoSubcommand => f.write_str(
                "no subcommand or input file given; try: mz c <file> | mz d <file> | mz --help",
            ),
            Self::ConflictingLevels => f.write_str("conflicting compression levels"),
            Self::TooManyInputs => f.write_str("only one input file is supported"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse the OS argument vector (skipping argv\[0\]).
pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Cli, ParseError> {
    let mut it = args.into_iter().peekable();
    // First non-flag positional is the subcommand (`c`, `d`, ...) — unless
    // the first positional looks like a filename, in which case we infer
    // the subcommand from the extension (matches Go's behaviour).
    let mut sub: Option<Subcommand> = None;
    let mut alias = SubcommandAlias::None;
    let mut opts = Options::default();
    let mut input: Option<PathBuf> = None;
    let mut help_only = false;

    while let Some(arg) = it.next() {
        let s = arg.to_string_lossy().to_string();
        if s == "--help" || s == "-h" {
            help_only = true;
            opts.help = true;
            continue;
        }
        if sub.is_none() && !s.starts_with('-') {
            if let Some(parsed) = parse_subcommand(&s) {
                sub = Some(parsed);
                alias = subcommand_alias(&s);
                continue;
            }
        }
        if s == "-" {
            // stdin/stdout sentinel; we still record it so we know it's
            // present.
            input = Some(PathBuf::from("-"));
            continue;
        }
        if let Some(rest) = s.strip_prefix("--") {
            apply_long_flag(rest, &mut it, &mut opts)?;
            continue;
        }
        if let Some(rest) = s.strip_prefix('-') {
            if rest.is_empty() {
                input = Some(PathBuf::from("-"));
                continue;
            }
            apply_short_flag(rest, &mut it, &mut opts)?;
            continue;
        }
        if input.is_some() {
            return Err(ParseError::TooManyInputs);
        }
        input = Some(PathBuf::from(s));
    }

    // Help — caller will print and exit 0; pass it through.
    if help_only {
        return Ok(Cli {
            subcommand: sub.unwrap_or(Subcommand::Compress),
            options: opts,
            input,
        });
    }

    // Infer subcommand from extension if none was given explicitly.
    let subcommand = match sub {
        Some(sc) => sc,
        None => {
            let input_ref = input.as_ref().ok_or(ParseError::NoSubcommand)?;
            if is_compressed_filename(input_ref) {
                Subcommand::Decompress
            } else {
                Subcommand::Compress
            }
        }
    };

    // Sanity: cannot combine -0 (uncompressed) with --level.
    if opts.uncompressed && opts.level.is_some() {
        return Err(ParseError::ConflictingLevels);
    }
    if subcommand == Subcommand::Bench && opts.bench_n.is_none() {
        opts.bench_n = Some(5);
    }

    // Subcommand-alias defaults — match Go `cmd/mz/main.go:110` where
    // `cat` / `tail` both forward to the decompress code path with
    // tweaked defaults.
    match alias {
        SubcommandAlias::Cat => {
            opts.stdout = true;
        }
        SubcommandAlias::Tail => {
            opts.stdout = true;
            if opts.tail.is_none() {
                opts.tail = Some(1024);
                opts.tail_next_nl = true;
            }
        }
        SubcommandAlias::None => {}
    }

    // `--follow` restrictions (match Go `cmd/mz/decompress.go:210-225`).
    if opts.follow {
        if opts.tail.is_some() {
            return Err(ParseError::InvalidValue(
                "--follow".into(),
                "cannot be combined with --tail".into(),
            ));
        }
        if opts.bench_n.is_some() {
            return Err(ParseError::InvalidValue(
                "--follow".into(),
                "cannot be combined with --bench".into(),
            ));
        }
        if input.as_deref() == Some(std::path::Path::new("-")) {
            return Err(ParseError::InvalidValue(
                "--follow".into(),
                "cannot follow stdin".into(),
            ));
        }
    }

    Ok(Cli {
        subcommand,
        options: opts,
        input,
    })
}

fn parse_subcommand(s: &str) -> Option<Subcommand> {
    match s {
        "c" | "compress" => Some(Subcommand::Compress),
        "d" | "decompress" | "cat" | "tail" => Some(Subcommand::Decompress),
        "verify" => Some(Subcommand::Verify),
        "bench" => Some(Subcommand::Bench),
        _ => None,
    }
}

/// Identify the alias the user typed so post-parse defaults can be
/// applied (e.g. `tail` forces stdout + default --tail).
fn subcommand_alias(s: &str) -> SubcommandAlias {
    match s {
        "cat" => SubcommandAlias::Cat,
        "tail" => SubcommandAlias::Tail,
        _ => SubcommandAlias::None,
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum SubcommandAlias {
    None,
    Cat,
    Tail,
}

fn apply_long_flag<I: Iterator<Item = OsString>>(
    name: &str,
    it: &mut std::iter::Peekable<I>,
    opts: &mut Options,
) -> Result<(), ParseError> {
    // `--name=value` form.
    if let Some((k, v)) = name.split_once('=') {
        return set_long_value(k, v, opts);
    }
    match name {
        "block" => opts.block = true,
        "stdout" => opts.stdout = true,
        "rm" => opts.remove = true,
        "quiet" => opts.quiet = true,
        "verify" => opts.verify = true,
        "help" => opts.help = true,
        "no-index" => opts.index = false,
        "follow" => opts.follow = true,
        _ => {
            // Long flags that take a separate argument.
            let val = it
                .next()
                .ok_or_else(|| ParseError::MissingValue(format!("--{name}")))?
                .into_string()
                .map_err(|_| ParseError::InvalidValue(format!("--{name}"), "<non-utf8>".into()))?;
            return set_long_value(name, &val, opts);
        }
    }
    Ok(())
}

fn set_long_value(name: &str, val: &str, opts: &mut Options) -> Result<(), ParseError> {
    match name {
        "level" => opts.level = Some(parse_level(val)?),
        "block-size" => {
            opts.block_size = Some(
                parse_size(val)
                    .map_err(|_| ParseError::InvalidValue("--block-size".into(), val.to_string()))?
                    as usize,
            )
        }
        "pad" => {
            opts.padding = Some(
                parse_size(val)
                    .map_err(|_| ParseError::InvalidValue("--pad".into(), val.to_string()))?
                    as u32,
            )
        }
        "output" => opts.output = Some(PathBuf::from(val)),
        "bench" => {
            opts.bench_n = Some(
                val.parse()
                    .map_err(|_| ParseError::InvalidValue("--bench".into(), val.to_string()))?,
            )
        }
        "threads" | "cpu" => {
            let n: usize = val
                .parse()
                .map_err(|_| ParseError::InvalidValue(format!("--{name}"), val.to_string()))?;
            opts.threads = Some(NonZeroUsize::new(n).ok_or_else(|| {
                ParseError::InvalidValue(format!("--{name}"), "must be ≥ 1".into())
            })?);
        }
        "index" => match val {
            "true" | "1" | "yes" => opts.index = true,
            "false" | "0" | "no" => opts.index = false,
            _ => return Err(ParseError::InvalidValue("--index".into(), val.to_string())),
        },
        "offset" => {
            let (n, nl) = parse_size_with_nl(val)
                .map_err(|_| ParseError::InvalidValue("--offset".into(), val.to_string()))?;
            opts.offset = Some(n);
            if nl {
                opts.tail_next_nl = true;
            }
        }
        "tail" => {
            let (n, nl) = parse_size_with_nl(val)
                .map_err(|_| ParseError::InvalidValue("--tail".into(), val.to_string()))?;
            opts.tail = Some(n);
            if nl {
                opts.tail_next_nl = true;
            }
        }
        _ => return Err(ParseError::UnknownFlag(format!("--{name}"))),
    }
    Ok(())
}

fn apply_short_flag<I: Iterator<Item = OsString>>(
    rest: &str,
    it: &mut std::iter::Peekable<I>,
    opts: &mut Options,
) -> Result<(), ParseError> {
    // Multi-char form like `-index=true` / `-offset 64K` — forward to
    // the long-flag handler for Go `cmd/mz` parity.
    let stem = rest.split_once('=').map(|(k, _)| k).unwrap_or(rest);
    if stem.len() > 1 {
        return apply_long_flag(rest, it, opts);
    }
    // Single-char shorts: -0, -1, -2, -3, -c, -q, -h, -o FILE
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '0' => opts.uncompressed = true,
            '1' => opts.level = Some(Level::Fastest),
            '2' => opts.level = Some(Level::Balanced),
            '3' => opts.level = Some(Level::Smallest),
            'c' => opts.stdout = true,
            'q' => opts.quiet = true,
            'h' => opts.help = true,
            'o' => {
                let val = if let Some(c) = chars.next() {
                    // `-oFILE` or `-o FILE`.
                    let mut s: String = std::iter::once(c).chain(chars.by_ref()).collect();
                    if s.is_empty() {
                        s = it
                            .next()
                            .ok_or_else(|| ParseError::MissingValue("-o".into()))?
                            .into_string()
                            .map_err(|_| {
                                ParseError::InvalidValue("-o".into(), "<non-utf8>".into())
                            })?;
                    }
                    s
                } else {
                    it.next()
                        .ok_or_else(|| ParseError::MissingValue("-o".into()))?
                        .into_string()
                        .map_err(|_| ParseError::InvalidValue("-o".into(), "<non-utf8>".into()))?
                };
                opts.output = Some(PathBuf::from(val));
            }
            _ => return Err(ParseError::UnknownFlag(format!("-{c}"))),
        }
    }
    Ok(())
}

fn parse_level(val: &str) -> Result<Level, ParseError> {
    match val {
        "1" | "fastest" => Ok(Level::Fastest),
        "2" | "balanced" => Ok(Level::Balanced),
        "3" | "smallest" => Ok(Level::Smallest),
        _ => Err(ParseError::InvalidValue("--level".into(), val.to_string())),
    }
}

/// Parse a `--tail` / `--offset` value: an optional `+nl` suffix asks
/// the decoder to advance to the next newline after seeking, so output
/// starts at a line boundary.  Returns `(size_bytes, next_nl)`.
pub fn parse_size_with_nl(s: &str) -> Result<(u64, bool), &'static str> {
    let (size_str, nl) = if let Some(stem) = s.strip_suffix("+nl") {
        (stem, true)
    } else {
        (s, false)
    };
    Ok((parse_size(size_str)?, nl))
}

/// Parse a size string with optional unit suffix (matches Go's `toSize`).
pub fn parse_size(s: &str) -> Result<u64, &'static str> {
    if s.is_empty() {
        return Ok(0);
    }
    let s = s.trim().to_uppercase();
    let (num, suffix) = s
        .find(|c: char| c.is_alphabetic())
        .map(|i| (&s[..i], &s[i..]))
        .unwrap_or((&s[..], ""));
    let n: u64 = num.parse().map_err(|_| "not a number")?;
    let mul: u64 = match suffix {
        "" | "B" => 1,
        "K" | "KB" | "KIB" => 1 << 10,
        "M" | "MB" | "MIB" => 1 << 20,
        "G" | "GB" | "GIB" => 1 << 30,
        "T" | "TB" | "TIB" => 1 << 40,
        _ => return Err("unknown size suffix"),
    };
    n.checked_mul(mul).ok_or("size overflow")
}

fn is_compressed_filename(path: &std::path::Path) -> bool {
    matches!(
        path.extension().and_then(|s| s.to_str()),
        Some("mz") | Some("mzb")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_argv(args: &[&str]) -> Result<Cli, ParseError> {
        parse(args.iter().map(|s| OsString::from(*s)))
    }

    #[test]
    fn compress_subcommand() {
        let cli = parse_argv(&["c", "file.txt"]).unwrap();
        assert_eq!(cli.subcommand, Subcommand::Compress);
        assert_eq!(cli.input.as_deref(), Some(std::path::Path::new("file.txt")));
    }

    #[test]
    fn infer_decompress_from_extension() {
        let cli = parse_argv(&["file.mz"]).unwrap();
        assert_eq!(cli.subcommand, Subcommand::Decompress);
    }

    #[test]
    fn level_flags() {
        let cli = parse_argv(&["-1", "x"]).unwrap();
        assert_eq!(cli.options.level, Some(Level::Fastest));
        let cli = parse_argv(&["-3", "x"]).unwrap();
        assert_eq!(cli.options.level, Some(Level::Smallest));
        let cli = parse_argv(&["--level", "2", "x"]).unwrap();
        assert_eq!(cli.options.level, Some(Level::Balanced));
    }

    #[test]
    fn level_zero_uncompressed() {
        let cli = parse_argv(&["-0", "x"]).unwrap();
        assert!(cli.options.uncompressed);
    }

    #[test]
    fn output_flag_short() {
        let cli = parse_argv(&["-o", "out.mz", "in.txt"]).unwrap();
        assert_eq!(
            cli.options.output.as_deref(),
            Some(std::path::Path::new("out.mz"))
        );
    }

    #[test]
    fn output_flag_long() {
        let cli = parse_argv(&["--output=foo.mz", "in.txt"]).unwrap();
        assert_eq!(
            cli.options.output.as_deref(),
            Some(std::path::Path::new("foo.mz"))
        );
    }

    #[test]
    fn block_flag() {
        let cli = parse_argv(&["--block", "in.txt"]).unwrap();
        assert!(cli.options.block);
    }

    #[test]
    fn stdin_sentinel() {
        let cli = parse_argv(&["c", "-"]).unwrap();
        assert_eq!(cli.input.as_deref(), Some(std::path::Path::new("-")));
    }

    #[test]
    fn parse_size_units() {
        assert_eq!(parse_size("").unwrap(), 0);
        assert_eq!(parse_size("1024").unwrap(), 1024);
        assert_eq!(parse_size("1K").unwrap(), 1 << 10);
        assert_eq!(parse_size("4MiB").unwrap(), 4 << 20);
        assert_eq!(parse_size("2GB").unwrap(), 2 << 30);
        assert!(parse_size("abc").is_err());
        assert!(parse_size("4XX").is_err());
    }

    #[test]
    fn conflicting_levels_error() {
        // -0 + --level should conflict.
        let err = parse_argv(&["-0", "--level", "2", "x"]).unwrap_err();
        matches!(err, ParseError::ConflictingLevels);
    }

    #[test]
    fn no_subcommand_no_input_errors() {
        let err = parse_argv(&[]).unwrap_err();
        matches!(err, ParseError::NoSubcommand);
    }

    #[test]
    fn block_size_pad() {
        let cli = parse_argv(&["--block-size", "64K", "--pad", "4096", "in.txt"]).unwrap();
        assert_eq!(cli.options.block_size, Some(64 << 10));
        assert_eq!(cli.options.padding, Some(4096));
    }

    #[test]
    fn cat_subcommand_forces_stdout() {
        let cli = parse_argv(&["cat", "file.mz"]).unwrap();
        assert_eq!(cli.subcommand, Subcommand::Decompress);
        assert!(cli.options.stdout);
        // No tail default — that's tail-specific.
        assert!(cli.options.tail.is_none());
    }

    #[test]
    fn tail_subcommand_sets_defaults() {
        let cli = parse_argv(&["tail", "file.mz"]).unwrap();
        assert_eq!(cli.subcommand, Subcommand::Decompress);
        assert!(cli.options.stdout);
        assert_eq!(cli.options.tail, Some(1024));
        assert!(cli.options.tail_next_nl);
    }

    #[test]
    fn tail_subcommand_explicit_size_overrides_default() {
        let cli = parse_argv(&["tail", "--tail=64K", "file.mz"]).unwrap();
        assert_eq!(cli.options.tail, Some(64 << 10));
        // No +nl suffix on the explicit value -> stay false.
        assert!(!cli.options.tail_next_nl);
    }

    #[test]
    fn tail_nl_suffix_parsed() {
        let cli = parse_argv(&["d", "--tail=4K+nl", "file.mz"]).unwrap();
        assert_eq!(cli.options.tail, Some(4 << 10));
        assert!(cli.options.tail_next_nl);
    }

    #[test]
    fn offset_nl_suffix_parsed() {
        let cli = parse_argv(&["d", "--offset=512+nl", "file.mz"]).unwrap();
        assert_eq!(cli.options.offset, Some(512));
        assert!(cli.options.tail_next_nl);
    }

    #[test]
    fn follow_flag_parses() {
        let cli = parse_argv(&["d", "--follow", "file.mz"]).unwrap();
        assert!(cli.options.follow);
    }

    #[test]
    fn follow_with_tail_errors() {
        let err = parse_argv(&["d", "--follow", "--tail=1K", "file.mz"]).unwrap_err();
        assert!(matches!(err, ParseError::InvalidValue(ref s, _) if s == "--follow"));
    }

    #[test]
    fn follow_with_stdin_errors() {
        let err = parse_argv(&["d", "--follow", "-"]).unwrap_err();
        assert!(matches!(err, ParseError::InvalidValue(ref s, _) if s == "--follow"));
    }

    #[test]
    fn follow_with_bench_errors() {
        let err = parse_argv(&["d", "--follow", "--bench=2", "file.mz"]).unwrap_err();
        assert!(matches!(err, ParseError::InvalidValue(ref s, _) if s == "--follow"));
    }

    #[test]
    fn go_style_single_dash_follow() {
        let cli = parse_argv(&["d", "-follow", "file.mz"]).unwrap();
        assert!(cli.options.follow);
    }
}
