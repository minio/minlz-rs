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

use minlz::{Level, cpu};

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
    /// The level to write. After [`parse`] it is a SIMD level unless `--classic` is given,
    /// and `None` only with `-0`.
    pub level: Option<Level>,
    pub uncompressed: bool,
    /// `--classic`: write classic MinLZ blocks (Go `-classic`).
    pub classic: bool,
    /// `--simd.delta`: SIMD delta literals (Go `-simd.delta`).
    pub simd_delta: bool,
    /// `minlz::cpu` features turned off by the hidden `--avx2/--avx512/--neon=false`.
    pub cpu_off: u32,
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
            classic: false,
            simd_delta: false,
            cpu_off: 0,
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

    while let Some(arg) = it.next() {
        let s = arg.to_string_lossy().to_string();
        if sub.is_none()
            && !s.starts_with('-')
            && let Some(parsed) = parse_subcommand(&s)
        {
            sub = Some(parsed);
            alias = subcommand_alias(&s);
            continue;
        }
        // As in Go's flag package, `-name` and `--name` are the same flag; `-` is stdin.
        if s != "-"
            && let Some(rest) = s.strip_prefix("--").or_else(|| s.strip_prefix('-'))
        {
            apply_long_flag(rest, &mut it, &mut opts)?;
            continue;
        }
        if input.is_some() {
            return Err(ParseError::TooManyInputs);
        }
        input = Some(PathBuf::from(s));
    }
    opts.cpu_off = cpu_mask(opts.cpu_off);

    // Help — caller will print and exit 0; pass it through.
    if opts.help {
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
    let level = write_level(&opts)?;
    if !opts.uncompressed {
        opts.level = Some(level);
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

/// The level Go's `mainCompress` writes: SIMD blocks unless `--classic`.
fn write_level(opts: &Options) -> Result<Level, ParseError> {
    let level = opts.level.unwrap_or(Level::Balanced);
    if opts.simd_delta && (opts.classic || opts.uncompressed || level == Level::Smallest) {
        return Err(ParseError::InvalidValue(
            "--simd.delta".into(),
            "cannot be used with --classic, -0 or -3".into(),
        ));
    }
    if opts.classic {
        return Ok(level);
    }
    Ok(match (level, opts.simd_delta) {
        (Level::SuperFast, false) => Level::SuperFastSimd,
        (Level::SuperFast, true) => Level::SuperFastSimdDelta,
        (Level::Fastest, false) => Level::FastestSimd,
        (Level::Fastest, true) => Level::FastestSimdDelta,
        (Level::Balanced, false) => Level::BalancedSimd,
        (Level::Balanced, true) => Level::BalancedSimdDelta,
        (Level::Smallest, _) => Level::SmallestSimd,
        (l, _) => l,
    })
}

/// Widens the features `--avx2/--avx512/--neon=false` turned off to all the `minlz::cpu`
/// features they cover. As in Go, AVX-512 needs AVX2, and the pivco kernels follow the exec
/// tiers.
fn cpu_mask(mut off: u32) -> u32 {
    if off & cpu::AVX2 != 0 {
        off |= cpu::PIVCO_AVX2 | cpu::AVX512;
    }
    if off & cpu::AVX512 != 0 {
        off |= cpu::PIVCO_AVX512 | cpu::PIVCO_HIST_AVX512;
    }
    off
}

/// Records a CPU toggle on its own bit, so the last value given wins, as in Go.
fn set_cpu(opts: &mut Options, feature: u32, on: bool) {
    if on {
        opts.cpu_off &= !feature;
    } else {
        opts.cpu_off |= feature;
    }
}

fn parse_bool(name: &str, val: &str) -> Result<bool, ParseError> {
    match val {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(ParseError::InvalidValue(
            format!("--{name}"),
            val.to_string(),
        )),
    }
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
    flag: &str,
    it: &mut std::iter::Peekable<I>,
    opts: &mut Options,
) -> Result<(), ParseError> {
    let (name, val) = match flag.split_once('=') {
        Some((k, v)) => (k, Some(v)),
        None => (flag, None),
    };
    // As in Go's flag package, a bool flag is `-name` or `-name=value`, never `-name value`.
    let on = || val.map_or(Ok(true), |v| parse_bool(name, v));
    match name {
        // Go's level flags are bools too; `=false` leaves the level alone.
        "xfast" | "1" | "2" | "3" if !on()? => {}
        "xfast" => opts.level = Some(Level::SuperFast),
        "1" => opts.level = Some(Level::Fastest),
        "2" => opts.level = Some(Level::Balanced),
        "3" => opts.level = Some(Level::Smallest),
        "0" => opts.uncompressed = on()?,
        "block" => opts.block = on()?,
        "c" | "stdout" => opts.stdout = on()?,
        "rm" => opts.remove = on()?,
        "q" | "quiet" => opts.quiet = on()?,
        "verify" => opts.verify = on()?,
        "h" | "help" => opts.help = on()?,
        "index" => opts.index = on()?,
        "no-index" => opts.index = !on()?,
        "follow" => opts.follow = on()?,
        "classic" => opts.classic = on()?,
        "simd.delta" => opts.simd_delta = on()?,
        "avx2" => set_cpu(opts, cpu::AVX2, on()?),
        "avx512" => set_cpu(opts, cpu::AVX512, on()?),
        "neon" => set_cpu(opts, cpu::NEON, on()?),
        _ => {
            let val = match val {
                Some(v) => v.to_owned(),
                None => it
                    .next()
                    .ok_or_else(|| ParseError::MissingValue(format!("--{name}")))?
                    .into_string()
                    .map_err(|_| {
                        ParseError::InvalidValue(format!("--{name}"), "<non-utf8>".into())
                    })?,
            };
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
        "o" | "output" => opts.output = Some(PathBuf::from(val)),
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
        let cli = parse_argv(&["-xfast", "--classic", "x"]).unwrap();
        assert_eq!(cli.options.level, Some(Level::SuperFast));
        let cli = parse_argv(&["-1", "--classic", "x"]).unwrap();
        assert_eq!(cli.options.level, Some(Level::Fastest));
        let cli = parse_argv(&["-3", "-classic", "x"]).unwrap();
        assert_eq!(cli.options.level, Some(Level::Smallest));
        let cli = parse_argv(&["--level", "2", "--classic=true", "x"]).unwrap();
        assert_eq!(cli.options.level, Some(Level::Balanced));
        let cli = parse_argv(&["--classic", "x"]).unwrap();
        assert_eq!(cli.options.level, Some(Level::Balanced));
    }

    #[test]
    fn simd_levels_by_default() {
        for (args, want) in [
            (&[][..], Level::BalancedSimd),
            (&["-xfast"], Level::SuperFastSimd),
            (&["--xfast"], Level::SuperFastSimd),
            (&["-1"], Level::FastestSimd),
            (&["-2"], Level::BalancedSimd),
            (&["-3"], Level::SmallestSimd),
            (&["--level", "1"], Level::FastestSimd),
            (&["--classic=false", "-1"], Level::FastestSimd),
            (&["--simd.delta"], Level::BalancedSimdDelta),
            (&["-xfast", "-simd.delta"], Level::SuperFastSimdDelta),
            (&["-1", "--simd.delta=true"], Level::FastestSimdDelta),
            (&["-2", "--simd.delta"], Level::BalancedSimdDelta),
            (&["-2", "--simd.delta=false"], Level::BalancedSimd),
        ] {
            let cli = parse_argv(&[&["c"], args, &["x"]].concat()).unwrap();
            assert_eq!(cli.options.level, Some(want), "{args:?}");
        }
    }

    #[test]
    fn simd_flag_conflicts() {
        for args in [
            &["--simd.delta", "--classic"][..],
            &["--simd.delta", "-0"],
            &["--simd.delta", "-3"],
        ] {
            let err = parse_argv(&[&["c"], args, &["x"]].concat()).unwrap_err();
            assert!(
                matches!(err, ParseError::InvalidValue(ref s, _) if s == "--simd.delta"),
                "{args:?}: {err:?}"
            );
        }
        let err = parse_argv(&["c", "-xfast", "-0", "x"]).unwrap_err();
        assert!(matches!(err, ParseError::ConflictingLevels), "{err:?}");
    }

    #[test]
    fn cpu_toggles() {
        const AVX512: u32 = cpu::AVX512 | cpu::PIVCO_AVX512 | cpu::PIVCO_HIST_AVX512;
        for (args, want) in [
            (&["-avx2", "--avx512=true", "-neon=1"][..], 0),
            (&["-avx2=false"], cpu::AVX2 | cpu::PIVCO_AVX2 | AVX512),
            (&["--avx512=false"], AVX512),
            (&["-neon=false"], cpu::NEON),
            (&["-avx512=false", "-avx2=true"], AVX512),
        ] {
            let cli = parse_argv(&[args, &["c", "x"]].concat()).unwrap();
            assert_eq!(cli.options.cpu_off, want, "{args:?}");
        }
        assert!(parse_argv(&["-avx2=maybe", "c", "x"]).is_err());
    }

    #[test]
    fn bool_flag_values() {
        for (args, want) in [
            (&["-xfast=true"][..], Level::SuperFastSimd),
            (&["-1", "-xfast=false"], Level::FastestSimd),
            (&["-3=1"], Level::SmallestSimd),
        ] {
            let cli = parse_argv(&[&["c"], args, &["x"]].concat()).unwrap();
            assert_eq!(cli.options.level, Some(want), "{args:?}");
        }
        let cli = parse_argv(&["c", "-c=true", "-block=1", "-rm=false", "-0=false", "x"]).unwrap();
        let o = &cli.options;
        assert!(o.stdout && o.block && !o.remove && !o.uncompressed);
        for args in [
            &["-avx2=false", "-avx2=true"][..],
            &["-neon=false", "--neon"],
            &["-avx512=false", "-avx512"],
        ] {
            let cli = parse_argv(&[args, &["c", "x"]].concat()).unwrap();
            assert_eq!(cli.options.cpu_off, 0, "{args:?}");
        }
        assert!(parse_argv(&["c", "-q=maybe", "x"]).is_err());
    }

    #[test]
    fn level_zero_uncompressed() {
        let cli = parse_argv(&["-0", "x"]).unwrap();
        assert!(cli.options.uncompressed);
        assert_eq!(cli.options.level, None);
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
    fn go_style_flag_values() {
        let cli = parse_argv(&["c", "-o=out.mz", "in.txt"]).unwrap();
        assert_eq!(
            cli.options.output.as_deref(),
            Some(std::path::Path::new("out.mz"))
        );
        let cli = parse_argv(&["c", "--o", "out.mz", "--1", "in.txt"]).unwrap();
        assert_eq!(
            cli.options.output.as_deref(),
            Some(std::path::Path::new("out.mz"))
        );
        assert_eq!(cli.options.level, Some(Level::FastestSimd));
        assert!(matches!(
            parse_argv(&["c", "a.txt", "-"]),
            Err(ParseError::TooManyInputs)
        ));
    }

    #[test]
    fn bare_index_is_a_toggle() {
        for args in [
            &["c", "--no-index", "--index", "e.txt"][..],
            &["c", "-index", "e.txt"],
        ] {
            let cli = parse_argv(args).unwrap();
            assert!(cli.options.index, "{args:?}");
            assert_eq!(cli.input, Some(PathBuf::from("e.txt")), "{args:?}");
        }
        let cli = parse_argv(&["c", "-index", "-c", "e.txt"]).unwrap();
        assert!(cli.options.index && cli.options.stdout);
        let cli = parse_argv(&["c", "-index=false", "e.txt"]).unwrap();
        assert!(!cli.options.index);
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
        assert!(matches!(err, ParseError::ConflictingLevels), "{err:?}");
    }

    #[test]
    fn no_subcommand_no_input_errors() {
        let err = parse_argv(&[]).unwrap_err();
        assert!(matches!(err, ParseError::NoSubcommand), "{err:?}");
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
    fn every_help_spelling_skips_validation() {
        for h in ["-h", "--help", "-help"] {
            for argv in [
                &[h][..],
                &["c", "--simd.delta", "-3", h],
                &["c", "-0", "--level", "2", h],
            ] {
                assert!(parse_argv(argv).unwrap().options.help, "{argv:?}");
            }
        }
    }

    #[test]
    fn go_style_single_dash_follow() {
        let cli = parse_argv(&["d", "-follow", "file.mz"]).unwrap();
        assert!(cli.options.follow);
    }
}
