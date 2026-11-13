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

//! `minlz-bench` measures the throughput of the MinLZ codec through its public API. It uses no
//! other part of the library, so the same source builds against an older commit, and
//! `cargo xtask bench <rev>` builds it against commit `rev` and against the working tree.
//!
//! ```text
//! minlz-bench list                 the cases
//! minlz-bench run <case> [secs]    one case, as `compare` runs it
//! minlz-bench compare <base> [--pairs N] [--secs S] [--only TEXT]
//! ```
//!
//! `compare` runs each case in the base build and in this one, in alternating order, `--pairs`
//! times (default 6), `--secs` seconds each (default 1), and prints the median of this build's
//! speed over the base's. A run reports its fastest sample, which leaves out the samples that
//! other processes interrupted. Pin the process to one core (`taskset -c 5` on Linux,
//! `start /affinity 20` on Windows): the runs it starts inherit the pin.
//!
//! The cases decode blocks at every level, and read and write streams at level Fastest. The
//! inputs are Tom Sawyer repeated to 4 MB and the Snappy benchmark files in `testdata/bench`,
//! which `cargo bench` downloads.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::hint::black_box;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::{Duration, Instant};
use std::{env, fs};

use minlz::Level;
use minlz::stream::{Reader, WriterBuilder};

/// The operation of a case.
#[derive(Clone, Copy, PartialEq)]
enum Op {
    /// `minlz::decode` of the block encoding.
    BlockDecode,
    /// A `Reader` over the stream encoding.
    StreamRead,
    /// A `Writer` that writes the stream encoding.
    StreamWrite,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::BlockDecode => "block-decode",
            Op::StreamRead => "stream-read",
            Op::StreamWrite => "stream-write",
        }
    }
}

/// One measurement: an operation on an input at a level.
struct Case {
    op: Op,
    input: &'static str,
    level: Level,
}

impl Case {
    fn name(&self) -> String {
        format!("{}/{}/{:?}", self.op.name(), self.input, self.level)
    }
}

/// Returns every case, in the order `compare` runs them.
fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    for input in ["twain-4m", "corpus"] {
        for level in [Level::Fastest, Level::Balanced, Level::Smallest] {
            v.push(Case {
                op: Op::BlockDecode,
                input,
                level,
            });
        }
    }
    for op in [Op::StreamRead, Op::StreamWrite] {
        for input in ["twain-4m", "corpus"] {
            v.push(Case {
                op,
                input,
                level: Level::Fastest,
            });
        }
    }
    v
}

/// Returns the folder of the test data: `$MINLZ_TESTDATA`, or `testdata/` of the workspace this
/// binary was built in.
fn testdata() -> PathBuf {
    match env::var_os("MINLZ_TESTDATA") {
        Some(d) => PathBuf::from(d),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata"),
    }
}

fn read(rel: &str) -> Vec<u8> {
    let p = testdata().join(rel);
    fs::read(&p).unwrap_or_else(|e| panic!("{}: {e} (set MINLZ_TESTDATA)", p.display()))
}

/// Returns the input `name`: `corpus`, the Snappy benchmark files one after another, or
/// `twain-<n>`, the first n bytes of Tom Sawyer repeated, with each copy XORed with its count as
/// Go's tests do (k is 1000 bytes, m 1000000).
fn input(name: &str) -> Vec<u8> {
    if name == "corpus" {
        let files = [
            "html",
            "urls.10K",
            "fireworks.jpeg",
            "paper-100k.pdf",
            "html_x_4",
            "alice29.txt",
            "asyoulik.txt",
            "lcet10.txt",
            "plrabn12.txt",
            "geo.protodata",
            "kppkn.gtb",
        ];
        return files
            .iter()
            .flat_map(|f| read(&format!("bench/{f}")))
            .collect();
    }
    let n = name
        .strip_prefix("twain-")
        .expect("input is corpus or twain-<n>");
    let n = match n.as_bytes()[n.len() - 1] {
        b'k' => n[..n.len() - 1].parse::<usize>().expect("size") * 1000,
        b'm' => n[..n.len() - 1].parse::<usize>().expect("size") * 1_000_000,
        _ => n.parse().expect("size"),
    };
    let src = read("Mark.Twain-Tom.Sawyer.txt");
    let mut dst = Vec::with_capacity(n);
    let mut cnt = 0u8;
    while dst.len() < n {
        let take = src.len().min(n - dst.len());
        dst.extend(src[..take].iter().map(|b| b ^ cnt));
        cnt = cnt.wrapping_add(1);
    }
    dst
}

/// Runs f for secs seconds in samples of at least 100 µs and returns the fastest time per call in
/// ns and the number of calls.
fn measure(secs: f64, mut f: impl FnMut()) -> (f64, u64) {
    let mut reps = 1u64;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        if t.elapsed() >= Duration::from_micros(100) {
            break;
        }
        reps *= 2;
    }
    let start = Instant::now();
    let (mut best, mut calls) = (f64::MAX, 0);
    while start.elapsed().as_secs_f64() < secs {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        best = best.min(t.elapsed().as_nanos() as f64 / reps as f64);
        calls += reps;
    }
    (best, calls)
}

/// Runs case for secs seconds and returns the fastest time per call in ns, the bytes of the input
/// and a hash of the encoding that the case decodes or writes, by which `compare` sees whether
/// both builds worked on the same encoding. Both builds come from one source and one toolchain,
/// so they hash alike.
fn run(case: &Case, secs: f64) -> (f64, usize, u64) {
    let src = input(case.input);
    let write_stream = |src: &[u8]| {
        let mut w = WriterBuilder::new()
            .level(case.level)
            .build(Vec::new())
            .expect("writer");
        w.write_all(src).expect("write");
        w.finish().expect("finish")
    };
    let (ns, enc) = match case.op {
        Op::BlockDecode => {
            let mut enc = Vec::new();
            minlz::encode(&mut enc, &src, case.level).expect("encode");
            let mut out = Vec::with_capacity(src.len());
            let (ns, _) = measure(secs, || {
                out.clear();
                minlz::decode(&mut out, black_box(&enc)).expect("decode");
                black_box(&out);
            });
            assert!(out == src, "decode differs from the input");
            (ns, enc)
        }
        Op::StreamRead => {
            let enc = write_stream(&src);
            let mut out = Vec::with_capacity(src.len());
            let (ns, _) = measure(secs, || {
                out.clear();
                Reader::new(black_box(&enc[..]))
                    .read_to_end(&mut out)
                    .expect("read");
                black_box(&out);
            });
            assert!(out == src, "read differs from the input");
            (ns, enc)
        }
        Op::StreamWrite => {
            let (ns, _) = measure(secs, || {
                black_box(write_stream(black_box(&src)));
            });
            (ns, write_stream(&src))
        }
    };
    let mut h = DefaultHasher::new();
    enc.hash(&mut h);
    (ns, src.len(), h.finish())
}

/// Runs case in exe and returns its fastest time per call in ns, the input's bytes and the
/// encoding's hash.
fn run_in(exe: &Path, case: &str, secs: f64) -> Result<(f64, usize, u64), String> {
    let out = Command::new(exe)
        .args(["run", case, &secs.to_string()])
        .env("MINLZ_TESTDATA", testdata())
        .output()
        .map_err(|e| format!("{}: {e}", exe.display()))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let f: Vec<&str> = text.split_whitespace().collect();
    if !out.status.success() || f.len() != 3 {
        return Err(format!(
            "{} run {case}: {}{}",
            exe.display(),
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let bad = || format!("{} run {case}: bad output {text}", exe.display());
    Ok((
        f[0].parse().map_err(|_| bad())?,
        f[1].parse().map_err(|_| bad())?,
        u64::from_str_radix(f[2], 16).map_err(|_| bad())?,
    ))
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Times every case whose name contains only, in base and in this build, alternating which runs
/// first, and prints a row per case and the geometric mean of the ratios per operation.
fn compare(base: &Path, pairs: usize, secs: f64, only: &str) -> Result<(), String> {
    let me = env::current_exe().map_err(|e| e.to_string())?;
    println!(
        "Speed of this build over {}: median of {pairs} alternating pairs",
        base.display()
    );
    println!("(smallest and largest pair), {secs} s per run, fastest sample of each run.\n");
    println!("| Case | Base MB/s | This MB/s | Ratio | Range |");
    println!("| --- | ---: | ---: | ---: | --- |");
    let mut logs: Vec<(Op, f64)> = Vec::new();
    for case in cases().iter().filter(|c| c.name().contains(only)) {
        let name = case.name();
        let (mut b, mut t, mut r) = (Vec::new(), Vec::new(), Vec::new());
        let mut same = true;
        for i in 0..pairs {
            let (nb, nt, bytes) = if i % 2 == 0 {
                let x = run_in(base, &name, secs)?;
                let y = run_in(&me, &name, secs)?;
                same &= x.2 == y.2;
                (x.0, y.0, x.1)
            } else {
                let y = run_in(&me, &name, secs)?;
                let x = run_in(base, &name, secs)?;
                same &= x.2 == y.2;
                (x.0, y.0, x.1)
            };
            b.push(bytes as f64 * 1e3 / nb);
            t.push(bytes as f64 * 1e3 / nt);
            r.push(nb / nt);
        }
        let ratio = median(&mut r);
        let note = if same {
            ""
        } else {
            " (the builds encode differently)"
        };
        println!(
            "| {name}{note} | {:.0} | {:.0} | {ratio:.3} | {:.2} to {:.2} |",
            median(&mut b),
            median(&mut t),
            r[0],
            r[r.len() - 1],
        );
        logs.push((case.op, ratio.ln()));
    }
    println!();
    for op in [Op::BlockDecode, Op::StreamRead, Op::StreamWrite] {
        let l: Vec<f64> = logs.iter().filter(|x| x.0 == op).map(|x| x.1).collect();
        if !l.is_empty() {
            let g = (l.iter().sum::<f64>() / l.len() as f64).exp();
            println!("Geometric mean, {}: {g:.3}", op.name());
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let usage = "usage: minlz-bench list | run <case> [secs] | \
                 compare <base> [--pairs N] [--secs S] [--only TEXT]";
    let result = match args.first().map(String::as_str) {
        Some("list") => {
            cases().iter().for_each(|c| println!("{}", c.name()));
            Ok(())
        }
        Some("run") if args.len() == 2 || args.len() == 3 => {
            let secs = match args.get(2).map(|s| s.parse::<f64>()) {
                None => Ok(1.0),
                Some(Ok(s)) if s.is_finite() && s > 0.0 => Ok(s),
                Some(Ok(s)) => Err(format!("secs must be a positive number, not {s}")),
                Some(Err(e)) => Err(format!("secs: {e}")),
            };
            secs.and_then(
                |secs| match cases().into_iter().find(|c| c.name() == args[1]) {
                    Some(case) => {
                        let (ns, bytes, hash) = run(&case, secs);
                        println!("{ns:.1} {bytes} {hash:016x}");
                        Ok(())
                    }
                    None => Err(format!("no case {} (see minlz-bench list)", args[1])),
                },
            )
        }
        Some("compare") if args.len() >= 2 => {
            let (mut pairs, mut secs, mut only) = (6, 1.0_f64, String::new());
            let mut i = 2;
            let mut bad = false;
            while i + 1 < args.len() {
                match args[i].as_str() {
                    "--pairs" => pairs = args[i + 1].parse().unwrap_or(0),
                    "--secs" => secs = args[i + 1].parse().unwrap_or(0.0),
                    "--only" => only = args[i + 1].clone(),
                    _ => bad = true,
                }
                i += 2;
            }
            if bad || i != args.len() || pairs == 0 || !secs.is_finite() || secs <= 0.0 {
                Err(usage.to_string())
            } else {
                compare(Path::new(&args[1]), pairs, secs, &only)
            }
        }
        _ => Err(usage.to_string()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("minlz-bench: {e}");
            ExitCode::FAILURE
        }
    }
}
