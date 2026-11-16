//! In-memory throughput comparison: MinLZ vs Snappy, LZ4, gzip, zstd.
//!
//! Loads each input fully into RAM (capped by --limit, default 1 GiB),
//! runs `--iters` rounds per codec, and records best-of-N encode/decode
//! MB/s plus the compressed size.  All codecs use their stream/frame
//! encoder so framing overhead and checksums are included on both sides.
//!
//! Measuring appends CSV rows to `--csv PATH`; `--report CSV...` prints the
//! markdown tables: one geomean table per CSV file, then the per-file
//! tables.  Single-threaded and MT runs are separate processes, so the
//! single-threaded one can be pinned to one CPU; both can append to one CSV.

use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use minlz::stream::{MtWriterBuilder, Reader, WriterBuilder};
use minlz::Level;

const DEFAULT_LIMIT: u64 = 1 << 30;
const DEFAULT_ITERS: u32 = 3;

const MINLZ: [Level; 11] = [
    Level::SuperFast,
    Level::Fastest,
    Level::Balanced,
    Level::Smallest,
    Level::SuperFastSimd,
    Level::FastestSimd,
    Level::BalancedSimd,
    Level::SmallestSimd,
    Level::SuperFastSimdDelta,
    Level::FastestSimdDelta,
    Level::BalancedSimdDelta,
];

#[derive(Clone, Copy, PartialEq)]
enum Codec {
    MinLz(Level),
    MinLzMt(Level),
    Snappy,
    Lz4,
    Gzip1,
    Zstd1,
}

impl Codec {
    /// Every codec, in report order.
    fn all() -> impl Iterator<Item = Codec> {
        MINLZ
            .into_iter()
            .map(Codec::MinLzMt)
            .chain(MINLZ.into_iter().map(Codec::MinLz))
            .chain([Codec::Snappy, Codec::Lz4, Codec::Gzip1, Codec::Zstd1])
    }

    // SIMD numbering follows the Go README ("MinLZ SIMD 1" = FastestSIMD)
    // and `mz -xfast`.
    fn name(self) -> &'static str {
        match self {
            Codec::MinLz(l) | Codec::MinLzMt(l) => match l {
                Level::SuperFast => "minlz-xfast",
                Level::Fastest => "minlz-1",
                Level::Balanced => "minlz-2",
                Level::Smallest => "minlz-3",
                Level::SuperFastSimd => "minlz-simd-xfast",
                Level::FastestSimd => "minlz-simd-1",
                Level::BalancedSimd => "minlz-simd-2",
                Level::SmallestSimd => "minlz-simd-3",
                Level::SuperFastSimdDelta => "minlz-simd-xfast+delta",
                Level::FastestSimdDelta => "minlz-simd-1+delta",
                Level::BalancedSimdDelta => "minlz-simd-2+delta",
                _ => unreachable!("no name for {l:?}"),
            },
            Codec::Snappy => "snappy",
            Codec::Lz4 => "lz4_flex",
            Codec::Gzip1 => "gzip-1",
            Codec::Zstd1 => "zstd-1",
        }
    }

    fn label(self) -> String {
        match self {
            Codec::MinLzMt(_) => format!("{} (MT)", self.name()),
            _ => self.name().into(),
        }
    }

    fn is_minlz(self) -> bool {
        matches!(self, Codec::MinLz(_) | Codec::MinLzMt(_))
    }

    fn is_delta(self) -> bool {
        matches!(
            self,
            Codec::MinLz(l) | Codec::MinLzMt(l)
                if matches!(l, Level::SuperFastSimdDelta | Level::FastestSimdDelta | Level::BalancedSimdDelta)
        )
    }

    // The output Vec is preallocated to src.len() so it never has to grow
    // mid-encode for compressible input.
    fn encode(self, src: &[u8], threads: usize) -> io::Result<Vec<u8>> {
        let out = Vec::with_capacity(src.len());
        match self {
            Codec::MinLz(l) => {
                let mut w = WriterBuilder::new().level(l).build(out)?;
                w.write_all(src)?;
                w.finish()
            }
            Codec::MinLzMt(l) => {
                let mut w = MtWriterBuilder::new()
                    .level(l)
                    .concurrency(threads)
                    .build(out)?;
                w.write_all(src)?;
                w.finish()
            }
            Codec::Snappy => {
                let mut w = snap::write::FrameEncoder::new(out);
                w.write_all(src)?;
                w.into_inner().map_err(|e| io::Error::other(e.to_string()))
            }
            Codec::Lz4 => {
                let mut w = lz4_flex::frame::FrameEncoder::new(out);
                w.write_all(src)?;
                w.finish().map_err(|e| io::Error::other(e.to_string()))
            }
            Codec::Gzip1 => {
                let mut w = flate2::write::GzEncoder::new(out, flate2::Compression::new(1));
                w.write_all(src)?;
                w.finish()
            }
            Codec::Zstd1 => {
                let mut w = zstd::stream::write::Encoder::new(out, 1)?;
                w.write_all(src)?;
                w.finish()
            }
        }
    }

    fn decode_verify(self, enc: &[u8], cap: usize, threads: usize) -> io::Result<Vec<u8>> {
        match self {
            Codec::MinLz(_) => to_vec(Reader::new(enc), cap),
            Codec::MinLzMt(_) => Ok(Reader::new(enc)
                .decode_concurrent(Vec::with_capacity(cap), threads)?
                .writer),
            Codec::Snappy => to_vec(snap::read::FrameDecoder::new(enc), cap),
            Codec::Lz4 => to_vec(lz4_flex::frame::FrameDecoder::new(enc), cap),
            Codec::Gzip1 => to_vec(flate2::read::GzDecoder::new(enc), cap),
            Codec::Zstd1 => to_vec(zstd::stream::read::Decoder::new(enc)?, cap),
        }
    }

    /// Decode to sink: decoder throughput without writing an output buffer,
    /// as `mz bench` reports `dec(st)` / `dec(mt)`.
    fn decode_sink(self, enc: &[u8], threads: usize) -> io::Result<u64> {
        let sink = &mut io::sink();
        match self {
            Codec::MinLz(_) => {
                // fill_buf loop: no memcpy out of the decoder's buffer.
                let mut r = Reader::new(enc);
                let mut acc = 0u64;
                loop {
                    let n = r.fill_buf()?.len();
                    if n == 0 {
                        return Ok(acc);
                    }
                    acc += n as u64;
                    r.consume(n);
                }
            }
            Codec::MinLzMt(_) => Ok(Reader::new(enc)
                .decode_concurrent(io::sink(), threads)?
                .bytes_written),
            Codec::Snappy => io::copy(&mut snap::read::FrameDecoder::new(enc), sink),
            Codec::Lz4 => io::copy(&mut lz4_flex::frame::FrameDecoder::new(enc), sink),
            Codec::Gzip1 => io::copy(&mut flate2::read::GzDecoder::new(enc), sink),
            Codec::Zstd1 => io::copy(&mut zstd::stream::read::Decoder::new(enc)?, sink),
        }
    }
}

fn to_vec<R: Read>(mut r: R, cap: usize) -> io::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(cap);
    r.read_to_end(&mut out)?;
    Ok(out)
}

struct Row {
    file: String,
    codec: Codec,
    threads: usize,
    src_len: usize,
    enc_len: usize,
    enc_mbps: f64,
    dec_mbps: f64,
}

impl Row {
    fn ratio(&self) -> f64 {
        self.src_len as f64 / self.enc_len.max(1) as f64
    }

    fn csv(&self) -> String {
        format!(
            "{},{},{},{},{},{:.3},{:.3}",
            self.file,
            self.codec.label(),
            self.threads,
            self.src_len,
            self.enc_len,
            self.enc_mbps,
            self.dec_mbps
        )
    }

    fn parse(line: &str) -> Option<Row> {
        // The file name may contain commas; the other six fields can't.
        let mut f = line.rsplitn(7, ',');
        let dec_mbps = f.next()?.parse().ok()?;
        let enc_mbps = f.next()?.parse().ok()?;
        let enc_len = f.next()?.parse().ok()?;
        let src_len = f.next()?.parse().ok()?;
        let threads = f.next()?.parse().ok()?;
        let label = f.next()?;
        let codec = Codec::all().find(|c| c.label() == label)?;
        Some(Row {
            file: f.next()?.to_string(),
            codec,
            threads,
            src_len,
            enc_len,
            enc_mbps,
            dec_mbps,
        })
    }
}

fn mb_per_sec(bytes: u64, secs: f64) -> f64 {
    if secs <= 0.0 {
        0.0
    } else {
        (bytes as f64 / 1e6) / secs
    }
}

fn measure(codec: Codec, file: &str, src: &[u8], iters: u32, threads: usize) -> io::Result<Row> {
    let mut best_enc: f64 = 0.0;
    let mut best_dec: f64 = 0.0;
    let mut encoded: Vec<u8> = Vec::new();

    for _ in 0..iters {
        let t0 = Instant::now();
        let buf = codec.encode(src, threads)?;
        let mbps = mb_per_sec(src.len() as u64, t0.elapsed().as_secs_f64());
        if mbps > best_enc {
            best_enc = mbps;
            encoded = buf;
        }
    }

    // Verified once, outside the timed loop; the Vec is dropped before
    // decode timing so its memory isn't held.
    if codec.decode_verify(&encoded, src.len(), threads)? != src {
        return Err(io::Error::other(format!(
            "{}: round-trip mismatch",
            codec.label()
        )));
    }

    for _ in 0..iters {
        let t0 = Instant::now();
        let n = codec.decode_sink(&encoded, threads)?;
        let dt = t0.elapsed().as_secs_f64();
        if n != src.len() as u64 {
            return Err(io::Error::other(format!(
                "{}: sink-decode len {n} != src len {}",
                codec.label(),
                src.len()
            )));
        }
        best_dec = best_dec.max(mb_per_sec(n, dt));
    }

    Ok(Row {
        file: file.to_string(),
        codec,
        threads: if matches!(codec, Codec::MinLzMt(_)) {
            threads
        } else {
            1
        },
        src_len: src.len(),
        enc_len: encoded.len(),
        enc_mbps: best_enc,
        dec_mbps: best_dec,
    })
}

struct Args {
    limit: u64,
    iters: u32,
    csv: Option<PathBuf>,
    mt_threads: usize,
    mt_only: bool,
    report: bool,
    paths: Vec<PathBuf>,
}

const USAGE: &str =
    "usage: cross_codec [--limit SIZE] [--iters N] [--mt-threads N] [--mt-only] [--csv OUT] PATH...
       cross_codec --report CSV...";

fn parse_size(s: &str) -> Option<u64> {
    let s = s.trim();
    let (num, mul): (&str, u64) = match s.as_bytes().last()? {
        b'K' | b'k' => (&s[..s.len() - 1], 1024),
        b'M' | b'm' => (&s[..s.len() - 1], 1024 * 1024),
        b'G' | b'g' => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        _ => (s, 1),
    };
    num.parse::<u64>().ok().map(|n| n * mul)
}

fn parse_args() -> Args {
    let mut a = Args {
        limit: DEFAULT_LIMIT,
        iters: DEFAULT_ITERS,
        csv: None,
        mt_threads: 0,
        mt_only: false,
        report: false,
        paths: Vec::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().unwrap_or_else(|| panic!("{name} needs a value"));
        match arg.as_str() {
            "--limit" => a.limit = parse_size(&val("--limit")).expect("bad --limit"),
            "--iters" => a.iters = val("--iters").parse().expect("bad --iters"),
            "--mt-threads" => a.mt_threads = val("--mt-threads").parse().expect("bad --mt-threads"),
            "--csv" => a.csv = Some(PathBuf::from(val("--csv"))),
            "--mt-only" => a.mt_only = true,
            "--report" => a.report = true,
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                std::process::exit(0);
            }
            _ => a.paths.push(PathBuf::from(arg)),
        }
    }
    if a.paths.is_empty() || (a.mt_only && a.mt_threads == 0) {
        eprintln!("{USAGE}");
        std::process::exit(2);
    }
    a
}

fn collect_files(p: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    let md = fs::metadata(p)?;
    if md.is_dir() {
        let mut entries: Vec<_> = fs::read_dir(p)?.collect::<io::Result<_>>()?;
        entries.sort_by_key(|e| e.path());
        for e in entries {
            collect_files(&e.path(), out)?;
        }
    } else if md.is_file() {
        out.push(p.to_path_buf());
    }
    Ok(())
}

fn load_capped(p: &Path, limit: u64) -> io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(limit.min(fs::metadata(p)?.len()) as usize);
    fs::File::open(p)?.take(limit).read_to_end(&mut buf)?;
    Ok(buf)
}

fn run(args: &Args) -> io::Result<()> {
    let mut files = Vec::new();
    for p in &args.paths {
        collect_files(p, &mut files)?;
    }
    files.sort();
    let mut csv = match &args.csv {
        Some(p) => Some(fs::OpenOptions::new().create(true).append(true).open(p)?),
        None => None,
    };
    let codecs: Vec<Codec> = Codec::all()
        .filter(|c| match c {
            Codec::MinLzMt(_) => args.mt_threads > 0,
            _ => !args.mt_only,
        })
        .collect();
    eprintln!(
        "cross_codec: {} files, iters={}, limit={} bytes, mt_threads={}",
        files.len(),
        args.iters,
        args.limit,
        args.mt_threads
    );
    for f in &files {
        let name = f
            .file_name()
            .map_or_else(|| f.display().to_string(), |s| s.to_string_lossy().into());
        let data = load_capped(f, args.limit)?;
        if data.is_empty() {
            eprintln!("skip {name}: empty");
            continue;
        }
        for &codec in &codecs {
            let r = measure(codec, &name, &data, args.iters, args.mt_threads)?;
            println!(
                "{:<28} {:<28} {:>14} {:>7.2}x {:>10.1} {:>10.1}",
                name,
                codec.label(),
                r.enc_len,
                r.ratio(),
                r.enc_mbps,
                r.dec_mbps
            );
            if let Some(w) = &mut csv {
                writeln!(w, "{}", r.csv())?;
            }
        }
    }
    Ok(())
}

/// Prints a markdown table with padded columns; `right[i]` right-aligns column i.
fn table(head: &[&str], right: &[bool], rows: &[Vec<String>]) {
    let w: Vec<usize> = (0..head.len())
        .map(|i| {
            rows.iter()
                .map(|r| r[i].chars().count())
                .chain([head[i].len()])
                .max()
                .unwrap()
        })
        .collect();
    let line = |cells: &[String]| {
        let c: Vec<String> = cells
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if right[i] {
                    format!("{s:>width$}", width = w[i])
                } else {
                    format!("{s:<width$}", width = w[i])
                }
            })
            .collect();
        println!("| {} |", c.join(" | "));
    };
    line(&head.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    let sep: Vec<String> = (0..head.len())
        .map(|i| {
            if right[i] {
                format!("{}:", "-".repeat(w[i] + 1))
            } else {
                "-".repeat(w[i] + 2)
            }
        })
        .collect();
    println!("|{}|", sep.join("|"));
    for r in rows {
        line(r);
    }
}

fn grouped(n: u64, sep: char) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(sep);
        }
        out.push(c);
    }
    out
}

fn geomean(v: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = v.fold((0.0, 0), |(s, n), x| (s + x.ln(), n + 1));
    (sum / n as f64).exp()
}

fn report(paths: &[PathBuf]) -> io::Result<()> {
    let mut sets = Vec::new();
    for p in paths {
        let rows: Vec<Row> = fs::read_to_string(p)?
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| Row::parse(l).ok_or_else(|| io::Error::other(format!("bad CSV line: {l}"))))
            .collect::<io::Result<_>>()?;
        let mut files: Vec<String> = Vec::new();
        for r in &rows {
            if !files.contains(&r.file) {
                files.push(r.file.clone());
            }
        }
        files.sort();
        sets.push((p, files, rows));
    }

    for (p, files, rows) in &sets {
        println!(
            "#### {} (geomean across {} files)\n",
            p.display(),
            files.len()
        );
        let mut out = Vec::new();
        for c in Codec::all().filter(|c| !c.is_delta()) {
            let rs: Vec<&Row> = rows.iter().filter(|r| r.codec == c).collect();
            if rs.is_empty() {
                continue;
            }
            let label = match c {
                Codec::MinLzMt(_) => format!("{}, MT × {}", c.name(), rs[0].threads),
                _ => c.label(),
            };
            let cells = [
                label,
                format!("{:.2}x", geomean(rs.iter().map(|r| r.ratio()))),
                grouped(geomean(rs.iter().map(|r| r.enc_mbps)).round() as u64, ' '),
                grouped(geomean(rs.iter().map(|r| r.dec_mbps)).round() as u64, ' '),
            ];
            out.push(
                cells
                    .map(|s| if c.is_minlz() { format!("**{s}**") } else { s })
                    .to_vec(),
            );
        }
        table(
            &["Codec", "Avg ratio", "Encode MB/s", "Decode MB/s"],
            &[false, true, true, true],
            &out,
        );
        println!();
    }

    for (_, files, rows) in &sets {
        for f in files {
            let src_len = rows.iter().find(|r| &r.file == f).unwrap().src_len;
            println!("### {f} ({:.2} MiB)\n", src_len as f64 / (1024.0 * 1024.0));
            let out: Vec<Vec<String>> = Codec::all()
                .filter_map(|c| rows.iter().find(|r| &r.file == f && r.codec == c))
                .map(|r| {
                    vec![
                        r.codec.label(),
                        grouped(r.enc_len as u64, ','),
                        format!("{:.2}x", r.ratio()),
                        format!("{:.1}", r.enc_mbps),
                        format!("{:.1}", r.dec_mbps),
                    ]
                })
                .collect();
            table(
                &["codec", "enc size", "ratio", "enc MB/s", "dec MB/s"],
                &[false, true, true, true, true],
                &out,
            );
            println!();
        }
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let args = parse_args();
    if args.report {
        report(&args.paths)
    } else {
        run(&args)
    }
}
