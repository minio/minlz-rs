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

//! Round-trip tests: drive the `mz` binary via `std::process::Command`,
//! compress + decompress each level, assert byte-equality.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

fn mz_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_mz"))
}

// PID + nanos isn't collision-proof: macOS' SystemTime::now() bottoms
// out at ~microsecond resolution, so two parallel #[test]s in the same
// binary can land on identical paths and clobber each other's files
// (root cause of the flaky `block_round_trip` byte-mismatch on macOS
// CI).  Append a process-local atomic counter to guarantee uniqueness.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_dir() -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "mz_test_{}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&p).unwrap();
    p
}

fn run_ok(cmd: &mut Command) {
    let out = cmd.output().expect("spawn mz");
    assert!(
        out.status.success(),
        "mz failed: status={} stderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}

fn sample_payload(seed: u64, size: usize) -> Vec<u8> {
    // Simple repeating-pattern + counter to get a compressible mix.
    let mut v = Vec::with_capacity(size);
    let mut x = seed;
    while v.len() < size {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(0x1234);
        let chunk = format!("line {x:016x}: {}\n", "the quick brown fox ".repeat(3));
        v.extend_from_slice(chunk.as_bytes());
    }
    v.truncate(size);
    v
}

#[test]
fn round_trip_each_level() {
    let dir = tmp_dir();
    let src = dir.join("input.txt");
    let payload = sample_payload(42, 64 * 1024);
    fs::write(&src, &payload).unwrap();
    for level in ["-1", "-2", "-3"] {
        let mz_file = dir.join(format!("input.txt{}", ".mz"));
        // Compress.
        run_ok(Command::new(mz_bin()).args(["c", level, "-q"]).arg(&src));
        // Decompress to a separate output (avoid clobbering source).
        let out = dir.join(format!("decoded.{level}.bin"));
        run_ok(
            Command::new(mz_bin())
                .args(["d", "-q"])
                .arg("-o")
                .arg(&out)
                .arg(&mz_file),
        );
        let got = fs::read(&out).unwrap();
        assert_eq!(got, payload, "round-trip mismatch at level {level}");
        fs::remove_file(&mz_file).unwrap();
    }
}

#[test]
fn block_round_trip() {
    let dir = tmp_dir();
    let src = dir.join("input.txt");
    let payload = sample_payload(7, 32 * 1024);
    fs::write(&src, &payload).unwrap();
    run_ok(
        Command::new(mz_bin())
            .args(["c", "--block", "-q"])
            .arg(&src),
    );
    let mzb = dir.join("input.txt.mzb");
    let out = dir.join("decoded.bin");
    run_ok(
        Command::new(mz_bin())
            .args(["d", "-q"])
            .arg("-o")
            .arg(&out)
            .arg(&mzb),
    );
    let got = fs::read(out).unwrap();
    assert_eq!(got, payload);
}

#[test]
fn verify_subcommand_succeeds_on_valid_mz() {
    let dir = tmp_dir();
    let src = dir.join("input.txt");
    let payload = sample_payload(11, 8 * 1024);
    fs::write(&src, &payload).unwrap();
    run_ok(Command::new(mz_bin()).args(["c", "-q"]).arg(&src));
    let mz_file = dir.join("input.txt.mz");
    run_ok(Command::new(mz_bin()).arg("verify").arg(&mz_file));
}

#[test]
fn verify_subcommand_fails_on_garbage() {
    let dir = tmp_dir();
    let mz_file = dir.join("garbage.mz");
    fs::write(&mz_file, b"\x00\x01\x02not a real mz stream").unwrap();
    let out = Command::new(mz_bin())
        .arg("verify")
        .arg(&mz_file)
        .output()
        .unwrap();
    assert!(!out.status.success(), "verify should fail on garbage");
}

#[test]
fn stdin_stdout_pipe() {
    let dir = tmp_dir();
    let src = dir.join("input.txt");
    let payload = sample_payload(99, 16 * 1024);
    fs::write(&src, &payload).unwrap();

    // mz c -c - < input.txt > compressed
    let mut child = Command::new(mz_bin())
        .args(["c", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    child.stdin.as_mut().unwrap().write_all(&payload).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "compress via stdin failed");
    let compressed = out.stdout;

    // mz d -c - < compressed > out
    let mut child = Command::new(mz_bin())
        .args(["d", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&compressed)
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "decompress via stdin failed");
    assert_eq!(out.stdout, payload);
}

#[test]
fn auto_detect_compress_from_extension() {
    let dir = tmp_dir();
    let src = dir.join("input.txt");
    let payload = sample_payload(3, 4 * 1024);
    fs::write(&src, &payload).unwrap();
    // No subcommand — extension is plain .txt so should compress.
    run_ok(Command::new(mz_bin()).args(["-1", "-q"]).arg(&src));
    assert!(dir.join("input.txt.mz").exists());
}

#[test]
fn auto_detect_decompress_from_extension() {
    let dir = tmp_dir();
    let src = dir.join("input.txt");
    let payload = sample_payload(5, 4 * 1024);
    fs::write(&src, &payload).unwrap();
    run_ok(Command::new(mz_bin()).args(["c", "-q"]).arg(&src));
    let mz_file = dir.join("input.txt.mz");
    let out = dir.join("input.out");
    // No subcommand — .mz extension implies decompress.
    run_ok(
        Command::new(mz_bin())
            .args(["-q"])
            .arg("-o")
            .arg(&out)
            .arg(&mz_file),
    );
    assert_eq!(fs::read(out).unwrap(), payload);
}

/// Runs `mz` with `input` on stdin and returns stdout.
fn mz_pipe(bin: &std::path::Path, args: &[&str], input: &[u8]) -> Vec<u8> {
    let mut child = Command::new(bin)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let data = input.to_vec();
    let feed = std::thread::spawn(move || stdin.write_all(&data));
    let out = child.wait_with_output().unwrap();
    feed.join().unwrap().unwrap();
    assert!(
        out.status.success(),
        "{} {args:?}: {}",
        bin.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// The stream identifier's flag byte has bit 6 set when the stream may hold SIMD chunks.
fn simd_flag(stream: &[u8]) -> bool {
    stream[9] & 1 << 6 != 0
}

/// Counts the 0x10 (SIMD) chunks of a stream.
fn simd_chunks(stream: &[u8]) -> usize {
    let (mut i, mut n) = (0, 0);
    while i + 4 <= stream.len() {
        n += usize::from(stream[i] == 0x10);
        i += 4
            + (usize::from(stream[i + 1])
                | usize::from(stream[i + 2]) << 8
                | usize::from(stream[i + 3]) << 16);
    }
    n
}

/// Reports whether `block` is a SIMD block: block type 1 in bits 24+ of the uvarint after the
/// zero byte.
fn simd_block(block: &[u8]) -> bool {
    let mut v = 0u64;
    for (i, &b) in block.iter().enumerate().skip(1).take(9) {
        v |= u64::from(b & 0x7f) << (7 * (i - 1));
        if b < 0x80 {
            return block[0] == 0 && v >> 24 == 1;
        }
    }
    false
}

/// Go mz's compress level flags and the level they write.
const LEVEL_FLAGS: &[(&[&str], minlz::Level)] = {
    use minlz::Level::*;
    &[
        (&[], BalancedSimd),
        (&["-xfast"], SuperFastSimd),
        (&["-1"], FastestSimd),
        (&["-2"], BalancedSimd),
        (&["-3"], SmallestSimd),
        (&["-xfast", "--simd.delta"], SuperFastSimdDelta),
        (&["-1", "--simd.delta"], FastestSimdDelta),
        (&["--simd.delta"], BalancedSimdDelta),
        (&["-xfast", "--classic"], SuperFast),
        (&["-1", "--classic"], Fastest),
        (&["--classic"], Balanced),
        (&["-3", "--classic"], Smallest),
    ]
};

/// Compress writes SIMD blocks unless `--classic`, at the levels Go's mz maps its flags to:
/// streams (one thread) and blocks equal the library's output at that level and round-trip.
#[test]
fn simd_level_flags() {
    use std::io::Write as _;
    let payload = sample_payload(17, 300 << 10);
    for &(flags, level) in LEVEL_FLAGS {
        let classic = flags.contains(&"--classic");
        let args = [
            &["c", "-c", "-q", "--cpu", "1", "--block-size", "64K"],
            flags,
            &["-"],
        ]
        .concat();
        let stream = mz_pipe(&mz_bin(), &args, &payload);
        let mut want = Vec::new();
        let mut w = minlz::stream::WriterBuilder::new()
            .level(level)
            .block_size(64 << 10)
            .append_index()
            .build(&mut want)
            .unwrap();
        w.write_all(&payload).unwrap();
        w.finish().unwrap();
        assert!(stream == want, "{flags:?}: stream differs from {level:?}");
        assert_eq!(simd_flag(&stream), !classic, "{flags:?}");
        if classic {
            assert_eq!(simd_chunks(&stream), 0, "{flags:?}");
        }
        for cpu in ["1", "4"] {
            let got = mz_pipe(&mz_bin(), &["d", "-c", "-q", "--cpu", cpu, "-"], &stream);
            assert!(got == payload, "{flags:?}: stream round trip, cpu {cpu}");
        }

        let args = [&["c", "--block", "-c", "-q"], flags, &["-"]].concat();
        let block = mz_pipe(&mz_bin(), &args, &payload);
        let mut want = Vec::new();
        minlz::encode(&mut want, &payload, level).unwrap();
        assert!(block == want, "{flags:?}: block differs from {level:?}");
        assert!(!classic || !simd_block(&block), "{flags:?}");
        let got = mz_pipe(&mz_bin(), &["d", "--block", "-c", "-q", "-"], &block);
        assert!(got == payload, "{flags:?}: block round trip");
    }
    let stream = mz_pipe(&mz_bin(), &["c", "-c", "-q", "-1", "-"], &payload);
    assert!(simd_chunks(&stream) > 0);
    let block = mz_pipe(
        &mz_bin(),
        &["c", "--block", "-c", "-q", "-1", "-"],
        &payload,
    );
    assert!(simd_block(&block));
}

#[test]
fn simd_flag_conflicts_fail() {
    for flags in [
        &["--simd.delta", "--classic"][..],
        &["--simd.delta", "-0"],
        &["--simd.delta", "-3"],
    ] {
        let out = Command::new(mz_bin())
            .args([&["c", "-c", "-q"], flags, &["-"]].concat())
            .stdin(Stdio::null())
            .output()
            .unwrap();
        assert!(!out.status.success(), "{flags:?} should fail");
    }
}

/// The hidden CPU toggles only pick lower tiers, so output and decoding stay the same.
#[test]
fn cpu_toggles() {
    let payload = sample_payload(23, 200 << 10);
    let want = mz_pipe(&mz_bin(), &["c", "-c", "-q", "-1", "-"], &payload);
    for off in [&["-avx512=false"][..], &["-avx2=false", "--neon=false"]] {
        let got = mz_pipe(
            &mz_bin(),
            &[off, &["c", "-c", "-q", "-1", "-"]].concat(),
            &payload,
        );
        assert!(got == want, "{off:?}: output differs");
        let dec = mz_pipe(&mz_bin(), &[off, &["d", "-c", "-q", "-"]].concat(), &got);
        assert!(dec == payload, "{off:?}: round trip");
    }
}

/// Cross-impl: Rust mz and a Go `cmd/mz` of the simd-blocks branch (`MINLZ_GO_CMD`) decode
/// each other's streams (one thread, 64 KiB blocks) and blocks, by default (SIMD) and with
/// `--classic`. Once Rust writes SIMD chunks at a level, its output equals Go's. Classic
/// output is not compared: it matches Go only when Go is built with `-tags=noasm`.
#[test]
fn go_mz_cross() {
    let Some(go) = std::env::var_os("MINLZ_GO_CMD").map(PathBuf::from) else {
        return;
    };
    let dir = tmp_dir();
    let src = dir.join("input.bin");
    let payload = sample_payload(29, 300 << 10);
    fs::write(&src, &payload).unwrap();
    for &(flags, _) in LEVEL_FLAGS {
        for block in [false, true] {
            // Go's flag package takes one or two dashes, so both get the same flags, except
            // for the block size.
            let (mode, bs, ext): (&[&str], &[&str], _) = if block {
                (&["-block"], &[], "mzb")
            } else {
                (&["-cpu=1"], &["-bs=64K", "--block-size=64K"], "mz")
            };
            let (rust, gos) = (dir.join(format!("r.{ext}")), dir.join(format!("g.{ext}")));
            for (bin, bs, out) in [(mz_bin(), bs.get(1), &rust), (go.clone(), bs.first(), &gos)] {
                run_ok(
                    Command::new(bin)
                        .args([&["c", "-q"], mode, flags].concat())
                        .args(bs)
                        .arg("-o")
                        .arg(out)
                        .arg(&src),
                );
            }
            let dec = dir.join("decoded.bin");
            for (bin, file) in [(&go, &rust), (&mz_bin(), &gos)] {
                run_ok(
                    Command::new(bin)
                        .args([&["d", "-q"], mode].concat())
                        .arg("-o")
                        .arg(&dec)
                        .arg(file),
                );
                assert!(
                    fs::read(&dec).unwrap() == payload,
                    "{flags:?} block={block}: {} d {}",
                    bin.display(),
                    file.display()
                );
            }
            let (rust, gos) = (fs::read(&rust).unwrap(), fs::read(&gos).unwrap());
            if flags.contains(&"--classic") {
                continue;
            }
            let rust_simd = if block {
                simd_block(&rust)
            } else {
                assert_eq!(simd_flag(&rust), simd_flag(&gos), "{flags:?}");
                simd_chunks(&rust) > 0
            };
            assert!(
                !rust_simd || rust == gos,
                "{flags:?} block={block}: Rust output differs from Go's"
            );
        }
    }
}

#[test]
fn output_same_as_input_is_rejected() {
    let dir = tmp_dir();
    let payload = sample_payload(11, 64 * 1024);
    for args in [
        &["c", "-q", "--cpu", "1"][..],
        &["c", "-q", "--cpu", "2"],
        &["c", "-q", "--block", "--rm"],
        &["d", "-q", "--cpu", "1"],
        &["d", "-q", "--cpu", "2"],
    ] {
        let src = dir.join("same.txt");
        fs::write(&src, &payload).unwrap();
        let alias = dir.join(".").join("same.txt");
        let out = Command::new(mz_bin())
            .args(args)
            .arg("-o")
            .arg(&alias)
            .arg(&src)
            .output()
            .unwrap();
        assert!(!out.status.success(), "{args:?}: accepted -o == input");
        assert_eq!(fs::read(&src).unwrap(), payload, "{args:?}: input modified");
    }
}

// Only Unix: std has no stable file identity on Windows, so mz cannot see hard links there.
#[cfg(unix)]
#[test]
fn output_hard_link_to_input_is_rejected() {
    let dir = tmp_dir();
    let payload = sample_payload(12, 64 * 1024);
    for args in [
        &["c", "-q", "--cpu", "1"][..],
        &["c", "-q", "--cpu", "2"],
        &["c", "-q", "--block", "--rm"],
        &["d", "-q", "--cpu", "1"],
        &["d", "-q", "--cpu", "2"],
    ] {
        let src = dir.join("hl-src.txt");
        let link = dir.join("hl-dst.txt");
        let _ = fs::remove_file(&link);
        fs::write(&src, &payload).unwrap();
        fs::hard_link(&src, &link).unwrap();
        let out = Command::new(mz_bin())
            .args(args)
            .arg("-o")
            .arg(&link)
            .arg(&src)
            .output()
            .unwrap();
        assert!(
            !out.status.success(),
            "{args:?}: accepted -o == hard link of input"
        );
        assert_eq!(fs::read(&src).unwrap(), payload, "{args:?}: input modified");
    }
}

/// `-0 --block` writes Go's uncompressed block (SPEC 1.1: size 0 means the rest is literals):
/// `0x00` when empty, else `0x00 0x00` and the input.
#[test]
fn block_level_zero_is_uncompressed() {
    let payload = sample_payload(9, 78_000);
    for input in [&payload[..], &[]] {
        let block = mz_pipe(&mz_bin(), &["c", "--block", "-0", "-c", "-q", "-"], input);
        let want = if input.is_empty() {
            vec![0]
        } else {
            [&[0, 0], input].concat()
        };
        assert!(
            block == want,
            "len {}: got {} bytes",
            input.len(),
            block.len()
        );
        let got = mz_pipe(&mz_bin(), &["d", "--block", "-c", "-q", "-"], &block);
        assert!(got == input, "len {}: round trip", input.len());
    }
}

#[test]
fn mt_stdout_summary_counts_output() {
    let dir = tmp_dir();
    let src = dir.join("in.txt");
    fs::write(&src, sample_payload(7, 100_000)).unwrap();
    let out = Command::new(mz_bin())
        .args(["c", "-c", "--threads", "2"])
        .arg(&src)
        .output()
        .unwrap();
    assert!(out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains(&format!("-> {} [", out.stdout.len())), "{err}");
}

#[test]
fn block_stdin_oversize_fails_without_buffering_all() {
    let mut child = Command::new(mz_bin())
        .args(["c", "--block", "-c", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn mz");
    let mut stdin = child.stdin.take().unwrap();
    let chunk = vec![0u8; 1 << 20];
    let mut written = 0usize;
    while written < 128 << 20 && stdin.write_all(&chunk).is_ok() {
        written += chunk.len();
    }
    drop(stdin);
    assert!(!child.wait().unwrap().success());
    assert!(
        written < 32 << 20,
        "mz consumed {written} bytes before rejecting"
    );
}

#[test]
fn block_decompress_stdin_oversize_fails_without_buffering_all() {
    let mut child = Command::new(mz_bin())
        .args(["d", "--block", "-c", "-q", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn mz");
    let mut stdin = child.stdin.take().unwrap();
    // Zeros form a valid literal block (0x00, size 0), so only a read bound can stop it.
    let chunk = vec![0u8; 1 << 20];
    let mut written = 0usize;
    while written < 128 << 20 && stdin.write_all(&chunk).is_ok() {
        written += chunk.len();
    }
    drop(stdin);
    assert!(!child.wait().unwrap().success());
    assert!(
        written < 32 << 20,
        "mz consumed {written} bytes before rejecting"
    );
}

#[test]
fn compress_stream_verify() {
    let dir = tmp_dir();
    let src = dir.join("input.txt");
    fs::write(&src, sample_payload(5, 64 * 1024)).unwrap();
    for threads in ["1", "4"] {
        let out = Command::new(mz_bin())
            .args(["c", "--verify", "--threads", threads, "-o"])
            .arg(dir.join("out.mz"))
            .arg(&src)
            .output()
            .unwrap();
        assert!(out.status.success(), "threads={threads}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains("Verified ok"),
            "threads={threads}: {stderr}"
        );
    }
}
