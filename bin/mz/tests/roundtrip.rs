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
