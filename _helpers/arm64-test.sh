#!/bin/bash
# Runs the minlz lib tests on aarch64 (NEON) under QEMU in docker, from Windows (Git Bash) or Linux.
#   _helpers/arm64-test.sh [libtest args, e.g. a test filter]
#
# The crate has no runtime dependencies, so rustc builds the lib tests directly: cargo would also
# build the dev-dependencies, and ring (via ureq) needs an aarch64 C compiler. The lib tests' own
# dev-dependencies (pure Rust) come from _helpers/arm64-deps. The musl target links a static binary
# with rust-lld and its own crt/libc, so no cross toolchain is needed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# Native tools get the Windows form; MSYS_NO_PATHCONV below stops Git Bash converting paths.
ROOT=$(cygpath -m "$ROOT" 2>/dev/null || echo "$ROOT")
TARGET=aarch64-unknown-linux-musl
BIN=target/$TARGET/arm64-test/minlz-test
rustup target list --installed | grep -qx $TARGET || rustup target add $TARGET
mkdir -p "$ROOT/$(dirname $BIN)"
cargo build -q --release --target $TARGET --manifest-path "$ROOT/_helpers/arm64-deps/Cargo.toml" \
	--target-dir "$ROOT/target/arm64-deps"
DEPS=$ROOT/target/arm64-deps/$TARGET/release/deps
# CARGO_MANIFEST_DIR is where the tests look for testdata: the mount below. Git Bash would turn it
# into a Windows path for the native rustc.
MSYS_NO_PATHCONV=1 CARGO_MANIFEST_DIR=/src/crates/minlz rustc --test --edition 2024 --crate-name minlz \
	--target $TARGET -C linker=rust-lld -C opt-level=2 -C debug-assertions -C overflow-checks \
	--extern flate2="$(ls -t "$DEPS"/libflate2-*.rlib | head -1)" -L dependency="$DEPS" \
	-o "$ROOT/$BIN" "$ROOT/crates/minlz/src/lib.rs"
# The binfmt handlers can drop when docker restarts; re-registering is cheap.
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes >/dev/null
# --init: as PID 1 the test would ignore the fatal signal qemu re-raises when the guest faults,
# and hang forever holding the binary open (rustc then cannot replace it on Windows). No core
# files: qemu would write them into the mounted crate directory.
MSYS_NO_PATHCONV=1 docker run --rm --init --ulimit core=0 --platform linux/arm64 -v "$ROOT:/src" -w /src/crates/minlz \
	arm64v8/alpine "/src/$BIN" "$@"
