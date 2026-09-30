#!/bin/bash
# Every test under Miri in its own process, $JOBS at a time (default: all
# cores). Prints each test's exit code and name as it finishes; logs go
# to $CARGO_TARGET_DIR/miri-logs. Miri runs a process on one core, so
# this is far faster than one `cargo miri test`. Pin it with taskset to
# keep cores free, e.g. `taskset -c 4-15 scripts/miri.sh`.
cd "$(dirname "$0")/.."
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$HOME/tmp/target/miri}
JOBS=${JOBS:-$(nproc)}
LOGS=$CARGO_TARGET_DIR/miri-logs
rm -rf "$LOGS"; mkdir -p "$LOGS"
cargo +nightly miri test -q --no-run || exit 1
cargo +nightly miri test -q -- --list --format terse 2>/dev/null | sed -n 's/: test$//p' |
  xargs -P "$JOBS" -I{} sh -c "cargo +nightly miri test -q -- --exact {} > '$LOGS/{}.log' 2>&1; echo \"\$? {}\""
