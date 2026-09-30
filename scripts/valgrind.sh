#!/bin/bash
# Every test in its own valgrind process, 12 at a time; definite and
# indirect leaks are errors. The optimized build with debug assertions,
# so model runs are a tenth of full size. Prints each test's exit code.
cd "$(dirname "$0")/.."
export CARGO_TARGET_DIR=$HOME/tmp/target/imhm-vg CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS=true CARGO_PROFILE_RELEASE_DEBUG=line-tables-only
BIN=$(cargo test -q --release --no-run --message-format=json 2>/dev/null | python3 -c 'import sys, json
for l in sys.stdin:
    m = json.loads(l)
    if m.get("reason") == "compiler-artifact" and m["target"]["kind"] == ["lib"] and m.get("executable"):
        print(m["executable"])')
OUT=$CARGO_TARGET_DIR/valgrind
rm -rf "$OUT"; mkdir -p "$OUT"
"$BIN" --list --format terse | sed 's/: test$//' | xargs -P 12 -I{} sh -c "valgrind --leak-check=full --errors-for-leak-kinds=definite,indirect --error-exitcode=99 --quiet $BIN --exact {} --test-threads=1 > $OUT/{}.log 2>&1; echo \"\$? {}\"" | sort
echo "logs in $OUT"
