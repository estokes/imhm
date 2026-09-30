#!/bin/bash
# instructions, cycles and branch misses per op on P-core 2: the difference
# between counts at 3r and at r rounds.
# usage: count.sh <getcount-binary> <kind> <u64|str> <n> <get|build|snap>
bin=$1; kind=$2; keys=$3; n=$4; op=$5
case $op in get) per=$n; budget=20000000;; build) per=$n; budget=4000000;; *) per=100; budget=200000;; esac
r1=$(( budget / per )); [ $r1 -lt 2 ] && r1=2; r2=$(( r1 * 3 ))
stat() { taskset -c 2 perf stat -x, -e cpu_core/instructions/u,cpu_core/cycles/u,cpu_core/branch-misses/u $bin $kind $keys $n $1 $op 2>&1 >/dev/null | awk -F, '{print $1}' | tr '\n' ' '; }
read i1 c1 b1 <<<"$(stat $r1)"; read i2 c2 b2 <<<"$(stat $r2)"
d=$(( (r2 - r1) * per ))
awk -v i=$((i2-i1)) -v c=$((c2-c1)) -v b=$((b2-b1)) -v d=$d 'BEGIN{printf "%7.0f ins %7.0f cyc %5.2f bm\n", i/d, c/d, b/d}'
