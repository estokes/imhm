#!/bin/bash
# ns per op on P-core 2: the difference between the fastest of $runs runs
# at 3r and at r rounds, over the ops in between; the builds take turns.
# usage: wall.sh <u64|str> <n> <get|build|snap> <runs> name=getcount-binary:kind ...
keys=$1; n=$2; op=$3; runs=$4; shift 4
case $op in get) per=$n; budget=20000000;; build) per=$n; budget=4000000;; *) per=100; budget=200000;; esac
r1=$(( budget / per )); [ $r1 -lt 2 ] && r1=2; r2=$(( r1 * 3 ))
t() { local a=$(date +%s%N); taskset -c 2 "$@" >/dev/null; echo $(( $(date +%s%N) - a )); }
declare -A lo hi
for i in $(seq $runs); do
  for v in "$@"; do
    name=${v%%=*}; rest=${v#*=}; bin=${rest%:*}; kind=${rest##*:}
    a=$(t $bin $kind $keys $n $r1 $op); b=$(t $bin $kind $keys $n $r2 $op)
    [ -z "${lo[$name]}" ] || [ $a -lt ${lo[$name]} ] && lo[$name]=$a
    [ -z "${hi[$name]}" ] || [ $b -lt ${hi[$name]} ] && hi[$name]=$b
  done
done
d=$(( (r2 - r1) * per ))
printf "%-4s %-8s %-5s" $keys $n $op
for v in "$@"; do name=${v%%=*}; awk -v x=$(( ${hi[$name]} - ${lo[$name]} )) -v d=$d -v nm=$name 'BEGIN{printf "  %s %7.1f", nm, x/d}'; done; echo
