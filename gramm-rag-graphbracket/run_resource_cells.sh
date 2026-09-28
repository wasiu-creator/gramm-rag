#!/usr/bin/env bash
# B -- serving/resource cells. 3 arms x 2 corpora x 2 states x 2 occurrences = 24 cells.
#
# Occurrence order is a PALINDROME (lightrag,s2,s5 then s5,s2,lightrag), the generalisation
# of A-B-B-A to three arms. All three arms are the same resource class (API-bound), so there
# is no GPU contention to interleave against; the drift this detects is OpenAI-side load
# across the session, which is the only drift source that can actually move these numbers.
# Comparing each arm's two occurrences is the drift test.
set -uo pipefail
GB=/media/volume/vol3/grammrag/gramm-rag-graphbracket
cd "$GB"
set -a; . /media/volume/vol3/grammrag/gramm-rag-mmlongbench/.env; set +a
: "${OPENAI_API_KEY:?FATAL: no key}"
PY=$GB/venvs/lightrag_current/bin/python
DR="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e TIKTOKEN_CACHE_DIR=/tmp/tiktoken"
N=${N:-500}
mkdir -p logs tier1/_resource_cells
say(){ echo "RESCELL $(date +%H:%M:%S) $*"; }

cell(){ # arm bench state occ
  local arm=$1 bench=$2 state=$3 occ=$4
  local out="tier1/_resource_cells/${arm}_${bench}_${state}_occ${occ}.json"
  if [ -f "$out" ]; then say "skip (present) $arm/$bench/$state occ$occ"; return 0; fi
  say "start $arm/$bench/$state occ$occ"
  local log="logs/rescell_${arm}_${bench}_${state}_occ${occ}.log"
  if [ "$arm" = "s5" ]; then
    $DR -e GRAPHRAG_API_KEY="$OPENAI_API_KEY" -e OPENAI_API_KEY="$OPENAI_API_KEY" \
      -v /media/volume/vol3/grammrag:/work -w /work/gramm-rag-graphbracket \
      grammrag-analysis:latest venvs/s5_graphrag/bin/python resource_cells.py \
      --arm "$arm" --bench "$bench" --state "$state" --occurrence "$occ" --n "$N" \
      > "$log" 2>&1
  else
    $PY resource_cells.py --arm "$arm" --bench "$bench" --state "$state" \
      --occurrence "$occ" --n "$N" > "$log" 2>&1
  fi
  local rc=$?
  if [ $rc -ne 0 ] || [ ! -f "$out" ]; then
    say "FAIL $arm/$bench/$state occ$occ rc=$rc"; tail -12 "$log"; return 1
  fi
  say "done  $(grep -oE '\-> .*' "$log" | tail -1)"
}

fails=0
for occ in 1 2; do
  if [ "$occ" = "1" ]; then order="lightrag s2 s5"; else order="s5 s2 lightrag"; fi
  say "===== occurrence $occ : order [$order] ====="
  for arm in $order; do
    for bench in mmlongbench docbench; do
      for state in warm cold; do
        cell "$arm" "$bench" "$state" "$occ" || fails=$((fails+1))
      done
    done
  done
done
say "ALL CELLS COMPLETE (failures: $fails)"
ls tier1/_resource_cells/*_occ[12].json | wc -l | xargs say "cells on disk:"
