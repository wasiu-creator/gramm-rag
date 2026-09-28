#!/usr/bin/env bash
# Chained after the 24-cell campaign: S5 decomposition on a single clock.
# Waits for the campaign to exit so it never contends for API bandwidth with a latency
# measurement -- the whole point of B running alone.
set -uo pipefail
GB=/media/volume/vol3/grammrag/gramm-rag-graphbracket
cd "$GB"
say(){ echo "S5DECOMP $(date +%H:%M:%S) $*"; }
# campaign already complete (24/24, 0 failures)
say "campaign finished; starting decomposition"
set -a; . /media/volume/vol3/grammrag/gramm-rag-mmlongbench/.env; set +a
: "${OPENAI_API_KEY:?FATAL: no key}"
DR="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e TIKTOKEN_CACHE_DIR=/tmp/tiktoken"
N=${N:-500}
fails=0
# smallest cell first: if the wrapper is not on graphrag's path the assertion fires here,
# for ~$0.03, instead of after four full cells.
say "probe n=20 (wrapper coverage assertion)"
$DR -e GRAPHRAG_API_KEY="$OPENAI_API_KEY" -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -v /media/volume/vol3/grammrag:/work -w /work/gramm-rag-graphbracket \
  grammrag-analysis:latest venvs/s5_graphrag/bin/python s5_decomposition.py \
  --arm s5 --bench mmlongbench --state warm --n 20 > logs/s5_decomp_probe.log 2>&1
if [ $? -ne 0 ]; then
  say "PROBE FAILED -- wrapper not on path; not spending on full cells"; tail -15 logs/s5_decomp_probe.log; exit 1
fi
say "probe ok: $(grep -oE '\-> .*' logs/s5_decomp_probe.log | tail -1)"
grep -oE 'wrapped litellm entry points: .*' logs/s5_decomp_probe.log | tail -1
rm -f tier1/_resource_cells/DECOMP_s5_mmlongbench_warm.json
for bench in mmlongbench docbench; do
  for state in warm cold; do
    say "start s5/$bench/$state"
    log="logs/s5_decomp_${bench}_${state}.log"
    $DR -e GRAPHRAG_API_KEY="$OPENAI_API_KEY" -e OPENAI_API_KEY="$OPENAI_API_KEY" \
      -v /media/volume/vol3/grammrag:/work -w /work/gramm-rag-graphbracket \
      grammrag-analysis:latest venvs/s5_graphrag/bin/python s5_decomposition.py \
      --arm s5 --bench "$bench" --state "$state" --n "$N" > "$log" 2>&1
    if [ $? -ne 0 ]; then say "FAIL s5/$bench/$state"; tail -10 "$log"; fails=$((fails+1)); continue; fi
    say "done  $(grep -oE '\-> .*' "$log" | tail -1)"
  done
done
say "DECOMPOSITION COMPLETE (failures: $fails)"
