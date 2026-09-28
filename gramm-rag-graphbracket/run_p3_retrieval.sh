#!/usr/bin/env bash
# D2b leg 1 -- retrieval for the three API graph arms. Secrets come from --env-file so they
# never appear in a process listing (the mechanism that exposed the key at 19:35).
set -uo pipefail
GB=/media/volume/vol3/grammrag/gramm-rag-graphbracket
cd "$GB"
set -a; . /media/volume/vol3/grammrag/gramm-rag-mmlongbench/.env; set +a
: "${OPENAI_API_KEY:?FATAL: no key}"
DR="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e TIKTOKEN_CACHE_DIR=/tmp/tiktoken --env-file /media/volume/vol3/grammrag/.secrets/p3.env"
say(){ echo "P3RET $(date +%H:%M:%S) $*"; }
fails=0
for arm in lightrag s2 s5; do
  out="tier1/p3_mpdocvqa/p3_retrieval_${arm}.json"
  [ -f "$out" ] && { say "skip $arm (present)"; continue; }
  say "=== $arm ==="
  if [ "$arm" = "s5" ]; then
    $DR -v /media/volume/vol3/grammrag:/work -w /work/gramm-rag-graphbracket \
      grammrag-analysis:latest venvs/s5_graphrag/bin/python retrieve_p3.py --arm $arm \
      > logs/p3_ret_${arm}.log 2>&1
  else
    PY=$GB/venvs/lightrag_current/bin/python
    [ "$arm" = "s2" ] && PY=$GB/venvs/raganything_legacy/bin/python
    $PY retrieve_p3.py --arm $arm > logs/p3_ret_${arm}.log 2>&1
  fi
  rc=$?
  if [ $rc -ne 0 ]; then say "FAIL $arm rc=$rc"; grep -vE '^INFO|^WARNING' logs/p3_ret_${arm}.log | tail -8; fails=$((fails+1)); continue; fi
  say "done $(grep -oE "${arm}: page-hit.*" logs/p3_ret_${arm}.log | tail -1)"
done
say "RETRIEVAL COMPLETE (failures: $fails)"
