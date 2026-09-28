#!/usr/bin/env bash
# D2 leg 1 -- index the full P3 subsample (175 docs) on the three API graph arms.
# The 50 projection docs are already stamped and resume-skip, so this adds ~125 each.
# Per-arm interpreters are pinned: RAG-Anything 1.2.8 lives in its own venv (rule 14) --
# running it under the LightRAG interpreter is what failed D1's first attempt.
set -uo pipefail
GB=/media/volume/vol3/grammrag/gramm-rag-graphbracket
cd "$GB"
set -a; . /media/volume/vol3/grammrag/gramm-rag-mmlongbench/.env; set +a
: "${OPENAI_API_KEY:?FATAL: no key}"
PY_LR=$GB/venvs/lightrag_current/bin/python
PY_RA=$GB/venvs/raganything_legacy/bin/python
DR="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e TIKTOKEN_CACHE_DIR=/tmp/tiktoken"
mkdir -p logs
say(){ echo "P3FULL $(date +%H:%M:%S) $*"; }
fails=0

say "=== 1/3 LightRAG, 175 docs ==="
$PY_LR index_tier1.py --bench mpdocvqa --corpus data/corpus_sizes_p3.json \
  --outdir p3_lightrag_full --budget-ceiling 25 > logs/p3_full_lightrag.log 2>&1
rc=$?; say "lightrag rc=$rc $(grep -oE 'INDEX COMPLETE.*' logs/p3_full_lightrag.log | tail -1)"
[ $rc -ne 0 ] && { fails=$((fails+1)); grep -vE '^INFO' logs/p3_full_lightrag.log | tail -8; }

say "=== 2/3 RAG-Anything, 175 docs ==="
$PY_RA index_s2_raganything.py --corpus data/corpus_sizes_p3.json \
  --outdir p3_s2_full --budget-ceiling 25 > logs/p3_full_s2.log 2>&1
rc=$?; say "s2 rc=$rc"
[ $rc -ne 0 ] && { fails=$((fails+1)); grep -vE '^INFO' logs/p3_full_s2.log | tail -8; }

say "=== 3/3 MS GraphRAG, 175 docs ==="
$DR -e GRAPHRAG_API_KEY="$OPENAI_API_KEY" -e OPENAI_API_KEY="$OPENAI_API_KEY" -e S5_OUTDIR=p3_s5_full \
  -v /media/volume/vol3/grammrag:/work -w /work/gramm-rag-graphbracket \
  grammrag-analysis:latest venvs/s5_graphrag/bin/python index_s5_graphrag.py \
  --docs data/p3_docs_full.json --tag p3full --ceiling 25 > logs/p3_full_s5.log 2>&1
rc=$?; say "s5 rc=$rc"
[ $rc -ne 0 ] && { fails=$((fails+1)); grep -vE '^INFO' logs/p3_full_s5.log | tail -8; }

say "INDEXING COMPLETE (failures: $fails)"
for d in p3_lightrag_full p3_s2_full p3_s5_full; do
  say "  $d: $(ls tier1/$d/stores 2>/dev/null | wc -l) stores"
done
