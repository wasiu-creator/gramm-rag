#!/usr/bin/env bash
# D1 -- P3 projection: index the stratified 50-doc set on the three API graph arms,
# measure, and project to the full 182-doc subsample BEFORE committing to it.
# Chained end-to-end so no stage waits on a human.
set -uo pipefail
GB=/media/volume/vol3/grammrag/gramm-rag-graphbracket
cd "$GB"
set -a; . /media/volume/vol3/grammrag/gramm-rag-mmlongbench/.env; set +a
: "${OPENAI_API_KEY:?FATAL: no key}"
PY=$GB/venvs/lightrag_current/bin/python
DR="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e TIKTOKEN_CACHE_DIR=/tmp/tiktoken"
mkdir -p logs tier1/p3_mpdocvqa
say(){ echo "P3PROJ $(date +%H:%M:%S) $*"; }

say "=== 1/3 LightRAG, 50 docs ==="
$PY index_tier1.py --bench mpdocvqa --corpus data/corpus_sizes_p3_proj50.json \
  --outdir p3_lightrag_proj --budget-ceiling 15 > logs/p3_proj_lightrag.log 2>&1
rc=$?; say "lightrag rc=$rc $(grep -oE 'docs, .*chars' logs/p3_proj_lightrag.log | tail -1)"
[ $rc -ne 0 ] && { say "FAIL lightrag"; tail -15 logs/p3_proj_lightrag.log; }

say "=== 2/3 RAG-Anything, 50 docs ==="
$GB/venvs/raganything_legacy/bin/python index_s2_raganything.py --corpus data/corpus_sizes_p3_proj50.json \
  --outdir p3_s2_proj --budget-ceiling 15 > logs/p3_proj_s2.log 2>&1
rc=$?; say "s2 rc=$rc"
[ $rc -ne 0 ] && { say "FAIL s2"; tail -15 logs/p3_proj_s2.log; }

say "=== 3/3 MS GraphRAG, 50 docs ==="
$DR -e GRAPHRAG_API_KEY="$OPENAI_API_KEY" -e OPENAI_API_KEY="$OPENAI_API_KEY" -e S5_OUTDIR=p3_s5_proj \
  -v /media/volume/vol3/grammrag:/work -w /work/gramm-rag-graphbracket \
  grammrag-analysis:latest venvs/s5_graphrag/bin/python index_s5_graphrag.py \
  --docs data/p3_docs_proj50.json --tag p3proj --ceiling 15 > logs/p3_proj_s5.log 2>&1
rc=$?; say "s5 rc=$rc"
[ $rc -ne 0 ] && { say "FAIL s5"; tail -15 logs/p3_proj_s5.log; }

say "=== projection ==="
$DR -v /media/volume/vol3/grammrag:/work -w /work/gramm-rag-graphbracket \
  grammrag-analysis:latest python p3_project.py > logs/p3_projection.log 2>&1
prc=$?
cat logs/p3_projection.log
if [ $prc -ne 0 ]; then say "PROJECTION REFUSED (rc=$prc) -- no verdict, D2 stays gated"; exit $prc; fi
say "PROJECTION COMPLETE"
