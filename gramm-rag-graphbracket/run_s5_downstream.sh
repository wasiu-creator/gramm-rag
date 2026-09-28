#!/usr/bin/env bash
# S5 downstream: wait for indexing, verify, retrieve -> stage -> generate -> score, both corpora.
# Fails fast at every stage; never emits a partial artefact under a canonical name.
set -euo pipefail
GB=/media/volume/vol3/grammrag/gramm-rag-graphbracket
CP=/media/volume/vol3/grammrag/gramm-rag-colpali
IMG_S5=grammrag-analysis:latest
IMG_GEN=grammrag-colpali-pinned:latest
cd "$GB"
set -a; . /media/volume/vol3/grammrag/gramm-rag-mmlongbench/.env; set +a
: "${OPENAI_API_KEY:?FATAL: no key}"
say(){ echo "S5DOWN $(date +%H:%M:%S) $*"; }
DR="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e TIKTOKEN_CACHE_DIR=/tmp/tiktoken"

# ---- 1. wait for the index ----
while docker ps --filter name=s5-full -q | grep -q .; do sleep 60; done
say "indexing container gone; verifying"
python3 - <<'PY'
import json
from pathlib import Path
OUT = Path('/media/volume/vol3/grammrag/gramm-rag-graphbracket/tier1/s5_graphrag')
log = json.loads((OUT / 'index_log_full.json').read_text())
docs = log['docs']
bad = [d['stem'] for d in docs if d['alignment'] < 0.95 or d['workflow_errors'] or d['missing_tables']]
assert len(docs) == 364, f"{len(docs)} documents in the log, expected 364"
assert not bad, f"guard violations: {bad[:5]}"
stamped = len(list((OUT / 'stores').glob('*/_INDEX_COMPLETE.json')))
assert stamped == 364, f"{stamped} completion stamps, expected 364"
print(f"  ok: 364/364 indexed, alignment min {min(d['alignment'] for d in docs):.4f}, "
      f"new spend ${log['spent_usd']:.2f}")
PY
say "index verified"

# ---- 2. retrieval, both corpora ----
for B in docbench mmlongbench; do
  if [ ! -f "tier1/s5_graphrag/s5_retrieval_${B}.json" ]; then
    say "retrieval $B"
    $DR -e GRAPHRAG_API_KEY="$OPENAI_API_KEY" -v /media/volume/vol3/grammrag:/work \
      -w /work/gramm-rag-graphbracket $IMG_S5 \
      venvs/s5_graphrag/bin/python retrieve_s5.py --bench $B --tag full --ceiling 15 \
      > "logs/s5_retr_${B}.log" 2>&1 || { say "FATAL retrieval $B"; tail -5 "logs/s5_retr_${B}.log"; exit 1; }
  fi
done

# ---- 3. verify and stage into the generator's input dir ----
python3 - <<'PY'
import json, shutil
from pathlib import Path
GB = Path('/media/volume/vol3/grammrag/gramm-rag-graphbracket/tier1/s5_graphrag')
CP = Path('/media/volume/vol3/grammrag/gramm-rag-colpali/retrieval')
for b, n in (('mmlongbench', 1091), ('docbench', 1102)):
    f = GB / f's5_retrieval_{b}.json'; d = json.loads(f.read_text())
    assert d['n_questions'] == n, f"{b}: {d['n_questions']} != {n}"
    assert not d['missing_stores'], f"{b}: missing stores {d['missing_stores'][:3]}"
    al = [r for r in d['alignment_rate'] if r is not None]
    assert len(al) == n, f"{b}: {len(al)} scored != {n}"
    af = [x for x in d['attributable_fraction'] if x is not None]
    print(f"  {b}: align {sum(al)/n:.4f}  attributable {sum(af)/len(af):.4f}  "
          f"cost ${d['retrieval_cost_usd']:.2f}")
    shutil.copy(f, CP / f's5_retrieval_{b}.json')
print("  staged both retrieval artefacts")
PY
say "retrieval verified and staged"

# ---- 4. generation ----
$DR -e OPENAI_API_KEY="$OPENAI_API_KEY" -v /media/volume/vol3/grammrag:/work \
  -w /work/gramm-rag-colpali $IMG_GEN \
  bash -lc 'for b in mmlongbench docbench; do echo "=== gen s5 / $b ==="; \
            python generate_colpali.py $b --arm s5 || exit 1; done' \
  > logs/s5_gen.log 2>&1 || { say "FATAL generation"; tail -15 logs/s5_gen.log; exit 1; }
say "generation complete"

# ---- 5. scoring ----
$DR -e OPENAI_API_KEY="$OPENAI_API_KEY" -v /media/volume/vol3/grammrag:/work \
  -w /work/gramm-rag-colpali $IMG_GEN \
  bash -lc 'for b in mmlongbench docbench; do echo "=== score s5 / $b ==="; \
            python score_colpali.py $b --arm s5 || exit 1; done' \
  > logs/s5_score.log 2>&1 || { say "FATAL scoring"; tail -15 logs/s5_score.log; exit 1; }
say "S5 DOWNSTREAM DONE"
