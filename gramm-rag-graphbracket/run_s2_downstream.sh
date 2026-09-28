#!/usr/bin/env bash
# S2 downstream driver: retrieval (docbench) -> generation -> scoring, both benchmarks.
# Fails fast at every stage. Never emits a partial artefact under a canonical name.
set -euo pipefail
GB=/media/volume/vol3/grammrag/gramm-rag-graphbracket
CP=/media/volume/vol3/grammrag/gramm-rag-colpali
PY=$GB/venvs/lightrag_current/bin/python
cd "$GB"
set -a; . /media/volume/vol3/grammrag/gramm-rag-mmlongbench/.env; set +a
: "${OPENAI_API_KEY:?FATAL: no key}"

say(){ echo "S2DOWN $(date +%H:%M:%S) $*"; }

# ---- 1. wait for the mmlongbench retrieval already in flight ----
while pgrep -f "retrieve_lightrag.py --arm s2 --bench mmlongbench" >/dev/null; do sleep 20; done
[ -f tier1/s2_raganything/s2_retrieval_mmlongbench.json ] \
  || { say "FATAL: mmlongbench retrieval produced no canonical artefact"; exit 1; }
say "mmlongbench retrieval present"

# ---- 2. docbench retrieval ----
if [ ! -f tier1/s2_raganything/s2_retrieval_docbench.json ]; then
  say "starting docbench retrieval"
  $PY retrieve_lightrag.py --arm s2 --bench docbench > logs/retr_s2_docbench.log 2>&1 \
    || { say "FATAL: docbench retrieval failed"; tail -5 logs/retr_s2_docbench.log; exit 1; }
fi
say "docbench retrieval present"

# ---- 3. verify both, then stage into the generator's input dir ----
$PY - <<'PY'
import json,sys
from pathlib import Path
GB=Path('/media/volume/vol3/grammrag/gramm-rag-graphbracket/tier1/s2_raganything')
CP=Path('/media/volume/vol3/grammrag/gramm-rag-colpali/retrieval')
exp={'mmlongbench':1091,'docbench':1102}
for b,n in exp.items():
    f=GB/f's2_retrieval_{b}.json'; d=json.loads(f.read_text())
    assert d['n_questions']==n, f"{b}: n_questions {d['n_questions']} != {n}"
    assert not d.get('missing_stores'), f"{b}: missing stores {d['missing_stores'][:3]}"
    al=[r for r in d['alignment_rate'] if r is not None]
    assert len(al)==n, f"{b}: {len(al)} scored != {n}"
    ir=d['alignment_route_split']
    print(f"  {b}: n={n} align={sum(al)/n:.4f} "
          f"image-route={ir['mean_image_route']} qs_with_images={ir['questions_with_image_chunks']}")
    (CP/f's2_retrieval_{b}.json').write_text(f.read_text())
print("  staged both retrieval artefacts into gramm-rag-colpali/retrieval/")
PY
say "retrieval verified and staged"

# ---- 4. generation, both benchmarks, in the pinned image ----
docker run --rm --name s2-gen -v /media/volume/vol3/grammrag:/work -w /work/gramm-rag-colpali \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" grammrag-colpali-pinned:latest \
  bash -lc 'for b in mmlongbench docbench; do echo "=== gen s2 / $b ==="; \
            python generate_colpali.py $b --arm s2 || exit 1; done' \
  > logs/gen_s2.log 2>&1 || { say "FATAL: generation failed"; tail -15 logs/gen_s2.log; exit 1; }
say "generation complete"

# ---- 5. scoring, both benchmarks ----
docker run --rm --name s2-score -v /media/volume/vol3/grammrag:/work -w /work/gramm-rag-colpali \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" grammrag-colpali-pinned:latest \
  bash -lc 'for b in mmlongbench docbench; do echo "=== score s2 / $b ==="; \
            python score_colpali.py $b --arm s2 || exit 1; done' \
  > logs/score_s2.log 2>&1 || { say "FATAL: scoring failed"; tail -15 logs/score_s2.log; exit 1; }
say "scoring complete"
say "S2 DOWNSTREAM DONE"
