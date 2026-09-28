#!/usr/bin/env bash
# D3 -- generation for all five P3 arms under ONE config (k=3, 12000 chars, T=0.0).
#
# The skip test is COMPLETENESS, not existence. `[ -f "$out" ]` tests that a file is there;
# it skipped an arm holding 6 of 978 predictions because a --limit probe had written the
# canonical name. A PARTIAL_ naming convention only works if every future author knows it,
# and the rule lived in another script's header -- headers don't travel. Asking the consumer
# "does this artefact have all the rows" holds regardless of what anyone names a file, and it
# emits the count it checked on every run rather than speaking only on failure.
set -uo pipefail
GB=/media/volume/vol3/grammrag/gramm-rag-graphbracket
cd "$GB"
DR="docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp --env-file /media/volume/vol3/grammrag/.secrets/p3.env"
EXPECT=$(python3 -c "import json;print(json.load(open('tier1/p3_mpdocvqa/subsample.json'))['n_questions'])")
say(){ echo "P3GEN $(date +%H:%M:%S) $*"; }

complete(){ # path -> prints "<n_generated>/<expected>"; rc 0 only if n_generated == expected
  python3 - "$1" "$EXPECT" <<'PY'
import json,sys
p,exp=sys.argv[1],int(sys.argv[2])
try:
    d=json.load(open(p))
except Exception:
    print(f"0/{exp} (absent)"); sys.exit(1)
n=sum(1 for x in d.get("predictions",[]) if x is not None)
q=d.get("n_questions",0)
print(f"{n}/{exp} generated, n_questions={q}")
sys.exit(0 if (n>0 and q==exp) else 1)
PY
}

say "expected predictions per arm: $EXPECT"
fails=0
for arm in lightrag s2 s5 flat gramm; do
  out="tier1/p3_mpdocvqa/generation/p3_generation_${arm}.json"
  st=$(complete "$out"); rc=$?
  if [ $rc -eq 0 ]; then say "skip $arm -- COMPLETE: $st"; continue; fi
  say "=== $arm === (current: $st)"
  $DR -v /media/volume/vol3/grammrag:/work -w /work/gramm-rag-graphbracket \
    grammrag-analysis:latest python generate_p3.py --arm "$arm" > logs/p3_gen_${arm}.log 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then say "FAIL $arm rc=$rc"; tail -8 logs/p3_gen_${arm}.log; fails=$((fails+1)); continue; fi
  st=$(complete "$out"); rc=$?
  if [ $rc -ne 0 ]; then say "FAIL $arm -- incomplete after run: $st"; fails=$((fails+1)); continue; fi
  say "done $arm -- $st"
done
say "GENERATION COMPLETE (failures: $fails)"
for arm in lightrag s2 s5 flat gramm; do
  say "  $arm: $(complete tier1/p3_mpdocvqa/generation/p3_generation_${arm}.json)"
done
