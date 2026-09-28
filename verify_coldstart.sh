#!/usr/bin/env bash
# H -- cold-start reproducibility test. Regenerates the evidence notebook from the HANDOFF
# PACKAGE ALONE (vol3 not mounted) and diffs every cell's output against the vol3 run.
#
# Why a diff and not an error count: the first cold-start run reported 0 errors while three
# cells printed different values -- index sizes read 0.00 GiB, and a pre-registered clause
# flipped FALSIFIED -> HELD at p = 4.6e-100. Silent fallbacks never raise, so only comparing
# OUTPUTS catches them. Pass = 0 errors AND 0 differing cells.
set -uo pipefail
# Defaults are portable: the package is the directory this script sits in, and scratch space is a
# temp dir -- a public reader has neither this instance's vol2 handoff nor its vol3 mount.
PKG=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
IMAGE=${IMAGE:-grammrag-analysis:latest}   # build: docker build -t grammrag-analysis -f _analysis_env/Dockerfile _analysis_env
# The reference is the executed notebook SHIPPED IN THE PACKAGE, read before the copy is
# regenerated -- so the test needs no instance and no vol3: "does this package reproduce what
# it claims". Override REF to compare against another run.
REF=${REF:-$PKG/gramm-rag-colpali/phaseC_evidence.ipynb}
CS=${CS:-$(mktemp -d)/coldstart}
# clean AS ROOT: an earlier run left root-owned files that plain rm could not remove, and a
# dirty tree lets a cell read leftovers the package does not ship -- a false pass.
docker run --rm -v "$(dirname "$CS")":/v "$IMAGE" rm -rf "/v/$(basename "$CS")"
mkdir -p "$CS"; rsync -a "$PKG/" "$CS/"
[ "$(find "$CS" ! -user "$(id -u)" | wc -l)" -eq 0 ] || { echo "FAIL: cold-start tree not clean"; exit 1; }
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$CS":/work -w /work/gramm-rag-colpali \
  "$IMAGE" bash -lc 'python gen_phaseC_notebook.py >/dev/null 2>&1 &&
  jupyter nbconvert --to notebook --execute --inplace --allow-errors \
    --ExecutePreprocessor.timeout=900 phaseC_evidence.ipynb >/dev/null 2>&1'
python3 - "$REF" "$CS/gramm-rag-colpali/phaseC_evidence.ipynb" <<'PY'
import json, re, sys
def outs(p):
    r=[]
    for c in json.load(open(p))['cells']:
        if c['cell_type']!='code': continue
        t=''
        for o in c.get('outputs',[]):
            if o.get('output_type')=='stream': t+=''.join(o['text'])
            elif o.get('output_type')=='error': t+=f"ERROR {o.get('ename')}"
            elif o.get('output_type') in ('execute_result','display_data'):
                t+=''.join(o.get('data',{}).get('text/plain',''))
        r.append(re.sub(r'/media/volume/vol3/grammrag','/work',t))
    return r
a,b=outs(sys.argv[1]),outs(sys.argv[2])
err=sum(1 for t in b if 'ERROR ' in t); diff=[i for i in range(min(len(a),len(b))) if a[i]!=b[i]]
print(f"  cells {len(a)} vs {len(b)} | cold-start errors {err} | differing outputs {len(diff)} {diff}")
sys.exit(0 if (len(a)==len(b) and err==0 and not diff) else 1)
PY
rc=$?; [ $rc -eq 0 ] && echo "COLD-START PASS" || echo "COLD-START FAIL"; exit $rc
