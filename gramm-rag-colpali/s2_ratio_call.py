#!/usr/bin/env python3
"""The Gate H 5b third clause, called properly.

Registered: RAG-Anything's false-decline rate "rises, and by less than 1.64x"
(1.64x = the MEASURED text/graph family ratio, fixed as the threshold before the run).
Two questions, deliberately kept apart:
  (a) did the registered prediction hold?      -- a point statement about the clause
  (b) is the violation statistically decisive? -- a claim about the world
"""
import json, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stratified_report as SR

S2, THR, B = 'RAG-Anything 1.2.8', 1.64, 20000
FL = {}
for b in ('mmlongbench', 'docbench'):
    out = SR.load(b); P, an = out[0][S2], out[1]
    FL[b] = [1 if SR.DEC.search(str(P[i])) else 0 for i, a in enumerate(an) if a]

mm, db = statistics.mean(FL['mmlongbench']), statistics.mean(FL['docbench'])
pt = db / mm
print(f"  MMLongBench false-decline {mm:.4f} (n={len(FL['mmlongbench'])})")
print(f"  DocBench    false-decline {db:.4f} (n={len(FL['docbench'])})")
print(f"  point ratio = {pt:.4f}   registered threshold {THR}")

random.seed(42)
rs = []
for _ in range(B):
    v = {}
    for b, f in FL.items():
        n = len(f)
        v[b] = sum(f[random.randrange(n)] for _ in range(n)) / n
    if v['mmlongbench'] > 0:
        rs.append(v['docbench'] / v['mmlongbench'])
rs.sort()
lo, hi = rs[int(.025*len(rs))], rs[int(.975*len(rs))]
p_above = sum(1 for x in rs if x > THR) / len(rs)
p_one_sided = 1 - p_above          # bootstrap evidence AGAINST the violation
print(f"  bootstrap ({B:,} resamples, corpora independent): 95% CI [{lo:.3f}, {hi:.3f}]")
print(f"  P(ratio > {THR}) = {p_above:.3f}   one-sided p against violation = {p_one_sided:.3f}")

print("\n=== THE CALL, in two separate sentences ===")
rises = pt > 1.0
below = pt < THR
print(f"  (a) the registered clause: 'rises' {'HOLDS' if rises else 'FAILS'}; "
      f"'by less than {THR}x' {'HOLDS' if below else 'FAILS'}")
print(f"      -> as registered, the prediction {'HOLDS' if (rises and below) else 'FAILS'} "
      f"({pt:.3f} vs {THR})")
print(f"  (b) is the violation statistically decisive at 95%? "
      f"{'YES' if lo > THR else 'NO'} -- the CI {'excludes' if lo > THR else 'contains'} {THR}")
print(f"      -> the world-claim 'RAG-Anything rises FASTER than the text/graph family' is "
      f"{'supported' if lo > THR else 'NOT established'}")
