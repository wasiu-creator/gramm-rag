#!/usr/bin/env python3
"""CIs for the S2 mechanism test.

Reuses stratified_report.py's DEC matcher and its answerable index by IMPORTING
them, never by reimplementing. A first version of this file rebuilt the regex and
produced 0.026/0.054 against the canonical 0.067/0.118 -- the D-32 class exactly:
a plausible number that diverges from the canonical one for an invisible reason.
"""
import json, random, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stratified_report as SR          # DEC, load(), paths -- one definition only

random.seed(42)
B = 4000
PRE = {'mmlongbench': (0.06, 0.12, 0.139, 0.041),
       'docbench':    (0.06, 0.18, 0.228, 0.023)}
S2 = 'RAG-Anything 1.2.8'

def boot(flags):
    n = len(flags)
    xs = sorted(sum(flags[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return statistics.mean(flags), xs[int(.025 * B)], xs[int(.975 * B)]

print(f"{'corpus':13s}{'n_ans':>7s}{'false-decline':>15s}{'95% CI':>21s}"
      f"{'pre-reg':>11s}{'call':>22s}")
res = {}
for b, (lo, hi, tx, vi) in PRE.items():
    out = SR.load(b)
    preds, an = out[0], out[1]
    P = preds[S2]
    flags = [1 if SR.DEC.search(str(P[i])) else 0 for i, a in enumerate(an) if a]
    m, l, h = boot(flags)
    res[b] = m
    inside = lo <= m <= hi
    robust = lo <= l and h <= hi
    call = ('BETWEEN' if inside else 'OUTSIDE') + \
           (', CI inside' if robust else ', CI straddles a bound')
    print(f"{b:13s}{len(flags):>7d}{m:>15.4f}   [{l:.4f},{h:.4f}]"
          f"{f'{lo}-{hi}':>11s}{call:>22s}")

r = res['docbench'] / res['mmlongbench']
print(f"\n  exact ratio DocBench/MMLongBench = {r:.3f}x  vs text/graph 1.64x, vision 0.56x")
print(f"  pre-registered third row (rises AND by less than 1.64x): "
      f"{'SATISFIED' if 1.0 < r < 1.64 else 'FAILS'}"
      + ("" if 1.0 < r < 1.64 else
         " -- it rises FASTER than the text/graph family it sits below in level"))

# ---- Is the ratio failure established, or is 1.747 vs 1.64 within noise? ----
# Paired-by-corpus is not available (different questions), so resample each corpus
# independently and form the ratio -- the two rates are independent samples.
print("\n  ratio uncertainty (4,000 resamples, seed 42, corpora resampled independently)")
FL = {}
for b in PRE:
    out = SR.load(b); P = out[0][S2]; an = out[1]
    FL[b] = [1 if SR.DEC.search(str(P[i])) else 0 for i, a in enumerate(an) if a]
random.seed(42)
rs = []
for _ in range(B):
    vals = {}
    for b, f in FL.items():
        n = len(f)
        vals[b] = sum(f[random.randrange(n)] for _ in range(n)) / n
    if vals['mmlongbench'] > 0:
        rs.append(vals['docbench'] / vals['mmlongbench'])
rs.sort()
lo95, hi95 = rs[int(.025 * len(rs))], rs[int(.975 * len(rs))]
above = sum(1 for x in rs if x > 1.64) / len(rs)
print(f"    ratio 95% CI [{lo95:.3f}, {hi95:.3f}]   P(ratio > 1.64) = {above:.3f}")
print(f"    -> the ratio clause fails on the point estimate; the CI "
      f"{'EXCLUDES' if lo95 > 1.64 else 'INCLUDES'} the 1.64 threshold, so the failure is "
      f"{'established' if lo95 > 1.64 else 'NOT established -- report as unresolved, not as a refutation'}")

# ---- Family-bracket robustness: the claim G-5 actually makes ----
print("\n  family-bracket robustness (the substantive G-5 claim, vs the sharper band)")
for b, (lo, hi, tx, vi) in PRE.items():
    f = FL[b]; n = len(f)
    random.seed(42)
    xs = sorted(sum(f[random.randrange(n)] for _ in range(n)) / n for _ in range(B))
    l, h = xs[int(.025 * B)], xs[int(.975 * B)]
    infam = min(vi, tx) < l and h < max(vi, tx)
    print(f"    {b:13s} CI [{l:.4f},{h:.4f}] inside family bracket "
          f"({min(vi,tx)}, {max(vi,tx)}): {'YES' if infam else 'NO'}"
          f"   | inside pre-reg band {lo}-{hi}: {'YES' if lo<=l and h<=hi else 'NO'}")
