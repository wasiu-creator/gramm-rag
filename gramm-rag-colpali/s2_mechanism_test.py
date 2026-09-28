#!/usr/bin/env python3
"""S2 mechanism test — Gate H §5b, evaluated exactly as pre-registered.

Reads decline_ans (= false-decline rate on answerable questions = 1 - TNR) for
RAG-Anything from stratified_report.json and applies the FIXED disposition table.
Nothing here is decided after seeing the numbers: the intervals, the ratio
threshold and the mixed-result reading were all written before the run.
"""
import json, sys
from pathlib import Path

SC = Path(__file__).resolve().parent / 'scores'
S2 = 'RAG-Anything 1.2.8'

# ---- pre-registered, Gate H section 5b ----
PRE = {'mmlongbench': dict(text=0.139, vision=0.041, lo=0.06, hi=0.12),
       'docbench':    dict(text=0.228, vision=0.023, lo=0.06, hi=0.18)}
FAMILY_RATIO = 1.64          # measured text/graph DocBench/MMLongBench ratio

sr = json.loads((SC / 'stratified_report.json').read_text())
missing = [b for b in PRE if S2 not in sr.get(b, {})]
if missing:
    sys.exit(f"FATAL: {S2} absent from stratified_report for {missing} -- "
             "regenerate stratified_report.py before evaluating the test")

print(f"{'corpus':14s}{'text/graph':>12s}{'vision':>9s}{'predicted':>13s}"
      f"{'RAG-Anything':>14s}{'verdict':>14s}")
inter = {}
obs = {}
for b, P in PRE.items():
    v = sr[b][S2]['decline_ans']
    obs[b] = v
    # "Between the families" is the claim G-5 makes; the [lo,hi] interval is the
    # sharper pre-registered form of it. Both are recorded -- the interval is the
    # test, the family bracket says which way a miss went.
    inside = P['lo'] <= v <= P['hi']
    inter[b] = inside
    between = min(P['vision'], P['text']) < v < max(P['vision'], P['text'])
    tag = 'BETWEEN' if inside else ('bracket-only' if between else
          ('with TEXT' if abs(v-P['text']) < abs(v-P['vision']) else 'with VISION'))
    print(f"{b:14s}{P['text']:>12.3f}{P['vision']:>9.3f}"
          f"{P['lo']:>7.2f}-{P['hi']:<5.2f}{v:>14.3f}{tag:>14s}")

ratio = obs['docbench'] / obs['mmlongbench'] if obs['mmlongbench'] else float('nan')
print(f"\n  ratio DocBench/MMLongBench: {ratio:.2f}x   "
      f"(text/graph {FAMILY_RATIO}x, vision 0.56x)")
print(f"  pre-registered third row -- rises AND by less than {FAMILY_RATIO}x: "
      f"{'SATISFIED' if 1.0 < ratio < FAMILY_RATIO else 'NOT satisfied'}")

n = sum(inter.values())
verdict = {2: ("CONFIRMS G-5", "the effect tracks how much of the pipeline is text-side"),
           0: ("FALSIFIES G-5 as stated", "intermediacy holds on neither corpus"),
           1: ("CONFIRMS NOTHING", "corpus-dependent; report which corpus and which family")}[n]
print(f"\n=== VERDICT (fixed before the run): {verdict[0]} ===")
print(f"    intermediacy holds on {n}/2 corpora -- {verdict[1]}")
if n == 1:
    miss = [b for b in PRE if not inter[b]][0]
    P, v = PRE[miss], obs[miss]
    fam = 'text/graph' if abs(v-P['text']) < abs(v-P['vision']) else 'vision'
    near = P['text'] if fam == 'text/graph' else P['vision']
    print(f"    the miss: {miss}, fell in with the {fam} family "
          f"({v:.3f} vs {near:.3f}, |delta|={abs(v-near):.3f}); "
          f"outside the predicted {P['lo']}-{P['hi']} by "
          f"{min(abs(v-P['lo']), abs(v-P['hi'])):.3f}")
