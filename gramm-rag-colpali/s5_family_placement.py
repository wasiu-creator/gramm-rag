#!/usr/bin/env python3
"""A2 -- where does a SECOND text-side system's false-decline rate land?

G-5 says false declines track how much of the pipeline is text-side. MS GraphRAG is
text-only, so G-5 predicts it lands WITH the text/graph family, not between the families.
RAG-Anything (partly vision) was predicted to land between, and did. A second text-side
system either replicates the family account or breaks it.

Family means are computed EXCLUDING the arm under test, so the arm is not compared against
a bracket it helps define.
"""
import json, random, re, sys, statistics
from pathlib import Path
sys.path.insert(0, "/work/gramm-rag-colpali")
import stratified_report as SR

B, SEED = 4000, 42
TEXT = ["Flat-vector", "GraMM-RAG", "LightRAG 1.5.7"]
VISION = ["ColPali", "ColPali size-matched", "Union(CP∪GraMM)"]
UNDER_TEST = "MS GraphRAG 2.7.2"


def rates(bench):
    out = SR.load(bench)
    preds, an = out[0], out[1]
    A = [i for i, a in enumerate(an) if a]
    flags = {k: [1 if SR.DEC.search(str(P[i])) else 0 for i in A] for k, P in preds.items()}
    return flags, len(A)


for bench in ("mmlongbench", "docbench"):
    flags, n = rates(bench)
    if UNDER_TEST not in flags:
        print(f"  {bench}: {UNDER_TEST} absent"); continue
    x = flags[UNDER_TEST]
    tmean = statistics.mean(statistics.mean(flags[a]) for a in TEXT if a in flags)
    vmean = statistics.mean(statistics.mean(flags[a]) for a in VISION if a in flags)
    rng = random.Random(SEED)
    draws = sorted(sum(x[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    lo, hi = draws[int(.025 * B)], draws[int(.975 * B)]
    pt = statistics.mean(x)
    tlo = min(statistics.mean(flags[a]) for a in TEXT if a in flags)
    thi = max(statistics.mean(flags[a]) for a in TEXT if a in flags)
    inside_text = tlo <= pt <= thi
    between = min(vmean, tmean) < pt < max(vmean, tmean)
    print(f"  {bench:12s} n={n}")
    print(f"    text/graph family (excl. arm under test): mean {tmean:.3f}  range [{tlo:.3f}, {thi:.3f}]")
    print(f"    vision family:                            mean {vmean:.3f}")
    print(f"    {UNDER_TEST}: {pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"    -> inside the text/graph range: {'YES' if inside_text else 'no'}"
          f"   |  strictly between the families: {'yes' if between else 'NO'}")
    print(f"    -> CI overlaps text/graph range: "
          f"{'YES' if not (hi < tlo or lo > thi) else 'no'}"
          f"   |  CI excludes the vision mean: {'YES' if lo > vmean or hi < vmean else 'no'}")
