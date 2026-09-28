#!/usr/bin/env python3
"""E / P4 -- failure attribution on P3 (MP-DocVQA), five arms, n=950 common set.

Every failed question is coded by joining two facts already measured, with no new API spend:

  LOCALISATION  did the gold page reach the generator's context (page-hit@3)?
  CONVERSION    given the context, did the answer come out right (ANLS)?

That 2x2 is the conversion-gap thesis made operational:
  * LOCALISATION_MISS  -- gold page absent, answer wrong. Retrieval's failure.
  * CONVERSION_MISS    -- gold page PRESENT, answer wrong. The generator had what it
                          needed and did not use it. This is the cell the thesis is about.
  * LUCKY_HIT          -- gold page absent, answer right. Answerable without the gold page
                          (priors, other pages, or a gold span duplicated elsewhere).
                          Counted, never discarded: it bounds how much of an arm's score
                          is independent of its retrieval.
  * BOTH_OK            -- gold page present, answer right.

A partial-credit metric needs a threshold to define "wrong". ANLS >= 0.5 is the metric's own
hard threshold (anls_score returns 0 below it), so scores are already effectively binary --
stated rather than chosen post hoc.
"""
import json, sys, statistics
from pathlib import Path

GB = Path(__file__).resolve().parent
ROOT = GB.parent
sys.path.insert(0, str(ROOT / "gramm-rag-mpdocvqa"))
sys.path.insert(0, str(GB))
from src.evaluation.metrics import anls_score      # noqa: E402
from p3_postproc import parse_answer               # noqa: E402

P3 = GB / "tier1/p3_mpdocvqa"
GEN = P3 / "generation"
ARMS = ["gramm", "flat", "lightrag", "s2", "s5"]
NAME = {"lightrag": "LightRAG", "s2": "RAG-Anything", "s5": "MS GraphRAG",
        "flat": "Flat-vector", "gramm": "GraMM-RAG"}
D3DIR = ROOT / "gramm-rag-3benchmarks/data/mpdocvqa"
THRESH = 0.5

sub = json.loads((P3 / "subsample.json").read_text())
qs = sub["questions"]
excl = set(json.loads((P3 / "common_set.json").read_text())["excluded_indices"])
common = [i for i in range(len(qs)) if i not in excl]

ctx = {}
for a in ARMS:
    rp = P3 / f"p3_retrieval_{a}.json"
    if rp.exists():
        d = json.loads(rp.read_text())
        ctx[a] = {i: d["context_pages"][i][:3] for i in range(len(qs))}
    else:
        fn = "baseline_vector_mpdocvqa_pilot.json" if a == "flat" else "gramm_mpdocvqa_pilot_s42.json"
        d = json.loads((D3DIR / fn).read_text())
        ctx[a] = {i: d["context_pages"][qs[i]["q_index"]][:3] for i in range(len(qs))}

rows = {}
for a in ARMS:
    g = json.loads((GEN / f"p3_generation_{a}.json").read_text())
    cells = {"LOCALISATION_MISS": [], "CONVERSION_MISS": [], "LUCKY_HIT": [], "BOTH_OK": []}
    for i in common:
        gold = qs[i]["gold_page_absolute"]
        hit = gold in ctx[a][i]
        pred = parse_answer(g["predictions"][i] or "")
        golds = list(dict.fromkeys(qs[i]["answers"])) or [""]
        ok = max(anls_score(pred, x) for x in golds) >= THRESH
        key = ("BOTH_OK" if (hit and ok) else "LUCKY_HIT" if (ok and not hit)
               else "CONVERSION_MISS" if (hit and not ok) else "LOCALISATION_MISS")
        cells[key].append(i)
    rows[a] = cells

n = len(common)
print(f"P4 failure attribution -- MP-DocVQA, n={n} common set, ANLS>={THRESH}\n")
print(f"{'arm':14s}{'BOTH_OK':>9s}{'CONV_MISS':>11s}{'LOC_MISS':>10s}{'LUCKY':>8s}"
      f"{'hit@3':>8s}{'conv|hit':>10s}")
for a in ARMS:
    c = rows[a]
    hit = len(c["BOTH_OK"]) + len(c["CONVERSION_MISS"])
    conv = len(c["BOTH_OK"]) / hit if hit else 0.0
    print(f"{NAME[a]:14s}{len(c['BOTH_OK'])/n:>9.3f}{len(c['CONVERSION_MISS'])/n:>11.3f}"
          f"{len(c['LOCALISATION_MISS'])/n:>10.3f}{len(c['LUCKY_HIT'])/n:>8.3f}"
          f"{hit/n:>8.3f}{conv:>10.3f}")

print("\nOf each arm's FAILURES, what share is conversion rather than localisation?")
for a in ARMS:
    c = rows[a]
    fails = len(c["CONVERSION_MISS"]) + len(c["LOCALISATION_MISS"])
    share = len(c["CONVERSION_MISS"]) / fails if fails else 0.0
    print(f"  {NAME[a]:14s} failures {fails:>4d}  conversion share {share:.3f}")

out = dict(benchmark="mpdocvqa", n_common=n, anls_threshold=THRESH,
           cells={NAME[a]: {k: len(v) for k, v in rows[a].items()} for a in ARMS},
           conversion_given_hit={NAME[a]: round(
               len(rows[a]["BOTH_OK"]) / max(len(rows[a]["BOTH_OK"]) + len(rows[a]["CONVERSION_MISS"]), 1), 4)
               for a in ARMS},
           sampled_indices={NAME[a]: {k: v[:100] for k, v in rows[a].items()} for a in ARMS})
(P3 / "p4_failure_attribution.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {P3/'p4_failure_attribution.json'}")
