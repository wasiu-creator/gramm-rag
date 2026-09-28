#!/usr/bin/env python3
"""D1 -- project full-P3 indexing cost and wall-clock from the stratified 50-doc set.

Denominator is CHARACTERS, pre-registered in P3_PREREGISTRATION.md: MP-DocVQA has zero
images, so C-48's image denominator is inapplicable, and that is stated as the reason rather
than discovered afterwards. Because there is only one physically available denominator here,
the S2/S4 "two denominators disagree by >2x -> NOT IDENTIFIED" rule cannot fire; what
replaces it is a per-document spread, reported so the projection's precision is visible
instead of implied.
"""
import json, statistics
from pathlib import Path

GB = Path(__file__).resolve().parent
FULL = json.loads((GB / "data/corpus_sizes_p3.json").read_text())
PROJ = json.loads((GB / "data/corpus_sizes_p3_proj50.json").read_text())
FULL_CH, PROJ_CH = sum(r["chars"] for r in FULL), sum(r["chars"] for r in PROJ)

ARMS = [("LightRAG",     "p3_lightrag_proj", ""),
        ("RAG-Anything", "p3_s2_proj",       "mpdocvqa__"),
        ("MS GraphRAG",  "p3_s5_proj",       "mpdocvqa__")]


def arm_totals(arm):
    """Sum the PER-DOCUMENT _INDEX_COMPLETE.json stamps, not the run log.

    The run log records what the LAST invocation did. All three indexers are resume-gated,
    so a rerun over already-built stores writes a log saying $0.0000 -- correct about that
    invocation, and meaningless as a measurement. The first version of this script read the
    log and duly projected $0.00 for two arms that had in fact cost $0.01 and $0.56, and
    still printed 'WITHIN estimate'. The stamps are written once, when the document is
    actually indexed, so they survive resume.

    Restricted to the CURRENT projection set: LightRAG's directory also holds 7 stores from
    the pre-floor draw, and summing those would inflate the measured base.
    """
    name, sub, pre = arm
    root = GB / "tier1" / sub / "stores"
    cost = wall = 0.0
    n = 0
    per_doc = []
    missing = []
    for r in PROJ:
        stem = r["doc"].replace(".json", "")
        f = root / f"{pre}{stem}" / "_INDEX_COMPLETE.json"
        if not f.exists():
            missing.append(stem); continue
        d = json.loads(f.read_text())
        c = d.get("cost_usd")
        w = d.get("seconds")
        ch = d.get("chars") or d.get("ingested_chars") or r["chars"]
        if c is None:
            missing.append(stem); continue
        cost += c; wall += (w or 0.0); n += 1
        if ch:
            per_doc.append(c / ch * 1e6)
    return dict(cost=cost, wall=wall, ndocs=n, missing=missing,
                per_doc_usd_per_mchar=per_doc)


print(f"projection set : {len(PROJ)} docs, {PROJ_CH:,} chars")
print(f"full subsample : {len(FULL)} docs, {FULL_CH:,} chars")
print(f"scale factor   : {FULL_CH/PROJ_CH:.3f}x on characters\n")

tot_cost = tot_wall = 0.0
rows = []
print(f"{'arm':14s}{'docs':>5s}{'measured $':>12s}{'measured h':>12s}"
      f"{'proj $':>10s}{'proj h':>9s}{'$/Mchar spread':>17s}")
k = FULL_CH / PROJ_CH
for arm in ARMS:
    t = arm_totals(arm)
    pc, pw = t["cost"] * k, t["wall"] * k / 3600
    tot_cost += pc; tot_wall += pw
    pd = t["per_doc_usd_per_mchar"]
    spread = (f"{min(pd):.2f}-{max(pd):.2f} ({max(pd)/max(min(pd),1e-9):.1f}x)"
              if len(pd) > 1 else "n/a")
    print(f"{arm[0]:14s}{t['ndocs']:>5}{t['cost']:>12.4f}{t['wall']/3600:>12.2f}"
          f"{pc:>10.2f}{pw:>9.2f}{spread:>17s}")
    rows.append((arm[0], t))

print(f"\n{'TOTAL projected':14s}{'':>5}{'':>12}{'':>12}{tot_cost:>10.2f}{tot_wall:>9.2f}")
print(f"\nPre-registered estimate: $20-30, 5-7 h.  Ceiling: $60.")

# Preconditions on emitting ANY verdict. Coverage was added after a verdict was printed
# from one document; the zero-cost check was added after a verdict was printed from two
# resume no-ops that passed the coverage check. Both are physical invariants: a metered
# arm cannot index 50 documents for exactly $0, and cannot do it in exactly 0 seconds.
MIN_DOCS = 40
bad = []
for name, t in rows:
    if t["ndocs"] < MIN_DOCS:
        bad.append(f"{name}: only {t['ndocs']}/{len(PROJ)} documents stamped (floor {MIN_DOCS})")
    if t["cost"] <= 0.0:
        bad.append(f"{name}: measured cost is exactly $0 over {t['ndocs']} documents -- "
                   f"impossible for a metered arm; the stamps were not read")
    if t["wall"] <= 0.0:
        bad.append(f"{name}: measured wall-clock is exactly 0 s -- same defect")
if bad:
    print("\nPROJECTION REFUSED -- cannot project from this:")
    for b in bad:
        print(f"  - {b}")
    print("\nNo verdict is emitted. Do NOT read the per-arm numbers above as a projection.")
    raise SystemExit(2)

verdict = ("WITHIN estimate" if tot_cost <= 30 else
           "ABOVE estimate, under ceiling" if tot_cost <= 60 else
           "ABOVE CEILING -- stop and report")
print(f"Projection verdict: {verdict}  (indexing only; retrieval+generation excluded)")
