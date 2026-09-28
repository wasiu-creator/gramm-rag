#!/usr/bin/env python3
"""D3 -- the five-arm common set, and whether the excluded questions are a fair sample.

Each arm's own row can look complete while the INTERSECTION across all five is smaller than
any of them. Every cross-arm contrast is computed on that intersection and the n is stated
beside the table, not in a footnote.

The exclusions are not random: RAG-Anything's 28 dropped questions failed at retrieval for
one specific system, and if they are systematically harder, dropping them flatters the arm
that dropped them. So the excluded set is compared against the retained set on every
covariate available before results are read.
"""
import json, statistics
from pathlib import Path

GB = Path(__file__).resolve().parent
P3 = GB / "tier1/p3_mpdocvqa"
GEN = P3 / "generation"
sub = json.loads((P3 / "subsample.json").read_text())
qs = sub["questions"]
N = len(qs)

ARMS = ["lightrag", "s2", "s5", "flat", "gramm"]
avail = {}
for a in ARMS:
    ok = set()
    rp = P3 / f"p3_retrieval_{a}.json"
    gp = GEN / f"p3_generation_{a}.json"
    if rp.exists():
        d = json.loads(rp.read_text())
        err = {e["i"] for e in d.get("query_errors", [])}
        ok = {i for i in range(N) if i not in err}
    else:
        ok = set(range(N))                       # archived arms: retrieval covers all
    if gp.exists():
        g = json.loads(gp.read_text())
        pred = g.get("predictions", [])
        ok &= {i for i in range(min(len(pred), N)) if pred[i] is not None}
    else:
        ok = set()                               # not generated yet
    avail[a] = ok

scored = {i for i, q in enumerate(qs) if q["gold_page_absolute"] is not None}
common = set.intersection(*[avail[a] for a in ARMS]) & scored if all(avail.values()) else set()

print(f"subsample: {N} questions\n")
print(f"{'arm':10s}{'available':>11s}{'missing':>9s}")
for a in ARMS:
    print(f"{a:10s}{len(avail[a]):>11d}{N-len(avail[a]):>9d}")
print(f"\nFIVE-ARM COMMON SET: n = {len(common)}")
if not all(avail.values()):
    print("  (incomplete -- some arms not yet generated; rerun after D3 finishes)")

excl = scored - common
print(f"excluded from the five-arm table: {len(excl)}")

if excl and common:
    def cov(idxs, key):
        v = [qs[i][key] for i in idxs if isinstance(qs[i].get(key), (int, float))]
        return v
    print(f"\nIS THE EXCLUDED SET A FAIR SAMPLE? (excluded n={len(excl)}, retained n={len(common)})")
    print(f"{'covariate':22s}{'excluded median':>17s}{'retained median':>17s}{'excl mean':>11s}{'ret mean':>10s}")
    for key in ("total_doc_pages", "octile"):
        e, r = cov(excl, key), cov(common, key)
        if e and r:
            print(f"{key:22s}{statistics.median(e):>17.1f}{statistics.median(r):>17.1f}"
                  f"{statistics.mean(e):>11.2f}{statistics.mean(r):>10.2f}")
    # question length as a crude difficulty proxy
    el = [len(qs[i]["question"]) for i in excl]
    rl = [len(qs[i]["question"]) for i in common]
    print(f"{'question chars':22s}{statistics.median(el):>17.1f}{statistics.median(rl):>17.1f}"
          f"{statistics.mean(el):>11.2f}{statistics.mean(rl):>10.2f}")
    # gold page depth -- how far into the document the answer sits
    ed = [qs[i]["gold_page_absolute"] for i in excl]
    rd = [qs[i]["gold_page_absolute"] for i in common]
    print(f"{'gold page index':22s}{statistics.median(ed):>17.1f}{statistics.median(rd):>17.1f}"
          f"{statistics.mean(ed):>11.2f}{statistics.mean(rd):>10.2f}")
    # documents contributing exclusions
    docs_e = {qs[i]["doc_id"] for i in excl}
    print(f"\nexclusions span {len(docs_e)} documents "
          f"({len(excl)/max(len(docs_e),1):.1f} questions each) -- clustered exclusions on a "
          f"few documents bias differently from scattered ones")
    out = dict(n_subsample=N, n_common=len(common), n_excluded=len(excl),
               per_arm_available={a: len(avail[a]) for a in ARMS},
               excluded_indices=sorted(excl), excluded_documents=sorted(docs_e),
               covariates=dict(
                   excluded=dict(pages_median=statistics.median(cov(excl, "total_doc_pages")),
                                 qchars_mean=statistics.mean(el),
                                 gold_depth_mean=statistics.mean(ed)),
                   retained=dict(pages_median=statistics.median(cov(common, "total_doc_pages")),
                                 qchars_mean=statistics.mean(rl),
                                 gold_depth_mean=statistics.mean(rd))))
    (P3 / "common_set.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {P3/'common_set.json'}")
