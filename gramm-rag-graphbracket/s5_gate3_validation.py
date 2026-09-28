#!/usr/bin/env python3
"""Gate H criterion 3 for S5 -- working, or merely loading?

A clean index and a clean alignment rate prove the plumbing, not the retrieval (C-22/C-24/C-33).
This scores S5's selected pages against a RANDOM-PAGE control drawn from the same document, on
exactly the questions S5 answered, using each benchmark's canonical localisation metric.
"""
import json, random, sys
from pathlib import Path

GB = Path(__file__).resolve().parent
sys.path.insert(0, str(GB))
import mapper                                                    # noqa: E402
OUT = GB / "tier1" / "s5_graphrag"
import os
REPS, SEED, TOPK, B = 200, 42, 3, 4000
SUF = os.environ.get("S5_RET_SUFFIX", "_pilot")


def docbench():
    sys.path.insert(0, "/work/gramm-rag-docbench")
    from src.evaluation.metrics import evidence_page_hit, _quote_tokens
    D3 = Path("/work/gramm-rag-3benchmarks/data/docbench")
    base = json.loads((D3 / "baseline_vector_docbench.json").read_text())
    qm = json.loads((D3 / "qmeta_docbench.json").read_text())["rows"]
    ret = json.loads((OUT / f"s5_retrieval_docbench{SUF}.json").read_text())
    ev, an = base["evidences"], base["is_answerable"]
    pdir = Path("/work/gramm-rag-docbench/cache/parsed"); cache = {}

    def page_text(doc_id, pg):
        if doc_id not in cache:
            f = pdir / f"{doc_id}.json"
            d = json.loads(f.read_text()) if f.exists() else {"elements": []}
            byp = {}
            for e in d.get("elements", []):
                try: k = int(e.get("page_no", -1))
                except (TypeError, ValueError): continue
                byp.setdefault(k, []).append((int(e.get("reading_order", 0) or 0), e.get("text", "") or ""))
            cache[doc_id] = {k: " ".join(t for _, t in sorted(v)) for k, v in byp.items()}
        return cache[doc_id].get(pg, "")

    idx = [i for i, p in enumerate(ret["context_pages"])
           if p and _quote_tokens(ev[i]) and an[i]]
    rng = random.Random(SEED)
    hits = [1 if evidence_page_hit([page_text(qm[i]["doc_id"], p) for p in ret["context_pages"][i]][:5],
                                   ev[i], 0.6) else 0 for i in idx]
    ctrls = []
    for i in idx:                                    # per-question control probability
        npg = int(qm[i].get("doc_pages") or 1); c = 0
        for _ in range(REPS):
            pgs = rng.sample(range(1, npg + 1), min(TOPK, npg))
            if evidence_page_hit([page_text(qm[i]["doc_id"], p) for p in pgs][:5], ev[i], 0.6): c += 1
        ctrls.append(c / REPS)
    return "DocBench", "quote-recall@5", hits, ctrls


def mmlongbench():
    D3 = Path("/work/gramm-rag-3benchmarks/data/mmlongbench")
    base = json.loads((D3 / "baseline_vector_mmlb.json").read_text())
    qm = json.loads((D3 / "qmeta_mmlb.json").read_text())["rows"]
    ret = json.loads((OUT / f"s5_retrieval_mmlongbench{SUF}.json").read_text())
    gold, an = base["gold_pages"], base["is_answerable"]
    idx = [i for i, p in enumerate(ret["context_pages"]) if p and an[i] and gold[i]]
    hits = [1 if any(g in ret["context_pages"][i] for g in gold[i]) else 0 for i in idx]
    rng = random.Random(SEED); ctrls = []
    for i in idx:
        npg = int(qm[i].get("doc_pages") or 1); c = 0
        for _ in range(REPS):
            if any(g in rng.sample(range(1, npg + 1), min(TOPK, npg)) for g in gold[i]): c += 1
        ctrls.append(c / REPS)
    return "MMLongBench", "ctx-hit", hits, ctrls


if __name__ == "__main__":
    # Amended rule: decide on a paired bootstrap CI of the margin, minimum n = 60 per corpus.
    print(f"{'corpus':13s}{'metric':>15s}{'n':>5s}{'S5':>8s}{'random':>8s}{'margin':>9s}"
          f"{'95% CI':>20s}  verdict")
    verdicts = []
    for fn in (docbench, mmlongbench):
        name, metric, hits, ctrls = fn()
        n = len(hits)
        s5, ctrl = sum(hits) / n, sum(ctrls) / n
        rng = random.Random(SEED); draws = []
        for _ in range(B):
            s = [rng.randrange(n) for _ in range(n)]
            draws.append(sum(hits[k] for k in s) / n - sum(ctrls[k] for k in s) / n)
        draws.sort(); lo, hi = draws[int(.025 * B)], draws[int(.975 * B)]
        if n < 60:
            v = f"UNDERPOWERED (n<60)"
        elif lo > 0:
            v = "PASS"
        elif hi < 0:
            v = "FAIL -> stop"
        else:
            v = "INDETERMINATE"
        verdicts.append((name, v))
        print(f"{name:13s}{metric:>15s}{n:>5d}{s5:>8.4f}{ctrl:>8.4f}{s5-ctrl:>+9.4f}"
              f"  [{lo:+.4f},{hi:+.4f}]  {v}")
    print("\n  GATE H CRITERION 3 (amended rule):")
    for name, v in verdicts:
        print(f"    {name:13s} {v}")
