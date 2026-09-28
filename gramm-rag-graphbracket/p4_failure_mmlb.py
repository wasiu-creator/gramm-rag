#!/usr/bin/env python3
"""E / P4 -- failure attribution on MMLongBench-Doc. Four arms, ctx-hit x correctness.

TWO DIFFERENCES FROM THE P3 TAXONOMY, both structural rather than cosmetic:

 1. AN UNANSWERABLE STRATUM EXISTS (244 questions; P3 has none -- that is C-61). A correct
    abstention is neither BOTH_OK nor a failure: no gold page exists to hit, so folding it
    into the 2x2 would score a correct refusal as a localisation miss. It gets its own cells
    (ABSTAIN_OK / ABSTAIN_MISS) and is excluded from the conv|hit denominator.
 2. ctx-hit is computed on the CANONICAL denominator: answerable questions WITH gold pages
    (n=838), matching ctx_hit_n in the shipped score files. The 9 answerable questions with
    empty evidence_pages (reconciliation item R1) are kept visible, never folded.
"""
import ast, json, sys
from pathlib import Path

GB = Path(__file__).resolve().parent
ROOT = GB.parent
sys.path.insert(0, str(ROOT / "gramm-rag-mmlongbench"))
sys.path.insert(0, str(GB))
from src.evaluation.metrics import anls_score            # noqa: E402

CP = ROOT / "gramm-rag-colpali"
D3 = ROOT / "gramm-rag-3benchmarks/data/mmlongbench"
SRC = ROOT / "gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json"
THRESH = 0.5
ARMS = {"lightrag": GB / "tier1/lightrag_mmlongbench/lightrag_retrieval_mmlongbench.json",
        "s2":       GB / "tier1/s2_raganything/s2_retrieval_mmlongbench.json",
        "s5":       GB / "tier1/s5_graphrag/s5_retrieval_mmlongbench.json"}
CKPT = {"lightrag": CP / "generation/ckpt_lightrag_mmlongbench.json",
        "s2":       CP / "generation/ckpt_s2_mmlongbench.json",
        "s5":       CP / "generation/ckpt_s5_mmlongbench.json"}
NAME = {"lightrag": "LightRAG", "s2": "RAG-Anything", "s5": "MS GraphRAG"}

src = json.loads(SRC.read_text())
n = len(src)
gold, answerable = [], []
for s in src:
    ep = s.get("evidence_pages")
    try:
        pages = ast.literal_eval(ep) if isinstance(ep, str) else (ep or [])
    except Exception:
        pages = []
    gold.append([int(x) for x in pages] if pages else [])
    answerable.append(str(s.get("answer", "")).strip().lower() != "not answerable")

print(f"MMLongBench-Doc: {n} questions, {sum(answerable)} answerable, "
      f"{n-sum(answerable)} unanswerable")
print(f"  answerable WITH gold pages (ctx-hit denominator): "
      f"{sum(1 for i in range(n) if answerable[i] and gold[i])}")
print(f"  answerable with EMPTY gold pages (R1 anomaly, kept visible): "
      f"{sum(1 for i in range(n) if answerable[i] and not gold[i])}\n")

print(f"{'arm':14s}{'BOTH_OK':>9s}{'CONV_MISS':>11s}{'LOC_MISS':>10s}{'LUCKY':>8s}"
      f"{'ctx-hit':>9s}{'conv|hit':>10s}{'abst_ok':>9s}")
out = {}
for a, rp in ARMS.items():
    if not rp.exists() or not CKPT[a].exists():
        print(f"{NAME[a]:14s}  -- missing artefact"); continue
    ret = json.loads(rp.read_text())
    ck = json.loads(CKPT[a].read_text())["predictions"]
    ctx = ret["context_pages"]
    cells = dict(BOTH_OK=0, CONVERSION_MISS=0, LOCALISATION_MISS=0, LUCKY_HIT=0,
                 ABSTAIN_OK=0, ABSTAIN_MISS=0, SKIPPED=0)
    for i in range(n):
        p = ck.get(str(i))
        if p is None:
            cells["SKIPPED"] += 1; continue
        refused = str(p).strip().lower() in ("not answerable", "unanswerable", "")
        if not answerable[i]:
            cells["ABSTAIN_OK" if refused else "ABSTAIN_MISS"] += 1
            continue
        if not gold[i]:
            continue                                   # R1: no gold page, not scoreable
        hit = bool(set(ctx[i]) & set(gold[i]))
        ok = anls_score(str(p), str(src[i]["answer"])) >= THRESH
        cells["BOTH_OK" if (hit and ok) else "LUCKY_HIT" if ok else
              "CONVERSION_MISS" if hit else "LOCALISATION_MISS"] += 1
    d = sum(cells[k] for k in ("BOTH_OK", "CONVERSION_MISS", "LOCALISATION_MISS", "LUCKY_HIT"))
    hitn = cells["BOTH_OK"] + cells["CONVERSION_MISS"]
    conv = cells["BOTH_OK"] / hitn if hitn else 0.0
    un = cells["ABSTAIN_OK"] + cells["ABSTAIN_MISS"]
    print(f"{NAME[a]:14s}{cells['BOTH_OK']/d:>9.3f}{cells['CONVERSION_MISS']/d:>11.3f}"
          f"{cells['LOCALISATION_MISS']/d:>10.3f}{cells['LUCKY_HIT']/d:>8.3f}"
          f"{hitn/d:>9.3f}{conv:>10.3f}{cells['ABSTAIN_OK']/max(un,1):>9.3f}")
    out[NAME[a]] = dict(cells=cells, n_scoreable=d, ctx_hit=round(hitn/d, 4),
                        conversion_given_hit=round(conv, 4),
                        abstain_accuracy=round(cells["ABSTAIN_OK"]/max(un, 1), 4))

(GB / "tier1/p4_failure_mmlongbench.json").write_text(json.dumps(
    dict(benchmark="mmlongbench", anls_threshold=THRESH, arms=out), indent=1))
print(f"\nwrote tier1/p4_failure_mmlongbench.json")
