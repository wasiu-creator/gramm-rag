#!/usr/bin/env python3
"""Persist the C-66 audit and the three P3 checks as result JSONs.

These numbers were first computed in throwaway /tmp scripts and quoted into
CORRECTION_2026-09-21_MPDOCVQA_APPA.md and P3_RESULT.md -- which left published figures with
no artefact behind them: exactly the drift the claims ledger exists to prevent. Recomputed
here, deterministically (seed 42), from the archived results and the imdb.
"""
import json, random, re, sys
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gramm-rag-mpdocvqa"))
from src.evaluation.metrics import compute_answer_page_accuracy as APPA, anls_score  # noqa: E402

D = ROOT / "gramm-rag-3benchmarks/data/mpdocvqa"
P3 = ROOT / "gramm-rag-graphbracket/tier1/p3_mpdocvqa"
recs = list(np.load(str(ROOT / "gramm-rag-mpdocvqa/data_volume/imdbs/imdb_val.npy"), allow_pickle=True)[1:])
N = len(recs)
truth = [int(re.search(r"_p(\d+)$", str(list(r["image_name"])[int(r["answer_page_idx"])])).group(1)) for r in recs]
buggy = [(list(r["pages"])[int(r["answer_page_idx"])] if int(r["answer_page_idx"]) < len(list(r["pages"])) else 0) for r in recs]
ai = [int(r["answer_page_idx"]) for r in recs]
A = {"Flat-vector": json.load(open(D / "baseline_vector_mpdocvqa_pilot.json")),
     "GraMM-RAG": json.load(open(D / "gramm_mpdocvqa_pilot_s42.json"))}

def boot(d, B=2000, seed=42):
    rng = random.Random(seed); n = len(d)
    b = sorted(float(np.mean(d[[rng.randrange(n) for _ in range(n)]])) for _ in range(B))
    return [round(b[int(.025 * B)], 4), round(b[int(.975 * B) - 1], 4)]

# ---------------- 1. APPA correction ----------------
appa = {}
for a, d in A.items():
    o, c = APPA(d["retrieved_pages"], d["gold_pages"]), APPA(d["retrieved_pages"], truth)
    assert abs(o["appa@1"] - d["metrics"]["appa@1"]) < 1e-9, f"{a}: archived APPA does not reproduce"
    assert all(int(x) == int(y) for x, y in zip(d["gold_pages"], buggy)), f"{a}: archive != line-311 formula"
    appa[a] = {"published": {k: round(o[k], 4) for k in ("appa@1", "appa@5", "appa@10")},
               "corrected": {k: round(c[k], 4) for k in ("appa@1", "appa@5", "appa@10")}}
def hits(rp, gold, k): return np.array([1 if gold[i] in list(rp[i])[:k] else 0 for i in range(N)])
contr = {}
for k in (1, 5, 10):
    for lab, gold in (("published", buggy), ("corrected", truth)):
        dd = hits(A["GraMM-RAG"]["retrieved_pages"], gold, k) - hits(A["Flat-vector"]["retrieved_pages"], gold, k)
        contr.setdefault(f"appa@{k}", {})[lab] = {"diff": round(float(dd.mean()), 4), "ci95": boot(dd)}
top0 = {a: round(sum(1 for p in d["retrieved_pages"] if p and p[0] == 0) / N, 4) for a, d in A.items()}
fb = [i for i in range(N) if ai[i] >= 2]
fb_credit = {a: round(sum(1 for i in fb if 0 in list(d["retrieved_pages"][i])[:1]) / len(fb), 4) for a, d in A.items()}
MB = json.load(open(ROOT / "gramm-rag-3benchmarks/data/mmlongbench/gramm_mmlongbench_s42.json"))["retrieved_pages"]
FP = json.load(open(D / "baseline_vector_mpdocvqa_pilot.json"))["retrieved_pages"]
correction = {
    "NOTE": "C-66 audit. Ground truth = image_name[answer_page_idx]. Archived gold = gen_notebook.py:311.",
    "n": N,
    "answer_page_idx_distribution": {"eq0": ai.count(0), "eq1": ai.count(1), "ge2": sum(1 for x in ai if x >= 2)},
    "gold_wrong": sum(1 for x, y in zip(truth, buggy) if x != y),
    "gold_wrong_frac": round(sum(1 for x, y in zip(truth, buggy) if x != y) / N, 4),
    "appa": appa, "gramm_minus_flat": contr,
    "top_ranked_page0_frac": top0,
    "true_gold_page0_frac": round(sum(1 for t in truth if t == 0) / N, 4),
    "fallback_questions": len(fb), "fallback_page0_credit_at1": fb_credit,
    "controls": {
        "gramm_mmlongbench_first_page_top_frac": round(sum(1 for p in MB if p and p[0] in (0, 1)) / len([p for p in MB if p]), 4),
        "flat_mpdocvqa_page01_top_frac": round(sum(1 for p in FP if p and p[0] in (0, 1)) / len([p for p in FP if p]), 4),
        "gramm_mpdocvqa_page01_top_frac": round(sum(1 for p in A["GraMM-RAG"]["retrieved_pages"] if p and p[0] in (0, 1)) / N, 4)},
}

# ---------------- 2. ANLS, archived full set ----------------
def per_q(d): return np.array([max(anls_score(p or "", g) for g in (gs if isinstance(gs, list) else [gs]) or [""])
                               for p, gs in zip(d["predictions"], d["golds"])])
f, g = per_q(A["Flat-vector"]), per_q(A["GraMM-RAG"])
assert abs(f.mean() - A["Flat-vector"]["metrics"]["anls"]) < 1e-9 and abs(g.mean() - A["GraMM-RAG"]["metrics"]["anls"]) < 1e-9
dd = g - f
correction["anls_fullset"] = {"gramm": round(float(g.mean()), 4), "flat": round(float(f.mean()), 4),
                              "diff": round(float(dd.mean()), 4), "ci95": boot(dd, 4000),
                              "gramm_better": int((dd > 0).sum()), "flat_better": int((dd < 0).sum()), "n": N}
out1 = ROOT / "gramm-rag-colpali/scores/mpdocvqa_appa_correction.json"
out1.write_text(json.dumps(correction, indent=1))

# ---------------- 3. P3 checks ----------------
sub = json.load(open(P3 / "subsample.json")); qs = sub["questions"]
excl = set(json.load(open(P3 / "common_set.json"))["excluded_indices"]); C = [i for i in range(len(qs)) if i not in excl]
ft = {i: int(re.search(r"_p(\d+)$", str(list(recs[qs[i]["q_index"]]["image_name"])[qs[i]["answer_page_idx"]])).group(1)) for i in C}
ctx = {a: {i: list(A[a]["context_pages"][qs[i]["q_index"]])[:3] for i in C} for a in A}
h = {a: np.array([1 if ft[i] in ctx[a][i] else 0 for i in C]) for a in A}
g0 = np.array([ft[i] == 0 for i in C])
split = {}
for lab, m in (("all", np.ones(len(C), bool)), ("true_gold_page0", g0), ("true_gold_not_page0", ~g0)):
    d2 = h["GraMM-RAG"][m] - h["Flat-vector"][m]
    split[lab] = {"n": int(m.sum()), "gramm": round(float(h["GraMM-RAG"][m].mean()), 4),
                  "flat": round(float(h["Flat-vector"][m].mean()), 4), "diff": round(float(d2.mean()), 4), "ci95": boot(d2, 4000)}
# ---- the k=3 control (C-77(ii)): page-hit@3 on the SAME common set and the SAME verified gold
# pages, so it is directly comparable with the k=2 row above. Its contexts are the archived ones
# plus one reranked page, so this is the page-budget confound removed, not a different retriever.
_fk3 = ROOT / "gramm-rag-colpali/retrieval/flatk3_retrieval_mpdocvqa.json"
if _fk3.exists():
    _f3 = json.load(open(_fk3))["context_pages"]
    ctx["Flat-vector (k=3)"] = {i: list(_f3[qs[i]["q_index"]])[:3] for i in C}
    h["Flat-vector (k=3)"] = np.array([1 if ft[i] in ctx["Flat-vector (k=3)"][i] else 0 for i in C])
    for lab, m in (("all", np.ones(len(C), bool)), ("true_gold_page0", g0), ("true_gold_not_page0", ~g0)):
        d3 = h["GraMM-RAG"][m] - h["Flat-vector (k=3)"][m]
        split.setdefault("vs_flatk3", {})[lab] = {
            "n": int(m.sum()), "gramm": round(float(h["GraMM-RAG"][m].mean()), 4),
            "flatk3": round(float(h["Flat-vector (k=3)"][m].mean()), 4),
            "diff": round(float(d3.mean()), 4), "ci95": boot(d3, 4000)}
    _loc = json.loads((P3 / "p3_localisation_common950.json").read_text())
    _loc["flatk3"] = {"page_hit_at_3": round(float(h["Flat-vector (k=3)"].mean()), 4),
                      "hit_any": None,
                      "note": "k=3 control (C-77(ii)): archived context + one reranked page. "
                              "hit_any is not computed: the reranked pool is this environment's, "
                              "not the archived one (C-79)."}
    (P3 / "p3_localisation_common950.json").write_text(json.dumps(_loc, indent=1))
    print(f"  flatk3 page-hit@3 on n={len(C)}: {h['Flat-vector (k=3)'].mean():.4f} "
          f"(k=2 was {h['Flat-vector'].mean():.4f})")

loc950 = json.load(open(P3 / "p3_localisation_common950.json"))
an = json.load(open(P3 / "p3_anls.json"))["per_arm"]
ARMS = ["Flat-vector", "GraMM-RAG", "LightRAG", "RAG-Anything", "MS GraphRAG"]
KEY = {"Flat-vector": "flat", "GraMM-RAG": "gramm", "LightRAG": "lightrag", "RAG-Anything": "s2", "MS GraphRAG": "s5"}
SR = json.load(open(ROOT / "gramm-rag-colpali/scores/stratified_report.json"))
SRN = {"Flat-vector": "Flat-vector", "GraMM-RAG": "GraMM-RAG", "LightRAG": "LightRAG 1.5.7",
       "RAG-Anything": "RAG-Anything 1.2.8", "MS GraphRAG": "MS GraphRAG 2.7.2"}
def sc(a, b, key): return json.load(open(ROOT / f"gramm-rag-colpali/scores/{a}_scores_{b}.json"))[key]
LOC = {"mmlongbench": {"Flat-vector": 0.2327, "GraMM-RAG": 0.4344, "LightRAG": sc("lightrag", "mmlongbench", "ctx_hit_rate"),
                       "RAG-Anything": sc("s2", "mmlongbench", "ctx_hit_rate"), "MS GraphRAG": sc("s5", "mmlongbench", "ctx_hit_rate")},
       "docbench": {"Flat-vector": 0.3959, "GraMM-RAG": 0.4534, "LightRAG": sc("lightrag", "docbench", "quote_recall@5"),
                    "RAG-Anything": sc("s2", "docbench", "quote_recall@5"), "MS GraphRAG": sc("s5", "docbench", "quote_recall@5")},
       "mpdocvqa": {a: loc950[KEY[a]]["page_hit_at_3"] for a in ARMS}}
ANS = {b: {a: SR[b][SRN[a]]["pooled"] for a in ARMS} for b in ("mmlongbench", "docbench")}
ANS["mpdocvqa"] = {a: an[a]["anls"] for a in ARMS}
loo = {}
for fam, T in (("localisation", LOC), ("answer_quality", ANS)):
    for c1, c2 in (("mmlongbench", "docbench"), ("mmlongbench", "mpdocvqa"), ("docbench", "mpdocvqa")):
        row = {"all": round(float(kendalltau([T[c1][a] for a in ARMS], [T[c2][a] for a in ARMS])[0]), 3)}
        for drop in ARMS:
            k = [a for a in ARMS if a != drop]
            row[f"drop_{drop}"] = round(float(kendalltau([T[c1][a] for a in k], [T[c2][a] for a in k])[0]), 3)
        loo.setdefault(fam, {})[f"{c1}_vs_{c2}"] = row
checks = {"NOTE": "C-66 checks on P3. Gold = image_name-derived truth, independent of any stored field.",
          "n_common": len(C), "stored_gold_matches_filename_truth": sum(1 for i in C if ft[i] == qs[i]["gold_page_absolute"]),
          "gramm_context_contains_page0_frac": round(float(np.mean([0 in ctx["GraMM-RAG"][i] for i in C])), 4),
          "flat_context_contains_page0_frac": round(float(np.mean([0 in ctx["Flat-vector"][i] for i in C])), 4),
          "true_gold_page0_frac": round(float(g0.mean()), 4),
          "gramm_minus_flat_page_hit_at_3": split, "leave_one_out_tau_b": loo,
          "loc_values": LOC, "answer_values": ANS}
out2 = P3 / "p3_c66_checks.json"; out2.write_text(json.dumps(checks, indent=1))
print(f"wrote {out1.relative_to(ROOT)}\nwrote {out2.relative_to(ROOT)}")
print(f"  APPA@1 GraMM-Flat published {contr['appa@1']['published']['diff']:+.4f} -> corrected {contr['appa@1']['corrected']['diff']:+.4f}")
print(f"  ANLS full set diff {correction['anls_fullset']['diff']:+.4f} {correction['anls_fullset']['ci95']}")
print(f"  P3 not-page-0 diff {split['true_gold_not_page0']['diff']:+.4f} {split['true_gold_not_page0']['ci95']}")
print(f"  LOO loc mmlb-mpdoc: all {loo['localisation']['mmlongbench_vs_mpdocvqa']['all']}  -Flat {loo['localisation']['mmlongbench_vs_mpdocvqa']['drop_Flat-vector']}  -GraMM {loo['localisation']['mmlongbench_vs_mpdocvqa']['drop_GraMM-RAG']}")
