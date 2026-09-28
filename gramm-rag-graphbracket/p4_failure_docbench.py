#!/usr/bin/env python3
"""E / P4 -- failure attribution on DocBench. Localisation is QUOTE-RECALL, not page-hit.

DocBench ships no gold_pages at all, so "did the gold page reach context" is undefined here.
The localisation axis is instead "did a top-5 retrieved page contain the gold evidence
quote", using the canonical _quote_tokens / min_overlap=0.6 rule from
src/evaluation/metrics.compute_quote_recall.

That is a DIFFERENT OPERATIONALISATION of the same axis, not the same measurement, and it is
labelled as such wherever the three corpora appear side by side. Pooling across corpora is
forbidden by the pre-registration for exactly this reason.

Unanswerable questions get their own cells, as on MMLongBench (P3 has no such stratum: C-61).
"""
import json, sys
from pathlib import Path

GB = Path(__file__).resolve().parent
ROOT = GB.parent
sys.path.insert(0, str(ROOT / "gramm-rag-docbench"))
sys.path.insert(0, str(GB))
from src.evaluation.metrics import anls_score, compute_quote_recall   # noqa: E402
import mapper                                                        # noqa: E402

CP = ROOT / "gramm-rag-colpali"
D3 = ROOT / "gramm-rag-3benchmarks/data/docbench"
THRESH, MIN_OVERLAP, K = 0.5, 0.6, 5
ARMS = {"lightrag": GB / "tier1/lightrag_docbench/lightrag_retrieval_docbench.json",
        "s2":       GB / "tier1/s2_raganything/s2_retrieval_docbench.json",
        "s5":       GB / "tier1/s5_graphrag/s5_retrieval_docbench.json"}
CKPT = {"lightrag": CP / "generation/ckpt_lightrag_docbench.json",
        "s2":       CP / "generation/ckpt_s2_docbench.json",
        "s5":       CP / "generation/ckpt_s5_docbench.json"}
NAME = {"lightrag": "LightRAG", "s2": "RAG-Anything", "s5": "MS GraphRAG"}

base = json.loads((D3 / "baseline_vector_docbench.json").read_text())
qm = json.loads((D3 / "qmeta_docbench.json").read_text())["rows"]
evid = base["evidences"]; ans = base["is_answerable"]; golds = base["golds"]
n = base["n_questions"]
PARSED = ROOT / "gramm-rag-docbench/cache/parsed"

# TOP-5 RETRIEVED PER ARM. context_pages is a k=3 truncation (mean 2.66), not an arm's
# retrieval depth: the graph arms carry 26.19 retrieved pages per question and flat-vector's
# stored page_texts hold its own top-5 (mean 4.33). Scoring graph arms from context_pages
# while flat-vector's published number came from its top-5 would handicap flat-vector.
#
# Mixing two text sources (flat-vector's stored text vs parse-assembled text for the graph
# arms) is licensed by a validation, not assumed: rebuilding flat-vector's pages from the
# parse reproduced quote-recall@5 0.3932 against the stored 0.3959 on an identical n=730 --
# the sources agree to 0.7%, so the remaining difference is page COUNT, not page CONTENT.
K = 5
_cache = {}
def page_text(doc, pg):
    if doc not in _cache:
        stem = str(doc).replace(".pdf", "").replace(".json", "")
        f = next((PARSED / c for c in (f"{doc}.json", f"{stem}.json", f"{stem}.pdf.json")
                  if (PARSED / c).exists()), None)
        d = {}
        if f is not None:
            for e in mapper.load_parse(f).get("elements", []):
                try:
                    k = int(e.get("page_no"))
                except (TypeError, ValueError):
                    continue
                d.setdefault(k, []).append(str(e.get("text", "")))
        _cache[doc] = {k: " ".join(v) for k, v in d.items()}
    return _cache[doc].get(pg, "")

# REFUSALS ARE PROSE, not a sentinel. The generator says "the provided document excerpts do
# not contain ..."; the golds say "Not mentioned." A literal match on "not answerable" never
# fires, which is why abstention accuracy came back 0.000 for all three arms -- zero by
# construction, not measured. Detect on the phrasings actually present.
REFUSAL_MARKS = ("do not contain", "does not contain", "do not mention", "does not mention",
                 "do not include", "does not include", "not mentioned", "no information",
                 "not answerable", "unanswerable", "cannot be determined", "does not have")
def is_refusal(t):
    t = (t or "").strip().lower()
    return (not t) or any(m in t for m in REFUSAL_MARKS)

print(f"DocBench: {n} questions, {sum(bool(a) for a in ans)} answerable, "
      f"{n - sum(bool(a) for a in ans)} unanswerable")
print(f"  localisation = quote-recall@{K} via canonical compute_quote_recall "
      f"(min_overlap=0.6), top-{K} retrieved per arm\n")

print(f"{'arm':14s}{'BOTH_OK':>9s}{'CONV_MISS':>11s}{'LOC_MISS':>10s}{'LUCKY':>8s}"
      f"{'qrec-ctx':>9s}{'conv|hit':>10s}{'abst_ok':>9s}{'qrec@5':>9s}")
out = {}
for a, rp in ARMS.items():
    if not rp.exists() or not CKPT[a].exists():
        print(f"{NAME[a]:14s}  -- missing artefact"); continue
    ret = json.loads(rp.read_text()); ck = json.loads(CKPT[a].read_text())["predictions"]
    rpg = ret.get("retrieved_pages") or ret["context_pages"]
    # TWO WINDOWS, because one cannot serve both purposes.
    #   LOCALISATION COLUMN  -- top-5 retrieved: compares what each arm's retriever found.
    #   CONVERSION DECOMPOSITION -- the k=3 pages actually placed in the prompt.
    # Using top-5 for both inflates CONVERSION_MISS with quotes on retrieved pages 4-5 that
    # the generator never saw: those are localisation failures from the generator's point of
    # view, and scoring them as conversion failures drove conv|hit to an implausible 0.11-0.14
    # while the same generator converts at 0.46-0.88 on the other two corpora.
    ctxp = ret["context_pages"]
    built5 = [[page_text(qm[i]["doc_id"], p) for p in (rpg[i] or [])[:K]] for i in range(n)]
    built3 = [[page_text(qm[i]["doc_id"], p) for p in (ctxp[i] or [])] for i in range(n)]
    qr = compute_quote_recall(built5, evid, ans, min_overlap=0.6)
    qr3 = compute_quote_recall(built3, evid, ans, min_overlap=0.6)
    hits = []
    for i in range(n):
        one = compute_quote_recall([built3[i]], [evid[i]], [ans[i]], min_overlap=0.6)
        hits.append(one.get("quote_recall@5", 0.0) >= 1.0 if one.get("quote_recall_n") else None)
    cells = dict(BOTH_OK=0, CONVERSION_MISS=0, LOCALISATION_MISS=0, LUCKY_HIT=0,
                 ABSTAIN_OK=0, ABSTAIN_MISS=0)
    for i in range(n):
        p = ck.get(str(i))
        if p is None:
            continue
        refused = is_refusal(p)
        if not ans[i]:
            cells["ABSTAIN_OK" if refused else "ABSTAIN_MISS"] += 1
            continue
        h = hits[i]
        if h is None:
            continue
        g = golds[i] if isinstance(golds[i], list) else [golds[i]]
        ok = max(anls_score(str(p), str(x)) for x in (g or [""])) >= THRESH
        cells["BOTH_OK" if (h and ok) else "LUCKY_HIT" if ok else
              "CONVERSION_MISS" if h else "LOCALISATION_MISS"] += 1
    d = sum(cells[k] for k in ("BOTH_OK", "CONVERSION_MISS", "LOCALISATION_MISS", "LUCKY_HIT"))
    hitn = cells["BOTH_OK"] + cells["CONVERSION_MISS"]
    conv = cells["BOTH_OK"] / hitn if hitn else 0.0
    un = cells["ABSTAIN_OK"] + cells["ABSTAIN_MISS"]
    print(f"{NAME[a]:14s}{cells['BOTH_OK']/max(d,1):>9.3f}{cells['CONVERSION_MISS']/max(d,1):>11.3f}"
          f"{cells['LOCALISATION_MISS']/max(d,1):>10.3f}{cells['LUCKY_HIT']/max(d,1):>8.3f}"
          f"{hitn/max(d,1):>9.3f}{conv:>10.3f}{cells['ABSTAIN_OK']/max(un,1):>9.3f}"
          f"{qr.get('quote_recall@5',0.0):>9.3f}")
    out[NAME[a]] = dict(cells=cells, n_scoreable=d,
                        quote_recall_at5_top5_retrieved=round(qr.get("quote_recall@5", 0.0), 4),
                        quote_recall_in_context_k3=round(qr3.get("quote_recall@5", 0.0), 4),
                        quote_recall_n=qr.get("quote_recall_n"),
                        quote_recall_at5=round(hitn/max(d,1), 4),
                        conversion_given_hit=round(conv, 4),
                        abstain_accuracy=round(cells["ABSTAIN_OK"]/max(un, 1), 4))

(GB / "tier1/p4_failure_docbench.json").write_text(json.dumps(
    dict(benchmark="docbench", localisation="quote-recall@5 (min_overlap=0.6)",
         anls_threshold=THRESH, arms=out), indent=1))
print(f"\nwrote tier1/p4_failure_docbench.json")
