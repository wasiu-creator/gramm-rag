#!/usr/bin/env python3
"""build_flatk3_retrieval.py -- turn the superset selection into the retrieval-shaped JSON that
`generate_colpali.py --arm flatk3` consumes.

No new generator is written for this arm. The canonical driver reads an arm's pages from
`retrieval/<arm>_retrieval_<bench>.json`, so the flat k=3 arm becomes a file, and generation is
byte-for-byte the same code path as every arm it will be compared against.

`context_pages` is the SUPERSET context: the archived context exactly and in order, plus the
highest-ranked page the canonical cross-encoder offers that is not already in it
(FLAT_K3_PREREGISTRATION.md, amendment 1). `retrieved_pages` is the pool this selection actually
reranked, not the archived list -- so the shipped result is internally consistent.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COST = HERE / "cost"
RET = HERE / "retrieval"
ARCHIVED = {
    "mmlongbench": "gramm-rag-3benchmarks/data/mmlongbench/baseline_vector_mmlb.json",
    "docbench": "gramm-rag-3benchmarks/data/docbench/baseline_vector_docbench.json",
    "mpdocvqa": "gramm-rag-3benchmarks/data/mpdocvqa/baseline_vector_mpdocvqa_pilot.json",
}

for bench, arch_rel in ARCHIVED.items():
    src = COST / f"flat_k3_selection_{bench}_sup2.json"
    if not src.exists():
        raise SystemExit(f"FATAL: {src} absent -- run flat_k3_select.py first")
    sel = json.loads(src.read_text())
    rows = sel["rows"]
    n = len(rows)
    if [r["i"] for r in rows] != list(range(n)):
        raise SystemExit(f"FATAL: {bench}: selection rows are not a contiguous 0..n-1 index")
    arch = json.loads((HERE.parent / arch_rel).read_text())
    if len(arch["context_pages"]) != n:
        raise SystemExit(f"FATAL: {bench}: {n} selection rows against "
                         f"{len(arch['context_pages'])} archived contexts")
    # The archived pages must survive verbatim -- that is the whole point of the construction.
    for r in rows:
        if r["superset"][:len(r["archived"])] != r["archived"]:
            raise SystemExit(f"FATAL: {bench} q{r['i']}: superset {r['superset']} does not begin "
                             f"with the archived context {r['archived']}")
        if list(arch["context_pages"][r["i"]]) != r["archived"]:
            raise SystemExit(f"FATAL: {bench} q{r['i']}: archived context drifted")

    out = dict(
        system="Flat-vector (k=3, superset of the archived k=2 context)",
        benchmark=bench, n_questions=n,
        RUN_CONFIG={
            "ARM": "flatk3", "RETRIEVER": "flat-vector: E5 node retrieval + cross-encoder rerank",
            "PAGE_CONTEXT_N": 3, "PAGE_CONTEXT_CHARS_3": 12000,
            "PAGE_SCORE_MODE": "first_seen", "RERANK_POOL": sel["run_config"]["RERANK_POOL"],
            "USE_CROSS_ENCODER": sel["run_config"]["USE_CROSS_ENCODER"],
            "QUERY_ENCODER": sel["query_encoder"], "INDEX_DIM": sel["index_dim"],
            "SEED": 42,
            "CONSTRUCTION": sel["construction"],
            "REPLACES": f"{Path(arch_rel).name} (PAGE_CONTEXT_N=2"
                        + (", 8000 chars)" if bench != "mpdocvqa" else ", 8000 chars, pilot)"),
            "ARCHIVE_FIDELITY": {
                "context_reproduction_rate": sel["gate1_reproduction_rate"],
                "retrieval_pool_reproduction_rate": sel.get("retrieval_reproduction_rate"),
                "note": "Gate 1 retired as a gate (amendment 1); these are fidelity measures. The "
                        "archived pages are preserved by construction, so a divergence can affect "
                        "only the ADDED page.",
            },
            "PAGES_MEAN": {"archived": sel["mean_pages_archived"],
                           "superset": sel["mean_pages_superset"],
                           "questions_with_no_page_added": sel["n_no_page_added"]},
        },
        context_pages=[r["superset"] for r in rows],
        retrieved_pages=[r["retrieved"] for r in rows],
        added_page=[r["added_page"] for r in rows],
    )
    RET.mkdir(parents=True, exist_ok=True)
    f = RET / f"flatk3_retrieval_{bench}.json"
    f.write_text(json.dumps(out))
    print(f"wrote {f.relative_to(HERE.parent)}  n={n}  mean pages "
          f"{sel['mean_pages_archived']} -> {sel['mean_pages_superset']}  "
          f"(no page added on {sel['n_no_page_added']})")
