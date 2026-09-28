#!/usr/bin/env python3
"""S5 -- retrieval through Microsoft GraphRAG local search, emitting our page schema.

Runs in the graphrag container (Python 3.11). Emits the same artefact shape as
`retrieve_lightrag.py`, so `generate_colpali.py --arm s5` runs unchanged.

WHAT IS ATTRIBUTABLE, AND WHAT IS NOT (§1b of the pre-registration)
  GraphRAG's local-search context has five parts. `sources` is verbatim text units from OUR
  injected parse and IS attributable to pages. `reports`, `entities` and `relationships` are
  LLM-GENERATED summaries with no character preimage -- C-49's problem, and here unfixable:
  a community report aggregates many chunks across many pages by construction, so there is no
  single page to attribute it to. Per the disposition fixed BEFORE any result existed, generated
  mass is EXCLUDED from attribution and the excluded fraction is PUBLISHED beside the metric.

GUARDS
  * every question emits its alignment and attributable fraction (rule 22)
  * blocking halt if >50% of questions attribute to zero pages -- a harness failure and a weak
    retriever are indistinguishable from the metric alone, and reporting the first as the second
    would understate a competitor
  * D-34/C-57 exhaustion classified on the exception chain; the run exits rather than scoring 0
"""
import argparse, asyncio, json, os, sys, time
from collections import defaultdict
from pathlib import Path

GB = Path(__file__).resolve().parent
sys.path.insert(0, str(GB))
import mapper                                                     # noqa: E402
import page_attribution as PA                                     # noqa: E402
from index_s5_graphrag import Meter, _chain                       # noqa: E402  (reuse the meter)

OUT = GB / "tier1" / "s5_graphrag"
BENCH = {
    "mmlongbench": dict(parsed="/work/gramm-rag-mmlongbench/cache/parsed",
                        d3="/work/gramm-rag-3benchmarks/data/mmlongbench",
                        base="baseline_vector_mmlb.json", qmeta="qmeta_mmlb.json",
                        src="/work/gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json"),
    "docbench":    dict(parsed="/work/gramm-rag-docbench/cache/parsed",
                        d3="/work/gramm-rag-3benchmarks/data/docbench",
                        base="baseline_vector_docbench.json", qmeta="qmeta_docbench.json", src=None),
}
TOPK, COMMUNITY_LEVEL = 3, 2
GEN_KEYS = ("reports", "entities", "relationships", "claims")


def ctx_chars(df):
    import pandas as pd
    if df is None or not len(df):
        return 0
    return int(df.select_dtypes("object").map(lambda x: len(str(x))).sum().sum())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="docbench", choices=list(BENCH))
    ap.add_argument("--tag", default="pilot")
    ap.add_argument("--ceiling", type=float, default=10.0)
    a = ap.parse_args()
    if not os.environ.get("GRAPHRAG_API_KEY", "").startswith("sk-"):
        raise SystemExit("FATAL: GRAPHRAG_API_KEY absent -- refusing to run")
    import litellm, pandas as pd
    import graphrag.api as api
    from graphrag.config.load_config import load_config
    meter = Meter(); litellm.callbacks = [meter]

    S = BENCH[a.bench]; D3 = Path(S["d3"]); PARSED = Path(S["parsed"])
    base = json.loads((D3 / S["base"]).read_text())
    qm = json.loads((D3 / S["qmeta"]).read_text())["rows"]
    n = base["n_questions"]
    questions = ([s["question"] for s in json.loads(Path(S["src"]).read_text())]
                 if S["src"] else list(base["questions"]))
    assert len(questions) == n == len(qm), "index alignment broken across metadata files"

    by_doc = defaultdict(list)
    for i, r in enumerate(qm):
        by_doc[r["doc_id"]].append(i)

    ctx_pages = [[] for _ in range(n)]
    ret_pages = [[] for _ in range(n)]
    align = [None] * n
    attrib_frac = [None] * n
    ctx_split = [None] * n
    missing, empty = [], []
    t0 = time.time()

    for di, (doc, idxs) in enumerate(sorted(by_doc.items()), 1):
        stem = str(doc).replace(".json", "").replace(".pdf", "")
        root = OUT / "stores" / f"{a.bench}__{stem}"
        if not (root / "_INDEX_COMPLETE.json").exists():
            missing.append(doc); continue
        pf = next((PARSED / c for c in (f"{doc}.json", f"{stem}.pdf.json", f"{stem}.json")
                   if (PARSED / c).exists()), None)
        assert pf is not None, f"no parse found for {doc}"
        shingles = PA.build_shingle_index(mapper.load_parse(pf))

        os.chdir(root)
        cfg = load_config(root)
        T = {t: pd.read_parquet(root / "output" / f"{t}.parquet")
             for t in ("entities", "communities", "community_reports", "text_units", "relationships")}
        for i in idxs:
            _resp, ctx = await api.local_search(
                config=cfg, entities=T["entities"], communities=T["communities"],
                community_reports=T["community_reports"], text_units=T["text_units"],
                relationships=T["relationships"], covariates=None,
                community_level=COMMUNITY_LEVEL, response_type="Multiple Paragraphs",
                query=questions[i])
            src = ctx.get("sources")
            chunks = src["text"].astype(str).tolist() if src is not None and "text" in src else []
            ranked, rate, total = PA.attribute_pages(chunks, shingles)[:3]
            gen = sum(ctx_chars(ctx.get(k)) for k in GEN_KEYS)
            attributable = ctx_chars(src)
            align[i] = rate
            attrib_frac[i] = round(attributable / (attributable + gen), 4) if (attributable + gen) else 0.0
            ctx_split[i] = {"sources_chars": attributable, "generated_chars": gen}
            ret_pages[i] = ranked
            ctx_pages[i] = ranked[:TOPK]
            if not ranked:
                empty.append(i)
            if meter.fatal:
                raise SystemExit(f"FATAL: credit exhaustion -- {meter.fatal}")
        if di % 5 == 0 or di == len(by_doc):
            print(f"  [{di:>3}/{len(by_doc)}] docs  {time.time()-t0:6.0f}s  "
                  f"${sum(v['cost_usd'] for v in meter.snapshot().values()):.4f}", flush=True)
        if sum(v["cost_usd"] for v in meter.snapshot().values()) > a.ceiling:
            raise SystemExit(f"FATAL: exceeds ceiling ${a.ceiling}")

    scored = [r for r in align if r is not None]
    if not scored:
        raise SystemExit("FATAL: no questions scored -- no index matched any question's document")
    zero = sum(1 for r in scored if r == 0.0) / len(scored)
    if zero > 0.5:
        raise SystemExit(f"FATAL: {zero:.1%} of questions attributed to ZERO pages -- that is a "
                         f"harness failure, not a retrieval result; refusing to emit it")
    af = [f for f in attrib_frac if f is not None]
    stages = meter.snapshot()
    out = {"RUN_CONFIG": {"RETRIEVER": "s5", "GRAPHRAG_VERSION": "2.7.2",
                          "MODE": "local search", "COMMUNITY_LEVEL": COMMUNITY_LEVEL, "TOPK": TOPK,
                          "SCOPE": "one GraphRAG root per document (C-35)",
                          "PAGE_ATTRIBUTION": "sources(text units) -> ingested-text shingle map; "
                                              "reports/entities/relationships EXCLUDED as generated (§1b)",
                          "GENERATION_OVERHEAD": "api.local_search also generates an answer this "
                                                 "harness discards; kept to use the public path "
                                                 "rather than reimplement context assembly",
                          "BENCHMARK": a.bench},
           "benchmark": a.bench, "n_questions": n, "context_pages": ctx_pages,
           "retrieved_pages": ret_pages, "alignment_rate": align,
           "attributable_fraction": attrib_frac, "context_split": ctx_split,
           "missing_stores": missing, "questions_with_no_attributed_page": empty,
           "alignment_summary": {"n_scored": len(scored), "mean": round(sum(scored)/len(scored), 4),
                                 "zero": sum(1 for r in scored if r == 0.0)},
           "attributable_summary": {"mean": round(sum(af)/len(af), 4) if af else None,
                                    "note": "share of retrieved context characters that are "
                                            "attributable to pages; the remainder is generated"},
           "retrieval_cost_usd": round(sum(v["cost_usd"] for v in stages.values()), 4),
           "stages": stages}
    f = OUT / f"s5_retrieval_{a.bench}{'' if a.tag == 'full' else '_' + a.tag}.json"
    f.write_text(json.dumps(out))
    print(f"\n  wrote {f}")
    print(f"  scored {len(scored)}/{n}  alignment mean {out['alignment_summary']['mean']}  "
          f"zero {out['alignment_summary']['zero']}  missing stores {len(missing)}")
    print(f"  attributable fraction mean {out['attributable_summary']['mean']}  "
          f"cost ${out['retrieval_cost_usd']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
