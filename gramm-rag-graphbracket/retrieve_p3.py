#!/usr/bin/env python3
"""D2b -- P3 retrieval for the three API graph arms over MP-DocVQA.

Driven from tier1/p3_mpdocvqa/subsample.json, NOT from directory contents. LightRAG's store
tree holds 7 extra documents indexed before the 500-char floor existed; retrieving over
directory contents would give that arm 7 documents of coverage the other two lack and break
pairing in its favour. The subsample is the single source of truth for which questions and
which documents every arm sees.

Localisation on MP-DocVQA is single-gold-page: each question carries answer_page_idx. So the
metric is page-hit@k -- did the retrieved page set contain the gold page -- rather than
MMLongBench's multi-page ctx-hit or DocBench's quote-recall. Stated here because the metric
differs per corpus and the pre-registration forbids pooling across benchmarks.
"""
import argparse, asyncio, json, os, sys, time
from collections import defaultdict
from pathlib import Path

GB = Path(__file__).resolve().parent
sys.path.insert(0, str(GB))
import mapper                                                    # noqa: E402
import page_attribution as PA                                    # noqa: E402

ROOT = GB.parent
PARSED = ROOT / "gramm-rag-mpdocvqa/cache/parsed"
SUB = GB / "tier1/p3_mpdocvqa/subsample.json"
OUT = GB / "tier1/p3_mpdocvqa"
TOPK = 3
ARMS = {"lightrag": dict(kind="lightrag", store="p3_lightrag_full", prefix=""),
        "s2":       dict(kind="lightrag", store="p3_s2_full",       prefix="mpdocvqa__"),
        "s5":       dict(kind="graphrag", store="p3_s5_full",       prefix="mpdocvqa__")}
COMMUNITY_LEVEL = 2


async def main(a):
    S = ARMS[a.arm]
    sub = json.loads(SUB.read_text())
    qs = sub["questions"]
    if a.limit:
        qs = qs[:a.limit]
    by_doc = defaultdict(list)
    for i, q in enumerate(qs):
        by_doc[q["doc_id"]].append(i)
    sdir = GB / "tier1" / S["store"] / "stores"
    pre = S["prefix"]

    # every arm must resolve the identical document set -- assert, do not assume
    missing = [d for d in by_doc if not (sdir / f"{pre}{d}" / "_INDEX_COMPLETE.json").exists()]
    assert not missing, (f"{a.arm}: {len(missing)} subsample documents have no store "
                         f"(e.g. {missing[:5]}) -- refusing to retrieve over a partial corpus")
    print(f"  {a.arm}: {len(qs)} questions over {len(by_doc)} documents, all stores present",
          flush=True)

    # SEMANTIC-NULL CONTROL. The existing validity assertion (>50% of questions attribute to
    # some page) passes trivially for a retriever that ignores the query and returns the same
    # document-level context every time. A nonsense query discriminates: real questions pull
    # ~50k chars of context, a semantic null pulls ~70. Measured on the first document of the
    # arm, before any scored retrieval, and asserted rather than logged.
    NONSENSE = "banana helicopter quantum tuesday"

    ret_pages = [[] for _ in qs]
    ctx_pages = [[] for _ in qs]
    align = [None] * len(qs)
    empty = []
    # A crashed query and a query that legitimately retrieved nothing are different events
    # that produce the same artefact (no attributed pages). Conflating them scores a harness
    # failure as a retrieval miss -- the D-32/D-34 shape, here in our own code rather than a
    # vendor's. RAG-Anything's pinned LightRAG v1.4.9.2 raises on some queries and the vendor
    # swallows it, returning None. Recorded as its own category, excluded from the scored
    # denominator, and reported as a count. Applied to EVERY arm, not only the one where it
    # surfaced: the conflation is latent in the harness regardless of which arm trips it.
    query_errors = []
    t0 = time.time()
    cost = 0.0

    if S["kind"] == "lightrag":
        from lightrag import LightRAG, QueryParam
        from lightrag.utils import EmbeddingFunc
        from lightrag.kg.shared_storage import initialize_pipeline_status, finalize_share_data
        from pilot_ingest import Meter, build_funcs
        key = os.environ["OPENAI_API_KEY"]
        meter = Meter()
        llm_f, emb_f = build_funcs(meter, "gpt-4o-mini", "text-embedding-3-small", key)
        dim = len((await emb_f(["dimension probe"]))[0])
        for di, (doc, idxs) in enumerate(sorted(by_doc.items()), 1):
            parsed = mapper.load_parse(PARSED / f"{doc}.json")
            shingles = PA.build_shingle_index(parsed)
            rag = LightRAG(working_dir=str(sdir / f"{pre}{doc}"), llm_model_func=llm_f,
                           llm_model_name="gpt-4o-mini",
                           embedding_func=EmbeddingFunc(embedding_dim=dim, func=emb_f))
            await rag.initialize_storages(); await initialize_pipeline_status()
            if di == 1:
                real = await rag.aquery(qs[idxs[0]]["question"],
                                        param=QueryParam(mode="hybrid", only_need_context=True))
                null = await rag.aquery(NONSENSE,
                                        param=QueryParam(mode="hybrid", only_need_context=True))
                lr_, ln_ = len(str(real)), len(str(null))
                print(f"  semantic-null control: real={lr_} null={ln_} ratio={lr_/max(ln_,1):.1f}x",
                      flush=True)
                assert lr_ > 4 * ln_, (
                    f"semantic-null control FAILED: a nonsense query returned {ln_} chars against "
                    f"{lr_} for a real question. The retriever is not responding to the query; "
                    f"refusing to emit numbers from it.")
            import retrieve_lightrag as RL
            for i in idxs:
                try:
                    ctx = await rag.aquery(qs[i]["question"],
                                           param=QueryParam(mode="hybrid", only_need_context=True))
                except Exception as e:
                    query_errors.append(dict(i=i, doc=doc, err=f"{type(e).__name__}: {e}"[:200]))
                    continue
                if not isinstance(ctx, str):
                    query_errors.append(dict(i=i, doc=doc,
                                             err=f"aquery returned {type(ctx).__name__}, not str"))
                    continue
                chunks = RL.parse_chunks(ctx)
                ranked, rate, _ = PA.attribute_pages(chunks, shingles)[:3]
                align[i] = rate; ret_pages[i] = ranked; ctx_pages[i] = ranked[:TOPK]
                if not ranked: empty.append(i)
            await rag.finalize_storages(); finalize_share_data()   # C-36
            if di % 20 == 0 or di == len(by_doc):
                print(f"  [{di}/{len(by_doc)}] docs {time.time()-t0:6.0f}s "
                      f"${meter.snapshot()['cost_usd']:.4f}", flush=True)
        cost = meter.snapshot()["cost_usd"]
    else:
        import pandas as pd, litellm
        import graphrag.api as api
        from graphrag.config.load_config import load_config
        from index_s5_graphrag import Meter
        meter = Meter(); litellm.callbacks = [meter]
        T_NAMES = ("entities", "communities", "community_reports", "text_units", "relationships")
        for di, (doc, idxs) in enumerate(sorted(by_doc.items()), 1):
            parsed = mapper.load_parse(PARSED / f"{doc}.json")
            shingles = PA.build_shingle_index(parsed)
            root = sdir / f"{pre}{doc}"
            os.chdir(root)
            cfg = load_config(root)
            T = {t: pd.read_parquet(root / "output" / f"{t}.parquet") for t in T_NAMES}
            for i in idxs:
              try:
                _resp, ctx = await api.local_search(
                    config=cfg, entities=T["entities"], communities=T["communities"],
                    community_reports=T["community_reports"], text_units=T["text_units"],
                    relationships=T["relationships"], covariates=None,
                    community_level=COMMUNITY_LEVEL, response_type="Multiple Paragraphs",
                    query=qs[i]["question"])
              except Exception as e:
                query_errors.append(dict(i=i, doc=doc, err=f"{type(e).__name__}: {e}"[:200]))
                continue
              if True:
                src = ctx.get("sources")
                chunks = src["text"].astype(str).tolist() if src is not None and "text" in src else []
                ranked, rate, _ = PA.attribute_pages(chunks, shingles)[:3]
                align[i] = rate; ret_pages[i] = ranked; ctx_pages[i] = ranked[:TOPK]
                if not ranked: empty.append(i)
                if meter.fatal:
                    raise SystemExit(f"FATAL: credit exhaustion -- {meter.fatal}")
            if di % 20 == 0 or di == len(by_doc):
                c = sum(v["cost_usd"] for v in meter.snapshot().values())
                print(f"  [{di}/{len(by_doc)}] docs {time.time()-t0:6.0f}s ${c:.4f}", flush=True)
        cost = sum(v["cost_usd"] for v in meter.snapshot().values())

    # BLOCKING validity assertion (constraint 7 / C-33 shape)
    scored = [r for r in align if r is not None]
    zero = sum(1 for r in scored if r == 0.0) / max(len(scored), 1)
    if zero > 0.5:
        raise SystemExit(f"FATAL: {zero:.1%} of questions attributed to ZERO pages -- that is a "
                         f"harness failure, not a retrieval result; refusing to emit it")

    # answer_page_idx indexes the question's PAGE WINDOW, not the document; the absolute
    # gold page is page_window[0] + answer_page_idx, precomputed and verified in the
    # subsample against window_page_names. Parse page_no is 0-based, so no +1.
    err_idx = {e["i"] for e in query_errors}
    hit = sum(1 for i, q in enumerate(qs)
              if i not in err_idx and q["gold_page_absolute"] is not None
              and q["gold_page_absolute"] in ctx_pages[i])
    hit_any = sum(1 for i, q in enumerate(qs)
                  if i not in err_idx and q["gold_page_absolute"] is not None
                  and q["gold_page_absolute"] in ret_pages[i])
    scoreable = sum(1 for i, q in enumerate(qs)
                    if i not in err_idx and q["gold_page_absolute"] is not None)
    out = dict(RUN_CONFIG=dict(RETRIEVER=a.arm, BENCHMARK="mpdocvqa", TOPK=TOPK,
                               MODE="hybrid" if S["kind"] == "lightrag" else "local search",
                               SCOPE="one store per document (C-35)",
                               DRIVEN_BY="subsample.json, not directory contents",
                               METRIC="page-hit@3 vs answer_page_idx (single gold page)"),
               benchmark="mpdocvqa", arm=a.arm, n_questions=len(qs),
               context_pages=ctx_pages, retrieved_pages=ret_pages, alignment_rate=align,
               questions_with_no_attributed_page=empty,
               query_errors=query_errors, n_query_errors=len(query_errors),
               query_error_rate=round(len(query_errors) / max(len(qs), 1), 4),
               scored_denominator_note=("questions whose query raised are EXCLUDED from the "
                                        "denominator, not scored as misses"),
               page_hit_at_3=round(hit / max(scoreable, 1), 4),
               page_hit_any_rank=round(hit_any / max(scoreable, 1), 4),
               n_scoreable=scoreable, n_hit=hit,
               alignment_summary=dict(n_scored=len(scored),
                                      mean=round(sum(scored)/max(len(scored),1), 4),
                                      zero=sum(1 for r in scored if r == 0.0)),
               retrieval_cost_usd=round(cost, 4), wall_seconds=round(time.time()-t0, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"p3_retrieval_{a.arm}.json").write_text(json.dumps(out, indent=1))
    print(f"  {a.arm}: page-hit@3 {out['page_hit_at_3']:.4f} ({hit}/{scoreable})  "
          f"qerr {len(query_errors)}  "
          f"align {out['alignment_summary']['mean']:.4f}  ${cost:.4f}  "
          f"{out['wall_seconds']/60:.1f}m", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--limit", type=int, default=None)
    asyncio.run(main(ap.parse_args()))
