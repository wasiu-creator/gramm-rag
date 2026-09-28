"""
WP-1 Tier 1 -- LightRAG 1.5.7 indexing over MMLongBench (135 docs / 6,522 pages).

RETRIEVAL SCOPE: ONE STORE PER DOCUMENT, and this is the control, not an optimisation.
Every other arm ranks within the question's own document -- ColPali's `retrieve.py` loops
`by_doc` and ranks only that document's pages; GraMM and flat-vector retrieve over that
document's element graph. MMLongBench questions are document-scoped by construction.
A single corpus-wide LightRAG store would therefore make this arm solve a strictly harder
problem -- locate the document AND the passage -- so retrieval would differ in SCOPE as
well as in METHOD. That is two changes, not one, and the resulting deficit would be
misread as LightRAG retrieving badly. See C-35.

Scope decision recorded in GATE_G_RESULT.md: MMLongBench only, $5.06 projected, the one
cell that clears the pre-registered $10 gate. The gate was NOT raised.

Unlike the pilot, this run produces reported numbers, so it carries the full guard set:
  T6  fail-fast: FAIL_FAST_N consecutive document failures aborts rather than producing a
      store with silent holes in it.
  T7  checkpoint/resume: LightRAG's own doc-status store skips completed documents, so an
      interrupted run resumes. Verified by asserting the processed-count grows.
  D-34 credit exhaustion is fatal, never retried (imported from pilot_ingest).
  D-32 no silent fallback anywhere; missing key or unlisted model rate raises.
  BUDGET a hard ceiling aborts mid-run. A projection is not a permission slip: if the real
      cost diverges upward from $5.06 the run stops rather than quietly spending the cap.
  R-42 every knob stamped into RUN_CONFIG and written beside the index.
"""
import argparse, asyncio, json, os, shutil, sys, time, traceback
from pathlib import Path

GB = Path(__file__).resolve().parent
sys.path.insert(0, str(GB))
import mapper                                                    # noqa: E402
from pilot_ingest import Meter, RATES, build_funcs, graph_size    # noqa: E402

# Per-arm ceiling from Gate H: abort rather than spend into the envelope. DocBench is
# projected at $18.00 indexing; 1.8x that is $32. Passed explicitly per run.
BUDGET_CEILING_USD = 9.00
FAIL_FAST_N = 5
BATCH = 10


async def run(a, docs, workdir, key):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import (initialize_pipeline_status,
                                            finalize_share_data)

    meter = Meter()
    llm_f, emb_f = build_funcs(meter, a.llm_model, a.emb_model, key)
    dim = len((await emb_f(["dimension probe"]))[0])
    stores = workdir / "stores"
    stores.mkdir(parents=True, exist_ok=True)

    async def make_rag(store_dir):
        r = LightRAG(working_dir=str(store_dir), llm_model_func=llm_f,
                     llm_model_name=a.llm_model,
                     embedding_func=EmbeddingFunc(embedding_dim=dim, func=emb_f))
        await r.initialize_storages()
        await initialize_pipeline_status()
        return r
    # probe store lives OUTSIDE stores/ so the isolation assertion sees only real
    # documents; a stray directory there made the assertion fail, correctly.
    rag = await make_rag(workdir / "_probe")

    cfg = {"ARM": "lightrag_current", "TIER": 1, "SCOPE": a.bench,
           "LIGHTRAG_VERSION": __import__("lightrag").__version__,
           "LLM_MODEL": a.llm_model, "EMB_MODEL": a.emb_model, "EMBEDDING_DIM": dim,
           "CHUNK_TOKEN_SIZE": rag.chunk_token_size,
           "GLEANING": rag.entity_extract_max_gleaning,
           "LLM_MAX_ASYNC": rag.llm_model_max_async,
           "MAX_PARALLEL_INSERT": rag.max_parallel_insert,
           "RETRIEVAL_MODE_PREREGISTERED": "hybrid",
           "RETRIEVAL_SCOPE": "one store per document -- matches every other arm (C-35)",
           "INGEST": "mapper.elements_to_text (canonical, shared with every text-side arm)",
           "BUDGET_CEILING_USD": BUDGET_CEILING_USD, "FAIL_FAST_N": FAIL_FAST_N,
           "GATE_G": "MMLongBench-only scope; $10 gate NOT raised", "RATES": RATES}
    (workdir / "RUN_CONFIG.json").write_text(json.dumps(cfg, indent=1))
    print("RUN_CONFIG: " + json.dumps({k: v for k, v in cfg.items() if k != "RATES"}), flush=True)

    done, consec_fail, rows = 0, 0, []
    base = Path(f"/media/volume/vol3/grammrag/gramm-rag-{a.bench}/cache/parsed")
    t_start = time.time()

    for d in docs:
        stem = d["doc"].replace(".json", "").replace(".pdf", "")
        store = stores / stem
        # T7 resume: a finished store carries its own completion stamp. Never trust the
        # mere existence of a directory -- R-42 is the precedent for a cache that reports
        # success for artefacts it should have rejected.
        stamp = store / "_INDEX_COMPLETE.json"
        if stamp.exists():
            done += 1
            print(f"  [{done:>3}/{len(docs)}] {stem[:44]:<44} resumed (already complete)",
                  flush=True)
            continue
        store.mkdir(parents=True, exist_ok=True)
        # C-36: seed the extraction cache so a re-index does not re-pay for identical
        # LLM calls. Keys are hashes of the extraction prompt, so a hit is exact reuse of
        # the same computation on the same input -- not a substitute for it.
        if a.seed_cache and not (store / "kv_store_llm_response_cache.json").exists():
            shutil.copy(a.seed_cache, store / "kv_store_llm_response_cache.json")
        parsed = mapper.load_parse(base / d["doc"])
        text = mapper.elements_to_text(parsed)
        before = meter.snapshot()["cost_usd"]
        t0 = time.time()
        try:
            rag = await make_rag(store)
            await rag.ainsert(text, ids=f"{a.bench}::{d['doc']}", file_paths=d["doc"])
            await rag.finalize_storages()
            # C-36: lightrag.kg.shared_storage holds MODULE-LEVEL globals. Without this
            # reset, every document after the first accumulates into the SAME in-memory
            # namespaces and each store persists the accumulation -- store N ends up
            # holding N documents. 135 directories, 135 completion stamps, 135 graphs,
            # all looking correct. This one line is the difference between per-document
            # stores and a progressively contaminated corpus-wide one.
            finalize_share_data()
            consec_fail = 0
            done += 1
            err = None
        except SystemExit:
            raise
        except Exception as e:
            consec_fail += 1
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            print(f"  DOC FAILED {stem}: {err}", flush=True)
            if consec_fail >= FAIL_FAST_N:
                raise SystemExit(f"FATAL T6: {consec_fail} consecutive document failures")
        el, snap = time.time() - t0, meter.snapshot()
        g = graph_size(store)
        row = {"doc": d["doc"], "chars": d["chars"], "pages": d["pages"],
               "seconds": round(el, 1), "cost_usd": round(snap["cost_usd"] - before, 6),
               "cum_cost_usd": snap["cost_usd"], "graph": g, "error": err}
        rows.append(row)
        # Rule 9, MECHANICAL. The prose version ("an order-of-magnitude anomaly on a quantity
        # with a physical prior is a defect until proven otherwise") existed when the C-36
        # re-index printed 13.1 s against ~2 s for a document's neighbours, and nobody ran it
        # against that line -- C-39. A rule applied only when someone remembers is closer to
        # an insight than to a check, so it lives here now instead of in a document.
        ok_rows = [r for r in rows if r["error"] is None]
        if len(ok_rows) >= 6:
            import statistics as _st
            prev = ok_rows[-6:-1]
            for field, unit in (("seconds", "s"), ("cost_usd", "$")):
                med = _st.median(r[field] / max(r["chars"], 1) for r in prev)
                cur = row[field] / max(row["chars"], 1)
                if med > 0 and cur > 5 * med:
                    print(f"    !! RULE-9 ANOMALY: {field} per char is {cur/med:.1f}x the "
                          f"median of the previous 5 documents ({unit}). Not fatal, but this "
                          f"is the C-39 signature -- inspect before trusting this store.",
                          flush=True)
        if err is None:
            stamp.write_text(json.dumps(row, indent=1))
        print(f"  [{done:>3}/{len(docs)}] {stem[:38]:<38} {d['chars']:>8,}ch "
              f"{el:6.1f}s ${snap['cost_usd']:>7.4f} ent={g['entities']} "
              f"{(time.time()-t_start)/60:5.1f}m", flush=True)
        if snap["cost_usd"] > BUDGET_CEILING_USD:
            raise SystemExit(f"FATAL BUDGET: ${snap['cost_usd']:.2f} exceeded ceiling "
                             f"${BUDGET_CEILING_USD:.2f} after {done}/{len(docs)} docs. "
                             f"Stopping rather than spending into the cap.")

    return cfg, rows, meter.snapshot(), time.time() - t_start, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--emb-model", default="text-embedding-3-small")
    ap.add_argument("--seed-cache", default=None,
                    help="kv_store_llm_response_cache.json to seed into each fresh store")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget-ceiling", type=float, default=None)
    ap.add_argument("--outdir", default="lightrag_mmlongbench")
    ap.add_argument("--bench", default="mmlongbench",
                    choices=["mmlongbench", "docbench", "mpdocvqa"])
    # P3 reads a SEPARATE corpus file. data/corpus_sizes.json is a canonical input that
    # other shipped indexers read without filtering by bench; appending to it would change
    # what a re-run of another arm indexes (R-42/C-53). Additive flag, default unchanged.
    ap.add_argument("--corpus", default="data/corpus_sizes.json")
    a = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not key.startswith("sk-"):
        raise SystemExit("FATAL: OPENAI_API_KEY absent or malformed -- refusing to run")

    global BUDGET_CEILING_USD
    if a.budget_ceiling:
        BUDGET_CEILING_USD = a.budget_ceiling
    corpus = json.loads((GB / a.corpus).read_text())
    docs = sorted([d for d in corpus if d["bench"] == a.bench], key=lambda d: d["chars"])
    print(f"{a.bench}: {len(docs)} docs, {sum(d['pages'] for d in docs):,} pages, "
          f"{sum(d['chars'] for d in docs):,} chars", flush=True)

    if a.limit:
        docs = docs[:a.limit]
    workdir = GB / "tier1" / a.outdir
    workdir.mkdir(parents=True, exist_ok=True)
    cfg, rows, total, wall, done = asyncio.run(run(a, docs, workdir, key))
    out = {"run_config": cfg, "batches": rows, "totals": total,
           "wall_seconds": round(wall, 1), "docs_indexed": done, "docs_expected": len(docs),
           "projected_usd": {"mmlongbench": 5.06, "docbench": 18.00}.get(a.bench),
           "benchmark": a.bench}
    (workdir / "index_log.json").write_text(json.dumps(out, indent=1))
    # The projection is benchmark-specific; hardcoding MMLongBench's $5.06 made the DocBench
    # completion line read as a 3x overrun ($15.23 "projected $5.06") when it was in fact
    # under its own $18.00 gate. A constant that is only correct for one input is a wrong
    # number waiting for the second input -- C-34's shape in a log line.
    PROJ = {"mmlongbench": 5.06, "docbench": 18.00}.get(a.bench)
    proj_txt = f"projected ${PROJ:.2f}" if PROJ is not None else "no pre-registered projection"
    print(f"\nINDEX COMPLETE: {done}/{len(docs)} docs, {wall/60:.1f} min, "
          f"${total['cost_usd']:.4f} ({proj_txt} for {a.bench})", flush=True)


if __name__ == "__main__":
    main()
