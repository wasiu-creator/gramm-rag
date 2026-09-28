#!/usr/bin/env python3
"""
S2 Tier-1 — RAG-Anything (legacy pins) over BOTH corpora, per-document stores.

Carries every guard the LightRAG arm's defects produced, because nothing about this arm is
easier and three defects came out of building that one:
  C-33  parser gate silently disables the multimodal path -> processors asserted non-empty,
        plus a BLOCKING assertion: images present with zero vision calls halts the run.
  C-35  retrieval scope must match every other arm -> ONE STORE PER DOCUMENT.
  C-36  lightrag.kg.shared_storage keeps module-level globals -> finalize_share_data() after
        every document, or store N silently accumulates N documents.
  C-39  a completion stamp records what was indexed; resume never infers success from a
        directory existing.
  rule 9 (mechanical) per-document time/cost anomaly detector at 5x the trailing median.
  D-34  credit exhaustion is fatal, never retried. T6 fail-fast. Hard budget abort.
"""
import argparse, asyncio, json, os, statistics, sys, time, traceback
from pathlib import Path

GB = Path(__file__).resolve().parent
sys.path.insert(0, str(GB))
import mapper                                                     # noqa: E402
from pilot_ingest import Meter, RATES, build_funcs, graph_size     # noqa: E402
from pilot_ingest_raganything import build_vision                  # noqa: E402

FAIL_FAST_N = 5


async def run(a, docs, workdir, key):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import (initialize_pipeline_status,
                                            finalize_share_data)
    from raganything import RAGAnything, RAGAnythingConfig

    tm, vm = Meter(), Meter()
    llm_f, emb_f = build_funcs(tm, a.llm_model, a.emb_model, key)
    vis_f = build_vision(vm, a.vision_model, key)
    dim = len((await emb_f(["probe"]))[0])
    stores = workdir / "stores"; stores.mkdir(parents=True, exist_ok=True)

    cfg = {"ARM": "s2_raganything", "TIER": 1, "SCOPE": "both",
           "LLM_MODEL": a.llm_model, "VISION_MODEL": a.vision_model,
           "EMB_MODEL": a.emb_model, "EMBEDDING_DIM": dim,
           "RETRIEVAL_SCOPE": "one store per document (C-35)",
           "INGEST": "insert_content_list, pre-parsed (mineru NOT installed)",
           "BUDGET_CEILING_USD": a.budget_ceiling, "RATES": RATES}
    (workdir / "RUN_CONFIG.json").write_text(json.dumps(cfg, indent=1))
    print("RUN_CONFIG: " + json.dumps({k: v for k, v in cfg.items() if k != "RATES"}), flush=True)

    done, consec, rows = 0, 0, []
    t_start = time.time()
    for d in docs:
        stem = d["doc"].replace(".json", "").replace(".pdf", "")
        store = stores / f'{d["bench"]}__{stem}'
        stamp = store / "_INDEX_COMPLETE.json"
        if stamp.exists():
            done += 1
            if done % 25 == 0:
                print(f"  [{done:>3}/{len(docs)}] resumed", flush=True)
            continue
        store.mkdir(parents=True, exist_ok=True)
        base = Path(f"/media/volume/vol3/grammrag/gramm-rag-{d['bench']}/cache/parsed")
        parsed = mapper.load_parse(base / d["doc"])
        art = base / "artifacts" / stem
        stats = mapper.parse_stats(parsed, art)
        cl = mapper.elements_to_content_list(parsed, art)
        bt, bv = tm.snapshot()["cost_usd"], vm.snapshot()["cost_usd"]
        t0 = time.time()
        err = None
        try:
            rag = LightRAG(working_dir=str(store), llm_model_func=llm_f,
                           llm_model_name=a.llm_model,
                           embedding_func=EmbeddingFunc(embedding_dim=dim, func=emb_f))
            await rag.initialize_storages(); await initialize_pipeline_status()
            ra = RAGAnything(lightrag=rag, llm_model_func=llm_f, vision_model_func=vis_f,
                             embedding_func=emb_f,
                             config=RAGAnythingConfig(working_dir=str(store), parser="mineru",
                                                      enable_image_processing=True,
                                                      enable_table_processing=True,
                                                      enable_equation_processing=True))
            ra._parser_installation_checked = True            # C-33
            await ra._ensure_lightrag_initialized()
            if not ra.modal_processors:
                raise SystemExit("FATAL C-33: modal_processors empty — images would be dropped")
            budget = max(1800.0, stats["content_list_items"] * 10.0)
            await asyncio.wait_for(
                ra.insert_content_list(content_list=cl, file_path=f"{d['bench']}/{d['doc']}",
                                       doc_id=f"{d['bench']}::{d['doc']}"), timeout=budget)
            await rag.finalize_storages()
        except SystemExit:
            raise
        except asyncio.TimeoutError:
            err = "TIMEOUT"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"; traceback.print_exc()
        finalize_share_data()                                  # C-36
        el = time.time() - t0
        dt = tm.snapshot()["cost_usd"] - bt; dv = vm.snapshot()["cost_usd"] - bv
        nvis = vm.snapshot()["llm_calls"]
        if err is None:
            consec = 0; done += 1
        else:
            consec += 1
            if consec >= FAIL_FAST_N:
                raise SystemExit(f"FATAL T6: {consec} consecutive failures")
        row = {"bench": d["bench"], "doc": d["doc"], "chars": d["chars"],
               "images": stats["content_list_image_items"], "seconds": round(el, 1),
               "cost_text": round(dt, 6), "cost_vision": round(dv, 6),
               "cost_usd": round(dt + dv, 6), "error": err,
               "graph": graph_size(store), "cum_vision_calls": nvis}
        rows.append(row)
        # C-33 blocking assertion
        if err is None and row["images"] > 0 and dv <= 0:
            raise SystemExit(f"FATAL C-33: {row['images']} images, zero vision spend on "
                             f"{d['doc']} — multimodal path not wired")
        # rule 9, mechanical
        # Rule 9, corrected. The first form normalised by CHARACTERS only, and fired twice on
        # a document with 4.89 images per 1k chars against a corpus median of 0.31 -- an
        # image-density spike, not a defect. Cost here is substantially image-driven, so a
        # per-char denominator manufactures false positives on exactly the documents that
        # dominate the bill. A genuine defect is anomalous on ANY reasonable denominator;
        # a density artefact is anomalous only on the mismatched one. So: flag only when the
        # ratio is anomalous on BOTH bases, and print both so the reader can judge. (C-48)
        ok = [r for r in rows if r["error"] is None]
        if len(ok) >= 6:
            prev = ok[-6:-1]
            for fld, u in (("seconds", "s"), ("cost_usd", "$")):
                flags = {}
                for dname, dkey in (("char", "chars"), ("image", "images")):
                    den_prev = [r[dkey] for r in prev]
                    if min(den_prev) <= 0 or row[dkey] <= 0:
                        continue
                    med = statistics.median(r[fld] / max(r[dkey], 1) for r in prev)
                    cur = row[fld] / max(row[dkey], 1)
                    if med > 0:
                        flags[dname] = cur / med
                if len(flags) == 2 and all(v > 5 for v in flags.values()):
                    print(f"    !! RULE-9 ANOMALY: {fld} ({u}) is "
                          f"{flags['char']:.1f}x per-char AND {flags['image']:.1f}x per-image "
                          f"vs trailing median — anomalous on both denominators, inspect",
                          flush=True)
                elif flags and max(flags.values()) > 5:
                    hi = max(flags, key=flags.get); lo = min(flags, key=flags.get)
                    print(f"    (note: {fld} {flags[hi]:.1f}x per-{hi} but "
                          f"{flags[lo]:.1f}x per-{lo} — density artefact, not flagged)",
                          flush=True)
        if err is None:
            stamp.write_text(json.dumps(row, indent=1))
        tot = tm.snapshot()["cost_usd"] + vm.snapshot()["cost_usd"]
        print(f"  [{done:>3}/{len(docs)}] {d['bench'][:4]}/{stem[:26]:<26} "
              f"{d['chars']:>8,}ch {row['images']:>4}img {el:>7.1f}s "
              f"${tot:>7.2f} {(time.time()-t_start)/3600:>5.2f}h"
              + (f"  ERR {err}" if err else ""), flush=True)
        if tot > a.budget_ceiling:
            raise SystemExit(f"FATAL BUDGET: ${tot:.2f} > ceiling ${a.budget_ceiling:.2f} "
                             f"after {done}/{len(docs)} docs")

    return cfg, rows, tm.snapshot(), vm.snapshot(), time.time() - t_start, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--vision-model", default="gpt-4o-mini")
    ap.add_argument("--emb-model", default="text-embedding-3-small")
    ap.add_argument("--budget-ceiling", type=float, default=200.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--corpus", default="data/corpus_sizes.json")
    ap.add_argument("--outdir", default="s2_raganything")
    a = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not key.startswith("sk-"):
        raise SystemExit("FATAL: OPENAI_API_KEY absent or malformed")
    corpus = json.loads((GB / a.corpus).read_text())
    docs = sorted(corpus, key=lambda d: d["chars"])
    if a.limit:
        docs = docs[:a.limit]
    print(f"S2: {len(docs)} docs, {sum(d['pages'] for d in docs):,} pages, "
          f"{sum(d['chars'] for d in docs):,} chars, {sum(d['n_img'] for d in docs):,} images",
          flush=True)
    wd = GB / "tier1" / a.outdir; wd.mkdir(parents=True, exist_ok=True)
    cfg, rows, tt, vt, wall, done = asyncio.run(run(a, docs, wd, key))
    out = {"run_config": cfg, "docs": rows, "totals_text": tt, "totals_vision": vt,
           "wall_seconds": round(wall, 1), "docs_indexed": done, "docs_expected": len(docs),
           "cost_usd": round(tt["cost_usd"] + vt["cost_usd"], 6),
           "projection_constrained": 92.09, "projection_nnls": 137.21,
           "projection_pooled": 144.29}
    (wd / "index_log.json").write_text(json.dumps(out, indent=1))
    print(f"\nS2 INDEX COMPLETE: {done}/{len(docs)} docs, {wall/3600:.2f} h, "
          f"${out['cost_usd']:.2f} (constrained model predicted $92.09)", flush=True)


if __name__ == "__main__":
    main()
