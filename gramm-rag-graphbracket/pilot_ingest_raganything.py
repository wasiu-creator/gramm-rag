"""
WP-1 / V2 -- 5-document pilot ingest for the RAG-Anything (legacy) arm.

Same Gate G contract as pilot_ingest.py: authorises or kills, never a reported number.

Two things differ from the LightRAG arm and both are deliberate:
  1. Ingest goes through RAGAnything.insert_content_list() with our OWN MinerU parse
     (mapper.elements_to_content_list). The document is NEVER re-parsed -- mineru is not
     even installed in this venv, which makes re-parsing impossible rather than merely
     discouraged. That is what preserves harness identity against the text-side arms.
  2. Images reach a vision model for description, so a THIRD cost line appears that the
     text-only arm does not have. It is metered separately, because whether the multimodal
     path is affordable is precisely the open question.

Version pinning: lightrag-hku is pinned to 1.4.9.2, the release contemporaneous with
arXiv:2510.12323 (2025-10-11). raganything 1.2.8 leaves the dependency UNPINNED, so a
plain install today resolves to lightrag-hku 1.5.7 -- eleven months newer than the system
whose published numbers (0.428 / 0.634) this arm exists to sit beside. See register C-32.
"""
import argparse, asyncio, json, os, sys, time, traceback
from pathlib import Path

GB = Path(__file__).resolve().parent
sys.path.insert(0, str(GB))
import mapper                       # noqa: E402
from pilot_ingest import Meter, RATES, fatal_if_credit_exhaustion, graph_size, \
                         CORPUS_CHARS, CORPUS_DOCS, CORPUS_PAGES   # noqa: E402


def build_vision(meter, model, api_key):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=api_key)

    async def vision_model_func(prompt, system_prompt=None, history_messages=[],
                                image_data=None, messages=None, **kw):
        kw.pop("hashing_kv", None); kw.pop("keyword_extraction", None)
        if messages is not None:
            msgs = messages
        elif image_data:
            content = [{"type": "text", "text": prompt},
                       {"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}]
            msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
                 + [{"role": "user", "content": content}]
        else:
            msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
                 + list(history_messages) + [{"role": "user", "content": prompt}]
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=model, messages=msgs, temperature=0.0, **kw)
                meter.add_llm(model, r.usage)
                return r.choices[0].message.content
            except Exception as e:
                fatal_if_credit_exhaustion(e)          # D-34
                if attempt == 4:
                    raise                              # D-32: never degrade
                await asyncio.sleep(2 ** attempt)
    return vision_model_func


async def run(arm, docs, workdir, llm_model, vis_model, emb_model, api_key, limit_docs):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from raganything import RAGAnything, RAGAnythingConfig
    from pilot_ingest import build_funcs

    text_meter, vis_meter = Meter(), Meter()
    llm_f, emb_f = build_funcs(text_meter, llm_model, emb_model, api_key)
    vis_f = build_vision(vis_meter, vis_model, api_key)
    dim = len((await emb_f(["dimension probe"]))[0])

    rag = LightRAG(working_dir=str(workdir), llm_model_func=llm_f, llm_model_name=llm_model,
                   embedding_func=EmbeddingFunc(embedding_dim=dim, func=emb_f))
    await rag.initialize_storages()
    await initialize_pipeline_status()

    cfg_obj = RAGAnythingConfig(working_dir=str(workdir), parser="mineru",
                                enable_image_processing=True, enable_table_processing=True,
                                enable_equation_processing=True)
    ra = RAGAnything(lightrag=rag, llm_model_func=llm_f, vision_model_func=vis_f,
                     embedding_func=emb_f, config=cfg_obj)

    # ---- DEFECT WORKAROUND, documented because it silently disables the whole
    # multimodal path (register C-33) ----------------------------------------------
    # _ensure_lightrag_initialized() checks parser installation FIRST and, on failure,
    # RETURNS {"success": False} rather than raising. insert_content_list() ignores that
    # return value and proceeds, so _initialize_processors() is never reached,
    # modal_processors stays empty, and every image is dropped with a WARNING while the
    # document is still marked complete and the process exits 0. The gate fires even on
    # the pre-parsed ingest route, which by construction never invokes a parser.
    # We assert the gate is moot rather than installing a parser we must never use:
    # mineru stays absent, so re-parsing remains structurally impossible (harness identity).
    ra._parser_installation_checked = True
    await ra._ensure_lightrag_initialized()
    if not ra.modal_processors:
        raise SystemExit("FATAL: modal_processors empty -- the multimodal path would be a "
                         "no-op and every image would be silently dropped")
    print(f"  modal processors registered: {sorted(ra.modal_processors)}", flush=True)

    import lightrag as _lr, raganything as _ra
    cfg = {"ARM": arm, "LIGHTRAG_VERSION": _lr.__version__,
           "RAGANYTHING_VERSION": getattr(_ra, "__version__", "1.2.8"),
           "LLM_MODEL": llm_model, "VISION_MODEL": vis_model, "EMB_MODEL": emb_model,
           "CHUNK_TOKEN_SIZE": rag.chunk_token_size,
           "GLEANING": rag.entity_extract_max_gleaning,
           "LLM_MAX_ASYNC": rag.llm_model_max_async, "EMBEDDING_DIM": dim,
           "INGEST_API": "insert_content_list (pre-parsed; mineru NOT installed)",
           "RATES": RATES, "PILOT_ONLY": "authorises or kills; never a reported number"}
    print("  RUN_CONFIG: " + json.dumps({k: v for k, v in cfg.items() if k != "RATES"}), flush=True)

    results = []
    tprev, vprev = text_meter.snapshot(), vis_meter.snapshot()
    ordered = sorted(docs, key=lambda d: d["chars"])
    if limit_docs:
        ordered = ordered[:limit_docs]

    for i, d in enumerate(ordered, 1):
        base = Path(f"/media/volume/vol3/grammrag/gramm-rag-{d['bench']}/cache/parsed")
        parsed = mapper.load_parse(base / d["doc"])
        art = base / "artifacts" / d["doc"].replace(".json", "").replace(".pdf", "")
        stats = mapper.parse_stats(parsed, art)
        content_list = mapper.elements_to_content_list(parsed, art)

        print(f"\n[{i}/{len(ordered)}] p{d['pct']} {d['bench']}/{d['doc']} {d['pages']}pg "
              f"{stats['content_list_items']} items "
              f"({stats['content_list_text_items']}txt/{stats['content_list_image_items']}img)",
              flush=True)
        budget = max(900.0, stats["content_list_items"] * 8.0)
        t0 = time.time()
        try:
            await asyncio.wait_for(
                ra.insert_content_list(content_list=content_list,
                                       file_path=f"{d['bench']}/{d['doc']}",
                                       doc_id=f"{d['bench']}::{d['doc']}"), timeout=budget)
            err = None
        except asyncio.TimeoutError:
            err = f"TIMEOUT after {budget:.0f}s"
        except SystemExit:
            raise
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        el = time.time() - t0
        td, vd = text_meter.delta(tprev), vis_meter.delta(vprev)
        tprev, vprev = text_meter.snapshot(), vis_meter.snapshot()
        g = graph_size(workdir)
        row = {**{k: d[k] for k in ("pct", "bench", "doc", "pages", "chars")},
               "parse_stats": stats, "text_chars": stats["text_chars"],
               "n_images": stats["content_list_image_items"],
               "seconds": round(el, 2), "error": err,
               "text_side": td, "vision_side": vd,
               "cost_usd": round(td["cost_usd"] + vd["cost_usd"], 6), "graph_after": g}
        if err is None and row["n_images"] > 0 and vd["llm_calls"] == 0:
            raise SystemExit(
                f"FATAL blocking assertion: {row['n_images']} images in "
                f"{d['bench']}/{d['doc']} produced ZERO vision calls. The multimodal path "
                f"is not wired -- this is the C-24 class (a green run from a component that "
                f"is not doing its job). Refusing to continue.")
        results.append(row)
        print(f"    {el:7.1f}s  text {td['llm_calls']:4d}call ${td['cost_usd']:.4f}  |  "
              f"vision {vd['llm_calls']:4d}call ${vd['cost_usd']:.4f}  |  "
              f"total ${row['cost_usd']:.4f}  ent={g['entities']}"
              + (f"  ERROR {err}" if err else ""), flush=True)
    return cfg, results, text_meter.snapshot(), vis_meter.snapshot()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="raganything_legacy")
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--vision-model", default="gpt-4o-mini")
    ap.add_argument("--emb-model", default="text-embedding-3-small")
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--docs", default="data/pilot_docs.json",
                    help="document list to ingest (relative to the graphbracket dir)")
    ap.add_argument("--only-new-vs", default=None,
                    help="skip documents already present in this doc list (augmentation runs)")
    a = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not key.startswith("sk-"):
        raise SystemExit("FATAL: OPENAI_API_KEY absent or malformed -- refusing to run")

    docs = json.loads((GB / a.docs).read_text())
    if a.only_new_vs:
        seen = {d["doc"] for d in json.loads((GB / a.only_new_vs).read_text())}
        before = len(docs)
        docs = [d for d in docs if d["doc"] not in seen]
        print(f"  augmentation: {before} -> {len(docs)} documents not already ingested", flush=True)
    workdir = GB / "pilot" / a.arm / "store"
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg, rows, tt, vt = asyncio.run(run(a.arm, docs, workdir, a.llm_model, a.vision_model,
                                        a.emb_model, key, a.limit_docs))
    ok = [r for r in rows if not r["error"]]
    out = {"run_config": cfg, "pilot_wall_seconds": round(time.time() - t0, 1), "docs": rows,
           "totals_text": tt, "totals_vision": vt,
           "pilot_totals": {"docs_ok": len(ok), "docs_failed": len(rows) - len(ok),
                            "text_chars": sum(r["text_chars"] for r in ok),
                            "images": sum(r["n_images"] for r in ok),
                            "ingest_seconds": round(sum(r["seconds"] for r in ok), 1),
                            "cost_text_usd": round(tt["cost_usd"], 6),
                            "cost_vision_usd": round(vt["cost_usd"], 6),
                            "cost_usd": round(tt["cost_usd"] + vt["cost_usd"], 6)},
           "corpus": {"docs": CORPUS_DOCS, "pages": CORPUS_PAGES, "chars": CORPUS_CHARS,
                      "images": 14581}}
    p = GB / "pilot" / a.arm / "pilot_log.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}\npilot: {len(ok)}/{len(rows)} docs, "
          f"${out['pilot_totals']['cost_usd']:.4f} "
          f"(text ${tt['cost_usd']:.4f} + vision ${vt['cost_usd']:.4f})")


if __name__ == "__main__":
    main()
