"""
WP-1 / V2 -- 5-document pilot ingest, instrumented for PROJECTION ONLY.

Gate G, pre-registered before any install: this script authorises or kills a comparator.
It NEVER produces a reported number. R-50/D-35 precedent, and its third recurrence in this
project (a 60-question sample gave joint AUC 0.731 where 1,088 gave 0.568).

Outputs per document: wall-clock, prompt/completion tokens, embedding tokens, $ at listed
rates, and the running graph size -- so the projection can be checked for superlinearity
rather than assumed linear.

Guards carried in from precedent, all of them load-bearing:
  D-34  OpenAI reports EXHAUSTED CREDITS as RateLimitError, the same class as ordinary
        throttling. 342/1,102 judge calls once silently scored 0 and produced a
        complete-looking, understated result. Any RateLimitError whose message mentions
        quota/credit/billing is FATAL here, never retried.
  D-32  No silent fallback of any kind. A missing key, a wrong model, an unreachable
        endpoint -> raise. Never degrade to something that returns a plausible number.
  R-42  Every knob is stamped into RUN_CONFIG and written beside the results.
"""
import argparse, asyncio, json, os, sys, time, traceback
from pathlib import Path

GB = Path(__file__).resolve().parent
sys.path.insert(0, str(GB))
import mapper  # noqa: E402

# Listed rates, stamped so a later price change cannot silently rewrite history.
RATES = {"gpt-4o-mini":       {"in": 0.15 / 1e6, "out": 0.60 / 1e6},
         "gpt-4.1-mini":      {"in": 0.40 / 1e6, "out": 1.60 / 1e6},
         "text-embedding-3-small": {"in": 0.02 / 1e6, "out": 0.0}}
CORPUS_CHARS, CORPUS_DOCS, CORPUS_PAGES = 45_876_082, 364, 20_530


class Meter:
    """Exact token accounting from provider usage objects -- not estimated from text."""
    def __init__(self):
        self.calls = self.pin = self.pout = self.emb_tok = self.emb_calls = 0
        self.by_model = {}

    def add_llm(self, model, usage):
        self.calls += 1
        pi, po = usage.prompt_tokens, usage.completion_tokens
        self.pin += pi; self.pout += po
        m = self.by_model.setdefault(model, {"calls": 0, "in": 0, "out": 0})
        m["calls"] += 1; m["in"] += pi; m["out"] += po

    def add_emb(self, model, usage):
        self.emb_calls += 1
        self.emb_tok += usage.prompt_tokens
        m = self.by_model.setdefault(model, {"calls": 0, "in": 0, "out": 0})
        m["calls"] += 1; m["in"] += usage.prompt_tokens

    def cost(self):
        c = 0.0
        for model, m in self.by_model.items():
            r = RATES.get(model)
            if r is None:
                raise RuntimeError(f"no listed rate stamped for model {model!r} -- refusing "
                                   f"to report a cost we cannot substantiate")
            c += m["in"] * r["in"] + m["out"] * r["out"]
        return c

    def snapshot(self):
        return {"llm_calls": self.calls, "prompt_tokens": self.pin,
                "completion_tokens": self.pout, "embedding_calls": self.emb_calls,
                "embedding_tokens": self.emb_tok, "cost_usd": round(self.cost(), 6),
                "by_model": {k: dict(v) for k, v in self.by_model.items()}}

    def delta(self, prev):
        cur = self.snapshot()
        return {k: (round(cur[k] - prev[k], 6) if isinstance(cur[k], (int, float)) else cur[k])
                for k in cur if k != "by_model"}


def fatal_if_credit_exhaustion(e):
    """D-34. Exhausted credits arrive as RateLimitError -- indistinguishable from throttling
    by class, distinguishable by message. Retrying it burns the whole run into zeros."""
    msg = str(e).lower()
    if any(s in msg for s in ("insufficient_quota", "no credits", "billing",
                              "exceeded your current quota", "credit balance")):
        raise SystemExit(f"FATAL D-34: credits exhausted, not throttling -- {e}")


def build_funcs(meter, llm_model, emb_model, api_key):
    from openai import AsyncOpenAI
    import numpy as np
    client = AsyncOpenAI(api_key=api_key)

    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kw):
        kw.pop("hashing_kv", None); kw.pop("keyword_extraction", None)
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
             + list(history_messages) + [{"role": "user", "content": prompt}]
        for attempt in range(5):
            try:
                r = await client.chat.completions.create(
                    model=llm_model, messages=msgs, temperature=0.0, **kw)
                meter.add_llm(llm_model, r.usage)
                return r.choices[0].message.content
            except Exception as e:
                fatal_if_credit_exhaustion(e)                      # D-34, before any retry
                if attempt == 4:
                    raise                                          # D-32: never degrade
                await asyncio.sleep(2 ** attempt)

    async def embedding_func(texts):
        for attempt in range(5):
            try:
                r = await client.embeddings.create(model=emb_model, input=texts)
                meter.add_emb(emb_model, r.usage)
                return np.array([d.embedding for d in r.data])
            except Exception as e:
                fatal_if_credit_exhaustion(e)
                if attempt == 4:
                    raise
                await asyncio.sleep(2 ** attempt)
    return llm_model_func, embedding_func


def graph_size(workdir):
    p = Path(workdir) / "graph_chunk_entity_relation.graphml"
    if not p.exists():
        return {"entities": None, "relations": None, "graphml_bytes": None}
    txt = p.read_text(errors="ignore")
    return {"entities": txt.count("<node "), "relations": txt.count("<edge "),
            "graphml_bytes": p.stat().st_size}


async def run(arm, docs, workdir, llm_model, emb_model, api_key, limit_docs=None):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status

    meter = Meter()
    llm_f, emb_f = build_funcs(meter, llm_model, emb_model, api_key)
    dim = len((await emb_f(["dimension probe"]))[0])
    print(f"  embedding dim probe: {dim}", flush=True)

    rag = LightRAG(working_dir=str(workdir), llm_model_func=llm_f, llm_model_name=llm_model,
                   embedding_func=EmbeddingFunc(embedding_dim=dim, func=emb_f))
    await rag.initialize_storages()
    await initialize_pipeline_status()

    cfg = {"ARM": arm, "LLM_MODEL": llm_model, "EMB_MODEL": emb_model,
           "CHUNK_TOKEN_SIZE": rag.chunk_token_size,
           "GLEANING": rag.entity_extract_max_gleaning,
           "LLM_MAX_ASYNC": rag.llm_model_max_async,
           "MAX_PARALLEL_INSERT": getattr(rag, "max_parallel_insert", None),
           "EMBEDDING_DIM": dim, "RATES": RATES,
           "PILOT_ONLY": "authorises or kills; never a reported number"}
    print(f"  RUN_CONFIG: {json.dumps({k: v for k, v in cfg.items() if k != 'RATES'})}", flush=True)

    results, prev = [], meter.snapshot()
    ordered = sorted(docs, key=lambda d: d["chars"])   # ascending: exposes growth effects
    if limit_docs:
        ordered = ordered[:limit_docs]

    for i, d in enumerate(ordered, 1):
        base = Path(f"/media/volume/vol3/grammrag/gramm-rag-{d['bench']}/cache/parsed")
        parsed = mapper.load_parse(base / d["doc"])
        art = base / "artifacts" / d["doc"].replace(".json", "").replace(".pdf", "")
        stats = mapper.parse_stats(parsed, art)
        text = mapper.elements_to_text(parsed)

        print(f"\n[{i}/{len(ordered)}] p{d['pct']} {d['bench']}/{d['doc']} "
              f"{d['pages']}pg {stats['text_chars']:,}ch", flush=True)
        t0 = time.time()
        # T6-class fail-fast: one run stalled after the graph was already written and had
        # to be killed at the shell timeout. A hang must announce itself, not silently
        # consume the window. Budget scales with size, floor 600s.
        budget = max(600.0, stats["text_chars"] / 1000.0 * 12.0)
        try:
            await asyncio.wait_for(
                rag.ainsert(text, ids=f"{d['bench']}::{d['doc']}",
                            file_paths=f"{d['bench']}/{d['doc']}"), timeout=budget)
            err = None
        except asyncio.TimeoutError:
            err = f"TIMEOUT after {budget:.0f}s -- ingest did not return"
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        el = time.time() - t0
        dl = meter.delta(prev); prev = meter.snapshot()
        g = graph_size(workdir)
        row = {**{k: d[k] for k in ("pct", "bench", "doc", "pages", "chars")},
               "parse_stats": stats, "text_chars": stats["text_chars"],
               "seconds": round(el, 2), "error": err, **dl, "graph_after": g,
               "sec_per_1k_chars": round(el / max(stats["text_chars"], 1) * 1000, 4),
               "usd_per_1k_chars": round(dl["cost_usd"] / max(stats["text_chars"], 1) * 1000, 8)}
        results.append(row)
        print(f"    {el:7.1f}s  {dl['llm_calls']:4d} llm  "
              f"{dl['prompt_tokens']:>9,}in {dl['completion_tokens']:>8,}out  "
              f"${dl['cost_usd']:.4f}  ent={g['entities']} rel={g['relations']}  "
              f"{row['sec_per_1k_chars']:.3f} s/1kch" + (f"  ERROR {err}" if err else ""), flush=True)

    return cfg, results, meter.snapshot()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--emb-model", default="text-embedding-3-small")
    ap.add_argument("--limit-docs", type=int, default=None)
    a = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key or not key.startswith("sk-"):          # D-32: fail loudly, never fall back
        raise SystemExit("FATAL: OPENAI_API_KEY absent or malformed -- refusing to run")

    docs = json.loads((GB / "data/pilot_docs.json").read_text())
    workdir = GB / "pilot" / a.arm / "store"
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg, rows, total = asyncio.run(run(a.arm, docs, workdir, a.llm_model, a.emb_model,
                                       key, a.limit_docs))
    wall = time.time() - t0

    ok = [r for r in rows if not r["error"]]
    ch = sum(r["text_chars"] for r in ok)
    sec = sum(r["seconds"] for r in ok)
    out = {"run_config": cfg, "pilot_wall_seconds": round(wall, 1),
           "docs": rows, "totals": total,
           "pilot_totals": {"docs_ok": len(ok), "docs_failed": len(rows) - len(ok),
                            "text_chars": ch, "ingest_seconds": round(sec, 1),
                            "cost_usd": round(total["cost_usd"], 6)},
           "corpus": {"docs": CORPUS_DOCS, "pages": CORPUS_PAGES, "chars": CORPUS_CHARS}}
    p = GB / "pilot" / a.arm / "pilot_log.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {p}")
    print(f"pilot: {len(ok)}/{len(rows)} docs, {ch:,} chars, {sec:.0f}s ingest, "
          f"${total['cost_usd']:.4f}")


if __name__ == "__main__":
    main()
