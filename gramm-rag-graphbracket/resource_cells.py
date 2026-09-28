#!/usr/bin/env python3
"""B -- serving / resource cells for the three API-mediated graph arms.

One (arm x benchmark x state x occurrence) cell per invocation.

States
  cold  the document's store is opened from disk INSIDE the timed region -- what a
        multi-tenant server pays on a first request for a document it is not holding.
  warm  the store is already open; only the query is timed -- steady state.
Both exclude answer generation where the arm lets us exclude it (see COMPARABILITY).

Design constraints that are not stylistic:
  * C-36. lightrag.kg.shared_storage keeps MODULE-LEVEL globals. Two LightRAG stores open
    in one process makes the second read the first's namespaces, silently. So "warm" can
    never mean an LRU of many stores -- it means one document open at a time, all of that
    document's questions, then finalize. Cold and warm therefore run the SAME (doc, question)
    pairs in the SAME order; only the placement of the open/close differs.
  * COMPARABILITY. LightRAG/RAG-Anything expose only_need_context=True, so their timed region
    is context assembly. graphrag's public api.local_search always generates an answer, and
    the context-only path (get_local_search_engine) needs a description_embedding_store we
    would have to build ourselves -- configuring it differently from the path that produced
    S5's published accuracy would measure a different system (the D-32 class). So S5 keeps
    the canonical path and we MEASURE the generation call's duration from litellm's own
    start/end timestamps and subtract it, reporting total AND context-only. Measured, not
    assumed.
  * rule 22. Peak GPU is measured and emitted every run, not asserted to be zero because
    these arms "should" be API-bound. "We expect zero" and "we observed zero" differ.
  * Warmup queries are discarded and COUNTED in the record.
"""
import argparse, json, os, random, statistics, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

GB = Path(__file__).resolve().parent
sys.path.insert(0, str(GB))

OUT = GB / "tier1" / "_resource_cells"
WARMUP, SEED = 20, 42

ARMS = {
    "lightrag": dict(kind="lightrag", store="tier1/lightrag_{bench}/stores", prefix=""),
    "s2":       dict(kind="lightrag", store="tier1/s2_raganything/stores",   prefix="{bench}__"),
    "s5":       dict(kind="graphrag", store="tier1/s5_graphrag/stores",      prefix="{bench}__"),
}
# The two stacks see different roots -- the lightrag venv runs on the host
# (/media/volume/vol3/grammrag), the graphrag venv inside a container (/work). Deriving
# ROOT from this file's own location makes one table correct in both, instead of two
# tables that can silently drift apart.
ROOT = GB.parent
BENCH = {
    "mmlongbench": dict(d3=ROOT / "gramm-rag-3benchmarks/data/mmlongbench",
                        base="baseline_vector_mmlb.json", qmeta="qmeta_mmlb.json",
                        src=ROOT / "gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json"),
    "docbench":    dict(d3=ROOT / "gramm-rag-3benchmarks/data/docbench",
                        base="baseline_vector_docbench.json", qmeta="qmeta_docbench.json", src=None),
}
COMMUNITY_LEVEL = 2


# ---------------------------------------------------------------- instrumentation
def gpu_peak_mib():
    """Measured, never assumed (rule 22). Returns (torch_peak, nvidia_smi_used)."""
    t = None
    try:
        import torch
        if torch.cuda.is_available():
            t = torch.cuda.max_memory_allocated() / 2 ** 20
    except Exception:
        pass
    s = None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        s = float(r.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return t, s


def pctl(v, q):
    v = sorted(v)
    return v[min(int(q * len(v)), len(v) - 1)]


def boot_ci(v, stat, B=2000, seed=SEED):
    rng = random.Random(seed)
    n = len(v)
    d = sorted(stat([v[rng.randrange(n)] for _ in range(n)]) for _ in range(B))
    return round(d[int(.025 * B)], 4), round(d[int(.975 * B)], 4)


def sample_pairs(arm, bench, n_needed):
    """Documents with a completed store, shuffled by DOCUMENT (seed 42), walked until we
    have n_needed (doc, question_index) pairs. Identical for cold and warm."""
    S, Bc = ARMS[arm], BENCH[bench]
    D3 = Path(Bc["d3"])
    base = json.loads((D3 / Bc["base"]).read_text())
    qm = json.loads((D3 / Bc["qmeta"]).read_text())["rows"]
    questions = ([s["question"] for s in json.loads(Path(Bc["src"]).read_text())]
                 if Bc["src"] else list(base["questions"]))
    assert len(qm) == len(questions), "index alignment broken across metadata files"
    sdir = GB / S["store"].format(bench=bench)
    pre = S["prefix"].format(bench=bench)

    by_doc = defaultdict(list)
    for i, r in enumerate(qm):
        by_doc[str(r["doc_id"])].append(i)

    docs = []
    for doc, idxs in sorted(by_doc.items()):
        stem = doc[:-4] if doc.lower().endswith(".pdf") else doc
        stem = stem[:-5] if stem.lower().endswith(".json") else stem
        root = sdir / f"{pre}{stem}"
        if (root / "_INDEX_COMPLETE.json").exists():
            docs.append((doc, root, idxs))
    random.Random(SEED).shuffle(docs)

    plan, tot = [], 0
    for doc, root, idxs in docs:
        take = idxs[:max(0, n_needed - tot)]
        if not take:
            break
        plan.append((doc, root, take))
        tot += len(take)
    return plan, questions, tot, len(docs), sdir


# ---------------------------------------------------------------- cells
async def cell_lightrag(a, plan, questions):
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status, finalize_share_data
    from pilot_ingest import Meter, build_funcs

    key = os.environ.get("OPENAI_API_KEY")
    if not key or not key.startswith("sk-"):
        raise SystemExit("FATAL: OPENAI_API_KEY absent or malformed -- refusing to run")
    meter = Meter()
    llm_f, emb_f = build_funcs(meter, "gpt-4o-mini", "text-embedding-3-small", key)
    dim = len((await emb_f(["dimension probe"]))[0])

    async def open_store(root):
        rag = LightRAG(working_dir=str(root), llm_model_func=llm_f,
                       llm_model_name="gpt-4o-mini",
                       embedding_func=EmbeddingFunc(embedding_dim=dim, func=emb_f))
        await rag.initialize_storages()
        await initialize_pipeline_status()
        return rag

    async def close_store(rag):
        await rag.finalize_storages()
        finalize_share_data()                      # C-36: reset module-level globals

    lat, k = [], 0
    for doc, root, idxs in plan:
        if a.state == "warm":
            rag = await open_store(root)
        for i in idxs:
            if a.state == "cold":
                t0 = time.perf_counter()
                rag = await open_store(root)
                await rag.aquery(questions[i], param=QueryParam(mode="hybrid", only_need_context=True))
                dt = time.perf_counter() - t0
                await close_store(rag)
            else:
                t0 = time.perf_counter()
                await rag.aquery(questions[i], param=QueryParam(mode="hybrid", only_need_context=True))
                dt = time.perf_counter() - t0
            k += 1
            if k > WARMUP:
                lat.append(dt)
            if k % 100 == 0:
                print(f"    {k} queries, {len(lat)} timed", flush=True)
        if a.state == "warm":
            await close_store(rag)
    snap = meter.snapshot()
    cost = snap["cost_usd"] if "cost_usd" in snap else sum(v["cost_usd"] for v in snap.values())
    return lat, [], cost, snap


async def cell_graphrag(a, plan, questions):
    import pandas as pd, litellm
    import graphrag.api as api
    from graphrag.config.load_config import load_config
    from index_s5_graphrag import Meter

    class TimingMeter(Meter):
        """Adds per-call wall durations so the generation leg can be SUBTRACTED rather
        than hand-waved. litellm hands us start_time/end_time on every success event."""
        def __init__(self):
            super().__init__()
            self.win = []

        def _stamp(self, kwargs, start_time, end_time):
            try:
                d = (end_time - start_time).total_seconds()
            except Exception:
                return
            self.win.append((self._classify(kwargs), d))

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            self._stamp(kwargs, start_time, end_time)
            return super().log_success_event(kwargs, response_obj, start_time, end_time)

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self._stamp(kwargs, start_time, end_time)
            return await super().async_log_success_event(kwargs, response_obj, start_time, end_time)

    meter = TimingMeter()
    litellm.callbacks = [meter]
    TABLES = ("entities", "communities", "community_reports", "text_units", "relationships")

    def open_store(root):
        os.chdir(root)
        cfg = load_config(root)
        T = {t: pd.read_parquet(root / "output" / f"{t}.parquet") for t in TABLES}
        return cfg, T

    lat, ctx_only, k, gen_seen = [], [], 0, 0
    for doc, root, idxs in plan:
        if a.state == "warm":
            cfg, T = open_store(root)
        for i in idxs:
            meter.win = []
            t0 = time.perf_counter()
            if a.state == "cold":
                cfg, T = open_store(root)
            await api.local_search(config=cfg, entities=T["entities"], communities=T["communities"],
                                   community_reports=T["community_reports"],
                                   text_units=T["text_units"], relationships=T["relationships"],
                                   covariates=None, community_level=COMMUNITY_LEVEL,
                                   response_type="Multiple Paragraphs", query=questions[i])
            dt = time.perf_counter() - t0
            gen = sum(d for stage, d in meter.win if stage != "embedding")
            k += 1
            if k > WARMUP:
                lat.append(dt)
                ctx_only.append(max(dt - gen, 0.0))
                # rule 22: the subtraction is only valid for queries whose generation
                # callback landed inside the window. Emit the coverage rather than
                # assume it -- an uncaptured call silently makes ctx_only == total.
                gen_seen += 1 if gen > 0 else 0
            if meter.fatal:
                raise SystemExit(f"FATAL: credit exhaustion -- {meter.fatal}")
            if k % 100 == 0:
                print(f"    {k} queries, {len(lat)} timed", flush=True)
    snap = meter.snapshot()
    cost = sum(v["cost_usd"] for v in snap.values())
    snap["_gen_capture_rate"] = round(gen_seen / max(len(lat), 1), 4)
    return lat, ctx_only, cost, snap


# ---------------------------------------------------------------- driver
async def run(a):
    plan, questions, tot, n_docs, sdir = sample_pairs(a.arm, a.bench, a.n + WARMUP)
    print(f"  {a.arm}/{a.bench}/{a.state} occ{a.occurrence}: {tot} pairs over "
          f"{len(plan)}/{n_docs} indexed docs", flush=True)
    if tot < a.n + WARMUP:
        print(f"  WARNING: corpus yields {tot} < {a.n + WARMUP} requested", flush=True)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    t_cell = time.time()
    fn = cell_lightrag if ARMS[a.arm]["kind"] == "lightrag" else cell_graphrag
    lat, ctx_only, cost, snap = await fn(a, plan, questions)
    wall = time.time() - t_cell
    if not lat:
        raise SystemExit("FATAL: no timed queries -- refusing to emit an empty cell")

    med, p95, p99 = statistics.median(lat), pctl(lat, .95), pctl(lat, .99)
    tpk, smi = gpu_peak_mib()
    # Filter by THIS corpus's store prefix. RAG-Anything and MS GraphRAG keep both corpora in
    # one stores/ directory (mmlongbench__*, docbench__*); an unfiltered rglob reported the
    # two-corpus total as each corpus's size (6.74 GiB for both, actual 2.32 / 4.43). LightRAG
    # was correct only because it happens to use one directory per corpus.
    pre = ARMS[a.arm]["prefix"].format(bench=a.bench)
    idx_bytes = sum(f.stat().st_size for d in sdir.glob(f"{pre}*") if d.is_dir()
                    for f in d.rglob("*") if f.is_file())
    rec = dict(
        arm=a.arm, bench=a.bench, state=a.state, occurrence=a.occurrence,
        n_timed=len(lat), warmup_discarded=WARMUP, n_docs_touched=len(plan),
        median_s=round(med, 4), p95_s=round(p95, 4), p99_s=round(p99, 4),
        tail_ratio_p95_over_median=round(p95 / med, 4),
        median_ci=boot_ci(lat, statistics.median),
        tail_ratio_ci=boot_ci(lat, lambda v: pctl(v, .95) / statistics.median(v)),
        mean_s=round(statistics.mean(lat), 4),
        peak_query_gpu_mib_torch=tpk,
        gpu_mib_nvidia_smi_at_end=smi,
        gpu_basis=("container started without --gpus and exposes no /dev/nvidia* device, "
                   "so this arm cannot allocate GPU memory at query time -- structurally 0, "
                   "not merely observed 0" if ARMS[a.arm]["kind"] == "graphrag" else
                   "host process; nvidia-smi read at cell end. torch absent from this venv, "
                   "so the smi figure is the measurement"),
        index_bytes_whole_corpus=idx_bytes,
        index_gib_whole_corpus=round(idx_bytes / 2 ** 30, 3),
        prep_a100_hours="0.0 -- indexing for this arm is API-mediated; it has no GPU stage",
        api_cost_usd=round(cost, 5),
        api_cost_per_query_usd=round(cost / len(lat), 8),
        cell_wall_s=round(wall, 1),
        meter_stages=snap,
    )
    if ctx_only:
        rec["context_only_median_s"] = round(statistics.median(ctx_only), 4)
        rec["context_only_p95_s"] = round(pctl(ctx_only, .95), 4)
        rec["generation_overhead_median_s"] = round(med - statistics.median(ctx_only), 4)
        rec["gen_capture_rate"] = snap.get("_gen_capture_rate")
        rec["context_only_note"] = (
            "api.local_search always generates; the generation leg's duration is measured from "
            "litellm start/end stamps and subtracted, so context_only_* is what compares against "
            "the other arms. Validity is bounded by gen_capture_rate: a query whose success "
            "callback lands outside its window subtracts nothing and reports context_only == "
            "total, which inflates the UPPER tail only. Read the median, treat context_only_p95 "
            "as an upper bound.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.arm}_{a.bench}_{a.state}_occ{a.occurrence}.json").write_text(json.dumps(rec, indent=1))
    print(f"  -> n={len(lat)} median {med:.3f}s p95 {p95:.3f}s p99 {p99:.3f}s "
          f"tail {p95/med:.2f} gpu(torch)={tpk} smi={smi} ${cost:.4f} wall {wall/60:.1f}m", flush=True)


if __name__ == "__main__":
    import asyncio
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--bench", required=True, choices=list(BENCH))
    ap.add_argument("--state", required=True, choices=["cold", "warm"])
    ap.add_argument("--occurrence", type=int, default=1)
    ap.add_argument("--n", type=int, default=500)
    asyncio.run(run(ap.parse_args()))
