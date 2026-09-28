#!/usr/bin/env python3
"""S5 -- Microsoft GraphRAG 2.7.2 indexing under harness identity, one root per document (C-35).

Runs inside the grammrag-analysis container: graphrag 2.7.2 requires Python >= 3.11.

GUARDS -- each emits its number on every document (rule 22):
  * harness identity: extraction model asserted gpt-4o-mini AT LOAD, not trusted from YAML, and
    chat type asserted 'chat' -- the LiteLLM path the meter hooks.
  * absorbed failures: run_pipeline catches workflow exceptions and yields errors=[e] rather than
    raising, so any non-empty .errors is fatal, and output tables are asserted to exist -- content
    checks outside the absorbing handler (rule 21, C-52).
  * parse provenance: text units aligned against the injected text, floor 0.95 (C-52).
  * meter liveness: a callback that never fires reports $0 and passes, so calls on each required
    stage are asserted > 0. Events are deduplicated on litellm_call_id, because sync and async
    hooks may both fire for one call and double-count cost silently.
  * D-34/C-57: credit exhaustion is classified on the exception CHAIN's structured fields --
    LiteLLM relabels it 'throttling_error' at the top of the chain -- and the process exits on
    the first rejected call, because a raise inside a callback is absorbed and GraphRAG would
    otherwise sleep through its retry backoff.
"""
import argparse, asyncio, hashlib, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path

GB = Path(__file__).resolve().parent
sys.path.insert(0, str(GB))
import mapper                                                    # noqa: E402

VENV_BIN = GB / "venvs" / "s5_graphrag" / "bin"
OUT = GB / "tier1" / os.environ.get("S5_OUTDIR", "s5_graphrag")
PARSED = {"mmlongbench": Path("/work/gramm-rag-mmlongbench/cache/parsed"),
          "docbench":    Path("/work/gramm-rag-docbench/cache/parsed"),
          "mpdocvqa":    Path("/work/gramm-rag-mpdocvqa/cache/parsed")}
RATES = {"chat_in": 1.5e-07, "chat_out": 6e-07, "emb_in": 2e-08}
SHINGLE, STRIDE, ALIGN_FLOOR = 50, 50, 0.95
TABLES = ("documents", "text_units", "entities", "relationships", "communities", "community_reports")
# Matched against the OPENING of a message only, so document text embedded later in a prompt
# cannot trigger a false stage match. Strings taken from graphrag 2.7.2's shipped prompt files.
SIGS = [("community_reports", ("perform general information discovery",
                               "write a comprehensive report of a community")),
        ("summarize",         ("responsible for generating a comprehensive summary",)),
        ("extraction",        ("-goal-", "many entities and relationships were missed",
                               "some entities and relationships may have still been missed"))]

from litellm.integrations.custom_logger import CustomLogger       # noqa: E402


class Meter(CustomLogger):
    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.fatal, self.failures, self.dupes = None, 0, 0
        self.stage, self.seen = {}, set()

    def reset(self):
        with self.lock:
            self.stage, self.failures, self.dupes = {}, 0, 0

    @staticmethod
    def _text(m):
        c = m.get("content", "") if isinstance(m, dict) else ""
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        return str(c)

    def _classify(self, kwargs):
        if kwargs.get("call_type") in ("embedding", "aembedding") or not kwargs.get("messages"):
            return "embedding"
        msgs = kwargs["messages"]
        heads = [self._text(msgs[0])[:400].lower()]
        if len(msgs) > 1:
            heads.append(self._text(msgs[-1])[:400].lower())
        for name, sigs in SIGS:
            if any(s in h for h in heads for s in sigs):
                return name
        return "unassigned"

    def _record(self, kwargs, resp):
        cid = kwargs.get("litellm_call_id")
        with self.lock:
            if cid is not None:
                if cid in self.seen:
                    self.dupes += 1
                    return
                self.seen.add(cid)
        st = self._classify(kwargs)
        u = getattr(resp, "usage", None)
        if u is None and isinstance(resp, dict):
            u = resp.get("usage")
        get = (lambda k: (u or {}).get(k)) if isinstance(u, dict) else (lambda k: getattr(u, k, None))
        pin, pout = get("prompt_tokens") or 0, get("completion_tokens") or 0
        cost = pin * RATES["emb_in"] if st == "embedding" else pin * RATES["chat_in"] + pout * RATES["chat_out"]
        with self.lock:
            s = self.stage.setdefault(st, [0, 0, 0, 0.0])
            s[0] += 1; s[1] += pin; s[2] += pout; s[3] += cost

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(kwargs, response_obj)

    # Verified 2026-09-10 against a live, rejected call: OpenAI returns 429 with
    # type=insufficient_quota, code=project_spend_limit_exceeded. Only these verified values are listed.
    FATAL_CODES = {"insufficient_quota", "project_spend_limit_exceeded"}
    FATAL_PROSE = ("insufficient_quota", "spend limit", "exceeded your current quota", "no credits")

    @staticmethod
    def exhausted(ex):
        """True if any exception in the chain is credit/spend exhaustion (C-57).

        LiteLLM re-raises OpenAI's exhaustion 429 as RateLimitError(type='throttling_error',
        code=429) with a rewritten message, so the field that separates exhaustion from throttling
        is RELABELLED at the top of the chain. The original openai.RateLimitError survives two links
        down, so structured fields are read along the chain first; prose is a fallback only."""
        for c in _chain(ex):
            if str(getattr(c, "code", "") or "").lower() in Meter.FATAL_CODES or \
               str(getattr(c, "type", "") or "").lower() == "insufficient_quota":
                return True
        s = str(ex).lower()
        return any(p in s for p in Meter.FATAL_PROSE)

    def _fail(self, kwargs, response_obj):
        ex = kwargs.get("exception")
        with self.lock:
            self.failures += 1
        if ex is None or not self.exhausted(ex):
            return
        # A raise here is absorbed by LiteLLM, and GraphRAG retries each call up to 10 times with
        # exponential backoff -- the first smoke run sat at 0.02% CPU for over ten minutes. Process
        # exit is a channel neither library can handle (rule 21).
        rec = {"fatal": "OpenAI credit/spend exhaustion (D-34, C-57)",
               "chain": [f"{type(c).__name__}(code={getattr(c, 'code', None)}, type={getattr(c, 'type', None)})"
                         for c in _chain(ex)],
               "message": str(ex)[:300], "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / "FATAL_credit_exhaustion.json").write_text(json.dumps(rec, indent=1))
        finally:
            print(f"\nFATAL: OpenAI credit/spend exhaustion -- {rec['chain']}", flush=True)
            os._exit(3)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._fail(kwargs, response_obj)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._fail(kwargs, response_obj)

    def snapshot(self):
        with self.lock:
            return {k: {"calls": v[0], "prompt_tokens": v[1], "completion_tokens": v[2],
                        "cost_usd": round(v[3], 6)} for k, v in self.stage.items()}

    def calls(self):
        with self.lock:
            return sum(v[0] for v in self.stage.values())


def _chain(ex):
    out, seen = [], set()
    while ex is not None and id(ex) not in seen:
        seen.add(id(ex)); out.append(ex); ex = ex.__cause__ or ex.__context__
    return out


def _norm(s):
    import re
    return re.sub(r"\s+", " ", s).strip().lower()


def alignment(chunks, injected):
    ref = _norm(injected)
    idx = {hash(ref[k:k + SHINGLE]) for k in range(0, max(len(ref) - SHINGLE + 1, 0))}
    m = t = 0
    for c in chunks:
        cn = _norm(c)
        for k in range(0, max(len(cn) - SHINGLE + 1, 0), STRIDE):
            t += SHINGLE
            m += SHINGLE if hash(cn[k:k + SHINGLE]) in idx else 0
    return round(m / t, 4) if t else 0.0


def make_template():
    tpl = OUT / "_template"
    if (tpl / "settings.yaml").exists():
        return tpl
    tpl.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(VENV_BIN / "graphrag"), "init", "--root", str(tpl)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FATAL: graphrag init failed: {r.stderr[-400:]}")
    s = (tpl / "settings.yaml").read_text()
    assert s.count("model: gpt-4-turbo-preview") == 1, "default chat model line not found exactly once"
    (tpl / "settings.yaml").write_text(s.replace("model: gpt-4-turbo-preview", "model: gpt-4o-mini"))
    if (tpl / ".env").exists():
        (tpl / ".env").unlink()          # key comes from the process env only, never disk (C-53)
    return tpl


def fatal(log, tag, reason):
    (OUT / f"index_log_{tag}.json").write_text(json.dumps(dict(docs=log, aborted=reason), indent=1))
    raise SystemExit(f"FATAL: {reason}")


async def index_doc(d, tpl, meter):
    import pandas as pd
    import graphrag.api as api
    from graphrag.config.load_config import load_config
    bench, doc = d["bench"], d["doc"]
    stem = doc.replace(".json", "").replace(".pdf", "")
    root = OUT / "stores" / f"{bench}__{stem}"
    if (root / "_INDEX_COMPLETE.json").exists():
        rec = json.loads((root / "_INDEX_COMPLETE.json").read_text())
        # Stamps written before text_md5 existed are backfilled from the store's own injected
        # text, so duplicate detection survives a resume across a code change. No API cost.
        if "text_md5" not in rec:
            f = root / "input" / f"{stem}.txt"
            rec["text_md5"] = (hashlib.md5(f.read_text(encoding="utf-8").encode()).hexdigest()
                               if f.exists() else None)
            (root / "_INDEX_COMPLETE.json").write_text(json.dumps(rec))
        return rec, True
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(tpl, root)
    text = mapper.elements_to_text(mapper.load_parse(PARSED[bench] / doc))
    text_md5 = hashlib.md5(text.encode()).hexdigest()
    (root / "input").mkdir(exist_ok=True)
    (root / "input" / f"{stem}.txt").write_text(text, encoding="utf-8")

    os.chdir(root)                       # relative input/output/cache dirs resolve inside the root
    cfg = load_config(root)
    chat, emb = cfg.models["default_chat_model"], cfg.models["default_embedding_model"]
    ctype = str(getattr(chat.type, "value", chat.type))
    assert chat.model == "gpt-4o-mini", f"harness identity: chat model is {chat.model}"
    assert ctype == "chat", f"chat type {ctype} is not the LiteLLM path the meter hooks"
    assert emb.model == "text-embedding-3-small", f"embedder is {emb.model}"
    assert str(chat.api_key or "").startswith("sk-"), "api key did not resolve from the environment"

    meter.reset()
    t0 = time.time()
    results = await api.build_index(config=cfg)
    last, stable = -1, 0                 # drain late-arriving callback events
    for _ in range(60):
        n = meter.calls()
        stable = stable + 1 if n == last else 0
        last = n
        if stable >= 3:
            break
        await asyncio.sleep(1)
    secs = time.time() - t0

    tab = {t: (pd.read_parquet(root / "output" / f"{t}.parquet")
               if (root / "output" / f"{t}.parquet").exists() else None) for t in TABLES}
    tu, ent, rel = tab["text_units"], tab["entities"], tab["relationships"]
    comm, rep = tab["communities"], tab["community_reports"]
    chunks = tu["text"].astype(str).tolist() if tu is not None and "text" in tu.columns else []
    stages = meter.snapshot()
    cost = sum(s["cost_usd"] for s in stages.values())
    rec = dict(
        bench=bench, doc=doc, stem=stem, stratum=d.get("stratum"), text_md5=text_md5,
        ingested_chars=len(text), seconds=round(secs, 1), cost_usd=round(cost, 6), stages=stages,
        unassigned_cost_share=round(stages.get("unassigned", {}).get("cost_usd", 0.0) / cost, 4) if cost else None,
        n_text_units=len(chunks), n_entities=0 if ent is None else len(ent),
        n_relationships=0 if rel is None else len(rel), n_communities=0 if comm is None else len(comm),
        n_levels=int(comm["level"].nunique()) if comm is not None and "level" in comm.columns and len(comm) else 0,
        n_reports=0 if rep is None else len(rep),
        report_chars=int(rep["full_content"].astype(str).str.len().sum())
                     if rep is not None and "full_content" in rep.columns else 0,
        alignment=alignment(chunks, text) if chunks else 0.0,
        workflow_errors=[(r.workflow, repr(e)[:300]) for r in results for e in (r.errors or [])],
        missing_tables=[t for t, v in tab.items() if v is None],
        llm_failures=meter.failures, duplicate_events_suppressed=meter.dupes)
    return rec, False


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=str(OUT / "pilot_docs.json"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--ceiling", type=float, default=20.0)       # pre-registered pilot ceiling
    ap.add_argument("--tag", default="pilot")
    a = ap.parse_args()
    if not os.environ.get("GRAPHRAG_API_KEY", "").startswith("sk-"):
        raise SystemExit("FATAL: GRAPHRAG_API_KEY absent or malformed -- refusing to run")
    import litellm
    meter = Meter()
    litellm.callbacks = [meter]
    tpl = make_template()
    docs = json.loads(Path(a.docs).read_text())["docs"]
    if a.only:
        docs = [d for d in docs if a.only in d["doc"] or a.only in (d.get("stratum") or "")]
    log, spent, seen_md5 = [], 0.0, {}
    for i, d in enumerate(docs, 1):
        rec, resumed = await index_doc(d, tpl, meter)
        log.append(rec)
        spent += 0.0 if resumed else rec["cost_usd"]
        st = rec["stages"]
        # Duplicate bookkeeping happens BEFORE the resume short-circuit: a run that resumes the
        # first copy of a duplicate pair and freshly indexes the second would otherwise never
        # register the hash, and the second copy's legitimate zero-call result would be fatal.
        md5 = rec.get("text_md5")
        dup_of = seen_md5.get(md5) if md5 else None
        rec["duplicate_of"] = dup_of
        if md5 and dup_of is None:
            seen_md5[md5] = rec["stem"]
        print(f"  [{i}/{len(docs)}] {rec['stem'][:24]:26s} {rec['ingested_chars']:>9,}ch "
              f"{rec['seconds']:>6.0f}s ${rec['cost_usd']:.4f} "
              f"[ext ${st.get('extraction', {}).get('cost_usd', 0):.3f} "
              f"sum ${st.get('summarize', {}).get('cost_usd', 0):.3f} "
              f"rep ${st.get('community_reports', {}).get('cost_usd', 0):.3f} "
              f"emb ${st.get('embedding', {}).get('cost_usd', 0):.4f} "
              f"unasg {rec['unassigned_cost_share']}] "
              f"comm={rec['n_communities']} lvl={rec['n_levels']} rep={rec['n_reports']} "
              f"align={rec['alignment']:.4f} dup={rec['duplicate_events_suppressed']}"
              + ("  (resumed)" if resumed else ""), flush=True)
        if resumed:
            continue
        if meter.fatal:
            fatal(log, a.tag, f"credit exhaustion (D-34): {meter.fatal}")
        if rec["workflow_errors"]:
            fatal(log, a.tag, f"{rec['stem']}: workflow errors absorbed by run_pipeline: {rec['workflow_errors'][:2]}")
        if rec["missing_tables"]:
            fatal(log, a.tag, f"{rec['stem']}: output tables missing {rec['missing_tables']}")
        if rec["alignment"] < ALIGN_FLOOR:
            fatal(log, a.tag, f"{rec['stem']}: text-unit alignment {rec['alignment']} < {ALIGN_FLOOR}")
        # EVERY call-count assertion is exempt for duplicate text, not just the liveness one.
        # MMLongBench has 3 byte-identical document pairs; the second copy processed in the same
        # process reuses the first's results, so extraction, embedding AND community-report call
        # counts are all legitimately zero while the outputs are correct. Guarding only the first
        # of them halted a healthy run twice -- at document 129 (liveness) and again at 259
        # (reports). The discriminator is the injected text's md5, never the call count or cost,
        # which look identical to a dead meter.
        if dup_of is None:
            if st.get("extraction", {}).get("calls", 0) == 0 or st.get("embedding", {}).get("calls", 0) == 0:
                fatal(log, a.tag, f"{rec['stem']}: meter not live (extraction or embedding calls = 0)")
            if rec["n_communities"] > 0 and (rec["n_reports"] == 0
                                             or st.get("community_reports", {}).get("calls", 0) == 0):
                fatal(log, a.tag, f"{rec['stem']}: communities exist but reports/report calls are zero")
        elif rec["cost_usd"] == 0.0:
            print(f"      (duplicate text of {dup_of}; results reused in-process, cost attributed "
                  f"to the first occurrence)", flush=True)
        # Content assertions stay unguarded: a duplicate still carries real output, so these hold
        # for every document regardless of how the output was produced.
        if rec["n_entities"] >= 5 and rec["n_communities"] == 0:
            fatal(log, a.tag, f"{rec['stem']}: {rec['n_entities']} entities but zero communities")
        if rec["n_communities"] > 0 and rec["n_reports"] == 0:
            fatal(log, a.tag, f"{rec['stem']}: communities exist but no reports were written")
        (OUT / "stores" / f"{rec['bench']}__{rec['stem']}" / "_INDEX_COMPLETE.json").write_text(json.dumps(rec))
        if spent > a.ceiling:
            fatal(log, a.tag, f"${spent:.2f} exceeds ceiling ${a.ceiling}")
    RUN_CONFIG = {"ARM": "s5_graphrag", "GRAPHRAG_VERSION": "2.7.2", "PYTHON": sys.version.split()[0],
                  "LLM_MODEL": "gpt-4o-mini (asserted at load)", "EMB_MODEL": "text-embedding-3-small",
                  "SCOPE": "one GraphRAG root per document (C-35)",
                  "CHUNKS": "1200/100 tokens (default)", "MAX_GLEANINGS": 1, "MAX_CLUSTER_SIZE": 10,
                  "CHAR_MEASURE": "len(mapper.elements_to_text) -- corpus_chars.json ingested_chars",
                  "STAGE_METER": "LiteLLM callback, classified by prompt-opening signature, dedup on litellm_call_id",
                  "RATES": RATES}
    (OUT / f"index_log_{a.tag}.json").write_text(json.dumps(dict(RUN_CONFIG=RUN_CONFIG, docs=log,
                                                                  spent_usd=round(spent, 6)), indent=1))
    print(f"\n  {a.tag} complete: {len(log)} docs, ${spent:.4f} new spend")


if __name__ == "__main__":
    asyncio.run(main())
