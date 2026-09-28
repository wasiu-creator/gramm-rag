#!/usr/bin/env python3
"""P2 -- judge-robustness sweep. Re-judge every arm's existing DocBench answers.

Only the judge MODEL varies. The judge function, prompt, parsing, temperature and reference
fields are the canonical ones, imported from gramm-rag-docbench -- reused, never reimplemented,
because a re-written judge would confound judge identity with judge implementation.

GUARDS
  * D-19: the judge model is asserted reachable before the sweep -- a dead model name once
    produced a confident judge_accuracy of 0.000 across a whole run.
  * D-34/C-57: credit exhaustion is classified on the exception CHAIN's structured fields and
    aborts the sweep; transient failures are counted and reported, never silently scored 0.
  * T7: per-arm checkpointing, so an abort never loses spend already made.
  * rule 22: every arm emits its counts and cost when it finishes, pass or fail.
"""
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/work/gramm-rag-colpali")
D3 = Path("/work/gramm-rag-3benchmarks/data/docbench")
OUT = ROOT / "scores" / "p2_judge_sweep.json"
# list prices, stamped rather than assumed silently; the provider dashboard is authoritative
RATES = {"gpt-4o": {"in": 2.5e-06, "out": 1.0e-05},
         "gpt-4.1": {"in": 2.0e-06, "out": 8.0e-06}}
ARMS = [("Flat-vector", None), ("GraMM-RAG", None), ("ColPali", "colpali"),
        ("Union(CP∪GraMM)", "union"), ("ColPali size-matched", "colpali_matched"),
        ("LightRAG 1.5.7", "lightrag"), ("RAG-Anything 1.2.8", "s2")]


def _chain(ex):
    out, seen = [], set()
    while ex is not None and id(ex) not in seen:
        seen.add(id(ex)); out.append(ex); ex = ex.__cause__ or ex.__context__
    return out


def exhausted(ex):
    for c in _chain(ex):
        if str(getattr(c, "code", "") or "").lower() in {"insufficient_quota", "project_spend_limit_exceeded"} \
           or str(getattr(c, "type", "") or "").lower() == "insufficient_quota":
            return True
    s = str(ex).lower()
    return any(p in s for p in ("insufficient_quota", "spend limit", "exceeded your current quota", "no credits"))


class Metered:
    """Wraps the OpenAI client so the canonical judge is called unmodified but metered."""
    def __init__(self, client, model):
        self._c, self.model = client, model
        self.lock = threading.Lock()
        self.calls = self.pin = self.pout = 0
        self.fatal = None

        orig = client.chat.completions.create

        def create(*a, **k):
            try:
                r = orig(*a, **k)
            except Exception as e:
                if exhausted(e):
                    with self.lock:
                        self.fatal = str(e)[:300]
                raise
            u = getattr(r, "usage", None)
            if u is not None:
                with self.lock:
                    self.calls += 1
                    self.pin += u.prompt_tokens or 0
                    self.pout += u.completion_tokens or 0
            return r

        client.chat.completions.create = create

    @property
    def cost(self):
        r = RATES[self.model]
        return self.pin * r["in"] + self.pout * r["out"]


def load_arm(label, key, base, gramm):
    if label == "Flat-vector":
        return base["predictions"], base["judge"]["judge_scores"]
    if label == "GraMM-RAG":
        return gramm["predictions"], gramm["judge"]["judge_scores"]
    gen = json.loads((ROOT / "generation" / f"{key}_gen_docbench.json").read_text())
    sc = json.loads((ROOT / "scores" / f"{key}_scores_docbench.json").read_text())
    return gen["predictions"], sc["judge_scores"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")          # pre-registered second judge
    ap.add_argument("--ceiling", type=float, default=25.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key.startswith("sk-"):
        raise SystemExit("FATAL: OPENAI_API_KEY absent -- refusing to run")
    sys.path.insert(0, "/work/gramm-rag-docbench")
    from openai import OpenAI
    from src.evaluation.metrics import judge_answer_gpt4      # canonical, unmodified

    client = OpenAI(api_key=key)
    meter = Metered(client, a.model)
    # D-19: prove the judge model answers before spending 7,714 calls against it
    probe = judge_answer_gpt4("What is 2+2?", "4", "4", "", client=client, model=a.model)
    assert probe in (0, 1), f"judge probe returned {probe!r}"
    print(f"  judge probe OK ({a.model}) -> {probe}", flush=True)

    base = json.loads((D3 / "baseline_vector_docbench.json").read_text())
    gramm = json.loads((D3 / "gramm_docbench_s42.json").read_text())
    qs, golds, ev = base["questions"], base["golds"], base.get("evidences") or [""] * len(base["questions"])
    an, n = base["is_answerable"], base["n_questions"]

    out = json.loads(OUT.read_text()) if OUT.exists() else {"RUN_CONFIG": {}, "arms": {}}
    out["RUN_CONFIG"] = {"SWEEP": "P2 judge robustness", "CANONICAL_JUDGE": "gpt-4-turbo",
                         "SECOND_JUDGE": a.model, "N": n, "TEMPERATURE": 0.0,
                         "JUDGE_FN": "gramm-rag-docbench src.evaluation.metrics.judge_answer_gpt4 (unmodified)",
                         "RATES_ASSUMED_LIST": RATES[a.model]}
    t0 = time.time()
    for label, akey in ARMS:
        if label in out["arms"]:
            print(f"  {label:24s} resumed", flush=True)
            continue
        preds, canon = load_arm(label, akey, base, gramm)
        assert len(preds) == n and len(canon) == n, f"{label}: length mismatch"
        new = [None] * n
        fails = {"n": 0}

        def one(i):
            try:
                return i, judge_answer_gpt4(qs[i], preds[i], golds[i], ev[i] or "",
                                            client=client, model=a.model)
            except Exception as e:
                if exhausted(e) or meter.fatal:
                    raise
                fails["n"] += 1
                return i, 0                     # transient only; counted and reported

        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for i, s in pool.map(one, range(n)):
                new[i] = s
        if meter.fatal:
            OUT.write_text(json.dumps(out, indent=1))
            raise SystemExit(f"FATAL: credit/spend exhaustion during {label} -- {meter.fatal}")
        acc = lambda idx: (sum(new[i] for i in idx) / len(idx)) if idx else None
        A = [i for i in range(n) if an[i]]
        U = [i for i in range(n) if not an[i]]
        agree = sum(1 for i in range(n) if new[i] == canon[i]) / n
        out["arms"][label] = {"scores_second_judge": new, "scores_canonical": canon,
                              "acc_second": acc(range(n)), "acc_canonical": sum(canon) / n,
                              "acc_second_answerable": acc(A), "acc_second_unanswerable": acc(U),
                              "raw_agreement": round(agree, 4), "transient_failures": fails["n"],
                              "cost_usd_cumulative": round(meter.cost, 4)}
        OUT.write_text(json.dumps(out, indent=1))                       # T7 checkpoint
        print(f"  {label:24s} acc {acc(range(n)):.4f} (canonical {sum(canon)/n:.4f})  "
              f"agree {agree:.4f}  fails {fails['n']}  ${meter.cost:.2f}", flush=True)
        if meter.cost > a.ceiling:
            raise SystemExit(f"FATAL: ${meter.cost:.2f} exceeds ceiling ${a.ceiling}")
    out["totals"] = {"calls": meter.calls, "cost_usd": round(meter.cost, 4),
                     "wall_seconds": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n  sweep complete: {meter.calls} calls, ${meter.cost:.2f}, "
          f"{out['totals']['wall_seconds']/60:.1f} min")


if __name__ == "__main__":
    main()
