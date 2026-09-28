#!/usr/bin/env python3
"""D3 -- P3 generation for all FIVE arms on MP-DocVQA, under ONE config.

Harness identity by IMPORT, not by copy. page_nodes, fatal_if_exhausted, the generator
constants (MODEL/PROVIDER/MAX_CONTEXT_CHARS/FAIL_FAST_N/CHECKPOINT_EVERY/WORKERS) and
build_prompt/VLMClient are imported from the modules the other two corpora already use, so
the two scripts cannot drift apart silently (C-49's shape: a path that differs between arms
and nobody notices until a metric moves). A startup assertion prints the resolved module
file of each import so divergence is visible in the log, not inferred.

TWO things are deliberately NOT shared, because they are exactly what must differ:

 1. PAGE INDEXING IS 0-BASED. generate_colpali's docstring states "1-based throughout" and
    its render() encodes it as `if page_no < 1: return None`. That is true for MMLongBench
    and DocBench and FALSE for MP-DocVQA, whose parse page_no runs 0..N-1 and whose imdb
    gold pages are 0-based. Retrofitting the 1-based path would have silently dropped every
    page-0 image and off-by-one the rest.
 2. PAGE IMAGES ARE THE CORPUS'S OWN JPGs, not a 120-dpi render. MP-DocVQA ships them
    (<doc>_p<N>.jpg). Measured: median 1701x2204 = 3.80 MP against a 120-dpi US-Letter
    render's 1.35 MP, i.e. 2.82x the pixel area. Stamped in RUN_CONFIG; the dispersion
    check in P3_PREREGISTRATION tests whether that inflates scores toward a ceiling.

ALL FIVE ARMS ARE REGENERATED under k=3 / 12,000 chars / T=0.0. The archived flat-vector
and GraMM MP-DocVQA answers used PAGE_CONTEXT_N=2 at 8,000 chars (stored contexts average
1.9 and 2.7 pages). Reusing them would attribute a context-budget difference to retrieval
quality. Their RETRIEVAL is untouched -- only generation is re-run, from their own stored
context_pages[:3].
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GB = Path(__file__).resolve().parent
ROOT = GB.parent
CP = ROOT / "gramm-rag-colpali"
MPR = ROOT / "gramm-rag-mpdocvqa"
sys.path.insert(0, str(CP))
sys.path.insert(0, str(MPR))

# --- shared downstream, by import -------------------------------------------------------
import generate_colpali as GC                      # noqa: E402
from generate_colpali import (page_nodes, fatal_if_exhausted, MODEL, PROVIDER,
                              MAX_CONTEXT_CHARS, FAIL_FAST_N, CHECKPOINT_EVERY,
                              WORKERS)              # noqa: E402
from src.generation.prompt_builder import build_prompt      # noqa: E402
from src.generation.vlm_client import VLMClient            # noqa: E402

IMGDIR = MPR / "data_volume/images/images"
P3 = GB / "tier1/p3_mpdocvqa"
OUT = P3 / "generation"
D3DIR = ROOT / "gramm-rag-3benchmarks/data/mpdocvqa"
PARSED = MPR / "cache/parsed"
TOPK = 3

ARMS = {
    "lightrag": ("graph", P3 / "p3_retrieval_lightrag.json"),
    "s2":       ("graph", P3 / "p3_retrieval_s2.json"),
    "s5":       ("graph", P3 / "p3_retrieval_s5.json"),
    "flat":     ("archived", D3DIR / "baseline_vector_mpdocvqa_pilot.json"),
    # C-77(ii): the k=3 control. Same shape as the archived arms -- context_pages indexed by
    # q_index over the full 5,187-question pilot -- so it needs no special case here. Its contexts
    # are the archived ones plus one reranked page (FLAT_K3_PREREGISTRATION.md, amendment 1).
    "flatk3":   ("archived", CP / "retrieval/flatk3_retrieval_mpdocvqa.json"),
    "gramm":    ("archived", D3DIR / "gramm_mpdocvqa_pilot_s42.json"),
}


def provenance():
    """rule 22: emit which file every shared import actually resolved to."""
    import inspect
    for name, obj in (("page_nodes", page_nodes), ("build_prompt", build_prompt),
                      ("VLMClient", VLMClient), ("fatal_if_exhausted", fatal_if_exhausted)):
        print(f"  import {name:18s} <- {inspect.getfile(obj)}", flush=True)
    print(f"  constants MODEL={MODEL} PROVIDER={PROVIDER} MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS} "
          f"WORKERS={WORKERS} FAIL_FAST_N={FAIL_FAST_N}", flush=True)
    assert MODEL == "gpt-4.1-mini" and MAX_CONTEXT_CHARS == 12000, \
        "generator constants drifted from the other corpora's values"


def img_for(doc: str, page0: int):
    """0-BASED page -> the corpus's own JPG. No render, no +1."""
    p = IMGDIR / f"{doc}_p{page0}.jpg"
    return str(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key or not key.startswith("sk-"):
        raise SystemExit("FATAL: OPENAI_API_KEY absent or malformed -- refusing to run")
    print(f"generate_p3 arm={a.arm}", flush=True)
    provenance()

    sub = json.loads((P3 / "subsample.json").read_text())
    qs = sub["questions"]
    kind, src = ARMS[a.arm]
    d = json.loads(src.read_text())

    if kind == "graph":
        err = {e["i"] for e in d.get("query_errors", [])}
        ctx_of = lambda i, q: (None if i in err else d["context_pages"][i][:TOPK])
    else:
        ctx_of = lambda i, q: d["context_pages"][q["q_index"]][:TOPK]

    RUN_CONFIG = {
        "ARM": a.arm, "BENCHMARK": "mpdocvqa", "GEN_MODEL": MODEL, "GEN_PROVIDER": PROVIDER,
        "TEMPERATURE": 0.0, "TOPK_PAGES": TOPK, "PAGE_CONTEXT_CHARS": MAX_CONTEXT_CHARS,
        "PROMPT_V": 3, "USE_PAGE_IMAGES": True,
        "PAGE_IMG_SOURCE": "MP-DocVQA native JPGs (median 1701x2204, 3.80 MP; 2.82x a "
                           "120-dpi US-Letter render) -- NOT re-rendered",
        "PAGE_INDEXING": "0-based (MP-DocVQA parse page_no and imdb gold pages are 0-based)",
        "REGENERATED": ("all five arms under one config; archived flat/gramm answers used "
                        "PAGE_CONTEXT_N=2 @8000 chars and are NOT reused"),
        "SEED": 42,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    ck = OUT / (f"ckpt_{a.arm}_limit{a.limit}.json" if a.limit else f"ckpt_{a.arm}.json")
    preds = {}
    if ck.exists():
        s = json.loads(ck.read_text())
        if s.get("RUN_CONFIG") == RUN_CONFIG:                    # R-42
            preds = {int(k): v for k, v in s["predictions"].items()}
            print(f"  resuming: {len(preds)} already generated", flush=True)
        else:
            print("  checkpoint RUN_CONFIG mismatch -- ignoring it", flush=True)

    idx = list(range(len(qs)))[:a.limit] if a.limit else list(range(len(qs)))
    todo = [i for i in idx if i not in preds]
    print(f"  {len(todo)} to generate of {len(idx)}", flush=True)

    client = VLMClient(provider=PROVIDER, openai_model=MODEL)
    pcache = {}
    consec, failures, n_img, n_noimg = 0, [], 0, 0
    t0 = time.time()

    def one(i):
        q = qs[i]
        doc = q["doc_id"]
        pages = ctx_of(i, q)
        if pages is None:
            return i, None, 0                                     # query errored upstream
        if doc not in pcache:
            pf = PARSED / f"{doc}.json"
            pcache[doc] = json.loads(pf.read_text()) if pf.exists() else {"elements": []}
        parsed = pcache[doc]
        nodes = page_nodes(parsed, pages)
        prompt = build_prompt(question=q["question"], retrieved_nodes=nodes, parsed=parsed,
                              reward_detail={}, refused=False, benchmark="mpdocvqa",
                              max_context_chars=MAX_CONTEXT_CHARS)
        imgs = [p for p in (img_for(doc, pg) for pg in pages) if p]
        txt = client.generate(prompt, temperature=0.0, max_tokens=512,
                              image_paths=imgs or None)
        return i, (txt or "").strip(), len(imgs)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for st in range(0, len(todo), CHECKPOINT_EVERY):
            batch = todo[st:st + CHECKPOINT_EVERY]
            for f in [ex.submit(one, i) for i in batch]:
                try:
                    i, txt, ni = f.result()
                except Exception as e:
                    fatal_if_exhausted(e)                         # D-34
                    failures.append(str(e)[:200]); consec += 1
                    if consec >= FAIL_FAST_N:                     # T6
                        raise SystemExit(f"FATAL: {consec} consecutive failures: {failures[-1]}")
                    continue
                consec = 0
                if txt is not None:
                    preds[i] = txt
                    n_img += ni
                    n_noimg += (1 if ni == 0 else 0)
            ck.write_text(json.dumps({"RUN_CONFIG": RUN_CONFIG,                    # T7
                                      "predictions": {str(k): v for k, v in preds.items()}}))
            print(f"  [{len(preds)}/{len(idx)}] {(time.time()-t0)/60:.1f} min", flush=True)

    out = {"RUN_CONFIG": RUN_CONFIG, "arm": a.arm, "benchmark": "mpdocvqa",
           "n_questions": len(idx), "n_generated": len(preds),
           "page_images_attached": n_img, "questions_with_no_image": n_noimg,
           "failures": failures[:20], "n_failures": len(failures),
           "predictions": [preds.get(i) for i in range(len(qs))],
           "wall_seconds": round(time.time() - t0, 1)}
    # A --limit run is a probe, not the arm. Writing it under the canonical name makes the
    # next driver pass skip the arm as "present" with a fraction of the questions -- which
    # is exactly what happened on the first launch (6/978 under the canonical name).
    name = (f"PARTIAL_p3_generation_{a.arm}_limit{a.limit}.json" if a.limit
            else f"p3_generation_{a.arm}.json")
    (OUT / name).write_text(json.dumps(out, indent=1))
    print(f"  {a.arm}: {len(preds)}/{len(idx)} generated, {n_img} images attached, "
          f"{len(failures)} failures, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
