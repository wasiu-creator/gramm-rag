#!/usr/bin/env python3
"""flat_k3_select.py -- re-run the flat-vector page selection, at k=2 (Gate 1) and k=3.

C-77(ii): the flat-vector control reads fewer pages than every arm compared against it. Its
context is NOT a prefix of its stored `retrieved_pages` (2.9% match on MMLongBench): those are the
pages of the 20 retrieved NODES, while the context is chosen by a cross-encoder rerank over them.
A third page therefore has to be selected, not sliced.

The selection path is reached by EXECUTING THE CANONICAL NOTEBOOK CELLS (the C-49 rule: shared code
is imported, never copied). The cells are located by content, and each one's index and SHA-256 is
printed before it runs, so a divergence is visible in the log rather than inferred (rule 22).

GATE 1, pre-registered in FLAT_K3_PREREGISTRATION.md: re-running at n_pages=2 must reproduce each
archived context exactly, >= 99.5% of questions per corpus. Below that, stop and spend nothing.

NO SPEND IS POSSIBLE HERE: VLMClient.generate is replaced with a raiser before any notebook cell is
executed, so a cell that tried to generate would crash rather than bill.

  docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \\
    -v /media/volume/vol3/grammrag:/work -w /work/gramm-rag-<corpus> \\
    grammrag-retrieval-pinned:latest python /work/gramm-rag-colpali/flat_k3_select.py --corpus <c>
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

W = Path("/work")
CORPORA = {
    "mmlongbench": dict(repo="gramm-rag-mmlongbench", nb="mmlongbench_e2e.ipynb",
                        archived="gramm-rag-3benchmarks/data/mmlongbench/baseline_vector_mmlb.json"),
    "docbench": dict(repo="gramm-rag-docbench", nb="docbench_e2e.ipynb",
                     archived="gramm-rag-3benchmarks/data/docbench/baseline_vector_docbench.json"),
    "mpdocvqa": dict(repo="gramm-rag-mpdocvqa", nb="mpdocvqa_e2e.ipynb",
                     archived="gramm-rag-3benchmarks/data/mpdocvqa/baseline_vector_mpdocvqa_pilot.json"),
}
# Cells are PINNED by index, per corpus, and their SHA-256 is recorded in the output. Content
# matching was tried first and picked the wrong set twice: it refused the shared import cell
# because that cell mentions "pip install" in a comment, leaving `os` undefined, and it pulled in
# unrelated cells that merely mention N_PILOT. The pinned set is: paths/constants, shared imports
# and DEVICE, directories, the question list, the retrieval helpers (embed_query / cross_rerank /
# page_context_nodes / node_page_of) and the in-memory FAISS build. Nothing that trains, plots,
# generates or judges is in it, and VLMClient.generate is disabled besides.
CELLS = {"mmlongbench": [1, 2, 3, 5, 27, 37],
         "docbench":    [1, 2, 3, 5, 21, 32],
         "mpdocvqa":    [1, 2, 3, 5, 26, 36]}


def cells_to_run(nb, corpus):
    out = []
    for i in CELLS[corpus]:
        c = nb["cells"][i]
        if c["cell_type"] != "code":
            raise SystemExit(f"FATAL: pinned cell {i} of {corpus} is not code -- notebook changed")
        src = "".join(c["source"])
        # Notebook magics (%matplotlib, !pip) are not Python: comment them out, as nbconvert does.
        src = "\n".join(("# [magic stripped] " + ln) if ln.lstrip()[:1] in ("%", "!") else ln
                         for ln in src.splitlines())
        out.append((i, src))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=list(CORPORA))
    ap.add_argument("--limit", type=int, default=None, help="probe only; never for a reported run")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                    help="force the torch device the canonical cells use. The archive's own "
                         "environment decides: MP-DocVQA was embedded on CPU (1024-d), so its "
                         "reranker ran on CPU too.")
    ap.add_argument("--tag", default="", help="suffix for the output file (determinism re-runs)")
    ap.add_argument("--root", default=None,
                    help="working dir for the notebook (the compose layout mounts the repo at /app "
                         "with cache/parsed -> /app/parsed etc.; the notebook resolves ROOT from CWD)")
    a = ap.parse_args()
    C = CORPORA[a.corpus]
    repo = Path(a.root) if a.root else W / C["repo"]
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    nb_path = repo / C["nb"]
    if not nb_path.exists():
        raise SystemExit(f"FATAL: {nb_path} absent -- is the repo mounted at {repo}?")

    # Make spend impossible before any canonical cell runs.
    import src.generation.vlm_client as _vc
    def _no_spend(*_a, **_k):
        raise SystemExit("FATAL: a cell tried to call the generator during the free selection pass")
    _vc.VLMClient.generate = _no_spend

    nb = json.loads(nb_path.read_text())
    G = {"__name__": "__canonical__"}
    print(f"executing canonical cells from {C['nb']}:", flush=True)
    cell_hashes = {}
    for i, s in cells_to_run(nb, a.corpus):
        h = hashlib.sha256(s.encode()).hexdigest()[:12]
        cell_hashes[i] = h
        print(f"  cell {i:3d}  sha256:{h}  {len(s):6d} ch", flush=True)
        exec(compile(s, f"{C['nb']}#cell{i}", "exec"), G)
    if a.device != "auto":
        G["DEVICE"] = a.device
        print(f"  DEVICE forced to {a.device} (archive-matching)", flush=True)
    for name in ("embed_query", "cross_rerank", "page_context_nodes", "faiss_indices",
                 "RERANK_POOL", "USE_CROSS_ENCODER", "PAGE_CONTEXT_N", "node_page_of", "PARSED_DIR"):
        if name not in G:
            raise SystemExit(f"FATAL: {name} is not defined after executing the canonical cells")
    print(f"  RERANK_POOL={G['RERANK_POOL']} USE_CROSS_ENCODER={G['USE_CROSS_ENCODER']} "
          f"PAGE_CONTEXT_N={G['PAGE_CONTEXT_N']} faiss docs={len(G['faiss_indices'])}", flush=True)

    # D-32 GUARD. load_text_model() falls back from E5-Mistral-7B (4096-d) to E5-large-v2
    # (1024-d) on ANY exception, including "no GPU visible". Running the container with --user
    # hides /dev/nvidia*, CUDA reads False, and the fallback fires silently: every query would be
    # embedded in the wrong space. Check the query against the index before touching the corpus.
    idims = {int(v.index.d) for v in G["faiss_indices"].values() if getattr(v, "index", None)}
    if not idims:
        raise SystemExit("FATAL: no FAISS index was built -- the embedding cache is not mounted")
    # The QUERY encoder is dictated by the archive's own index, not by what this host can load.
    # MP-DocVQA's 927 documents are embedded with E5-large-v2 (1024-d) because that corpus was
    # embedded on CPU and the notebook asks for 4-bit E5-Mistral only when DEVICE == 'cuda'.
    # Querying its index with E5-Mistral would be a different space, so the encoder follows the
    # index and the choice is recorded in the artefact.
    encoder = "notebook default: load_text_model(use_4bit=DEVICE=='cuda')"
    if idims == {1024}:
        from src.graph.embeddings import load_text_model
        G["_QUERY_MODEL"], _ = load_text_model(use_4bit=False, device=G.get("DEVICE", "cpu"))
        encoder = "intfloat/e5-large-v2 (forced: the stored index is 1024-d)"
    print(f"  query encoder: {encoder}", flush=True)
    _probe = G["embed_query"]("dimension guard probe")
    qdim = int(getattr(_probe, "shape", [0])[-1])
    print(f"  query dim {qdim}; index dims {sorted(idims)}", flush=True)
    if idims != {qdim}:
        raise SystemExit(f"FATAL: query dim {qdim} against index dims {sorted(idims)} -- the "
                         f"E5-Mistral 4-bit load fell back to E5-large-v2 (D-32). Refusing to run.")

    arch = json.loads((W / C["archived"]).read_text())
    ctx_archived = arch["context_pages"]
    ret_archived = arch.get("retrieved_pages")   # pages of the top-RERANK_POOL nodes, pre-rerank
    questions = G.get("pilot_questions") or G.get("questions")
    if questions is None:
        raise SystemExit("FATAL: the canonical cells did not leave the question list in scope")
    n = len(ctx_archived)
    if len(questions) != n:
        raise SystemExit(f"FATAL: {len(questions)} questions against {n} archived contexts")
    idx = range(n if a.limit is None else min(a.limit, n))

    embed_query, cross_rerank = G["embed_query"], G["cross_rerank"]
    page_context_nodes = G["page_context_nodes"]
    # MMLongBench defines parsed_path; DocBench and MP-DocVQA glob PARSED_DIR instead.
    parsed_path = G.get("parsed_path") or (lambda d: G["PARSED_DIR"] / f"{d}.json")
    faiss_indices, pool = G["faiss_indices"], G["RERANK_POOL"]

    def ctx_pages(gen_nodes, parsed):
        t2e = {}
        for e in (parsed or {}).get("elements", []):
            t2e.setdefault(e["type"], []).append(e)
        out = []
        for nd in gen_nodes:
            pg = G["node_page_of"](nd, t2e)
            if pg is not None and pg not in out:
                out.append(pg)
        return out

    rows, mismatches, pcache = [], [], {}
    t0 = time.time()
    for j, i in enumerate(idx):
        item = questions[i]
        doc_id = Path(item["doc_id"]).stem
        q = item["question"]
        vi = faiss_indices.get(doc_id)
        nodes = vi.search(embed_query(q), top_k=pool) if vi and vi.index else []
        pf = parsed_path(doc_id)
        if doc_id not in pcache:
            pcache[doc_id] = json.loads(pf.read_text(encoding="utf-8")) if pf.exists() else {"elements": []}
        parsed = pcache[doc_id]
        ranked = cross_rerank(list(nodes), q, doc_id, parsed) if G["USE_CROSS_ENCODER"] else \
            sorted(nodes, key=lambda x: x.get("score", 0.0), reverse=True)
        # Where does a divergence come from -- FAISS retrieval, or the cross-encoder? The archive
        # stores the pages of its own top-20 nodes, so the two stages can be separated.
        retro = G["node_pages"](nodes, parsed) if "node_pages" in G else None
        ret_ok = None if (ret_archived is None or retro is None) else (retro == ret_archived[i])
        k2 = ctx_pages(page_context_nodes(ranked, parsed, 2), parsed)
        k3 = ctx_pages(page_context_nodes(ranked, parsed, 3), parsed)
        ok = (k2 == ctx_archived[i])
        # SUPERSET CONSTRUCTION (amendment 1): keep the archived context exactly, then append the
        # highest-ranked page this reranker offers that is not already in it. The published pages
        # survive by construction, so reproduction is a fidelity measure and not a precondition.
        order = ctx_pages(page_context_nodes(ranked, parsed, 6), parsed)
        added = next((pg for pg in order if pg not in ctx_archived[i]), None)
        sup = list(ctx_archived[i]) + ([added] if added is not None else [])
        if not ok:
            mismatches.append(dict(i=i, doc=doc_id, archived=ctx_archived[i], reproduced=k2))
        rows.append(dict(i=i, doc=doc_id, k2=k2, k3=k3, reproduces=ok, retrieval_matches=ret_ok,
                         retrieved=retro,
                         archived=list(ctx_archived[i]), added_page=added, superset=sup,
                         n_superset=len(sup)))
        if (j + 1) % 100 == 0:
            r = sum(x["reproduces"] for x in rows) / len(rows)
            print(f"  {j+1}/{len(list(idx))}  reproduction {r:.3%}  {time.time()-t0:.0f}s", flush=True)

    rate = sum(x["reproduces"] for x in rows) / len(rows)
    rr = [x["retrieval_matches"] for x in rows if x["retrieval_matches"] is not None]
    ret_rate = (sum(rr) / len(rr)) if rr else None
    # Of the questions whose CONTEXT differs, how many also had a different retrieval pool?
    bad = [x for x in rows if not x["reproduces"] and x["retrieval_matches"] is not None]
    ret_bad = sum(1 for x in bad if not x["retrieval_matches"])
    if ret_rate is not None:
        print(f"  retrieval (top-{pool} node pages) reproduces on {ret_rate:.3%}; of {len(bad)} context "
              f"mismatches, {ret_bad} also differ in retrieval -> "
              f"{'FAISS' if ret_bad > len(bad)/2 else 'reranker'} is the dominant source", flush=True)
    out = W / f"gramm-rag-colpali/cost/flat_k3_selection_{a.corpus}{a.tag}.json"
    out.write_text(json.dumps(dict(
        corpus=a.corpus, n=len(rows), limit=a.limit,
        gate1_threshold=0.995, gate1_reproduction_rate=round(rate, 5),
        gate1_pass=bool(rate >= 0.995 and a.limit is None),
        cells_executed=cell_hashes, query_encoder=encoder, index_dim=sorted(idims),
        device=G.get("DEVICE"), retrieval_reproduction_rate=ret_rate,
        context_mismatches_with_retrieval_mismatch=ret_bad,
        run_config=dict(RERANK_POOL=pool, USE_CROSS_ENCODER=G["USE_CROSS_ENCODER"],
                        PAGE_CONTEXT_N_archived=G["PAGE_CONTEXT_N"], K3=3, SEED=42),
        construction="superset: archived context + highest-ranked reranked page not already in it "
                     "(FLAT_K3_PREREGISTRATION.md amendment 1)",
        mean_pages_archived=round(sum(len(x["archived"]) for x in rows) / len(rows), 3),
        mean_pages_superset=round(sum(len(x["superset"]) for x in rows) / len(rows), 3),
        n_no_page_added=sum(1 for x in rows if x["added_page"] is None),
        mean_pages_k2=round(sum(len(x["k2"]) for x in rows) / len(rows), 3),
        mean_pages_k3=round(sum(len(x["k3"]) for x in rows) / len(rows), 3),
        mismatches=mismatches[:200], n_mismatches=len(mismatches), rows=rows), indent=1))
    print(f"\nGATE 1 {a.corpus}: reproduction {rate:.3%} of {len(rows)} "
          f"(threshold 99.5%) -> {'PASS' if rate >= 0.995 else 'FAIL'}"
          + ("  [LIMITED PROBE, not a gate result]" if a.limit else ""))
    print(f"  mean pages k=2 {sum(len(x['k2']) for x in rows)/len(rows):.2f} -> "
          f"k=3 {sum(len(x['k3']) for x in rows)/len(rows):.2f}")
    print(f"  SUPERSET: archived {sum(len(x['archived']) for x in rows)/len(rows):.2f} pages -> "
          f"{sum(len(x['superset']) for x in rows)/len(rows):.2f} pages; "
          f"{sum(1 for x in rows if x['added_page'] is None)} questions gained no page")
    print(f"  wrote {out}")
    return 0 if (rate >= 0.995 or a.limit) else 1


if __name__ == "__main__":
    sys.exit(main())
