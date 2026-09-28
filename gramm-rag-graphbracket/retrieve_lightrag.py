"""
WP-1 Tier 1 stage 2 -- LightRAG retrieval for MMLongBench, emitted in the ColPali arm's
exact retrieval format so that `generate_colpali.py --arm lightrag` runs the ENTIRE
downstream path unchanged: same synthetic page nodes, same prompt builder (PROMPT_V=3,
12,000 chars), same generator (gpt-4.1-mini, temp 0.0), same 120-dpi page images, same
scorer. Only retrieval differs -- which is the whole claim.

PAGE PROVENANCE. LightRAG chunks text and does not track page numbers, but our chunks are
built from `mapper.elements_to_text`, whose elements each carry `page_no`. So pages are
RECOVERED by matching returned chunk text back to source elements. This changes nothing
about what was ingested -- the alternative (injecting page markers into the ingested text)
would have given this arm a different input from every other text-side arm.

The recovery is measured, not assumed: `alignment_rate` (matched element characters /
returned chunk characters) is computed per question and reported with the results. A
mapping whose quality is asserted rather than measured is exactly the class of claim this
project keeps finding to be wrong.

Retrieval mode `hybrid` was pre-registered in GATE_G_GRAPH_BRACKET.md before any quality
number existed; four modes is four chances to report the best one.

Scope: one store per document (C-35), matching every other arm.
"""
import argparse, asyncio, json, os, re, sys, time
from collections import defaultdict
from pathlib import Path

GB = Path(__file__).resolve().parent
sys.path.insert(0, str(GB))
import mapper                                          # noqa: E402
from pilot_ingest import Meter, build_funcs            # noqa: E402

# Benchmark-specific wiring. DocBench differs in three ways and each is handled explicitly
# rather than by a default that silently does the MMLongBench thing:
#   1. questions live in the baseline file, not a separate source JSON;
#   2. there are no gold_pages -- localisation is quote-recall@5, computed by the scorer from
#      page_texts, so retrieval only has to emit context_pages;
#   3. the judge is gpt-4-turbo (handled downstream in score_colpali.py).
BENCHES = {
    "mmlongbench": dict(
        parsed="/media/volume/vol3/grammrag/gramm-rag-mmlongbench/cache/parsed",
        d3="/media/volume/vol3/grammrag/gramm-rag-3benchmarks/data/mmlongbench",
        base="baseline_vector_mmlb.json", qmeta="qmeta_mmlb.json",
        src="/media/volume/vol3/grammrag/gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json",
        store="lightrag_mmlongbench"),
    "docbench": dict(
        parsed="/media/volume/vol3/grammrag/gramm-rag-docbench/cache/parsed",
        d3="/media/volume/vol3/grammrag/gramm-rag-3benchmarks/data/docbench",
        base="baseline_vector_docbench.json", qmeta="qmeta_docbench.json",
        src=None, store="lightrag_docbench"),
}
TOPK = 3                     # matches the ColPali arm's context budget exactly
MIN_MATCH_CHARS = 30         # below this, substring matching is noise, not evidence


def parse_chunks(ctx: str) -> list[str]:
    """Pull the `content` fields out of the Document Chunks JSON block."""
    i = ctx.find("Document Chunks")
    if i < 0:
        return []
    out = []
    for line in ctx[i:].splitlines():
        line = line.strip()
        if not line.startswith('{"reference_id"'):
            continue
        try:
            out.append(json.loads(line).get("content", "") or "")
        except json.JSONDecodeError:
            continue
    return out


def _norm(s: str) -> str:
    """Whitespace-insensitive, case-insensitive view for matching."""
    return re.sub(r"\s+", " ", s).strip().lower()


SHINGLE = 50          # characters per probe
INDEX_STRIDE = 1      # element side: EVERY offset -- see below
PROBE_STRIDE = 50     # chunk side: non-overlapping, so each hit accounts for 50 chars


def build_shingle_index(parsed: dict):
    """Index the FULL ingested text at stride 1, mapping each offset to its page.

    Three attempts, recorded because the differences are instructive and the number is
    reported rather than assumed:

      1. whole-element substring match -> 0.418, but over-credits: a matched element is
         credited its entire length, and an element appearing in two chunks counts twice.
      2. 50-char shingles, stride 50 on BOTH sides -> 0.027. Two non-overlapping griddings
         coincide only when their offsets agree modulo 50; that is an artefact of grid
         alignment, not a property of the retrieval.
      3. stride 1 on the element side -> 0.325, honest but still floored: elements shorter
         than 50 characters were skipped entirely, so their characters could never match
         however faithfully they were returned.

    This version indexes the exact string that was ingested -- `mapper.elements_to_text`
    output, headings and all -- so every ingested character is eligible, and carries a
    parallel offset->page map so attribution survives element boundaries.
    """
    import bisect
    order = sorted(parsed["elements"], key=lambda x: x.get("reading_order", 0))
    parts, starts, pages, pos = [], [], [], 0
    for el in order:
        raw = (el.get("text") or "").strip()
        if not raw:
            continue
        rendered = f"## {raw}" if el.get("type") == "section" else raw
        norm = _norm(rendered)
        if not norm:
            continue
        starts.append(pos); pages.append(int(el.get("page_no", 0) or 0))
        parts.append(norm); pos += len(norm) + 1          # +1 for the joining space
    joined = " ".join(parts)
    idx = {}
    for k in range(0, len(joined) - SHINGLE + 1, INDEX_STRIDE):
        h = hash(joined[k:k + SHINGLE])
        if h not in idx:
            j = bisect.bisect_right(starts, k) - 1
            idx[h] = pages[j] if j >= 0 else 0
    return idx


IMG_PATH_RE = re.compile(r"Image Path:\s*(\S+)")


def build_image_page_map(parsed: dict, artifacts_dir: Path) -> dict:
    """Exact img_path -> 1-based page_no map for RAG-Anything's multimodal chunks.

    RAG-Anything ingests image items and stores a VLM-written "Image Content Analysis"
    block for each. That text is GENERATED, so it has no character preimage in the
    parse and the shingle index cannot attribute a single character of it -- measured
    align 0.000 on those chunks. Scoring the S2 arm on the text half alone would
    understate its localisation exactly on the multimodal content that distinguishes
    it, and would bias the pre-registered G-5 mechanism test against it.

    The recovery is exact, not fuzzy: `mapper.elements_to_content_list` writes the
    resolved absolute `img_path` into the item alongside `page_idx`, and RAG-Anything
    echoes that same path back in the chunk. So this is a dictionary lookup on a key
    the mapper itself produced.

    CONVENTION: content_list uses 0-based `page_idx` (= page_no - 1, mapper D3), while
    build_shingle_index keys on 1-based `page_no`. This returns page_no so both routes
    land in ONE coordinate system. An off-by-one here would shift every multimodal
    attribution by exactly one page and still look like a plausible result.
    """
    m, n_img_items = {}, 0
    for it in mapper.elements_to_content_list(parsed, artifacts_dir):
        if it.get("type") == "image" and it.get("img_path"):
            n_img_items += 1
            m[it["img_path"]] = int(it.get("page_idx", 0)) + 1
    # A route that silently maps nothing is indistinguishable, downstream, from an arm
    # that simply retrieves no images -- and the second is a result while the first is
    # a bug. Fail here instead of emitting the ambiguity.
    assert n_img_items == 0 or m, (
        f"image page map empty despite {n_img_items} image items in {artifacts_dir}")
    return m


def attribute_pages(chunks: list[str], shingles: dict, img_pages: dict | None = None):
    """Map returned chunk text back to pages via shingle lookup.

    Returns (pages ranked by attributed character mass, alignment_rate, chunk chars).
    alignment_rate = attributed characters / returned characters. Probes are
    non-overlapping, so the rate is a genuine coverage fraction bounded by 1.
    """
    mass, best_rank, matched, total = defaultdict(int), {}, 0, 0
    img_matched, img_chunks = 0, 0
    for ci, c in enumerate(chunks):
        cn = _norm(c)
        total += len(cn)
        # ROUTE 2 -- multimodal chunks, attributed by exact image path, not by text.
        # Probes are non-overlapping (PROBE_STRIDE == SHINGLE), so a fully-attributed
        # text chunk accrues exactly len(cn) mass; giving an image chunk len(cn) puts
        # both routes on the SAME mass scale, so ranking stays comparable.
        mp = IMG_PATH_RE.search(c) if img_pages else None
        if mp and mp.group(1) in img_pages:
            pg = img_pages[mp.group(1)]
            mass[pg] += len(cn)
            matched += len(cn)
            img_matched += len(cn)
            img_chunks += 1
            best_rank.setdefault(pg, ci)
            continue
        for k in range(0, max(len(cn) - SHINGLE + 1, 0), PROBE_STRIDE):
            pg = shingles.get(hash(cn[k:k + SHINGLE]))
            if pg:
                mass[pg] += SHINGLE
                matched += SHINGLE
                if pg not in best_rank:
                    best_rank[pg] = ci          # LightRAG's OWN relevance order
    rate = (matched / total) if total else 0.0
    img_rate = (img_matched / total) if total else 0.0
    # Rank pages by the retriever's ordering, NOT by how much text they happen to hold.
    # LightRAG returns Document Chunks in relevance order, so a page's rank is the rank of
    # the best chunk that maps to it; character mass breaks ties only.
    # Ranking by mass alone (the first implementation) discarded the retriever's ranking
    # entirely and measured "which pages carry the most text among those LightRAG touched"
    # -- a different quantity, and one no other arm is scored on. ColPali ranks by MaxSim,
    # GraMM by graph score; both are relevance. See C-37.
    ranked = [pg for pg, _ in sorted(mass.items(),
                                     key=lambda kv: (best_rank.get(kv[0], 10**6), -kv[1]))]
    return (ranked, round(min(rate, 1.0), 4), total,
            round(min(img_rate, 1.0), 4), img_chunks)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-model", default="gpt-4o-mini")
    ap.add_argument("--emb-model", default="text-embedding-3-small")
    ap.add_argument("--mode", default="hybrid")          # pre-registered
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bench", default="mmlongbench", choices=list(BENCHES))
    ap.add_argument("--no-image-route", action="store_true",
                    help="C-49 SENSITIVITY ONLY: disable the img_path->page route, "
                         "reproducing the text-only attribution that understated the "
                         "multimodal arm. Writes to a _textonly filename; never "
                         "overwrites the canonical artefact.")
    ap.add_argument("--arm", default="lightrag",
                    help="'lightrag' (per-bench store dirs) or 's2' (RAG-Anything, "
                         "stores named <bench>__<stem> in one directory)")
    a = ap.parse_args()
    SPEC = BENCHES[a.bench]
    BENCH, PARSED = a.bench, Path(SPEC["parsed"])
    D3 = Path(SPEC["d3"])
    key = os.environ.get("OPENAI_API_KEY")
    if not key or not key.startswith("sk-"):
        raise SystemExit("FATAL: OPENAI_API_KEY absent or malformed -- refusing to run")

    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import (initialize_pipeline_status,
                                            finalize_share_data)

    base = json.loads((D3 / SPEC["base"]).read_text())
    qm = json.loads((D3 / SPEC["qmeta"]).read_text())["rows"]
    n = base["n_questions"]
    if SPEC["src"]:
        srcq = json.loads(Path(SPEC["src"]).read_text())
        assert len(qm) == len(srcq) == n, "index alignment broken across metadata files"
        questions = [s["question"] for s in srcq]
    else:
        assert len(qm) == n, "index alignment broken across metadata files"
        assert "questions" in base, f"{SPEC['base']} carries no questions list"
        questions = list(base["questions"])
    docs = [r["doc_id"] for r in qm]

    meter = Meter()
    llm_f, emb_f = build_funcs(meter, a.llm_model, a.emb_model, key)
    dim = len((await emb_f(["dimension probe"]))[0])
    stores = (GB / "tier1" / "s2_raganything" / "stores") if a.arm == "s2" \
             else (GB / "tier1" / SPEC["store"] / "stores")

    by_doc = defaultdict(list)
    idx = list(range(n))[:a.limit] if a.limit else list(range(n))
    for i in idx:
        by_doc[docs[i]].append(i)

    ctx_pages = [[] for _ in range(n)]
    ret_pages = [[] for _ in range(n)]
    align = [None] * n
    img_align = [None] * n
    img_nchunks = [0] * n
    missing_stores, empty_attr = [], []
    t0 = time.time()

    for di, (doc, idxs) in enumerate(sorted(by_doc.items()), 1):
        stem = doc[:-4] if doc.lower().endswith(".pdf") else doc
        store = stores / (f"{BENCH}__{stem}" if a.arm == "s2" else stem)
        if not (store / "_INDEX_COMPLETE.json").exists():
            missing_stores.append(doc)                    # never silently score a gap
            continue
        pf = PARSED / f"{doc}.json"
        if not pf.exists():
            pf = PARSED / f"{stem}.json"
        if not pf.exists():                       # DocBench parses are named "<id>.json"
            pf = PARSED / f"{doc.split('.')[0]}.json"
        assert pf.exists(), f"no parse found for {doc}"
        parsed = mapper.load_parse(pf)
        shingles = build_shingle_index(parsed)
        # Route 2 is armed for the S2 arm only: it is the only arm whose store holds
        # VLM-generated image analyses. LightRAG ingests text alone, so the map would
        # be empty and the route inert -- but keeping it arm-gated makes that explicit
        # rather than incidental.
        img_pages = (build_image_page_map(parsed, PARSED / "artifacts" / stem)
                     if (a.arm == "s2" and not a.no_image_route) else None)

        rag = LightRAG(working_dir=str(store), llm_model_func=llm_f,
                       llm_model_name=a.llm_model,
                       embedding_func=EmbeddingFunc(embedding_dim=dim, func=emb_f))
        await rag.initialize_storages()
        await initialize_pipeline_status()
        for i in idxs:
            ctx = await rag.aquery(questions[i],
                                   param=QueryParam(mode=a.mode, only_need_context=True))
            chunks = parse_chunks(ctx if isinstance(ctx, str) else "")
            ranked, rate, _, irate, ichunks = attribute_pages(chunks, shingles, img_pages)
            align[i] = rate
            img_align[i] = irate
            img_nchunks[i] = ichunks
            ret_pages[i] = ranked
            ctx_pages[i] = ranked[:TOPK]
            if not ranked:
                empty_attr.append(i)
        await rag.finalize_storages()
        # C-36: lightrag.kg.shared_storage keeps MODULE-LEVEL globals (_shared_dicts,
        # _init_flags, ...). Opening a second LightRAG in the same process without
        # resetting them makes every document after the first read the FIRST document's
        # in-memory namespaces -- queries return nothing attributable, silently, with
        # exit 0. finalize_share_data() resets those globals in single-process mode.
        finalize_share_data()
        if di % 10 == 0 or di == len(by_doc):
            print(f"  [{di:>3}/{len(by_doc)}] docs  {time.time()-t0:6.0f}s  "
                  f"${meter.snapshot()['cost_usd']:.4f}", flush=True)

    rates = [r for r in align if r is not None]
    # BLOCKING VALIDITY ASSERTION (constraint 7's shape). If most questions attribute to no
    # page at all, that is a broken harness, not a weak retriever -- and the two are
    # indistinguishable from the metrics alone. Reporting the first as the second would
    # understate a competitor, which is the direction this project is least likely to
    # challenge. Halt instead.
    # Second blocking guard, S2 only: the combined corpus holds 14,581 images and the
    # S2 stores demonstrably carry "Image Content Analysis" chunks. If NOT ONE was
    # attributed across the whole run, route 2 silently no-op'd (a path change, a
    # resolve() difference) and the arm is being scored on its text half alone.
    if a.arm == "s2" and not a.no_image_route and not any(img_nchunks):
        raise SystemExit(
            "FATAL: S2 run attributed zero image chunks across all questions. "
            "Route 2 is not matching img_path keys -- refusing to emit a text-only "
            "score for a multimodal arm.")
    zero_frac = (sum(1 for r in rates if r == 0.0) / len(rates)) if rates else 1.0
    if zero_frac > 0.5:
        raise SystemExit(
            f"FATAL: {zero_frac:.1%} of questions attributed to ZERO pages. That is a "
            f"harness failure, not a retrieval result -- refusing to emit it. "
            f"(mean alignment {sum(rates)/len(rates):.4f})")
    RUN_CONFIG = {"RETRIEVER": a.arm, "LIGHTRAG_VERSION": "1.5.7",
                  "LIGHTRAG_MODE": a.mode, "MODE_PREREGISTERED": True,
                  "RETRIEVAL_SCOPE": "per-document store (C-35)",
                  "LLM_MODEL": a.llm_model, "EMB_MODEL": a.emb_model,
                  "TOPK": TOPK,
                  "PAGE_ATTRIBUTION": ("chunk->ingested-text shingle map"
                                       + (" + img_path->page_idx exact map (S2 only)"
                                          if a.arm == "s2" else "")),
                  "PAGE_RANKING": "LightRAG chunk relevance order; mass breaks ties (C-37)",
                  "SHINGLE_CHARS": SHINGLE, "INDEX_STRIDE": INDEX_STRIDE, "PROBE_STRIDE": PROBE_STRIDE, "BENCHMARK": BENCH}
    out = {"system": "LightRAG 1.5.7 (hybrid)", "benchmark": BENCH, "n_questions": n,
           "RUN_CONFIG": RUN_CONFIG, "context_pages": ctx_pages,
           "retrieved_pages": ret_pages,
           "alignment_rate": align,
           "alignment_image_route": img_align,
           "image_chunks_per_q": img_nchunks,
           "alignment_route_split": {
               "note": ("total alignment = text-shingle route + image-path route. "
                        "The image route attributes VLM-generated 'Image Content "
                        "Analysis' chunks, which have no character preimage in the "
                        "parse, via the exact img_path the mapper itself wrote."),
               "mean_image_route": (round(sum(r for r in img_align if r is not None)
                                          / max(len([r for r in img_align
                                                     if r is not None]), 1), 4)),
               "questions_with_image_chunks": sum(1 for c in img_nchunks if c),
           },
           "alignment_summary": {
               "n_scored": len(rates),
               "mean": round(sum(rates) / len(rates), 4) if rates else None,
               "median": round(sorted(rates)[len(rates) // 2], 4) if rates else None,
               "below_0.5": sum(1 for r in rates if r < 0.5),
               "zero": sum(1 for r in rates if r == 0.0)},
           "questions_with_no_attributed_page": len(empty_attr),
           "missing_stores": missing_stores,
           "retrieval_cost_usd": round(meter.snapshot()["cost_usd"], 6),
           # C-38 class, self-inflicted: a --limit run writes a file with n_questions=N and
           # only `limit` entries populated. Downstream it looks complete and scores the rest
           # as retrieval misses. The flag makes partial artefacts announce themselves.
           "COMPLETE": a.limit is None,
           "n_populated": sum(1 for p in ctx_pages if p),
           "limit_used": a.limit}
    d = GB / "tier1" / ("s2_raganything" if a.arm == "s2" else SPEC["store"])
    if a.limit is not None:                       # never let a partial occupy the real path
        out_name = (f"PARTIAL_{a.arm}_retrieval_{BENCH}"
                    + ("_textonly" if a.no_image_route else "") + ".json")
        print(f"  NOTE: --limit run -> writing {out_name}, not the canonical filename")
    else:
        out_name = (f"{a.arm}_retrieval_{BENCH}"
                    + ("_textonly" if a.no_image_route else "") + ".json")
    (d / out_name).write_text(json.dumps(out))
    print(f"\nwrote {d}/{out_name}")
    print(f"  alignment: mean {out['alignment_summary']['mean']} "
          f"median {out['alignment_summary']['median']}  "
          f"zero-attribution {out['alignment_summary']['zero']}  "
          f"missing stores {len(missing_stores)}")
    print(f"  retrieval cost ${out['retrieval_cost_usd']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
