#!/usr/bin/env python3
"""S4 -- MMGraphRAG indexing under harness identity.

Injects THIS project's MinerU parse into MMGraphRAG rather than letting it re-parse the
PDF (which would reintroduce parsing as a confound). The vendor's `_run_mineru` already
short-circuits when <working>/<stem>/auto/ holds a .md, so injection needs no vendor
patch: we write that directory ourselves.

GUARDS
  * C-52: the vendor's broad `except Exception` absorbs any guard that signals by raising
    or by exit code, so parse provenance is verified by a CONTENT assertion outside the
    vendor call -- store chunk text vs the markdown we injected, per document.
  * rule 22: that assertion emits its number on EVERY document, pass or fail, so the guard
    is visibly live rather than silent-on-success.
  * C-48: cost is projected on IMAGES (the driver in a multimodal arm) as well as chars;
    rule 9 anomaly flags require both denominators.
  * D-34: credit exhaustion is fatal, never a degraded score.
"""
import argparse, asyncio, json, os, re, shutil, statistics, sys, time
from pathlib import Path

GB = Path(__file__).resolve().parent
VEND = GB / "vendor" / "mmgraphrag"
sys.path.insert(0, str(GB))
import mapper                                                   # noqa: E402

BENCHES = {
    "mmlongbench": dict(parsed="/media/volume/vol3/grammrag/gramm-rag-mmlongbench/cache/parsed"),
    "docbench":    dict(parsed="/media/volume/vol3/grammrag/gramm-rag-docbench/cache/parsed"),
}
SHINGLE, STRIDE = 50, 50
ALIGN_FLOOR = 0.95          # pre-registered in S4_PILOT_PREREGISTRATION.md
RATES = {"gpt-4o-mini": {"in": 1.5e-07, "out": 6e-07}}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


class Meter:
    """Token/cost meter installed at the openai module level, so the vendor is unmodified."""
    def __init__(self):
        self.calls = self.pin = self.pout = 0
        self.vision_calls = 0

    def add(self, usage, had_image: bool):
        self.calls += 1
        self.pin += getattr(usage, "prompt_tokens", 0) or 0
        self.pout += getattr(usage, "completion_tokens", 0) or 0
        if had_image:
            self.vision_calls += 1

    @property
    def cost(self):
        r = RATES["gpt-4o-mini"]
        return self.pin * r["in"] + self.pout * r["out"]

    def snapshot(self):
        return dict(calls=self.calls, vision_calls=self.vision_calls,
                    prompt_tokens=self.pin, completion_tokens=self.pout,
                    cost_usd=round(self.cost, 6))


def install_meter(meter: Meter):
    """Wrap AsyncOpenAI/OpenAI before the vendor imports them. No vendor code is touched."""
    import openai
    from openai import AsyncOpenAI, OpenAI

    def _has_image(kwargs):
        for m in kwargs.get("messages", []) or []:
            c = m.get("content")
            if isinstance(c, list) and any(
                    isinstance(x, dict) and x.get("type") == "image_url" for x in c):
                return True
        return False

    def _fatal_if_exhausted(e):
        # D-34: OpenAI reports exhausted credit as RateLimitError, the same class as ordinary
        # throttling. Scoring 0 for those is how a complete-looking, understated run happens.
        msg = str(e).lower()
        if any(k in msg for k in ("insufficient_quota", "no credits", "billing")):
            raise SystemExit(f"FATAL: OpenAI credit exhausted -- {e}")

    class MeteredAsync(AsyncOpenAI):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            orig = self.chat.completions.create
            async def create(*aa, **kk):
                try:
                    r = await orig(*aa, **kk)
                except Exception as e:
                    _fatal_if_exhausted(e); raise
                if getattr(r, "usage", None):
                    meter.add(r.usage, _has_image(kk))
                return r
            self.chat.completions.create = create

    class MeteredSync(OpenAI):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            orig = self.chat.completions.create
            def create(*aa, **kk):
                try:
                    r = orig(*aa, **kk)
                except Exception as e:
                    _fatal_if_exhausted(e); raise
                if getattr(r, "usage", None):
                    meter.add(r.usage, _has_image(kk))
                return r
            self.chat.completions.create = create

    openai.AsyncOpenAI, openai.OpenAI = MeteredAsync, MeteredSync
    return meter


def inject_parse(parsed: dict, artifacts: Path, dest: Path) -> tuple[str, int, int]:
    """Write our parse into <dest>/auto/ in MinerU's on-disk shape.

    Returns (markdown, n_images, n_chars). img_path is written RELATIVE to the auto folder
    because the vendor resolves it as os.path.join(folder_path, item['img_path']); the
    mapper emits absolute paths for RAG-Anything, which would silently miss here.
    """
    auto = dest / "auto"
    (auto / "images").mkdir(parents=True, exist_ok=True)
    md = mapper.elements_to_text(parsed)
    (auto / f"{dest.name}.md").write_text(md, encoding="utf-8")

    cl = mapper.elements_to_content_list(parsed, artifacts)      # same D1-D4 decisions as S2
    out, n_img = [], 0
    for it in cl:
        if it.get("type") == "image" and it.get("img_path"):
            src = Path(it["img_path"])
            if not src.exists():
                continue
            rel = f"images/{src.name}"
            shutil.copy(src, auto / rel)
            out.append({**it, "img_path": rel})
            n_img += 1
        else:
            out.append(it)
    (auto / f"{dest.name}_content_list.json").write_text(json.dumps(out, ensure_ascii=False))
    return md, n_img, len(md)


def alignment(chunks: list[str], injected_md: str) -> float:
    """Fraction of returned chunk characters traceable to the markdown we injected.

    This is the C-52 guard: it sits OUTSIDE the vendor call and asserts on content, so no
    handler inside the vendor can absorb it. PyMuPDF extraction of the same PDF produces
    materially different text and scores far below the floor.
    """
    ref = _norm(injected_md)
    idx = {hash(ref[k:k + SHINGLE]) for k in range(0, max(len(ref) - SHINGLE + 1, 0))}
    m = t = 0
    for c in chunks:
        cn = _norm(c)
        for k in range(0, max(len(cn) - SHINGLE + 1, 0), STRIDE):
            t += SHINGLE
            if hash(cn[k:k + SHINGLE]) in idx:
                m += SHINGLE
    return round(m / t, 4) if t else 0.0


def pick_docs(bench: str, n: int, seed: int = 42) -> list[dict]:
    """Stratify on IMAGES per document -- C-48: images drive cost in a multimodal arm, so a
    pilot stratified on characters measures the wrong axis."""
    import random
    base = Path(BENCHES[bench]["parsed"])
    rows = []
    for f in sorted(base.glob("*.json")):
        stem = f.name[:-5]
        stem = stem[:-4] if stem.lower().endswith(".pdf") else stem
        try:
            parsed = mapper.load_parse(f)
        except Exception:
            continue
        st = mapper.parse_stats(parsed, base / "artifacts" / stem)
        rows.append(dict(doc=f.name, stem=stem, chars=st["text_chars"],
                         images=st["content_list_image_items"]))
    rows.sort(key=lambda r: r["images"])
    picks, rng = [], random.Random(seed)
    # equal-count strata over the image distribution, one document sampled from each
    per = max(len(rows) // n, 1)
    for i in range(n):
        band = rows[i * per:(i + 1) * per] if i < n - 1 else rows[i * per:]
        if band:
            picks.append(rng.choice(band))
    return picks


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="mmlongbench", choices=list(BENCHES))
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--ceiling", type=float, default=15.0)   # pre-registered pilot ceiling
    ap.add_argument("--tag", default="pilot")
    a = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY", "")
    if not key.startswith("sk-"):
        raise SystemExit("FATAL: OPENAI_API_KEY absent or malformed -- refusing to run")

    meter = Meter()
    install_meter(meter)                       # MUST precede the vendor import
    os.chdir(VEND)                             # parameter.py resolves ./models relatively
    sys.path.insert(0, str(VEND))
    from src.builder import MMKGBuilder
    from src import parameter

    assert parameter.MODEL_NAME == "gpt-4o-mini" and parameter.MM_MODEL_NAME == "gpt-4o-mini", \
        f"harness identity: expected gpt-4o-mini, got {parameter.MODEL_NAME}/{parameter.MM_MODEL_NAME}"

    RUN_CONFIG = {
        "ARM": "s4_mmgraphrag", "TIER": 1, "BENCH": a.bench,
        "MMGRAPHRAG_COMMIT": "ec0065d2e026e42c797b42723b1fffd0347a82c4",
        "LLM_MODEL": parameter.MODEL_NAME, "VISION_MODEL": parameter.MM_MODEL_NAME,
        "EMB_MODEL": "all-MiniLM-L6-v2 (384-d, vendor default)",
        "INGEST": "our MinerU parse injected at <working>/<stem>/auto (no re-parse)",
        "PARSE_GUARD": f"content alignment vs injected markdown, floor {ALIGN_FLOOR} (C-52)",
        "PILOT_N": a.n, "STRATIFIED_ON": "images per document (C-48)",
        "CEILING_USD": a.ceiling, "RATES": RATES,
    }

    out = GB / "tier1" / "s4_mmgraphrag"
    (out / "stores").mkdir(parents=True, exist_ok=True)
    docs = pick_docs(a.bench, a.n)
    print(f"  S4 {a.tag}: {len(docs)} docs, stratified on images "
          f"[{docs[0]['images']}..{docs[-1]['images']}]", flush=True)

    log, t0 = [], time.time()
    for i, d in enumerate(docs, 1):
        stem, base = d["stem"], Path(BENCHES[a.bench]["parsed"])
        work = out / "stores" / f"{a.bench}__{stem}"
        if (work / "_INDEX_COMPLETE.json").exists():
            print(f"  [{i}/{len(docs)}] {stem[:34]:36s} resumed", flush=True)
            continue
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)

        parsed = mapper.load_parse(base / d["doc"])
        md, n_img, n_chars = inject_parse(parsed, base / "artifacts" / stem, work / stem)

        b_cost, b_calls, t1 = meter.cost, meter.calls, time.time()
        err = None
        try:
            parameter.WORKING_DIR = str(work)
            parameter.OUTPUT_DIR = str(work / "output")
            parameter.MMKG_NAME = stem
            builder = MMKGBuilder(working_dir=str(work))
            await builder.aindex(str(work / stem / f"{stem}.pdf"))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"

        # ---- C-52 guard: content assertion, outside the vendor call ----
        chunk_f = work / "kv_store_text_chunks.json"
        chunks = []
        if chunk_f.exists():
            chunks = [c.get("content", "") for c in json.load(open(chunk_f)).values()]
        al = alignment(chunks, md) if chunks else 0.0

        rec = dict(bench=a.bench, doc=d["doc"], stem=stem, chars=n_chars, images=n_img,
                   seconds=round(time.time() - t1, 1),
                   cost_usd=round(meter.cost - b_cost, 6),
                   calls=meter.calls - b_calls, chunks=len(chunks),
                   alignment=al, error=err)
        log.append(rec)
        # rule 22: emit the number every document, pass or fail -- a guard you can see run
        print(f"  [{i}/{len(docs)}] {stem[:34]:36s} img={n_img:>4d} "
              f"{rec['seconds']:>6.0f}s ${rec['cost_usd']:.4f} chunks={len(chunks):>4d} "
              f"align={al:.4f}" + (f"  ERROR {err[:50]}" if err else ""), flush=True)

        if err:
            (out / f"index_log_{a.tag}_{a.bench}.json").write_text(json.dumps(
                dict(RUN_CONFIG=RUN_CONFIG, docs=log, aborted="error"), indent=1))
            raise SystemExit(f"FATAL: {stem} failed -- {err}")
        if al < ALIGN_FLOOR:
            (out / f"index_log_{a.tag}_{a.bench}.json").write_text(json.dumps(
                dict(RUN_CONFIG=RUN_CONFIG, docs=log, aborted="alignment"), indent=1))
            raise SystemExit(
                f"FATAL: {stem} parse-provenance alignment {al:.4f} < {ALIGN_FLOOR}. "
                f"The arm did not ingest our parse -- most likely the vendor's silent "
                f"PyMuPDF fallback (C-52). Refusing to continue.")
        if meter.cost > a.ceiling:
            raise SystemExit(f"FATAL: ${meter.cost:.2f} exceeds pilot ceiling ${a.ceiling}")
        (work / "_INDEX_COMPLETE.json").write_text(json.dumps(rec))

    tot = meter.snapshot()
    als = [r["alignment"] for r in log]
    summary = dict(RUN_CONFIG=RUN_CONFIG, docs=log, totals=tot,
                   wall_seconds=round(time.time() - t0, 1),
                   alignment_min=min(als) if als else None,
                   alignment_mean=round(statistics.mean(als), 4) if als else None,
                   per_image_usd=round(tot["cost_usd"] / max(sum(r["images"] for r in log), 1), 6),
                   per_1m_chars_usd=round(tot["cost_usd"] / max(sum(r["chars"] for r in log), 1) * 1e6, 4))
    (out / f"index_log_{a.tag}_{a.bench}.json").write_text(json.dumps(summary, indent=1))
    print(f"\n  {a.tag} complete: {len(log)} docs, {summary['wall_seconds']/60:.1f} min, "
          f"${tot['cost_usd']:.4f}, {tot['calls']} calls ({tot['vision_calls']} with images)")
    print(f"  alignment min {summary['alignment_min']} mean {summary['alignment_mean']}")
    print(f"  unit rates: ${summary['per_image_usd']:.5f}/image  "
          f"${summary['per_1m_chars_usd']:.2f}/1M chars")


if __name__ == "__main__":
    asyncio.run(main())
