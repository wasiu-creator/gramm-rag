#!/usr/bin/env python3
"""D1 -- P3 subsample, drawn from corpus metadata ONLY, before any indexing.

Pre-registered in P3_PREREGISTRATION.md §3:
  ~1,000 questions from 5,187; stratified on document length in PAGES in equal-count
  octiles over documents; seed 42; drawn WHOLE-DOCUMENT so no arm ever retrieves against
  a partially indexed corpus; written to disk before indexing starts.

Also verifies that every sampled question's page images exist on disk. MP-DocVQA ships
page images natively (64,057 JPGs), which removes the rendering pass -- but "the directory
has 64k files" is not "this subsample is covered", so coverage is MEASURED per sampled
document rather than inferred from the total.
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MP = ROOT / "gramm-rag-mpdocvqa"
IMDB = MP / "data_volume/imdbs/imdb_val.npy"
IMGDIR = MP / "data_volume/images/images"
PARSED = MP / "cache/parsed"
OUT = Path(__file__).resolve().parent / "tier1" / "p3_mpdocvqa"
SEED, TARGET_Q, NOCT = 42, 1000, 8

def main():
    imdb = np.load(str(IMDB), allow_pickle=True)
    recs = list(imdb[1:])
    print(f"imdb_val: {len(recs)} questions")

    by_doc = defaultdict(list)
    for i, r in enumerate(recs):
        by_doc[str(r["image_id"])].append(i)
    print(f"documents in val split: {len(by_doc)}")

    parsed = {p.stem for p in PARSED.glob("*.json")}
    docs = sorted(d for d in by_doc if d in parsed)
    print(f"documents with a MinerU parse: {len(docs)} "
          f"({len(by_doc)-len(docs)} unparsed, excluded and reported)")
    assert docs, "no parsed documents -- refusing to draw a subsample"

    pages = {d: int(recs[by_doc[d][0]]["total_doc_pages"]) for d in docs}
    ordered = sorted(docs, key=lambda d: (pages[d], d))          # deterministic tie-break
    octiles = [ordered[i::NOCT] for i in range(NOCT)] if False else \
              [ordered[round(i*len(ordered)/NOCT):round((i+1)*len(ordered)/NOCT)]
               for i in range(NOCT)]

    rng = random.Random(SEED)
    picked, nq = [], 0
    pools = []
    for o in octiles:
        p = list(o); rng.shuffle(p); pools.append(p)
    # round-robin across octiles so the draw stays balanced if it stops early
    while nq < TARGET_Q and any(pools):
        progressed = False
        for oi, p in enumerate(pools):
            if not p or nq >= TARGET_Q:
                continue
            d = p.pop()
            picked.append((d, oi)); nq += len(by_doc[d]); progressed = True
        if not progressed:
            break

    picked.sort()
    qidx = [i for d, _ in picked for i in by_doc[d]]
    qidx.sort()
    oct_of = {d: oi for d, oi in picked}

    # page-image coverage, measured per sampled document
    missing_img, checked = [], 0
    for d, _ in picked:
        for i in by_doc[d]:
            for nm in recs[i]["image_name"]:
                checked += 1
                if not (IMGDIR / f"{nm}.jpg").exists():
                    missing_img.append(nm)
    chars = 0
    for d, _ in picked:
        try:
            pj = json.loads((PARSED / f"{d}.json").read_text())
            chars += sum(len(str(e.get("text", ""))) for e in pj.get("elements", []))
        except Exception as e:
            print(f"  WARN: cannot read parse for {d}: {e}")

    rows = []
    for i in qidx:
        r = recs[i]
        rows.append(dict(q_index=int(i), question_id=int(r["question_id"]),
                         doc_id=str(r["image_id"]), question=str(r["question"]),
                         answers=[str(a) for a in r["answers"]],
                         answer_page_idx=(None if r["answer_page_idx"] is None
                                          else int(r["answer_page_idx"])),
                         total_doc_pages=int(r["total_doc_pages"]),
                         octile=oct_of[str(r["image_id"])]))

    per_oct = defaultdict(lambda: [0, 0])
    for d, oi in picked:
        per_oct[oi][0] += 1; per_oct[oi][1] += len(by_doc[d])

    out = dict(
        RUN_CONFIG=dict(SEED=SEED, TARGET_QUESTIONS=TARGET_Q, OCTILES=NOCT,
                        STRATUM="total_doc_pages, equal-count octiles over documents",
                        DRAW="whole-document, round-robin across octiles",
                        SOURCE=str(IMDB.relative_to(ROOT)),
                        PARSED_ONLY=True, BENCHMARK="mpdocvqa"),
        n_documents=len(picked), n_questions=len(rows),
        total_chars=chars,
        page_image_refs_checked=checked, page_images_missing=len(missing_img),
        missing_examples=missing_img[:10],
        documents=[d for d, _ in picked],
        per_octile={str(k): dict(docs=v[0], questions=v[1]) for k, v in sorted(per_oct.items())},
        page_range={str(k): [min(pages[d] for d, o in picked if o == k),
                             max(pages[d] for d, o in picked if o == k)]
                    for k in sorted(per_oct)},
        questions=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subsample.json").write_text(json.dumps(out, indent=1))

    print(f"\nsubsample: {len(picked)} documents, {len(rows)} questions, {chars:,} chars")
    print(f"page-image refs checked {checked}, missing {len(missing_img)}")
    for k in sorted(per_oct):
        lo, hi = out["page_range"][str(k)]
        print(f"  octile {k}: {per_oct[k][0]:>3} docs {per_oct[k][1]:>4} q   pages {lo}-{hi}")
    if missing_img:
        print(f"  WARNING: {len(missing_img)} referenced page images absent, e.g. {missing_img[:3]}")

if __name__ == "__main__":
    main()
