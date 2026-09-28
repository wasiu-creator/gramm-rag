#!/usr/bin/env python3
"""D1 -- corpus_sizes rows for the P3 subsample, plus the stratified projection set.

Written to its OWN file. data/corpus_sizes.json is a canonical input three shipped indexers
read; appending to it would change what a re-run of S2 (which does not filter by bench)
would index. Additive file + additive --corpus flag instead.

The projection set is drawn ACROSS OCTILES, not as "the first 50 by size". The indexers sort
ascending on chars, so the first 50 are the smallest documents in the corpus -- projecting
from them is C-59 exactly (an extrapolation biased by the order the sample was taken in,
which already cost this project once at $35/38 h vs $28/7.1 h).
"""
import json
from pathlib import Path

GB = Path(__file__).resolve().parent
ROOT = GB.parent
PARSED = ROOT / "gramm-rag-mpdocvqa/cache/parsed"
SUB = GB / "tier1/p3_mpdocvqa/subsample.json"
N_PROJ = 50

sub = json.loads(SUB.read_text())
oct_of, pages_of = {}, {}
for q in sub["questions"]:
    oct_of[q["doc_id"]] = q["octile"]
    pages_of[q["doc_id"]] = q["total_doc_pages"]

rows = []
for d in sub["documents"]:
    pj = json.loads((PARSED / f"{d}.json").read_text())
    chars = sum(len(str(e.get("text", ""))) for e in pj.get("elements", []))
    rows.append(dict(bench="mpdocvqa", doc=f"{d}.json", chars=chars,
                     pages=pages_of[d], octile=oct_of[d]))
rows.sort(key=lambda r: r["chars"])
(GB / "data/corpus_sizes_p3.json").write_text(json.dumps(rows, indent=1))

# projection set: round-robin across octiles, largest-first within each, so the 50 span the
# size range instead of hugging its floor
by_oct = {}
for r in rows:
    by_oct.setdefault(r["octile"], []).append(r)
for v in by_oct.values():
    v.sort(key=lambda r: -r["chars"])
proj, i = [], 0
while len(proj) < N_PROJ and any(by_oct.values()):
    for k in sorted(by_oct):
        if by_oct[k] and len(proj) < N_PROJ:
            proj.append(by_oct[k].pop())
proj.sort(key=lambda r: r["chars"])
(GB / "data/corpus_sizes_p3_proj50.json").write_text(json.dumps(proj, indent=1))

tc, pc = sum(r["chars"] for r in rows), sum(r["chars"] for r in proj)
print(f"P3 corpus: {len(rows)} docs, {tc:,} chars")
print(f"projection set: {len(proj)} docs, {pc:,} chars ({pc/tc:.1%} of corpus)")
print(f"  proj chars  min {min(r['chars'] for r in proj):,}  max {max(r['chars'] for r in proj):,}")
print(f"  corpus chars min {min(r['chars'] for r in rows):,}  max {max(r['chars'] for r in rows):,}")
first50 = rows[:50]
print(f"  CONTRAST -- 'first 50 by size' would cover {sum(r['chars'] for r in first50):,} chars "
      f"({sum(r['chars'] for r in first50)/tc:.1%}), max {max(r['chars'] for r in first50):,}")
