#!/usr/bin/env python3
"""In-flight validation of the S5 projection, immune to processing order.

The full run indexes documents ascending by size, so a cumulative per-character rate taken
early is computed over the corpus's SMALLEST documents, where per-document fixed costs
dominate -- it overstates, and comparing it to the projection is not like-for-like. This
instead compares ACTUAL cost and time for the documents already completed against what the
pilot's fitted model PREDICTS for those same documents, which no processing order can bias.
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import project_s5 as P

GB = Path(__file__).resolve().parent
OUT = GB / "tier1" / "s5_graphrag"

pilot = json.loads((OUT / "index_log_pilot.json").read_text())["docs"]
p = P.fit(pilot)
if p is None:
    raise SystemExit("pilot fit degenerate")
corpus = [d["ingested_chars"] for d in json.loads((OUT / "corpus_chars.json").read_text())["docs"]]

cost_of = lambda c: (p["a_ext"] + p["a_emb"] + p["a_una"]) * c + p["r_rep"] * p["k"] * c ** p["beta"]
secs_of = lambda c: p["s"] * c ** p["gamma"]

txt = Path(GB / "logs" / "s5_full.log").read_text(errors="ignore")
rows = [(int(c.replace(",", "")), int(s), float(d))
        for line in txt.splitlines() if "/364]" in line and "(resumed)" not in line
        for c, s, d in re.findall(r"([\d,]+)ch\s+(\d+)s\s+\$([\d.]+)", line)]
if not rows:
    print("  no freshly-indexed documents completed yet"); raise SystemExit

ac, as_, pc, ps = sum(r[2] for r in rows), sum(r[1] for r in rows), 0.0, 0.0
for c, s, d in rows:
    pc += cost_of(c); ps += secs_of(c)
ch = sum(r[0] for r in rows)
print(f"  completed (fresh): {len(rows)} docs, {ch:,} chars")
print(f"  cost   actual ${ac:.3f}  predicted ${pc:.3f}   ratio {ac/pc:.3f}")
print(f"  time   actual {as_/60:.0f} min  predicted {ps/60:.0f} min   ratio {as_/ps:.3f}")
print(f"\n  full-corpus projection ${sum(cost_of(c) for c in corpus):.2f}, "
      f"{sum(secs_of(c) for c in corpus)/3600:.1f} h")
print(f"  same, rescaled by the observed cost ratio: ${sum(cost_of(c) for c in corpus)*ac/pc:.2f}")
print(f"  (a naive cumulative per-char rate here would read "
      f"${ac/ch*sum(corpus):.2f} -- biased upward by size-ascending order)")
