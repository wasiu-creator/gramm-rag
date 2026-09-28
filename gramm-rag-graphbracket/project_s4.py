#!/usr/bin/env python3
"""S4 full-corpus cost projection, under the model pre-registered before the pilot.

The constrained model is TWO TERMS ONLY:  cost = a*chars + b*images,  a,b >= 0.
A vision-cost-per-character coefficient is FORBIDDEN BY CONSTRUCTION, not rejected after
the fact -- that single phantom term is what made S2's NNLS and pooled estimators
overestimate by 36-44% while the constrained model came in at +8.4% (C-43 made structural).

Reports three projections and applies the pre-registered NOT-IDENTIFIED rule.
"""
import json, sys
from pathlib import Path

GB = Path(__file__).resolve().parent
CORPUS_CHARS, CORPUS_IMAGES = 45_876_082, 14_581      # both benchmarks, 364 docs

def nnls2(rows):
    """Non-negative least squares on exactly two terms, by grid + refinement.
    Two parameters, so a coarse-to-fine search is exact enough and has no solver dependency."""
    import itertools
    C = [r["chars"] for r in rows]; I = [r["images"] for r in rows]; Y = [r["cost_usd"] for r in rows]
    def sse(a, b):
        return sum((a*c + b*i - y)**2 for c, i, y in zip(C, I, Y))
    a_hi = max((y/c if c else 0) for c, y in zip(C, Y)) * 1.5 or 1e-6
    b_hi = max((y/i if i else 0) for i, y in zip(I, Y)) * 1.5 or 1e-3
    best, lo_a, hi_a, lo_b, hi_b = None, 0.0, a_hi, 0.0, b_hi
    for _ in range(6):
        step_a, step_b = (hi_a-lo_a)/40 or 1e-12, (hi_b-lo_b)/40 or 1e-9
        cand = min(((sse(lo_a+i*step_a, lo_b+j*step_b), lo_a+i*step_a, lo_b+j*step_b)
                    for i, j in itertools.product(range(41), repeat=2)), key=lambda t: t[0])
        best = cand
        lo_a, hi_a = max(0.0, cand[1]-step_a), cand[1]+step_a
        lo_b, hi_b = max(0.0, cand[2]-step_b), cand[2]+step_b
    return best[1], best[2]

def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    bench = sys.argv[2] if len(sys.argv) > 2 else "mmlongbench"
    f = GB / "tier1" / "s4_mmgraphrag" / f"index_log_{tag}_{bench}.json"
    d = json.loads(f.read_text())
    rows = [r for r in d["docs"] if not r.get("error")]
    tot_c = sum(r["chars"] for r in rows); tot_i = sum(r["images"] for r in rows)
    tot_y = sum(r["cost_usd"] for r in rows)
    print(f"  pilot: {len(rows)} docs, {tot_c:,} chars, {tot_i:,} images, ${tot_y:.4f}")
    print(f"  images per doc: {sorted(r['images'] for r in rows)}")

    a, b = nnls2(rows)
    con = a*CORPUS_CHARS + b*CORPUS_IMAGES
    print(f"\n  CONSTRAINED (2-term, no cross term -- the pre-registered model)")
    print(f"    text  {a*1e6:.4f} $/1M chars   vision {b:.6f} $/image")
    print(f"    -> corpus ${con:.2f}   (text ${a*CORPUS_CHARS:.2f} + vision ${b*CORPUS_IMAGES:.2f})")
    print(f"    implied vision share {b*CORPUS_IMAGES/con:.3f}")

    per_img = tot_y/tot_i if tot_i else float("nan")
    per_chr = tot_y/tot_c if tot_c else float("nan")
    p_img, p_chr = per_img*CORPUS_IMAGES, per_chr*CORPUS_CHARS
    print(f"\n  SINGLE-DENOMINATOR projections (reported to test identifiability, not to be used)")
    print(f"    on images only: ${p_img:.2f}   (${per_img:.5f}/image)")
    print(f"    on chars  only: ${p_chr:.2f}   (${per_chr*1e6:.2f}/1M chars)")
    ratio = max(p_img, p_chr)/min(p_img, p_chr)
    print(f"    disagreement {ratio:.2f}x")

    print(f"\n  === PRE-REGISTERED IDENTIFIABILITY RULE (>2x -> NOT IDENTIFIED) ===")
    if ratio > 2.0:
        print(f"  NOT IDENTIFIED: the two single-denominator projections disagree by {ratio:.2f}x.")
        print(f"  Report the constrained estimate ${con:.2f} as the projection and the split as")
        print(f"  unresolved -- exactly the S2 disposition, applied in advance rather than after.")
    else:
        print(f"  IDENTIFIED: {ratio:.2f}x <= 2x. Constrained projection ${con:.2f} stands, with")
        print(f"  the single-denominator figures bracketing it (${min(p_img,p_chr):.2f}-${max(p_img,p_chr):.2f}).")

    print(f"\n  vs RAG-Anything (S2 measured): $100.49 total, $0.00386/image")
    print(f"  S4 is {per_img/0.00386:.2f}x S2 per image")
    print(f"\n  Gate H S4 estimate was $128-213 (central ~$157): "
          f"constrained ${con:.2f} lands "
          f"{'INSIDE' if 128 <= con <= 213 else 'OUTSIDE'} that band")

if __name__ == "__main__":
    main()
