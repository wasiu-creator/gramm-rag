#!/usr/bin/env python3
"""C-49 sensitivity: how much of the S2 arm's localisation came from the image route?

Recomputes ctx-hit from context_pages alone, under the canonical denominator
(answerable AND has gold pages, n=838), for the shipped S2 retrieval and for a
text-only rerun with route 2 disabled. Everything else is byte-identical: same
stores, same mode, same k, same seed.
"""
import json
from pathlib import Path
GB = Path('/work/gramm-rag-graphbracket/tier1/s2_raganything') if Path('/work').exists() \
     else Path('/media/volume/vol3/grammrag/gramm-rag-graphbracket/tier1/s2_raganything')
D3 = Path('/work/gramm-rag-3benchmarks/data/mmlongbench') if Path('/work').exists() \
     else Path('/media/volume/vol3/grammrag/gramm-rag-3benchmarks/data/mmlongbench')

base = json.loads((D3/'baseline_vector_mmlb.json').read_text())
an = base['is_answerable']
gold = base['gold_pages']          # canonical source, same field the scorer reads
idx = [i for i in range(len(an)) if an[i] and gold[i]]
print(f"  canonical ctx-hit denominator: n={len(idx)} (answerable AND gold pages)")

def ctx_hit(cp):
    return sum(1 for i in idx if any(g in cp[i] for g in gold[i])) / len(idx)

res = {}
for lab, f in (('S2 shipped (text + image route)', 's2_retrieval_mmlongbench.json'),
               ('S2 text-only (route 2 disabled)', 's2_retrieval_mmlongbench_textonly.json')):
    p = GB/f
    if not p.exists():
        print(f"  MISSING: {f}"); continue
    d = json.loads(p.read_text())
    al = [r for r in d['alignment_rate'] if r is not None]
    res[lab] = (ctx_hit(d['context_pages']), sum(al)/len(al))
    print(f"  {lab:34s} ctx-hit {res[lab][0]:.4f}   alignment {res[lab][1]:.4f}")

OUTJ = (Path('/work/gramm-rag-colpali/scores') if Path('/work').exists()
        else Path('/media/volume/vol3/grammrag/gramm-rag-colpali/scores'))/'c49_sensitivity.json'
blob = json.loads(OUTJ.read_text()) if OUTJ.exists() else {}
blob['mmlongbench'] = {
    'metric': 'ctx-hit', 'denominator': len(idx),
    'shipped': res.get('S2 shipped (text + image route)', (None, None))[0],
    'text_only': res.get('S2 text-only (route 2 disabled)', (None, None))[0],
    'align_shipped': res.get('S2 shipped (text + image route)', (None, None))[1],
    'align_text_only': res.get('S2 text-only (route 2 disabled)', (None, None))[1],
}
OUTJ.write_text(json.dumps(blob, indent=1))
print(f"  wrote {OUTJ}")

if len(res) == 2:
    (a, aa), (b, ba) = res.values()
    print(f"\n  the C-49 fix moved ctx-hit by {a-b:+.4f} ({b:.4f} -> {a:.4f})")
    print(f"  and alignment by {aa-ba:+.4f} ({ba:.4f} -> {aa:.4f})")
    print("\n  RANKING CHECK against the arms this sits between:")
    for nm, v in (('GraMM-RAG', 0.4344), ('LightRAG 1.5.7', 0.2816), ('ColPali', 0.7995)):
        print(f"    vs {nm:16s} {v:.4f}:  shipped {'ABOVE' if a>v else 'below':5s} | "
              f"text-only {'ABOVE' if b>v else 'below'}"
              + ("   <-- FIX FLIPPED THIS RANKING" if (a > v) != (b > v) else ""))
