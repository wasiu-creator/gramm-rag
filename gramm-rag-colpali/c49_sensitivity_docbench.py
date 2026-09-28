#!/usr/bin/env python3
"""C-49 sensitivity, DocBench: quote-recall@5 with and without the image route.

Reuses score_colpali.py's canonical page_text reconstruction and the canonical
evidence_page_hit / _quote_tokens from src.evaluation.metrics -- no reimplementation.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, '/work/gramm-rag-docbench')
from src.evaluation.metrics import evidence_page_hit, _quote_tokens

D3 = Path('/work/gramm-rag-3benchmarks/data/docbench')
GBS = Path('/work/gramm-rag-graphbracket/tier1/s2_raganything')
base = json.loads((D3/'baseline_vector_docbench.json').read_text())
gramm = json.loads((D3/'gramm_docbench_s42.json').read_text())
qm = json.loads((D3/'qmeta_docbench.json').read_text())['rows']
ev, an = base['evidences'], base['is_answerable']
n = base['n_questions']
pdir = Path('/work/gramm-rag-docbench/cache/parsed')
cache = {}

def page_text(doc_id, pg):
    if doc_id not in cache:
        f = pdir / f'{doc_id}.json'
        d = json.loads(f.read_text()) if f.exists() else {'elements': []}
        byp = {}
        for e in d.get('elements', []):
            try:
                k = int(e.get('page_no', -1))
            except (TypeError, ValueError):
                continue
            byp.setdefault(k, []).append((int(e.get('reading_order', 0) or 0),
                                          e.get('text', '') or ''))
        cache[doc_id] = {k: ' '.join(t for _, t in sorted(v)) for k, v in byp.items()}
    return cache[doc_id].get(pg, '')

scoreable = [i for i in range(n) if _quote_tokens(ev[i]) and an[i]]
print(f"  canonical quote-recall denominator: n={len(scoreable)}")
# drift check on the archived arms, same rule as the scorer
for lab, res, exp in (('gramm', gramm, 0.4534), ('baseline', base, 0.3959)):
    h = sum(1 for i in scoreable if evidence_page_hit(res['page_texts'][i][:5], ev[i], 0.6))
    got = h/len(scoreable)
    assert abs(got-exp) < 5e-4, f'{lab} QRec drift {got} vs {exp}'
print("  drift check OK: archived quote-recall reproduced (0.3959 / 0.4534)")

rows = []
for lab, fn in (('shipped (text + image route)', 's2_retrieval_docbench.json'),
                ('text-only (route 2 disabled)', 's2_retrieval_docbench_textonly.json')):
    f = GBS/fn
    if not f.exists():
        print(f"  absent: {fn}"); continue
    d = json.loads(f.read_text()); cp = d['context_pages']
    pt = [[page_text(qm[i]['doc_id'], pg) for pg in cp[i]] for i in range(n)]
    qr = sum(1 for i in scoreable if evidence_page_hit(pt[i][:5], ev[i], 0.6))/len(scoreable)
    al = [r for r in d['alignment_rate'] if r is not None]
    rows.append((lab, qr, sum(al)/len(al), sum(1 for c in cp if not c), len(cp)))

print(f"\n{'DocBench variant':32s}{'qrec@5':>9s}{'align':>8s}{'no-page qs':>13s}")
for lab, q, a, z, N in rows:
    print(f"{lab:32s}{q:>9.4f}{a:>8.4f}{z:>8d}/{N}")
OUTJ = Path('/work/gramm-rag-colpali/scores')/'c49_sensitivity.json'
blob = json.loads(OUTJ.read_text()) if OUTJ.exists() else {}
blob['docbench'] = {
    'metric': 'quote-recall@5', 'denominator': len(scoreable),
    'shipped': rows[0][1] if rows else None,
    'text_only': rows[1][1] if len(rows) > 1 else None,
    'align_shipped': rows[0][2] if rows else None,
    'align_text_only': rows[1][2] if len(rows) > 1 else None,
}
OUTJ.write_text(json.dumps(blob, indent=1))
print(f"  wrote {OUTJ}")

if len(rows) == 2:
    print(f"\n  route contributes {rows[0][1]-rows[1][1]:+.4f} quote-recall@5")
    for nm, v in (('GraMM-RAG', 0.4534), ('LightRAG 1.5.7', 0.5068), ('ColPali', 0.5986)):
        f1, f2 = rows[0][1] > v, rows[1][1] > v
        print(f"    vs {nm:16s}{v:.4f}: shipped {'above' if f1 else 'below'}, "
              f"text-only {'above' if f2 else 'below'}"
              + ("   <-- RANKING DEPENDS ON THE ROUTE" if f1 != f2 else ""))
