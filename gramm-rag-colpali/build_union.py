#!/usr/bin/env python3
"""
build_union.py — B'-2: union arm (ColPali u GraMM) + a size-matched ColPali control.

Cowork asked for the union arm to test whether the graph adds anything on top of a
strong retriever. Run alone it is not interpretable: the union averages 5.22 pages on
MMLongBench against 3 for each single arm, so a gain could simply be more context.

Two arms are therefore emitted:
  union        ColPali top-3 in rank order, then GraMM's pages not already present.
               ColPali-first because it is the stronger retriever and the 12k-char
               budget truncates from the end -- this makes the test "does adding the
               graph's pages on top of ColPali help?"
  colpali_matched   ColPali top-N, where N is exactly the union size for that same
               question. Identical context budget, different composition. This is the
               control that separates "more pages" from "complementary pages".
"""
import json
from pathlib import Path

RET = Path('/work/gramm-rag-colpali/retrieval')
ARCH = Path('/work/gramm-rag-3benchmarks/data')
GRAMM = {'mmlongbench': 'mmlongbench/gramm_mmlongbench_s42.json',
         'docbench': 'docbench/gramm_docbench_s42.json'}

for bench, gf in GRAMM.items():
    cp = json.loads((RET / f'colpali_retrieval_{bench}.json').read_text())
    gm = json.loads((ARCH / gf).read_text())
    ranked, ctx = cp['retrieved_pages'], cp['context_pages']
    gctx = gm['context_pages']
    n = cp['n_questions']
    union, matched = [], []
    for i in range(n):
        u = list(ctx[i])
        for p in gctx[i]:
            if p not in u:
                u.append(p)
        union.append(u)
        matched.append(ranked[i][:len(u)])      # same budget, ColPali only
    for arm, pages in (('union', union), ('colpali_matched', matched)):
        cfg = dict(cp['RUN_CONFIG'])
        cfg.update({'ARM': arm, 'RETRIEVER': 'colpali+gramm' if arm == 'union' else 'colpali',
                    'CONTEXT_POLICY': ('colpali_topk_then_gramm_union' if arm == 'union'
                                       else 'colpali_topN_size_matched_to_union'),
                    'TOPK_CONTEXT': 'variable'})
        out = {'system': ('Union (ColPali u GraMM)' if arm == 'union'
                          else 'ColPali size-matched'),
               'benchmark': bench, 'n_questions': n, 'RUN_CONFIG': cfg,
               'retrieved_pages': ranked, 'context_pages': pages}
        (RET / f'{arm}_retrieval_{bench}.json').write_text(json.dumps(out))
    avg = sum(len(x) for x in union) / n
    print(f'{bench}: union mean {avg:.2f} pages (matched control identical by construction)')
