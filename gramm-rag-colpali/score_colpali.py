#!/usr/bin/env python3
"""
score_colpali.py — T-C2 stage 6b: score the ColPali arm with the project's OWN metrics.

Every scorer here is imported from the benchmark repo, never reimplemented. Two
denominator rules in particular are easy to get subtly wrong and were each got wrong
once already this phase (C-8, C-9):
  * ctx-hit      -- unscoreable when the question is unanswerable OR has no gold pages
  * quote-recall -- scoreable only when `_quote_tokens(evidence)` is non-empty AND the
                    question is answerable
Both are taken from the canonical implementations, and the GraMM/baseline figures are
recomputed alongside so any drift shows up immediately.

DocBench judging uses gpt-4-turbo, matching the archived D-38 config, with the D-34
credit-exhaustion guard.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path('/work/gramm-rag-colpali')
GEN = ROOT / 'generation'
OUT = ROOT / 'scores'
JUDGE_MODEL = 'gpt-4-turbo'

REPO = {'mmlongbench': '/work/gramm-rag-mmlongbench',
        'docbench': '/work/gramm-rag-docbench'}
ARCH = {'mmlongbench': '/work/gramm-rag-3benchmarks/data/mmlongbench',
        'docbench': '/work/gramm-rag-3benchmarks/data/docbench'}
ARMS = {'mmlongbench': ('baseline_vector_mmlb.json', 'gramm_mmlongbench_s42.json'),
        'docbench': ('baseline_vector_docbench.json', 'gramm_docbench_s42.json')}

FATAL = ('no credits', 'insufficient_quota', 'insufficient quota',
         'exceeded your current quota', 'billing', 'payment required')


def ctx_hit_flags(context_pages, gold_pages, is_answerable, n):
    """Canonical rule, ported verbatim from gen_notebook.py::_ctx_hit."""
    out = []
    for i in range(n):
        golds = set(gold_pages[i]) if gold_pages and gold_pages[i] else set()
        if not golds or (is_answerable and not is_answerable[i]):
            out.append(None)
            continue
        out.append(bool(golds & set(context_pages[i])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bench', choices=list(REPO))
    ap.add_argument('--arm', default='colpali')
    a = ap.parse_args()
    sys.path.insert(0, REPO[a.bench])
    from src.evaluation.metrics import (compute_anls, compute_f1, compute_accuracy,
                                        evidence_page_hit, _quote_tokens)
    OUT.mkdir(parents=True, exist_ok=True)

    gen = json.loads((GEN / f'{a.arm}_gen_{a.bench}.json').read_text())
    arch = Path(ARCH[a.bench])
    base = json.loads((arch / ARMS[a.bench][0]).read_text())
    gramm = json.loads((arch / ARMS[a.bench][1]).read_text())
    preds = gen['predictions']
    golds = gen['golds']
    n = len(preds)
    assert n == base['n_questions'], f'{n} predictions vs {base["n_questions"]} questions'

    m = {'n_questions': n, 'n_empty': sum(1 for p in preds if not p),
         'n_failures': gen.get('n_failures', 0)}
    # These return plain floats, not dicts.
    m['anls'] = compute_anls(preds, golds)
    m['f1'] = compute_f1(preds, golds)
    m['accuracy'] = compute_accuracy(preds, golds)

    if a.bench == 'mmlongbench':
        hit = ctx_hit_flags(gen['context_pages'], base['gold_pages'],
                            base.get('is_answerable'), n)
        sc = [h for h in hit if h is not None]
        m['ctx_hit_rate'] = sum(sc) / len(sc)
        m['ctx_hit_n'] = len(sc)
        # GraMM / baseline recomputed on the SAME rule, as a drift check
        for lab, res in (('gramm', gramm), ('baseline', base)):
            h2 = ctx_hit_flags(res['context_pages'], base['gold_pages'],
                               base.get('is_answerable'), n)
            s2 = [x for x in h2 if x is not None]
            m[f'ctx_hit_rate_{lab}'] = sum(s2) / len(s2)
        assert abs(m['ctx_hit_rate_baseline'] - 0.2327) < 5e-4, 'baseline ctx-hit drift'
        assert abs(m['ctx_hit_rate_gramm'] - 0.4344) < 5e-4, 'GraMM ctx-hit drift'
        print('  drift check OK: archived ctx-hit reproduced (0.2327 / 0.4344)')
    else:
        ev, an = base['evidences'], base['is_answerable']
        # base['page_texts'] is ordered by the ARCHIVED arm's ranking, so it cannot be
        # reused here -- ColPali selected different pages. Rebuild page text from the
        # parsed document, in ColPali's own rank order.
        qm = json.loads(Path('/work/gramm-rag-3benchmarks/data/docbench/'
                             'qmeta_docbench.json').read_text())['rows']
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
                cache[doc_id] = {k: ' '.join(t for _, t in sorted(v))
                                 for k, v in byp.items()}
            return cache[doc_id].get(pg, '')

        colpali_pt = [[page_text(qm[i]['doc_id'], pg) for pg in gen['context_pages'][i]]
                      for i in range(n)]
        scoreable = [i for i in range(n) if _quote_tokens(ev[i]) and an[i]]
        for k in (1, 3, 5):
            hits = sum(1 for i in scoreable
                       if evidence_page_hit(colpali_pt[i][:k], ev[i], 0.6))
            m[f'quote_recall@{k}'] = hits / len(scoreable) if scoreable else 0.0
        m['quote_recall_n'] = len(scoreable)
        # Drift check on the archived arms, same canonical rule.
        for lab, res in (('gramm', gramm), ('baseline', base)):
            h = sum(1 for i in scoreable
                    if evidence_page_hit(res['page_texts'][i][:5], ev[i], 0.6))
            m[f'quote_recall@5_{lab}'] = h / len(scoreable)
        assert abs(m['quote_recall@5_baseline'] - 0.3959) < 5e-4, 'baseline QRec drift'
        assert abs(m['quote_recall@5_gramm'] - 0.4534) < 5e-4, 'GraMM QRec drift'
        print('  drift check OK: archived quote-recall@5 reproduced (0.3959 / 0.4534)')

        # ---- GPT-4 judge, matching the archived D-38 config ----
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            sys.exit('FATAL: OPENAI_API_KEY not set (judge required for DocBench).')
        from openai import OpenAI
        from src.evaluation.metrics import judge_answer_gpt4
        from concurrent.futures import ThreadPoolExecutor
        client = OpenAI(api_key=key)
        qs = base['questions']

        def one(i):
            return i, judge_answer_gpt4(qs[i], preds[i], golds[i], ev[i] or '',
                                        client=client, model=JUDGE_MODEL)
        scores = [0] * n
        fails = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for f in [ex.submit(one, i) for i in range(n)]:
                try:
                    i, s = f.result()
                except Exception as e:
                    if any(x in str(e).lower() for x in FATAL):
                        raise RuntimeError(f'JUDGE CREDIT EXHAUSTION (D-34): {e}')
                    fails.append(str(e)[:150]); continue
                scores[i] = s
        m['judge_accuracy'] = sum(scores) / n
        m['judge_scores'] = scores
        m['judge_n'] = n
        m['judge_failures'] = len(fails)
        assert not fails, f'{len(fails)} judge failures: {fails[:2]}'
        ty = base['types']
        m['per_type'] = {t: sum(scores[i] for i in range(n) if ty[i] == t) /
                            max(1, sum(1 for i in range(n) if ty[i] == t))
                         for t in sorted(set(ty))}

    (OUT / f'{a.arm}_scores_{a.bench}.json').write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))


if __name__ == '__main__':
    main()
