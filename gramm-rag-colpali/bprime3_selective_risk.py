#!/usr/bin/env python3
"""
bprime3_selective_risk.py — B'-3 per PHASE_B_GO_NO_GO_REVIEW.md 6.4.

Question: does stronger retrieval surface better *distractors* on unanswerable questions,
and does that defeat threshold-based abstention?

The systems store no comparable confidence signal (only GraMM has `reward_scores`; Flat and
ColPali have none). A sweep therefore needs a **uniform instrument**: ColQwen2 MaxSim between
the question and whatever pages that system put in context. Applied identically to all three,
it measures "how relevant does this system's context look", which is exactly the distractor
quantity 6.4 asks for. Page embeddings already exist in the index, so this costs $0 API.

Using ColPali's scorer as the instrument does not privilege ColPali: it scores each system's
own page choices with one fixed yardstick.

PRIMARY (not gameable): selective risk on the ANSWERABLE stratum at matched coverage.
Abstain when confidence < tau; sweep tau; a policy that games unanswerable accuracy by
abstaining more pays for it in coverage here.

SECONDARY: distribution of that confidence on unanswerable vs answerable items, per system.
If a stronger retriever's unanswerable contexts look as relevant as its answerable ones, the
threshold cannot separate them -- the distractor hypothesis.
"""
import json, sys, statistics
import numpy as np
import torch
from pathlib import Path

ROOT = Path('/work/gramm-rag-colpali')
IDX, GEN, OUT = ROOT / 'index', ROOT / 'generation', ROOT / 'scores'
ARCH = Path('/work/gramm-rag-3benchmarks/data')
MODEL_ID = 'vidore/colqwen2-v1.0'
sys.path.insert(0, '/work/gramm-rag-mmlongbench')
from src.evaluation.metrics import anls_score


def load_model():
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    m, info = ColQwen2.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                       device_map='cuda:0', output_loading_info=True)
    miss = [k for k in info.get('missing_keys', []) if 'lora' not in k.lower()]
    if miss:
        raise RuntimeError(f'{len(miss)} params newly initialised — refusing (C-22)')
    return m.eval(), ColQwen2Processor.from_pretrained(MODEL_ID)


def setup(bench):
    if bench == 'mmlongbench':
        b = json.loads((ARCH / 'mmlongbench/baseline_vector_mmlb.json').read_text())
        g = json.loads((ARCH / 'mmlongbench/gramm_mmlongbench_s42.json').read_text())
        src = json.loads(Path('/work/gramm-rag-mmlongbench/data/mmlongbench/'
                              'mmlongbench_doc.json').read_text())
        qs = [s['question'] for s in src]
        gold = b['golds']
        sc = {'Flat': [anls_score(p, t) for p, t in zip(b['predictions'], gold)],
              'GraMM': [anls_score(p, t) for p, t in zip(g['predictions'], gold)]}
        cp = json.loads((GEN / 'colpali_gen_mmlongbench.json').read_text())
        sc['ColPali'] = [anls_score(p, t) for p, t in zip(cp['predictions'], gold)]
        ctx = {'Flat': b['context_pages'], 'GraMM': g['context_pages'],
               'ColPali': cp['context_pages']}
        qm = json.loads((ARCH / 'mmlongbench/qmeta_mmlb.json').read_text())['rows']
    else:
        b = json.loads((ARCH / 'docbench/baseline_vector_docbench.json').read_text())
        g = json.loads((ARCH / 'docbench/gramm_docbench_s42.json').read_text())
        qs = list(b['questions'])
        sc = {'Flat': b['judge']['judge_scores'], 'GraMM': g['judge']['judge_scores'],
              'ColPali': json.loads((OUT / 'colpali_scores_docbench.json').read_text())['judge_scores']}
        cp = json.loads((GEN / 'colpali_gen_docbench.json').read_text())
        ctx = {'Flat': b['context_pages'], 'GraMM': g['context_pages'],
               'ColPali': cp['context_pages']}
        qm = json.loads((ARCH / 'docbench/qmeta_docbench.json').read_text())['rows']
    return qs, sc, ctx, b['is_answerable'], [r['doc_id'] for r in qm]


@torch.no_grad()
def confidence(bench, qs, ctx, docs, model, proc):
    """MaxSim(question, best context page) per system — one instrument, all systems."""
    conf = {k: [None] * len(qs) for k in ctx}
    by_doc = {}
    for i, d in enumerate(docs):
        by_doc.setdefault(d, []).append(i)
    for di, (doc, idxs) in enumerate(sorted(by_doc.items()), 1):
        stem = doc[:-4] if doc.lower().endswith('.pdf') else doc
        f = IDX / bench / f'{stem}.npz'
        if not f.exists():
            continue
        z = np.load(f)
        emb = {int(p): torch.from_numpy(z[f'p{int(p)}'].astype(np.float32)) for p in z['pages']}
        qe = []
        for j in range(0, len(idxs), 8):
            bt = proc.process_queries([qs[i] for i in idxs[j:j+8]]).to('cuda:0')
            e = model(**bt)
            qe += [e[t].float().cpu() for t in range(e.shape[0])]
        for row, i in enumerate(idxs):
            for k in ctx:
                pages = [p for p in ctx[k][i] if p in emb]
                if not pages:
                    conf[k][i] = float('nan'); continue
                s = proc.score_multi_vector([qe[row]], [emb[p] for p in pages])[0]
                conf[k][i] = float(s.max())
        if di % 50 == 0:
            print(f'  [{di}/{len(by_doc)} docs]', flush=True)
    return conf


def sweep(scores, conf, ansI, grid=41):
    """Risk-coverage on the ANSWERABLE stratum: abstain when conf < tau."""
    c = np.array([conf[i] for i in ansI], float)
    s = np.array([scores[i] for i in ansI], float)
    ok = ~np.isnan(c)
    c, s = c[ok], s[ok]
    out = []
    for tau in np.quantile(c, np.linspace(0, 0.95, grid)):
        keep = c >= tau
        if keep.sum() < 20:
            continue
        out.append((float(keep.mean()), float(1.0 - s[keep].mean())))   # coverage, risk
    return out


def risk_at(curve, target):
    """Selective risk at matched coverage, linear interp."""
    cs = sorted(curve)
    for (c0, r0), (c1, r1) in zip(cs, cs[1:]):
        if c0 <= target <= c1:
            w = 0 if c1 == c0 else (target - c0) / (c1 - c0)
            return r0 + w * (r1 - r0)
    return float('nan')


if __name__ == '__main__':
    model, proc = load_model()
    report = {}
    for bench in sys.argv[1:] or ['mmlongbench', 'docbench']:
        qs, sc, ctx, an, docs = setup(bench)
        print(f'\n=== {bench} — computing uniform confidence ===', flush=True)
        conf = confidence(bench, qs, ctx, docs, model, proc)
        ansI = [i for i, x in enumerate(an) if x]
        unI = [i for i, x in enumerate(an) if not x]
        print(f'\n{bench}: answerable n={len(ansI)}  unanswerable n={len(unI)}')
        print(f'{"system":9s}{"conf|ans":>10s}{"conf|unans":>12s}{"separation":>12s}'
              f'{"AUC":>7s}')
        rep = {}
        for k in ('Flat', 'GraMM', 'ColPali'):
            a = np.array([conf[k][i] for i in ansI], float)
            u = np.array([conf[k][i] for i in unI], float)
            a, u = a[~np.isnan(a)], u[~np.isnan(u)]
            # AUC that confidence separates answerable from unanswerable
            # mid-rank averaging: argsort ranks resolve ties against the positive class
            from scipy.stats import rankdata
            allv = np.concatenate([a, u]); lab = np.concatenate([np.ones(len(a)), np.zeros(len(u))])
            ranks = rankdata(allv)
            auc = (ranks[lab == 1].sum() - len(a)*(len(a)+1)/2) / (len(a)*len(u))
            rep[k] = {'conf_ans': float(a.mean()), 'conf_unans': float(u.mean()),
                      'separation': float(a.mean()-u.mean()), 'auc': float(auc)}
            print(f'{k:9s}{a.mean():>10.3f}{u.mean():>12.3f}{a.mean()-u.mean():>12.3f}{auc:>7.3f}')
        print(f'\n  selective risk on ANSWERABLE at matched coverage')
        print(f'  {"system":9s}{"cov=1.0":>10s}{"cov=0.8":>10s}{"cov=0.6":>10s}{"cov=0.4":>10s}')
        for k in ('Flat', 'GraMM', 'ColPali'):
            cur = sweep(sc[k], conf[k], ansI)
            rep[k]['curve'] = cur
            vals = [risk_at(cur, t) for t in (1.0, 0.8, 0.6, 0.4)]
            rep[k]['risk_at'] = dict(zip(('1.0', '0.8', '0.6', '0.4'), vals))
            print(f'  {k:9s}' + ''.join(f'{v:>10.3f}' for v in vals))
        report[bench] = rep
    (OUT / 'bprime3_selective_risk.json').write_text(json.dumps(report, indent=1))
    print('\nlower selective risk = better. AUC 0.5 = confidence cannot tell '
          'answerable from unanswerable.')
