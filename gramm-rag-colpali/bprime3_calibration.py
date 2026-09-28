#!/usr/bin/env python3
"""
bprime3_calibration.py — B'-3 calibration study on the Multi-attribute Utility Function.

Object under study is U itself (GraMM's `reward_scores`, R = aC + bS + lT), NOT the ColQwen2
MaxSim instrument used for cross-system selective risk. That instrument answered a different
question and its verdict stands; this one asks whether U can drive abstention at all.

Reported against CEILINGS, never against an unbounded target:
  * ceiling_U   = best achievable balanced accuracy from ANY single threshold on U
  * achieved_gen = the generator's own decline decision on identical context
A policy that reaches ceiling_U has hit the limit of the signal, not failed.

Per benchmark, never pooled: the two behave differently and a pooled number would average a
live effect against a null one.

Per-stratum taus are fitted with 5-fold CV. Fitting a threshold per stratum on the same data
it is evaluated on manufactures a gain out of overfitting; the held-out column is the honest
one.
"""
import json, re, sys
import numpy as np
from pathlib import Path

ARCH = Path('/work/gramm-rag-3benchmarks/data')
GEN = Path('/work/gramm-rag-colpali/generation')
OUT = Path('/work/gramm-rag-colpali/scores')
SEED = 42
DEC = re.compile(
    r"not\s*answerable|unanswerable|insufficient"
    r"|cannot\s+(?:be\s+answered|provide|determine|answer)"
    r"|unable\s+to|no\s+(?:answer|information|mention)"
    r"|(?:do|does|did)\s+not\s+(?:include|contain|mention|provide|specify|have|appear|reference)"
    r"|(?:is|are)\s+not\s+(?:stated|provided|mentioned|specified|available|found|included|contained)"
    r"|not\s+(?:stated|provided|mentioned|specified|available|found)", re.I)


def balacc(pred_abstain, unans):
    """Balanced accuracy of 'abstain' as a detector of 'is unanswerable'."""
    tp = np.sum(pred_abstain & unans); fn = np.sum(~pred_abstain & unans)
    tn = np.sum(~pred_abstain & ~unans); fp = np.sum(pred_abstain & ~unans)
    if (tp + fn) == 0 or (tn + fp) == 0:
        return float('nan')
    return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def auc(score, pos):
    """Mann-Whitney AUC with MID-RANK averaging for ties.

    The earlier version assigned distinct ranks via argsort with no tie averaging. Because
    positives are concatenated first and the sort is stable, every tie was resolved AGAINST
    the positive class, biasing AUC downward in proportion to how tied the variable is. A
    continuous feature was unaffected; a 3-valued one was wrecked (T read 0.231 when it is
    0.484). Invariant that catches it: AUC(x) + AUC(-x) must equal 1.
    """
    from scipy.stats import rankdata
    a, b = score[pos], score[~pos]
    if len(a) == 0 or len(b) == 0:
        return float('nan')
    allv = np.concatenate([a, b])
    lab = np.concatenate([np.ones(len(a)), np.zeros(len(b))])
    r = rankdata(allv)
    return (r[lab == 1].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def best_tau(u, unans, grid=None):
    grid = grid if grid is not None else np.unique(np.quantile(u, np.linspace(0, 1, 200)))
    best = (-1, None)
    for t in grid:
        s = balacc(u < t, unans)
        if not np.isnan(s) and s > best[0]:
            best = (s, float(t))
    return best


def setup(bench):
    if bench == 'mmlongbench':
        g = json.loads((ARCH / 'mmlongbench/gramm_mmlongbench_s42.json').read_text())
        b = json.loads((ARCH / 'mmlongbench/baseline_vector_mmlb.json').read_text())
        qm = json.loads((ARCH / 'mmlongbench/qmeta_mmlb.json').read_text())['rows']
        strata = {'doc_type': [r['doc_type'] for r in qm],
                  'len_bin': [('1-20' if r['doc_pages'] <= 20 else '21-60'
                               if r['doc_pages'] <= 60 else '61+') for r in qm]}
        preds = g['predictions']
    else:
        g = json.loads((ARCH / 'docbench/gramm_docbench_s42.json').read_text())
        b = json.loads((ARCH / 'docbench/baseline_vector_docbench.json').read_text())
        qm = json.loads((ARCH / 'docbench/qmeta_docbench.json').read_text())['rows']
        strata = {'type': list(b['types']),
                  'len_bin': [('1-20' if r['doc_pages'] <= 20 else '21-60'
                               if r['doc_pages'] <= 60 else '61+') for r in qm]}
        preds = g['predictions']
    return (np.asarray(g['reward_scores'], float),
            ~np.asarray(b['is_answerable'], bool), strata, preds)


def per_stratum_cv(u, unans, labels, k=5):
    """5-fold CV: fit a tau per stratum on train, evaluate on held-out."""
    rng = np.random.default_rng(SEED)
    n = len(u); fold = rng.permutation(n) % k
    pred = np.zeros(n, bool)
    labels = np.asarray(labels)
    for f in range(k):
        tr, te = fold != f, fold == f
        gt, _ = best_tau(u[tr], unans[tr])          # global fallback for thin strata
        gtau = _
        for s in np.unique(labels):
            m_tr = tr & (labels == s); m_te = te & (labels == s)
            if m_te.sum() == 0:
                continue
            # need both classes present to fit a threshold
            if m_tr.sum() >= 30 and unans[m_tr].sum() >= 5 and (~unans[m_tr]).sum() >= 5:
                _, t = best_tau(u[m_tr], unans[m_tr])
            else:
                t = gtau
            pred[m_te] = u[m_te] < t
    return balacc(pred, unans)


def global_cv(u, unans, k=5):
    rng = np.random.default_rng(SEED)
    n = len(u); fold = rng.permutation(n) % k
    pred = np.zeros(n, bool)
    for f in range(k):
        tr, te = fold != f, fold == f
        _, t = best_tau(u[tr], unans[tr])
        pred[te] = u[te] < t
    return balacc(pred, unans)


if __name__ == '__main__':
    rep = {}
    for bench in sys.argv[1:] or ['mmlongbench', 'docbench']:
        u, unans, strata, preds = setup(bench)
        gen_dec = np.array([bool(DEC.search(str(p))) for p in preds])
        a = auc(u, ~unans)                      # high U -> answerable
        ceil_u, tau_u = best_tau(u, unans)
        ceil_gen = balacc(gen_dec, unans)
        print(f'\n=== {bench} — calibrating the utility function U ===')
        print(f'  n={len(u)}  unanswerable={int(unans.sum())}  U unique={len(np.unique(u))}'
              f'  sd={u.std():.4f}')
        print(f'  AUC of U as answerability signal      {a:.3f}')
        print(f'  CEILING  best single-tau on U          {ceil_u:.3f}  (tau={tau_u:.4f})')
        print(f'  ACHIEVED  generator own decision        {ceil_gen:.3f}   <-- same context')
        print(f'  shipped  tau=0.0, refusals=0           {balacc(u < 0.0, unans):.3f}')
        g_cv = global_cv(u, unans)
        print(f'\n  {"policy":34s}{"balanced acc":>14s}{"vs ceiling_U":>14s}')
        print(f'  {"global tau (5-fold CV)":34s}{g_cv:>14.3f}{g_cv-ceil_u:>+14.3f}')
        rep[bench] = {'auc_U': float(a), 'ceiling_U': float(ceil_u), 'tau_U': tau_u,
                      'achieved_generator': float(ceil_gen),
                      'shipped_tau0': float(balacc(u < 0.0, unans)),
                      'global_tau_cv': float(g_cv), 'per_stratum_cv': {}}
        for name, lab in strata.items():
            ps = per_stratum_cv(u, unans, lab)
            print(f'  {"per-stratum tau by "+name+" (CV)":34s}{ps:>14.3f}{ps-ceil_u:>+14.3f}')
            rep[bench]['per_stratum_cv'][name] = float(ps)
        print(f'  {"":34s}{"":>14s}   ceiling_U={ceil_u:.3f}  achieved_gen={ceil_gen:.3f}')
    (OUT / 'bprime3_calibration.json').write_text(json.dumps(rep, indent=1))
    print('\nPer-attribute (C/S/T) AUC not computed: only the scalar R is archived, the '
          'components are not. Recovering them requires re-running GraMM retrieval.')
