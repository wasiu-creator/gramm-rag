#!/usr/bin/env python3
"""
stratified_report.py — every metric by answerable / unanswerable / pooled, all systems.

Cowork's standing instruction: "Pooled-only numbers are no longer acceptable for this
project." This reports all five systems on both benchmarks, stratified.

One caveat is measured and printed rather than left implicit. On unanswerable questions
the gold answer *is* "not answerable", so scoring well there and declining to answer are
the same event -- verified below at ~100% agreement. The unanswerable-accuracy row and
the abstention-rate row are therefore ONE measurement in two units, not two independent
lines of evidence. The effect is real; the corroboration is not.
"""
import json, re, sys, statistics
import numpy as np
from pathlib import Path

sys.path.insert(0, '/work/gramm-rag-mmlongbench')
from src.evaluation.metrics import anls_score

ARCH = Path('/work/gramm-rag-3benchmarks/data')
GEN = Path('/work/gramm-rag-colpali/generation')
OUT = Path('/work/gramm-rag-colpali/scores')
BOOT, SEED = 2000, 42
# v2 matcher (Cowork, 2026-08-24). v1 missed "does not include/contain" -- the dominant
# decline phrasing on DocBench -- which understated its decline rate 17x (0.040 vs 0.702)
# and produced a spurious "DocBench measures something different" reading. Never narrow
# this without re-running the confound check below.
DEC = re.compile(
    r"not\s*answerable|unanswerable|insufficient"
    r"|cannot\s+(?:be\s+answered|provide|determine|answer)"
    r"|unable\s+to|no\s+(?:answer|information|mention)"
    r"|(?:do|does|did)\s+not\s+(?:include|contain|mention|provide|specify|have|appear|reference)"
    r"|(?:is|are)\s+not\s+(?:stated|provided|mentioned|specified|available|found|included|contained)"
    r"|not\s+(?:stated|provided|mentioned|specified|available|found)", re.I)

BANNER = (
    "REPORTING RULE: on both benchmarks the gold answer to an unanswerable question is a\n"
    "  statement of non-availability, so 'accuracy on the unanswerable stratum' and\n"
    "  'abstention rate' are ONE measurement (agreement 0.92-1.00 below). Never quote the\n"
    "  unanswerable column as evidence independent of the decline column.")


def ci(d, n=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    b = np.asarray(d)[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.mean(d)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def load(bench):
    if bench == 'mmlongbench':
        b = json.loads((ARCH / 'mmlongbench/baseline_vector_mmlb.json').read_text())
        g = json.loads((ARCH / 'mmlongbench/gramm_mmlongbench_s42.json').read_text())
        gold, an = b['golds'], b['is_answerable']
        sysd = {'Flat-vector': b['predictions'], 'GraMM-RAG': g['predictions']}
        for arm, lab in (('colpali', 'ColPali'), ('union', 'Union(CP∪GraMM)'),
                         ('colpali_matched', 'ColPali size-matched'),
                         ('lightrag', 'LightRAG 1.5.7'),         # WP-1 graph bracket
                         ('s2', 'RAG-Anything 1.2.8'),          # S2 multimodal graph
                         ('s5', 'MS GraphRAG 2.7.2'),            # S5 community-text graph
                         ('flatk3', 'Flat-vector (k=3)')):       # C-77(ii) page-matched control
            f = GEN / f'{arm}_gen_{bench}.json'
            if f.exists():
                sysd[lab] = json.loads(f.read_text())['predictions']
        score = lambda P: [anls_score(p, t) for p, t in zip(P, gold)]
        return sysd, an, score, 'ANLS'
    b = json.loads((ARCH / 'docbench/baseline_vector_docbench.json').read_text())
    g = json.loads((ARCH / 'docbench/gramm_docbench_s42.json').read_text())
    an = b['is_answerable']
    sc = {'Flat-vector': b['judge']['judge_scores'], 'GraMM-RAG': g['judge']['judge_scores']}
    pr = {'Flat-vector': b['predictions'], 'GraMM-RAG': g['predictions']}
    for arm, lab in (('colpali', 'ColPali'), ('union', 'Union(CP∪GraMM)'),
                     ('colpali_matched', 'ColPali size-matched'),
                     ('lightrag', 'LightRAG 1.5.7'),            # WP-1 graph bracket
                     ('s2', 'RAG-Anything 1.2.8'),             # S2 multimodal graph
                     ('s5', 'MS GraphRAG 2.7.2'),               # S5 community-text graph
                     ('flatk3', 'Flat-vector (k=3)')):          # C-77(ii) page-matched control
        s = OUT / f'{arm}_scores_docbench.json'
        gg = GEN / f'{arm}_gen_docbench.json'
        if s.exists() and gg.exists():
            sc[lab] = json.loads(s.read_text())['judge_scores']
            pr[lab] = json.loads(gg.read_text())['predictions']
    return pr, an, (lambda P, _sc=sc: None), 'Judge accuracy', sc


def report(bench):
    r = load(bench)
    if bench == 'mmlongbench':
        preds, an, score, label = r
        scores = {k: score(v) for k, v in preds.items()}
    else:
        preds, an, _, label, scores = r
    A = [i for i, x in enumerate(an) if x]
    U = [i for i, x in enumerate(an) if not x]
    print(f'\n{"="*78}\n{bench}  ({label})   answerable n={len(A)}  unanswerable n={len(U)}\n{"="*78}')
    print(BANNER)
    print(f'\n{"system":22s}{"pooled":>9s}{"answerable":>12s}{"unanswerable":>14s}'
          f'{"dec|unans":>11s}{"dec|ans":>9s}{"balAcc":>8s}')
    rows = {}
    for k, s in scores.items():
        p_, a_, u_ = (statistics.mean(s), statistics.mean(s[i] for i in A),
                      statistics.mean(s[i] for i in U))
        d = statistics.mean(bool(DEC.search(str(preds[k][i]))) for i in U)
        da = statistics.mean(bool(DEC.search(str(preds[k][i]))) for i in A)
        # abstention treated as a classifier of 'is unanswerable' -- a uniform-threshold
        # shift shows up as high decline on BOTH strata, not as discrimination.
        tpr = d
        tnr = 1 - da
        bal = 0.5 * (tpr + tnr)
        rows[k] = (p_, a_, u_, d, da, bal)
        print(f'{k:22s}{p_:>9.4f}{a_:>12.4f}{u_:>14.4f}{d:>11.3f}{da:>9.3f}{bal:>8.3f}')

    # measurement-overlap check
    print('\n  overlap check (is the abstention row a second measurement?)')
    for k, s in scores.items():
        ag = statistics.mean((s[i] >= 0.5) == bool(DEC.search(str(preds[k][i]))) for i in U)
        print(f'    {k:22s} accuracy>=0.5 agrees with decline-regex on {ag*100:5.1f}% of unanswerable')

    base = 'GraMM-RAG'
    print(f'\n  paired contrasts vs {base} (2000-resample bootstrap)')
    print(f'    {"contrast":34s}{"answerable":>26s}{"unanswerable":>26s}')
    for k in scores:
        if k == base:
            continue
        out = []
        for idx in (A, U):
            o, lo, hi = ci([scores[k][i] - scores[base][i] for i in idx])
            star = '*' if (lo > 0 or hi < 0) else ' '
            out.append(f'{o:+.4f} [{lo:+.4f},{hi:+.4f}]{star}')
        print(f'    {k+" - "+base:34s}{out[0]:>26s}{out[1]:>26s}')
    return rows


if __name__ == '__main__':
    allr = {}
    for b in sys.argv[1:] or ['mmlongbench', 'docbench']:
        allr[b] = report(b)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'stratified_report.json').write_text(json.dumps(
        {b: {k: dict(zip(('pooled', 'answerable', 'unanswerable', 'decline_unans',
                          'decline_ans', 'abstention_balanced_acc'), v))
             for k, v in r.items()} for b, r in allr.items()}, indent=1))
    print('\n* = 95% CI excludes zero')
