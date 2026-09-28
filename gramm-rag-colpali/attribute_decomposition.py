#!/usr/bin/env python3
"""
attribute_decomposition.py — does ANY reweighting of the retrieval-side attributes
(C/S/T) detect answerability?

Motivation: U's ceiling is 0.560 on MMLongBench against the generator's achieved 0.806 on
identical context. The open objection is that this measures one weighted sum, not the
attribute family. Marginal per-attribute AUC answers "were our weights wrong"; the decisive
test is a cross-validated JOINT logistic regression on (C,S,T) -- "does any reweighting
work". If CV'd joint AUC stays near 0.56, the attribute-family claim is closed.

INTEGRITY GATE (blocking, runs first): recomputed U must reproduce the archived
`reward_scores` per question. A re-run that does not reproduce is a different system and its
decomposition measures nothing. After C-22 -- a checkpoint reporting missing_keys == [] while
holding random weights -- behavioural equality is the only acceptable evidence that the
configuration was restored.
"""
import argparse, json, sys, time
import numpy as np
import torch
from pathlib import Path

BENCH = {
 'mmlongbench': dict(repo='/work/gramm-rag-mmlongbench',
   arch='/work/gramm-rag-3benchmarks/data/mmlongbench',
   gramm='gramm_mmlongbench_s42.json', base='baseline_vector_mmlb.json',
   qmeta='qmeta_mmlb.json', hgt='hgt_mmlb', reward='reward_mmlb.json',
   src='/work/gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json'),
 'docbench': dict(repo='/work/gramm-rag-docbench',
   arch='/work/gramm-rag-3benchmarks/data/docbench',
   gramm='gramm_docbench_s42.json', base='baseline_vector_docbench.json',
   qmeta='qmeta_docbench.json', hgt='hgt_docbench', reward='reward_docbench.json',
   src=None),
}
OUT = Path('/work/gramm-rag-colpali/scores')
ACHIEVED_GEN = {'mmlongbench': 0.806, 'docbench': 0.750}   # GraMM generator, same context
CEILING_U = {'mmlongbench': 0.560, 'docbench': 0.686}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bench', choices=list(BENCH))
    ap.add_argument('--limit', type=int, default=None, help='gate-test on a sample first')
    a = ap.parse_args()
    s = BENCH[a.bench]
    sys.path.insert(0, s['repo'])
    from src.retrieval.self_healing import SelfHealingRetriever, RewardFunction
    from src.retrieval.hgt_model import DocumentHGT
    import torch_geometric  # noqa

    arch = Path(s['arch'])
    g = json.loads((arch / s['gramm']).read_text())
    b = json.loads((arch / s['base']).read_text())
    qm = json.loads((arch / s['qmeta']).read_text())['rows']
    archived = np.asarray(g['reward_scores'], float)
    if a.bench == 'mmlongbench':
        questions = [x['question'] for x in json.loads(Path(s['src']).read_text())]
    else:
        questions = list(b['questions'])
    docs = [r['doc_id'] for r in qm]
    n = len(questions)

    # --- components: E5 query embedding, spaCy entities, graphs, HGT ---
    import spacy
    nlp = spacy.load('en_core_web_lg')
    from sentence_transformers import SentenceTransformer
    print('loading E5-Mistral-7B (4-bit)...', flush=True)
    from transformers import BitsAndBytesConfig
    st = SentenceTransformer('intfloat/e5-mistral-7b-instruct', device='cuda',
                             model_kwargs={'quantization_config': BitsAndBytesConfig(
                                 load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)})
    cache = Path(s['repo']) / 'cache'
    rfn = RewardFunction(params_path=str(cache / 'results/models' / s['reward']))
    hgt_dir = cache / 'results/models' / s['hgt']
    # DocumentHGT needs the graph METADATA (node/edge types) before weights can load;
    # take it from the first available graph, as the notebook does.
    _g0 = None
    for r in qm:
        d0 = r['doc_id']
        st0 = d0[:-4] if d0.lower().endswith('.pdf') else d0
        f0 = cache / 'graphs' / f'{st0}.pt'
        if f0.exists():
            _g0 = torch.load(f0, map_location='cuda', weights_only=False); break
    if _g0 is None:
        sys.exit('no graphs found')
    hgt = DocumentHGT(metadata=_g0.metadata()).to('cuda')
    hgt.load_state_dict(torch.load(hgt_dir / 'best_model.pt', map_location='cuda',
                                   weights_only=False))
    hgt.eval()
    retr = SelfHealingRetriever(hgt_model=hgt, reward_fn=rfn, top_k=10,
                                expansion_k=5, max_rounds=2)

    idx = list(range(n))[:a.limit] if a.limit else list(range(n))
    rows, gate = [], []
    t0 = time.time()
    for c, i in enumerate(idx, 1):
        doc = docs[i]
        stem = doc[:-4] if doc.lower().endswith('.pdf') else doc
        gp = cache / 'graphs' / f'{stem}.pt'
        pp = None
        for cand in (f'{stem}.pdf.json', f'{stem}.json'):
            if (cache / 'parsed' / cand).exists():
                pp = cache / 'parsed' / cand; break
        if not gp.exists() or pp is None:
            rows.append((np.nan,)*4); gate.append(np.nan); continue
        graph = torch.load(gp, map_location='cuda', weights_only=False)
        parsed = json.loads(pp.read_text())
        q = f'Represent this question for retrieving relevant document passages: {questions[i]}'
        qv = torch.from_numpy(np.asarray(st.encode([q], normalize_embeddings=True)[0],
                                         dtype=np.float32))
        ents = [e.text for e in nlp(questions[i] or '').ents]
        try:
            res = retr.retrieve(qv, graph, parsed, ents)
        except Exception as e:
            print(f'  q{i} retrieve failed: {type(e).__name__}: {str(e)[:90]}', flush=True)
            rows.append((np.nan,)*4); gate.append(np.nan); continue
        rd = res.get('reward_detail') or {}
        rows.append((rd.get('coverage', np.nan), rd.get('similarity', np.nan),
                     rd.get('temporal', np.nan), res.get('reward', np.nan)))
        gate.append(res.get('reward', np.nan))
        if c % 25 == 0:
            print(f'  [{c}/{len(idx)}] {(time.time()-t0)/60:.1f} min', flush=True)

    arr = np.array(rows, float)
    rec = np.array(gate, float)
    ok = ~np.isnan(rec)
    diff = np.abs(rec[ok] - archived[np.array(idx)][ok])
    print(f'\n=== INTEGRITY GATE ({a.bench}, n={ok.sum()}) ===')
    print(f'  max |recomputed - archived| = {diff.max():.3e}   mean = {diff.mean():.3e}')
    print(f'  within 1e-6: {np.mean(diff < 1e-6)*100:.1f}%   within 1e-3: {np.mean(diff < 1e-3)*100:.1f}%')
    # Bit-exactness is unattainable here: E5-Mistral runs 4-bit via bitsandbytes, which is
    # not reproducible to the ULP across runs. The gate is therefore BEHAVIOURAL, which is
    # what the review actually asked for -- the recomputed signal must behave identically,
    # not hash identically. Both conditions must hold:
    #   (a) max|diff| < 1e-3   -- residual is quantisation noise, not configuration drift
    #   (b) the recomputed U reproduces the archived U's AUC to <0.005
    # Loosening (a) alone would be exactly the anti-pattern this project guards against;
    # (b) is the one that certifies the signal is the same object.
    _un_all = ~np.asarray(b['is_answerable'], bool)[np.array(idx)][ok]
    auc_new = auc(rec[ok], ~_un_all)
    auc_old = auc(archived[np.array(idx)][ok], ~_un_all)
    # A handful of questions sit on a self-healing expansion boundary: a quantisation-level
    # embedding difference flips an expand/stop decision, producing a different node set and
    # so a visibly different reward. Those are genuine non-reproductions, not noise, and are
    # EXCLUDED from the decomposition rather than absorbed by widening the tolerance.
    frac_ok = float(np.mean(diff < 1e-3))
    passed = bool(frac_ok >= 0.99 and abs(auc_new - auc_old) < 5e-3)
    n_excl = int((diff >= 1e-3).sum())
    print(f'  AUC recomputed {auc_new:.4f} vs archived {auc_old:.4f} '
          f'(delta {abs(auc_new-auc_old):.4f})')
    print(f'  non-reproducing (>=1e-3): {n_excl} of {ok.sum()} '
          f'-- excluded from the decomposition, not tolerated')
    print(f'  GATE: {"PASS" if passed else "FAIL"}  '
          f'[>=99% within 1e-3 AND |dAUC|<5e-3, boundary flips excluded]')
    res_gate = {'auc_recomputed': float(auc_new), 'auc_archived': float(auc_old)}
    res = {'bench': a.bench, 'n': int(ok.sum()), 'gate_max_abs_diff': float(diff.max()),
           'gate_mean_abs_diff': float(diff.mean()), 'gate_pass': passed,
           'limit': a.limit, 'frac_within_1e-3': frac_ok,
           'n_excluded_boundary': n_excl, **res_gate}
    if not passed:
        print('  Configuration not restored -- decomposition NOT computed (it would '
              'measure a different system).')
        (OUT / f'attr_decomp_{a.bench}{"_gate" if a.limit else ""}.json').write_text(json.dumps(res, indent=1))
        return

    unans = ~np.asarray(b['is_answerable'], bool)[np.array(idx)]
    keep = np.zeros(len(rec), bool); keep[np.where(ok)[0][diff < 1e-3]] = True
    unans, C, S, T = unans[keep], arr[keep, 0], arr[keep, 1], arr[keep, 2]
    rec_k = rec[keep]
    print(f'  decomposition computed on {keep.sum()} reproducing questions')
    # Invariant: AUC(x) + AUC(-x) == 1. Violated by any tie-mishandling implementation.
    for _nm, _v in (('C', C), ('S', S), ('T', T)):
        _s = auc(_v, ~unans) + auc(-_v, ~unans)
        assert abs(_s - 1.0) < 1e-9, f'AUC helper broken on {_nm}: AUC(x)+AUC(-x)={_s:.4f}'
    print('  AUC invariant OK: AUC(x)+AUC(-x)=1 for all three attributes')
    print(f'\n  marginal AUC (high value -> answerable)')
    for nm, v in (('coverage C', C), ('similarity S', S), ('temporal T', T),
                  ('U (weighted)', rec_k)):
        print(f'    {nm:16s} AUC {auc(v, ~unans):.3f}   unique={len(np.unique(v))}')
        res[f'auc_{nm.split()[0]}'] = float(auc(v, ~unans))
    # joint: does ANY reweighting work? 5-fold CV logistic regression
    # Persist per-question components so the joint fit can be diagnosed and refitted
    # WITHOUT re-running retrieval. Not saving these the first time made a fitting bug
    # un-diagnosable offline.
    np.savez(OUT / f'attr_components_{a.bench}.npz',
             C=C, S=S, T=T, U=rec_k, unans=unans)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X = np.column_stack([C, S, T]); y = (~unans).astype(int)
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    ja = auc(oof, ~unans)
    res['joint_cv_auc'] = float(ja)
    print(f'\n  JOINT 5-fold CV logistic on (C,S,T): AUC {ja:.3f}')
    print(f'  contrast: ceiling_U {CEILING_U[a.bench]:.3f} | '
          f'achieved_gen {ACHIEVED_GEN[a.bench]:.3f} (same context)')
    (OUT / f'attr_decomp_{a.bench}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
