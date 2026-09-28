#!/usr/bin/env python3
"""
retrieve.py — ColQwen2 MaxSim retrieval + the Gate C validity assertion (T-C2 stage 4).

Retrieval is scoped WITHIN each question's own document, because that is what every
other arm does: each MMLongBench/DocBench question names one document. Corpus-wide
retrieval would change the task, not just the retriever, and would break the control.

Emits, per question:
  * `retrieved_pages` -- every page of the document, ranked by MaxSim
  * `context_pages`   -- the top-k that would be handed to the generator

Context budget: k=3, matching GraMM's UNION context (2 cross-encoder pages + 1 graph
page). k=2 (the flat-vector budget) is reported alongside so the comparison can be read
against either control.

**Gate C2 validity assertion (blocking).** ColPali's APPA / quote-recall MUST differ
from GraMM's. Identical values mean the adapter is not actually routing ColPali's
retrieval through, which is a bug and not a result -- and it is the failure that would
most easily masquerade as a finding.
"""
import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/work/gramm-rag-colpali')
IDX = ROOT / 'index'
OUT = ROOT / 'retrieval'
MODEL_ID = 'vidore/colqwen2-v1.0'
TOPK = 3
BATCH_Q = 8

BENCH = {
    'mmlongbench': dict(
        qmeta='/work/gramm-rag-3benchmarks/data/mmlongbench/qmeta_mmlb.json',
        src='/work/gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json',
        gramm='/work/gramm-rag-3benchmarks/data/mmlongbench/gramm_mmlongbench_s42.json',
        base='/work/gramm-rag-3benchmarks/data/mmlongbench/baseline_vector_mmlb.json',
    ),
    'docbench': dict(
        qmeta='/work/gramm-rag-3benchmarks/data/docbench/qmeta_docbench.json',
        src=None,
        gramm='/work/gramm-rag-3benchmarks/data/docbench/gramm_docbench_s42.json',
        base='/work/gramm-rag-3benchmarks/data/docbench/baseline_vector_docbench.json',
    ),
}


def load_model():
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    model, info = ColQwen2.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda:0',
        output_loading_info=True)
    missing = [k for k in info.get('missing_keys', []) if 'lora' not in k.lower()]
    if missing:
        raise RuntimeError(f'{MODEL_ID}: {len(missing)} params newly initialised. '
                           f'Refusing to proceed (C-22).')
    return model.eval(), ColQwen2Processor.from_pretrained(MODEL_ID)


@torch.no_grad()
def embed_queries(model, proc, queries):
    out = []
    for i in range(0, len(queries), BATCH_Q):
        b = proc.process_queries(queries[i:i+BATCH_Q]).to('cuda:0')
        e = model(**b)
        out += [e[j].float().cpu() for j in range(e.shape[0])]
    return out


def load_doc(bench, stem):
    f = IDX / bench / f'{stem}.npz'
    if not f.exists():
        return None, None
    z = np.load(f)
    pages = [int(x) for x in z['pages']]
    embs = [torch.from_numpy(z[f'p{p}'].astype(np.float32)) for p in pages]
    return pages, embs


def questions_for(bench, spec):
    """Question text + doc_id + gold pages, aligned to the archived result JSONs."""
    qm = json.loads(Path(spec['qmeta']).read_text())['rows']
    base = json.loads(Path(spec['base']).read_text())
    if bench == 'mmlongbench':
        src = json.loads(Path(spec['src']).read_text())
        qs = [s['question'] for s in src]
        gold = [ast.literal_eval(s['evidence_pages']) for s in src]
    else:
        qs = list(base['questions'])
        gold = [None] * len(qs)          # DocBench has no page-level gold
    docs = [r['doc_id'] for r in qm]
    return qs, docs, gold, base


def run(bench, model, proc):
    spec = BENCH[bench]
    qs, docs, gold, base = questions_for(bench, spec)
    n = len(qs)
    OUT.mkdir(parents=True, exist_ok=True)

    by_doc = {}
    for i, d in enumerate(docs):
        by_doc.setdefault(d, []).append(i)

    ranked = [None] * n
    t0 = time.time()
    missing_docs = []
    for di, (doc, idxs) in enumerate(sorted(by_doc.items()), 1):
        stem = doc[:-4] if doc.lower().endswith('.pdf') else doc
        pages, embs = load_doc(bench, stem)
        if pages is None:
            missing_docs.append(doc)
            continue
        qe = embed_queries(model, proc, [qs[i] for i in idxs])
        scores = proc.score_multi_vector(qe, embs)
        for row, i in enumerate(idxs):
            order = torch.argsort(scores[row], descending=True).tolist()
            ranked[i] = [pages[j] for j in order]
        if di % 20 == 0 or di == len(by_doc):
            el = time.time() - t0
            print(f'  [{di}/{len(by_doc)} docs] {el/60:.1f} min elapsed', flush=True)
    if missing_docs:
        raise RuntimeError(f'{bench}: {len(missing_docs)} documents have no index '
                           f'({missing_docs[:3]}). Embedding is incomplete.')
    assert all(r is not None for r in ranked), 'some questions were not ranked'

    res = {
        'system': 'ColPali (ColQwen2-v1.0)', 'benchmark': bench, 'n_questions': n,
        'RUN_CONFIG': {
            'STUDY': 'retriever-iso', 'ARM': 'colpali', 'RETRIEVER': 'colpali',
            'COLPALI_MODEL': MODEL_ID, 'COLPALI_MAX_SIDE': 1024,
            'TOPK_CONTEXT': TOPK, 'SEED': 42,
            'STACK': 'colpali-engine==0.3.4, transformers==4.46.3',
        },
        'retrieved_pages': ranked,
        'context_pages': [r[:TOPK] for r in ranked],
        'context_pages_k2': [r[:2] for r in ranked],
    }
    (OUT / f'colpali_retrieval_{bench}.json').write_text(json.dumps(res))
    print(f'{bench}: ranked {n} questions over {len(by_doc)} docs '
          f'in {(time.time()-t0)/60:.1f} min', flush=True)
    return res, gold, base


def appa(ranked, gold, is_answerable, ks=(1, 5, 10)):
    """Answer-page prediction accuracy: is a gold page within the top-k ranking?"""
    out = {}
    scoreable = [i for i, g in enumerate(gold)
                 if g and (is_answerable is None or is_answerable[i])]
    for k in ks:
        hit = sum(1 for i in scoreable if any(g in ranked[i][:k] for g in gold[i]))
        out[f'appa@{k}'] = hit / len(scoreable) if scoreable else 0.0
    out['n_appa_scoreable'] = len(scoreable)
    return out


def gate_c2(bench, mine, spec):
    """BLOCKING: competitor retrieval metrics must DIFFER from GraMM's."""
    gm = json.loads(Path(spec['gramm']).read_text())['metrics']
    same = []
    for k, v in mine.items():
        if k.startswith('appa@') and k in gm and abs(gm[k] - v) < 1e-9:
            same.append(f'{k}: both {v:.6f}')
    if same:
        raise RuntimeError(
            f'GATE C2 FAILED for {bench}: ColPali retrieval metrics are IDENTICAL to '
            f"GraMM's ({'; '.join(same)}). The adapter is not routing ColPali's "
            f'retrieval through. This is a bug, not a result.')
    print(f'  Gate C2 OK ({bench}): retrieval metrics differ from GraMM', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('benchmarks', nargs='*', default=['mmlongbench', 'docbench'])
    a = ap.parse_args()
    assert torch.cuda.is_available(), 'GPU not reclaimed'
    model, proc = load_model()
    summary = {}
    for b in a.benchmarks:
        res, gold, base = run(b, model, proc)
        if b == 'mmlongbench':
            m = appa(res['retrieved_pages'], gold, base.get('is_answerable'))
            gm = json.loads(Path(BENCH[b]['gramm']).read_text())['metrics']
            print(f"  ColPali APPA@1/@5/@10: {m['appa@1']:.4f} / {m['appa@5']:.4f} / "
                  f"{m['appa@10']:.4f}   (n={m['n_appa_scoreable']})", flush=True)
            print(f"  GraMM   APPA@1/@5/@10: {gm['appa@1']:.4f} / {gm['appa@5']:.4f} / "
                  f"{gm['appa@10']:.4f}", flush=True)
            gate_c2(b, m, BENCH[b])
            summary[b] = {'colpali': m, 'gramm': {k: gm[k] for k in
                                                  ('appa@1', 'appa@5', 'appa@10')}}
        else:
            summary[b] = {'note': 'DocBench has no page-level gold; quote-recall is '
                                  'computed in the generation stage from page_texts'}
    (OUT / 'retrieval_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
