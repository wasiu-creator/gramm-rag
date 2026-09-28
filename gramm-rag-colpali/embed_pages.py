#!/usr/bin/env python3
"""
embed_pages.py — ColQwen2.5 multi-vector embeddings for every corpus page (T-C2).

Model: vidore/colqwen2-v1.0 on the PINNED stack (colpali-engine 0.3.4,
transformers 4.46.3) -- the configuration these checkpoints were released for.

Three configurations were rejected first, and only one of them announced itself:
  * colqwen2-v1.0 @ transformers 5.15 -- 2 params randomly initialised (C-22)
  * colqwen2.5-v0.2 @ transformers 5.15 -- LoRA adapter never applied. Weights load
    CLEAN (missing_keys == []) and the model returns confident, plausible scores while
    ranking the gold page 12th of 20 (C-24). A weight-integrity gate passes this.
  * colSmol-256M -- works, but a 256M model would understate the ColPali family and
    bias Gate C in our own favour.

Two preflight gates, both blocking:
  1. `missing_keys == []` from `output_loading_info=True`
  2. behavioural: rank the gold page against EVERY page of its own document and require
     performance well above the k/n_pages random baseline. Gate 1 alone passed a model
     that retrieved at chance, so gate 2 is the one that actually protects Gate C.

Storage: float16 multi-vector, ~190 KB/page -> ~3.9 GB for 20,530 pages.
Resumable per document. Prints rate, ETA and peak VRAM.

    python embed_pages.py --preflight-only
    python embed_pages.py mmlongbench
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _resize(im, maxside=1024):
    w, h = im.size
    s = maxside / max(w, h)
    return im.resize((max(1, int(w*s)), max(1, int(h*s))), Image.LANCZOS) if s < 1 else im

ROOT = Path('/work/gramm-rag-colpali')
IMG = ROOT / 'page_images_full'
IDX = ROOT / 'index'
MODEL_ID = 'vidore/colqwen2-v1.0'
# ColQwen2 v1.0 does not downscale internally. Memory does NOT scale linearly with
# batch: attention over the patch grid is quadratic, so extrapolating from batch 2's
# 9.4 GiB peak to batch 6 was wrong -- batch 6 asked for a single 19.1 GiB allocation
# on top of 25.9 GiB already resident. Rather than keep guessing, the batch adapts:
# start here and halve on OOM, permanently, for the rest of the run.
BATCH = 4
MAX_SIDE = 1024
SEED = 42

QMETA = {
    'mmlongbench': Path('/work/gramm-rag-3benchmarks/data/mmlongbench/qmeta_mmlb.json'),
    'docbench': Path('/work/gramm-rag-3benchmarks/data/docbench/qmeta_docbench.json'),
}


def load_model():
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    t0 = time.time()
    model, info = ColQwen2.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map='cuda:0',
        output_loading_info=True)
    missing = [k for k in info.get('missing_keys', []) if 'lora' not in k.lower()]
    unexpected = info.get('unexpected_keys', [])
    # GATE 1 -- C-22. Never downgrade this to a warning.
    if missing:
        raise RuntimeError(
            f'{MODEL_ID}: {len(missing)} parameters were newly initialised instead of '
            f'loaded ({missing[:5]}). The model would emit plausible but meaningless '
            f'embeddings. Refusing to proceed.')
    model = model.eval()
    proc = ColQwen2Processor.from_pretrained(MODEL_ID)
    print(f'model OK: {MODEL_ID} in {time.time()-t0:.0f}s | '
          f'missing=0 unexpected={len(unexpected)} | '
          f'VRAM {torch.cuda.memory_allocated()/2**30:.1f} GiB', flush=True)
    return model, proc


_BATCH = [BATCH]        # mutable so an OOM downgrade persists across documents


@torch.no_grad()
def embed_images(model, proc, paths):
    out = []
    i = 0
    while i < len(paths):
        n = _BATCH[0]
        chunk = paths[i:i+n]
        imgs = [_resize(Image.open(p).convert('RGB'), MAX_SIDE) for p in chunk]
        try:
            b = proc.process_images(imgs).to('cuda:0')
            e = model(**b)
        except torch.OutOfMemoryError:
            for im in imgs:
                im.close()
            torch.cuda.empty_cache()
            if n == 1:
                raise
            _BATCH[0] = max(1, n // 2)
            print(f'  OOM at batch {n} -> retrying at {_BATCH[0]}', flush=True)
            continue          # same slice, smaller batch
        for j in range(e.shape[0]):
            out.append(e[j].to(torch.float16).cpu().numpy())
        for im in imgs:
            im.close()
        i += len(chunk)
    return out


@torch.no_grad()
def embed_queries(model, proc, queries):
    out = []
    for i in range(0, len(queries), BATCH):
        b = proc.process_queries(queries[i:i+BATCH]).to('cuda:0')
        e = model(**b)
        for j in range(e.shape[0]):
            out.append(e[j].to(torch.float16).cpu().numpy())
    return out


def preflight(model, proc, n_docs=6, max_q=4):
    """GATE 2 -- behavioural, and the gate that actually matters.

    Rank the gold evidence page against EVERY page of its own document, exactly as
    retrieval will. A pairwise test against one random page is far too weak: pages
    within a document are topically similar, so a broken model still wins ~50% and an
    earlier version of this gate passed a model retrieving at chance. Full-document
    ranking has a meaningful random baseline (k / n_pages) to compare against.

    Uses proc.score_multi_vector -- the library's own scorer. A hand-rolled MaxSim
    scored 7/20 where the official one scored 9/15 on identical embeddings.
    """
    import ast
    src = json.loads(Path('/work/gramm-rag-mmlongbench/data/mmlongbench/'
                          'mmlongbench_doc.json').read_text())
    qm = json.loads(QMETA['mmlongbench'].read_text())['rows']
    rng = random.Random(SEED)
    cands = [i for i, r in enumerate(qm)
             if 8 <= r['doc_pages'] <= 60 and ast.literal_eval(src[i]['evidence_pages'])]
    rng.shuffle(cands)
    by_doc = {}
    for i in cands:
        by_doc.setdefault(qm[i]['doc_id'], []).append(i)

    hits = {1: 0, 5: 0}
    rand = {1: 0.0, 5: 0.0}
    tot = 0
    for doc in list(by_doc)[:n_docs]:
        r = qm[by_doc[doc][0]]
        stem = doc[:-4] if doc.lower().endswith('.pdf') else doc
        paths = [IMG / 'mmlongbench' / f'{stem}_p{x}.png'
                 for x in range(1, r['doc_pages'] + 1)]
        paths = [x for x in paths if x.exists()]
        if len(paths) < 5:
            continue
        pe = [torch.from_numpy(e).float() for e in embed_images(model, proc, paths)]
        pages = [int(x.stem.rsplit('_p', 1)[1]) for x in paths]
        qs = by_doc[doc][:max_q]
        # Queries are ragged too (differing token counts), so pass a LIST --
        # score_multi_vector handles ragged inputs; torch.stack cannot.
        qe = [torch.from_numpy(e).float()
              for e in embed_queries(model, proc, [src[i]['question'] for i in qs])]
        scores = proc.score_multi_vector(qe, pe)
        for row, i in enumerate(qs):
            gp = [g for g in ast.literal_eval(src[i]['evidence_pages'])
                  if 1 <= g <= r['doc_pages']]
            if not gp:
                continue
            order = [pages[j] for j in torch.argsort(scores[row], descending=True).tolist()]
            tot += 1
            for k in hits:
                hits[k] += any(g in order[:k] for g in gp)
                rand[k] += min(1.0, k * len(gp) / len(pages))
    if not tot:
        raise RuntimeError('preflight found no scoreable questions')
    t1, t5 = hits[1] / tot, hits[5] / tot
    r1, r5 = rand[1] / tot, rand[5] / tot
    print(f'preflight: {tot} questions over {n_docs} docs', flush=True)
    print(f'  gold-in-top1  {t1:.2f}  (random {r1:.2f})', flush=True)
    print(f'  gold-in-top5  {t5:.2f}  (random {r5:.2f})', flush=True)
    # Require a clear margin over the random baseline, not merely "above" it.
    if not (t5 > r5 + 0.20 and t1 > r1):
        raise RuntimeError(
            f'BEHAVIOURAL PREFLIGHT FAILED: top5 {t5:.2f} vs random {r5:.2f}, '
            f'top1 {t1:.2f} vs random {r1:.2f}. The model is not discriminating '
            f'evidence pages; retrieval built on it would be noise and Gate C would '
            f'be a fabrication.')
    print('preflight PASSED -- model ranks gold pages well above chance.', flush=True)


def run(bench, model, proc):
    rows = json.loads(QMETA[bench].read_text())['rows']
    docs = {}
    for r in rows:
        docs.setdefault(r['doc_id'], r['doc_pages'])
    odir = IDX / bench
    odir.mkdir(parents=True, exist_ok=True)
    total = sum(docs.values())
    print(f'{bench}: {len(docs)} docs, {total:,} pages', flush=True)

    done_pages = 0
    t0 = time.time()
    for di, (doc_id, npages) in enumerate(sorted(docs.items()), 1):
        stem = doc_id[:-4] if doc_id.lower().endswith('.pdf') else doc_id
        out = odir / f'{stem}.npz'
        if out.exists():
            done_pages += npages
            continue
        paths, pages = [], []
        for p in range(1, npages + 1):
            f = IMG / bench / f'{stem}_p{p}.png'
            if f.exists():
                paths.append(f); pages.append(p)
        if not paths:
            print(f'  NO IMAGES for {doc_id} -- skipping', flush=True)
            continue
        embs = embed_images(model, proc, paths)
        # Ragged (patch counts vary with page aspect); store object array + page map.
        np.savez_compressed(
            out, pages=np.array(pages, dtype=np.int32),
            **{f'p{pg}': e for pg, e in zip(pages, embs)})
        done_pages += len(paths)
        el = time.time() - t0
        rate = done_pages / el if el else 0
        eta = (total - done_pages) / rate if rate else float('nan')
        if di % 5 == 0 or di == len(docs):
            print(f'  [{di}/{len(docs)}] pages={done_pages:,}/{total:,} | '
                  f'{rate:.1f} pg/s | ETA {eta/60:.1f} min | '
                  f'VRAM peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB',
                  flush=True)
    print(f'{bench} DONE: {done_pages:,} pages in {(time.time()-t0)/60:.1f} min',
          flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('benchmarks', nargs='*', default=['mmlongbench', 'docbench'])
    ap.add_argument('--preflight-only', action='store_true')
    a = ap.parse_args()
    assert torch.cuda.is_available(), 'GPU not reclaimed -- recreate the container'
    model, proc = load_model()
    preflight(model, proc)
    if a.preflight_only:
        sys.exit(0)
    for b in a.benchmarks:
        run(b, model, proc)
