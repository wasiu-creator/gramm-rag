#!/usr/bin/env python3
"""
render_pages.py — rasterise EVERY page of both corpora for the ColPali arm.

Why this exists: `cache/results/page_images/` holds only 1,546 / 1,834 images,
because the pipeline rendered just the pages its own retrievers selected for the
generator. Indexing those would confine ColPali to pages our system already found
and make Gate C unfalsifiable by construction. ColPali must see the whole corpus:
6,522 + 14,008 = 20,530 pages.

CPU-only (PyMuPDF) so it can run while the GPU is busy. Resumable: an existing
non-empty PNG is skipped, so a crash costs only the page in flight.

Output: page_images_full/<bench>/<doc_id>_p<N>.png  (1-based page numbers, matching
the project's gold_pages convention -- R-9 established that the parser is 0-based
and gold is 1-based, and getting this wrong silently destroys every APPA number).
"""
import json
import sys
import time
from pathlib import Path

import fitz

ROOT = Path('/work/gramm-rag-colpali')
OUT = ROOT / 'page_images_full'
DPI = 150          # ColQwen2 downsamples anyway; 150 keeps text legible at ~0.4 MB/page

SPEC = {
    'mmlongbench': dict(
        pdf_dir=Path('/work/gramm-rag-mmlongbench/pdfs_volume'),
        qmeta=Path('/work/gramm-rag-3benchmarks/data/mmlongbench/qmeta_mmlb.json'),
        # doc_id already carries '.pdf'
        resolve=lambda pdf_dir, doc_id: pdf_dir / doc_id,
    ),
    'docbench': dict(
        pdf_dir=Path('/work/gramm-rag-docbench/data_volume'),
        qmeta=Path('/work/gramm-rag-3benchmarks/data/docbench/qmeta_docbench.json'),
        # The folder name is the doc_id, but the PDF inside keeps its ORIGINAL name
        # (0/P19-1598.pdf, 1/W18-4401.pdf ...) -- it is not <doc_id>.pdf. Each folder
        # holds exactly one PDF, so glob for it. Assuming <doc_id>.pdf failed all
        # 14,008 pages on the first pass (C-23).
        resolve=lambda pdf_dir, doc_id: next(
            iter(sorted((pdf_dir / doc_id).glob('*.pdf'))), pdf_dir / doc_id / 'MISSING'),
    ),
}


def main():
    benches = sys.argv[1:] or list(SPEC)
    grand_t0 = time.time()
    for bench in benches:
        s = SPEC[bench]
        rows = json.loads(s['qmeta'].read_text())['rows']
        docs = {}
        for r in rows:
            docs.setdefault(r['doc_id'], r['doc_pages'])
        odir = OUT / bench
        odir.mkdir(parents=True, exist_ok=True)
        total_pages = sum(docs.values())
        print(f'{bench}: {len(docs)} docs, {total_pages:,} pages -> {odir}', flush=True)

        done = skipped = failed = 0
        t0 = time.time()
        for di, (doc_id, npages) in enumerate(sorted(docs.items()), 1):
            pdf = s['resolve'](s['pdf_dir'], doc_id)
            if not pdf.exists():
                print(f'  MISSING PDF: {doc_id}', flush=True)
                failed += npages
                continue
            try:
                d = fitz.open(pdf)
            except Exception as e:
                print(f'  OPEN FAILED {doc_id}: {e}', flush=True)
                failed += npages
                continue
            stem = doc_id[:-4] if doc_id.lower().endswith('.pdf') else doc_id
            for p in range(len(d)):
                out = odir / f'{stem}_p{p+1}.png'      # 1-based, matches gold_pages
                if out.exists() and out.stat().st_size > 0:
                    skipped += 1
                    continue
                try:
                    pix = d[p].get_pixmap(dpi=DPI)
                    pix.save(out)
                    done += 1
                except Exception as e:
                    print(f'  RENDER FAILED {doc_id} p{p+1}: {e}', flush=True)
                    failed += 1
            d.close()
            if di % 10 == 0 or di == len(docs):
                el = time.time() - t0
                seen = done + skipped
                rate = done / el if el and done else 0
                left = total_pages - seen
                eta = left / rate if rate else float('nan')
                print(f'  [{di}/{len(docs)} docs] rendered={done} skipped={skipped} '
                      f'failed={failed} | {rate:.1f} pg/s | ETA {eta/60:.1f} min',
                      flush=True)
        print(f'{bench} DONE: rendered={done} skipped={skipped} failed={failed} '
              f'in {(time.time()-t0)/60:.1f} min', flush=True)
    print(f'ALL DONE in {(time.time()-grand_t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
