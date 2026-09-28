#!/usr/bin/env python3
"""
generate_colpali.py — T-C2 stage 5/6: generate answers for the ColPali arm.

Only retrieval differs. Everything downstream is the project's own code, reached by
adding the benchmark repo to sys.path:
  * context assembly  -- synthetic element nodes for the selected pages, in
                         (page order, reading_order), the tail of the pipeline's own
                         `page_context_nodes`
  * prompt            -- src.generation.prompt_builder.build_prompt, PROMPT_V=3,
                         max_context_chars = PAGE_CONTEXT_CHARS_3 = 12000
  * generator         -- src.generation.vlm_client.VLMClient, gpt-4.1-mini + page images
  * scoring/judge     -- src.evaluation.metrics (separate step)

Page images are re-rendered at **dpi=120**, matching `render_page_image` in the
benchmark notebook. My ColPali renders are 150 DPI; feeding those would give this arm a
different visual input from every other arm and quietly break the control.

Page numbering is 1-based throughout: parsed `element.page_no` runs 1..N (verified), gold
pages are 1-based, and the ColPali index was built as `_p1..N`. R-9 is the precedent for
checking rather than assuming.

Hardening: T6 fail-fast at 15 consecutive errors, T7 checkpoint every 100 questions with
a RUN_CONFIG match, and the D-34 credit-exhaustion guard (an exhausted balance arrives as
RateLimitError, indistinguishable by type from throttling).
"""
import argparse
import ast
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/work/gramm-rag-colpali')
RET = ROOT / 'retrieval'
OUT = ROOT / 'generation'
IMGC = ROOT / 'gen_page_images'          # 120-dpi, generator-facing
MODEL = 'gpt-4.1-mini'
PROVIDER = 'openai'
MAX_CONTEXT_CHARS = 12000                # PAGE_CONTEXT_CHARS_3
DPI = 120
FAIL_FAST_N = 15
CHECKPOINT_EVERY = 100
WORKERS = 8

BENCH = {
    'mmlongbench': dict(
        repo='/work/gramm-rag-mmlongbench',
        parsed='/work/gramm-rag-mmlongbench/cache/parsed',
        pdfs='/work/gramm-rag-mmlongbench/pdfs_volume',
        qmeta='/work/gramm-rag-3benchmarks/data/mmlongbench/qmeta_mmlb.json',
        base='/work/gramm-rag-3benchmarks/data/mmlongbench/baseline_vector_mmlb.json',
        src='/work/gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json',
    ),
    'docbench': dict(
        repo='/work/gramm-rag-docbench',
        parsed='/work/gramm-rag-docbench/cache/parsed',
        pdfs='/work/gramm-rag-docbench/data_volume',
        qmeta='/work/gramm-rag-3benchmarks/data/docbench/qmeta_docbench.json',
        base='/work/gramm-rag-3benchmarks/data/docbench/baseline_vector_docbench.json',
        src=None,
    ),
}

FATAL = ('no credits', 'insufficient_quota', 'insufficient quota',
         'exceeded your current quota', 'billing', 'payment required')


def fatal_if_exhausted(e):
    if any(s in str(e).lower() for s in FATAL):
        raise RuntimeError(f'CREDIT EXHAUSTION (D-34) — stopping rather than '
                           f'emitting empty answers: {e}')


def parsed_path(pdir: Path, doc_id: str):
    """R-36: strip ONLY a trailing '.pdf'; Path().stem truncates at any dot."""
    stem = doc_id[:-4] if doc_id.lower().endswith('.pdf') else doc_id
    for c in (f'{stem}.pdf.json', f'{stem}.json'):
        p = pdir / c
        if p.exists():
            return p
    return None


def pdf_for(bench, pdfs: Path, doc_id: str):
    if bench == 'mmlongbench':
        p = pdfs / doc_id
        return p if p.exists() else None
    hits = sorted((pdfs / doc_id).glob('*.pdf'))   # C-23: original filename, not <id>.pdf
    return hits[0] if hits else None


def render(bench, pdf: Path, doc_id: str, page_no: int):
    """Render one page at the pipeline's dpi=120. Cached."""
    import fitz
    stem = doc_id[:-4] if doc_id.lower().endswith('.pdf') else doc_id
    d = IMGC / bench
    d.mkdir(parents=True, exist_ok=True)
    out = d / f'{stem}_p{page_no}.png'
    if out.exists() and out.stat().st_size:
        return out
    try:
        doc = fitz.open(str(pdf))
        if page_no < 1 or page_no > doc.page_count:   # D-31: 1-based bounds
            doc.close()
            return None
        doc[page_no - 1].get_pixmap(dpi=DPI).save(out)
        doc.close()
        return out
    except Exception:
        return None


def page_nodes(parsed: dict, pages: list):
    """Synthetic nodes for every element on `pages`, in (page order, reading_order).

    This is the tail of the notebook's `page_context_nodes`; build_prompt re-resolves
    the text from `parsed`, so only node_type/local_idx are needed.
    """
    t2e = {}
    for e in parsed.get('elements', []):
        t2e.setdefault(e['type'], []).append(e)
    sel = list(pages)
    out = []
    for nt, elems in t2e.items():
        for li, e in enumerate(elems):
            try:
                pg = int(e.get('page_no', -1))
            except (TypeError, ValueError):
                continue
            if pg in sel:
                try:
                    ro = int(e.get('reading_order', 0))
                except (TypeError, ValueError):
                    ro = 0
                out.append((sel.index(pg), ro, {'node_type': nt, 'local_idx': li}))
    out.sort(key=lambda t: (t[0], t[1]))
    return [nd for _, _, nd in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bench', choices=list(BENCH))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--arm', default='colpali',
                    help='colpali | union | colpali_matched')
    a = ap.parse_args()
    spec = BENCH[a.bench]
    sys.path.insert(0, spec['repo'])
    from src.generation.prompt_builder import build_prompt
    from src.generation.vlm_client import VLMClient
    import inspect
    # R-49 added `answer_format` to MMLongBench's build_prompt only; DocBench's
    # signature predates it. Passing it there raises on every question -- which the
    # T6 guard caught after 15 consecutive failures rather than emitting 1,102 blanks.
    ACCEPTS_AFMT = 'answer_format' in inspect.signature(build_prompt).parameters

    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        sys.exit('FATAL: OPENAI_API_KEY not set.')          # never fall back silently
    assert PROVIDER == 'openai', 'Together is deprecated on this project'

    _pre = 'colpali_retrieval' if a.arm == 'colpali' else f'{a.arm}_retrieval'
    ret = json.loads((RET / f'{_pre}_{a.bench}.json').read_text())
    base = json.loads(Path(spec['base']).read_text())
    qm = json.loads(Path(spec['qmeta']).read_text())['rows']
    n = ret['n_questions']
    if a.bench == 'mmlongbench':
        srcq = json.loads(Path(spec['src']).read_text())
        questions = [s['question'] for s in srcq]
        afmt = [s.get('answer_format') for s in srcq]        # R-49
    else:
        questions = list(base['questions'])
        afmt = [None] * n
    docs = [r['doc_id'] for r in qm]
    ctx_pages = ret['context_pages']

    RUN_CONFIG = dict(ret['RUN_CONFIG'])
    RUN_CONFIG.update({'GEN_PROVIDER': PROVIDER, 'GEN_MODEL': MODEL,
                       'PAGE_CONTEXT_CHARS_3': MAX_CONTEXT_CHARS,
                       'USE_PAGE_IMAGES': True, 'PAGE_IMG_DPI': DPI,
                       'PROMPT_V': 3, 'BENCHMARK': a.bench})

    OUT.mkdir(parents=True, exist_ok=True)
    ck = OUT / f'ckpt_{a.arm}_{a.bench}.json'
    preds = {}
    if ck.exists():
        s = json.loads(ck.read_text())
        if s.get('RUN_CONFIG') == RUN_CONFIG:                # R-42
            preds = {int(k): v for k, v in s['predictions'].items()}
            print(f'resuming: {len(preds)} already generated', flush=True)
        else:
            print('checkpoint RUN_CONFIG mismatch — ignoring it', flush=True)

    pcache, pdfcache = {}, {}
    idx = list(range(n))[:a.limit] if a.limit else list(range(n))
    todo = [i for i in idx if i not in preds]
    print(f'{a.bench}: {len(todo)} to generate of {len(idx)}', flush=True)

    client = VLMClient(provider=PROVIDER, openai_model=MODEL)
    usage = {'calls': 0}
    consec = 0
    failures = []
    t0 = time.time()

    def one(i):
        doc = docs[i]
        if doc not in pcache:
            pf = parsed_path(Path(spec['parsed']), doc)
            pcache[doc] = json.loads(pf.read_text()) if pf else {'elements': []}
        parsed = pcache[doc]
        pages = ctx_pages[i]
        nodes = page_nodes(parsed, pages)
        kw = dict(question=questions[i], retrieved_nodes=nodes, parsed=parsed,
                  reward_detail={}, refused=False, benchmark=a.bench,
                  max_context_chars=MAX_CONTEXT_CHARS)
        if ACCEPTS_AFMT:
            kw['answer_format'] = afmt[i]
        prompt = build_prompt(**kw)
        if doc not in pdfcache:
            pdfcache[doc] = pdf_for(a.bench, Path(spec['pdfs']), doc)
        imgs = []
        if pdfcache[doc]:
            for pg in pages:
                p = render(a.bench, pdfcache[doc], doc, pg)
                if p:
                    imgs.append(str(p))
        txt = client.generate(prompt, temperature=0.0, max_tokens=512,
                              image_paths=imgs or None)
        usage['calls'] += 1
        return i, (txt or '').strip()

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for st in range(0, len(todo), CHECKPOINT_EVERY):
            batch = todo[st:st+CHECKPOINT_EVERY]
            for f in [ex.submit(one, i) for i in batch]:
                try:
                    i, txt = f.result()
                except Exception as e:
                    fatal_if_exhausted(e)                     # D-34
                    failures.append(str(e)[:200])
                    consec += 1
                    if consec >= FAIL_FAST_N:                 # T6
                        raise RuntimeError(
                            f'{FAIL_FAST_N} consecutive generation failures — '
                            f'aborting. Last: {failures[-1]}')
                    continue
                consec = 0
                preds[i] = txt
            ck.write_text(json.dumps({'RUN_CONFIG': RUN_CONFIG,
                                      'predictions': preds}))  # T7
            el = time.time() - t0
            done = len(preds)
            rate = usage['calls'] / el if el else 0
            left = len(idx) - done
            print(f'  {done}/{len(idx)} | {rate:.2f} calls/s | '
                  f'ETA {left/rate/60 if rate else float("nan"):.1f} min | '
                  f'failures {len(failures)}', flush=True)

    order = sorted(preds)
    res = {'system': ret.get('system', 'ColPali (ColQwen2-v1.0)'), 'benchmark': a.bench,
           'n_questions': len(order), 'RUN_CONFIG': RUN_CONFIG,
           'predictions': [preds[i] for i in order],
           'question_index': order,
           'golds': [base['golds'][i] for i in order],
           'context_pages': [ctx_pages[i] for i in order],
           'retrieved_pages': [ret['retrieved_pages'][i] for i in order],
           'n_failures': len(failures), 'failures': failures[:5]}
    if 'is_answerable' in base:
        res['is_answerable'] = [base['is_answerable'][i] for i in order]
    if a.bench == 'docbench':
        for k in ('types', 'evidences', 'questions'):
            if k in base:
                res[k] = [base[k][i] for i in order]
    tag = f'_pilot{a.limit}' if a.limit else ''
    (OUT / f'{a.arm}_gen_{a.bench}{tag}.json').write_text(json.dumps(res))
    empt = sum(1 for p in res['predictions'] if not p)
    print(f'{a.bench} DONE: {len(order)} predictions, {empt} empty, '
          f'{len(failures)} failures, {usage["calls"]} calls in '
          f'{(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
