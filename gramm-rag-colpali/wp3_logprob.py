#!/usr/bin/env python3
"""
WP-3 signals S-b (answer-token logprob confidence) and S-c (k=5 self-consistency).

Both re-generate answers for MMLongBench and read a confidence signal off the generation.
Pre-registered in WP3_PREREGISTRATION.md.

**`vlm_client.py` is deliberately NOT modified.** It is the shared generator every archived
arm used, and adding logprobs to it would put a change into a component with published results
downstream. Instead this script calls the API directly with the SAME model, temperature,
max_tokens, prompt (`build_prompt`, PROMPT_V=3, 12,000-char budget) and the SAME cached 120-dpi
page images. The only difference from the shipped call is `logprobs=True`, which is additive:
it returns extra fields and changes nothing about sampling. Nothing archived can be affected,
because nothing archived is touched — which is stronger than verifying a modification.

S-b: temperature 0.0, one sample, mean and min answer-token logprob.
S-c: temperature 1.0, k=5 samples, agreement rate over normalised answers, on a 300-question
     stratified subsample (R-50/D-35: reported with n stated, never carrying a headline).

Guards: D-34 fatal on credit exhaustion, T6 fail-fast at 15, T7 checkpoint every 100 with
RUN_CONFIG match (R-42), no silent fallback.
"""
import argparse, json, math, os, re, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, '/work/gramm-rag-mmlongbench')
from src.generation.prompt_builder import build_prompt          # noqa: E402

ARCH = Path('/work/gramm-rag-3benchmarks/data/mmlongbench')
PARSED = Path('/work/gramm-rag-mmlongbench/cache/parsed')
PDFS = Path('/work/gramm-rag-mmlongbench/pdfs_volume')
IMGC = Path('/work/gramm-rag-colpali/gen_page_images')
OUT = Path('/work/gramm-rag-colpali/scores')
MODEL, MAX_CTX, DPI = 'gpt-4.1-mini', 12000, 120
FAIL_FAST_N, CHECKPOINT_EVERY, WORKERS = 15, 100, 8
FATAL = ('no credits', 'insufficient_quota', 'insufficient quota',
         'exceeded your current quota', 'billing', 'payment required')


def fatal_if_exhausted(e):
    if any(s in str(e).lower() for s in FATAL):
        sys.exit(f'FATAL D-34: credits exhausted, not throttling -- {e}')


def page_nodes(parsed, pages):
    """Identical to generate_colpali.page_nodes — synthetic nodes in (page, reading) order."""
    t2e = {}
    for e in parsed.get('elements', []):
        t2e.setdefault(e['type'], []).append(e)
    sel, out = list(pages), []
    for nt, elems in t2e.items():
        for li, e in enumerate(elems):
            try:
                pg = int(e.get('page_no', -1))
            except (TypeError, ValueError):
                continue
            if pg in sel:
                try:
                    ro = int(e.get('reading_order', 0) or 0)
                except (TypeError, ValueError):
                    ro = 0
                out.append((sel.index(pg), ro, {'node_type': nt, 'local_idx': li}))
    out.sort(key=lambda x: (x[0], x[1]))
    return [n for _, _, n in out]


def render(pdf, doc_id, page_no):
    import fitz
    stem = doc_id[:-4] if doc_id.lower().endswith('.pdf') else doc_id
    d = IMGC / 'mmlongbench'; d.mkdir(parents=True, exist_ok=True)
    out = d / f'{stem}_p{page_no}.png'
    if out.exists() and out.stat().st_size:
        return out
    try:
        doc = fitz.open(str(pdf))
        if page_no < 1 or page_no > doc.page_count:
            doc.close(); return None
        doc[page_no - 1].get_pixmap(dpi=DPI).save(out); doc.close(); return out
    except Exception:
        return None


def norm_answer(s):
    return re.sub(r'\s+', ' ', (s or '').strip().lower()).rstrip('.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signal', choices=['sb', 'sc'], required=True)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--subsample', type=int, default=300)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        sys.exit('FATAL: OPENAI_API_KEY not set')
    import base64
    from openai import OpenAI
    client = OpenAI(api_key=key)

    base = json.loads((ARCH / 'baseline_vector_mmlb.json').read_text())
    gramm = json.loads((ARCH / 'gramm_mmlongbench_s42.json').read_text())
    qm = json.loads((ARCH / 'qmeta_mmlb.json').read_text())['rows']
    srcq = json.loads(Path('/work/gramm-rag-mmlongbench/data/mmlongbench/'
                           'mmlongbench_doc.json').read_text())
    n = base['n_questions']
    questions = [s['question'] for s in srcq]
    afmt = [s.get('answer_format') for s in srcq]
    ctx_pages, ans = gramm['context_pages'], base['is_answerable']

    if a.signal == 'sc':
        # Stratified subsample: preserve the answerable/unanswerable ratio exactly.
        import random
        rnd = random.Random(42)
        U = [i for i in range(n) if not ans[i]]; A = [i for i in range(n) if ans[i]]
        kU = round(a.subsample * len(U) / n); kA = a.subsample - kU
        idx = sorted(rnd.sample(U, kU) + rnd.sample(A, kA))
        temp, ksamp = 1.0, a.k
    else:
        idx = list(range(n)); temp, ksamp = 0.0, 1
    if a.limit:
        idx = idx[:a.limit]

    RUN_CONFIG = {'SIGNAL': 'S-b logprob' if a.signal == 'sb' else f'S-c self-consistency k={a.k}',
                  'MODEL': MODEL, 'TEMPERATURE': temp, 'K': ksamp,
                  'MAX_CONTEXT_CHARS': MAX_CTX, 'PAGE_IMG_DPI': DPI, 'PROMPT_V': 3,
                  'N_QUESTIONS': len(idx), 'BENCHMARK': 'mmlongbench',
                  'VLM_CLIENT_MODIFIED': False}
    print('RUN_CONFIG: ' + json.dumps(RUN_CONFIG), flush=True)

    ck = OUT / f'ckpt_wp3_{a.signal}.json'
    got = {}
    if ck.exists():
        s = json.loads(ck.read_text())
        if s.get('RUN_CONFIG') == RUN_CONFIG:
            got = {int(k): v for k, v in s['res'].items()}
            print(f'resuming: {len(got)} done', flush=True)
        else:
            print('checkpoint RUN_CONFIG mismatch — ignoring', flush=True)

    pc, pdfc = {}, {}
    usage = {'calls': 0, 'in': 0, 'out': 0}
    failures, consec = [], 0

    def one(i):
        doc = qm[i]['doc_id']
        if doc not in pc:
            f = PARSED / f'{doc}.json'
            pc[doc] = json.loads(f.read_text()) if f.exists() else {'elements': []}
        parsed = pc[doc]; pages = ctx_pages[i]
        kw = dict(question=questions[i], retrieved_nodes=page_nodes(parsed, pages),
                  parsed=parsed, reward_detail={}, refused=False,
                  benchmark='mmlongbench', max_context_chars=MAX_CTX)
        import inspect
        if 'answer_format' in inspect.signature(build_prompt).parameters:
            kw['answer_format'] = afmt[i]
        prompt = build_prompt(**kw)
        if doc not in pdfc:
            p = PDFS / doc
            pdfc[doc] = p if p.exists() else None
        content = [{'type': 'text', 'text': prompt}]
        if pdfc[doc]:
            for pg in pages:
                ip = render(pdfc[doc], doc, pg)
                if ip:
                    b64 = base64.b64encode(ip.read_bytes()).decode('ascii')
                    content.append({'type': 'image_url',
                                    'image_url': {'url': f'data:image/png;base64,{b64}'}})
        msgs = [{'role': 'user', 'content': content if len(content) > 1 else prompt}]
        kwargs = dict(model=MODEL, messages=msgs, temperature=temp, max_tokens=512)
        if a.signal == 'sb':
            kwargs['logprobs'] = True
        if ksamp > 1:
            kwargs['n'] = ksamp
        r = client.chat.completions.create(**kwargs)
        usage['calls'] += ksamp
        usage['in'] += r.usage.prompt_tokens; usage['out'] += r.usage.completion_tokens
        if a.signal == 'sb':
            ch = r.choices[0]
            lps = [t.logprob for t in (ch.logprobs.content or [])] if ch.logprobs else []
            if not lps:
                raise RuntimeError('logprobs requested but absent -- refusing to guess')
            return i, {'answer': (ch.message.content or '').strip(),
                       'mean_lp': sum(lps) / len(lps), 'min_lp': min(lps),
                       'ppl': math.exp(-sum(lps) / len(lps)), 'n_tok': len(lps)}
        answers = [(c.message.content or '').strip() for c in r.choices]
        norm = [norm_answer(x) for x in answers]
        top, cnt = Counter(norm).most_common(1)[0]
        return i, {'answers': answers, 'agreement': cnt / len(norm), 'k': len(norm)}

    todo = [i for i in idx if i not in got]
    print(f'{len(todo)} to run of {len(idx)}', flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for st in range(0, len(todo), CHECKPOINT_EVERY):
            batch = todo[st:st + CHECKPOINT_EVERY]
            for f in [ex.submit(one, i) for i in batch]:
                try:
                    i, v = f.result(); got[i] = v; consec = 0
                except Exception as e:
                    fatal_if_exhausted(e)
                    failures.append(str(e)[:200]); consec += 1
                    if consec >= FAIL_FAST_N:
                        sys.exit(f'FATAL T6: {consec} consecutive failures')
            ck.write_text(json.dumps({'RUN_CONFIG': RUN_CONFIG,
                                      'res': {str(k): v for k, v in got.items()}}))
            el = time.time() - t0
            print(f'  {len(got)}/{len(idx)} | {usage["calls"]/el:.2f} calls/s | '
                  f'failures {len(failures)}', flush=True)

    order = sorted(got)
    res = {'RUN_CONFIG': RUN_CONFIG, 'question_index': order,
           'is_answerable': [ans[i] for i in order],
           'result': [got[i] for i in order], 'n': len(order),
           'n_failures': len(failures), 'usage': usage,
           'cost_usd': round(usage['in'] * 0.40 / 1e6 + usage['out'] * 1.60 / 1e6, 6),
           'seconds': round(time.time() - t0, 1)}
    (OUT / f'wp3_{a.signal}_mmlongbench.json').write_text(json.dumps(res))
    print(f"\n{a.signal.upper()} DONE: {len(order)} items, {len(failures)} failures, "
          f"${res['cost_usd']:.4f}, {res['seconds']/60:.1f} min")


if __name__ == '__main__':
    main()
