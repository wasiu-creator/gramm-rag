#!/usr/bin/env python3
"""
WP-3 signal S-a — sufficiency autorater over ARCHIVED contexts.

Asks, over the retrieved context only: *is this sufficient to answer the question?*
The abstention decision is then: decline when the autorater says insufficient.

Pre-registered in WP3_PREREGISTRATION.md. Reads only archived artefacts: GraMM's shipped
`context_pages` and the MinerU parse, reassembled with the SAME page/reading order and the
SAME 12,000-character budget the generator conditioned on. **No re-retrieval and no
regeneration**, so this signal cannot differ from the shipped run through any path except the
autorater call itself.

Guards: D-34 credit-exhaustion is fatal and never retried; T6 fail-fast at 15 consecutive
failures; T7 checkpoint every 100 with a RUN_CONFIG match (R-42); no silent fallback anywhere.
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/work/gramm-rag-colpali')
ARCH = Path('/work/gramm-rag-3benchmarks/data/mmlongbench')
PARSED = Path('/work/gramm-rag-mmlongbench/cache/parsed')
OUT = ROOT / 'scores'
MODEL = 'gpt-4.1-mini'          # same family as the generator; stamped
MAX_CONTEXT_CHARS = 12000       # PAGE_CONTEXT_CHARS_3, identical to the shipped run
FAIL_FAST_N, CHECKPOINT_EVERY, WORKERS = 15, 100, 8
FATAL = ('no credits', 'insufficient_quota', 'insufficient quota',
         'exceeded your current quota', 'billing', 'payment required')

PROMPT = """You are assessing whether a retrieved context contains enough information to answer a question.

Question: {q}

Retrieved context:
{ctx}

Does the context contain sufficient information to answer the question?
Answer with exactly one word: SUFFICIENT or INSUFFICIENT."""


def fatal_if_exhausted(e):
    if any(s in str(e).lower() for s in FATAL):
        sys.exit(f'FATAL D-34: credits exhausted, not throttling -- {e}')


def context_for(parsed, pages, budget=MAX_CONTEXT_CHARS):
    """Reassemble the context block in (page order, reading_order) under the same budget.

    This mirrors build_prompt's context assembly: elements on the selected pages, ordered by
    the page's rank then reading_order, truncated at the character budget.
    """
    sel = list(pages)
    items = []
    for e in parsed.get('elements', []):
        try:
            pg = int(e.get('page_no', -1))
        except (TypeError, ValueError):
            continue
        if pg in sel:
            items.append((sel.index(pg), int(e.get('reading_order', 0) or 0),
                          (e.get('text') or '').strip()))
    items.sort(key=lambda x: (x[0], x[1]))
    out, n = [], 0
    for _, _, txt in items:
        if not txt:
            continue
        if n >= budget:
            break
        out.append(txt); n += len(txt)
    return '\n'.join(out)[:budget]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--pages', choices=['retrieved', 'gold'], default='retrieved',
                    help="D-1: 'gold' rebuilds context from gold_pages instead of the "
                         "retrieved pages, to separate rater failure from retrieval failure")
    a = ap.parse_args()
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        sys.exit('FATAL: OPENAI_API_KEY not set')
    from openai import OpenAI
    client = OpenAI(api_key=key)

    base = json.loads((ARCH / 'baseline_vector_mmlb.json').read_text())
    gramm = json.loads((ARCH / 'gramm_mmlongbench_s42.json').read_text())
    qm = json.loads((ARCH / 'qmeta_mmlb.json').read_text())['rows']
    srcq = json.loads(Path('/work/gramm-rag-mmlongbench/data/mmlongbench/'
                           'mmlongbench_doc.json').read_text())
    n = base['n_questions']
    assert len(qm) == len(srcq) == n, 'index alignment broken'
    questions = [s['question'] for s in srcq]
    ans = base['is_answerable']; ans_all = ans
    if a.pages == 'gold':
        gp = base['gold_pages']
        # Unanswerable questions have no gold pages by definition -- they keep the retrieved
        # context, because there is no evidence to inject. Substituting anything else would
        # change what the rater is asked about on that stratum and break the comparison.
        ctx_pages = [gp[i] if (ans_all[i] and gp[i]) else gramm['context_pages'][i]
                     for i in range(n)]
        n_inj = sum(1 for i in range(n) if ans_all[i] and gp[i])
        print(f'D-1: gold pages injected for {n_inj} answerable questions; '
              f'{n - n_inj} keep retrieved context', flush=True)
    else:
        ctx_pages = gramm['context_pages']

    RUN_CONFIG = {'SIGNAL': 'S-a autorater', 'MODEL': MODEL,
                  'MAX_CONTEXT_CHARS': MAX_CONTEXT_CHARS,
                  'CONTEXT_SOURCE': ('GOLD PAGES (D-1 diagnostic)' if a.pages=='gold'
                                      else 'archived GraMM context_pages, reassembled'),
                  'PAGES': a.pages,
                  'BENCHMARK': 'mmlongbench', 'TEMPERATURE': 0.0}
    print('RUN_CONFIG: ' + json.dumps(RUN_CONFIG), flush=True)

    ck = OUT / f'ckpt_wp3_autorater_{a.pages}.json'
    got = {}
    if ck.exists():
        s = json.loads(ck.read_text())
        if s.get('RUN_CONFIG') == RUN_CONFIG:                 # R-42
            got = {int(k): v for k, v in s['scores'].items()}
            print(f'resuming: {len(got)} already scored', flush=True)
        else:
            print('checkpoint RUN_CONFIG mismatch — ignoring', flush=True)

    pcache = {}
    idx = list(range(n))[:a.limit] if a.limit else list(range(n))
    todo = [i for i in idx if i not in got]
    print(f'{len(todo)} to score of {len(idx)}', flush=True)
    usage = {'calls': 0, 'in': 0, 'out': 0}
    failures, consec = [], 0

    def one(i):
        doc = qm[i]['doc_id']
        if doc not in pcache:
            f = PARSED / f'{doc}.json'
            pcache[doc] = json.loads(f.read_text()) if f.exists() else {'elements': []}
        ctx = context_for(pcache[doc], ctx_pages[i])
        r = client.chat.completions.create(
            model=MODEL, temperature=0.0, max_tokens=4,
            messages=[{'role': 'user',
                       'content': PROMPT.format(q=questions[i], ctx=ctx or '(empty)')}])
        usage['calls'] += 1
        usage['in'] += r.usage.prompt_tokens; usage['out'] += r.usage.completion_tokens
        txt = (r.choices[0].message.content or '').strip().upper()
        # No silent fallback: an unparseable verdict is recorded as such, not coerced.
        if txt.startswith('SUFFICIENT'):
            return i, 1
        if txt.startswith('INSUFFICIENT'):
            return i, 0
        return i, -1

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
                                      'scores': {str(k): v for k, v in got.items()}}))
            el = time.time() - t0
            print(f'  {len(got)}/{len(idx)} | {usage["calls"]/el:.2f} calls/s | '
                  f'failures {len(failures)}', flush=True)

    order = sorted(got)
    unparsed = sum(1 for i in order if got[i] == -1)
    res = {'RUN_CONFIG': RUN_CONFIG, 'n': len(order), 'question_index': order,
           'sufficient': [got[i] for i in order],
           'is_answerable': [ans[i] for i in order],
           'n_unparsed': unparsed, 'n_failures': len(failures),
           'usage': usage,
           'cost_usd': round(usage['in'] * 0.40 / 1e6 + usage['out'] * 1.60 / 1e6, 6),
           'seconds': round(time.time() - t0, 1)}
    tag = '' if a.pages == 'retrieved' else '_goldpages'
    (OUT / f'wp3_autorater_mmlongbench{tag}.json').write_text(json.dumps(res))
    print(f"\nS-a DONE: {len(order)} scored, {unparsed} unparsed, {len(failures)} failures, "
          f"${res['cost_usd']:.4f}, {res['seconds']/60:.1f} min")


if __name__ == '__main__':
    main()
