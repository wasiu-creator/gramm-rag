#!/usr/bin/env python3
"""
Rule 10 calibration for the DocBench gpt-4-turbo judge, run BEFORE authorising a full
1,102-question judge pass.

Gate H priced the S1 judge at $4-11 and could not narrow it, because the prompt embeds
`ref_text` of unbounded length. This measures it instead of estimating it.

Calibrated on an ALREADY-JUDGED arm's predictions (ColPali), not on the arm being
authorised: the judge config, prompt and corpus are identical, so $/call transfers, and it
does not require the new arm's generation to have finished. The measured rate is then
extrapolated with the observed spread reported alongside -- a mean $/call hides a long tail
when one field of the prompt is unbounded.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, '/work/gramm-rag-docbench')
from src.evaluation.metrics import DOCBENCH_JUDGE_PROMPT      # noqa: E402

ARCH = Path('/work/gramm-rag-3benchmarks/data/docbench')
GEN = Path('/work/gramm-rag-colpali/generation')
JUDGE_MODEL = 'gpt-4-turbo'
RATE_IN, RATE_OUT = 10.0/1e6, 30.0/1e6                        # listed, stamped

FATAL = ('no credits', 'insufficient_quota', 'insufficient quota',
         'exceeded your current quota', 'billing', 'payment required')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--arm', default='colpali')
    a = ap.parse_args()
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        sys.exit('FATAL: OPENAI_API_KEY not set')
    from openai import OpenAI
    client = OpenAI(api_key=key)

    base = json.loads((ARCH/'baseline_vector_docbench.json').read_text())
    gen = json.loads((GEN/f'{a.arm}_gen_docbench.json').read_text())
    preds, golds = gen['predictions'], gen['golds']
    qs = list(base['questions'])
    evid = base.get('evidences') or ['']*len(qs)

    # Stratified by prompt length, not the first N: ref_text is the unbounded field and the
    # whole point is to price its tail. Taking the first 50 would price the head.
    lens = [len(str(evid[i] or '')) for i in range(len(qs))]
    order = sorted(range(len(qs)), key=lambda i: lens[i])
    idx = [order[int(k*(len(order)-1)/(a.n-1))] for k in range(a.n)]

    tin = tout = 0
    per_call, t0 = [], time.time()
    for j, i in enumerate(idx, 1):
        prompt = DOCBENCH_JUDGE_PROMPT.format(
            question=qs[i], sys_ans=(preds[i] or '')[:4000],
            ref_ans=golds[i] or '', ref_text=str(evid[i] or ''))
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL, temperature=0.0, max_tokens=8,
                messages=[{'role': 'system', 'content': 'You are a helpful evaluator.'},
                          {'role': 'user', 'content': prompt}])
        except Exception as e:
            if any(s in str(e).lower() for s in FATAL):        # D-34
                sys.exit(f'FATAL: credits exhausted, not throttling -- {e}')
            raise
        u = r.usage
        tin += u.prompt_tokens; tout += u.completion_tokens
        per_call.append(u.prompt_tokens*RATE_IN + u.completion_tokens*RATE_OUT)
        if j % 10 == 0:
            print(f'  {j}/{a.n}  cum ${sum(per_call):.4f}', flush=True)

    n_full = base['n_questions']
    mean = sum(per_call)/len(per_call)
    srt = sorted(per_call)
    p50, p90, p99 = srt[len(srt)//2], srt[int(.9*len(srt))], srt[int(.99*len(srt))]
    out = {'model': JUDGE_MODEL, 'n_calibration': a.n, 'calibrated_on_arm': a.arm,
           'sampling': 'stratified by ref_text length (evenly spaced quantiles)',
           'rates': {'in_per_1m': 10.0, 'out_per_1m': 30.0},
           'measured': {'prompt_tokens': tin, 'completion_tokens': tout,
                        'cost_usd': round(sum(per_call), 6),
                        'mean_per_call': mean, 'p50': p50, 'p90': p90, 'p99': p99},
           'projection_full_run': {'n_questions': n_full,
                                   'mean_based': round(mean*n_full, 2),
                                   'p90_based': round(p90*n_full, 2)},
           'gate_h_estimate': '$4-11', 'seconds': round(time.time()-t0, 1)}
    Path('/work/gramm-rag-colpali/scores/judge_calibration_docbench.json').write_text(
        json.dumps(out, indent=1))
    print(f"\nmeasured {a.n} calls: ${sum(per_call):.4f}  "
          f"mean ${mean:.5f}/call  p50 ${p50:.5f}  p90 ${p90:.5f}")
    print(f"prompt tokens {tin:,} in / {tout:,} out")
    print(f"PROJECTED full judge run ({n_full} questions): "
          f"${mean*n_full:.2f} (mean) / ${p90*n_full:.2f} (p90-based upper)")
    print(f"Gate H estimate was $4-11 -> "
          f"{'WITHIN' if 4 <= mean*n_full <= 11 else 'OUTSIDE'} the stated range")


if __name__ == '__main__':
    main()
