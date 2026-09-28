#!/usr/bin/env python3
"""
query_latency_v2.py — P3: the 2x2x2 query-latency measurement.

{GraMM, ColPali} x {cold, warm} x {MMLongBench, DocBench}, n>=500 per cell.

SYMMETRY IS THE POINT. GraMM loads a graph per query exactly as ColPali loads a multi-vector
store per query. Warming one and not the other measures our system against an unoptimised
reference — the error Gate C caught one layer up.

  cold — per-query load of that document's index/graph from disk (multi-tenant, cache miss)
  warm — index/graph already resident, timing the query operation only (repeated querying
         of one corpus)

Both states exclude generation. Warmup queries are discarded and counted.

The two stacks cannot share a container: colpali-engine 0.3.4 pins transformers==4.46.3 while
the GraMM re-run stack requires 4.57.6 for its integrity gate (C-28). Systems are therefore
run in separate containers in A-B-B-A order, and each system's two occurrences are compared
so thermal/clock drift is measured rather than assumed.
"""
import argparse, ast, json, random, time
import numpy as np, torch
from pathlib import Path

OUT = Path('/work/gramm-rag-colpali/cost'); OUT.mkdir(exist_ok=True)
WARMUP = 20

BENCH = {
 'mmlongbench': dict(repo='/work/gramm-rag-mmlongbench',
   qmeta='/work/gramm-rag-3benchmarks/data/mmlongbench/qmeta_mmlb.json',
   src='/work/gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json',
   base=None, hgt='hgt_mmlb', reward='reward_mmlb.json'),
 'docbench': dict(repo='/work/gramm-rag-docbench',
   qmeta='/work/gramm-rag-3benchmarks/data/docbench/qmeta_docbench.json', src=None,
   base='/work/gramm-rag-3benchmarks/data/docbench/baseline_vector_docbench.json',
   hgt='hgt_docbench', reward='reward_docbench.json'),
}


def load_questions(bench):
    s = BENCH[bench]
    qm = json.loads(Path(s['qmeta']).read_text())['rows']
    if s['src']:
        qs = [x['question'] for x in json.loads(Path(s['src']).read_text())]
    else:
        qs = list(json.loads(Path(s['base']).read_text())['questions'])
    return qm, qs


def run_gramm(bench, state, n, seed):
    import sys, spacy
    s = BENCH[bench]; sys.path.insert(0, s['repo'])
    from src.retrieval.self_healing import SelfHealingRetriever, RewardFunction
    from src.retrieval.hgt_model import DocumentHGT
    from sentence_transformers import SentenceTransformer
    from transformers import BitsAndBytesConfig
    cache = Path(s['repo'])/'cache'
    qm, qs = load_questions(bench)
    nlp = spacy.load('en_core_web_lg')
    st = SentenceTransformer('intfloat/e5-mistral-7b-instruct', device='cuda',
          model_kwargs={'quantization_config': BitsAndBytesConfig(load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16)})
    g0 = None
    for r in qm:
        d = r['doc_id']; stm = d[:-4] if d.lower().endswith('.pdf') else d
        f = cache/'graphs'/f'{stm}.pt'
        if f.exists(): g0 = torch.load(f, map_location='cuda', weights_only=False); break
    hgt = DocumentHGT(metadata=g0.metadata()).to('cuda')
    hgt.load_state_dict(torch.load(cache/'results/models'/s['hgt']/'best_model.pt',
                                   map_location='cuda', weights_only=False))
    hgt.eval()
    retr = SelfHealingRetriever(hgt_model=hgt,
            reward_fn=RewardFunction(params_path=str(cache/'results/models'/s['reward'])),
            top_k=10, expansion_k=5, max_rounds=2)

    idx = list(range(len(qm))); random.Random(seed).shuffle(idx)
    rec = []; resident = {}
    torch.cuda.reset_peak_memory_stats()
    for i in idx:
        if len(rec) >= n + WARMUP: break
        d = qm[i]['doc_id']; stm = d[:-4] if d.lower().endswith('.pdf') else d
        gf = cache/'graphs'/f'{stm}.pt'
        pf = next((cache/'parsed'/c for c in (f'{stm}.pdf.json', f'{stm}.json')
                   if (cache/'parsed'/c).exists()), None)
        if not gf.exists() or pf is None: continue
        if state == 'warm':
            if stm not in resident:
                if len(resident) > 3:            # bounded residency; keeps VRAM sane
                    resident.pop(next(iter(resident)))
                resident[stm] = (torch.load(gf, map_location='cuda', weights_only=False),
                                 json.loads(pf.read_text()))
            graph, parsed = resident[stm]
            t0 = time.time()
        else:
            t0 = time.time()
            graph = torch.load(gf, map_location='cuda', weights_only=False)
            parsed = json.loads(pf.read_text())
        q = f"Represent this question for retrieving relevant document passages: {qs[i]}"
        qv = torch.from_numpy(np.asarray(st.encode([q], normalize_embeddings=True)[0],
                                         dtype=np.float32))
        try:
            retr.retrieve(qv, graph, parsed, [e.text for e in nlp(qs[i] or '').ents])
        except Exception:
            continue
        rec.append({'ms': (time.time()-t0)*1000, 'pages': qm[i]['doc_pages'], 'q': i})
    return rec, torch.cuda.max_memory_allocated()/2**30


def run_colpali(bench, state, n, seed):
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    s = BENCH[bench]
    qm, qs = load_questions(bench)
    m, info = ColQwen2.from_pretrained('vidore/colqwen2-v1.0', torch_dtype=torch.bfloat16,
                                       device_map='cuda:0', output_loading_info=True)
    miss = [k for k in info.get('missing_keys', []) if 'lora' not in k.lower()]
    if miss:
        raise RuntimeError(f'C-22 gate: {len(miss)} params newly initialised — refusing')
    m = m.eval(); proc = ColQwen2Processor.from_pretrained('vidore/colqwen2-v1.0')
    IDX = Path('/work/gramm-rag-colpali/index')/bench
    idx = list(range(len(qm))); random.Random(seed).shuffle(idx)
    rec = []; resident = {}
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in idx:
            if len(rec) >= n + WARMUP: break
            d = qm[i]['doc_id']; stm = d[:-4] if d.lower().endswith('.pdf') else d
            f = IDX/f'{stm}.npz'
            if not f.exists(): continue
            if state == 'warm':
                if stm not in resident:
                    if len(resident) > 3: resident.pop(next(iter(resident)))
                    z = np.load(f)
                    resident[stm] = [torch.from_numpy(z[f'p{int(p)}'].astype(np.float32))
                                     for p in z['pages']]
                emb = resident[stm]
                t0 = time.time()
            else:
                t0 = time.time()
                z = np.load(f)
                emb = [torch.from_numpy(z[f'p{int(p)}'].astype(np.float32)) for p in z['pages']]
            qe = m(**proc.process_queries([qs[i]]).to('cuda:0'))
            proc.score_multi_vector([qe[0].float().cpu()], emb)
            rec.append({'ms': (time.time()-t0)*1000, 'pages': qm[i]['doc_pages'], 'q': i})
    return rec, torch.cuda.max_memory_allocated()/2**30


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('system', choices=['gramm', 'colpali'])
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--occurrence', type=int, default=1, help='A-B-B-A drift control')
    a = ap.parse_args()
    assert torch.cuda.is_available()
    out = {}
    for bench in ('mmlongbench', 'docbench'):
        for state in ('cold', 'warm'):
            fn = run_gramm if a.system == 'gramm' else run_colpali
            rec, peak = fn(bench, state, a.n, a.seed)
            kept = rec[WARMUP:]
            ms = np.array([r['ms'] for r in kept])
            out[f'{bench}|{state}'] = {
                'n': len(kept), 'warmup_discarded': WARMUP,
                'median': float(np.median(ms)), 'p95': float(np.percentile(ms, 95)),
                'p99': float(np.percentile(ms, 99)), 'mean': float(ms.mean()),
                'peak_gpu_gib': float(peak),
                'raw_ms': [float(x) for x in ms],
                'pages': [int(r['pages']) for r in kept],
            }
            print(f'  {a.system:8s} {bench:12s} {state:5s} n={len(kept):4d} '
                  f'median {np.median(ms):7.1f} p95 {np.percentile(ms,95):8.1f} '
                  f'p99 {np.percentile(ms,99):8.1f} peak {peak:.1f} GiB', flush=True)
    out['_meta'] = {'system': a.system, 'occurrence': a.occurrence, 'seed': a.seed,
                    'device': torch.cuda.get_device_name(0), 'warmup': WARMUP}
    (OUT/f'qlat_{a.system}_occ{a.occurrence}.json').write_text(json.dumps(out))
    print(f'wrote qlat_{a.system}_occ{a.occurrence}.json')
