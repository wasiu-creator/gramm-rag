#!/usr/bin/env python3
"""Measure the GraMM stages with no recorded rate, plus query-time for both systems.

Device is named for every stage. Wall-clock only; no normalisation is applied here --
everything below runs on the same A100-SXM4-40GB, so the normalisation factor is 1.0 and
is stated rather than silently assumed (Rule 10).
"""
import json, sys, time
import numpy as np, torch
from pathlib import Path

sys.path.insert(0, '/work/gramm-rag-mmlongbench')
OUT = Path('/work/gramm-rag-colpali/cost'); OUT.mkdir(exist_ok=True)
res = {'device': torch.cuda.get_device_name(0), 'normalisation_factor': 1.0}
qm = json.loads(Path('/work/gramm-rag-3benchmarks/data/mmlongbench/qmeta_mmlb.json').read_text())['rows']
src = json.loads(Path('/work/gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json').read_text())
cache = Path('/work/gramm-rag-mmlongbench/cache')

# ---- 1. E5 node-embedding throughput (GraMM stage 2, text half) ----
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig
t0 = time.time()
st = SentenceTransformer('intfloat/e5-mistral-7b-instruct', device='cuda',
        model_kwargs={'quantization_config': BitsAndBytesConfig(load_in_4bit=True,
                      bnb_4bit_compute_dtype=torch.float16)})
res['e5_model_load_s'] = time.time() - t0
docs = sorted({r['doc_id'] for r in qm})[:4]
texts, npages = [], 0
for d in docs:
    stem = d[:-4] if d.lower().endswith('.pdf') else d
    for c in (f'{stem}.pdf.json', f'{stem}.json'):
        f = cache / 'parsed' / c
        if f.exists():
            pj = json.loads(f.read_text())
            texts += [e.get('text', '') for e in pj.get('elements', []) if e.get('text')]
            npages += pj.get('num_pages') or 0
            break
texts = texts[:600]
t0 = time.time(); st.encode(texts, batch_size=8, normalize_embeddings=True); dt = time.time() - t0
res['e5_elements'] = len(texts); res['e5_seconds'] = dt
res['e5_s_per_element'] = dt / max(1, len(texts))
print(f'E5 embedding: {len(texts)} elements in {dt:.1f}s = {dt/len(texts)*1000:.1f} ms/element', flush=True)

# elements per page, to convert to a per-page rate
allel = 0; allpg = 0
for d in sorted({r['doc_id'] for r in qm}):
    stem = d[:-4] if d.lower().endswith('.pdf') else d
    for c in (f'{stem}.pdf.json', f'{stem}.json'):
        f = cache / 'parsed' / c
        if f.exists():
            pj = json.loads(f.read_text()); allel += len(pj.get('elements', []))
            allpg += pj.get('num_pages') or 0; break
res['elements_per_page'] = allel / max(1, allpg)
res['e5_s_per_page'] = res['e5_s_per_element'] * res['elements_per_page']
print(f'  {allel} elements over {allpg} pages = {res["elements_per_page"]:.1f} el/page '
      f'-> {res["e5_s_per_page"]:.2f} s/page', flush=True)

# ---- 2. Query-time latency, both systems ----
from src.retrieval.self_healing import SelfHealingRetriever, RewardFunction
from src.retrieval.hgt_model import DocumentHGT
import spacy
nlp = spacy.load('en_core_web_lg')
g0 = None
for r in qm:
    d = r['doc_id']; stem = d[:-4] if d.lower().endswith('.pdf') else d
    f = cache / 'graphs' / f'{stem}.pt'
    if f.exists(): g0 = torch.load(f, map_location='cuda', weights_only=False); break
hgt = DocumentHGT(metadata=g0.metadata()).to('cuda')
hgt.load_state_dict(torch.load(cache/'results/models/hgt_mmlb/best_model.pt',
                               map_location='cuda', weights_only=False)); hgt.eval()
rfn = RewardFunction(params_path=str(cache/'results/models/reward_mmlb.json'))
retr = SelfHealingRetriever(hgt_model=hgt, reward_fn=rfn, top_k=10, expansion_k=5, max_rounds=2)

sample = [i for i in range(len(qm))][:60]
torch.cuda.reset_peak_memory_stats()
lat_g = []
for i in sample:
    d = qm[i]['doc_id']; stem = d[:-4] if d.lower().endswith('.pdf') else d
    gf = cache/'graphs'/f'{stem}.pt'
    pf = next((cache/'parsed'/c for c in (f'{stem}.pdf.json', f'{stem}.json')
               if (cache/'parsed'/c).exists()), None)
    if not gf.exists() or pf is None: continue
    graph = torch.load(gf, map_location='cuda', weights_only=False)
    parsed = json.loads(pf.read_text())
    t0 = time.time()
    q = f"Represent this question for retrieving relevant document passages: {src[i]['question']}"
    qv = torch.from_numpy(np.asarray(st.encode([q], normalize_embeddings=True)[0], dtype=np.float32))
    ents = [e.text for e in nlp(src[i]['question'] or '').ents]
    try: retr.retrieve(qv, graph, parsed, ents)
    except Exception: continue
    lat_g.append(time.time() - t0)
res['gramm_query_ms'] = {'median': float(np.median(lat_g)*1000), 'p95': float(np.percentile(lat_g,95)*1000), 'n': len(lat_g)}
res['gramm_peak_gpu_gib'] = torch.cuda.max_memory_allocated()/2**30
print(f'GraMM query: median {res["gramm_query_ms"]["median"]:.0f} ms  p95 '
      f'{res["gramm_query_ms"]["p95"]:.0f} ms  peak {res["gramm_peak_gpu_gib"]:.1f} GiB', flush=True)
# ColPali query-time is measured separately in cost_measure_colpali.py: colpali_engine is
# pinned to a different image (C-22), so the two halves cannot share one container.
(OUT/'cost_measurements.json').write_text(json.dumps(res, indent=1))
print('\nwrote cost/cost_measurements.json')
