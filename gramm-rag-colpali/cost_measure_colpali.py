import json, time, numpy as np, torch
from pathlib import Path
from colpali_engine.models import ColQwen2, ColQwen2Processor
qm = json.loads(Path('/work/gramm-rag-3benchmarks/data/mmlongbench/qmeta_mmlb.json').read_text())['rows']
src = json.loads(Path('/work/gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json').read_text())
IDX = Path('/work/gramm-rag-colpali/index/mmlongbench')
m, info = ColQwen2.from_pretrained('vidore/colqwen2-v1.0', torch_dtype=torch.bfloat16,
                                   device_map='cuda:0', output_loading_info=True)
assert not [k for k in info.get('missing_keys', []) if 'lora' not in k.lower()], 'C-22 gate'
m = m.eval(); proc = ColQwen2Processor.from_pretrained('vidore/colqwen2-v1.0')
torch.cuda.reset_peak_memory_stats()
lat = []
with torch.no_grad():
    for i in range(60):
        d = qm[i]['doc_id']; stem = d[:-4] if d.lower().endswith('.pdf') else d
        f = IDX / f'{stem}.npz'
        if not f.exists(): continue
        t0 = time.time()
        z = np.load(f)
        emb = [torch.from_numpy(z[f'p{int(p)}'].astype(np.float32)) for p in z['pages']]
        qe = m(**proc.process_queries([src[i]['question']]).to('cuda:0'))
        proc.score_multi_vector([qe[0].float().cpu()], emb)
        lat.append(time.time() - t0)
out = {'colpali_query_ms': {'median': float(np.median(lat)*1000),
                            'p95': float(np.percentile(lat, 95)*1000), 'n': len(lat)},
       'colpali_peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30,
       'device': torch.cuda.get_device_name(0)}
print(f"ColPali query: median {out['colpali_query_ms']['median']:.0f} ms  "
      f"p95 {out['colpali_query_ms']['p95']:.0f} ms  peak {out['colpali_peak_gpu_gib']:.1f} GiB  n={len(lat)}")
Path('/work/gramm-rag-colpali/cost/colpali_query.json').write_text(json.dumps(out, indent=1))
