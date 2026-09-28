#!/usr/bin/env python3
"""
graph_provenance.py — emit the graph-provenance block that RUN_CONFIG never stamped.

The archived RUN_CONFIG carries 18 keys (MMLongBench) / 21 (DocBench) and **none of them
describes the graph**. `GRAPH_PAGE_GUIDE` / `GRAPH_PAGE_UNION` record that the graph was
consulted, not what was in it. Chapter 3 claims 8 typed edges and a trained 2-layer HGT;
before this script the archive could not confirm either was active (C-30).

Emits, per corpus:
  * graph_manifest_sha256  — hash over (doc_id, per-relation edge counts) for every graph
  * edge_type_counts       — docs>0, total, median, max per relation
  * hgt_checkpoint_sha256  — plus size, so a swapped checkpoint is detectable
  * hgt_behavioural        — APPA@10 of the checkpoint vs fresh random init

Stamp the output into RUN_CONFIG of every future graph-path run.
"""
import hashlib, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

CORPORA = {'mmlongbench': ('/work/gramm-rag-mmlongbench',
                           '/work/gramm-rag-3benchmarks/data/mmlongbench/qmeta_mmlb.json',
                           'hgt_mmlb'),
           'docbench': ('/work/gramm-rag-docbench',
                        '/work/gramm-rag-3benchmarks/data/docbench/qmeta_docbench.json',
                        'hgt_docbench')}
OUT = Path('/work/gramm-rag-colpali/provenance'); OUT.mkdir(exist_ok=True)


def build(bench):
    repo, qf, hgt = CORPORA[bench]
    qm = json.loads(Path(qf).read_text())['rows']
    gdir = Path(repo) / 'cache' / 'graphs'
    per = defaultdict(list); dw = defaultdict(int); manifest = []
    for d in sorted({r['doc_id'] for r in qm}):
        stem = d[:-4] if d.lower().endswith('.pdf') else d
        f = gdir / f'{stem}.pt'
        if not f.exists():
            continue
        g = torch.load(f, map_location='cpu', weights_only=False)
        row = {}
        for et in g.edge_types:
            c = int(g[et].edge_index.shape[1]) if hasattr(g[et], 'edge_index') else 0
            row[et[1]] = row.get(et[1], 0) + c
        for rel, c in row.items():
            per[rel].append(c)
            if c > 0:
                dw[rel] += 1
        manifest.append((stem, tuple(sorted(row.items()))))
    n = len(manifest)
    counts = {rel: {'docs_nonzero': dw[rel], 'pct_docs': round(dw[rel]/n*100, 1),
                    'total': int(np.sum(v)), 'median': int(np.median(v)),
                    'max': int(np.max(v))}
              for rel, v in ((r, np.array(x)) for r, x in per.items())}
    ck = Path(repo) / 'cache/results/models' / hgt / 'best_model.pt'
    return {
        'benchmark': bench, 'n_graphs': n,
        'graph_manifest_sha256': hashlib.sha256(repr(sorted(manifest)).encode()).hexdigest(),
        'edge_type_counts': counts,
        'n_edge_types': len(counts),
        'model_extracted_types': ['kg'],   # only kg calls an LLM; the other 7 are deterministic
        'hgt_checkpoint_sha256': hashlib.sha256(ck.read_bytes()).hexdigest() if ck.exists() else None,
        'hgt_checkpoint_bytes': ck.stat().st_size if ck.exists() else None,
        'degenerate_types': [r for r, c in counts.items()
                             if c['pct_docs'] < 50 or c['median'] == 0],
    }


if __name__ == '__main__':
    out = {b: build(b) for b in (sys.argv[1:] or list(CORPORA))}
    # behavioural result is measured separately (needs GPU + E5); recorded here by reference
    out['hgt_behavioural_mmlongbench'] = {
        'appa10_trained': 0.806, 'appa10_fresh_init': 0.448, 'delta': 0.358,
        'n': 67, 'verdict': 'PASS', 'method': 'APPA@10, checkpoint vs same-shape random init',
        'note': 'HGTConv params are lazy; weight statistics are unavailable pre-forward, '
                'so the check is necessarily behavioural (C-22).'}
    (OUT / 'graph_provenance.json').write_text(json.dumps(out, indent=1))
    for b in CORPORA:
        if b in out:
            d = out[b]
            print(f"{b}: {d['n_graphs']} graphs, {d['n_edge_types']} edge types, "
                  f"manifest {d['graph_manifest_sha256'][:16]}, "
                  f"hgt {d['hgt_checkpoint_sha256'][:16]}")
            print(f"   degenerate: {d['degenerate_types']}")
    print(f"\nwrote {OUT/'graph_provenance.json'}")
