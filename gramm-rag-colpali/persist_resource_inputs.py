#!/usr/bin/env python3
"""persist_resource_inputs.py -- write cost/resource_inputs.json, the inputs RESOURCE_PROFILE.md
needs that existed only on disk or as numbers typed into prose.

Before this file, three kinds of resource value had no artefact to cite:
  * index size on disk for GraMM and ColPali -- measured once by hand, reported as "GB" although the
    figures were GiB (4.43 / 2.83 are 2**30-byte units), and never split per corpus;
  * embedder dimensionality -- in no RUN_CONFIG; asserted in prose, never read from the stored vectors;
  * three preparation constants typed into the notebook (ColPali embed/index and render minutes, the
    MinerU parse rate), whose primary logs were not retained.
The first two are MEASURED here from the artefacts. The third cannot be re-measured, so each value is
stored with the verbatim quote it came from; build_reports.py re-finds every quote in its source at
render time and refuses to build if one has changed or vanished.

Runs in grammrag-retrieval-pinned (torch, to read .pt shapes):
  docker run --rm --user $(id -u):$(id -g) -v /media/volume/vol3/grammrag:/work \
    grammrag-retrieval-pinned:latest python /work/gramm-rag-colpali/persist_resource_inputs.py
Absence raises: an empty glob is a missing store, never a zero-size index.
"""
import collections, datetime, glob, json, statistics
from pathlib import Path

import numpy as np
import torch

W = Path("/work")
OUT = W / "gramm-rag-colpali/cost/resource_inputs.json"
CORPORA = ("mmlongbench", "docbench")


def tree_bytes(d):
    files = [f for f in Path(d).rglob("*") if f.is_file()]
    if not files:
        raise SystemExit(f"FATAL: {d} holds no files -- store missing, refusing to record a zero")
    return sum(f.stat().st_size for f in files), len(files)


out = {"measured_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
       "producer": "gramm-rag-colpali/persist_resource_inputs.py",
       "units": "bytes; GiB = 2**30 bytes", "index_bytes": {}, "embedding_census": {}, "corpus": {}}

for c in CORPORA:
    g_b, g_n = tree_bytes(W / f"gramm-rag-{c}/cache/graphs")
    e_b, e_n = tree_bytes(W / f"gramm-rag-{c}/cache/embeddings")
    p_b, p_n = tree_bytes(W / f"gramm-rag-colpali/index/{c}")
    out["index_bytes"][f"gramm|{c}"] = dict(graphs=g_b, graphs_files=g_n, embeddings=e_b,
                                            embeddings_files=e_n, total=g_b + e_b)
    out["index_bytes"][f"colpali|{c}"] = dict(index=p_b, index_files=p_n, total=p_b)

    # GraMM: every per-document tensor, not a sample -- D-32 was ONE document in a different space.
    cen = {}
    for kind in ("text_raw", "text", "img"):
        fs = sorted(glob.glob(str(W / f"gramm-rag-{c}/cache/embeddings/*_{kind}.pt")))
        if not fs:
            raise SystemExit(f"FATAL: no *_{kind}.pt under gramm-rag-{c}/cache/embeddings")
        seen = collections.Counter()
        for f in fs:
            t = torch.load(f, map_location="cpu", weights_only=False)
            seen[(int(t.shape[-1]), str(t.dtype).replace("torch.", ""))] += 1
        if len(seen) != 1:
            raise SystemExit(f"FATAL: gramm/{c}/{kind} has mixed dims {dict(seen)} -- a D-32-class fallback")
        (dim, dt), n = next(iter(seen.items()))
        cen[kind] = dict(dim=dim, dtype=dt, files=n)
    out["embedding_census"][f"gramm|{c}"] = cen

    # ColPali: every page matrix in every document store.
    dims, dts, toks, npz = collections.Counter(), collections.Counter(), [], 0
    for f in sorted(glob.glob(str(W / f"gramm-rag-colpali/index/{c}/*.npz"))):
        z = np.load(f); npz += 1
        for k in z.files:
            if k == "pages":
                continue
            a = z[k]; dims[int(a.shape[-1])] += 1; dts[str(a.dtype)] += 1; toks.append(int(a.shape[0]))
    if npz == 0 or len(dims) != 1 or len(dts) != 1:
        raise SystemExit(f"FATAL: colpali/{c}: files={npz} dims={dict(dims)} dtypes={dict(dts)}")
    out["embedding_census"][f"colpali|{c}"] = dict(
        dim=next(iter(dims)), dtype=next(iter(dts)), files=npz, page_matrices=len(toks),
        tokens_per_page_median=statistics.median(toks), tokens_per_page_min=min(toks),
        tokens_per_page_max=max(toks))

    # Corpus size, from the same parse every GPU stage consumed -- the per-corpus split of the
    # corpus-level rates (parse, E5) is allocated on these, and says so in the report.
    pages = elems = docs = 0
    for f in sorted(glob.glob(str(W / f"gramm-rag-{c}/cache/parsed/*.json"))):
        d = json.loads(Path(f).read_text()); docs += 1
        el = d.get("elements")
        if el is None:
            raise SystemExit(f"FATAL: {f} has no 'elements' key")
        elems += len(el)
        pn = set()
        for e in el:
            try:
                pn.add(int(e.get("page_no")))
            except (TypeError, ValueError):
                pass
        pages += len(pn)
    if docs == 0:
        raise SystemExit(f"FATAL: no parsed documents for {c}")
    out["corpus"][c] = dict(docs=docs, pages_with_elements=pages, elements=elems)

# MP-DocVQA was not part of the resource campaign (no cost-gate cell), but its EMBEDDER matters:
# its 927 documents are embedded with E5-large-v2 (1024-d), not the 4096-d E5-Mistral-7B the other
# two corpora use, because gen_notebook.py asks for 4-bit E5-Mistral only when DEVICE == 'cuda'
# and that corpus was embedded on CPU. Nothing in any RUN_CONFIG records which encoder built an
# index, so it is measured here from the stored tensors.
cen = {}
for kind in ("text_raw", "text", "img"):
    fs = sorted(glob.glob(str(W / f"gramm-rag-mpdocvqa/cache/embeddings/*_{kind}.pt")))
    if not fs:
        raise SystemExit(f"FATAL: no *_{kind}.pt under gramm-rag-mpdocvqa/cache/embeddings")
    seen = collections.Counter()
    for f in fs:
        t = torch.load(f, map_location="cpu", weights_only=False)
        seen[(int(t.shape[-1]), str(t.dtype).replace("torch.", ""))] += 1
    if len(seen) != 1:
        raise SystemExit(f"FATAL: mpdocvqa/{kind} has mixed dims {dict(seen)}")
    (dim, dt), n = next(iter(seen.items()))
    cen[kind] = dict(dim=dim, dtype=dt, files=n)
out["embedding_census"]["gramm|mpdocvqa"] = cen
g_b, g_n = tree_bytes(W / "gramm-rag-mpdocvqa/cache/graphs")
e_b, e_n = tree_bytes(W / "gramm-rag-mpdocvqa/cache/embeddings")
out["index_bytes"]["gramm|mpdocvqa"] = dict(graphs=g_b, graphs_files=g_n, embeddings=e_b,
                                            embeddings_files=e_n, total=g_b + e_b)

# API-mediated arms: embedder width from EVERY store's vector file header, model from its config.
GB = W / "gramm-rag-graphbracket/tier1"
for arm, pat in (("lightrag", "lightrag_{c}/stores/*/vdb_chunks.json"),
                 ("s2", "s2_raganything/stores/{c}__*/vdb_chunks.json")):
    for c in CORPORA:
        fs = sorted(glob.glob(str(GB / pat.format(c=c))))
        if not fs:
            raise SystemExit(f"FATAL: no vector stores for {arm}/{c}")
        seen = collections.Counter()
        for f in fs:
            with open(f) as fh:
                head = fh.read(64)
            key = '"embedding_dim": '
            if key not in head:
                raise SystemExit(f"FATAL: {f} header carries no embedding_dim: {head!r}")
            seen[int(head.split(key)[1].split(",")[0])] += 1
        if len(seen) != 1:
            raise SystemExit(f"FATAL: {arm}/{c} mixed embedding dims {dict(seen)}")
        out["embedding_census"][f"{arm}|{c}"] = dict(dim=next(iter(seen)), stores=len(fs),
                                                     basis="vdb_chunks.json embedding_dim, every store")
for c in CORPORA:
    fs = sorted(glob.glob(str(GB / f"s5_graphrag/stores/{c}__*/settings.yaml")))
    models = collections.Counter()
    for f in fs:
        txt = Path(f).read_text()
        blk = txt.split("default_embedding_model:")[1].split("\n  default_")[0] if "default_embedding_model:" in txt else ""
        m = [ln.split("model:")[1].strip() for ln in blk.splitlines() if ln.strip().startswith("model:")]
        if len(m) != 1:
            raise SystemExit(f"FATAL: {f}: cannot read the embedding model ({m})")
        models[m[0]] += 1
    if not fs or len(models) != 1:
        raise SystemExit(f"FATAL: s5/{c}: settings={len(fs)} models={dict(models)}")
    out["embedding_census"][f"s5|{c}"] = dict(model=next(iter(models)), stores=len(fs), dim=None,
                                             basis="settings.yaml default_embedding_model, every store; "
                                                   "stored vector width not read (LanceDB)")

# Values whose primary record was not retained. Stored WITH the quote they were read from.
out["transcribed"] = [
    dict(key="mineru_parse_rate", seconds=141, pages=196,
         source="gramm-rag-mmlongbench/REVISION_LOG.md", quote="196-page document in 141 s",
         note="one document, A100; the notebook and SECTION_4_13 use the rounded 0.7 s/page"),
    dict(key="colpali_embed_index_minutes", minutes=[116.8, 262.5],
         source="GraphRAG_baseline/SECTION_4_13_REBUILD.md", quote="container log (116.8 + 262.5 min)",
         note="container log not retained; which term is which corpus is not recorded"),
    dict(key="colpali_render_minutes", minutes=[10.2, 17.7],
         source="gramm-rag-colpali/gen_phaseC_notebook.py", quote="(10.2+17.7)/60",
         note="CPU stage; container log not retained"),
]
OUT.write_text(json.dumps(out, indent=1))
print(f"wrote {OUT}")
for k, v in out["index_bytes"].items():
    print(f"  {k:22s} {v['total'] / 2**30:.4f} GiB")
for k, v in out["embedding_census"].items():
    print(f"  {k:22s} {v}")
print(f"  corpus {out['corpus']}")
