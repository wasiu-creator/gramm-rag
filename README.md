# GraMM-RAG — public reproducibility release

**Verify first:** `./verify_coldstart.sh` regenerates the evidence notebook from this directory alone
and compares every cell's output with the notebook shipped here. A pass means **52 cells, 0 errors,
0 differing outputs**. It compares outputs rather than counting errors on purpose: an early internal
run printed 0 errors while three cells were wrong, one of them reversing a pre-registered verdict.

Requires Docker. Build the analysis image once:
`docker build -t grammrag-analysis -f _analysis_env/Dockerfile _analysis_env`

## Contents

| area | what | path |
|---|---|---|
| **system** | GraMM-RAG: parsing, graph construction, HGT retrieval, router, generation, metrics | `gramm-rag-{mmlongbench,docbench,mpdocvqa}/src/`, each repo's `gen_notebook.py` |
| **harness** | comparator adapters (LightRAG, RAG-Anything, MS GraphRAG, ColPali), indexing, retrieval, generation, scoring | `gramm-rag-graphbracket/*.py`, `gramm-rag-colpali/*.py` |
| **environments** | Dockerfiles, including pinned variants; `requirements.txt` per benchmark repo | `_analysis_env/`, `gramm-rag-*/Dockerfile` |
| **graph-arm pins** | frozen lockfiles for the four comparator venvs | `gramm-rag-graphbracket/env_locks/` |
| **per-question outputs** | retrievals, generations, scores and per-question results for every arm | `gramm-rag-colpali/{scores,generation,retrieval,cost,provenance}/`, `gramm-rag-graphbracket/tier1/`, `gramm-rag-3benchmarks/data/` |
| **evidence** | executed notebook and its generator; all figures | `gramm-rag-colpali/phaseC_evidence.ipynb`, `gen_phaseC_notebook.py`, `GraphRAG_baseline/figures/` |

**Version pins that change results.** RAG-Anything 1.2.8 requires **lightrag-hku 1.4.9.2**; the LightRAG
arm runs **lightrag-hku 1.5.7**. They are not interchangeable: the older version raises on some queries,
and those failures were measured and excluded from the scored set (not counted as misses). MS GraphRAG
is **graphrag 2.7.2** on Python 3.11. See `env_locks/`.

## Deliberately not included

| excluded | why |
|---|---|
| graph stores, embeddings, parses, page images | bulk data, tens of GB. Regenerable from the source PDFs with the code here, at API cost. |
| installed virtualenvs, vendored third-party source | the lockfiles pin them; third-party code is not ours to redistribute |
| superseded, discarded, partial and backup runs | replaced results; kept internally so every correction stays auditable |
| development transcripts, planning documents, internal reviews, evidence register | internal working material |
| API keys | none are present; the build refuses to emit credential-shaped strings |

## Dataset content — what was removed, and what remains

**DocBench corpus content is REMOVED from this package** (authors' decision, 2026-09-28). The build
strips `page_texts`, `evidences`, `questions` and `golds` from every DocBench artefact — 35 keys
across 11 files — because that material is DocBench's to redistribute, not ours. What remains for
those questions is **our own output**: predictions, judge scores, context pages and metrics.

Nothing is lost to the reader. Everything the evidence notebook computed from the removed keys is
**precomputed** into `gramm-rag-colpali/scores/package_precomputed.json` — per-question quote-recall
hits for all three arms, the scoreable index, and the structural/content split of the unanswerable
stratum — and the precompute asserts the published **0.4534 / 0.3959 / 0.5068** before it writes, so
no value in the notebook changes. `verify_coldstart.sh` passes on the stripped package: 52 cells,
0 errors, 0 differing outputs. A `_STRIPPED` note records the removal in each affected file.

**Still present, and still the authors' call:** MMLongBench-Doc and MP-DocVQA outputs carry their
questions and gold answers. Those licences have not been checked here.

Built by `build_public_release.sh` (default-deny: nothing ships unless named).
