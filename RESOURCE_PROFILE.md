# Resource profile — preparation, index size, serving, embedders

**Rendered from the result JSONs by `reports/build_reports.py`. Do not edit.** `build_reports.py --check` re-renders it and the sync refuses to ship a difference.

Sizes are **GiB** (2³⁰ bytes), measured from the stores on vol3 by `gramm-rag-colpali/persist_resource_inputs.py` and recorded in `cost/resource_inputs.json`. `COST_GATE_RESULT.md` and `SECTION_4_13_REBUILD.md` print the same figures labelled GB; the values are GiB and the 1.57× ratio is unaffected.

Corpus sizes used for every per-corpus allocation: MMLongBench 135 docs / 6,522 pages / 50,751 parsed elements; DocBench 229 docs / 14,008 pages / 120,196 elements. Their totals reproduce `e5_10pct.json` exactly (20,530 pages, 170,947 elements).

## 1. Preparation

Stages measured per corpus are marked *measured*; corpus-level rates are **allocated** on pages, elements, images or documents, and say so. Allocation is arithmetic on a measured total, not a second measurement.

| arm | stage | MMLongBench | DocBench | total | basis |
|---|---|---|---|---|---|
| GraMM-RAG | MinerU parse (A100) | 1.30 h | 2.80 h | 4.10 h | allocated on pages; rate 0.7194 s/page from one document (141 s / 196 pages, `REVISION_LOG.md` R-10) |
| GraMM-RAG | E5 node embedding (A100) | 0.20 h | 0.47 h | 0.67 h | allocated on parsed elements; 14.198 ms/element (10% sample, `e5_10pct.json`) |
| GraMM-RAG | SigLIP image embedding (A100) | 0.02 h | 0.04 h | 0.06 h | allocated on image elements; 14.70 ms/image (400-crop sample, `b1_siglip_edges.json`) |
| GraMM-RAG | HGT training, 50 epochs (A100) | 2.20 h | 2.44 h | 4.64 h | **measured per corpus** (250 timed steps each, `hgt_training.json`) |
| GraMM-RAG | edge construction, 7 deterministic types (CPU) | 0.35 h | 0.59 h | 0.95 h | allocated on documents; 9.35 s/doc (20-doc sample) |
| GraMM-RAG | model-extracted edges, `kg` (API) | not measured (g) | not measured (g) | not measured (g) | — |
| ColPali | embedding + index build (A100) | not measured (h) | not measured (h) | 6.32 h | transcribed: "container log (116.8 + 262.5 min)" |
| ColPali | page rendering (CPU) | not measured (h) | not measured (h) | 0.46 h | transcribed: `(10.2+17.7)/60` |
| LightRAG 1.5.7 | indexing (API-mediated) | **0 A100-h** | **0 A100-h** | **0 A100-h** | `_resource_cells/lightrag_*_occ1.json` → `prep_a100_hours`: "0.0 -- indexing for this arm is API-mediated; it has no GPU stage" |
| LightRAG 1.5.7 | indexing API spend | not measured (i) | not measured (i) | not measured (i) | — |
| RAG-Anything 1.2.8 | indexing (API-mediated) | **0 A100-h** | **0 A100-h** | **0 A100-h** | `_resource_cells/s2_*_occ1.json` → `prep_a100_hours`: "0.0 -- indexing for this arm is API-mediated; it has no GPU stage" |
| RAG-Anything 1.2.8 | indexing API spend | not measured (i) | not measured (i) | not measured (i) | — |
| MS GraphRAG 2.7.2 | indexing (API-mediated) | **0 A100-h** | **0 A100-h** | **0 A100-h** | `_resource_cells/s5_*_occ1.json` → `prep_a100_hours`: "0.0 -- indexing for this arm is API-mediated; it has no GPU stage" |
| MS GraphRAG 2.7.2 | indexing API spend | not measured (i) | not measured (i) | not measured (i) | — |
| Flat-vector | — | not measured (j) | not measured (j) | not measured (j) | — |
| ColPali size-matched | — | not measured (j) | not measured (j) | not measured (j) | — |
| Union(CP∪GraMM) | — | not measured (j) | not measured (j) | not measured (j) | — |

**(g)** not measured — the stage's status in the shipped runs is **unverified**: no KG or edge flag appears in the archived `RUN_CONFIG` for either benchmark (`SECTION_4_13_REBUILD.md` §1, an R-42-class stamping gap). Its cost is *order $0.1–0.2* for 364 documents.
**(h)** not measured per corpus — the record is two container-log terms (116.8 + 262.5 min) that are not attributed to corpora; the log was not retained, so the split cannot be recovered.
**(i)** not measured here — indexing spend is spread across several logs per arm (MS GraphRAG alone has pilot, full and two expansion logs). Consolidating them is a measurement, not a rendering: see `tier1/*/index_log*.json`, `S2_INDEXING_RESULT.md`, `GATE_G_RESULT.md`.
**(j)** not measured — the resource campaign (cost gate; work order B) covered GraMM-RAG, ColPali and the three API-mediated arms. Flat-vector, ColPali size-matched and Union have no resource cell of their own.

**GraMM-RAG total: 9.48 A100-h** (+0.95 CPU-h), against **ColPali 6.32 A100-h** (+0.46 CPU-h) — a ratio of **1.50×** on GPU hours. The pre-registered cost-gate falsifier fires either way (GraMM > ColPali).

**Two published totals differ from this one, both explained:**

- `SECTION_4_13_REBUILD.md` and `COST_GATE_RESULT.md` print **9.36 A100-h**. They apply the parse rate rounded to 0.7 s/page; the measurement is 141 s / 196 pages = 0.7194 s/page, worth 4.10 h rather than 3.99 h.
- The notebook's `pc_fig4` prints **9.31 A100-h** and labels SigLIP and edge construction "still unmeasured". Both have since been measured (`b1_siglip_edges.json`); the panel omits the SigLIP stage (0.06 h) and its label is stale.

## 2. Index size on disk

| arm | MMLongBench | DocBench | total | components |
|---|---|---|---|---|
| GraMM-RAG | 1.153 GiB | 3.286 GiB | **4.439 GiB** | graphs 3.083 + node embeddings 1.356 (364 graph files, 1092 tensor files) |
| ColPali | 0.903 GiB | 1.932 GiB | **2.834 GiB** | multi-vector page store, 364 `.npz` files |
| LightRAG 1.5.7 | 12.004 GiB | 2.679 GiB | **14.684 GiB** | `package_precomputed.json` → `index_gib`, prefix-filtered per corpus (C-62) |
| RAG-Anything 1.2.8 | 2.318 GiB | 4.425 GiB | **6.744 GiB** | `package_precomputed.json` → `index_gib`, prefix-filtered per corpus (C-62) |
| MS GraphRAG 2.7.2 | 3.597 GiB | 1.272 GiB | **4.868 GiB** | `package_precomputed.json` → `index_gib`, prefix-filtered per corpus (C-62) |
| Flat-vector | not measured (j) | not measured (j) | not measured (j) | — |
| ColPali size-matched | not measured (j) | not measured (j) | not measured (j) | — |
| Union(CP∪GraMM) | not measured (j) | not measured (j) | not measured (j) | — |

**(j)** as above — no resource cell was measured for these arms.

The per-corpus figures for RAG-Anything and MS GraphRAG come from the prefix-filtered precomputation, **not** from `index_gib_whole_corpus` in the cell files, which is the two-corpus total that C-62 corrected (both arms keep both corpora in one directory). C-62's table records MS GraphRAG as 3.62 / 1.29 GiB where the shipped artefact says 3.5967 / 1.2717; the precomputation counts prefix-matched store directories only, and it is the value used here.

## 3. Query latency and peak query memory

**Cold** = the document's index or graph is opened inside the timed region (multi-tenant, cache miss); **warm** = already resident. All retrieval/context only, no generation, same A100 host. Two harnesses, so the two blocks are not interchangeable: GraMM and ColPali were timed by the cost gate (n = 1,000 per cell, A-B-B-A, 20 warm-ups discarded); the API arms by work order B (n = 500, two occurrences, 20 warm-ups discarded), and their latency includes an embedding-API round trip.

| arm | corpus | state | n | median | p95 | p99 | p95/median [95% CI] | peak query GPU | drift |
|---|---|---|---|---|---|---|---|---|---|
| GraMM-RAG | mmlongbench | cold | 1,000 | 88.2 ms | 102.0 ms | 133.3 ms | 1.16 [1.15, 1.18] | 6.73 GiB | 0.58% |
| GraMM-RAG | mmlongbench | warm | 1,000 | 84.8 ms | 95.3 ms | 112.1 ms | 1.12 [1.11, 1.15] | 8.60 GiB | 0.19% |
| GraMM-RAG | docbench | cold | 1,000 | 87.7 ms | 196.5 ms | 340.0 ms | 2.24 [1.78, 2.65] | 20.46 GiB | 0.10% |
| GraMM-RAG | docbench | warm | 1,000 | 84.0 ms | 139.2 ms | 240.9 ms | 1.66 [1.50, 2.04] | 20.50 GiB | 0.06% |
| ColPali | mmlongbench | cold | 1,000 | 113.6 ms | 323.9 ms | 438.2 ms | 2.85 [2.31, 3.14] | 4.28 GiB | 1.78% |
| ColPali | mmlongbench | warm | 1,000 | 66.3 ms | 90.0 ms | 109.0 ms | 1.36 [1.30, 1.40] | 4.28 GiB | 1.33% |
| ColPali | docbench | cold | 1,000 | 92.0 ms | 575.1 ms | 1032.5 ms | 6.25 [5.05, 6.66] | 4.28 GiB | 7.14% |
| ColPali | docbench | warm | 1,000 | 62.2 ms | 109.0 ms | 142.9 ms | 1.75 [1.62, 1.79] | 4.28 GiB | 1.17% |
| LightRAG 1.5.7 | mmlongbench | cold | 500 | 1811 ms | 2168 ms | 2568 ms | 1.20 [1.17, 1.24] | **0 (m)** | +94 ms |
| LightRAG 1.5.7 | mmlongbench | warm | 500 | 246 ms | 488 ms | 2243 ms | 1.98 [1.88, 2.21] | **0 (m)** | -5 ms |
| LightRAG 1.5.7 | docbench | cold | 500 | 346 ms | 1161 ms | 1412 ms | 3.35 [2.97, 3.56] | **0 (m)** | +39 ms |
| LightRAG 1.5.7 | docbench | warm | 500 | 248 ms | 391 ms | 526 ms | 1.58 [1.52, 1.63] | **0 (m)** | +19 ms |
| RAG-Anything 1.2.8 | mmlongbench | cold | 500 | 402 ms | 1038 ms | 1754 ms | 2.58 [2.31, 2.75] | **0 (m)** | +13 ms |
| RAG-Anything 1.2.8 | mmlongbench | warm | 500 | 295 ms | 528 ms | 742 ms | 1.79 [1.66, 1.94] | **0 (m)** | -5 ms |
| RAG-Anything 1.2.8 | docbench | cold | 500 | 370 ms | 1588 ms | 3532 ms | 4.30 [3.84, 5.01] | **0 (m)** | +4 ms |
| RAG-Anything 1.2.8 | docbench | warm | 500 | 250 ms | 454 ms | 648 ms | 1.81 [1.68, 1.90] | **0 (m)** | +25 ms |
| MS GraphRAG 2.7.2 | mmlongbench | cold | 500 | 268 ms | 379 ms | not measured (k) | 1.42 (l) | **0 (m)** | -23 ms |
| MS GraphRAG 2.7.2 | mmlongbench | warm | 500 | 240 ms | 368 ms | not measured (k) | 1.53 (l) | **0 (m)** | -81 ms |
| MS GraphRAG 2.7.2 | docbench | cold | 500 | 274 ms | 397 ms | not measured (k) | 1.45 (l) | **0 (m)** | +47 ms |
| MS GraphRAG 2.7.2 | docbench | warm | 500 | 252 ms | 393 ms | not measured (k) | 1.56 (l) | **0 (m)** | -66 ms |
| Flat-vector | — | — | — | not measured (j) | not measured (j) | not measured (j) | not measured (j) | not measured (j) | — |
| ColPali size-matched | — | — | — | not measured (j) | not measured (j) | not measured (j) | not measured (j) | not measured (j) | — |
| Union(CP∪GraMM) | — | — | — | not measured (j) | not measured (j) | not measured (j) | not measured (j) | not measured (j) | — |

**20 measured latency cells**: 2 GPU arms × 2 corpora × cold/warm (8) plus 3 API arms × 2 corpora × cold/warm (12). The API block is backed by 24 occurrence files (two palindrome occurrences per cell; occurrence 1 is reported, occurrence 2 is the drift check) plus 4 MS GraphRAG decompositions -- the 28 files under `tier1/_resource_cells/`.

**(k)** not measured — the B′ decomposition that separates MS GraphRAG's retrieval from its generation records the context-only median and p95 only; no context-only p99 exists.
**(l)** derived, not measured — p95/median of the two context-only values; no bootstrap interval was computed for the decomposed series. MS GraphRAG's `api.local_search` always generates, so its rows are context-only and its tail is an upper bound.
**(m)** structurally zero, not merely observed: "container started without --gpus and exposes no /dev/nvidia* device, so this arm cannot allocate GPU memory at query time -- structurally 0, not merely observed 0"
**(j)** no resource cell was measured for these arms.

Drift is occurrence 2 − occurrence 1 for the API arms, and the cost gate's own drift percentage for GraMM and ColPali. Peak query GPU for GraMM and ColPali is the per-cell figure from the same run.

## 4. Embedder and dimensionality

Widths are read from the stored vectors themselves — every file, not a sample — because no `RUN_CONFIG` stamps them and D-32 was a single document silently embedded in a different space.

| arm | stage | model | stored width | dtype | basis |
|---|---|---|---|---|---|
| GraMM-RAG | text, raw | `intfloat/e5-mistral-7b-instruct` | 4096 | float16 | every tensor: 135 + 229 files, one dimension |
| GraMM-RAG | text, projected node vector | linear projection to a common space | 256 | float32 | every tensor: 135 + 229 files, one dimension |
| GraMM-RAG | image, projected node vector | SigLIP-SO400M → linear projection | 256 | float32 | every tensor: 135 + 229 files, one dimension |
| GraMM-RAG | image, raw | SigLIP-SO400M | not measured (n) | — | only the projection is stored |
| ColPali | page multi-vector | `vidore/colqwen2-v1.0` | 128 per token | float16 | every page matrix: 20,530 pages, median 755 tokens/page |
| LightRAG 1.5.7 | text chunk | `text-embedding-3-small` | 1536 | — | vdb_chunks.json embedding_dim, every store (135 + 229 stores) |
| RAG-Anything 1.2.8 | text chunk | `text-embedding-3-small` | 1536 | — | vdb_chunks.json embedding_dim, every store (135 + 229 stores) |
| MS GraphRAG 2.7.2 | text unit | `text-embedding-3-small` | not measured (o) | — | settings.yaml default_embedding_model, every store; stored vector width not read (LanceDB) |

**(n)** not measured — SigLIP's raw output is projected before storage; only the 256-d projection is on disk.
**(o)** not measured — the stored width was not read back from the LanceDB store; the model is taken from every store's `settings.yaml`.

GraMM's text encoder is the 4,096-d E5-Mistral-7B in every one of the 364 documents: none fell back to the 1,024-d E5-large-v2, so the D-32 fallback did not recur on these corpora.
