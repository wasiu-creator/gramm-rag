# Stratified report — every arm × corpus × metric × stratum

**Rendered from the result JSONs by `reports/build_reports.py`. Do not edit.** Every cell
is read from an artefact at render time; `build_reports.py --check` re-renders this file and
the sync refuses to ship a difference. Cells that carry no measurement print
`not measured (x)` and the reason is printed under the table.

Localisation metrics differ by corpus and are **comparable within a corpus only** (C-43): ctx-hit on MMLongBench, quote-recall@5 on DocBench, page-hit@3 on MP-DocVQA.

## 0. Two things to settle before quoting the DocBench localisation column

**(i) The published DocBench contrast scores two arms on a different page set from the others.** GraMM-RAG and Flat-vector are scored on their top-5 **retrieved** pages (`metrics.quote_recall@5` in the archived run); every other arm is scored on the pages its **generator saw** (`context_pages`, page text rebuilt from the parse). Scored the second way — the only like-for-like available — the numbers move:

| arm | on retrieved top-5 (published) | on generator context (uniform) |
|---|---|---|
| GraMM-RAG | 0.4534 | **0.5096** |
| Flat-vector | 0.3959 | 0.3932 |
| LightRAG 1.5.7 | 0.5068 (context) | 0.5068 |

So LightRAG − GraMM is **+0.0534 as published** (S1, McNemar p=0.0013, ledger S-13) and **-0.0027 [-0.0342, +0.0288], CI includes zero** scored uniformly (`scores/cell_table.json`). Both tables below print both columns and this report does not choose between them: the DocBench reversal (S-10, S-13, `SINGLE_BENCHMARK_FINDING.md` leg 1) rests on the published pairing, so which page set is the pre-registered one is a decision for the authors, not a rendering choice.

**(ii) The control arm reads fewer pages than the arms compared with it, on all three corpora.** Flat-vector's generator sees 2 pages on MMLongBench and DocBench against 3 for GraMM-RAG (`GRAPH_PAGE_UNION`) and 3 for every external arm (the harness's 3-page budget, C-37); on MMLongBench and DocBench it also read a **8,000**-character context where the 3-page arms read 12,000. On MP-DocVQA the gap is smaller but present: flat-vector averages **1.92** pages and never reaches 3, against GraMM 2.74 and 2.45–2.52 for the graph arms. P3 stamps `TOPK_PAGES = 3` for all five arms, but its driver slices `context_pages[:3]` from each arm's own stored context, and a slice cannot add a page that was never retrieved — so the stamped k is the setting, not the realised budget. Both localisation metrics count a hit anywhere in the pages scored, so they rise with the page count: every contrast against Flat-vector — including η, which is defined against it — carries a page-budget difference as well as a retrieval difference. The `context pages / q` column in every table below is counted from the artefacts for exactly this reason.

## MMLongBench-Doc — n = 1,091 (847 answerable, 244 unanswerable; localisation scoreable 838)

| arm | context pages / q | ANLS pooled | ANLS answerable | ANLS unanswerable | ctx-hit pooled | ctx-hit (n=838) | ctx-hit unanswerable | abstention bal. acc. | decline rate, unanswerable | false-decline, answerable |
|---|---|---|---|---|---|---|---|---|---|---|
| GraMM-RAG | 3.00 [3–3] | 0.3522 | 0.2436 | 0.7295 | not measured (a) | 0.4344 | not measured (a) | 0.806 | 0.734 | 0.122 |
| Flat-vector (k=2) | 2.00 [2–2] | 0.3189 | 0.1841 | 0.7869 | not measured (a) | 0.2327 | not measured (a) | 0.812 | 0.791 | 0.166 |
| Flat-vector (k=3) | 3.00 [3–3] | 0.3229 | 0.1987 | 0.7541 | not measured (a) | 0.2780 | not measured (a) | 0.809 | 0.758 | 0.140 |
| LightRAG | 2.95 [0–3] | 0.3376 | 0.2199 | 0.7459 | not measured (a) | 0.2816 | not measured (a) | 0.814 | 0.758 | 0.130 |
| RAG-Anything | 2.99 [0–3] | 0.3850 | 0.3401 | 0.5410 | not measured (a) | 0.5465 | not measured (a) | 0.749 | 0.566 | 0.067 |
| MS GraphRAG | 2.97 [0–3] | 0.3201 | 0.1998 | 0.7377 | not measured (a) | 0.2482 | not measured (a) | 0.813 | 0.754 | 0.128 |
| ColPali | 3.00 [3–3] | 0.4683 | 0.4745 | 0.4467 | not measured (a) | 0.7995 | not measured (a) | 0.711 | 0.463 | 0.041 |
| ColPali size-m. | 5.22 [3–6] | 0.4811 | 0.5017 | 0.4098 | not measured (a) | 0.8461 | not measured (a) | 0.706 | 0.447 | 0.035 |
| Union | 5.22 [3–6] | 0.4628 | 0.4804 | 0.4016 | not measured (a) | 0.8461 | not measured (a) | 0.695 | 0.434 | 0.045 |

**(a)** not measured — localisation is undefined on unanswerable questions, and on answerable ones lacking gold evidence (C-8/C-9); the denominator is the scoreable subset in the neighbouring column

**Sources, per column.**

- *context pages / q* — derived: mean `context_pages` length in each arm's generation JSON [min–max]
- *ANLS pooled* — `scores/cell_table.json` → `mmlongbench.rows[arm]['ANLS pooled']`
- *ANLS answerable* — `scores/cell_table.json` → `['ANLS answerable']`
- *ANLS unanswerable* — `scores/cell_table.json` → `['ANLS unanswerable']`
- *ctx-hit pooled* — undefined; see reason (a)
- *ctx-hit (n=838)* — `scores/cell_table.json` → `['ctx-hit (n=838)']`; equals `ctx_hit_rate` in each arm's `scores/*_scores_mmlongbench.json`
- *ctx-hit unanswerable* — undefined; see reason (a)
- *abstention bal. acc.* — `scores/cell_table.json` → `['decline balAcc']`
- *decline rate, unanswerable* — `scores/cell_table.json` → `['decline on unanswerable']`
- *false-decline, answerable* — `scores/cell_table.json` → `['false-decline (answerable)']`

**Caveat.** 1.7% of MMLongBench questions (19 of 1,091) are answered against the wrong document, uniformly across arms: comparisons unbiased, absolute ANLS depressed by 0.0023–0.0054 per arm (`CORPUS_PARSE_INTEGRITY.md`).

## DocBench — n = 1,102 (978 answerable, 124 unanswerable; localisation scoreable 730)

| arm | context pages / q | judge acc. pooled | judge acc. answerable | judge acc. unanswerable | quote-recall pooled | quote-recall@5 — context (n=730) | quote-recall@5 — as published | quote-recall unanswerable | abstention bal. acc. | decline rate, unanswerable | false-decline, answerable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GraMM-RAG | 2.69 [1–3] | 0.5327 | 0.5092 | 0.7177 | not measured (a) | 0.5096 | 0.4534 | not measured (a) | 0.750 | 0.726 | 0.226 |
| Flat-vector (k=2) | 1.84 [1–2] | 0.4728 | 0.4458 | 0.6855 | not measured (a) | 0.3932 | 0.3959 | not measured (a) | 0.703 | 0.702 | 0.297 |
| Flat-vector (k=3) | 2.69 [1–3] | 0.5054 | 0.4867 | 0.6532 | not measured (a) | 0.4192 | 0.4192 | not measured (a) | 0.696 | 0.645 | 0.254 |
| LightRAG | 2.66 [0–3] | 0.5672 | 0.5562 | 0.6532 | not measured (a) | 0.5068 | 0.5068 | not measured (a) | 0.750 | 0.661 | 0.162 |
| RAG-Anything | 2.67 [0–3] | 0.6125 | 0.6135 | 0.6048 | not measured (a) | 0.5014 | 0.5014 | not measured (a) | 0.748 | 0.613 | 0.118 |
| MS GraphRAG | 2.63 [0–3] | 0.4864 | 0.4611 | 0.6855 | not measured (a) | 0.4384 | 0.4384 | not measured (a) | 0.727 | 0.669 | 0.215 |
| ColPali | 2.69 [1–3] | 0.7740 | 0.7945 | 0.6129 | not measured (a) | 0.5986 | 0.5986 | not measured (a) | 0.785 | 0.597 | 0.028 |
| ColPali size-m. | 4.39 [1–6] | 0.7759 | 0.7935 | 0.6371 | not measured (a) | 0.6178 | 0.6178 | not measured (a) | 0.788 | 0.597 | 0.020 |
| Union | 4.39 [1–6] | 0.7768 | 0.7965 | 0.6210 | not measured (a) | 0.6274 | 0.6274 | not measured (a) | 0.788 | 0.597 | 0.020 |

**(a)** not measured — localisation is undefined on unanswerable questions, and on answerable ones lacking gold evidence (C-8/C-9); the denominator is the scoreable subset in the neighbouring column

**Sources, per column.**

- *context pages / q* — derived: mean `context_pages` length in each arm's generation JSON [min–max]
- *judge acc. pooled* — `scores/cell_table.json` → `docbench.rows[arm]['judge accuracy pooled']`
- *judge acc. answerable* — `scores/cell_table.json` → `['judge accuracy answerable']`
- *judge acc. unanswerable* — `scores/cell_table.json` → `['judge accuracy unanswerable']`
- *quote-recall pooled* — undefined; see reason (a)
- *quote-recall@5 — context (n=730)* — `scores/cell_table.json` → `['quote-recall@5 (n=730)']` — every arm scored on the pages its generator saw
- *quote-recall@5 — as published* — GraMM/Flat: `metrics.quote_recall@5` in the archived run (top-5 **retrieved** pages); others: `quote_recall@5` in `scores/*_scores_docbench.json` (context pages)
- *quote-recall unanswerable* — undefined; see reason (a)
- *abstention bal. acc.* — `scores/cell_table.json` → `['decline balAcc']`
- *decline rate, unanswerable* — `scores/cell_table.json` → `['decline on unanswerable']`
- *false-decline, answerable* — `scores/cell_table.json` → `['false-decline (answerable)']`

**Judge.** `gpt-4-turbo`, the canonical judge. A second judge (`gpt-4o`, 7,715 calls) shifts every arm down by a mean 0.0258 with no significant pairwise reversal (`P2_RESULT.md`); the ordering here is not judge-specific.

## MP-DocVQA (P3) — n = 950 (950 answerable, 0 unanswerable)

| arm | context pages / q | ANLS pooled | ANLS answerable | ANLS unanswerable | page-hit@3 pooled | page-hit@3 (n=950) | hit@any | abstention bal. acc. | decline rate, unanswerable | false-decline, answerable |
|---|---|---|---|---|---|---|---|---|---|---|
| GraMM-RAG | 2.74 [1–3] | 0.6978 | 0.6978 | not measured (c) | 0.7189 | 0.7189 | not comparable (f) | not measured (c) | not measured (c) | not measured (d) |
| Flat-vector (k=2) | 1.92 [1–2] | 0.6663 | 0.6663 | not measured (c) | 0.6632 | 0.6632 | 0.9011 | not measured (c) | not measured (c) | not measured (d) |
| LightRAG | 2.49 [0–3] | 0.4438 | 0.4438 | not measured (c) | 0.3442 | 0.3442 | 0.7263 | not measured (c) | not measured (c) | not measured (d) |
| RAG-Anything | 2.52 [0–3] | 0.4335 | 0.4335 | not measured (c) | 0.3463 | 0.3463 | 0.7168 | not measured (c) | not measured (c) | not measured (d) |
| MS GraphRAG | 2.45 [0–3] | 0.3914 | 0.3914 | not measured (c) | 0.3263 | 0.3263 | 0.5863 | not measured (c) | not measured (c) | not measured (d) |
| ColPali | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) |
| ColPali size-m. | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) |
| Union | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) | not measured (e) |

**(c)** not measured — MP-DocVQA has no unanswerable stratum (C-61), so every abstention quantity and the unanswerable column are undefined
**(d)** not measured — false-decline on answerable questions IS measurable on MP-DocVQA; P3 did not compute it, and no artefact carries it
**(e)** not measured — the ColPali family and the Union arm are excluded from P3 by pre-registration (`P3_RESULT.md` §1)
**(f)** **a value exists but is not comparable and is withheld** — `retrieved_pages` covers GraMM's graph leg only, while its context comes from hybrid fusion with a dense leg, so the two fields measure different stages, and hit@any for GraMM would not mean what it means for the other arms (`P3_RESULT.md` §3 footnote †; the stored figure is 0.4989)

**Sources, per column.**

- *context pages / q* — derived: pages actually passed to the generator — `context_pages[:3]` from each arm's own retrieval, exactly as `generate_p3.py` slices them (**not** the stamped `TOPK_PAGES`)
- *ANLS pooled* — `tier1/p3_mpdocvqa/p3_anls.json` → `per_arm[arm].anls` (n=950 common set)
- *ANLS answerable* — same value: every MP-DocVQA question is answerable, so pooled = answerable
- *ANLS unanswerable* — no such stratum; see reason (c)
- *page-hit@3 pooled* — `p3_localisation_common950.json` → `[arm].page_hit_at_3` (defined on all 950: no unanswerable stratum)
- *page-hit@3 (n=950)* — same value as pooled
- *hit@any* — `p3_localisation_common950.json` → `[arm].hit_any`
- *abstention bal. acc.* — no unanswerable stratum; see reason (c)
- *decline rate, unanswerable* — no unanswerable stratum; see reason (c)
- *false-decline, answerable* — measurable, but not computed by P3; see reason (d)

**Caveat.** GraMM's row measures a model trained on corrupted MP-DocVQA gold-page labels (C-66, ledger S-51); the bias runs both ways — the APPA defect flattered GraMM, the training defect plausibly penalised it.

## Paired contrasts against GraMM-RAG

Each entry is *arm − GraMM-RAG*, with a 95% bootstrap interval; **\*** marks an interval excluding zero. MMLongBench and DocBench: 4,000 resamples, seed 42, one RNG per contrast (`scores/cell_table.json`). MP-DocVQA: `tier1/p3_mpdocvqa/p3_anls.json` (signs inverted here, which stores GraMM − arm).

**MMLongBench-Doc**

| arm | ANLS pooled | ANLS answerable | ANLS unanswerable | ctx-hit |
|---|---|---|---|---|
| Flat-vector (k=2) | -0.0334 [-0.0529, -0.0139]\* | -0.0595 [-0.0821, -0.0375]\* | +0.0574 [+0.0205, +0.0943]\* | -0.2017 [-0.2303, -0.1730]\* |
| Flat-vector (k=3) | -0.0293 [-0.0506, -0.0083]\* | -0.0449 [-0.0687, -0.0215]\* | +0.0246 [-0.0205, +0.0656] | -0.1563 [-0.1862, -0.1265]\* |
| LightRAG | -0.0147 [-0.0428, +0.0138] | -0.0236 [-0.0560, +0.0079] | +0.0164 [-0.0492, +0.0820] | -0.1527 [-0.1969, -0.1110]\* |
| RAG-Anything | +0.0328 [+0.0024, +0.0642]\* | +0.0965 [+0.0633, +0.1291]\* | -0.1885 [-0.2541, -0.1189]\* | +0.1122 [+0.0680, +0.1551]\* |
| MS GraphRAG | -0.0321 [-0.0604, -0.0031]\* | -0.0437 [-0.0743, -0.0112]\* | +0.0082 [-0.0533, +0.0738] | -0.1862 [-0.2279, -0.1432]\* |
| ColPali | +0.1160 [+0.0852, +0.1499]\* | +0.2309 [+0.1975, +0.2634]\* | -0.2828 [-0.3484, -0.2172]\* | +0.3652 [+0.3270, +0.4021]\* |
| ColPali size-m. | +0.1289 [+0.0968, +0.1626]\* | +0.2581 [+0.2252, +0.2903]\* | -0.3197 [-0.3893, -0.2500]\* | +0.4117 [+0.3747, +0.4475]\* |
| Union | +0.1105 [+0.0795, +0.1436]\* | +0.2368 [+0.2027, +0.2694]\* | -0.3279 [-0.3934, -0.2623]\* | +0.4117 [+0.3783, +0.4451]\* |



**DocBench**

| arm | judge accuracy pooled | judge accuracy answerable | judge accuracy unanswerable | quote-recall@5 |
|---|---|---|---|---|
| Flat-vector (k=2) | -0.0599 [-0.0826, -0.0372]\* | -0.0634 [-0.0879, -0.0389]\* | -0.0323 [-0.0887, +0.0161] | -0.1164 [-0.1425, -0.0918]\* |
| Flat-vector (k=3) | -0.0272 [-0.0517, -0.0036]\* | -0.0225 [-0.0481, +0.0041] | -0.0645 [-0.1210, -0.0081]\* | -0.0904 [-0.1178, -0.0658]\* |
| LightRAG | +0.0345 [+0.0027, +0.0653]\* | +0.0470 [+0.0133, +0.0818]\* | -0.0645 [-0.1290, +0.0000] | -0.0027 [-0.0342, +0.0288] |
| RAG-Anything | +0.0799 [+0.0490, +0.1107]\* | +0.1043 [+0.0716, +0.1380]\* | -0.1129 [-0.1774, -0.0484]\* | -0.0082 [-0.0397, +0.0260] |
| MS GraphRAG | -0.0463 [-0.0762, -0.0163]\* | -0.0481 [-0.0798, -0.0153]\* | -0.0323 [-0.1048, +0.0403] | -0.0712 [-0.1068, -0.0370]\* |
| ColPali | +0.2414 [+0.2105, +0.2713]\* | +0.2853 [+0.2536, +0.3170]\* | -0.1048 [-0.1774, -0.0323]\* | +0.0890 [+0.0616, +0.1178]\* |
| ColPali size-m. | +0.2432 [+0.2132, +0.2731]\* | +0.2843 [+0.2526, +0.3160]\* | -0.0806 [-0.1452, -0.0161]\* | +0.1082 [+0.0808, +0.1370]\* |
| Union | +0.2441 [+0.2151, +0.2740]\* | +0.2873 [+0.2566, +0.3180]\* | -0.0968 [-0.1694, -0.0323]\* | +0.1178 [+0.0945, +0.1438]\* |

*quote-recall@5 contrast is on the uniform (generator-context) definition for both corpora.*

**MP-DocVQA (ANLS, n = 950)**

| arm | ANLS − GraMM-RAG |
|---|---|
| Flat-vector (k=2) | -0.0315 [-0.0482, -0.0147]\* |
| Flat-vector (k=3) | +0.0005 [-0.0160, +0.0167] |
| LightRAG | -0.2540 [-0.2892, -0.2196]\* |
| RAG-Anything | -0.2643 [-0.2985, -0.2304]\* |
| MS GraphRAG | -0.3064 [-0.3402, -0.2712]\* |
| ColPali | not measured (e) |
| ColPali size-m. | not measured (e) |
| Union | not measured (e) |

**(e)** not measured — the ColPali family and the Union arm are excluded from P3 by pre-registration (`P3_RESULT.md` §1)
