# GraMM-RAG — Phase C (`retriever-iso`) Revision Log

Running record of every substantive change, defect and specification correction made
while executing Phase C on JetStream2 `grammrag3` (A100-40GB, `vol3`). Same intent as
`REVISION_LOG.md` in the benchmark repos: each entry records the **defect**, its
**root cause**, the **evidence** that established it, the **fix**, and the **impact on
results** — i.e. whether any reported number would have been wrong without it.

Convention: **C-n** = Phase C revision number. Register holds 81 entries, numbered C-1…C-82; C-25 was never assigned. Phase C spans several repos
(`gramm-rag-3benchmarks`, `gramm-rag-colpali`, and the three benchmark repos read-only),
so the log lives here beside `PHASE_C_EXECUTION_BRIEF.md` rather than in any one repo.

Environment: A100-SXM4-40GB, torch 2.6.0+cu124, Docker data-root on the boot volume
(see C-4). Seed 42 throughout. Started 2026-08-21.

**Two pinned stacks, and which component each governs** (they are not interchangeable;
running either component on the other's stack is a silent-failure path):

| stack | pins | governs | why pinned |
|---|---|---|---|
| `Dockerfile.colpali_pinned` | `colpali-engine==0.3.4`, `transformers==4.46.3` | the **ColPali/ColQwen2 retrieval arm** (T-C2, union, size-matched) | under transformers 5.x the ColQwen LoRA adapters do not apply; the model loads clean and retrieves at chance (C-22/C-24) |
| `Dockerfile.gramm_retrieval_pinned` | `transformers==4.57.6`, `sentence-transformers<4` | **GraMM retrieval re-runs** (attribute decomposition) | the archived GraMM runs used 4.57.6; on 5.15.1 the integrity gate failed at max abs diff 0.239 vs 7.8e-4 (C-28 gate) |

Both are required, in separate images. The appendix must carry both.

---

## Category A — Environment / infrastructure
*(No effect on reported metrics.)*

### C-1 · Analyses run in a slim CPU image, not the benchmark image
- **Situation.** This is a fresh instance: no `gramm-rag-*` images exist, and the host
  Python has no numpy/pandas/matplotlib, so the joint notebook could not run at all.
- **Decision.** Built `grammrag-analysis` (`python:3.11-slim` + numpy/pandas/matplotlib/
  nbconvert/openai, ~600 MB) rather than the CUDA 12.4 + torch + PyG + MinerU image.
  T-C0's analyses are pure post-processing of archived JSONs — they need no GPU and no
  parser.
- **Why it matters.** Building the heavy image to run CPU post-processing would have
  written tens of GB to the boot volume for no benefit, and would have put the $0
  analyses on the GPU's critical path, defeating the plan's A-and-C concurrency.
- **Impact on results.** None. `anls_score` is pure stdlib and imports unchanged.

### C-2 · Container-written artefacts were root-owned
- **Defect.** `results/ragany/` and every notebook-written CSV/PNG were owned by root,
  because the containers ran as root against a bind mount. The first full T-C1 launch
  died with `Permission denied` writing its log beside them.
- **Evidence.** `ls -ld results/ragany` → `drwxr-xr-x root root`; run exited 1 before
  issuing a single API call.
- **Fix.** `chown -R 1001:1001` over the output tree, and every subsequent container run
  uses `--user 1001:1001 -e HOME=/tmp`.
- **Impact on results.** None — zero API calls were made on the failed attempt, and the
  T-C1 checkpoints written earlier were intact and resumed. Had this gone unnoticed it
  would have broken the vol3 mirror (C-3) later.

### C-3 · Session state was on the ephemeral boot volume
- **Defect.** The plan file and session memory live in `/home/exouser/.claude/`, which is
  on `/dev/sda1` — the boot volume, which dies with the instance. The standing project
  rule is that all work lives on `vol3`.
- **Evidence.** `lsblk`/`findmnt`: `/dev/sdb` → `/media/volume/vol3` and `/dev/sdc` →
  `/media/volume/boot-vol2-grammrag` are attached volumes and survive; `/dev/sda1` → `/`
  does not. Proof vol3 already survived a transition: this instance has **zero**
  `gramm-rag-*` images yet carries repos with June/August git history and
  `session_backup_20260812_2222/`.
- **Fix.** Mirror `~/.claude/plans/` and the memory directory to
  `vol3/grammrag/_session_state/`, refreshed at every milestone (~2 MB).
- **Impact on results.** None. Research artefacts were never at risk; Docker images are
  rebuildable from Dockerfiles on vol3.

### C-4 · Docker data-root left on the boot volume (deliberate)
- **Situation.** `/etc/docker/daemon.json` carries no `data-root`, so all images land in
  `/var/lib/docker` on `/dev/sda1` (97 GB). A previous instance was lost to a full boot
  volume, hence the standing "stop at 80%" rule.
- **Decision (user).** Do not relocate. Relocation is a config edit plus a daemon
  restart — not a reinstall or reboot — but it would stop the running Guacamole
  containers serving the desktop session.
- **Mitigation.** Monitor `df -h /` before and during every image build, with a
  self-imposed **stop at 60%** to leave margin under the 80% rule. Observed: 14% at
  start → 25% after the ColPali image. The genuinely large assets (MinerU weights
  ~16 GB, HF cache ~15 GB) are already bind-mounted to vol3 by the compose files.
- **Contingency, unused.** `/dev/sdc` is an empty, persistent 98 GB volume; data-root can
  point there if the ceiling is ever approached.

### C-5 · Separate image for PDF export
- **Situation.** `nbconvert --to pdf` needs XeLaTeX, which the slim analysis image lacks.
- **Fix.** `Dockerfile.pdf` layers pandoc + texlive-xetex on top of the analysis image as
  `grammrag-pdf`, kept separate so the everyday image stays ~600 MB.
- **Impact on results.** None, but see C-11: re-rendering the PDF is an explicit PHASE_A
  acceptance criterion that was initially skipped.

### C-6 · ColPali image avoids both the CUDA base image and MinerU
- **Decision.** `Dockerfile.colpali` = `python:3.11-slim` + torch cu124 + colpali-engine
  + PyMuPDF. No `nvidia/cuda:*-devel` base (the torch cu124 wheels bundle the CUDA
  runtime; the host driver is injected by nvidia-container-toolkit), and no magic-pdf.
- **Why.** Parsing is already done and is *reused unchanged* — that reuse is the control
  the whole arm rests on, so the parser has no place in this image. Saves ~8 GB on the
  boot volume, which matters given C-4.
- **Verified.** `torch.cuda.is_available()` true, A100-SXM4-40GB detected,
  torch 2.6.0+cu124.

---

## Category B — Analysis defects caught before publication
*(Each of these would have produced a plausible, wrong number.)*

### C-7 · The A1 integrity check was tautological — **the worst defect this phase**
- **Defect.** The first A1 draft "verified" itself by asserting that the size-weighted
  per-stratum means reconstruct the pooled mean.
- **Root cause.** That identity holds by arithmetic no matter what the stratification
  does. Rounding aside, it can never fail. It was a check that *looked* like
  verification while verifying nothing.
- **Fix.** Replaced with three assertions against independently archived quantities:
  pooled ctx-hit **0.2327** (baseline, quoted as 0.233 in the execution brief),
  **0.4344** (GraMM, quoted as 0.434 in `SESSION_JOURNEY.md`), and a scoreable
  denominator of exactly **838 = 478 single + 360 cross**. The ratio 1.87× reproduces
  the journal's "finds the gold page 86% more often".
- **Impact on results.** None as shipped — caught on review before the section was ever
  executed. Recorded because it is the exact failure mode of D-19/D-32/D-34 reappearing
  in *verification code* rather than in a pipeline error path.

### C-8 · ctx-hit was reimplemented instead of reused, and lost the unanswerable filter
- **Defect.** The A1 draft wrote a fresh `_hit()` helper that treated a question as
  scoreable whenever it had gold pages, ignoring `is_answerable`.
- **Root cause.** The canonical `_ctx_hit` already exists at
  `gramm-rag-mmlongbench/gen_notebook.py:2638` and excludes a question when it is
  unanswerable **or** has no gold pages. Brief §4 ("reuse, do not rebuild") was not
  followed.
- **Evidence.** 7 questions are flagged unanswerable yet carry gold pages, so the
  reimplementation would have scored them and shifted the denominator off 838.
- **Fix.** Ported the canonical function verbatim, with a comment explaining that the
  denominator rule is what makes the rate reproduce the archived figure.
- **Impact on results.** None as shipped — the C-7 assertions would now catch it.

### C-9 · The same bug class recurred with quote-recall
- **Defect.** While building A3's DocBench localisation series, the scoreable set was
  filtered on "evidence string is non-empty" alone. Result: n=751 and
  quote_recall@5 = 0.3848 against an archived 0.3959.
- **Root cause.** `compute_quote_recall` in `src/evaluation/metrics.py` requires
  `_quote_tokens(ev)` non-empty **and** `is_answerable[i]`. Identical omission to C-8,
  made a second time, in a second metric.
- **Fix.** Adopted the canonical predicate. Reproduces n=730 and 0.2219/0.3178/0.3959
  (baseline) and 0.2699/0.3699/0.4534 (GraMM) exactly.
- **Impact on results.** None as shipped — caught by asserting against the archived
  aggregate rather than by inspection.
- **Lesson.** Twice in one session, the same denominator mistake slipped past reading the
  code and was caught only by reproduction assertions. Any new per-question metric must
  assert against an archived aggregate before it is trusted.

### C-10 · Figure 9's twin axis was actively misleading
- **Defect.** The A3 conversion curves plotted localisation Δ and answer Δ on twin y-axes
  with a single crimson zero line.
- **Root cause.** One reference line can only sit at the *left* axis's zero. Each panel's
  right axis has its own. On DocBench the answer-metric axis spanned ~0.05–0.13, so zero
  was **not on it at all** — the answer series appeared to touch zero at 101+ pages when
  the true value there is **+0.0472**.
- **Fix.** Rebuilt as stacked rows: localisation (line) above, answer metric (bars)
  below, each with its own axis and its own real zero. The one negative bar (MMLongBench
  101+, −0.0087) is coloured distinctly.
- **Impact on results.** No table was ever wrong. The figure told a different story from
  the tables, which for a figure destined for a paper is a defect of the same family.
- **Lesson.** Numbers were verified obsessively and no figure was *looked at* until
  prompted. Rendering correctness needs its own check.

### C-11 · Figure 8 shipped 2 panels where the deliverable specifies 3
- **Defect.** `PHASE_A_CURRENT_WORK.md` §1 specifies "3-panel figure: single-page /
  cross-page / unanswerable". The draft plotted only single-page and cross-page.
- **Why it mattered.** The omitted stratum carries the only **significant regression** in
  the study (−0.0574, CI [−0.0943, −0.0205]). The figure hid the most consequential
  finding on the page.
- **Fix.** Rebuilt as three panels, one per stratum, each annotated with Δ and CI, the
  regression styled in red.
- **Related.** The phase acceptance criterion "Joint notebook regenerated, **PDF
  re-rendered**, 0 errors" was also initially skipped; PDF and HTML are now current
  (see C-5).

### C-22 · ColQwen2 v1.0 loaded with **randomly initialised weights** — silent, and plausible
- **Defect.** `vidore/colqwen2-v1.0` under transformers 5.15.1 loads with
  `language_model.embed_tokens.weight` and `language_model.norm.weight` reported
  **MISSING** — "newly initialized because missing from the checkpoint". The token
  embedding table and final norm were therefore **random**.
- **Why this is the most dangerous defect of the phase.** Nothing raised. The model
  produced a page embedding of shape (1, 755, 128) and a MaxSim score of **8.08** — an
  entirely plausible number. Had the arm been built on it, ColPali would have retrieved
  noise, Gate C would have reported "GraMM ctx-hit >> ColPali", and that **false result
  would have confirmed the project's central claim**. This is D-32's failure mode
  (a silent representation fallback accepted by every downstream stage) transplanted into
  the competitor arm, where it would have been self-serving rather than merely wrong.
- **Root cause.** Not a version conflict in the declared sense: colpali-engine 0.3.17
  requires `transformers>=5.3.0,<6.0.0`, and 5.15.1 satisfies it. The checkpoint predates
  the Qwen2-VL parameter rename (`model.*` -> `language_model.*`), so the keys simply do
  not map. `from_pretrained` treats unmapped parameters as newly initialised rather than
  as an error.
- **Evidence.** Probed three checkpoints with `output_loading_info=True` and inspected
  `missing_keys` programmatically instead of reading stderr:

  | checkpoint | result |
  |---|---|
  | `vidore/colqwen2.5-v0.2` | **CLEAN** — 0 missing, 0 unexpected |
  | `vidore/colqwen2-v1.0` | 2 missing (embed table, final norm) |
  | `vidore/colqwen2-v1.0-hf` | 731 missing / 731 unexpected |

- **Fix.** Use **`vidore/colqwen2.5-v0.2`** (`ColQwen2_5` / `ColQwen2_5_Processor`), which
  maps cleanly under transformers 5.15.1 and is the stronger model. The embedding stage
  asserts `missing_keys == []` and aborts otherwise, and additionally runs a behavioural
  preflight (a relevant page must outscore an irrelevant one) — weight-loading cleanliness
  alone would not catch a processor/config mismatch.
- **Stamped.** `COLPALI_MODEL='vidore/colqwen2.5-v0.2'` goes into `RUN_CONFIG` per R-42.
- **Impact on results.** None — caught before any page was embedded. Recorded because the
  failure was silent, self-serving, and would not have been visible in any downstream
  metric.


### C-23 · DocBench PDFs keep their original filenames; the first render pass failed all 14,008 pages
- **Defect.** `render_pages.py` resolved a DocBench document as `data_volume/<id>/<id>.pdf`.
  The folder name is the doc_id, but the PDF inside keeps its **original** name
  (`0/P19-1598.pdf`, `1/W18-4401.pdf`). Every lookup missed.
- **Evidence.** `docbench DONE: rendered=0 skipped=0 failed=14008 in 0.0 min`, with
  `MISSING PDF: 0 / 1 / 10` in the log. MMLongBench, whose `doc_id` *does* carry `.pdf`,
  rendered 6,522/6,522 in the same run — so the failure was corpus-specific and total.
- **Root cause.** An id-convention assumption carried from one corpus to another. Same family
  as R-18/R-36/R-48, which are all doc_id-resolution defects in this codebase.
- **Fix.** Glob for the single PDF in the folder:
  `next(iter(sorted((pdf_dir/doc_id).glob('*.pdf'))), ...)`. Re-run rendered 14,008/14,008.
- **Impact on results.** None. The failure was **complete and loud** — 0 of 14,008 rendered —
  rather than partial and silent. A convention mismatch that had matched *some* files would
  have produced a partial corpus and a quietly wrong ColPali index. Cited from
  `render_pages.py` and `generate_colpali.py`.

### C-24 · ColQwen2.5-v0.2 loads with `missing_keys == []` and still retrieves at chance
- **Defect.** After C-22 rejected `colqwen2-v1.0` for randomly-initialised parameters,
  `vidore/colqwen2.5-v0.2` was adopted because it loaded **clean**: `missing_keys == []`,
  `unexpected_keys == []`. It retrieved at chance anyway.
- **Evidence.** Ranking the gold page against **every** page of its own document:
  gold-in-top1 **0.08** against a random baseline of 0.11; top5 **0.38** vs 0.40; top10
  **0.50** vs 0.69 — at or *below* chance on all three. Score spread across a 20-page
  document was **32.88–34.96**, a near-flat distribution, with the gold page ranked 12th.
- **Root cause.** The checkpoint is a **LoRA adapter** (244 MB, `adapter_config.json` →
  `base_model_name_or_path: vidore/colqwen2.5-base`). Under transformers 5.15.1 the 7 GB base
  loads correctly and the trained adapter never applies. `output_loading_info` reports on the
  base load, so the integrity gate was answering a different question than the one asked.
- **Why this is worse than C-22.** C-22 announced itself — two parameters reported MISSING.
  C-24 reports a **perfectly clean load**. A weight-integrity gate passes it. Only a
  behavioural gate, ranking gold pages against a random baseline over the full document,
  detected it — and an earlier pairwise version of that gate (gold vs one random page) also
  passed it, because pages within a document are topically similar enough that a broken model
  still wins ~50%.
- **Fix.** Adopted `colqwen2-v1.0` on the **pinned** stack (colpali-engine 0.3.4,
  transformers 4.46.3), which ranks gold-in-top1 **0.79** / top5 **0.96** against random
  0.11 / 0.40. `colSmol-256M` also loaded and behaved correctly but was rejected as a 256M
  model that would have understated the ColPali family and biased Gate C in our favour.
- **Impact on results.** None — caught before any page was embedded. Had it shipped, Gate C
  would have reported "GraMM localisation ≫ ColPali" and **confirmed the project's central
  claim from a model retrieving noise.** This is the incident behind standing rule 5 and, with
  the lazy-parameter finding of C-30, rule 5b.

### C-25 · number not assigned
Reserved during drafting and never used. Recorded here rather than renumbered, because other
documents cite this register by number and a shift would invalidate them. **There is no
C-25 incident.**

### C-26 · `ln -sf` destroyed a completed result file; recovered from the vol2 handoff
- **Defect.** While making the generator arm-aware I ran
  `ln -sf colpali_gen_mmlongbench.json generation/colpali_gen_mmlongbench.json` intending an
  alias. Both paths resolve to the same file, so `ln -sf` replaced the real 1,091-prediction
  output with a **self-referential broken symlink**. The `cp ... .bak` issued immediately
  afterwards then failed silently, because its source was by then the broken link.
- **Evidence.** `ls -l` showed
  `colpali_gen_mmlongbench.json -> colpali_gen_mmlongbench.json`; the stratified report simply
  omitted the ColPali row rather than erroring, which is how it was noticed.
- **Impact on results.** **None.** The file was recovered intact from
  `/media/volume/boot-vol2-grammrag/grammrag_handoff/`, and the restored copy recomputes
  ANLS 0.468252 against the archived 0.468252 (exact). No regeneration, no additional spend.
- **Two lessons.** (1) The review handoff, built for Cowork rather than as a backup, was the
  thing that made this a non-event -- an argument for cutting the export earlier rather than
  at the end. (2) A missing arm surfaced as a *quietly shorter table*, not an error. The
  stratified report now has no guard requiring an expected system set; adding one would have
  caught it immediately.


### C-27 · A decline matcher was reused without validating it against the data
- **Defect.** `stratified_report.py` adopted the v1 `DECLINE` regex verbatim from the review
  harness. It matched MMLongBench phrasing but missed `does not include` / `does not contain`
  -- the dominant decline phrasing on DocBench.
- **Measured impact.** DocBench decline-on-unanswerable read **0.040 / 0.056 / 0.040** where
  the true rates are **0.702 / 0.726 / 0.597**: a 17x understatement. Two downstream claims
  were built on it and both were wrong -- that DocBench abstention was negligible, and that
  its unanswerable accuracy and abstention rate were separable measurements (reported as
  32-41% agreement; the true figure is 92-94%).
- **Root cause.** Reuse without verification. The same failure class as C-8/C-9, inverted:
  there the fix was to reuse the canonical helper, here the reused artefact was itself wrong.
  A matcher is an instrument; reuse does not exempt it from calibration against real output.
- **Fix.** v2 matcher covering the auxiliary-verb forms, plus a confound check printed on
  every run so the narrow version cannot be reintroduced silently. Standing rules 6 and 7.
- **Impact on results.** No archived metric changed. Two interpretive claims were retracted
  and the "negative scaling law for retrieval-augmented abstention" was withdrawn: with the
  correct matcher the benchmarks disagree on whether stronger retrieval abstains better
  (balanced accuracy MMLongBench 0.812/0.806/0.711 vs DocBench 0.703/0.750/0.785).

### C-28 · F-3 reinterpreted — tau -> 0 was optimal, not a tuning failure
- **Prior reading.** The reward's tau tuning to 0.0 on every benchmark, with 0 refusals in
  2,193 questions, was recorded as a defect of the grid search (F-2/F-3).
- **Measured.** Scoring U directly as an answerability signal: MMLongBench AUC 0.557, best
  achievable single threshold 0.560, against the generator's own decline decision at **0.806
  on identical context**. DocBench: AUC 0.723, ceiling 0.686, generator 0.750.
- **Reading.** On MMLongBench no threshold on U beats 0.560, so refusing nothing was the
  correct response to a signal that weak -- tau -> 0 is the optimum, not a bug. The
  answerability signal demonstrably exists in the context (the generator finds it at 0.806);
  it is simply **not in U**. Since Rel/Co/Mod are all retrieval-side, the failure is of the
  attribute family, not of the threshold policy. That is the mechanistic negative result and
  it names the future-work direction: generator-side attributes.
- **On DocBench the failure is real and fixable**: signal present, a global tau recovers
  0.673 of a 0.750 ceiling, and the shipped system still refused nothing.
- **Per-stratum calibration is not supported.** 5-fold CV: DocBench identical to global
  (0.673 both); MMLongBench +0.012 by doc_type but **-0.031** by length bin -- inconsistent
  in sign, i.e. noise.


### C-29 · AUC helper mishandled ties; a 3-valued attribute read 0.231 when it is 0.484
- **Defect.** `auc()` assigned distinct ranks via `argsort` with **no mid-rank averaging**.
  Positives are concatenated first and the sort is stable, so every tie was resolved
  *against* the positive class, biasing AUC downward in proportion to how tied the variable
  is. Continuous features were unaffected; low-cardinality ones were wrecked.
- **How it surfaced.** Not by inspection -- by an impossibility. Review noted the joint CV
  logistic (0.568) scored **below the best single feature it contains** (-T at a reported
  0.769), which cannot happen for a correctly fitted model of that class.
- **Diagnostic that localised it in one step.** `AUC(x) + AUC(-x)` must equal 1.
  Measured: T 0.231+0.263 = **0.494**; C 0.386+0.250 = **0.636**; S (continuous)
  0.507+0.493 = **1.000**. The invariant holds only where there are no ties.

| feature | unique | reported | **correct** |
|---|---|---|---|
| MMLongBench C | 13 | 0.386 | **0.568** |
| MMLongBench T | 3 | 0.231 | **0.484** |
| DocBench C | 8 | 0.568 | **0.717** |
| DocBench T | 3 | 0.316 | **0.499** |

- **Effect on conclusions.** §7's central claim **stands and is better supported**. T is at
  chance, not strongly inverted, so there is no retrieval-side attribute detecting
  answertability at 0.769 that was being missed. The impossibility resolves: joint 0.568 =
  best marginal (C, 0.568) on MMLongBench, and 0.728 > 0.717 on DocBench -- joint >= best
  marginal on both, as required. U's AUC (0.556 / 0.723), the joint CV figures, and every
  selective-risk number were computed on continuous values and are **unchanged**.
- **Two hypotheses tested and rejected before the real cause was found.** Standardisation
  (features *were* standardised; refits with no penalty, C=100, and raw features all give
  0.568 to three decimals) and tie handling as tested by coarsening U to 3 values, which
  moved AUC only 0.006 -- a false negative, because the bias depends on class balance
  *within* tie groups, not on cardinality alone.
- **Fix.** `scipy.stats.rankdata` (mid-rank averaging) in every script, plus an assertion
  that `AUC(x)+AUC(-x)==1` for each attribute, printed and enforced on every run.
- **Conceptual error this caused, worth stating separately.** The write-up read "T takes 3
  distinct values, therefore near-degenerate", conflating **cardinality with
  informativeness**. Low cardinality bounds resolution, not information -- a 3-valued
  feature at a true AUC of 0.231 would be strong and inverted. That slip is plausibly why an
  apparently strong inverted predictor was filed as a degenerate one and never chased.


### C-30 · Graph provenance was unstamped; verified now, and the stop condition does NOT fire
- **Defect.** The archived `RUN_CONFIG` carries 18 keys (MMLongBench) / 21 (DocBench) and
  **none describes the graph**. `GRAPH_PAGE_GUIDE` / `GRAPH_PAGE_UNION` record that the graph
  was *consulted*, not what was in it. Chapter 3 claims 8 typed edges and a trained 2-layer
  HGT; the archive could confirm neither. R-42's stamping rule was applied to reward
  parameters and never extended to the graph itself.
- **Why it was treated as blocking.** Same shape as C-22: a component asserted to be active,
  with no artefact proving it. If the model-extracted edges were absent, Gate A -- load-bearing
  under the current framing -- would be confounded.

**Edge-type census (all 8 types present in the schema on both corpora):**

| relation | MMLB docs>0 | MMLB median | DocBench docs>0 | DocBench median |
|---|---|---|---|---|
| hierarchy | 100% | 122 | 100% | 78 |
| reading_order | 100% | 75 | 100% | 63 |
| coref | 99% | 1,870 | 98% | 1,766 |
| crosspage | 99% | 1,592 | 76% | 1,630 |
| temporal | 70% | 42 | 88% | 506 |
| **kg (model-extracted)** | **73%** | **28** | **91%** | **38** |
| spatial | 64% | 1 | **41%** | **0** |
| caption | **6%** | **0** | **3%** | **0** |

- **Correction to the review's premise.** There is **exactly one** model-extracted edge type,
  not four: only `kg` calls an LLM (`extract_triplets` -> `gpt-4o-mini`). The other seven are
  deterministic functions of the parsed elements. Verified by inspecting every builder in
  `src/graph/edges.py`.
- **Stop condition does NOT fire.** `kg` is present and non-trivial on both corpora (98/135
  and 208/229 documents; 10,298 and 22,086 edges). **Gate A is not confounded by absent
  model-extracted edges.**
- **Two degenerate types, both deterministic.** `caption` is effectively dead -- 14 edges
  total across MMLongBench and 8 across DocBench, present in 6% / 3% of documents, matching
  the earlier note that it is "sparse by construction". `spatial` is sparse on DocBench (41%
  of documents, median 0). Neither is model-extracted, so neither triggers the stop, but the
  Chapter 3 claim of 8 *contributing* edge types is not supported: **6 contribute on
  MMLongBench, 6 on DocBench**.
- **HGT checkpoint verified behaviourally, not by load-success.** `HGTConv` parameters are
  **lazy** -- they do not exist until a forward pass -- so weight statistics are unavailable
  before use, and the framework itself forces the C-22 lesson. Compared the checkpoint against
  a same-shape fresh random init on identical inputs:

  | | APPA@10 | n |
  |---|---|---|
  | trained checkpoint | **0.806** | 67 |
  | fresh random init | 0.448 | 67 |
  | delta | **+0.358** | |

  The checkpoint holds genuinely trained weights and materially drives retrieval. sha256
  `625dd7c0…` (MMLongBench, 20.5 MB) / `dc24f6e5…` (DocBench).
- **Fix, going forward.** `gramm-rag-colpali/graph_provenance.py` emits a provenance block --
  graph manifest sha256 over (doc_id, per-relation edge counts), full edge-type census, HGT
  checkpoint sha256 and size, and the behavioural result -- to be stamped into `RUN_CONFIG`
  for every future graph-path run. Current values in
  `gramm-rag-colpali/provenance/graph_provenance.json`; manifests `ec177c55…` / `6f2b8732…`.
- **Impact on results.** None retracted. Gate A stands. The Chapter 3 description of the
  architecture must be corrected from 8 contributing edge types to 6.


### C-31 · Three Chapter 3 / §4.13 claims corrected by the census and the cost gate
- **"Four model-extracted edge types" → one.** Only `kg` invokes an LLM
  (`extract_triplets` → `gpt-4o-mini`); the other seven are deterministic functions of parsed
  elements. Verified builder-by-builder in `src/graph/edges.py` (C-30).
- **"Eight typed edges" → eight defined, six contributing.** `caption` is dead (14 edges
  across MMLongBench, 8 across DocBench; 6% / 3% of documents) and `spatial` is sparse on
  DocBench (41%, median 0). A designed component that does not materialise in practice is a
  finding, not an embarrassment, and reporting it costs nothing that is still held.
- **"≈$45 API cost" → order $0.1–0.2, and mis-scoped.** That figure sits on the same
  unsourced planning list as the "~8 h HGT" estimate (`PHASE_A_CURRENT_WORK.md` §12,
  *"assemble rather than measure"*), and it is scoped to **MP-DocVQA (927 docs)**, not the
  364-document corpora under test. The project's own measurement of the same stage is
  *"38 triplets/document in 37.6 s (20 elements/doc, `gpt-4o-mini`) → ≈46 min for 73
  documents, **a few cents**"*. Scaling to 364 documents gives **order $0.1–0.2** — a ~100×
  discrepancy with the planning figure. Re-derive from the measured rate; do not cite $45.
- **Impact.** No experimental result changes. Three descriptive claims in the write-up were
  wrong, and all three traced to numbers carried forward without provenance checks.


### C-32 · An unpinned transitive dependency would have silently substituted the comparator
- **What.** `raganything==1.2.8` — the release contemporaneous with arXiv:2510.12323, whose
  published numbers (MMLongBench **0.428**, DocBench **0.634**) this arm exists to sit beside —
  declares its dependency as bare `lightrag-hku`, with no upper bound. A plain
  `pip install raganything==1.2.8` today resolves that to **lightrag-hku 1.5.7**, released
  2026-09-02: **eleven months newer** than the system that produced the published numbers.
- **Why it matters.** RAG-Anything is a thin multimodal layer; LightRAG underneath it is where
  chunking, entity extraction, gleaning and graph storage actually live. Substituting eleven
  months of LightRAG development is not a patch-level difference — it changes the system being
  compared while leaving the version string in the write-up (`raganything 1.2.8`) *correct*.
  Every artefact would have named the right package at the wrong system.
- **How it was caught.** Reading `requires_dist` from the PyPI metadata **before** installing,
  because the pilot's job is to record exact versions. It would not have announced itself: the
  install succeeds, imports succeed, and indexing produces plausible numbers.
- **Fix.** `lightrag-hku` pinned explicitly to **1.4.9.2** (2025-10-11), the release
  contemporaneous with the paper. Both pins stamped into the arm's `RUN_CONFIG` and asserted in
  the pilot log.
- **Standing rule 14** (below): when reproducing a published system, pin its *transitive*
  dependencies to the versions contemporaneous with the publication, and record the date
  evidence for each. A version pin on the named package alone is not a pin on the system.

### C-82 · The ledger carried the corrected claim in its uncorrected form, and the frontier's trend line was anchored on an arm we built

- **What (i).** After C-80 and C-81 the register, the pre-registration, the reports and the
  notebook all carried the page-matched results — and **`CLAIMS_LEDGER.md` did not**. S-04 still
  read *"at 101+ pages the answer delta turns negative"*, value *"turns negative (−0.0087)"*,
  status **stands**, no caveat, no supersede pointer: the exact claim C-81 had just overturned,
  sitting in the artefact built to stop precisely that. Found by the advisor, not by a guard.
- **The systematic version.** S-04 was one instance. Every ledger claim whose value is a contrast
  against the flat-vector control had been computed against the 2-page arm, and the C-80/C-81
  re-anchoring covered the notebook and the figures, not the claim rows. A sweep of all 64 claims
  found **nine** more: S-01, S-03, S-07, S-11, S-12, S-15, S-16, S-43, S-45.
- **Two of them changed status on the evidence, not the wording.** S-03 — *GraMM regresses
  **significantly** on unanswerable questions* — is **−0.0246 [−0.0656, +0.0205]** page-matched,
  no longer significant (now S-58). S-11's *"evidenced two independent ways"* loses one way: A3's
  101+ leg was a k=2-control contrast (C-81), leaving the between-systems leg alone.
- **Disposition.** S-04 keeps its pre-registration intact: **caveated**, standing as measured
  against the shipped configuration, with the page-budget artefact and the matched value
  (+0.0087, no arm negative) stated on its face. S-01, S-03 superseded; S-07, S-11, S-12, S-16
  caveated; S-15 and S-43 scoped (APPA is a retrieval-stage metric, so the context budget does not
  enter it; P3's granularity gap is *wider* against the matched control). New: S-58, S-59, S-60.
- **The guard that now exists.** `build_ledger.py` refuses to render a LIVE claim resting on a
  flat-vector contrast unless its face says which control produced it. A text trigger alone was
  not enough — **S-04 never names the baseline it is measured against** — so the check reads an
  explicit `CONTROL_DEPENDENT` registry as well. Both were tested by rendering a deliberately
  broken copy through `LEDGER_SOURCE`, and the guard fires on S-04 in exactly the state the
  advisor found it.
- **What (ii): the conversion frontier's trend line was anchored on a derived arm.** The dashed
  line every system is read against joined the control to the **best-localising** arm — which is
  `Union(CP∪GraMM)` or `ColPali size-matched`, both built here and shipped by nobody. The figure
  and notebook §11 now draw **two** lines: the primary to the best arm that exists outside this
  project, and a faint dotted one to the derived arm, each named in the caption.
- **The generalisable form.** **A correction is not finished when the analysis is corrected; it is
  finished when every artefact that indexes the analysis is corrected** — and the index is the
  easiest one to forget precisely because it is not where the computation lives. C-75 was this
  shape in the documents, C-81 in the derived quantities, and C-82 in the ledger itself.

### C-81 · The "101+ collapse" was a property of the control, and no arm is negative there once it is matched

- **What.** A3's length-bin finding — *GraMM's answer-quality advantage turns negative on documents
  past 100 pages (dANLS −0.009) while its localisation keeps rising* — was measured against the
  k=2 flat-vector control. Against the **page-matched** control the sign flips:

  | 101+ bin, n=115 | vs archived k=2 control | vs page-matched control |
  |---|---|---|
  | GraMM-RAG | **−0.0087** (reproduces the published −0.009) | **+0.0087** [−0.0348, +0.0522] |
  | arms with a negative point estimate | GraMM alone | **none, of eight** |

  The claim that GraMM is the only arm below zero at 101+ is **withdrawn**. It was never
  significant — its interval always spanned zero — but the point estimate, which is what the claim
  rested on, does not keep its sign.
- **What survives.** GraMM's 101+ advantage is its **smallest**: +0.0087 against +0.0233…+0.0376 in
  every other bin, about a third, while its 101+ localisation advantage stays positive (+0.0659).
  The conversion gap on long documents is a *shrinking ratio*, not a collapse into deficit. The
  vision arms' 101+ advantage still survives Bonferroni (ColPali +0.2450 [+0.0970, +0.3889]), so
  *vision retains, graph does not demonstrably retain* stands — without needing a GraMM collapse.
- **A finding about the control itself, not about any system.** At 101+ the **k=2** control
  outscores the k=3 control (+0.0174): past 100 pages the third context page costs more in
  distraction than it returns in evidence. Every other length bin runs the other way. That is a
  property of long documents and a context budget, measured on the control.
- **How it was found.** Not by re-rendering: the advisor noticed that §§11, 13 and 15 of the
  evidence notebook still computed every delta and every η against the k=2 control after C-80 had
  retired it, and asked whether §13's conclusion was a number change or a finding change. It was a
  finding change.
- **A second casualty, found the same way.** The comparator-validation number goes too: *LightRAG
  genuinely beats the flat-vector control (+0.0489 ctx-hit, p=0.014)*, used to retire the suspicion
  that the LightRAG arm was simply broken, is **+0.0036 [−0.0346, +0.0418], p=0.901** against the
  page-matched control — LightRAG is not distinguishable from a dense baseline reading the same
  three pages. The comparator remains validated by its DocBench result (+0.1110) and by C-37's
  ranking fix, but not by that number. Noted in `GRAPH_BRACKET_RESULT.md` and
  `GATE_H_COMPARATOR_SET.md`; §9.1 of the notebook now prints both contrasts side by side.
- **The generalisable form.** **When a control moves, conclusions ABOUT OTHER ARMS move with it,
  and a sign is not more robust than the baseline it is measured from.** C-75 was the same shape
  one level up: retracted numbers that were never removed. A retired control has to be chased
  through every derived quantity — per-bin deltas, ratios, and the prose that reads them — not
  only through the headline table.

### C-80 · At equal page budget the flat-vector control matches GraMM-RAG on MP-DocVQA — the published advantage there was the page budget

- **What.** The pre-registered falsifier in `FLAT_K3_PREREGISTRATION.md` (prediction 3) **FIRED on
  MP-DocVQA**. Re-running the control with three context pages instead of ~1.9, changing nothing
  else, GraMM-RAG's advantage disappears:

  | MP-DocVQA, n=950 | published (control k=2) | page-matched (control k=3) |
  |---|---|---|
  | ANLS | **+0.0315** [+0.0147, +0.0482] | **−0.0005** [−0.0167, +0.0160] |
  | page-hit@3 | **+0.0558** [+0.0400, +0.0716] | **−0.0158** [−0.0358, +0.0032] |
  | page-hit@3, answer NOT on page 0 (n=750) | +0.0480 [+0.0320, +0.0653] | **−0.0333 [−0.0547, −0.0120]** |

  The control gained **+0.0320 [+0.0171, +0.0477]** ANLS from its third page alone — almost exactly
  the +0.0315 that GraMM's published lead consisted of. On the stratum where the answer is not on
  page 0, the page-matched control is **significantly better** than GraMM at finding the page.
- **What survives.** On the other two corpora GraMM still leads the page-matched control:
  MMLongBench ANLS **+0.0293** [+0.0083, +0.0506] and ctx-hit **+0.1563** [+0.1265, +0.1862];
  DocBench judge accuracy **+0.0272** [+0.0040, +0.0520] pooled and quote-recall@5 **+0.0904**
  [+0.0660, +0.1180]. DocBench's *answerable-stratum* quality advantage does **not** survive:
  +0.0225 [−0.0040, +0.0480].
- **So the floor claim changes.** "GraMM-RAG beats the flat-vector baseline on all three benchmarks"
  is true only of a control reading fewer pages. At equal budget it holds on **two of three**, and
  on MP-DocVQA the honest statement is that GraMM and a page-matched dense baseline are
  indistinguishable.
- **η moves too**, because its denominator is a localisation difference that the page gap inflated:
  MMLongBench 0.1654 → **0.1877**, DocBench 0.5144 → **0.3011**. No η computed against the k=2
  control may be quoted (prediction 4).
- **How it was found.** Not by a guard: by counting the pages in each arm's stored context while
  building `STRATIFIED_REPORT.md` (C-77). Every arm's own numbers were correct; the *comparison*
  was between arms reading different amounts of the document.
- **The generalisable form.** **A control must be matched on everything the metric is sensitive to,
  not only on the component under test.** This project already knew that — `ColPali size-matched`
  exists solely to hold page count fixed against the Union arm — and still compared its own system
  against a control reading a third fewer pages for the entire length of the project. A retrieval
  metric that scores a hit anywhere in the context is a function of context size; holding the
  retriever fixed while varying the budget measures the budget.

### C-79 · The archive is deterministic but not reproducible: a quantized encoder defeated an exact-reproduction gate

- **What.** `FLAT_K3_PREREGISTRATION.md` gated the flat-vector k=3 re-run on reproducing each archived
  k=2 context exactly, ≥ 99.5% per corpus. Measured: **DocBench 99.909% (pass), MMLongBench 98.350%,
  MP-DocVQA 96.221%** — the gate failed on two corpora and no money was spent.
- **Where the divergence starts, measured rather than guessed.** The archived **retrieval pool**
  reproduces on only **81.118%** of MMLongBench questions, and **all 18** context mismatches sit
  inside that 19%. So the reranker is not the source; the candidate list is.
- **Deterministic ≠ reproducible.** Two runs of this environment produced **bit-identical** output
  (0 differing rows of 1,091). The instability is *between* environments, not within one. The only
  component upstream of an exact `IndexFlatIP` search that varies that way is the **4-bit quantized
  E5-Mistral-7B query encoder** — bitsandbytes kernels differ by build — with the FAISS build a
  second candidate. The original run's library versions were never stamped (an R-42-class gap), so
  the comparison cannot be closed.
- **Disposition (advisor-authorised).** Gate 1 is **retired as a gate** and its three rates are
  reported as a fidelity measure. The k=3 context is built as a **superset**: the archived context
  exactly, plus the highest-ranked page the canonical reranker offers that is not already in it. The
  published pages then survive by construction. Residual risk, recorded with the result: on the
  1.7% / 0.1% / 3.8% of questions where this environment's rank-2 differs, the *added* page may
  differ from the one the original environment would have added.
- **The generalisable form.** **A quantized component makes a pipeline deterministic within an
  environment and irreproducible across environments**, and the two properties look identical from
  inside a single machine. Any quantized model in a retrieval path must have its inference stack
  (bitsandbytes, transformers, the SIMD build of the index) pinned and stamped into `RUN_CONFIG`, or
  its outputs are reproducible only as artefacts, never as computations.

### C-78 · MP-DocVQA is embedded with a different text encoder from the other two corpora, and nothing recorded it

- **What.** All **927** MP-DocVQA documents carry **1024-d E5-large-v2** node embeddings, where
  MMLongBench (135) and DocBench (229) carry **4096-d E5-Mistral-7B**. Cause:
  `load_text_model(use_4bit=(DEVICE == 'cuda'))` — the fallback is *designed* to fire when no GPU is
  visible, and that corpus was embedded on CPU. It fired silently and correctly, and was never
  recorded: no `RUN_CONFIG` on any corpus stamps which encoder built its index.
- **How it surfaced.** The D-32 dimension guard added to `flat_k3_select.py` refused to search a
  1024-d index with a 4096-d query vector. Without that guard the run would have crashed deep in
  FAISS with an opaque assertion, or — had the dimensions happened to match — returned nonsense.
- **What it does and does not touch.** Within MP-DocVQA every E5 consumer uses the same encoder, so
  the GraMM-vs-flat contrasts there are internally consistent, and P3's primary prediction
  (RAG-Anything vs LightRAG) uses neither. What it does forbid is a **cross-corpus** statement about
  GraMM's dense retrieval: its MP-DocVQA leg runs on a 1024-d encoder and its other two legs on a
  4096-d one, which is a capability difference, not a corpus difference.
- **Fix.** The query encoder now follows the index it is searching, chosen by measured dimension and
  printed. The census is persisted (`cost/resource_inputs.json`) and printed in
  `RESOURCE_PROFILE.md` §4 for all three corpora, measured from every stored tensor rather than
  asserted from a config that does not record it.

### C-77 · Two DocBench localisation numbers are not the same measurement, and the control reads fewer pages

- **What (i).** On DocBench, localisation is scored on a **different page set per arm**. GraMM-RAG and
  Flat-vector are scored on their top-5 **retrieved** page texts (`metrics.quote_recall@5` in the
  archived run); every external arm is scored on the pages its **generator saw** (`context_pages`,
  page text rebuilt from the parse — `score_colpali.py`, and the notebook's S1 cell). The published
  contrast pairs the two: LightRAG 0.5068 (context) against GraMM 0.4534 (retrieved top-5) =
  **+0.0534, McNemar p=0.0013**. Scored the same way for both arms — `scores/cell_table.json`, which
  recomputes every arm on its generator context — GraMM is **0.5096** and the contrast is
  **−0.0027 [−0.0342, +0.0288]**, not distinguishable from zero.
- **Why it hid.** Both figures are "quote-recall@5, n=730", both are correct for their own definition,
  and the metric name records the cut-off but not the input set. The like-for-like value had been
  sitting in `cell_table.json` since 2026-09-19; nothing compared the two until the stratified report
  had to print one number per cell.
- **Scope, checked rather than assumed.** MMLongBench is **unaffected**: `ctx-hit` reads
  `context_pages` for every arm, and cell_table reproduces each arm's own score file to 5e-4.
  MP-DocVQA is **also affected, and this was got wrong first**: P3 stamps `TOPK_PAGES = 3` for all
  five arms, but `generate_p3.py` slices `context_pages[:3]` from each arm's own stored context, and
  a slice cannot add a page that was never retrieved. Counted from the artefacts, flat-vector
  averages **1.92** pages and never reaches 3, against GraMM **2.74** and 2.45–2.52 for the graph
  arms (the per-arm `img/q` column of `P3_RESULT.md` §3 carried this all along). The first render of
  `STRATIFIED_REPORT.md` printed "k = 3 (harness)" for that column because it read `RUN_CONFIG`
  instead of the context lists — **C-76's own lesson, repeated inside the report built to expose
  it, within a day of writing it down.** The column is now counted from the artefacts for every
  corpus.
- **What it touches.** The DocBench reversal — ledger S-13 and S-10, the retraction of R-01, and
  **leg 1** of `SINGLE_BENCHMARK_FINDING.md`. Legs 2 (abstention τ-b = −0.828) and 3 (the C-49
  ablation) do not depend on it.
- **What (ii): the control arm reads fewer pages than the arms compared with it, on every corpus.**
  Flat-vector's generator sees **2** pages on MMLongBench and DocBench — at an **8,000**-character
  context where the 3-page arms get 12,000 — and averages **1.92** on MP-DocVQA; GraMM-RAG sees
  **3** (k=2+1 via `GRAPH_PAGE_UNION`, 2.74 on MP-DocVQA) and every external arm sees 3 (C-37's
  3-page budget). Both localisation metrics score a hit anywhere in
  the pages scored, so they rise with page count: every contrast against Flat-vector, and η — which is
  *defined* against it — carries a page-budget difference as well as a retrieval difference. The
  project already controls this confound elsewhere: **ColPali size-matched exists precisely to hold
  page count fixed** against the Union arm.
- **Status: OPEN — the authors' decision, not repaired here.** Which page set is the pre-registered
  DocBench localisation metric, and whether a k=3 flat-vector control is needed, both change published
  claims; a rendering script must not decide either.
- **What was done.** `STRATIFIED_REPORT.md` prints **both** DocBench definitions side by side, plus a
  derived `context pages / q` column for every arm on every corpus, so neither can be quoted without
  the other being visible. `build_reports.py` fails if the two definitions ever converge, which would
  make that section stale.
- **The generalisable form.** *A metric name is not a definition.* Two arms can be scored by the same
  function, at the same cut-off, on the same denominator, and still not be comparable, because the
  inputs came from different stages of the pipeline. Candidate standing rule, for adoption: a
  localisation metric is quoted with the **page set** it scored (retrieved top-k, or generator
  context), and a contrast between arms scored on different page sets is not reported as a contrast.

### C-76 · Publication figures were certified 3.5 in wide from figsize; two shipped at 3.626 in

- **What.** Six IEEE-column figures were reported as "widths ≤ 3.5 in". The number came from
  `figsize=(3.5, …)`, not from the files. With `savefig.bbox="tight"` the page grows to hold labels that
  overhang the figure: both quality radars wrote **3.626 in** PDFs, because their side-axis labels
  extend past the polar disc. Placed at `\columnwidth` they would scale to 96.5%, putting the 6.5 pt
  type floor at 6.3 pt.
- **Found** when regenerating the figures from the public subset: the PNGs were 2,175 px wide at 600 dpi.
- **Fix.** Polar axes narrowed (`subplots_adjust(left=0.19, right=0.81)`), which also moved the legend
  into the bottom axis label (a collision, seen at print size and fixed). `save()` now measures the
  **written PNG** and raises `FIG FAIL` over 3.5 in. The guard was run on the old geometry first, and
  it failed there. Result: radars 3.140 × 3.697 in, all six ≤ 3.385 in, zero Type 3 fonts. The other
  four figures are pixel-identical to their previous versions.
- **Rule 21 instance:** a guard that checks a setting instead of the artefact it controls.

### C-75 · Retracted claims were still asserted in live artefacts after the ledger existed

- **What.** Retracting a claim in the register did not remove it from the documents a writer would
  actually open. The ledger sweep checked each retracted wording for a withdrawal marker within
  12 lines, and it found:
  - the evidence notebook's closing *Summary of verdicts* stating R-01 as a standing verdict
    ("GraMM localises +0.1527 over LightRAG"), with no DocBench reversal;
  - `pc_fig4` titled with R-06 ("GraMM wins, and is far more predictable"), plotting a pre-cost-gate
    measurement;
  - `GRAPH_BRACKET_RESULT.md`, whose headline states R-01 with no notice (its only withdrawal banner
    covered R-02, 75 lines further down);
  - `HANDOFF.md`'s file index repeating R-01;
  - R-07 and R-11 unflagged at their original sites (`GATE_G_RESULT.md`, `PHASE_B_DISSERTATION.md`
    §1, `ADVANCED_GRAMMRAG_IMPLEMENTATION_PLAN.md`).
- **Also found: a superseded value carried forward.** C-66's knock-on paragraph, in this register and
  in `MPDOCVQA_GOLDPAGE_DEFECT.md`, cited GraMM's P3 page-hit@3 as **0.7127**. That is the n = 978
  value C-67 had already replaced with **0.7189** (n = 950).
- **Fix.**
  - The summary row and `pc_fig4` now state the surviving claims (S-10, S-21), with the reversal and
    cold/warm data.
  - Withdrawal banners were added at each original site. Original text is kept unedited, following the
    precedent of the R-02 banner.
  - New ledger claim **S-51** pins GraMM's P3 row to the JSON in all three documents that print it. It
    was run against the stale value and failed there.
  - R-06's index wording "GraMM wins" was too generic (it matched §14.1 "GraMM wins no column") and was
    narrowed.
  - 12 hits remain, each read in context: negations, withdrawal records, or both-corpora statements.
- **Also, not shipped:** the first re-execution mounted the tree at its host path instead of
  `/work` and wrote a notebook with 38 errors in 10 s. It was caught by counting errors before
  accepting the run, and re-run correctly: 0 errors, and only the two intended cells changed.

### C-74 · Stating an intention was treated as having acted — thirteen times in one work order

- **What.** Across the 2026-09-19→21 work order the same failure recurred in distinct forms: a
  pipeline chain was **described but never written** (D2, 2026-09-20); a launch was **announced but
  not executed** ("Launching all three arms now", then the turn ended — 25 min idle); three idle gaps
  of **3.5 h, 6.8 h and 5.3 h** where a job finished and nothing picked it up; and **eight findings
  declared "belong in the register" and never written** — the entries now numbered C-62–C-73.
- **Why it hides.** A sentence of intent reads, in the transcript, exactly like a record of action.
  The reader — including the author, one turn later — cannot tell them apart.
- **Related, and recurring three times: C-59 applied to progress logs.** "0.7 h" for LightRAG
  (actual 1.88 h), "2.5× faster" for RAG-Anything (actual ~1.2×) and one further walk-back — all
  extrapolated from an **ascending-by-size** progress log, whose early documents are the cheap ones.
  Identifying that bias when *designing* the projection set did not stop it being committed when
  *reading* a log: the discipline transferred to the artefact being built, not to routine observation.
- **Fix.** Act, confirm from output, then report — a status line is written from a PID or a file, never
  from a plan. Long runs are chained end-to-end and polled on a timer (a 29-min cron was added and
  instructed to start the next task, not merely report idleness). ETAs are not extrapolated from
  size-ordered progress.

### C-73 · A credential leak, and three defects introduced while containing it

- **What.** A status check ran `ps -eo pid,etime,cmd`, which printed a `docker run -e
  OPENAI_API_KEY=…` line in full: the live key entered the session transcript, which `sync_handoff.sh`
  ships to vol2. The pre-copy secret scan (ASSERTION -1) covers copyable *sources*; the transcript is
  generated by the sync, so it was never scanned.
- **Contained, 2026-09-21.** The transcript is now **redacted at export** (4 occurrences of 1 key) and
  a new ASSERTION 5c checks what *landed* in `_chat_history/`, independently of what the export
  intended. The shipped copy had never carried the key — the last sync predated the leak.
- **Three defects introduced while fixing it, each caught:** (1) moving the key to `--env-file
  .p3_env` put the secret **inside a synced repo** — ASSERTION -1 aborted before any copy; it now lives
  in `grammrag/.secrets/`, outside every synced tree, and the repo rsync gained env-file exclusions.
  (2) 5c **reintroduced C-53 verbatim**: `ncred=$(grep … | wc -l)` under `pipefail` killed the script
  silently in the zero-match *success* case, skipping every later assertion — the `|| true` fix sat in
  the same file for the same reason. (3) An ad-hoc check matched `ri`**`sk-on-the-…`**, an HTML anchor
  slug, as a leaked key (no word boundary); traced before acting, so nothing was rewritten on it.
- **Not fixable by redaction.** The key remains in the live transcript on the instance. **Rotation is
  the fix, and it is the account owner's action.**
- **The generalisable form.** A lesson recorded as a code comment protects the line it sits on and
  nothing else: C-53's fix was forty lines above the new guard that repeated C-53.

### C-72 · The package regenerated with 0 errors and reversed a pre-registered verdict

- **What.** Cold-start regeneration of `phaseC_evidence.ipynb` from the handoff package alone (vol3
  unmounted) reported **0 errors** and three cells whose **output differed** from vol3:
  - **index sizes** read **0.00 GiB** — measured by walking store directories the package excludes;
    an empty glob sums to zero without raising (a regression introduced in F, which switched §10.3
    from a wrong JSON field to measuring disk);
  - **LightRAG DocBench quote-recall** read **0.0000** instead of **0.5068**: §16 rebuilt page texts
    from the MinerU parse, and `f.read_text() if f.exists() else {'elements': []}` swapped every missing
    parse for an **empty document** — D-32 verbatim;
  - which flipped **pre-registered clause 1** ("GraMM leads DocBench localisation") from **FALSIFIED
    to HELD**, at LightRAG − GraMM **−0.4534, p = 4.6e-100**.
  A reader regenerating from the package would have received the opposite conclusion, stated with
  near-certainty, and no error.
- **Why the existing guard missed it.** §16's drift assertion checked GraMM (0.4534) and Flat-vector
  (0.3959) — the two arms reading **stored** page texts, which cannot drift — and **omitted LightRAG**,
  the one arm rebuilt from excluded data. Rule 20 in its purest form: the guard covered the arms that
  could not fail.
- **A dirty tree would have faked a pass.** The first cold-start container ran without `--user`; its
  root-owned leftovers survived `rm`, and rsync overlays rather than replaces — a cell reading a
  leftover would pass while depending on something the package does not ship. Cleaned as root.
- **Fix.** `precompute_package_artifacts.py` computes excluded-data values on vol3, **mirroring the
  notebook's exact assembly** (a first version appended elements in file order instead of by
  `reading_order`, gave 0.494, and was not shipped) and asserts reproduction (0.5068, n=730) before
  writing `scores/package_precomputed.json`. The notebook reads it and **raises** if absent. LightRAG
  gained its drift assertion. `verify_coldstart.sh` diffs every cell: **52 cells, 0 errors, 0 diffs.**
- **The generalisable form.** **An error count cannot verify reproduction; only an output comparison
  can.** Silent fallbacks never raise, so the test that finds them compares what was produced.

### C-71 · The paper's figure palette failed accessibility, and three figures contradicted their data

- **What.** Run through the dataviz validator (not eyeballed), the tab10 map used by every radar and
  comparison figure **failed**: GraMM ↔ Flat-vector normal-vision ΔE **13.3**, below the hard floor
  of 15 — the two arms the paper contrasts most; MS GraphRAG outside the lightness band; three arms
  under 3:1 contrast. It was also inconsistent: GraMM `#4C72B0` in §1, `#1f77b4` in §10.
- **Replaced** by one map defined once (`ARM_COL`), passing every check in both modes; Flat-vector kept
  neutral gray **on purpose** as the control, with a dashed stroke as secondary encoding.
- **Found only by looking at the rendered figures:** three stale "LightRAG has no DocBench arm" claims —
  one printed **directly beneath a chart plotting LightRAG's DocBench bars**; a stale resource-radar
  footnote ("never measured") after B measured it; label collisions on two frontiers; two occluded
  points (the P3 near-coincidence of LightRAG and RAG-Anything *is* the primary prediction, made
  visible); and near-zero bars (MS GraphRAG ±0.0021) indistinguishable from absent arms.
- **Not added to a GPU-centric radar:** the three API arms score 0 A100-h and 0 GiB; under
  ratio-to-best scaling that reads as a win when the cost moved to API spend. They got their own
  figure (§10.3) framed as capex/opex.
- **The generalisable form.** The validator checks colour; nothing but rendering and reading checks
  that a figure agrees with its own data.

### C-70 · DocBench failure attribution: three convention errors and one metric artefact

- **What.** P4 on DocBench first returned quote-recall 0.24 flat across arms, conv|hit 0.07, and
  abstention accuracy **exactly 0.000** for all three arms. Three independent errors, all mine:
  **(1)** `page_texts[i]` is a **rank-ordered list of five retrieved page texts**, not a
  document-indexed array — indexing it by page number read arbitrary pages; **(2)** refusals are
  **prose** ("the provided document excerpts do not contain…"), so a sentinel match never fired —
  0.000 by construction; **(3)** localisation used top-5 *retrieved* pages while the generator saw
  the k=3 *context*.
- **Validated before trusting:** `compute_quote_recall` on stored arrays reproduced **0.3959, n=730**
  exactly; parse-rebuilt text reproduced 0.3932 on identical pages, licensing the mixed-source
  comparison (the 0.7% gap is page count, not content).
- **What survives as real:** conv|hit stays ~**0.12** after every fix — because DocBench's golds are
  free-form prose scored with **ANLS at a 0.5 edit threshold**, while its own headline metric is
  judge accuracy. That is a metric artefact, marked on the figure, not a corpus property.
- **The generalisable form.** This is the third convention error in this corpus family (1- vs
  0-based pages; window offset vs absolute page; indexed array vs ranked list). Each failed silently
  with a plausible number; each was caught the same way — **reproduce a published value through the
  canonical function before trusting anything derived from it.** Twice in one script the error was
  reimplementing a canonical function instead of calling it.

### C-69 · The canonical scorer was reused without the canonical post-processor

- **What.** P3 ANLS returned **0.001 for all five arms, 99.8% exact zeros**, against archived 0.669.
  Predictions were raw generator output (`EVIDENCE: …\nANSWER: 7.5`); `anls_score` has a hard 0.5
  edit-distance threshold, so a 90-character prediction against a 3-character gold scores 0. The
  missing step is `parse_answer` (POSTPROC_V=2).
- **The config stamp matched.** `POSTPROC_V=2` had been checked and agreed two steps earlier. **Matching
  the stamp is not running the code the stamp names.**
- **A second bug in the fix.** Post-processing was restricted to the three graph arms on the grounds
  that flat-vector and GraMM were "archived". Their **retrieval** is archived; their **predictions**
  were regenerated in D3. The label travelled from the source of one input to a decision about a
  different one. Symptom: two distinct systems at identical SD to four decimals and a CI of exactly
  [0, 0] — an impossibility no plausible-looking output can hide (the C-36 shape).
- **Fix.** `parse_answer` extracted verbatim with an md5 stamp (`p3_postproc.py`, the
  `page_attribution.py` precedent), verified by reproducing archived flat-vector ANLS **to 1e-9
  (0.6694875391)**, applied to all five arms. Flat-vector at k=3 lands at 0.6663 against its Phase A/B
  0.6695 — an external anchor.

### C-68 · A probe artefact under a canonical name made the driver skip an arm

- **What.** A `--limit 6` smoke test of `generate_p3.py` wrote `p3_generation_lightrag.json` — the
  **canonical** name. The driver's `[ -f "$out" ] && skip` then skipped LightRAG as complete holding
  **6 of 978** predictions. The project's own rule — `run_s2_downstream.sh`: *"Never emits a partial
  artefact under a canonical name"* — lived in another script's header; the new script did not carry it.
- **Detected by a component with no verification role.** The resume logic logged `skip lightrag
  (present)` one line after a launch known to have generated nothing. It announced what it did; the
  announcement contradicted what was known. A silent skip passes identically (rule 22, from something
  nobody designed as a check).
- **Fix.** The driver tests **completeness, not existence**: the artefact must hold `n_questions == 978`,
  and it prints the count it checked on every run. `PARTIAL_` naming is kept as redundancy, not as
  the mechanism.
- **The generalisable form.** **Conventions enforced by naming are enforced by whoever remembers
  them; conventions enforced by the consumer are enforced always.** Headers do not travel between
  scripts. Same shape as rule 21 — the guard belongs where the decision is made.

### C-67 · Vendor query crashes were scored as retrieval misses, and flipped the primary contrast

- **What.** RAG-Anything's pinned LightRAG **v1.4.9.2** raised `'NoneType' object does not support
  item assignment` on **28–29** P3 queries; the vendor swallowed it and `aquery` returned `None`. The
  harness mapped `None` → empty chunks → zero attributed pages → **scored as a miss**. A crashed query
  and a legitimate empty retrieval produce the same artefact.
- **It mattered at exactly the resolution of the result.** The pre-registered contrast was LightRAG −
  RAG-Anything **+0.0031** (3 questions) with errors scored as misses; excluding them from the
  denominator gives **−0.0068** — a **sign flip**. The numerator (329 hits) never changed. Primary
  prediction, final: **+0.0021 [−0.0074, +0.0116], n=950** — HELD.
- **Errors are not random.** The 28 are significantly *easier* than chance (document pages p=0.0196,
  question length p<0.0001, gold depth p=0.0411; one-sided toward lower; 28 over 23 documents). So
  RAG-Anything's row is **conservative**, and every cross-arm contrast is computed on the five-arm
  intersection, n=950, stated beside the table.
- **Fix.** `query_error` recorded as its own category, excluded from the scored denominator, reported
  as a count — applied to **every** arm, not only the one where it surfaced. The error count moved
  29→28 between passes: the vendor fault is partly non-deterministic.
- **The table that stated the rule then broke it.** `P3_RESULT.md`'s first draft headed its five-arm
  table "n = 950" while the localisation columns carried **n = 978** values for four of five arms —
  each row complete, the header wrong. Found while re-scoring for C-66's check 1, before the document
  shipped. Recomputed on the intersection: every value moves ≤ 0.006, no ranking changes, and
  RAG-Anything − LightRAG then equals the paired primary contrast exactly (+0.0021) — which the mixed
  table did not.

### C-66 · Phase A/B's MP-DocVQA gold pages are wrong on 48.9% of questions — reversing its localisation claim

- **What.** `gramm-rag-mpdocvqa/gen_notebook.py:311` builds the gold page as
  `pages_list[ans_idx] if ans_idx < len(pages_list) else 0`. In MP-DocVQA's schema `pages` is a
  **two-element `[start, end)` range**, not a list of pages. So `answer_page_idx = 0` gives the right
  page (51.1%); `= 1` gives the window's **end** (17.0%); `≥ 2` falls **silently to 0** (31.8%).
  **2,535 of 5,187 gold pages are wrong.** Both archived result files match that formula on 5,187/5,187
  questions. Ground truth is `image_name[answer_page_idx]` — the gold page's own filename.
- **The published APPA claim reverses.** Recomputed with `compute_answer_page_accuracy` (the archived
  values reproduce exactly first):

  | GraMM − Flat-vector | as published | corrected |
  |---|---|---|
  | APPA@1 | **+0.2296** [+0.211, +0.247] | **−0.2928** [−0.309, −0.278] |
  | APPA@5 | +0.1513 | −0.3574 |
  | APPA@10 | +0.1199 | −0.3233 |

- **Why it reverses rather than shrinks.** GraMM's `retrieved_pages` begins with page 0 on **98.0%** of
  questions; page 0 is the true answer on **23.1%**. On the 1,652 questions whose gold fell back to 0,
  GraMM is credited at APPA@1 on **96.2%** (Flat-vector: 2.8%). `node_pages` defaults to −1, so this is
  not a sentinel — page-0 nodes are genuinely retrieved first.
- **The same labels trained the model.** Lines 1254–1261 select the HGT's **positive** evidence nodes
  with `_actual_answer_page`; line 331 stores it as `_evidence_pages` "for router training"; line 1554
  validates reward calibration against it. **Probable chain:** wrong labels → HGT trained toward page 0
  → GraMM retrieves page 0 first → APPA scores that against the same wrong labels. An error present in
  both training and evaluation confirms itself.
- **Supported by two independent controls (not yet by intervention).** *Same architecture, correct
  labels:* on MMLongBench GraMM ranks the first page top on **7.8%** of questions, its top pages spread
  across the document. *Same corpus, no label training:* Flat-vector ranks page 0/1 top on **32.1%** of
  MP-DocVQA questions. GraMM on MP-DocVQA: **99.1%** (5,083 / 5,187). The bias is specific to the one
  corpus whose training labels are corrupted, and absent from an untrained arm on that same corpus —
  which rules out "MP-DocVQA documents are just short". Only a retrain on corrected labels would
  establish it.
- **Scope.** MP-DocVQA only — the defect is in its `imdb` schema handling; MMLongBench (evidence-page
  lists) and DocBench (quotes) use separate code. **ANLS is not directly affected** (it scores answer
  text), but it measures a model trained on corrupted supervision. P3's GraMM row (context
  page-hit@3 0.7189, ANLS 0.6978; n = 950 common set — 0.7127 was the n = 978 value, C-67) is a measurement of that same model and carries the caveat.
- **Bias direction (rule 20) runs both ways.** The APPA defect **flattered** GraMM; the training defect
  plausibly **penalised** it. Correcting APPA alone would therefore understate a correctly trained
  GraMM.
- **Scope narrowed from the shipped config.** Two of the three label consumers were **inert**: the
  DeBERTa router was disabled (`TRAIN_ROUTER=False`, heuristic router), and refusal was disabled at
  generation, so the reward calibration touched logged values only. **Only HGT training used the
  wrong labels.**
- **Three checks before disposition (2026-09-21).** (1) **P3 is not contaminated**: its golds match
  the filename truth on 950 / 950, and GraMM's P3 lead holds on the 750 questions whose answer is not
  page 0, +0.0480 [+0.0320, +0.0653]. (2) **P3's contribution to corpus dependence does not rest on
  GraMM**: in leave-one-out τ-b, removing Flat-vector moves MP-DocVQA *toward* the other corpora and
  removing GraMM moves it *away*. The mover is an arm that never trains on labels. (3) **GraMM still
  beats flat-vector on MP-DocVQA ANLS**, +0.0141 [+0.0076, +0.0205] (n = 5,187) and +0.0315
  [+0.0147, +0.0482] (P3, n = 950).
- **Disposition.** **B done** — `CORRECTION_2026-09-21_MPDOCVQA_APPA.md` retracts the APPA claim and
  restates the ANLS claim with its caveat; no archived file is edited. **C (retrain the HGT) is no
  longer a validity requirement:** every GraMM-involving result that survives does so despite a defect
  that handicapped it, so each is a lower bound. It stays optional, as the way to turn lower bounds
  into estimates and to test the page-0 hypothesis by intervention. Estimate: ~$15–22 API, ~15–30 h
  A100, pending a one-epoch probe for A100 time and 40 GB fit. See `MPDOCVQA_GOLDPAGE_DEFECT.md`.

### C-65 · P3's subsample stored a window offset where an absolute page was needed

- **What.** The first P3 retrieval smoke test returned **page-hit@3 0.0000 at alignment 0.93** —
  chunks attributed fine, no hits. `answer_page_idx` indexes the question's **page window**; the
  subsample stored it alone and discarded `pages`. Fix: `gold_page_absolute = page_window[0] +
  answer_page_idx`, verified **978/978** against `window_page_names[idx]`.
- **The same misreading of the same field is C-66.** The one used here was verified against the
  file's own image names and holds; the one in Phase A/B was never checked. A spurious "confirmation"
  was also logged: the archive's first three gold pages agreed with the fix, and three agreeing rows
  were generalised to a whole convention — the agreement rate over 978 was 50.5%.
- **A separate retraction.** Four near-identical questions returning byte-identical page rankings were
  reported as "the query is not reaching the retriever". A two-minute nonsense-query control (real
  ~50k chars, "banana helicopter quantum tuesday" 70 chars) showed the retriever was working; the
  questions shared a source page. That control is now a startup assertion in `retrieve_p3.py` (535–699×
  separation) — the discriminating invariant, as against the misread symptom.

### C-64 · The projection gate passed two false verdicts in a row

- **What.** `p3_project.py` printed **"WITHIN estimate"** twice on invalid data. **First**, from **one**
  indexed document: two arms had crashed, their missing logs were skipped as rows, and what remained
  was summed and graded. A coverage precondition was added. **Second**, with all three arms at 50
  documents, it passed that gate — reading **run logs** of resume no-ops that honestly recorded
  "$0.0000 new spend". The true measured base, from per-document `_INDEX_COMPLETE.json` stamps written
  when each document was actually indexed, moved the projection from **$4.51 to $14.77 (3.3×)**.
- **The shape, recorded as it recurred.** The coverage gate guarded the failure just seen; the next
  failure was different. The gate now asserts **physical invariants** — a metered arm cannot index 50
  documents for exactly $0 or in exactly 0 s — and refuses with "do NOT read the per-arm numbers above
  as a projection" rather than degrading to a number. The driver propagates the exit code instead of
  swallowing it through `tee`.
- **Denominators are quantity-specific.** Characters projected cost to within ~16–26% on all three arms
  and overstated wall-clock by ~46–54% (per-document fixed overhead), a split confirmed on two arms
  before it was stated. Entity density (5.6× entities for 15% more characters on one document) and, for
  GraphRAG, community-report generation are cost drivers no character denominator can see.

### C-63 · Two broken timing instruments passed a call-count guard

- **What.** B′ decomposed MS GraphRAG's query latency into context assembly and answer generation.
  **Attempt 1** subtracted litellm's self-reported `end − start` from a `perf_counter` wall time: two
  clocks. Context-only came out **10.0 ms warm / 2.7 ms cold** — below one embedding round-trip, and
  cold *cheaper* than warm. litellm's self-report runs **1.24×** wall. **Attempt 2** wrapped
  `LitellmChatModel`, but installed a *sync* wrapper on `achat_stream` (an async generator) and timed
  generator **creation — ~1 ns**; context-only then equalled the total.
- **The guard passed both.** Its coverage check counted wrapper *calls*: 1.000 on both broken
  instruments. It could not tell a wrapper that measured correctly from one that measured a nanosecond.
  **Both were caught by an external physical invariant** — context-only must exceed an embedding
  round-trip, and cold must exceed warm — never by the guard written for the purpose.
- **Fix.** Dispatch on what the call *returns* (awaitable / async generator / generator), and assert
  the invariant directly (generation ≥ 20% of total; context ≤ 80%). Final: context assembly
  **0.24–0.27 s**, generation **81–86%** of total; cold > warm on both corpora.
- **Also corrected in the same work:** an earlier report that S5 context assembly was "an order of
  magnitude faster than LightRAG" (it is comparable), and an upper-bound caveat written in the wrong
  direction.
- **The generalisable form.** **A guard written from the author's model of what will break tests the
  author's model, not the code.** Assert what must be true of a working system, not what was false in
  the last broken one.

### C-62 · Index size reported the two-corpus total as each corpus's size

- **What.** `resource_cells.py` summed `sdir.rglob("*")` over the whole stores directory. LightRAG uses
  one directory per corpus and was correct; **RAG-Anything and MS GraphRAG keep both corpora in one
  directory**, so their "MMLongBench" and "DocBench" cells both reported the combined total.

  | arm | corpus | reported | actual |
  |---|---|---|---|
  | RAG-Anything | MMLongBench | 6.74 GiB | **2.32** |
  | RAG-Anything | DocBench | 6.74 GiB | **4.43** |
  | MS GraphRAG | MMLongBench | 4.87 GiB | **3.62** |
  | MS GraphRAG | DocBench | 4.87 GiB | **1.29** |

- **Found in F, by reading the rendered table** — identical values across two corpora for two arms,
  and distinct ones for the third. **Fixed at the source** (prefix-filtered glob), not only in the
  display; the notebook now reads a shipped precomputed value (C-72).
- **Also corrected in the same table:** `$/query` printed `0.00000` for LightRAG and RAG-Anything —
  a rounding artefact reading as "free". True cost is **~7.3e-07** per query, embedding-only, because
  the on-disk LLM response cache absorbed keyword extraction (`llm_calls = 0`). These are cache-warm
  service costs, and the table now says so.

### C-61 · A third corpus was scoped and costed before checking whether it had the stratum

- **What.** P3 was designed, argued and budgeted as the falsifier for this project's headline
  corpus-dependence finding — whose strongest leg is that **abstention rankings invert between
  corpora (τ-b −0.828)**. MP-DocVQA was the chosen third corpus: already parsed, 927 documents on
  disk. It has **no `is_answerable` field**: all 5,187 questions are answerable. **The abstention leg
  cannot be tested on it at all**, because decline-decision balanced accuracy requires both strata.
- **Found before any measurement**, while checking the subsample design — but *after* the arm set,
  cost and schedule had been argued over, including a decision to include three vision arms whose
  entire justification was that they anchor the abstention ranking. That justification evaporated
  with one `grep`.
- **A second property, same shape.** MP-DocVQA contains **zero images** (250/250 documents sampled),
  against ~40 per document in both other corpora. That one turned out useful — it converts leg 3 into
  a sharp mechanism test, since RAG-Anything's image route must contribute exactly nothing there —
  but it was equally unchecked when the corpus was chosen.
- **The generalisable form.** **The space of available falsifiers is smaller than the space of
  benchmarks, and which claims a corpus can falsify is a property of its strata, not of its size or
  domain.** An abstention claim needs a corpus with something to abstain from; a modality claim needs
  images; a judge-robustness claim needs a judge. Of three multi-page document-QA benchmarks used
  here, **two carry an unanswerable stratum, one carries a judge, and one has no images**. Written up
  as `BENCHMARK_LANDSCAPE_CONSTRAINT.md` because it is useful to anyone designing an abstention
  experiment, not only to this project.
- **Fix.** Check the strata a claim needs **before** selecting, costing or scheduling its falsifier.
  Concretely, a corpus proposed as a falsifier must be checked for: the stratum the claim is about,
  the metric family the claim uses, and the modality the mechanism depends on — three greps, none of
  which was run until the design was otherwise complete.
- **Consequence for the write-up, not a loss.** DUDE is named as the abstention leg's real falsifier
  and scoped as future work. "Untestable here, and here is exactly what would test it" is a stronger
  position than an unexplained absence — but it is only available because the gap was found before
  the run rather than in review.

### C-60 · Three corpus documents are duplicates, and a liveness guard halted a healthy run over one

- **What.** The S5 full index aborted at document 129 of 364 on the rule-22 liveness assertion —
  *"meter not live (extraction or embedding calls = 0)"*. `mi_phone` had indexed in **4 seconds at
  $0.0000 with zero LLM calls**, yet produced 82 entities, 8 communities and 8 reports at alignment
  1.0000. Two explanations with opposite consequences: a dead meter recording $0 while real money was
  spent, or a legitimate zero.
- **It was a legitimate zero, and the cause is a property of the corpus.** `mi_phone.pdf` and
  `NYU_graduate.pdf` parse to **byte-identical text** (md5 `bb693ba563bb5784`, 27,788 chars).
  Processing the second in the **same process** reused the first's results. Confirmed by a
  single-document control: re-run alone in a **fresh process**, the same document took **38 s and
  $0.0221** and made real calls.
- **MMLongBench contains 135 documents but only 132 distinct texts.** Three duplicate pairs:
  `NYU_graduate` / `mi_phone`, `SAO-StudentSupport_Guidebook-Content` / `StudentSupport_Guidebook`,
  and `digitalmeasurementframework22feb2011v6no` / `dr-vorapptchapter1emissionsources-121120`.
  **This is a corpus property that affects every arm in this study, not an S5 artefact**, and no
  document in the project had previously stated it. Questions attached to both members of a pair are
  questions about the same text.
- **The guard was right to be suspicious and wrong to halt.** Cost alone cannot separate the two
  explanations — both look like $0.00. The discriminator is the **injected text's md5**, which is
  available before the call is made. The guard now exempts a document whose text hash was already
  seen in the run, records `duplicate_of`, and states that the cost is attributed to the first
  occurrence. Everything else about the assertion is unchanged, because the failure it was written
  for — a callback that silently stops firing — remains real.
- **A second finding from the control: GraphRAG's clustering is stochastic.** The same input gave
  **8 communities** in one run and **13** in another. Community count therefore varies run to run,
  which matters twice: the **β fit** and the cost model inherit that variance, so the projection's
  precision is bounded by clustering noise rather than by sampling alone; and **a re-run of this arm
  will not reproduce its community counts exactly**, which must be stated wherever they are reported.
- **The in-process reuse is benign and is kept.** Identical input yielding identical output is what a
  cache is *for*, and it is preferable here to the alternative: fresh-process indexing would give two
  copies of the same document **different** clusterings, injecting noise unrelated to the method into
  a comparison between arms. The reuse is recorded rather than removed.
- **Class.** A guard false-positive caused by an unstated property of the data. Related to C-36 in
  shape — cross-document state inside one process — but opposite in valence: C-36's shared state
  silently corrupted results, this one silently produced correct ones.
- **The first fix was incomplete, in the register's most familiar shape.** Exempting duplicates
  from the *liveness* assertion left the sibling assertion — *communities exist but report calls are
  zero* — unguarded, and the run halted again at document **259** on the second duplicate pair
  (`dr-vorappt` reusing `digitalmeasurementframework`). **C-35 → C-36 again, this time inside a
  guard I had just written to fix the first instance**: one route closed, the state leaking through
  another that looked unrelated. The rule that generalises is narrow and checkable: **when a
  condition exempts a class of input, every assertion resting on the same signal must take the
  exemption together** — here, all three call-count assertions, since in-process reuse zeroes
  extraction, embedding and report calls alike. Content assertions stay unguarded, because a
  duplicate still carries real output.
- **A second, quieter bug in the same fix:** the hash was registered *after* the resume
  short-circuit, so a run that resumed the first copy and freshly indexed the second would never
  recognise the pair. Moved before it.
- **The duplicate check then found something larger, and it is not about S5** (see
  `CORPUS_PARSE_INTEGRITY.md`). Two of the three MMLongBench collisions are **not duplicates but
  mis-assignments**: `mi_phone` parses to the NYU housing guide, and
  `dr-vorapptchapter1emissionsources` parses to the digital-measurement deck. The questions prove it
  — `mi_phone` asks how many applications appear on pages 3–6, its parse is a university housing
  guide. **The rendered page images are mis-assigned too** (byte-identical page 1), so the vision
  arms and the shared generator inherit it as well. **19 questions, 1.7% of MMLongBench, are answered
  against the wrong document by every arm in the study.** Uniform, so comparisons are unbiased;
  absolute ANLS is depressed by +0.0023 to +0.0054 per arm, with no ordering change. Reported with
  its sensitivity rather than silently corrected, because re-parsing would invalidate the do-not-touch
  canonical runs. **A guard written to catch a dead meter found a corpus defect that had survived
  every arm of the study** — the value of emitting a number on every document (rule 22) rather than
  only on failure.

### C-59 · A progress extrapolation was biased by the order the work was queued in

- **What.** The S5 full-corpus index processes documents **ascending by size** (smallest first, so
  defects surface cheaply). An in-flight monitor extrapolating cumulative cost per character read
  **$35.02 against a projected $28.02**, and cumulative time read **38.1 h against a projected
  7.1 h** — an apparent 25% cost overrun and a 5× time overrun, on a run that was tracking fine.
- **Why it was wrong.** The denominator was right; the **sample** was not. At that point the run had
  covered the smallest **1.3%** of corpus characters, where per-document fixed costs — a community
  report generated even for one or two communities, the embedding call, pipeline overhead — dominate.
  The pilot's own documents show it: the small ones cost **$0.68–0.73 per 1M chars** against a corpus
  average near **$0.61**. Any cumulative rate taken early is therefore an estimate of *small
  documents*, not of the corpus.
- **The general form.** **A progress-based extrapolation is biased whenever the work is ordered by
  the same variable the projection divides by.** Ordering smallest-first is good practice for failing
  cheaply, and it silently invalidates the most natural progress metric. The two are in direct
  tension, and nothing about the monitor's output reveals which regime it is in.
- **Relation to C-48.** C-48 was the same *consequence* — a false alarm from a wrong progress
  figure — by a different mechanism: there the denominator itself was wrong (characters for an
  image-driven cost), here the denominator is correct and the sample is unrepresentative. **Twice
  now a progress monitor has produced an alarm large enough to justify aborting a healthy run**
  ($893 against a $200 ceiling in C-48; $35 against a $28 projection here).
- **Fix: compare against the model's prediction for the documents actually completed**, which no
  processing order can bias. `s5_inflight_check.py` computes actual/predicted ratios over the
  completed subset: **cost ratio 1.155, time ratio 1.031** at the same moment the naive figures read
  25% and 5× over. Rescaling the projection by the observed ratio gives **$32.35**, against a naive
  $35.00 — and, unlike the naive figure, it is a statement the projection can actually be scored on.
- **Standing consequence.** A progress monitor for a run ordered by size must report
  **predicted-vs-actual on completed work**, never a cumulative rate, and must say which fraction of
  the driving variable has been covered so a reader can see how far the estimate can be trusted.

### C-58 · A registered gate criterion returned a stop/go verdict from a point estimate on n=13

- **What.** Gate H criterion 3 for S5 — *"does a pilot show it working rather than degenerate:
  behavioural validation against a random or control baseline"* — was registered with **no minimum
  sample size and no rule for how uncertainty enters the verdict**. Its first evaluation returned
  **FAIL** on MMLongBench from **13 questions**: S5 ctx-hit 0.1538 against a random-page control of
  0.1658, a margin of **−0.0119** in a sample where one question moves the rate by **0.077**.
  DocBench passed the same test at n=24 with a margin of **+0.1567**.
- **Why this is expensive rather than untidy.** Criterion 3's registered disposition is **stop**. A
  point-estimate verdict on n=13 would have stopped a funded arm on noise — and, had the sign fallen
  the other way, would have passed it on noise just as readily, with nobody looking again.
- **The asymmetry is the part worth generalising.** A criterion whose branches carry *asymmetric
  consequences* — here stop versus proceed — cannot be evaluated by a rule that treats both
  directions alike. An underpowered FAIL is acted on; an underpowered PASS is silently absorbed into
  the arm's results. **Underpowered tests do not merely add noise; they add noise that is
  preferentially cashed in one direction.**
- **Class: C-51's family, entering from the other side.** C-51 was a *guard* misapplied to a
  registered prediction, deciding on a point estimate what needed an interval. Here the
  **pre-registration itself** was underspecified, so the clause could not be evaluated correctly by
  anyone following it. Same root in one sentence: **a decision rule that does not say how
  uncertainty enters is not a decision rule — it is a preference with a threshold attached.**
- **Fix, written before the additional data existed.** Criterion 3 amended: minimum **n = 60 scored
  questions per corpus**; decision on a **paired bootstrap CI (4,000 resamples, seed 42)** of the
  margin; three outcomes rather than two — CI excludes 0 positive → **PASS**, excludes 0 negative →
  **FAIL (stop)**, contains 0 → **INDETERMINATE**, which is *not* a failure and lets the arm proceed
  only with the indeterminacy published beside that corpus's localisation result. The control
  probability is computed per question over its own document's page range, so it is matched to
  document length instead of pooled across documents of very different sizes.
- **The sample was enlarged to make the registered test evaluable, not to change its threshold.**
  The expansion documents were drawn by the same seeded octile rule, excluding those already indexed,
  and the decision rule was fixed before they were indexed — it applies whichever way the result
  falls, including FAIL → stop.
- **Also fixed: the checking script.** `s5_gate3_validation.py` compared point estimates and printed
  a bare PASS/FAIL. Rebuilt to report the margin with its CI and to refuse a verdict below the
  minimum n (`UNDERPOWERED`), so the instrument can no longer produce the verdict the criterion
  forbade.
- **Standing consequence.** Every registered gate criterion must state **(a)** the minimum sample on
  which it may be evaluated and **(b)** how uncertainty enters its verdict. A criterion missing
  either is evaluable only by accident.

### C-57 · A wrapper library relabelled credit exhaustion as throttling

- **What.** The S5 smoke test hit OpenAI's project spend limit — HTTP 429, `type: insufficient_quota`,
  `code: project_spend_limit_exceeded`, 54 rejected calls from 01:27:15, after only GraphRAG's three
  non-LLM workflows had completed. The run **went silent rather than failing**: 0.02% CPU for over ten
  minutes. GraphRAG treats `RateLimitError` as throttling and retries each call up to 10 times with
  exponential backoff, and `run_pipeline` would then have absorbed the final error. Stopped by hand;
  almost nothing spent (no entities, empty cache).
- **D-34 recurs one layer up.** D-34 was OpenAI reporting credit exhaustion under the same exception
  *class* as throttling, and the fix everywhere since has been to read the error text for
  `insufficient_quota`. Reproduced at zero cost against the exhausted limit:

  | client | top-level exception | `type` | `code` | text contains `insufficient_quota` |
  |---|---|---|---|---|
  | OpenAI SDK | `openai.RateLimitError` | `insufficient_quota` | `project_spend_limit_exceeded` | yes |
  | LiteLLM (S5) | `litellm…RateLimitError` | **`throttling_error`** | **`429`** | **no** — only "spend limit" survives |

  **LiteLLM does not merely drop the distinguishing field; it relabels exhaustion as throttling.**
  Every D-34 guard that calls the OpenAI SDK directly — generation, judge, WP-3, calibration, and S4's
  adapter — fires correctly. S5's, written the same way, would not have.
- **The information is buried, not destroyed.** The original `openai.RateLimitError`, with its
  structured `code` and `type`, survives two links down the exception chain — in the raised exception
  and in the failure callback's `kwargs["exception"]` alike. The HTTP response body does not carry it;
  it has already been consumed.
- **Mechanism, and why it generalises.** Rule 21 says a guard must *signal* on a channel the code under
  test cannot handle. This is the same principle applied to *classification*: **a guard must read the
  field it depends on at a layer that still carries it.** A wrapper owns its top-level exception and
  can rewrite its type, code and message; matching on what the wrapper surfaces trusts the wrapper's
  taxonomy, and here that taxonomy is inverted.
- **Fix.** S5's failure hook walks the exception chain and classifies on structured provider fields
  first (`code` ∈ {`project_spend_limit_exceeded`, `insufficient_quota`}, or `type == insufficient_quota`),
  with prose only as a fallback. On the first exhausted call it writes `FATAL_credit_exhaustion.json` and
  **exits the process**: a raise inside a LiteLLM callback is absorbed, and GraphRAG's backoff would
  otherwise keep the run silently asleep.
- **Correction recorded.** The first reading was that none of the guard keywords appear in the error.
  That holds for the message prose only; the OpenAI SDK's exception text does carry `insufficient_quota`,
  so the gap is specific to the LiteLLM path.
- **A finding about the envelope as well as the code.** OpenAI's enforced limit bound while the Phase C
  ledger reads ≈$187 of a $500 new-spend envelope. Either the configured project cap sits below $500, or
  the project's spend includes Phase A/B usage that no repo artefact records. The provider dashboard,
  not this register, can say which.

### C-56 · A pre-registration reasoned about the comparator as it ships, not as the harness runs it

- **What.** `S5_PILOT_PREREGISTRATION.md` §1a argued that GraphRAG's community-report cost scales
  with **corpus** size — Leiden clustering runs over the whole indexed graph — and designed a nested
  4-vs-8-document pilot to observe community growth across documents. But **C-35** binds every arm to
  per-document retrieval scope: S1 and S2 run one store per document, and S5 must too. Under
  per-document indexes communities never span documents, so the 4-vs-8 comparison measures nothing
  about community growth — eight documents cost the sum of eight independent single-document costs.
- **Caught before any spend**, while writing the adapter: GraphRAG's `build_index` treats one input
  directory as one corpus, which forced the question of what that directory would contain.
- **Mechanism.** The pre-registration modelled the comparator's *native* architecture correctly and
  forgot the *harness override* already on record as C-35. Every sentence was true of GraphRAG; the
  design was wrong for this study. **A pre-registration can be internally rigorous and still test the
  wrong quantity, if it reasons about the system rather than about the configuration being measured.**
- **The consequence would not have been neutral.** The withdrawn design would have run cleanly,
  produced a tidy community-growth ratio, and supported a projection — for a scaling regime the arm
  never operates in. Meanwhile the real non-linearity went unexamined: under per-document scope,
  community count grows with **document size**, and the corpus's largest documents are where a
  superlinear exponent would dominate cost.
- **Fix.** §1a rewritten: 9 pilot documents stratified on document size, including the corpus's
  largest; community count modelled as `k · chars^β` per document; the projection summed over each
  document's actual size, because with β > 1 the total depends on the size distribution and not only
  its sum. §1c's NOT-IDENTIFIED definition and operative total rewritten to match.
- **A second instance in the same document, caught one step later.** The rewritten §1a said stage
  costs would be *"separated using GraphRAG's own per-workflow cache"* — an implementation mechanism
  written in as fact without being read. graphrag 2.7.2 keys the chat cache by **model name**, so all
  stages share one cache and the mechanism does not exist. Replaced before the pilot with per-call
  classification in a LiteLLM success callback. Same shape as the entry above, one level down: the
  first claim was about the architecture, this one about the code, and neither was checked against
  what would actually run.
- **Rule of thumb.** Before pre-registering a comparator's cost or behaviour, write down the *harness
  configuration* it will run under, and check each structural claim against that configuration —
  not against the paper or the README.

### C-55 · A precedent inherited by name did not carry its operative clause

- **What.** `S4_PILOT_PREREGISTRATION.md` declared that if the two cost projections disagree by
  >2×, the split is NOT IDENTIFIED *"— the S2 precedent, applied in advance."* It did not carry
  that precedent's **operative clause**, which lives in Gate H §5a: *"any projection with an
  unidentified cost model → proceed only on the **conservative** (higher) estimator."* It also
  contained **no full-arm cost or scope disposition at all** — only a $15 pilot ceiling. When the
  pilot fired the rule (2.29×), the S4 result's budget table quoted the **central** $245.55 as the
  operative figure. Caught by external review, which asked what NOT IDENTIFIED obliges.
- **Mechanism: inheritance by reference.** Citing a precedent by name transfers its *authority*
  and none of its *content*. The clause that mattered at decision time lived in another document,
  and the citation made re-reading it feel unnecessary. A close relative of **C-50** — there a
  number gained authority by repetition; here a rule lost its content by citation.
- **The precedent does not resolve cleanly, which is a finding about Gate H, not only S4.** S2's
  "conservative estimator" was the **highest point estimator** ($144.29 pooled — itself
  misspecified). Applied literally to S4 it selects **chars-only $552.66**, physically wrong for an
  arm whose vision share is 0.967 (C-48). The reviewer's planning figure, **$340.24**, is the upper
  bound of the physically-specified model's bootstrap CI — an interval endpoint, not an estimator.
  **"Conservative estimator" is undefined whenever the estimator set contains misspecified
  members**, and S2 never exposed that because its conservative choice happened not to matter.
- **Not decision-relevant for S4:** every reading ($245.55 / $340.24 / $552.66) exceeds Gate H's
  $150 arm cap, and two exceed the $250 scope-down threshold.
- **Fix.** Each arm's pre-registration carries its **own copy** of the operative disposition,
  never a pointer; and the conservative figure is **defined operationally before the pilot**.
  Applied to S5 before its pilot runs: the operative total is the upper 95% bootstrap bound of the
  physically-specified model, explicitly *not* the highest single-denominator estimator.

### C-54 · A constrained fit returned a boundary solution that contradicted its own data

- **What.** The pre-registered S4 cost model (`cost = a·chars + b·images`, both ≥ 0, no cross
  term) returned **a = exactly 0** — text extraction costs nothing. The pilot contains a
  **zero-image document that cost $0.0109**: a direct measurement that the zeroed term is not zero.
- **Why.** The text term is **3.7%** of the corpus total. Squared error over 7 documents is
  dominated by the image-heavy ones, so a small but real coefficient was driven onto the
  non-negativity boundary. Squared error can afford to discard 3.7%; a corpus projection cannot.
- **Fix.** Identify `a` on the uncontaminated pure-text observation (**$0.177 / 1M chars → $8.10
  corpus**) and fit `b` on the residual: **$245.55**, against the boundary fit's $218.67.
- **Rule of thumb, cheap to apply: a constrained optimum sitting on a boundary is a flag, not a
  result.** Check it against any observation that directly measures the zeroed term. The pilot
  *had* such an observation only because it was stratified on images (C-48) — so this is also
  evidence that the C-48 stratification rule earns its keep beyond the purpose it was written for.
- **Recorded late.** `S4_PILOT_RESULT.md` cited "C-54" before this entry existed — a document
  pointing at a register entry with no referent, caught in review.

### C-53 · The secret guard validated the destination, so it detects leaks instead of preventing them

- **What.** `sync_handoff.sh` runs its rsync copies (lines 31-105) **before** ASSERTION 4 greps
  the destination for credential-length keys (line ~205). A secret in a source file is therefore
  **written to the shared vol2 handoff and then detected** — the guard reports a leak that has
  already happened. Surfaced while configuring S4: the vendor's `parameter.py` is its documented
  config surface, and writing an OpenAI key into it would have shipped that key to a volume whose
  entire purpose is being shared with a reviewer.
- **A fourth guard-failure mechanism, and it completes §7.4b's taxonomy.** This guard is correct
  (it greps the right pattern), audible (rule 21 — it prints, on a channel nothing absorbs),
  well-aimed (rule 20 — it fires against the author's interest), and emits a value every run
  (rule 22). It satisfies every rule the register has, **and it still cannot do the job**, because
  it runs downstream of the event it guards:

  | | mechanism | instance |
  |---|---|---|
  | cannot fail | tautological | C-7 |
  | cannot point | right rule, wrong target | C-51 |
  | cannot be heard | signal absorbed by the system | C-52 |
  | **cannot prevent** | **correct, audible, well-aimed — but ordered after the event** | **C-53** |

  **Detection and prevention are different guarantees, and passing a detection check is not
  evidence that prevention occurred.** For a reversible artefact detection suffices; for a secret
  reaching a shared volume it does not, because the leak is not undone by noticing it. A guard's
  *position in the sequence* is as load-bearing as its correctness — and position is invisible in
  every property the other three rules test.
- **Fix.** ASSERTION -1: a **pre-copy** scan of the *sources* that aborts before any rsync runs,
  printing `ok: pre-copy source scan clean` every run. ASSERTION 4 is retained — the two now
  cover prevention and detection respectively, rather than one being mistaken for both.
- **Second finding, a recurrence.** `rsync --exclude` stops a path being **copied**, never being
  **present**. `vendor/` (77M of third-party source) had shipped in runs predating its exclusion
  and would have stayed in the package indefinitely — no assertion looked for it. The script
  already carried this exact lesson in a comment about `chat_history/`, **so this is the second
  occurrence of a failure the file itself documents.** Knowing a rule and having it enforced are
  different things: the exclusion list is now *asserted* against the destination and offending
  paths removed, rather than trusted.
- **Third finding, self-inflicted inside the fix.** The pre-copy scan was first written as
  `hits=$(grep ... | head -5)`. Under `set -euo pipefail`, a grep matching **nothing** exits 1 and
  killed the whole script with **no output and exit 1** — the precise silent-failure shape
  ASSERTION 0 was written to prevent, reintroduced by a guard added to prevent a different one.
  Caught immediately because the script produced no output where it always produces some, which
  is rule 22's liveness property doing exactly its job.
- **Class.** Guard-design, ordering. Caught during implementation rather than by review or by a
  failing run.

### C-52 · A mitigation for a silent fallback selected the other silent fallback

- **What.** MMGraphRAG's preprocessing has two D-32-class downgrade paths to a different parser:
  (1) `mineru` binary absent → PyMuPDF with a warning; (2) `_process_mineru` raises → a broad
  `except Exception` → PyMuPDF, continue. The mitigation drafted into the S4 pilot
  pre-registration was a `mineru` shim on `PATH` that exits non-zero if invoked, described as
  converting the fallback into fail-fast.
- **Why it fails.** Installing the shim makes the binary **exist**, so path (1) can no longer
  fire. The shim's non-zero exit raises `CalledProcessError` inside `_process_mineru`, which the
  broad handler catches and routes to **PyMuPDF, silently** — path (2). **The mitigation does not
  prevent the fallback; it selects the other one.** The run continues under a parser that is
  neither the vendor's intended one nor ours, which is the exact confound the guard existed to
  prevent. Caught in review, before implementation — nothing was run under it.
- **Class: C-35 → C-36, recurring in a *mitigation* rather than in pipeline code.** A fix closes
  one route while the state it guards leaks through another, and looks correct from outside.
  C-36 was the same shape (per-document stores fixed the corpus-wide confound; module-level
  globals reinstated it). **The register now has this shape three times, and the third instance
  is in a guard**, which is the more dangerous location: pipeline defects produce wrong numbers,
  guard defects produce wrong *confidence*.
- **Unreachability — the property to lead with, because it is the one that makes this dangerous
  rather than merely wrong.** If cache injection succeeds, `_run_mineru` short-circuits and the
  shim is **never exercised**. It would have sat unfired through every successful run and been
  wrong on exactly the one run it existed to protect: **its failure invisible until precisely the
  moment it mattered.** A guard that cannot fire on the happy path and is wrong on the unhappy
  one is not neutral — it manufactures false assurance. That is C-7's property arriving by a
  different route: **C-7's check could not fail; C-52's could not be reached.** Both read as
  rigour, and neither can catch anything.
- **The structural lesson, stated at the level that generalises: channel ownership (rule 21).**
  The first draft of this lesson was "assert on content, not control flow" — true, but it names
  one instance and misses the class. The operative property is that **a guard must signal on a
  channel the code under test cannot handle**, and *any system with error recovery will absorb
  signals sent through channels it owns.* Three candidate guards here signalled on three
  channels, two of them owned:

  | guard | channel | outcome |
  |---|---|---|
  | shim exiting non-zero | process exit code | absorbed → routed to the guarded fallback |
  | pre-flight check raising | exception | absorbed by the same handler → same fallback |
  | comparing store chunks to injected markdown after the call | content, outside the call | **works** — no handler exists for a fact about its own output |

  The vendor's `except Exception` swallows *our own* injection errors too: a malformed
  `content_list` raises `ValueError` inside `_process_mineru` and lands in PyMuPDF exactly as the
  shim did. **Error recovery is precisely the authority to discard error signals**, so the better
  a system's recovery, the more channels it owns and the fewer remain usable for guarding it. The
  same shape appears far from parsers — a log the system rotates, a flag it resets on retry, a
  timeout it retries past, a sentinel file it cleans up.
- **Fix.** A per-document **content-level post-condition**: the store's chunk text is compared to
  the markdown we injected, via the stride-1 shingle index already built for C-49; alignment must
  be **≥ 0.95**, asserted immediately after each document, halting on first failure. Decisive
  regardless of which path executed, requires no vendor patching, and answers *"did this arm
  ingest our parse?"* instead of the proxy *"was a binary invoked?"*. The pre-condition check is
  retained but explicitly recorded as **cheap and insufficient** — only the post-condition is
  load-bearing.
- **Caught by external review**, not by an internal guard — the same bound on guard credit that
  C-51 established (§7.6).

### C-51 · A guard against overclaiming was used to suppress a prediction's failure

- **What.** Gate H §5b's third clause predicted RAG-Anything's false-decline rate would rise
  between corpora **by less than the text/graph family's 1.64×**. Measured **1.747×** — above
  the threshold, so the registered prediction fails. It was filed instead as **"unresolved"**,
  citing **C-41** (do not assert a direction the interval does not support), on the ground that
  the ratio's CI [1.313, 2.410] contains 1.64.
- **Why that is wrong, and it is wrong twice.**
  1. **C-41 governs claims about the world, not outcomes of tests.** "RAG-Anything's
     false-decline rate rises faster than the text/graph family" is a world-claim, and it is
     correctly reported as *not established* at 95%. "The registered prediction failed" is a
     statement about **the clause**, decided by the point estimate, because the clause was
     written as a point prediction. Collapsing the second into the first suppresses a negative
     result about the author's own pre-registration.
  2. **This is the clause that was deliberately sharpened so it could fail.** An earlier draft
     said "less than 2×", which a 1.9× rise would have satisfied *while moving faster than the
     family it is predicted to sit below* — a clause that cannot really fail (rule 2). The
     threshold was tightened to the measured family ratio for exactly that reason. Filing a
     point-estimate violation of the sharpened clause under "unresolved" **restores the vacuity
     the sharpening removed**.
- **Direction, and this is the entry's point.** Every other guard in this register fires against
  overclaiming. This one was applied to a **failure**, and a guard applied to a failure is a
  shield: it protected a pre-registration from being falsified. **The register's own instruments
  can run in the confirmatory direction when pointed at the author's predictions rather than at
  the author's claims** — and nothing in the guard's wording flags the misuse, because the
  wording is about intervals and directions, which were both present.
- **Fix — a two-sentence discipline, not a new interval.** When a registered clause is scored,
  report **both** of these and never let one stand for the other:
  (a) *did the registered prediction hold?* — decided as the clause was written (here: point
  estimate, so **FAILS**); (b) *is the effect established?* — decided by the interval (here:
  **not** established, CI contains the threshold). Two sentences, always both. **Collapsing them
  in either direction is a different error**: collapsing (a) into (b) suppresses a failure;
  collapsing (b) into (a) asserts a world-claim the data do not carry.
- **The general form — this is why the entry matters beyond its own clause.** The specific fix
  repairs one scoring. The generalisation is **rule 20**: *guards have direction, and their
  direction depends on what they are pointed at.* C-41 aimed at a world-claim fires **against**
  the author; aimed at "my prediction failed" the same rule fires **for** them, because what it
  suppresses has changed from a claim that flatters to a conclusion that costs. The test that
  catches it is not semantic but **directional** — before invoking any guard, ask which way it
  points relative to your own interest. Reading the guard more carefully does not help; it was
  applied correctly on its own terms.
- **Pairs with C-7: the two ways diligence manufactures false assurance.** C-7 was a check that
  **could not fail** — a tautological identity that verified nothing while looking like
  verification. C-51 is a check that **fired correctly, in the wrong direction**. Both present
  from outside as rigour; **neither is caught by executing the process more carefully**, because
  in both cases the process *was* executed correctly. C-7 needs a check that can fail; C-51 needs
  a check on the direction the check points. Together they mark the boundary of what
  conscientiousness alone can buy, and they are the strongest pair in the register for the claim
  that method must be audited by something other than more method.
- **Class.** Directional. Caught by external review, not by any internal guard — which bounds
  the credit the guards can take (§7.6) and adds to the tally of directional incidents surfaced
  from outside rather than inside.

### C-50 · An unsourced number acquired authority by being repeated across the collaboration

- **What.** `S2_INDEXING_RESULT.md` scored the measured 28.93 h wall-clock against estimators of
  **25.7 h / 54.3 h**. The pre-registered figures are **26.4 h fitted / 56.2 h pooled**
  (`GATE_G_RESULT.md`, repeated in Gate H §5c-ter). The pair 25.7/54.3 appears in no
  pre-registration, no log and no projection field — it has **no provenance at all**.
- **How it survived, and this is the part the register has not documented before.** The figure
  did not sit in one document waiting to be checked. It **propagated through the collaboration**:
  it was written into an executor status message, echoed back by the reviewer in the next turn as
  "the pre-registered 26–54 h range", and only then written into a result document — where it was
  used to score a run *against the very pre-registration it misquoted*. Neither party traced it.
- **The transmission mechanism.** *A number gains authority by being repeated, and the second use
  is harder to question than the first, because by then it has a witness.* On first appearance it
  is one party's assertion and reads as something to check. On the second it reads as shared
  context — already agreed, already survived a reading. Nothing was asserted more strongly at any
  step; the authority came from the **repetition**, not from any claim about the source. This is
  how a figure with no provenance survives long enough to be cited.
- **Why it is distinct from C-39 and C-46.** C-39 was a claim outrunning its evidence in one
  author's head; C-46 was a review conducted against a stale artefact. This is neither: the number
  was **created in conversation and laundered by being quoted back**. The register's existing
  provenance guards all assume a single author checking a single document. None of them watch the
  channel between two parties, which is where this one lived.
- **Fix.** A figure being scored against a pre-registration must be **read from the
  pre-registration at the moment of scoring**, never from conversational memory or from a prior
  message — including a prior message of one's own, and including one the other party has since
  repeated. Agreement between two parties is not provenance. Concretely: any number appearing in a
  result document as a registered threshold, band or estimator must cite the file it came from.
- **Class.** Provenance, propagating through the collaboration channel rather than within one
  artefact. **The first entry in this register whose failure mode requires two parties to occur.**

### C-49 · The multimodal arm was being scored on half of what it retrieved

- **What.** The S2 (RAG-Anything) retrieval smoke test reported mean page-attribution
  alignment **0.4995**, against **0.9765** for LightRAG on the same benchmark under a
  byte-identical retrieval config (same `hybrid` mode, same k=3, same shingle parameters —
  the two `RUN_CONFIG`s differ in the `RETRIEVER` label alone). Roughly half of everything
  RAG-Anything retrieved could not be attributed to any page.
- **Cause, and it is not a bug in either system.** The page index is built from
  `mapper.elements_to_text`, the text-side rendering in which images and tables contribute
  no characters (mapping decisions D1–D4). RAG-Anything additionally ingests each image and
  stores a VLM-written **"Image Content Analysis"** block for it. That text is *generated*,
  so it has **no character preimage in the parse** and the shingle index cannot attribute a
  single character of it — measured `align 0.000` on exactly those chunks, against 0.91–0.99
  on every text chunk in the same context.
- **Why it mattered more than a cosmetic number.** The unattributable half is not arbitrary:
  it is precisely the multimodal content that distinguishes RAG-Anything from the text-only
  graph arm. Scoring the arm on its text half alone would have **understated its localisation
  exactly where its differentiator lives**, and the pre-registered G-5 mechanism test asks
  whether RAG-Anything falls *between* the text/graph and vision families. A defect that
  biases one arm toward the text family is a defect aimed at the test's own conclusion.
- **The check that caught it was a comparison, not a threshold.** The existing blocking
  assertion halts only when >50% of questions attribute to **zero** pages; here
  `zero-attribution` was **0** and every question attributed to something, so the guard passed.
  What exposed it was holding the number against the other arm measured the same way: 0.4995
  beside 0.9765, under configs that differ in one label.
- **Fix — a second attribution route, exact rather than fuzzy.**
  `mapper.elements_to_content_list` writes the resolved absolute `img_path` into each image
  item **alongside its `page_idx`**, and RAG-Anything echoes that same path back inside the
  chunk. So provenance is recovered by a dictionary lookup on a key this project's own mapper
  produced — verified **95/95 exact key matches** on the probe document, every attributed page
  within the document's 117-page range. Alignment **0.4995 → 0.9599**, against LightRAG's
  0.9765; the residual is `## ` heading boundaries, which affect both arms equally.
- **The coordinate trap, recorded because it would not have shown.** `content_list` uses
  0-based `page_idx` (= `page_no − 1`, mapper D3) while the shingle index keys on **1-based**
  `page_no`. Both routes had to be normalised into one system. An off-by-one would have
  shifted every multimodal attribution by exactly one page and produced a fully plausible,
  silently wrong ctx-hit — with alignment looking *repaired*, which is worse than looking broken.
- **Guards added, because a route that maps nothing looks like an arm that retrieves nothing.**
  Those two are indistinguishable downstream, and the second is a result while the first is a
  bug. (i) map construction asserts non-empty whenever the parse holds image items; (ii) an
  S2 run that attributes **zero** image chunks across all questions halts rather than emitting
  a text-only score for a multimodal arm. Mass is assigned on the same scale as the text route
  (probes are non-overlapping, so a fully-attributed text chunk accrues exactly its length),
  so C-37's relevance ordering is preserved across both routes.
- **A measurement, not only a repair.** Over the full MMLongBench corpus (n=1,091) the image
  route supplies **0.5177 of the 0.9589** total — **54.0% of RAG-Anything's attributed retrieval
  mass is VLM-generated description rather than document text** — and **1,061 of 1,091 questions
  (97.3%)** pull at least one image chunk. So this is not a handful of image-heavy documents; it
  is nearly every question in the benchmark. That is a property of the arm worth reporting in its
  own right: the multimodal comparator spends the majority of its retrieval budget on text that
  does not exist in the document. Both routes are emitted separately
  (`alignment_image_route`, `image_chunks_per_q`) so the split stays auditable rather than folded
  into one number. (The 15-question smoke test that caught the defect gave 0.4942 of 0.9599 —
  close, but the corpus figure is the one reported.)
- **How much of the result the fix moved — measured, not asserted.** A rerun with route 2
  disabled (`--no-image-route`, writing to a separate `_textonly` artefact that cannot overwrite
  the canonical one; same stores, same mode, same k, same seed) gives, on the canonical n=838
  denominator:

  | corpus | metric | shipped | text-only | delta | align shipped | align text-only |
  |---|---|---|---|---|---|---|
  | MMLongBench | ctx-hit (n=838) | **0.5465** | **0.3294** | **+0.2172** | 0.9589 | 0.4544 |
  | DocBench | quote-recall@5 (n=730) | 0.5014 | 0.4973 | **+0.0041** | 0.9577 | 0.7501 |

  **On MMLongBench the fix moved localisation by +0.2172 and flipped a ranking**: text-only,
  RAG-Anything sits *below* GraMM-RAG's 0.4344; with the route, *above* it. The statement
  "RAG-Anything localises better than GraMM-RAG on MMLongBench" is therefore **conditional on
  this attribution route**, and is reported that way rather than as a bare ordering. Orderings
  against LightRAG and ColPali are unchanged on both corpora, as is every DocBench ordering.

- **The corpus split is a finding, not just an impact estimate.** The same code buys **+0.2172**
  on MMLongBench and **+0.0041** on DocBench — a **53× difference**. RAG-Anything's multimodal
  branch contributes to page selection where the corpus is image-dense (research papers, slide
  decks) and essentially nothing where it is not (filings and reports, whose documents are also
  short enough that top-3 often spans most of the page range). A single-corpus reading of this
  ablation would have generalised a 53× corpus effect from one side of it — the S1 post-mortem's
  error, avoided here only because the second corpus was run. **This is the third independent
  leg** of the single-benchmark finding (`SINGLE_BENCHMARK_FINDING.md`), joining S1's ranking
  reversal and the abstention rank inversion at τ-b −0.828 — three structurally unrelated
  quantities, three corpus-dependent answers.
- **The route is load-bearing for usability, not only for a metric.** Without it, **102 of 1,091
  MMLongBench questions (9.3%) retrieve no page at all** — against 2/1,091 shipped — because
  everything RAG-Anything returned for them was image analysis; the generator would have received
  no document context whatsoever. On DocBench the same figures are 14/1,102 against 5/1,102.
- **Why the route is nonetheless the right measurement, stated so a reader can disagree.** The
  text-only figure measures how well RAG-Anything localises *using only the half of its retrieved
  content that happens to have a character preimage in the parse* — a property of this harness's
  index, not of the retriever. Every arm is attributed on all of its own retrieved content:
  LightRAG has no image content to miss, ColPali returns pages directly and needs no attribution
  at all. Excluding route 2 would single out the one arm whose retrieval is majority-multimodal.
  The route uses the retriever's **own** chunk ordering for rank (C-37) and the mapper's own
  `img_path` key for page, so it adds no judgement of ours. A reader who rejects the route has
  the text-only number here to use instead — which is why it is published beside the headline
  rather than replaced by it.
- **Class.** Same family as C-36: caught by an *arithmetic-plausibility* comparison against a
  sibling arm, not by any assertion written in advance. Both times the run exited 0 and produced
  a complete-looking artefact. **Tally: this is the fourth defect found by comparing a number to
  what it must be, rather than by a guard that anticipated it.**

### C-48 · The referent error appeared in two guards I wrote to prevent it, in one run
- **What.** Within a single hour of S2 indexing, the same wrong-denominator error appeared in
  **two independent monitoring guards**, both written by the executor **after** rule 18 was
  recorded, and both normalising an image-driven cost by **characters**:

  | guard | what it reported | what was true |
  |---|---|---|
  | progress monitor | **$893 projected** against a $200 ceiling, "176 h left" | by images — the actual cost driver — **$97.02**, ~18 h |
  | rule 9 anomaly detector | two anomalies on one document, "inspect before trusting this store" | the document has **4.89 images per 1k chars** against a corpus median of 0.31 — a density spike, not a defect |

- **Consequences of each, and they run opposite ways.** The monitor's figure would have
  triggered an unnecessary abort of a run that is tracking its projection. The anomaly detector
  cried wolf, and a detector that fires on ordinary variation trains its reader to ignore it —
  which is worse than not having one.
- **The convergent evidence is the useful part.** Projecting on images gives **$97.02** against
  the constrained model's **$92.09**. The physically well-specified estimate is being confirmed
  in flight, and the two misspecified 5-document estimators ($137.21, $144.29) look
  correspondingly high. That was pre-registered as the discriminator (Gate H §5c-ter) and it is
  resolving toward the **$74–111 band**, though the run is 5.25% through by images and no band
  is claimed yet.
- **Why this is worth an entry rather than two corrections.** Rule 18 says: report every rate as
  *"X per Y, measured on Z"*. Both guards were written to enforce discipline **after** that rule
  existed, by the author who wrote it, and both silently chose the wrong Y. **Knowing the rule,
  having just written it, and being actively engaged in enforcing it were jointly insufficient.**
  This is C-47's finding one level out: not only can a check fail to compel consultation, a
  check can *embed* the very error it exists to catch.
- **Fix.** The monitor reports progress by **images** (the cost driver) with characters shown
  beside it. Rule 9's mechanical form now computes the ratio on **both** denominators and flags
  only when **both** exceed the threshold — a genuine defect is anomalous on any reasonable
  denominator, a density artefact only on the mismatched one; a single-denominator excursion
  prints as a note, not a flag.
- **Standing consequence.** A guard that normalises by a denominator must state which driver it
  assumes and be checked against a second denominator before it is trusted. **Guards are
  artefacts too, and rule 18 applies to them.**

### C-47 · Rule 17's first form was satisfied and failed — adjacency is not the mechanism
- **Not a defect in a result; a defect in a guard.** Logged separately from the claim it failed
  to catch (instance 5 of the claim-exceeds-computation class) because the finding is about the
  **remedy**, not the error.
- **What happened.** Rule 17 was amended to require an ordering claim to appear **adjacent to
  its interval**. Instance 5 satisfied that requirement exactly: the CI `[0.5689, 0.6224]` was
  printed **two lines above** the sentence asserting the value was "still below the 0.614
  FAILURE threshold". The upper bound exceeds the threshold. **The rule was followed and the
  claim was still wrong.**
- **Why the first form looked sufficient.** Across instances 1–4 the relevant number was
  usually **not present at all** — M6's ranking sat in another document, C-40's arm count was
  never printed. Adjacency appeared to be the operative property because absence was the
  visible symptom. Instance 5 separates the two: the number was present, adjacent, and
  unconsulted.
- **The mechanism: availability is not consultation.** Placing a number nearby makes it
  *available*. Reading it and checking a claim against it are **separate cognitive acts**, and
  composing prose reliably invokes the first without the second. A guard that depends on the
  author choosing to look is not a mechanism — it is awareness wearing a procedural costume.
- **Fix — the comparison must be written, not the inputs displayed.** Rule 17 now requires the
  ordering and its check in **one sentence**: *"0.5956, CI upper 0.6224, which does not exclude
  0.614."* Omitting the comparison then requires writing something **visibly incomplete**,
  which is a different and much harder failure than not looking something up.
  **This form would have caught instance 5; adjacency demonstrably would not, and instance 5 is
  the proof.**
- **Consequence for the chapter.** §7.4 must not credit rule 17 as a fix that worked from the
  start. The honest sequence is: **instances 1–4 motivated it, instance 5 falsified its first
  form, and the second form is what the counterexample produced.** A rule improved by its own
  counterexample is better evidence for the method than a rule that never needed revising.
- **Corollary to §7.3's "awareness is not the mechanism; the check is."** Not every check is a
  mechanism. A check that relies on the author looking inherits exactly the failure mode of
  awareness, and is distinguishable from it only by having a name.

### C-46 · A review was conducted against a stale copy, and nothing in the package could detect it
- **What.** WP-3's pre-registration was revised (three defects fixed), synced, and reported as
  synced. The reviewer, reading a copy pulled **before** the revision, reported all three
  defects as still present and declined approval. Verification: vol3 and vol2 were
  **byte-identical** (md5 `73a7939e…`), all three fixes present in both, and the phrase quoted
  as remaining — *"tripled to allow generously"* — existed in **neither**. The handoff was
  current; the local copy was not.
- **Why it is a defect in the process rather than an error by either party.** The reviewer had
  **no way to check** whether what they were reading was what had been changed. The agent
  asserted "synced"; the reviewer held a file; nothing connected the two. A review executed
  competently against the wrong artefact produces **false assurance** — and had the three
  defects *not* been fixed, the same process would have produced a correct rejection, so the
  outcome carried no information about the document either way.
- **Class: C-45's shape relocated into the review loop.** C-45 was a check correctly executed
  against a criterion that should not have been one. This is a check correctly executed against
  an **artefact that was not the one under discussion**. In both, diligence is not merely
  insufficient — it actively manufactures confidence, because the process ran exactly as
  designed.
- **Second occurrence.** This is the **second** stale review in one session, which moves it from
  incident to pattern and is why the fix is mechanical rather than a resolution to be careful.
- **Fix.** `sync_handoff.sh` now emits `GraphRAG_baseline/REVISION_MANIFEST.md` on every sync:
  every review-facing document with its **md5 and mtime, derived from content, never typed**.
  A reviewer checks the hash of what they hold against the manifest before reading. A mismatch
  identifies a stale pull immediately and distinguishes it from a defect in the document —
  which is precisely the distinction this incident could not make.
- **Standing consequence.** Any artefact submitted for external review must carry a
  content-derived revision identifier that the reviewer can check independently. "I synced it"
  is an assertion by the author about a file the reviewer cannot verify — the same class as a
  claim that two computations are identical, evidenced by cost (rule 16).

### C-45 · An illustrative number became a pre-registered criterion
- **What.** Reviewing the plan to augment the S2 pilot, the reviewer proposed a decorrelation
  target — *"say r < 0.6"* — as an **illustrative** figure, offered to make the shape of a
  criterion concrete. It was adopted verbatim into `GATE_H_COMPARATOR_SET.md` §5c as a
  **pre-registered gate**, without deriving it or asking whether it was meant as a threshold.
- **Consequence.** The deterministic selection rule produced r = **0.802** against the target,
  and the criterion bit: by its own terms the augmentation should not have proceeded. But the
  number had no basis. It was never a computed identifiability boundary — and 0.802 corresponds
  to a **variance inflation factor of 2.8**, inside the conventionally acceptable range, against
  5.7 at the original 0.908. **A criterion with no derivation was about to block a substantively
  worthwhile measurement.**
- **The trap that makes this worth an entry.** The obvious escape — *"it was only a proxy"* — is
  what is always said when a criterion bites, and accepting it would corrode every other gate in
  this project. The resolution is not to override the criterion but to observe that it was
  **badly specified**, replace it with one that tests the thing actually cared about, and record
  the substitution **before** any augmented numbers exist.
- **Replacement.** The r-target measured the *design*; the question is whether the **estimates
  separate the hypotheses**. Now: fit with the physical constraint (vision cost depends only on
  images), bootstrap over documents, and report a **95% CI on the vision share** — a direction
  is reported only if that interval **excludes 50%**, otherwise "not identifiable in this
  design". One attempt, no third.
- **Also recorded: the pre-registration was silent on this case.** It specified a stopping rule
  for the *identification outcome* but not for a **pre-flight target miss**, so the document
  said "report the miss" and nothing more — leaving the decision to be made with the miss
  already in hand, which is the position pre-registration exists to avoid.
- **Standing consequence.** When proposing a threshold, **derive it or label it illustrative**;
  the difference is invisible once it is written into a gate. And every pre-registered criterion
  needs a stated disposition for **failing its own precondition**, not only for failing the test.
- **Class: its own shape — a correctly executed check on an incorrectly specified criterion.**
  It is **not** another possess-a-check-without-applying-it instance, and conflating them would
  attach the wrong fix:
  - *avalaunch (C-39)*: rule 9 existed and **was not run** against the log line. Fix: make the
    check mechanical.
  - *the 1,091 coincidence*: rule 2 existed and **was not applied** to proposed evidence. Fix:
    apply the rule to one's own evidence.
  - *C-45*: the check **was** applied — faithfully, on time, exactly as written — to a criterion
    that should never have been one. **Diligence made it worse, not better**, because executing
    a bad gate correctly produces false assurance. A checklist does nothing here, and the
    mechanical-check fix that addresses avalaunch does nothing either.

  **This is C-7 inverted.** C-7 was a check that *could not fail* (a tautological identity
  asserted as verification). C-45 is a check that *could fail and did* — on the wrong quantity.
  Both produce false assurance; they need opposite remedies. C-7 needs a check capable of
  failing; C-45 needs a criterion whose derivation is checked before it is trusted.
- **Attribution.** The illustrative figure was the reviewer's, adopted without challenge by the
  executor; neither role questioned its provenance until it fired.

### C-44 · The C-33 guard fired correctly on a second, independent comparator
- **Not a defect.** Recorded because it is the cheapest available evidence for the transfer
  claim, and because a register that logs only failures cannot show that a guard ever worked.
- **What happened.** C-33 was the unwired multimodal path: `insert_content_list()` returned
  `{"success": False}` from a parser check, the caller discarded it, `modal_processors` stayed
  empty, and every image was dropped while the run exited 0 and reported a plausible cost. The
  fix added (a) an explicit processor initialisation and (b) a **blocking assertion** — images
  present with zero vision calls halts the run.
- **On S2's first execution against a new arm**, the guard reported
  `modal processors registered: ['equation', 'generic', 'image', 'table']` before any document
  was ingested, and the vision meter recorded non-zero calls on every document containing
  images. The failure C-33 documents cannot recur silently on this arm.
- **Why it is worth the entry.** It is a small, clean instance of the pattern §5 argues for: a
  guard written for one comparator's defect **carrying to the next comparator without being
  re-derived**. Unlike the C-24 → C-33 transfer, which crossed systems and mechanisms, this one
  is narrow — same library, same code path. It is therefore weaker evidence, and is logged as
  the weak kind: **a guard doing its job on a new arm**, not a rule generalising to a new shape.
- **Filed separately from the correction-direction tally.** It is neither a correction nor a
  claim withdrawn, so it does not enter §7.6's counts — consistent with the rule that the
  register counts defects while outcomes of tests live in the notebook.

### C-43 · Benchmark and metric family were perfectly confounded in the conversion comparison
- **What.** Conversion efficiency eta was computed as ΔANLS/Δctx-hit on MMLongBench and
  Δjudge-accuracy/Δquote-recall@5 on DocBench. The 6.3x difference in GraMM's eta
  (0.165 vs 1.041) was described as evidence that **conversion is benchmark-specific**.
  Benchmark and metric family vary together, so the comparison cannot separate them.
- **De-confound attempted** (G-1): DocBench also archives ANLS, so eta was recomputed there
  with ANLS as the numerator, making the quality metric common across corpora. The cross-corpus
  gap shrank substantially and, for LightRAG, reversed direction — **most of the effect
  attributed to corpus was metric scaling**. The corrected values are **not reported here**;
  see the impact note below for why.
- **Impact: the cross-corpus eta comparison is UNAVAILABLE in this design, not merely
  uncertain.** Two attempts, two distinct failures, and no third option exists:
  1. judge/quote-recall vs ANLS/ctx-hit — numerator **and** denominator definitions differ;
  2. ANLS on both — fixes the numerator, but DocBench ANLS is degenerate (near floor), and the
     denominator still differs because DocBench has no `gold_pages` and ctx-hit cannot be
     computed there at all.

  **No number is reported.** A hedged figure is worse than none: readers retain figures and
  discard qualifications, so "1.7x, upper bound, degenerate numerator" would be cited as 1.7x.
  The corrected value is deliberately omitted from every document; the limitation is reported
  with its reason instead.

- **The boundary that survives: within-benchmark comparisons hold, across-benchmark magnitudes
  do not.** Everything load-bearing is within a corpus, where metric choice is common to the
  arms being compared and cancels:

  | evidence | corpus | status |
  |---|---|---|
  | GraMM localises +0.1527 over LightRAG with no measurable ANLS gain | MMLongBench | **holds** |
  | A3's 101+ page bin: answer delta negative while ctx-hit is +0.0989 | MMLongBench | **holds** |
  | LightRAG's +0.0534 localisation edge converts to +0.0345 judge accuracy | DocBench | **holds** |

  So the write-up may say **conversion fails on MMLongBench, evidenced two independent ways,
  and succeeds on DocBench**. It may not say **by how much** the corpora differ. That is
  narrower than the claim held before G-1 and wider than "withdrawn" — the boundary is stated
  explicitly so a reader does not infer the whole finding collapsed.

- **What is NOT affected.** The S1 falsification (C-42) is untouched: it compares two systems
  **within** DocBench on metrics common to both arms.

- **A finding worth keeping, not just a caveat.** DocBench ANLS runs **0.065-0.082** where
  judge accuracy runs **0.47-0.57** on the same predictions. Edit-distance scoring is close to
  useless on long free-text answers. This defends a design choice the thesis otherwise has to
  argue for: the per-benchmark metric families were **forced by answer format**, not chosen.
  Belongs in Chapter 5's metric discussion, and is a real caution for anyone reusing ANLS
  across answer formats.

- **Standing consequence.** Before comparing a derived quantity across conditions, check
  whether anything *else* changes with the condition. A ratio compared across two settings that
  differ in both numerator and denominator definition measures the definitions at least as much
  as the settings — and when no common definition exists, the correct output is "not
  measurable here", not a corrected number.

### C-42 · S1 falsified its own pre-registered prediction, and a headline merit reversed
- **What was predicted** (Gate H §5, before generation and scoring): GraMM leads LightRAG on
  DocBench localisation by a smaller margin than on MMLongBench, and the answer-quality
  difference again fails to reach significance.
- **What happened.** Both clauses false. **LightRAG leads localisation** (+0.0534
  quote-recall@5 [+0.0219, +0.0849], McNemar p=0.0013) **and answer quality** (+0.0345 judge
  accuracy [+0.0045, +0.0662], p=0.034). Drift checks passed on both metrics before either
  contrast was computed.
- **Consequence 1 — M9 reverses.** M9 was filled from MMLongBench as "localisation leadership
  over an external graph-RAG system" (+0.1527 ctx-hit, p=7.7e-12). The same comparison on
  DocBench runs the other way, significantly. **Two corpora, opposite directions.** The merit
  is restated as leadership *on MMLongBench* and a deficit *on DocBench*; it can no longer be
  stated as leadership over LightRAG.
- **Consequence 2 — a pre-registered falsifier fires.** Gate H stated that a failure of the
  quality null on DocBench would make the between-systems conversion-gap claim
  benchmark-specific. It failed; the claim is withdrawn as general. Chapter 6's three-line
  argument keeps one unqualified leg (Gate C); dimension B had already shown the internal leg
  is GraMM-specific.
- **The instructive part is why the prediction missed.** Its rationale was *correct*: GraMM's
  localisation edge over the control is 3.5x larger on MMLongBench (+0.2017) than on DocBench
  (+0.0575), which does imply a smaller GraMM margin. What it ignored is that **LightRAG's edge
  moves too, and in the opposite direction** — +0.0489 on MMLongBench against +0.1110 on
  DocBench, a 2.3x increase. Reasoning about one system's trend while holding the comparator
  implicitly fixed is an **incomplete enumeration of what varies** — rule 15's shape applied to
  a prediction rather than to a gate.
- **Standing consequence.** A prediction about a *contrast* must state an expected trend for
  **both** sides of it. Predicting how our system moves across corpora, and treating the
  comparator as a constant, is a one-sided model of a two-sided quantity — and it will be wrong
  exactly when the comparator is the thing that changed.
- **What this cost, stated plainly.** A headline merit and one leg of the central argument, for
  $23.9. Both were reported by the protocol working as designed: prediction written down before
  the run, falsifier attached, result reported against it.

### C-41 · A shipped superlative did not survive its own confidence interval
- **What.** `GRAPH_BRACKET_RESULT.md` stated, as a headline: *"GraMM converts localisation into
  answer quality least efficiently of any system measured, on both corpora."* It was written
  from a ranking of eta point estimates (GraMM 0.17 against LightRAG 0.38, ColPali 0.26), with
  eta explicitly labelled "descriptive, not a test" — and then used to carry an argument.
- **What the test showed.** Bootstrapping the ratio (dimension D): **no pairwise eta difference
  reaches significance**, uncorrected or Bonferroni-corrected over 4 tests. ColPali − GraMM
  is +0.098 [−0.001, +0.192]. And **LightRAG's eta is uninterpretable** — 0.382 with a CI of
  [−0.204, +1.854], because its localisation delta is small enough that the ratio is dominated
  by denominator noise. The arm that looked like the best converter cannot be compared at all.
- **A borderline contrast flipped between bootstrap draws.** ColPali − GraMM came out
  [+0.002, +0.196] on one seed and [−0.001, +0.192] on another. **A contrast whose significance
  depends on the resampling draw is not a result**, and reporting the first draw would have
  been a seed-dependent claim.
- **Class: rule 17** — the computation was right and the sentence overstated. It is the fourth
  instance of that class, and the first found in a document already shipped to the reviewer.
- **A second defect, in the guard itself.** The stability rule first written for this analysis
  marked a ratio interpretable whenever the denominator CI excluded zero. LightRAG passed that
  test (dLoc CI [+0.0107, +0.0883]) while producing a CI two orders of magnitude wider than its
  own estimate. Excluding zero is not the same as being bounded away from it. Tightened to
  require the denominator's lower bound to exceed 25% of its point estimate.
- **Fix.** Claim withdrawn and replaced with what the data support: every eta is far below 1
  (the conversion gap is present in every arm), GraMM's point estimate is lowest, and no arm
  differs significantly. The withdrawal is recorded in the result document, not silently edited.
- **Standing consequence.** A quantity labelled "descriptive, not a test" must not then be used
  to support a comparative claim. Either it carries an interval and can be compared, or it is
  descriptive and cannot — the label is not a disclaimer that licenses the comparison anyway.

### C-40 · A silently-narrowed sample produced a confident class-level verdict that was false
- **What.** Dimension A re-runs Gate A's cross-page stratification across every comparator, to
  ask whether the absence of cross-page concentration is a property of GraMM or of the task.
  The analysis cell built its score table from a dict populated three sections earlier
  (`Flat-vector`, `GraMM-RAG`, `LightRAG 1.5.7`) and **silently tested 2 arms of 5**. The three
  ColPali-family arms were never added, and nothing said so.
- **What it produced.** A verdict cell that printed:
  *"cross-page concentration is absent for EVERY arm measured … it fails for a flat-vector
  control, an external graph-RAG system **and a vision retriever** alike — so the finding is a
  property of the TASK, not of one architecture."*
  **No vision retriever was in the analysis.** The prose asserted a fixed architectural list
  rather than describing what had run, so it credited an arm that never entered.
- **The verdict was not merely under-powered — it was wrong.** With all five arms:

  | arm | interaction (cross − single) | |
  |---|---|---|
  | GraMM-RAG | −0.0245 [−0.0693, +0.0211] | null |
  | LightRAG 1.5.7 | +0.0045 [−0.0550, +0.0627] | null |
  | ColPali | **−0.2150 [−0.2826, −0.1495]** | **concentrates** |
  | Union(CP∪GraMM) | **−0.2024 [−0.2690, −0.1407]** | **concentrates** |
  | ColPali size-matched | **−0.1897 [−0.2581, −0.1220]** | **concentrates** |

  Three of five arms show strong, significant concentration. The class-level claim
  ("absent for every arm") is **false**, and the two-arm run would have published it.
- **How it was caught.** Not by an assertion — there was none. By reading the printed arm count
  (`systems tested: 2`) against the five arms known to exist, and by noticing the prose named
  an architecture family absent from the table.
- **Fix.** Two, and the second is the one that generalises.
  1. Rebuild the score table over every arm with predictions, and **assert the arm set matches
     the systems present in the stratified report** — a missing arm now fails loudly.
  2. **Derive the verdict prose from what actually ran.** The architecture families named in
     the conclusion are now computed from the tested arms, so the text cannot credit an arm
     that was not analysed. A conclusion that hard-codes its own scope will keep asserting that
     scope after the scope changes.
- **Class.** This is C-38's shape (a partial input silently accepted as complete) combined with
  the reviewer's seventh error (a conclusion outrunning its evidence). What makes it the more
  dangerous instance is that both halves failed in the *same direction*: a narrowed sample
  produced a cleaner-looking result, and confident prose made it sound better-supported than a
  five-arm null would have been.
- **Standing consequence.** Any cell that emits a **conclusion over a set** must print the set
  it used and assert its size. Prose stating "every X" is a claim about the denominator, and
  the denominator must be shown next to the claim.

### C-39 · "Exact cache reuse" was 134/135, and the hash check is what established the number
- **Claim under test.** The C-36 re-index dropped from **$4.93 to $0.0975** and was reported as
  running on *exact* reuse of the seeded extraction cache — "a hit is exact reuse of the same
  computation on the same input, not a substitute for it". A follow-on proposal was to
  deduplicate 135 supposedly byte-identical 85 MB cache copies and reclaim ~11 GB.
- **The check.** SHA-256 of all 135 caches, against each other and against the seed. Run
  *before* deduplicating, because identical files were the premise of the proposal and because
  divergence would be a provenance fact about a published arm either way.
- **Result: 135 distinct hashes, none matching the seed, none matching each other.**
  Deduplication is therefore **off the table** — byte-identical copies carry no information a
  hash does not, but these were not byte-identical. Nothing was deleted or merged.
- **What the divergence actually is — census across all 135 stores, not a sample:**

  | | |
  |---|---|
  | seed cache entries | 4,326 |
  | stores containing NEW entries | 135 / 135 |
  | **new entries, by type** | **1,090 `keywords` + 1 `extract` = 1,091** |
  | stores whose new entries include `extract` | **1** |
  | stores missing seed entries | **1** (the same store) |

  Every new entry is accounted for, and the accounting closes exactly:

  - **1,090 `keywords`** = query-time keyword extraction written during *retrieval*, after
    indexing. MMLongBench has 1,091 questions but only **1,090 distinct (document, question)
    pairs** — one question text appears twice within `PRE_2022.09.29_NSL-politics_REPORT.pdf`,
    and LightRAG keys this cache by (query, mode) within a store, so the pair collapses to one
    entry. **1,090 expected, 1,090 measured.**
  - **1 `extract`** = the single genuinely recomputed extraction, in the outlier store.

- **A correction to this entry's own first draft, which is the reason it was re-measured.**
  The draft reported the type distribution as **"100% `cache_type=keywords`" from a 3-store
  sample**, stated with the confidence of a census — while the outlier store's fresh `extract`
  entries were already a known counter-example in the same entry. That is the C-39 failure
  class applied to C-39: a sampled figure carrying census-strength language. The census
  above replaces it.
- **And the headline arithmetic was right for the wrong reason.** The draft framed
  "1,091 new entries = 1,091 questions" as the entry's strongest evidence — an exact identity
  confirming the query-caching account. The identity is real, but it is a **coincidence of two
  unrelated quantities**: 1,090 keyword entries (which match *distinct pairs*, not questions)
  plus 1 fresh extraction. Had the corpus contained no duplicate question, the totals would
  have been 1,091 and 1,092 and the "exactness" would have evaporated without the underlying
  account changing at all. **A numerical coincidence accepted as confirmation is the same
  error as a tautological check accepted as verification (C-7)** — it feels like evidence and
  constrains nothing. The corrected evidence is the *decomposed* match: 1,090 measured against
  1,090 independently-derived distinct pairs, with the residual single `extract` located in a
  named store.

- **Impact.** Bounded and small: one 1,004-character document, 0.01% of the corpus by
  characters. No result is withdrawn. But the *claim* was wrong as stated — "exact reuse" was
  134/135, not 135/135 — and the corrected claim is now in the record.
- **Why it matters more than its size.** The cost evidence pointed the right way ($4.93 →
  $0.0975 is overwhelmingly reuse) and would have carried the claim indefinitely. A
  cost-shaped observation cannot distinguish "all reused" from "almost all reused", and the
  difference is the one a reproducibility reviewer would ask about. Only per-artefact hashing
  answered it.
- **Standing consequence.** A claim that two computations are *identical* is a claim about
  bytes and must be tested on bytes. Aggregate evidence — cost, wall-clock, output equality —
  can support "equivalent" but never establishes "exact", and the two words should not be
  used interchangeably in a methods chapter.

### C-38 · A partial argument list silently truncated a two-benchmark artefact
- **What.** `stratified_report.py` writes `scores/stratified_report.json` containing **only
  the benchmarks named on its command line** (`sys.argv[1:] or ['mmlongbench','docbench']`).
  Running `stratified_report.py mmlongbench` to add the LightRAG arm rewrote the file with
  the DocBench half **deleted**. No warning; the file is written, not merged.
- **How it surfaced.** The Phase C notebook failed at an unrelated, previously-passing cell
  with `KeyError: 'docbench'`. Had the notebook not been rebuilt, the truncation would have
  sat in the artefact directory looking like a valid report.
- **Fix.** Re-ran with no arguments, restoring both benchmarks; verified LightRAG is present
  on MMLongBench and correctly **absent** from DocBench (the loader skips arms with no
  generation file, which is the right behaviour and now confirmed rather than assumed).
- **Why it is in the register.** It is C-26's shape — an artefact destroyed by a routine
  command — but with a sharper edge: `ln -sf` at least looks destructive, whereas passing one
  argument to a report generator looks like *narrowing the output*, not *deleting the rest of
  the file*. Nothing warned, and the loss was silent and complete.
- **Standing consequence.** A script that writes a whole artefact from a subset of its inputs
  must either merge with what is already there or refuse to write a partial file. Reporting
  scripts should be run at full scope, and any narrowing flag should write to a different
  path.

### C-37 · The comparator's page ranking discarded the retriever's own ordering
- **What.** LightRAG returns a median of **26 attributable pages** per question, which must
  be cut to the harness's 3-page context budget. The first implementation ranked those pages
  by **attributed character mass**. That measures *which pages carry the most text among
  those LightRAG touched* — not which pages LightRAG judged relevant. Its `Document Chunks`
  are returned in relevance order, and that ordering was thrown away.
- **Why it is a control violation, not a tuning choice.** Every other arm is reduced to its
  top-3 by a **relevance** score: ColPali by MaxSim, GraMM by graph retrieval score. The
  LightRAG arm was the only one ranked by page length. The comparison was therefore not
  "same everything, different retriever" but "same everything, different retriever, *and* a
  different page-selection criterion applied to one arm only".
- **Effect, measured on `context_pages` alone at zero cost:**

  | ranking | ctx-hit (n=838) |
  |---|---|
  | mass-ranked | 0.1480 |
  | relevance-order | **0.2816** |
  | delta | **+0.1337** |

  Only **175 of 1,091 (16.0%)** page sets were unchanged — the defect was rewriting 84% of
  the retrieval. It also **inverts the qualitative reading**: mass-ranked put LightRAG
  *below* the flat-vector control (0.2327), relevance-order puts it *above*.
- **How it was caught, and what did not catch it.** The drift check passed — genuinely, the
  archived ctx-hit reproduced to 4 decimals. The scorer was canonical and unmodified. The
  generation run reported 1,091 predictions, 2 empty, 0 failures. Nothing in the pipeline
  objected, because nothing in the pipeline knows what a page ranking *ought* to be. What
  triggered the audit was the **direction of the result**: a competitor landing below the
  internal control is the shape this project is least motivated to challenge, and is
  therefore the shape that must trigger a harness audit before it is written down.
- **Fix.** Rank by the retriever's own chunk order; character mass breaks ties only.
  `PAGE_RANKING` stamped into `RUN_CONFIG`. The invalid generation and scores are retained
  under `tier1/_invalid_massranked/` rather than deleted, so the corrected result can be
  compared against the artefact it replaces.
- **Cost of the defect.** One wasted generation run (~$4.15) — paid because the harness flaw
  was found *after* generating rather than before. Cheap relative to reporting it.
- **Standing consequence.** When an arm's native output must be reduced to the harness's
  budget, the reduction is part of the control and must use the arm's own relevance
  ordering. Any reduction criterion applied to one arm and not the others is a second
  independent variable, which is C-35's shape at the scoring stage rather than the indexing
  stage — the same lesson, in the same arm, one pipeline stage later.

### C-36 · Module-level globals silently defeated the C-35 fix, and the fix looked like it worked
- **What.** `lightrag.kg.shared_storage` keeps its state in **module-level globals**
  (`_shared_dicts`, `_init_flags`, `_initialized`, …) keyed by workspace, and the default
  workspace is empty. Constructing a second `LightRAG` in the same process therefore reuses
  the **first** instance's in-memory namespaces. Indexing 135 documents into 135 separate
  working directories from one process accumulated every document into one shared namespace,
  and each store persisted the accumulation.
- **Measured contamination**, in indexing order (ascending by size):

  | position | documents in store | characters | own characters |
  |---|---|---|---|
  | 1 | 1 | 1,141 | 1,004 |
  | 2 | 2 | 4,501 | 3,122 |
  | 46 | 45 | 607,632 | 27,788 |
  | 135 | **132** | **9,890,495** | 790,976 |

  Store N held N documents. The final store held essentially the entire corpus.
- **Why this is the worst of the three failure shapes so far.** C-35 was a *design* error I
  caught by reasoning. C-36 defeated the fix for C-35 **through a mechanism I did not know
  existed**, and left behind every artefact that would ordinarily evidence success: 135
  directories, 135 graphs, 135 `_INDEX_COMPLETE.json` stamps — including the stamp added
  *specifically* so that resume would not infer success from a directory's existence. The
  stamp was not wrong. The document really was indexed. It simply did not say that 134
  others had been indexed alongside it.
- **What the contamination would have done.** Retrieval scope would have varied
  **systematically with document size**, because indexing ran size-ascending: small documents
  retrieve over few competitors, large ones over the whole corpus. That is worse than either
  clean design — it is a confound *correlated with the stratifying variable* of the project's
  central length-scaling analysis.
- **How it was caught.** Not by the exit code, the stamps, the graph sizes, or the cost —
  cost matched the projection to 2.5% ($4.9330 against $5.06), because extraction still ran
  once per document. It was caught by the **alignment rate**, an integrity number added to
  measure something else entirely: retrieved chunks totalled 104,560 characters against a
  33,661-character document. Content cannot exceed its source. That arithmetic impossibility
  is the whole detection.
- **Fix.** `finalize_share_data()` after each document, which resets the globals in
  single-process mode. Verified by assertion, not by inspection: every store must contain
  exactly **one** document. Re-index seeded with the accumulated extraction cache — keys are
  hashes of the extraction prompt, so a hit is exact reuse of the same computation on the
  same input, not a substitute for it — which reduced the re-run from $4.93 to near zero.
- **Standing consequence.** A library's per-instance API does not imply per-instance state.
  When constructing several instances of a stateful component in one process, assert
  isolation by **measuring a property that must differ between them** — here, document count
  per store — rather than trusting that separate constructor arguments produced separate
  state.

### C-35 · Tier 1 was indexing into a corpus-wide store, which would have confounded scope with method
- **What.** The first Tier 1 indexing run built **one shared LightRAG store** over all 135
  MMLongBench documents. Every other arm in this project retrieves **within the question's own
  document**: ColPali's `retrieve.py` loops `by_doc` and ranks only that document's pages, and
  GraMM and flat-vector retrieve over that document's element graph. MMLongBench questions are
  document-scoped by construction.
- **The confound.** A corpus-wide store makes LightRAG solve a strictly harder problem —
  locate the *document* and then the passage — so retrieval would have differed in **scope**
  as well as in **method**. That is two changes at once in a comparison whose entire claim is
  that only one thing differs. The resulting deficit would have been read as "LightRAG
  retrieves worse", and it would have been partly an artefact of asking it a harder question
  than every arm it was being compared against.
- **Why it would not have announced itself.** Nothing errors. The store builds, retrieval
  returns context, generation produces answers, and every metric computes to a plausible
  number. The arm would simply have scored low, in the direction that flatters this project's
  own system — which is the direction a result is least likely to be challenged from.
- **Caught by** working forward from the harness-identity rule to the *next* stage rather than
  only the current one: writing the retrieval step forced the question "retrieval over what
  scope?", and the answer had already been decided wrongly one stage earlier.
- **Fix.** One store per document (135 stores), with a per-store `_INDEX_COMPLETE.json`
  completion stamp so resume never infers success from a directory's existence (R-42's shape:
  a cache that reports success for artefacts it should reject). `RETRIEVAL_SCOPE` stamped into
  `RUN_CONFIG`. Indexing cost is unchanged — the same text is extracted either way — so the
  correct control was not the expensive one; it was merely the one that had to be noticed.
  Sunk cost of the discarded run: **$0.14**.
- **Standing consequence.** When adding an arm, the controlled variable list must be checked
  against **every** stage the arm touches, not only the stage being written. Scope, budget,
  ordering and units are all things that can differ silently between arms while the named
  method difference is the only one anyone is watching.

### C-34 · The sprint brief predicted the wrong binding constraint for the graph bracket
- **What was predicted.** `SPRINT_BRIEF_SEP2026.md` §2b: *"The long pole is comparator
  indexing wall-clock, and it is unmeasured."* The whole V2 pilot requirement was built
  around that premise, and the Sep 18 risk valve was specified as a **wall-clock** valve.
- **What was measured.** Wall-clock was never close to binding. LightRAG indexes all 364
  documents in **3.30 h** at its default insert concurrency — against a 12-day window. Cost
  bound instead, and bound hard: **$23.05** against a $10 gate, and **$136.73** for the
  multimodal arm. The valve that actually fired was the one nobody designated as a valve.
- **Why it is worth recording rather than shrugging off.** The anchor itself was sound and
  correctly sourced — GraMM's own measured 37.6 s/document, from the project's own log. The
  error was **enumerating one resource axis and not the other**: reasoning from a *time*
  anchor to a *time* conclusion, when the same pilot that measured time would measure cost.
  The pre-registration escaped the consequence only because Gate G happened to specify a
  cost ceiling *and* a wall-clock deadline; had it inherited the brief's framing and gated on
  wall-clock alone, both comparators would have been **authorised**, and the $136.73 arm would
  have been discovered by spending it.
- **Attribution.** The prediction and the anchor were the external reviewer's, and are recorded
  here as their miss at their own instruction. Note the class: the reviewer's four previously
  logged errors were all **provenance** failures (reusing a number without checking where it
  came from). This one is not — the provenance was impeccable. It is an **incomplete axis
  enumeration**, and it is the first reviewer error in this register that breaks that pattern.
  `METHODS_CHAPTER_ARTEFACTS.md` §6 is amended accordingly, because a claim of monotony that
  has acquired a counter-example should stop being stated as monotony.
- **Standing consequence.** A feasibility gate states a threshold on **every** resource the
  work consumes — wall-clock, money, storage, and reviewer time — or it is not a feasibility
  gate, it is a gate on the resource that happened to be salient when it was written.

### C-33 · A parser-installation gate silently disabled the entire multimodal path
- **What.** `RAGAnything.insert_content_list()` — the pre-parsed ingest route, which by
  construction never invokes a parser — begins with `await self._ensure_lightrag_initialized()`.
  That method checks parser installation **first** and, on failure,
  **returns `{"success": False, "error": ...}` rather than raising**. `insert_content_list`
  discards the return value and proceeds. `_initialize_processors()` is therefore never reached,
  `modal_processors` stays `{}`, and every image is dropped with
  `WARNING: No processor found for type: image`.
- **Why it is the C-24 class, not a mere bug.** The run **exits 0**. The document is marked
  complete. A graph is built, entities are extracted, and a per-document cost is reported. On the
  p5 pilot document the multimodal arm reported **$0.0041 and 0 vision calls** against 36 images
  present on disk — a clean, cheap, complete-looking result from an arm that was silently running
  **text-only**. Reported as-is it would have made the multimodal comparator look ~56× cheaper
  than it is, and its quality numbers would have been a text-only system wearing a multimodal name.
- **How it was caught.** Not by the exit code, the logs' INFO stream, or the cost line — all
  three looked healthy. By reading the **vision meter**: 36 images in, zero vision calls out.
- **Fix.** Two parts, and only the second generalises.
  1. Assert the parser gate is moot (`_parser_installation_checked = True`) and initialise
     processors explicitly. MinerU is deliberately **left uninstalled**, so re-parsing stays
     *structurally impossible* rather than merely discouraged — which is what makes the
     harness-identity claim (same parse to every arm) enforceable rather than asserted.
  2. A **blocking validity assertion** in the pilot: a document containing images that produces
     zero vision calls raises and halts the run. This is the same shape as the standing
     retrieval-metric-differs assertion — a multimodal arm making no vision calls is not wired,
     and that must stop the run rather than reach a table.
- **Impact.** No published result was affected: the defect was found in the pilot, which is what
  the pilot is for. Cost consequence in the other direction is large — see `GATE_G_RESULT.md`.


---

## Category C — Specification corrections to the brief
*(The brief was wrong or under-specified; recorded so the write-up does not inherit it.)*

### C-12 · T-C2's page-image estimate counts context pages, not corpora — **10× low on DocBench**
- **Defect.** The brief sizes the ColPali index from `cache/results/page_images/`
  ("MMLongBench ~6.4k pages ≈ 3.4 GB; DocBench ~1.4k pages ≈ 0.7 GB"). Those directories
  hold 1,546 and 1,834 images.
- **Root cause.** The pipeline rendered images only for pages **its own retrievers
  selected** for the generator (`USE_PAGE_IMAGES=True` acts on context pages). They are
  an output of our retrieval, not a copy of the corpus.
- **Why this is fatal if missed.** Indexing only those pages would confine ColPali to
  pages GraMM/flat-vector already found. Gate C would become unfalsifiable by
  construction — ColPali could never beat us on localisation because it could never see
  a page we missed.
- **Evidence.** True corpus sizes from `qmeta` `doc_pages` (whose page counts reproduce
  the archived `difficulty_bins` exactly): MMLongBench **6,522** pages / 135 docs;
  DocBench **14,008** pages / 229 docs. Source PDFs are present for both (135 and 229).
- **Fix.** `gramm-rag-colpali/render_pages.py` rasterises every page of both corpora at
  150 DPI to `page_images_full/<bench>/<doc_id>_p<N>.png`, 1-based to match the
  `gold_pages` convention (R-9 established the 0-based parser vs 1-based gold hazard).
- **Impact on budget.** None: the ~$10 is *generation* cost, which is per question, not
  per page. Index storage rises ~4 GB → ~10 GB against 806 GB free; GPU and rendering
  time rise.

### C-13 · RAG-Anything's judge prompt is not extractable from the paper
- **Defect.** The brief instructs T-C1 to use "their published Appendix A.4 prompt".
  Appendix A.4 renders the prompt as **Figure 10, an image** — the text is not in the
  HTML or PDF.
- **Fix.** Took the prompt verbatim from the authors' own evaluation code instead:
  `HKUDS/RAG-Anything`, `reproduce/llm_answer_evaluator.py`
  (`get_accuracy_evaluation_prompt`, line 46), sha256
  `40f812a19bf866eb867af504bcd22cabce6cf3c5fc0e57e56aecf5ce0da3651c`, 79,241 bytes,
  retrieved 2026-08-21. Config: `gpt-4o-mini`, temperature 0.1, max_tokens 1000, system
  prompt "You are an expert evaluator…", output JSON `{"accuracy": 0|1, "reasoning": …}`.
- **Why this source is stronger.** It is the code that produced their published numbers,
  not a typeset rendering of it.
- **Provenance.** Archived at `vol3/grammrag/_reference/raganything/` with
  `PROVENANCE.md`.
- **Impact on results.** Makes T-C1 reproducible. Inventing an approximation would have
  produced a plausible positioning number with no defensible meaning.

### C-14 · Reconciliation item R1 resolved
- **Item.** `PHASE_A_CURRENT_WORK.md` §7 R1: the published MMLongBench-Doc reports 1,082
  questions / 223 unanswerable; our runs use 1,091 / 244. The action was to "identify the
  source of the 9-question difference and state it explicitly".
- **Resolution.** The difference is exactly **9 questions flagged answerable that carry an
  empty `gold_pages` list**. A second, separate irregularity: **7 questions flagged
  unanswerable that do carry gold pages** (this is why our unanswerable count is 244 and
  not 223).
- **Handling.** The 9 are kept as an explicit fourth stratum in A1 rather than dropped or
  folded into either bucket. T-C1 reports MMLongBench on both N=1,091 and their N=1,082.
- **Impact on results.** Immaterial: dropping the 9 moves `judge_accuracy_ragany` from
  0.3584 to 0.3586. R2–R7 remain open; all are writing actions in §7, which the brief
  places outside the execution agent's scope.

---

## Category D — Deliberate deviations from the brief
*(Recorded so they are visible rather than silent.)*

### C-15 · `judge_accuracy_ragany` written to new files, not into the result JSONs
- **Brief.** T-C1 verification says "both fields present in every result JSON".
- **Deviation.** The re-judge writes `results/ragany/ragany_judge_<bench>_<arm>.json` and
  leaves the archived result JSONs untouched.
- **Rationale.** The archived runs are on the do-not-touch list. Mutating them to add a
  positioning metric risks the canonical artefacts for no analytical gain; the joint
  notebook joins the new files by index. The requirement's intent — both metrics
  available side by side, neither replacing the other — is met.

### C-16 · `prep_qmeta.py` added to preserve the joint folder's self-containment
- **Situation.** A2 needs MMLongBench `doc_type`, A3 needs per-document page counts, A4
  uses `evidence_sources`. None are in the archived result JSONs; all live in the sibling
  benchmark repos.
- **Deviation.** Rather than have the notebook reach across repos at run time (its README
  promises `data/` is a complete copy), a prep script resolves them **once**, proves the
  joins, and freezes them into `data/<bench>/qmeta_<bench>.json`.
- **Checks it enforces.** MMLongBench index alignment proven on golds (0 mismatches) and
  on `evidence_pages` (0 after parsing the JSON-string field); DocBench ordering proven to
  be `sorted(iterdir())` **string** order (0 question mismatches); page counts must
  reproduce all five archived `difficulty_bins` (N and dANLS) or the script exits.

### C-17 · A1 gained an interaction test beyond its stated deliverable
- **Deviation.** §1 asks for "Δ per stratum, each with its own paired bootstrap 95% CI".
  That cannot adjudicate the hypothesis, which is about *concentration*: two overlapping
  CIs are not a difference test, and "cross-page significant, single-page not" is the
  difference-in-significance fallacy.
- **Addition.** An unpaired bootstrap on the contrast Δ_cross − Δ_single, with its own CI,
  for both the answer metric and localisation. The Gate A verdict is derived from it
  programmatically rather than written by hand.
- **Also added.** Per-stratum baseline level and relative Δ, because cross-page questions
  start from a lower base (0.1243 vs 0.2248) and a larger absolute Δ on a harder stratum
  can be a floor artefact.

### C-18 · T-C1 cost ran 32% over estimate
- **Estimate.** ~$0.30. **Observed.** $0.3958 total ($0.0180 calibration + $0.3778 full),
  at $0.000090/call over 4,386 calls.
- **Handling.** The brief's §7.3 rule is to stop and report if observed exceeds estimate
  by >50%. 32% is inside that, so the run proceeded after the 50-question calibration
  measured the true per-call cost.
- **Note.** The brief's "~2,193 calls" undercounts by half: positioning requires judging
  **both** arms on both benchmarks (1,091×2 + 1,102×2 = 4,386).

---

## Category E — Results recorded

### C-19 · Gate A — pre-registered cross-page hypothesis **NOT SUPPORTED**
Δ_cross − Δ_single = **−0.0245**, CI [−0.0687, +0.0216]; localisation contrast −0.0468,
CI [−0.1074, +0.0102]. Both include zero.

| Stratum | n | Base ANLS | Δ ANLS | 95% CI | ctx-hit Δ |
|---|---|---|---|---|---|
| single-page | 478 | 0.2248 | +0.0714 | [+0.0397, +0.1046] | +0.2218 |
| cross-page | 360 | 0.1243 | +0.0470 | [+0.0134, +0.0803] | +0.1750 |
| unanswerable | 244 | 0.7869 | **−0.0574** | [−0.0943, −0.0205] | n/a |
| empty gold_pages (R1) | 9 | 0.4074 | −0.0741 | [−0.2222, +0.0000] | n/a |

Consequence per §13's routing table: the architectural claim narrows — GraMM is a general
retrieval improvement, not a cross-page mechanism — and Phase B should lead with the
conversion-gap / calibrated-abstention story rather than cross-page chaining (B4).

**Unplanned finding.** GraMM regresses *significantly* on unanswerable questions. This
gives F-3 (τ collapsing to 0.0; the reward cannot detect answerability) a measured cost:
failing to abstain is not a missed opportunity, the extra retrieved context actively
degrades unanswerable handling. It also explains why the pooled delta (+0.0334) sits far
below either answerable stratum.

### C-20 · A2–A5
- **A2.** No category regresses on either benchmark. MMLongBench: all 7 `doc_type`s
  positive, only Tutorial/Workshop (+0.055) and Financial report (+0.059) clear zero.
  DocBench: text-only **+0.097** (significant); una-web scores 0.0 on both arms (n=7).
- **A3.** Localisation Δ is positive in every length bin on both benchmarks, but
  MMLongBench's answer Δ turns **negative (−0.0087) at 101+ pages** (n=115) while ctx-hit
  is still +0.0989. DocBench converts in all five bins. The conversion gap is therefore an
  extreme-length MMLongBench phenomenon, **not** a general property — narrower than the
  plan assumed.
- **A4.** The SigLIP figure branch does not demonstrably earn its complexity. DocBench's
  advantage is largest and only significant on **text-only** (+0.097); multimodal-t
  (+0.064) and multimodal-f (+0.034) are not significant. MMLongBench `evidence_sources`
  agrees independently: Table +0.094, Chart +0.049, Pure-text +0.049 significant, while
  **Figure (+0.034) is not**. (Multi-label; those rows overlap and do not sum to N.)
- **A5.** Retrieval control **holds bit-identical**: all six metrics (MMLongBench
  appa@1/@5/@10, DocBench quote_recall@1/@3/@5) are exactly equal between the superseded
  (R-50 / D-35) and current runs, Δ = 0.0. Answer-quality changes came from generation and
  context assembly, not retrieval drift.

### C-21 · T-C1 — positioning, and judge robustness
Positioning is **indicative only**; two of five differences (generator, one-sentence
answer constraint) remain unresolved.

| Benchmark | Flat-vector | GraMM-RAG | RAG-Anything (published) |
|---|---|---|---|
| MMLongBench-Doc (their N=1,082) | 0.3198 | 0.3586 | **0.428** |
| DocBench (N=1,102) | 0.5145 | 0.5771 | **0.634** |

Our scores *rose* under their judge (DocBench baseline 0.4728 → 0.5145), so the remaining
gap is not judge harshness.

The controlled contrast — same predictions, same questions, only the judge changes — is
that **the verdict survives the judge swap**: MMLongBench Δ +0.0334 → +0.0376
(CI [+0.0174, +0.0596], McNemar p=5.23e-04); DocBench +0.0599 → +0.0626
(CI [+0.0390, +0.0844], p=2.34e-07). Both still exclude zero. 4,386 calls, **0 failures**.

---

## Standing checks adopted this phase

*23 numbered standing rules plus rule 5b (24 in total).*

1. Every new per-question metric must reproduce an archived aggregate before it is
   trusted (C-8, C-9).
2. A verification assertion must be capable of failing — identities that hold by
   arithmetic are not checks (C-7).
3. Every figure must be **looked at**, not only its underlying table verified (C-10, C-11).
4. Any corpus-scale input must be confirmed to be the corpus, not an output of our own
   pipeline (C-12).
5. Any externally loaded model must assert `missing_keys == []` **and** pass a
   behavioural check before use. A model with randomly initialised weights returns
   plausible numbers and fails silently (C-22).
   **5b. For lazily-initialised modules, `missing_keys == []` is not weak evidence — it is
   unavailable evidence.** PyG's `HGTConv` does not materialise its parameters until a
   forward pass, so static weight inspection is *structurally impossible*, not merely
   unreliable. The framework forces a behavioural check rather than merely rewarding one.
   Verify by comparing the checkpoint against a same-shape fresh random init on identical
   inputs (C-30: APPA@10 0.806 vs 0.448).
6. **A text matcher is a measuring instrument and must be validated against the data it
   will measure**, not reused because it exists. The v1 decline regex missed "does not
   include/contain", the dominant phrasing on DocBench, understating its decline rate
   **17x** (0.040 vs 0.702) and producing a spurious "DocBench measures something
   different" reading that was then reasoned from (C-27).
7. **Never quote an unanswerable-stratum accuracy without stating it is equivalent to an
   abstention rate on that benchmark.** On both corpora the gold answer to an unanswerable
   question is a statement of non-availability, so the two agree on 0.92-1.00 of items.
   They are one measurement in two units and must never be presented as converging
   evidence. `stratified_report.py` prints this rule and the agreement figures on every run.
8. **A joint model can never score below the best single feature it contains.** If it does,
   the fit or the metric is broken -- check `AUC(x)+AUC(-x)==1` first (C-29).
9. **Cardinality is not informativeness.** A low-cardinality feature bounds resolution, not
   information; never infer "degenerate" from "few distinct values" (C-29).
10. **Any comparison of a policy against a signal must state the signal's ceiling.** A
   calibration reaching 0.55 where the best possible single threshold is 0.560 has hit the
   limit of the signal, not failed (C-28).
11. **Any externally supplied figure is unverified until traced to a measurement.** Three
    numbers taken from a planning list (`PHASE_A_CURRENT_WORK.md` §12, since marked
    contaminated) were wrong by 6.7×, ~100× and ~1.7×. Check provenance before use, and
    apply this to figures supplied by reviewers as well as by the team (C-31).
12. `df -h /` before and during every image build; stop at 60% (C-4).
13. **A required-reading file must have one canonical copy, or an assertion that its
    duplicates match.** A stale `HANDOFF.md` misstated three settled results through weeks
    of review because the package root and `GraphRAG_baseline/` held separate copies:
    everyone read one, nobody re-derived the other. This is C-12's shape applied to
    documentation — an input everyone trusts and nobody checks. `sync_handoff.sh` now fails
    the sync if duplicated required-reading files differ, if the register's stated entry
    count disagrees with the register, or if a secret appears in the package.

14. **Pinning a package is not pinning a system: pin the transitive dependencies too, to
    versions contemporaneous with the publication being reproduced.** `raganything==1.2.8`
    declares a bare, unbounded `lightrag-hku`, so the exact-looking pin would have installed
    a LightRAG eleven months newer than the one that produced the numbers we are sitting
    beside — while the write-up's version string stayed correct (C-32). Record the release
    date evidence for each pin, not just the version. The general shape: a dependency that
    is *named* but not *bounded* is an unstamped config knob, which is R-42 with a different
    file extension.

15. **A gate must enumerate every resource axis on which it could fail, not only the axis the
    prediction named.** The sprint brief predicted indexing *wall-clock* as the binding
    constraint on the graph bracket and specified the Sep 18 risk valve as a wall-clock valve.
    The measured binding constraint was **cost**, and not narrowly: 3.30 h against a 12-day
    window, versus $23.05 and $136.73 against a $10 ceiling (C-34). Gate G survived only
    because it happened to state a cost ceiling **and** a deadline; had it inherited the
    brief's framing and gated on wall-clock alone, **both comparators would have been
    authorised** and the $136.73 arm discovered by spending it. The anchor behind the wrong
    prediction was sound and correctly sourced — this is not rule 6 in a new costume. The
    failure is upstream of verification: choosing *what to check*. Before a gate is written,
    enumerate the resources the work consumes — wall-clock, money, storage, memory, and
    human review time — and state a threshold or an explicit "not binding, because …" for
    each. Cheap insurance: a gate silent on an axis cannot fail on it.

23. **A metric name is not a definition. Quote a metric with the INPUT SET it scored, and never
    contrast two arms scored on different ones.** Two arms can be scored by the same function, at
    the same cut-off, on the same denominator, with the same name, and still not be comparable,
    because the inputs came from different stages of the pipeline. DocBench "quote-recall@5,
    n=730" meant the top-5 **retrieved** pages for GraMM-RAG and Flat-vector and the pages the
    **generator saw** for every external arm; the contrast between them was published as a
    significant reversal (+0.0534, p=0.0013) and is null when both are scored the same way
    (−0.0027 [−0.0342, +0.0288]) — C-77.

    **Why the usual guards do not catch it.** Every arm reproduced its own archived value
    exactly, so every drift check passed; the denominators matched, so every n agreed; the metric
    name was identical, so nothing looked inconsistent. The defect is not in any arm's number but
    in the *pairing*, which no per-arm check inspects. The operative form is therefore a property
    of the table, not of a cell: **a localisation column may contain only values computed from
    the same pipeline stage, and the stage is named in the column header.**
    `STRATIFIED_REPORT.md` prints both definitions side by side for exactly this reason, and
    `reports/build_reports.py` fails if they ever converge, which would make the distinction
    invisible again.

22. **Prefer guards that emit an observable on every run over guards that speak only on
    failure. A measurement is proof of liveness; silence is evidence of nothing.** This is the
    constructive form of rule 21's corollary — *a guard whose failure mode is silence cannot be
    distinguished from a guard that is working.* A guard printing 0.98, 0.99, 0.97 per document
    is self-verifying: you can see that it ran. A shim that stays quiet is indistinguishable from
    a shim that was never reached, which is exactly C-52.

    **Two tiers, and the register's own guards separate on both.** The first draft of this rule
    was "value-emitting guards catch things, boolean guards do not." Checking it against this
    project's actual guards falsified the simple version and produced a better one.

    | tier | property | what it buys | evidence in this register |
    |---|---|---|---|
    | **1. liveness** | announces itself **every run**, pass or fail | you can tell the guard executed at all — defeats unreachability | `sync_handoff.sh` prints an `ok:` line per check on every run; ASSERTION 3 caught a stale register count and ASSERTION 4 caught a live Jupyter token in the handoff, **both reporting only `ok:` on success** |
    | **2. readable value** | emits a **number** on every run | the value can be read against invariants the guard was never built to test | `alignment_rate` was built to report page-attribution coverage; **0.4995 against a sibling arm's 0.9765 exposed C-49**, and an arithmetically impossible rate exposed **C-36** — neither was the property it was written for |

    **The simple version is false and the refined one is the point.** Boolean guards are not the
    problem — two of this project's boolean-reporting checks have real catches to their name. The
    problem is guards that are **silent on success**, because silence and non-execution are the
    same observation. Tier 1 is the necessary property; tier 2 is what makes a guard catch things
    it was not designed for.

    **Tier 2 explains the incidental transfers, which is the strongest evidence for it.** A guard
    emitting a value offers a *surface* — the number can be checked against an unrelated
    invariant by anyone who sees it. A boolean offers no such surface: `ok` cannot be compared to
    anything. Every incidental catch in this register came through a value read against something
    it was not built for; none came from a boolean firing unexpectedly, because a boolean cannot
    fire unexpectedly.

    **The two guards that manufactured false assurance are silent-on-success.** C-7's tautological
    identity printed nothing meaningful because it could not fail; C-52's shim would have printed
    nothing because it was never reached. Neither offered liveness *or* a value.

    **Class B.** A reader applies it to a paper they did not write: for each validity check
    described, does the paper report what it *observed*, or only that it passed? "We verified X"
    is a boolean with no liveness evidence; "X measured 0.98 (threshold 0.95)" is a check the
    reader can audit and reuse.

21. **A guard must signal on a channel the code under test cannot handle. Any system with
    error recovery will absorb signals sent through channels it owns.** This is the general form;
    "assert on content, not control flow" is one instance of it, and stating only the instance
    misses most of the cases.

    **The channel is the thing to check, not the assertion.** In C-52 three candidate guards all
    signalled through channels MMGraphRAG owns, and all three are absorbed by the same broad
    `except Exception`:

    | guard | channel | outcome |
    |---|---|---|
    | `mineru` shim exiting non-zero | **process exit code** | caught as `CalledProcessError` → routed to the fallback it guarded |
    | pre-flight check raising | **exception** | caught by the same handler → same fallback |
    | comparing store chunks to the injected markdown **after** the call | **content, outside the call** | works — the system has no handler for a fact about its own output |

    **The failure generalises well past parsers.** A guard writing to a log the system rotates,
    a guard setting a flag the system resets on retry, a timeout the system retries past, a
    sentinel file the system cleans up, a non-zero status a wrapper script swallows — each is the
    same shape: **the signal is delivered into a subsystem with authority to discard it.** Error
    recovery is precisely the authority to discard error signals, so the better a system's
    recovery, the more channels it owns and the fewer remain usable for guarding it.

    **The operational test, applicable before writing the guard:** name the channel your guard
    signals on, then ask whether the code under test has a handler for it. If it does — or if it
    might acquire one, since `except Exception` is one refactor away in most code — the guard is
    advisory, not blocking, and must be recorded as such. Prefer a channel the system cannot
    reach: an assertion outside its call frame, on an artefact it has already produced.

    **Corollary, and it is the part that makes this checkable rather than merely wise:** a guard
    whose failure mode is *silence* cannot be distinguished from a guard that is working. Both
    look identical from outside. So a guard on an owned channel does not merely fail — it fails
    **invisibly**, which is C-52's unreachability property and the reason guard defects produce
    false confidence rather than wrong numbers (§7.4b).

    **Class B.** A reader applies it to a paper they did not write: for each validity check the
    authors describe, what channel does it signal on, and does the system under test have a
    handler for that channel? A check that signals through the system it is checking is not
    evidence the check passed — it is evidence the check was not heard.

20. **Guards have direction, and their direction depends on what they are pointed at. Before
    invoking one, ask which way it points relative to your own interest — a guard that protects
    your prediction, your method or your system is being misused.** C-41 forbids claiming more
    than an interval supports. Aimed at a **world-claim**, it fires *against* the author: it
    withdraws a superlative they wanted. Aimed at *"my registered prediction failed"*, the same
    rule fires *for* them: what it suppresses is now a conclusion that **costs** them. Identical
    wording, inverted direction, and **nothing in the wording distinguishes the two cases** —
    which is precisely why the misuse passed unnoticed (C-51).

    **The check is directional, not semantic**, and that is what makes it usable. Reading the
    guard more carefully does not help; the guard was applied correctly on its own terms, to a
    real interval containing a real threshold. What separates use from misuse is the **relation
    between the guard's output and the author's interest**. Every instrument in this register was
    built to fire against its author. **So one running the other way is diagnostic on its own,
    whatever its wording says** — and that test can be applied without re-reading the rule, which
    matters because re-reading it is the step that failed.

    **Scope, stated so it is not overclaimed.** This does not say a guard may never produce a
    convenient result — sometimes the honest answer is the one the author wanted. It says a
    convenient result from a guard is a **flag requiring the reasoning to be stated explicitly**,
    not a licence to proceed quietly. Convenience shifts the burden; it does not settle it.

    **Class B.** A reader applies it to a paper they did not write, with no access to the
    pipeline: for each methodological caveat the authors invoke, does it withdraw a claim they
    wanted, or suppress a conclusion against them? A paper whose caveats consistently run the
    second way is reporting selectively **through its rigour**, which is harder to see than
    selective reporting through its results — and is invisible to every arithmetic check.

19. **A composite metric that inverts across settings must be decomposed into its components
    before the inversion is interpreted.** Abstention balanced accuracy reversed between corpora
    (Kendall's tau-b −0.828) while localisation and answer quality stayed concordant (+0.828,
    +0.600). Balanced accuracy is `0.5×(TPR+TNR)` — **the only reported metric mixing two error
    types whose relative sizes depend on the corpus**. Decomposition showed the inversion was
    driven by the TNR term (false declines on answerable questions, 0.139→0.228 for text/graph)
    and not by the detection term it appears to be about. Interpreting the composite directly
    would have produced a claim about *detecting unanswerability* that the components contradict.
    **Class B:** a reader asks of any paper whether an inverting composite was decomposed before
    the inversion was explained.

18. **Report every rate as "X per Y, measured on Z", and state what else changes whenever a
    ratio is compared across conditions.** The referent family — five occurrences across two
    analysts, the largest in this register — is invisible to arithmetic checking: every figure
    was correctly sourced and every computation right. It is catchable at **writing time**,
    because each instance omits one of three things adjacent to the number:
    - **unit** — "4.7 s/page" omitted the *device* (consumer GPU, applied to A100-hours);
    - **denominator** — "13 GB per corpus" omitted *what was in it* (92.5% a cache counted 135
      times); "over 20,530 pages" omitted that the loop iterates 14,581 crops;
    - **condition/scope** — "$18.00" omitted *which stages* (indexing only, not generation or
      judge); "the long pole is wall-clock" omitted the other resource axes (C-34).

    And the ratio clause, generalised from C-43: a ratio compared across two conditions must
    state what else varies with the condition. Where no common definition exists, the correct
    output is **"not measurable here"**, not a corrected number.

    **A sixth occurrence, and it names the question that catches the whole family.** "$92 was
    S2's cost" conflated the constrained model's **prediction** ($92.09) with the run's
    **measurement** ($100.49). The omitted referent was not unit, denominator or scope but
    **kind**: a prediction and a measurement of the same quantity, in the same units, on the same
    corpus, differing only in whether the number had happened yet. Arithmetic checking cannot see
    it — both numbers are real, both correctly sourced.

    The catching question generalises the three bullets above and this one: **ask "of what?" of
    your own sentence.** Of what device, of what denominator, of what stages, and — the case this
    adds — *of what status: predicted or observed?* A number's kind is part of its referent.

    Worth noting because it cuts the other way: conflating them here would have made the reported
    figure *closer* to the model's prediction and the model look better than it was. The
    measured +8.4% miss is what actually vindicates the constrained estimator against the
    phantom-term ones — a real miss on a well-specified model is stronger evidence than an
    apparent hit, because it shows the model was scored rather than fitted.

    **Class B, all clauses.** A reader applies them to someone else's paper with no access to
    anything: does each rate name its unit, denominator and condition? Does each cross-condition
    ratio say what co-varies? Does each number say whether it was predicted or observed? This is
    what turns the family from *the failure you should expect to make* into *the failure you
    should expect to make, and the line that catches it*.

17. **A conclusion over a set must print the set it used and assert its size; and an ordering
    claim must state the COMPARISON IN WORDS, not merely display its inputs nearby.**
    *Amended after C-47.* The first form required the interval to appear **adjacent** to the
    ordering. That is insufficient, and its own counterexample is instance 5: the interval
    `[0.5689, 0.6224]` was printed **two lines above** the sentence claiming the value was
    "still below the 0.614 threshold". The rule was satisfied and the claim still failed,
    because **availability is not consultation** — putting a number nearby makes it available;
    reading it and checking a claim against it are separate acts, and composing prose invokes
    the first without the second.
    Required form: *"0.5956, CI upper 0.6224, which does not exclude 0.614"* — claim and check
    in **one sentence**, so failing to compare requires writing something visibly incomplete
    rather than merely not looking up. Where intervals overlap, the claim is stated as **"not
    distinguishable"** rather than ordered.
    The interval clause was added after four instances of one shape — an ordering claim made on
    point estimates that the intervals do not support:
    - **M6 (1st)** "best decline-decision accuracy" where the table said **second**;
    - **dimension B** "the only arm whose advantage goes negative at 101+" where GraMM's CI is
      [−0.0437, +0.0261] and **spans zero**;
    - **M6 (3rd)** "smallest rank shift" where a ±1.0 shift over six systems has **no interval
      at all**;
    - **M9** a magnitude asymmetry (+0.1527 vs +0.0534) across **incommensurable denominators**
      (ctx-hit n=838 vs quote-recall@5 n=730).

    Each was true of the point estimates and unsupported as stated. All four would have been
    caught at writing time by requiring the interval beside the ordering.

    **Class B, both clauses.** A reader checks any published table: does each "best" sit beside
    its ranking? Does each ordering sit beside intervals, and are overlapping ones reported as
    indistinguishable rather than ranked? Three failures where the
    computation was correct and the *sentence* was not: M6 said "best" where the table said
    second; C-40 said "every arm … and a vision retriever" with two arms of five analysed and
    no vision arm present; the C-39 draft called a numerical coincidence the entry's strongest
    evidence. Reviewers check tables, and in each case the table was right — so this class
    passes review that the other three fail. **Class B**: a reader needs no pipeline access to
    check whether an "every system" claim names its systems, or a "best" sits beside its
    ranking.

16. **Aggregate evidence can establish *equivalent*; only per-artefact evidence can establish
    *exact*.** The C-36 re-index cost $0.0975 against an original $4.93 — a 98% reduction that
    is overwhelming evidence of cache reuse, and was reported as *exact* reuse. It was not:
    hashing all 135 caches showed 134 exact and one document re-extracted fresh (C-39). Cost,
    wall-clock and matching outputs are **aggregate** signals; they cannot distinguish "all"
    from "almost all", and that distinction is what a reproducibility reviewer asks about. A
    claim that two computations are identical is a claim about bytes and must be tested on
    bytes. Do not use "exact" and "equivalent" interchangeably.
    **This is a Class B guard** — it needs no access to the pipeline, only the published claim
    and its evidence. It applies to any paper asserting exact reproduction on the strength of
    cost or timing agreement, including papers this project did not produce.
