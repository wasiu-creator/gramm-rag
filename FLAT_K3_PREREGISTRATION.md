# Pre-registration — flat-vector control re-run at k = 3

**Written before any measurement or spend.** Authorised 2026-09-28: all three corpora, ~$14.2
estimated, and the k=3 arm **replaces** the archived k=2 flat-vector as the reported control.

## Why

C-77(ii). The control reads fewer pages than every arm compared against it, on all three corpora:

| corpus | flat-vector | GraMM-RAG | external arms | context chars |
|---|---|---|---|---|
| MMLongBench | 2 | 3 | 3 | flat 8,000 · others 12,000 |
| DocBench | 2 | 3 | 3 | flat 8,000 · others 12,000 |
| MP-DocVQA | 1.92 mean, never 3 | 2.74 | 2.45–2.52 | 12,000 all |

Both localisation metrics score a hit anywhere in the pages given to the generator, so they rise
with the page count. Every flat-vector contrast, and all of η (defined against flat-vector), mixes a
page-budget difference with a retrieval difference. The project already treats this as unacceptable:
**ColPali size-matched exists solely to hold page count fixed** against the Union arm.

## What changes, and what is held fixed

**Changes:** the number of pages the flat-vector generator receives (2 → 3) and, on MMLongBench and
DocBench, its character budget (8,000 → 12,000), matching every arm it is compared against.

**Held fixed** — nothing else may move, or the re-run confounds a second variable:
the E5 node retrieval and its embeddings on disk; `RERANK_POOL = 20`; the cross-encoder
`cross-encoder/ms-marco-MiniLM-L-6-v2`; `PAGE_SCORE_MODE = 'first_seen'`; generator `gpt-4.1-mini`
at `T = 0.0`, `PROMPT_V = 3`; page images (120-dpi renders on MMLongBench/DocBench, the corpus's own
JPGs on MP-DocVQA, 0-based there); the DocBench judge `gpt-4-turbo`; every scorer; `SEED = 42`.

Shared code is reached by **executing the canonical source**, never by copying functions out of it
(the C-49 rule). The driver prints the resolved file of every shared object before it runs.

## Gate 1 — reproduction, free, blocking

The selection path is re-run at **n_pages = 2** and must reproduce each archived context exactly.

- **Threshold, fixed now: ≥ 99.5% of questions identical, per corpus**, page ids and order.
- Below threshold on any corpus: **stop, spend nothing, report**. A selection path that cannot
  reproduce its own published output cannot be trusted to extend to a third page (C-70's rule:
  reproduce a published value through the canonical function before trusting anything derived
  from it).
- The per-question mismatch list is written to the artefact whatever the outcome.

## Gate 2 — cost

Estimated from measured figures: MMLongBench generation ~$4.2, DocBench generation ~$4.2 and judge
~$4.3 (`S1_DOCBENCH_RESULT.md` cost table), MP-DocVQA generation ~$1.5 (P3's ~$6 for five arms).
**Estimate $14.2; hard ceiling $17.** Spend is metered per corpus and the run aborts at the ceiling.
One discarded run is budgeted. Keys are passed by `--env-file`, never on a command line.

## Predictions, registered before the run

1. **Flat-vector localisation rises on every corpus.** Mechanical: a third page can only add hits.
   This is not a finding; it is the confound being removed.
2. **Direction of the answer-quality change is not predicted.** A third page adds evidence and
   distraction at once; on the 101+-page MMLongBench bin the project has already measured
   localisation rising while ANLS falls.
3. **Falsifier for the headline localisation claim.** If GraMM − flat-vector localisation stops
   excluding zero on a corpus, the claim "GraMM localises better than the flat-vector baseline" does
   not survive on that corpus at equal page budget, and is retracted there. Reported either way,
   per corpus, never pooled.
4. **η is recomputed against the new control on every corpus.** Its denominator moves by
   construction, so no η value from before this run may be quoted afterwards.

## What must be recomputed afterwards

Every flat-vector contrast on all three corpora; η and the conversion frontier; the Gate C column;
`STRATIFIED_REPORT.md` and `RESOURCE_PROFILE.md` (rendered, so they follow automatically); and the
ledger claims that rest on a flat-vector contrast — at least S-14, S-15, S-40, S-42, S-43 — each
re-checked and restated or retracted with its interval.

## Artefacts

The archived k=2 runs are **kept**, renamed `*_k2_archived.json`, never deleted: they are the
provenance of every Phase A/B number. New artefacts carry `PAGE_CONTEXT_N = 3`,
`PAGE_CONTEXT_CHARS = 12000` and `REPLACES` in their `RUN_CONFIG`.

**Registered:** 2026-09-28, before Gate 1 ran.

---

## Amendment 1 — 2026-09-28, after Gate 1 ran and failed, authorised by the advisor

**Gate 1 result, as pre-registered.** Exact reproduction of the archived k=2 contexts:

| corpus | reproduction | verdict |
|---|---|---|
| DocBench | **99.909%** (1 miss / 1,102) | PASS |
| MMLongBench | **98.350%** (18 / 1,091) | FAIL |
| MP-DocVQA | **96.221%** (196 / 5,187) | FAIL |

**Why, measured rather than assumed.** The divergence is upstream of the reranker: the archived
**retrieval pool** itself reproduces on only **81.118%** of MMLongBench questions, and *all 18*
context mismatches are among those. Two runs of this environment are **bit-identical** (0 differing
rows of 1,091), so the selection path is deterministic; what differs is the environment. The only
component upstream of an exact `IndexFlatIP` search that varies across environments is the **4-bit
quantized E5-Mistral-7B query encoder** (bitsandbytes/transformers build), with the FAISS build a
second candidate. The archive's own library versions were never recorded, so exact reproduction is
not reachable here. *(Separately: MP-DocVQA's index is 1024-d **E5-large-v2**, not E5-Mistral —
C-78 — so its queries are encoded with the matching model.)*

**Amended construction.** The k=3 context is:

> the **archived context, exactly and in order**, plus the highest-ranked page from the canonical
> cross-encoder rerank that is not already in it.

The published pages are preserved *by construction*, so exact reproduction is no longer a
precondition and **Gate 1 is retired as a gate**. The three reproduction rates are reported as a
**fidelity measure** instead, in the result and in this file.

**Residual risk, stated rather than buried.** On the questions where this environment's rank-2 page
differs from the archive's (1.7% MMLongBench, 0.1% DocBench, 3.8% MP-DocVQA), the *added third*
page may not be the page the original environment would have added. The first two pages are
unaffected. This is a property of the added page only, and it is recorded with the result.

**Scope correction.** MP-DocVQA's published claims (S-40, the ANLS full-set contrast) are computed
on **n = 5,187**, not on P3's 950-question common set, so replacing the control there means
regenerating 5,187 answers — 7,380 generations in total, not 3,143. APPA is a retrieval metric and
is **unaffected** by a generation-context change.

**Gate 2 becomes binding.** The $14.2 estimate was built on 3,143 generations. Before any full run,
generation cost is calibrated on **50 questions per corpus** (rule 10, as the judge was calibrated)
and the projection is reported. If the projection exceeds the **$17 ceiling**, the run does not
start without a fresh authorisation.


---

## Outcome — 2026-09-28, run complete

**Spend:** generation $10.53 metered (1,041 + 1,052 + 978 calls; 26.0M prompt tokens), calibration
$0.41, DocBench gpt-4-turbo judge ~$4.30 → **≈ $15.2 against the $17 ceiling**. 3,171 generations,
**0 failures, 0 empty predictions**.

**Prediction 1 — held.** Flat-vector localisation rose on every corpus: MMLongBench ctx-hit
0.2327 → 0.2780, DocBench quote-recall@5 0.3932 → 0.4192, MP-DocVQA page-hit@3 0.6632 → 0.7347.

**Prediction 2 — quality also rose**, though no direction was predicted: MMLongBench ANLS
0.3189 → 0.3229, DocBench judge accuracy 0.4728 → 0.5054, MP-DocVQA ANLS 0.6663 → 0.6983.

**Prediction 3 — the falsifier FIRED on MP-DocVQA.** GraMM − page-matched control:

| corpus | quality | localisation |
|---|---|---|
| MMLongBench | ANLS **+0.0293** [+0.0083, +0.0506] | ctx-hit **+0.1563** [+0.1265, +0.1862] |
| DocBench | judge **+0.0272** [+0.0040, +0.0520] pooled; **+0.0225 [−0.0040, +0.0480] answerable (ns)** | quote-recall@5 **+0.0904** [+0.0660, +0.1180] |
| **MP-DocVQA** | ANLS **−0.0005** [−0.0167, +0.0160] **(ns)** | page-hit@3 **−0.0158** [−0.0358, +0.0032] **(ns)**; on the 750 not-page-0 questions **−0.0333 [−0.0547, −0.0120]**, the control ahead |

So "GraMM-RAG beats the flat-vector baseline on all three benchmarks" holds against a control
reading fewer pages. Page-matched it holds on two, and on MP-DocVQA the two are indistinguishable.
Registered as C-80; ledger S-54, S-55, S-56, S-57 supersede S-14, S-40 and S-42.

**Prediction 4 — η recomputed against the new control.** MMLongBench 0.1654 → **0.1877**;
DocBench 0.5144 → **0.3011**. No η against the k=2 control may be quoted.
