# Single-benchmark conclusions are wrong in this design — two independent legs

**Status: a result about evaluation practice, not a caution.** It was a caution when it rested
on one reversal. Two independent quantities, measured for unrelated reasons, each give a
**corpus-dependent answer that would have been reported confidently from either corpus alone.**
Neither is about any particular system.

> **LEG 1 WITHDRAWN 2026-09-28 (C-77) — it was a measurement defect, not corpus dependence.** The
> GraMM-vs-LightRAG ranking did not reverse: the DocBench contrast scored GraMM on its top-5
> **retrieved** pages and LightRAG on the pages its **generator saw**. Scored the same way for both,
> the DocBench difference is **−0.0027 [−0.0342, +0.0288]** — indistinguishable from zero, not a
> reversal (ledger S-52, S-53). The row is kept below, struck, because the near-miss it records is
> still true and is now a different near miss: the project nearly published a reversal that its own
> uniform cell table already contradicted. **Legs 2 and 3 are untouched** — neither involves that
> comparison — and the finding now rests on two structurally unrelated quantities, not three.

## The legs

> **MMLongBench caveat (`CORPUS_PARSE_INTEGRITY.md`).** 1.7% of MMLongBench questions (19 of
> 1,091) are answered against the wrong document's text *and* page images, uniformly across
> every arm. Comparisons are unbiased; absolute ANLS is depressed by 0.0023–0.0054 per arm,
> with no ordering change.


| # | quantity | MMLongBench says | DocBench says | source |
|---|---|---|---|---|
| ~~1~~ | ~~**Ranking** — GraMM vs LightRAG on localisation~~ | ~~GraMM +0.1527~~ | ~~LightRAG +0.0534~~ | **WITHDRAWN, C-77** — mixed page sets; uniformly scored the DocBench contrast is −0.0027 [−0.0342, +0.0288] |
| 1 | **Abstention rank order** across all systems | — | — | S1: Kendall **τ-b = −0.828** between corpora |
| 2 | **Contribution of a multimodal branch** to localisation | **+0.2172** ctx-hit | **+0.0041** quote-recall@5 | S2 (C-49 ablation) |

**Leg 1 (withdrawn) — a ranking that appeared to reverse.** See the banner above: the two arms were
scored on different page sets. What survives is narrower and is not corpus dependence: GraMM leads
LightRAG on MMLongBench localisation and is indistinguishable from it on DocBench (S-53).

**Leg 1 — an entire rank order that inverts.** τ-b = −0.828 on abstention, against +0.828 on
localisation and +0.600 on answer quality in the same comparison. A single-benchmark abstention
comparison is not merely noisy: it will order systems **approximately backwards** relative to the
other corpus. Because τ-b is rank-based it is scale-free, so the C-43 benchmark/metric-family
confound does not apply to it — which is what makes this leg citable rather than suggestive.

**Leg 2 — an architectural contribution that all but vanishes.** Identical code, a **53×
spread**: RAG-Anything's multimodal branch is worth +0.2172 ctx-hit on MMLongBench and +0.0041
quote-recall@5 on DocBench. Reported from MMLongBench alone, "the multimodal branch contributes
substantially to page selection" is true and would be the headline. Reported from DocBench alone,
"the multimodal branch contributes nothing measurable" is also true. Neither is the finding.

## The obvious objection, now measured (P2)

A reviewer's first question is whether these rankings are also **judge**-dependent. Re-judging all
seven arms' DocBench answers with a second, comparable-capability judge (`gpt-4o` against the
canonical `gpt-4-turbo`, 7,715 calls, only the model string varied) answers it:

| varied | abstention τ-b |
|---|---|
| **corpus** | **−0.828** — near-total inversion |
| **judge** | **+0.781**, with **no significant pairwise reversal** among 21 arm pairs |

The judge swap produces a **uniform strictness shift** (every arm down, mean −0.0258), and every
rank movement is confined to three arms whose scores differ by less than 0.003. **The instability
documented here is specific to corpus choice, not a general fragility of evaluation.** Details and
the interval caveat: `P2_RESULT.md`.

## Why two legs is still categorically different from one

The two quantities are **structurally unrelated**: a full rank order over seven systems, and an
ablation of one architectural component. They were produced by two different work packages, for
two different purposes, months apart in the plan. A single reversal invites the reading that one
benchmark is anomalous, or that one metric is fragile. Two, on unrelated quantities, weakens that
reading — the common factor is the **practice of concluding from one corpus**, not any property
of a particular corpus or metric. It is weaker evidence than three would have been, and the
withdrawal of leg 1 is stated here rather than absorbed quietly.

**Each leg is also a near miss, observed on the authors.** The C-49 ablation was written up from
MMLongBench and the +0.2172 figure was on its way to being reported as *the* contribution of the
multimodal branch when the DocBench run returned +0.0041. The abstention rank order was read from
DocBench before MMLongBench inverted it. The withdrawn leg 1 is a near miss of a different kind,
and a sharper one: a reversal was published from a contrast whose two arms were scored on
different page sets, while the project's own uniformly scored cell table already said the
difference was null (C-77).

## What it licenses, and what it does not

**Licensed.** In this design — long-document multimodal RAG, retriever-isolating harness — a
conclusion drawn from one benchmark should be treated as **unsupported until replicated on the
other**, and this applies to rank orders and component ablations. Pairwise rankings are no longer
part of the evidence: the one instance the project had was withdrawn (C-77). Abstention is
the most exposed family (τ-b −0.828), localisation the least (+0.828).

**Not licensed.** This is **two corpora within one project's harness**. It is not evidence about
how many benchmarks a field needs in general, nor that these two corpora are representative of
long-document QA, nor that the effect transfers to other task families. The honest scope is:
*two unrelated quantities, two corpora, one design, two corpus-dependent answers.* Extending
it needs a third corpus, and a third corpus is the obvious next test — it would also
discriminate between "corpora disagree pairwise" and "MMLongBench and DocBench are unusually
dissimilar", which two corpora cannot separate.

**A cheap falsifier exists.** If a third corpus agreed with both on both quantities, the
finding would weaken sharply. Nothing here predicts that it would not.

## Why this is S2's most portable output

S2 was funded to test G-5's mechanism, and it did (Gate H §5b, confirmed on both corpora). But
the mechanism result is about RAG-Anything and about this project's abstention explanation. Leg 3
is about **evaluation practice**, needs no access to this pipeline to apply, and survives contact
with work this project did not produce — the Class B property that §7.2 argues is the
generalisable contribution. It is the strongest thing S2 produced that is not about RAG-Anything.

**Recorded:** 2026-09-09 13:08:11 EDT
