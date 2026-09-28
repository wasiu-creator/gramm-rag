"""Claims ledger -- SOURCE OF TRUTH. CLAIMS_LEDGER.md is RENDERED from this file; never edit it.

Every claim this project has made, including retracted ones, indexed under its ORIGINAL wording.

value provenance, one of:
  J  "{alias:path/to/key|fmt}"   resolved LIVE from a result JSON at render time
  Q  a literal quote that MUST appear verbatim (whitespace-normalised) in `src` -- else render fails
docs_must_show: for a J value, further documents that must print the rendered number. A document
  disagreeing with the data fails the build: the drift the sync assertions exist to catch.
original: phrases from the claim as FIRST published. check_draft.py flags them in any draft.
"""
ROOT_ALIASES = {
    "anls":   "gramm-rag-graphbracket/tier1/p3_mpdocvqa/p3_anls.json",
    "loc950": "gramm-rag-graphbracket/tier1/p3_mpdocvqa/p3_localisation_common950.json",
    "chk":    "gramm-rag-graphbracket/tier1/p3_mpdocvqa/p3_c66_checks.json",
    "p4":     "gramm-rag-graphbracket/tier1/p3_mpdocvqa/p4_failure_attribution.json",
    "p4mm":   "gramm-rag-graphbracket/tier1/p4_failure_mmlongbench.json",
    "p4db":   "gramm-rag-graphbracket/tier1/p4_failure_docbench.json",
    "corr":   "gramm-rag-colpali/scores/mpdocvqa_appa_correction.json",
    "sr":     "gramm-rag-colpali/scores/stratified_report.json",
    "lrdb":   "gramm-rag-colpali/scores/lightrag_scores_docbench.json",
    "lrmm":   "gramm-rag-colpali/scores/lightrag_scores_mmlongbench.json",
    "s2mm":   "gramm-rag-colpali/scores/s2_scores_mmlongbench.json",
    "s5mm":   "gramm-rag-colpali/scores/s5_scores_mmlongbench.json",
    "pre":    "gramm-rag-colpali/scores/package_precomputed.json",
    "ct":     "gramm-rag-colpali/scores/cell_table.json",
    "lb":     "gramm-rag-colpali/scores/length_bins_k3.json",
    "gmm":    "gramm-rag-3benchmarks/data/mmlongbench/gramm_mmlongbench_s42.json",
    "fmm":    "gramm-rag-3benchmarks/data/mmlongbench/baseline_vector_mmlb.json",
}
B = "GraphRAG_baseline/"
S, C, R, SU, NE = "stands", "caveated", "retracted", "superseded", "not established"

# Claims whose VALUE is a contrast against the flat-vector control. Every one was computed
# against the 2-page control before C-80 re-ran it at k=3. build_ledger.py refuses to render a LIVE
# claim in this set unless its face says which control it used -- a text trigger is not enough,
# because S-04 ("at 101+ pages the answer delta turns negative") never names the baseline it is
# measured against, and that is the one the ledger missed.
CONTROL_DEPENDENT = {
    "S-01", "S-03", "S-04", "S-07", "S-11", "S-12", "S-14", "S-15", "S-16", "S-40", "S-42",
    "S-43", "S-45", "S-54", "S-55", "S-56", "S-57", "S-58", "S-59", "S-60", "R-11", "R-18",
}

CLAIMS = [
# ============================== RETRACTED / SUPERSEDED -- indexed by original wording ==============================
dict(id="R-01", status=SU, area="localisation",
     claim="GraMM-RAG shows localisation leadership over an external graph-RAG system (LightRAG).",
     original=["localisation leadership over an external graph-RAG system", "leadership over LightRAG", "+0.1527"],
     value=("Q", "+0.1527 ctx-hit, p=7.7e-12"), src=(B+"PHASE_C_REVISION_LOG.md", "C-42"),
     scope="Stated from MMLongBench alone (M9).", register="C-42, C-77", superseded_by="S-53",
     why="Withdrawn in C-42 because DocBench ran the other way; C-77 showed that contrast scored GraMM "
         "on its retrieved pages and LightRAG on its context pages. Under uniform scoring the DocBench "
         "difference is null, so the claim survives SCOPED TO MMLongBench (S-53) -- it is still not "
         "'leadership over LightRAG' without a corpus attached."),
dict(id="R-02", status=R, area="conversion",
     claim="The between-systems conversion gap is a general property: GraMM's localisation lead does not become an answer-quality lead.",
     original=["between-systems conversion-gap", "between-systems conversion gap", "between systems"],
     value=("Q", "withdrawn as general"), src=(B+"S1_DOCBENCH_RESULT.md", "§1b"),
     scope="Both corpora, as first claimed.", register="C-42", superseded_by="S-11",
     why="A pre-registered falsifier fired: on DocBench LightRAG's localisation edge converts to +0.0345 judge accuracy."),
dict(id="R-03", status=R, area="conversion",
     claim="GraMM-RAG converts localisation into answer quality least efficiently of any system measured, on both corpora.",
     original=["converts localisation into answer quality least efficiently", "converts worst", "least efficiently of any system"],
     value=("Q", "no pairwise eta difference"), src=(B+"PHASE_C_REVISION_LOG.md", "C-41"),
     scope="Ranking of eta point estimates.", register="C-41", superseded_by="S-12",
     why="No pairwise eta difference reaches significance; LightRAG's eta CI is [-0.204, +1.854]."),
dict(id="R-04", status=NE, area="conversion",
     claim="Conversion efficiency is benchmark-specific: GraMM's eta differs 6.3x between MMLongBench and DocBench.",
     original=["conversion is benchmark-specific", "6.3x difference", "eta differs"],
     value=("Q", "(0.165 vs 1.041)"), src=(B+"PHASE_C_REVISION_LOG.md", "C-43"),
     scope="Cross-corpus eta magnitude.", register="C-43", superseded_by="S-11",
     why="Benchmark and metric family are perfectly confounded; no common definition exists, so no number is reported."),
dict(id="R-05", status=R, area="cost",
     claim="GraMM-RAG is cheaper to prepare and serve than ColPali.",
     original=["cheaper to prepare and serve", "GraMM-RAG is cheaper"],
     value=("Q", "the falsifier FIRED. The general cost claim is WITHDRAWN."), src=(B+"COST_GATE_RESULT.md", "Verdict"),
     scope="General cost claim, pre-registered with falsifiers.", register="COST_GATE", superseded_by="S-20",
     why="Preparation falsified at 1.47x; both pre-registered falsifiers fired."),
dict(id="R-06", status=R, area="cost",
     claim="GraMM-RAG is faster and more predictable than ColPali at query time.",
     original=["faster and more predictable", "query-time advantage", "far more predictable", "GraMM wins, and is far more predictable"],
     value=("Q", "Query-time advantage: COLD ONLY."), src=(B+"COST_GATE_RESULT.md", "§3"),
     scope="Warm query latency, both corpora.", register="COST_GATE", superseded_by="S-21",
     why="ColPali is faster warm on both corpora (66.3 vs 84.8 ms MMLongBench; 62.2 vs 84.0 ms DocBench)."),
dict(id="R-07", status=R, area="cost",
     claim="RAG-Anything's indexing cost is two-thirds vision.",
     original=["two-thirds vision", "two thirds of it vision", "one-third vision"],
     value=("Q", "two thirds of it vision"), src=(B+"GATE_G_RESULT.md", "§projection"),
     scope="S2 indexing cost split.", register="C-48", superseded_by="S-22",
     why="Collinearity artefact: NNLS gave two-thirds, pooled gave one-third; the split is not identified."),
dict(id="R-08", status=R, area="cost",
     claim="GraMM-RAG's graph construction costs approximately $45 in API calls.",
     original=["≈$45 API cost", "$45 API", "~$45"],
     value=("Q", "order $0.1–0.2"), src=(B+"PHASE_C_REVISION_LOG.md", "C-31"),
     scope="Graph construction, 364-document corpora.", register="C-31", superseded_by=None,
     why="An unsourced planning figure, scoped to MP-DocVQA's 927 documents; measured rate gives ~100x less."),
dict(id="R-09", status=R, area="architecture",
     claim="GraMM-RAG's graph has four model-extracted edge types and eight contributing typed edges.",
     original=["Four model-extracted edge types", "four model-extracted edge types", "Eight typed edges"],
     value=("Q", "eight defined, six contributing"), src=(B+"PHASE_C_REVISION_LOG.md", "C-31"),
     scope="Graph builder, both corpora.", register="C-31", superseded_by=None,
     why="Only `kg` invokes an LLM; `caption` is dead (14 edges on MMLongBench) and `spatial` sparse on DocBench."),
dict(id="R-10", status=R, area="abstention",
     claim="Stronger retrieval suppresses abstention: retrieval-augmented systems get worse at declining as they get better at finding evidence (a negative scaling law).",
     original=["stronger retrieval suppresses abstention", "scaling law", "negative scaling law for retrieval-augmented abstention"],
     value=("Q", "no general negative scaling law for retrieval-augmented abstention"), src=(B+"PHASE_B_GO_NO_GO_REVIEW.md", "§findings"),
     scope="Abstention vs retrieval strength, both corpora.", register="C-28", superseded_by="S-31",
     why="The two benchmarks disagree in direction, so no law is available."),
dict(id="R-11", status=R, area="localisation",
     claim="GraMM-RAG localises the answer page better than the flat-vector baseline on MP-DocVQA.",
     original=["APPA@1 +75%", "+75% (MP-DocVQA)", "localises better than flat-vector on MP-DocVQA", "+0.2296"],
     value=("J", "{corr:gramm_minus_flat/appa@1/published/diff|+.4f} as published -> {corr:gramm_minus_flat/appa@1/corrected/diff|+.4f} [{corr:gramm_minus_flat/appa@1/corrected/ci95/0|+.3f}, {corr:gramm_minus_flat/appa@1/corrected/ci95/1|+.3f}] corrected, n={corr:n}"),
     src=(B+"CORRECTION_2026-09-21_MPDOCVQA_APPA.md", "Retracted"), docs_must_show=[B+"CORRECTION_2026-09-21_MPDOCVQA_APPA.md", B+"MPDOCVQA_GOLDPAGE_DEFECT.md", B+"PHASE_C_REVISION_LOG.md"],
     scope="MP-DocVQA, APPA@1/5/10, Phase A/B run8.", register="C-66", superseded_by="S-40",
     why="Gold pages wrong on 48.9% of questions (gen_notebook.py:311); the direction reverses at every cut-off."),
dict(id="R-12", status=R, area="task",
     claim="Cross-page concentration is absent for every arm measured, so the finding is a property of the task, not of one architecture.",
     original=["cross-page concentration is absent for EVERY arm", "property of the TASK"],
     value=("Q", "silently tested 2 arms of 5"), src=(B+"PHASE_C_REVISION_LOG.md", "C-40"),
     scope="Dimension A, MMLongBench.", register="C-40", superseded_by="S-02",
     why="All three ColPali-family arms DO concentrate (ColPali -0.2150 [-0.2826, -0.1495])."),
dict(id="R-13", status=SU, area="reward",
     claim="The reward's tau tuning to 0.0 on every benchmark is a grid-search defect.",
     original=["tau tuning to 0.0", "grid search defect", "tuning failure"],
     value=("Q", "tau -> 0 is the optimum, not a bug"), src=(B+"PHASE_C_REVISION_LOG.md", "C-28"),
     scope="MMLongBench and DocBench reward.", register="C-28", superseded_by="S-30",
     why="No threshold on U beats 0.560 on MMLongBench; refusing nothing was optimal for so weak a signal."),
dict(id="R-14", status=R, area="latency",
     claim="MS GraphRAG's context assembly is an order of magnitude faster than LightRAG's.",
     original=["an order of magnitude faster", "order of magnitude faster than LightRAG"],
     value=("Q", "comparable"), src=(B+"PHASE_C_REVISION_LOG.md", "C-63"),
     scope="Serving, B/B'.", register="C-63", superseded_by="S-24",
     why="A cross-clock subtraction over-subtracted; the single-clock measurement gives 0.24-0.27 s, comparable."),
dict(id="R-15", status=SU, area="cost",
     claim="S4 (MMGraphRAG) full-corpus indexing costs $245.55.",
     original=["$245.55", "S4 full corpus"],
     value=("Q", "Planning number: $340"), src=(B+"S4_PILOT_RESULT.md", "§4"),
     scope="S4 cost model, not identified.", register="C-55", superseded_by="S-50",
     why="The model is NOT IDENTIFIED (2.29x > 2x); the operative clause binds to the conservative estimator."),
dict(id="R-16", status=SU, area="comparators",
     claim="MS GraphRAG is the first arm in the comparator set to be beaten significantly by GraMM.",
     original=["first arm in the comparator set"],
     value=("Q", "first non-control comparator"), src=(B+"S5_RESULT.md", "§headline"),
     scope="S5, both corpora.", register="A3", superseded_by=None,
     why="Flat-vector is the control, so 'first arm' miscounted it."),
dict(id="R-17", status=SU, area="judge",
     claim="Rankings are judge-robust at the pooled level (tau-b clears 0.90).",
     original=["pooled clause", "+0.905"],
     value=("Q", "mis-specified"), src=(B+"P2_RESULT.md", "§2"),
     scope="P2 judge sweep, DocBench.", register="P2", superseded_by="S-33",
     why="Gate H binds at stratum level: +0.905 clears 0.90 while neither stratum (0.810, 0.781) does."),
dict(id="R-18", status=SU, area="P3",
     claim="On MP-DocVQA page-hit@3, GraMM-RAG 0.7127 and Flat-vector 0.6575 (five-arm table, n = 950).",
     original=["0.7127", "0.6575"],
     value=("J", "GraMM {loc950:gramm/page_hit_at_3|.4f}, Flat-vector {loc950:flat/page_hit_at_3|.4f} on the true common set"),
     src=(B+"P3_RESULT.md", "§3"), docs_must_show=[B+"P3_RESULT.md"],
     scope="P3 five-arm table.", register="C-67", superseded_by="S-43",
     why="The draft table was headed n=950 but its localisation column carried n=978 values for four of five arms."),
dict(id="R-19", status=R, area="P4",
     claim="Conversion given a hit is near-constant across architecturally unrelated arms (arms differ in finding, not in using).",
     original=["near-constant across five architecturally unrelated arms", "differ almost entirely in finding"],
     value=("Q", "holds *within* a corpus"), src=(B+"P3_RESULT.md", "§5"),
     scope="Stated first from MP-DocVQA alone.", register="P4", superseded_by="S-44",
     why="Holds within a corpus only; the conversion rate itself moves across corpora (0.46-0.51 vs 0.84-0.88)."),

# ============================== STANDING / CAVEATED ==============================
dict(id="S-01", status=SU, area="headline",
     claim="GraMM-RAG improves pooled answer quality over the flat-vector baseline on MMLongBench-Doc.",
     value=("Q", "the pooled delta (+0.0334)"), src=(B+"PHASE_C_REVISION_LOG.md", "C-19"),
     original=["+0.0334", "improves pooled answer quality over the flat-vector baseline"],
     scope="MMLongBench, ANLS, pooled, against a control reading 2 context pages.",
     superseded_by="S-54",
     why="The control read 2 pages against GraMM's 3 (C-77 ii). Page-matched the improvement is "
         "+0.0293 [+0.0083, +0.0506] -- smaller, and still significant.", register="C-19",
     caveat="Sits far below either answerable stratum because GraMM regresses significantly on unanswerable questions (S-03)."),
dict(id="S-02", status=NE, area="task",
     claim="GraMM-RAG's advantage concentrates on cross-page questions (Gate A hypothesis).",
     value=("Q", "Δ_cross − Δ_single = **−0.0245**, CI [−0.0687, +0.0216]"), src=(B+"PHASE_C_REVISION_LOG.md", "C-19"),
     scope="MMLongBench, single- vs cross-page strata.", register="C-19",
     caveat="Pre-registered; NOT SUPPORTED. GraMM is a general retrieval improvement, not a cross-page mechanism."),
dict(id="S-03", status=SU, area="abstention",
     claim="GraMM-RAG regresses significantly on unanswerable questions relative to the flat-vector baseline.",
     value=("Q", "**−0.0574** | [−0.0943, −0.0205]"), src=(B+"PHASE_C_REVISION_LOG.md", "C-19"),
     original=["regresses significantly on unanswerable", "-0.0574", "−0.0574"],
     scope="MMLongBench, unanswerable stratum, n=244, against the k=2 control.",
     superseded_by="S-58",
     why="Page-matched the regression is -0.0246 [-0.0656, +0.0205] and NO LONGER SIGNIFICANT "
         "(C-80/C-81): part of what looked like a GraMM deficit on unanswerable questions was the "
         "control's smaller context.", register="C-19"),
dict(id="S-04", status=C, area="conversion",
     claim="On MMLongBench the conversion gap is length-gated: at 101+ pages the answer delta turns negative while localisation still improves.",
     value=("Q", "turns **negative (−0.0087) at 101+ pages**"), src=(B+"PHASE_C_REVISION_LOG.md", "C-20"),
     scope="MMLongBench only, against the SHIPPED k=2 control; DocBench converts in all five bins.",
     caveat="Stands as PRE-REGISTERED and as measured against the shipped configuration: the "
            "-0.0087 reproduces exactly. But the negative sign was a page-budget artefact (C-81). "
            "Against a page-matched control the 101+ delta is +0.0087 [-0.0348, +0.0522] and NO "
            "arm is negative at 101+. What survives is a shrinking advantage -- GraMM's smallest "
            "bin, about a third of its others -- not a sign change. Quote it with that attached, "
            "or quote S-59 instead.", register="C-20"),
dict(id="S-05", status=S, area="architecture",
     claim="The SigLIP figure branch does not demonstrably earn its complexity.",
     value=("Q", "**Figure (+0.034) is not**"), src=(B+"PHASE_C_REVISION_LOG.md", "C-20"),
     scope="Both corpora, evidence-type strata.", register="C-20"),
dict(id="S-06", status=S, area="validity",
     claim="The retrieval control holds bit-identical between superseded and current runs.",
     value=("Q", "Δ = 0.0"), src=(B+"PHASE_C_REVISION_LOG.md", "C-20"),
     scope="MMLongBench APPA@1/5/10, DocBench quote-recall@1/3/5.", register="C-20"),
dict(id="S-07", status=C, area="judge",
     claim="GraMM-RAG's lead over flat-vector survives a change of judge.",
     value=("Q", "Δ +0.0334 → +0.0376"), src=(B+"PHASE_C_REVISION_LOG.md", "C-21"),
     scope="MMLongBench and DocBench, positioning judge swap, against the k=2 control.",
     caveat="Both deltas are contrasts with the 2-page control; page-matched the lead is +0.0293 "
            "(C-80). The finding is about ROBUSTNESS TO THE JUDGE, which the control does not "
            "touch -- but the numbers quoted are k=2 numbers.", register="C-21"),
dict(id="S-08", status=C, area="positioning",
     claim="GraMM-RAG trails RAG-Anything's published scores.",
     value=("Q", "Positioning is **indicative only**"), src=(B+"PHASE_C_REVISION_LOG.md", "C-21"),
     scope="Published numbers; two of five confounds unresolved.", register="C-21",
     caveat="Indicative only — generator and answer-constraint differences remain."),
dict(id="S-10", status=SU, area="localisation",
     claim="GraMM-RAG leads LightRAG on localisation on MMLongBench and trails it on DocBench — the ranking reverses between corpora.",
     original=["the ranking reverses between corpora", "trails it on DocBench", "opposite directions, both significant"],
     value=("Q", "Two benchmarks, opposite directions, both significant."), src=(B+"S1_DOCBENCH_RESULT.md", "§1"),
     scope="MMLongBench ctx-hit; DocBench quote-recall@5.", register="C-42", supersedes="R-01",
     superseded_by="S-53",
     why="The DocBench half was a mixed-page-set contrast (C-77): GraMM scored on its top-5 RETRIEVED "
         "pages, LightRAG on the pages its generator saw. Scored uniformly the DocBench difference is "
         "-0.0027 [-0.0342, +0.0288]. There is no reversal to report."),
dict(id="S-11", status=C, area="conversion",
     claim="Conversion fails on MMLongBench (evidenced two independent ways) and succeeds on DocBench; the magnitude of the difference is not measurable in this design.",
     value=("Q", "conversion fails on MMLongBench, evidenced two independent ways"), src=(B+"PHASE_C_REVISION_LOG.md", "C-43"),
     scope="Within-corpus comparisons only.", register="C-43, C-81", supersedes="R-02, R-04",
     caveat="ONE of the two MMLongBench legs survives, and the lost one was a k=2-control contrast. "
            "The surviving leg is between systems: GraMM "
            "localises +0.1527 over LightRAG with no measurable ANLS gain. The second leg -- A3's "
            "101+ bin, where the answer delta was negative while ctx-hit rose -- was measured "
            "against the 2-page control and does not survive page matching (C-81). Say 'evidenced "
            "between systems', not 'evidenced two independent ways'."),
dict(id="S-12", status=C, area="conversion",
     claim="Every arm's conversion efficiency is far below 1; GraMM's point estimate is lowest, and no arm differs significantly.",
     value=("J", "MMLongBench eta vs the page-matched control: GraMM {ct:mmlongbench/eta_vs_flatk3/GraMM-RAG/eta|.3f}, "
                 "RAG-Anything {ct:mmlongbench/eta_vs_flatk3/RAG-Anything 1.2.8/eta|.3f}, "
                 "ColPali {ct:mmlongbench/eta_vs_flatk3/ColPali/eta|.3f}, "
                 "Union {ct:mmlongbench/eta_vs_flatk3/Union(CP∪GraMM)/eta|.3f}"),
     src=(B+"PHASE_C_REVISION_LOG.md", "C-41"),
     scope="MMLongBench only, eta against the PAGE-MATCHED control.", register="C-41, C-80", supersedes="R-03",
     caveat="Two qualifications. (1) MMLongBench only: on DocBench most arms' eta EXCEEDS 1 "
            "(RAG-Anything 1.30, ColPali 1.50), so 'far below 1' is not a cross-corpus statement. "
            "(2) LightRAG and MS GraphRAG are omitted because their localisation delta against the "
            "page-matched control is ~0.004 and ~0.030: the ratio is undefined-in-practice, not "
            "small, and the figures mark it UNSTABLE rather than printing a value."),
dict(id="S-13", status=SU, area="localisation",
     claim="LightRAG leads GraMM-RAG on DocBench localisation and answer quality.",
     original=["LightRAG leads GraMM-RAG on DocBench localisation", "+0.0534", "b=90 c=51"],
     value=("Q", "**+0.0534** | **[+0.0219, +0.0849]** | b=90 c=51, **p=0.0013**"), src=(B+"S1_DOCBENCH_RESULT.md", "§results"),
     scope="DocBench; quote-recall@5 and judge accuracy.", register="C-42, C-77", superseded_by="S-52",
     why="The localisation half compared two different page sets (C-77) and does not survive uniform "
         "scoring. The answer-quality half is unaffected and stands as S-52."),
dict(id="S-14", status=SU, area="headline",
     claim="GraMM-RAG improves on the flat-vector baseline on DocBench judge accuracy.",
     original=["+0.0599", "improves on the flat-vector baseline on DocBench"],
     value=("Q", "| GraMM − Flat | +0.0599 | [+0.0372, +0.0817] | p=2.7e-07"), src=(B+"S1_DOCBENCH_RESULT.md", "§results"),
     scope="DocBench, judge accuracy, against a control reading 2 context pages.",
     register="S1, C-80", superseded_by="S-55",
     why="The control read 1.84 pages against GraMM's 3 (C-77 ii). Page-matched, the pooled "
         "advantage is +0.0272 and the answerable-stratum advantage is no longer significant."),
dict(id="S-15", status=S, area="localisation",
     claim="On MMLongBench, GraMM-RAG places the answer page first 166% more often than the flat-vector baseline (APPA@1).",
     value=("J", "APPA@1 GraMM {gmm:metrics/appa@1|.4f} vs flat-vector {fmm:metrics/appa@1|.4f}"),
     src=(B+"PHASE_B_DISSERTATION.md", "§1, §3.2"),
     scope="MMLongBench only; unaffected by C-66 (evidence-page lists, separate code), and "
           "unaffected by C-80: APPA scores the RETRIEVAL ranking, not the generator's context, so "
           "the control's page budget does not enter it.", register="Phase A/B"),
dict(id="S-16", status=C, area="headline",
     claim="GraMM-RAG improves answer quality over the flat-vector baseline by +2.1% to +12.7% across benchmarks.",
     original=["+2.1% to +12.7%"],
     value=("Q", "answer-quality +2.1% to +12.7%"), src=(B+"PHASE_B_DISSERTATION.md", "§1"),
     scope="MP-DocVQA ANLS (+2.1%) through DocBench judge accuracy (+12.7%).", register="C-43, C-66",
     caveat="Three qualifications now. (1) The range spans two metric families across corpora, "
            "which cannot be compared (C-43); state per corpus. (2) Its low end is MP-DocVQA, "
            "measured on a model trained on corrupted labels (C-66). (3) EVERY figure in the range "
            "is a contrast with the 2-page control (C-80): page-matched, MMLongBench is +0.0293, "
            "DocBench's answerable stratum is no longer significant, and the MP-DocVQA end is "
            "-0.0005 and not distinguishable from zero. The range as written no longer holds."),
dict(id="S-20", status=S, area="cost",
     claim="GraMM-RAG is more expensive to prepare than ColPali; ColPali becomes cheaper overall only after very high query volume.",
     value=("Q", "becomes cheaper only after ~346,000 queries"), src=(B+"COST_GATE_RESULT.md", "§amortisation"),
     scope="Preparation and serving, one corpus; a lower bound.", register="COST_GATE", supersedes="R-05"),
dict(id="S-21", status=S, area="latency",
     claim="GraMM-RAG is faster and more predictable than ColPali at cold start only.",
     value=("Q", "Query-time advantage: COLD ONLY."), src=(B+"COST_GATE_RESULT.md", "§3"),
     scope="Cold query latency, both corpora.", register="COST_GATE", supersedes="R-06"),
dict(id="S-22", status=S, area="cost",
     claim="RAG-Anything's indexing cost is vision-dominated; the vision/text split is not identified.",
     value=("Q", "were retired as collinearity artefacts"), src=(B+"S2_INDEXING_RESULT.md", "§cost"),
     scope="S2 indexing.", register="C-48", supersedes="R-07"),
dict(id="S-23", status=S, area="latency",
     claim="The three API-mediated graph arms use no GPU at query time; their cost is API spend rather than GPU hours.",
     value=("Q", "score 0 A100-h and 0 GiB"), src=(B+"PHASE_C_REVISION_LOG.md", "C-71"),
     scope="LightRAG, RAG-Anything, MS GraphRAG; both corpora; B resource cells.", register="B"),
dict(id="S-24", status=S, area="latency",
     claim="Answer generation is 81-86% of MS GraphRAG's query latency; context assembly is 0.24-0.27 s.",
     value=("Q", "context assembly\n  **0.24–0.27 s**, generation **81–86%** of total"), src=(B+"PHASE_C_REVISION_LOG.md", "C-63"),
     scope="MS GraphRAG, both corpora, single-clock decomposition.", register="C-63", supersedes="R-14"),
dict(id="S-25", status=S, area="multimodal",
     claim="RAG-Anything's image route is worth +0.2172 ctx-hit on MMLongBench and +0.0041 quote-recall@5 on DocBench — a 53x spread.",
     value=("Q", "**+0.2172** ctx-hit | **+0.0041** quote-recall@5"), src=(B+"SINGLE_BENCHMARK_FINDING.md", "leg 3"),
     scope="Identical code, two corpora.", register="C-49"),
dict(id="S-26", status=S, area="corpus-dependence",
     claim="Abstention rank order inverts between corpora (Kendall tau-b = -0.828).",
     value=("Q", "Kendall **τ-b = −0.828** between corpora"), src=(B+"SINGLE_BENCHMARK_FINDING.md", "leg 2"),
     scope="MMLongBench vs DocBench, all systems.", register="S1"),
dict(id="S-30", status=S, area="abstention",
     claim="The answerability signal exists in the context but not in the retrieval-side utility U; the generator detects it at 0.806 on identical context.",
     value=("Q", "generator's own decline decision at **0.806\n  on identical context**"), src=(B+"PHASE_C_REVISION_LOG.md", "C-28"),
     scope="MMLongBench.", register="C-28", supersedes="R-13"),
dict(id="S-31", status=S, area="abstention",
     claim="Abstention behaviour does not scale monotonically with retrieval strength; the two benchmarks disagree in direction.",
     value=("Q", "The two\n   benchmarks disagree in direction, so no law is available."), src=(B+"PHASE_B_PRIME_1_FINDINGS.md", "§1"),
     scope="Both corpora.", register="C-28", supersedes="R-10"),
dict(id="S-32", status=S, area="abstention",
     claim="GraMM-RAG's advantage on unanswerable questions is a threshold artefact; under matched coverage a stronger retriever dominates it.",
     value=("Q", "GraMM's advantage on unanswerable questions is a threshold artefact."), src=(B+"PHASE_B_PRIME_1_FINDINGS.md", "§1"),
     scope="Both corpora, matched coverage.", register="B'-1"),
dict(id="S-33", status=S, area="judge",
     claim="Rankings are judge-robust at stratum level: no significant pairwise reversal among 21 arm pairs under a second judge.",
     value=("Q", "**+0.781**, with **no significant pairwise reversal** among 21 arm pairs"), src=(B+"SINGLE_BENCHMARK_FINDING.md", "P2"),
     scope="DocBench, all seven arms, gpt-4o vs gpt-4-turbo.", register="P2", supersedes="R-17"),
dict(id="S-34", status=S, area="abstention",
     claim="Generator-side abstention signals fail to beat the retrieval-side ceiling; the best (a sufficiency autorater) reaches 0.5431 balanced accuracy.",
     value=("Q", "| S-a autorater | 1,091 | 0.939 | 0.148 | **0.5431** | [0.5235, 0.5625] | **FAILURE** |"), src=(B+"WP3_RESULT.md", "Verdict"),
     scope="MMLongBench, pre-registered.", register="WP-3"),
dict(id="S-35", status=S, area="abstention",
     claim="Injecting gold pages raises the autorater's decision quality to 0.6098, still short of the success threshold.",
     value=("Q", "D-1 reaches 0.6098, 95% CI [0.5883, 0.6306]"), src=(B+"D1_RESULT.md", "§1"),
     scope="MMLongBench, gold pages injected.", register="D-1"),
dict(id="S-36", status=S, area="comparators",
     claim="MS GraphRAG beats a random-page control on both corpora (Gate H criterion 3).",
     value=("Q", "| DocBench | 715 | 0.4476 | 0.2821 | +0.1654 | [+0.1367, +0.1945] | **PASS** |"), src=(B+"S5_RESULT.md", "§3"),
     scope="Both corpora.", register="S5"),

# ============================== P3 (MP-DocVQA) and P4 -- values resolved live ==============================
dict(id="S-40", status=SU, area="headline",
     claim="GraMM-RAG answers MP-DocVQA more accurately than the flat-vector baseline.",
     value=("J", "ANLS {corr:anls_fullset/diff|+.4f} [{corr:anls_fullset/ci95/0|+.4f}, {corr:anls_fullset/ci95/1|+.4f}], n={corr:anls_fullset/n}; replicated in P3 {anls:contrasts/GraMM-RAG - Flat-vector/diff|+.4f} [{anls:contrasts/GraMM-RAG - Flat-vector/lo|+.4f}, {anls:contrasts/GraMM-RAG - Flat-vector/hi|+.4f}], n={anls:n_common}"),
     src=(B+"CORRECTION_2026-09-21_MPDOCVQA_APPA.md", "Not retracted"), docs_must_show=[B+"CORRECTION_2026-09-21_MPDOCVQA_APPA.md"],
     scope="MP-DocVQA, ANLS, against a control reading 1.92 context pages.",
     register="C-66, C-80", supersedes="R-11", superseded_by="S-56",
     caveat="GraMM here was trained on corrupted labels (C-66); read the lead as a lower bound.",
     why="The whole margin was the page budget: page-matched, the contrast is -0.0005 "
         "[-0.0167, +0.0160] on n=950, and the control gains +0.0320 from its third page alone."),
dict(id="S-41", status=S, area="multimodal",
     claim="On a zero-image corpus, RAG-Anything's localisation advantage over LightRAG collapses to indistinguishable (P3 primary prediction held).",
     value=("J", "page-hit@3 {loc950:s2/page_hit_at_3|.4f} vs {loc950:lightrag/page_hit_at_3|.4f}; ANLS {anls:contrasts/RAG-Anything - LightRAG/diff|+.4f} [{anls:contrasts/RAG-Anything - LightRAG/lo|+.4f}, {anls:contrasts/RAG-Anything - LightRAG/hi|+.4f}], n={anls:n_common}"),
     src=(B+"P3_RESULT.md", "§2"), docs_must_show=[B+"P3_RESULT.md"],
     scope="MP-DocVQA; both arms share one attribution path, so artefacts cancel.", register="P3"),
dict(id="S-42", status=SU, area="localisation",
     claim="GraMM-RAG's final context localises the MP-DocVQA answer page better than flat-vector, including where the answer is not page 0.",
     value=("J", "{chk:gramm_minus_flat_page_hit_at_3/true_gold_not_page0/diff|+.4f} [{chk:gramm_minus_flat_page_hit_at_3/true_gold_not_page0/ci95/0|+.4f}, {chk:gramm_minus_flat_page_hit_at_3/true_gold_not_page0/ci95/1|+.4f}] on n={chk:gramm_minus_flat_page_hit_at_3/true_gold_not_page0/n} answers not on page 0"),
     src=(B+"P3_RESULT.md", "§3 footnote"), docs_must_show=[B+"P3_RESULT.md"],
     scope="MP-DocVQA P3, page-hit@3, verified gold, against a control reading 1.92 pages.",
     register="C-66, C-80", superseded_by="S-56",
     caveat="Trained on corrupted labels; a lower bound.",
     why="Page-matched the direction REVERSES on this stratum: -0.0333 [-0.0547, -0.0120] on the "
         "750 questions whose answer is not on page 0 -- the control localises better."),
dict(id="S-43", status=S, area="P3",
     claim="On MP-DocVQA the two page-level arms (GraMM, flat-vector) localise far better at k=3 than the three chunk-level graph arms, which find the gold page but rank it 4th at the median.",
     value=("J", "page-hit@3: GraMM {loc950:gramm/page_hit_at_3|.4f}, Flat k=2 {loc950:flat/page_hit_at_3|.4f}, "
                 "Flat k=3 {loc950:flatk3/page_hit_at_3|.4f}, LightRAG {loc950:lightrag/page_hit_at_3|.4f}, "
                 "RAG-Anything {loc950:s2/page_hit_at_3|.4f}, MS GraphRAG {loc950:s5/page_hit_at_3|.4f} (n=950)"),
     src=(B+"P3_RESULT.md", "§3"), docs_must_show=[B+"P3_RESULT.md"],
     scope="MP-DocVQA, retrieval granularity — an architectural property, not a defect. The gap to "
           "the chunk-level arms is WIDER against the page-matched control (0.7347), so page "
           "budget is not what produces it.", register="P3, C-80", supersedes="R-18"),
dict(id="S-44", status=S, area="P4",
     claim="Within a corpus, arms differ far more in localisation than in conversion; across corpora the conversion rate itself moves.",
     value=("J", "MP-DocVQA conv|hit {p4:conversion_given_hit/MS GraphRAG|.3f}-{p4:conversion_given_hit/LightRAG|.3f}; MMLongBench {p4mm:arms/MS GraphRAG/conversion_given_hit|.3f}-{p4mm:arms/RAG-Anything/conversion_given_hit|.3f}"),
     src=(B+"P3_RESULT.md", "§5"),
     scope="MMLongBench and MP-DocVQA; DocBench excluded (ANLS on prose golds).", register="P4", supersedes="R-19"),
dict(id="S-45", status=S, area="corpus-dependence",
     claim="P3's contribution to the corpus-dependence finding does not rest on GraMM-RAG: removing flat-vector, not GraMM, brings MP-DocVQA into line with the other corpora.",
     value=("J", "tau-b MMLongBench vs MP-DocVQA (localisation): all {chk:leave_one_out_tau_b/localisation/mmlongbench_vs_mpdocvqa/all|+.3f}, without flat-vector {chk:leave_one_out_tau_b/localisation/mmlongbench_vs_mpdocvqa/drop_Flat-vector|+.3f}, without GraMM {chk:leave_one_out_tau_b/localisation/mmlongbench_vs_mpdocvqa/drop_GraMM-RAG|+.3f}"),
     src=(B+"P3_RESULT.md", "§3 corpus dependence"),
     scope="Five shared arms, with flat-vector at its SHIPPED k=2 budget; direction only (tau-b "
           "steps of 0.2). The page-matched k=3 control is deliberately NOT in this rank "
           "correlation: it is a sixth arm that did not exist when the rank order was measured, "
           "and adding it would change every tau-b here (C-80).", register="C-66",
     caveat="No inference attempted with 5 arms; the direction is consistent, the magnitudes are not estimates."),
dict(id="S-46", status=S, area="validity",
     claim="The handoff package regenerates the evidence notebook with no output differences from the original run.",
     value=("Q", "**52 cells, 0 errors, 0 diffs.**"), src=(B+"PHASE_C_REVISION_LOG.md", "C-72"),
     scope="phaseC_evidence.ipynb, package-only cold start.", register="C-72"),
dict(id="S-47", status=S, area="validity",
     claim="Phase A/B's MP-DocVQA gold pages are wrong on 48.9% of questions, and the same labels trained GraMM's HGT.",
     value=("J", "{corr:gold_wrong} of {corr:n} gold pages wrong ({corr:gold_wrong_frac|.1%}); GraMM ranks page 0 first on {corr:top_ranked_page0_frac/GraMM-RAG|.1%} vs {corr:controls/gramm_mmlongbench_first_page_top_frac|.1%} on MMLongBench"),
     src=(B+"MPDOCVQA_GOLDPAGE_DEFECT.md", "§1-2"),
     scope="MP-DocVQA only; router and refusal uses were inert.", register="C-66",
     caveat="The training-induced page-0 bias is supported by two controls, not yet by a retrain."),
dict(id="S-50", status=NE, area="comparators",
     claim="MMGraphRAG (S4) was not run as a comparator; it is reported as a piloted Tier 3 feasibility finding.",
     value=("Q", "piloted, projected, **deferred without a full run**"), src=(B+"S4_TIER3_DISPOSITION.md", "Status"),
     scope="S4.", register="C-55", supersedes="R-15",
     caveat="External multimodal-graph comparators are n=1; scene-graph construction is unrepresented."),
# ===================== the page-matched control (C-77 ii, C-80) =====================
dict(id="S-54", status=S, area="headline",
     claim="On MMLongBench, GraMM-RAG beats a PAGE-MATCHED flat-vector control on both answer quality and localisation.",
     value=("J", "ANLS {ct:mmlongbench/contrasts_vs_gramm/Flat-vector (k=3)/ANLS pooled/delta|+.4f} "
                 "[{ct:mmlongbench/contrasts_vs_gramm/Flat-vector (k=3)/ANLS pooled/ci/1|+.4f}, "
                 "{ct:mmlongbench/contrasts_vs_gramm/Flat-vector (k=3)/ANLS pooled/ci/0|+.4f}] and ctx-hit "
                 "{ct:mmlongbench/contrasts_vs_gramm/Flat-vector (k=3)/ctx-hit/delta|+.4f} "
                 "[{ct:mmlongbench/contrasts_vs_gramm/Flat-vector (k=3)/ctx-hit/ci/1|+.4f}, "
                 "{ct:mmlongbench/contrasts_vs_gramm/Flat-vector (k=3)/ctx-hit/ci/0|+.4f}] in favour of the control, i.e. GraMM leads"),
     src=(B+"STRATIFIED_REPORT.md", "MMLongBench table"), docs_must_show=[B+"STRATIFIED_REPORT.md"],
     scope="MMLongBench; control at 3 context pages and 12,000 chars, identical to GraMM's budget.",
     register="C-80"),
dict(id="S-55", status=C, area="headline",
     claim="On DocBench, GraMM-RAG beats a page-matched flat-vector control on pooled judge accuracy and on localisation.",
     value=("J", "judge {ct:docbench/contrasts_vs_gramm/Flat-vector (k=3)/judge accuracy pooled/delta|+.4f}; "
                 "answerable {ct:docbench/contrasts_vs_gramm/Flat-vector (k=3)/judge accuracy answerable/delta|+.4f} "
                 "[{ct:docbench/contrasts_vs_gramm/Flat-vector (k=3)/judge accuracy answerable/ci/0|+.4f}, "
                 "{ct:docbench/contrasts_vs_gramm/Flat-vector (k=3)/judge accuracy answerable/ci/1|+.4f}]; "
                 "quote-recall@5 {ct:docbench/contrasts_vs_gramm/Flat-vector (k=3)/quote-recall@5/delta|+.4f}"),
     src=(B+"STRATIFIED_REPORT.md", "DocBench table"), docs_must_show=[B+"STRATIFIED_REPORT.md"],
     scope="DocBench; page-matched control.", register="C-80", supersedes="S-14",
     caveat="The ANSWERABLE-stratum quality advantage does NOT survive page matching: its interval "
            "includes zero. Quote the pooled figure only with the stratum result beside it."),
dict(id="S-56", status=S, area="headline",
     claim="On MP-DocVQA, GraMM-RAG and a page-matched flat-vector control are not distinguishable, on either answer quality or localisation; the published advantage was the page budget.",
     value=("J", "ANLS {anls:contrasts/GraMM-RAG - Flat-vector (k=3)/diff|+.4f} "
                 "[{anls:contrasts/GraMM-RAG - Flat-vector (k=3)/lo|+.4f}, "
                 "{anls:contrasts/GraMM-RAG - Flat-vector (k=3)/hi|+.4f}]; the control's third page alone is worth "
                 "{anls:contrasts/Flat-vector (k=3) - Flat-vector/diff|+.4f} "
                 "[{anls:contrasts/Flat-vector (k=3) - Flat-vector/lo|+.4f}, "
                 "{anls:contrasts/Flat-vector (k=3) - Flat-vector/hi|+.4f}]; page-hit@3 "
                 "{chk:gramm_minus_flat_page_hit_at_3/vs_flatk3/all/diff|+.4f} "
                 "[{chk:gramm_minus_flat_page_hit_at_3/vs_flatk3/all/ci95/0|+.4f}, "
                 "{chk:gramm_minus_flat_page_hit_at_3/vs_flatk3/all/ci95/1|+.4f}], and on the 750 questions whose "
                 "answer is not page 0 the control leads: "
                 "{chk:gramm_minus_flat_page_hit_at_3/vs_flatk3/true_gold_not_page0/diff|+.4f} "
                 "[{chk:gramm_minus_flat_page_hit_at_3/vs_flatk3/true_gold_not_page0/ci95/0|+.4f}, "
                 "{chk:gramm_minus_flat_page_hit_at_3/vs_flatk3/true_gold_not_page0/ci95/1|+.4f}]"),
     src=(B+"P3_RESULT.md", "§3"), docs_must_show=[B+"P3_RESULT.md"],
     scope="MP-DocVQA, n=950 common set, control at 2.74 images/question against GraMM's 2.74.",
     register="C-80", supersedes="S-40, S-42"),
dict(id="S-57", status=S, area="headline",
     claim="The claim that GraMM-RAG beats the flat-vector baseline on ALL THREE benchmarks holds only against a control reading fewer pages; page-matched it holds on two.",
     value=("Q", "the falsifier FIRED on MP-DocVQA"), src=(B+"FLAT_K3_PREREGISTRATION.md", "Outcome"),
     scope="All three corpora, page-matched control.", register="C-80"),
dict(id="S-58", status=S, area="abstention",
     claim="Against a page-matched control, GraMM-RAG's regression on MMLongBench unanswerable questions is not distinguishable from zero.",
     value=("J", "{ct:mmlongbench/contrasts_vs_gramm/Flat-vector (k=3)/ANLS unanswerable/delta|+.4f} "
                 "for the control minus GraMM, i.e. GraMM "
                 "{ct:mmlongbench/contrasts_vs_gramm/Flat-vector (k=3)/ANLS unanswerable/ci/0|+.4f} to "
                 "{ct:mmlongbench/contrasts_vs_gramm/Flat-vector (k=3)/ANLS unanswerable/ci/1|+.4f} against it"),
     src=(B+"STRATIFIED_REPORT.md", "MMLongBench table"), docs_must_show=[B+"STRATIFIED_REPORT.md"],
     scope="MMLongBench, unanswerable stratum, n=244, page-matched control.",
     register="C-80", supersedes="S-03",
     caveat="This does not make GraMM good at abstention: its balanced accuracy still trails the "
            "control (0.806 vs 0.809), and Gate A's unanswerable regression against the SHIPPED "
            "configuration is what the gate recorded. What is withdrawn is the word 'significantly'."),
dict(id="S-59", status=S, area="conversion",
     claim="At 101+ pages no arm's answer-quality advantage over a page-matched control is negative; GraMM's is its smallest bin while its localisation advantage persists.",
     value=("J", "GraMM dANLS by bin {lb:d_quality/GraMM-RAG/0|+.4f}, {lb:d_quality/GraMM-RAG/1|+.4f}, "
                 "{lb:d_quality/GraMM-RAG/2|+.4f}, {lb:d_quality/GraMM-RAG/3|+.4f}, "
                 "{lb:d_quality/GraMM-RAG/4|+.4f} (101+); d ctx-hit at 101+ {lb:d_localisation/GraMM-RAG/4|+.4f}"),
     src=(B+"PHASE_C_REVISION_LOG.md", "C-81"),
     scope="MMLongBench, five archived length bins, page-matched control; 101+ is n=115.",
     register="C-81", supersedes="S-04",
     caveat="Every graph-arm interval in the 101+ bin spans zero (n=115): this is a statement about "
            "point estimates and about the SIGN not surviving, not a demonstration that GraMM "
            "improves there."),
dict(id="S-60", status=S, area="conversion",
     claim="On documents past 100 pages a third context page costs the flat-vector control more in distraction than it returns in evidence.",
     value=("J", "the k=2 control beats the k=3 control at 101+ by {lb:d_quality/Flat-vector/4|+.4f} ANLS, "
                 "against {lb:d_quality/Flat-vector/0|+.4f} in the 1-20 bin"),
     src=(B+"PHASE_C_REVISION_LOG.md", "C-81"),
     scope="MMLongBench, flat-vector control at two page budgets; a property of the corpus, measured on the control.",
     register="C-81"),
dict(id="S-52", status=S, area="localisation",
     claim="On DocBench, LightRAG leads GraMM-RAG on answer quality; the localisation difference is not distinguishable from zero once both arms are scored on the pages their generators saw.",
     value=("J", "judge accuracy {ct:docbench/contrasts_vs_gramm/LightRAG 1.5.7/judge accuracy pooled/delta|+.4f} "
                 "[{ct:docbench/contrasts_vs_gramm/LightRAG 1.5.7/judge accuracy pooled/ci/0|+.4f}, "
                 "{ct:docbench/contrasts_vs_gramm/LightRAG 1.5.7/judge accuracy pooled/ci/1|+.4f}]; "
                 "quote-recall@5 {ct:docbench/contrasts_vs_gramm/LightRAG 1.5.7/quote-recall@5/delta|+.4f} "
                 "[{ct:docbench/contrasts_vs_gramm/LightRAG 1.5.7/quote-recall@5/ci/0|+.4f}, "
                 "{ct:docbench/contrasts_vs_gramm/LightRAG 1.5.7/quote-recall@5/ci/1|+.4f}]"),
     src=(B+"STRATIFIED_REPORT.md", "§0, DocBench table"), docs_must_show=[B+"STRATIFIED_REPORT.md"],
     scope="DocBench; uniform generator-context scoring for every arm.", register="C-77", supersedes="S-13"),
dict(id="S-53", status=S, area="localisation",
     claim="GraMM-RAG leads LightRAG on localisation on MMLongBench; on DocBench the two are not distinguishable. The ranking does not reverse.",
     value=("J", "MMLongBench ctx-hit GraMM {ct:mmlongbench/rows/GraMM-RAG/ctx-hit (n=838)|.4f} vs LightRAG "
                 "{ct:mmlongbench/rows/LightRAG 1.5.7/ctx-hit (n=838)|.4f}; DocBench quote-recall@5 "
                 "{ct:docbench/rows/GraMM-RAG/quote-recall@5 (n=730)|.4f} vs "
                 "{ct:docbench/rows/LightRAG 1.5.7/quote-recall@5 (n=730)|.4f}"),
     src=(B+"STRATIFIED_REPORT.md", "§0"), docs_must_show=[B+"STRATIFIED_REPORT.md"],
     scope="Localisation only: ctx-hit on MMLongBench, quote-recall@5 on DocBench, both on the pages the generator saw.",
     register="C-77", supersedes="R-01, S-10",
     caveat="Not 'leadership over LightRAG' unqualified: one corpus leads, the other is null, and DocBench answer quality still favours LightRAG (S-52)."),
dict(id="S-51", status=C, area="P3",
     claim="GraMM-RAG's P3 row on MP-DocVQA measures a model trained on corrupted gold-page labels.",
     value=("J", "page-hit@3 {loc950:gramm/page_hit_at_3|.4f}, ANLS {anls:per_arm/GraMM-RAG/anls|.4f}, n={anls:n_common}"),
     src=(B+"MPDOCVQA_GOLDPAGE_DEFECT.md", "§2f"),
     docs_must_show=[B+"MPDOCVQA_GOLDPAGE_DEFECT.md", B+"PHASE_C_REVISION_LOG.md", B+"P3_RESULT.md"],
     scope="MP-DocVQA P3 only; MMLongBench and DocBench use separate gold code.", register="C-66",
     caveat="Bias runs both ways: the APPA defect flattered GraMM, the training defect plausibly penalised it."),
]

# ============================== TABLES checked CELL BY CELL ==============================
# A number-presence check cannot catch a hand-transcribed cell that drifts: "0.477" may appear
# elsewhere in the file. P3_RESULT.md's table had exactly one such cell (MS GraphRAG SD 0.478 vs
# the data's 0.4775) -- so hand-written tables are checked cell by cell against their JSON.
_P3K = {"GraMM-RAG": "gramm", "Flat-vector": "flat", "Flat-vector (k=3)": "flatk3",
        "LightRAG": "lightrag", "RAG-Anything": "s2", "MS GraphRAG": "s5"}
TABLES = [dict(
    id="T-P3", doc=B + "P3_RESULT.md", header="| arm | page-hit@3 | hit@any",
    also_show=["gramm-rag-colpali/phaseC_evidence.ipynb"],       # computed there -- presence suffices
    cols=["page-hit@3", "hit@any", "ANLS", "SD", "img/q"],
    rows={arm: {"page-hit@3": "{loc950:%s/page_hit_at_3|.4f}" % k,
                "hit@any": ("—" if k in ("gramm", "flatk3") else "{loc950:%s/hit_any|.4f}" % k),
                "ANLS": "{anls:per_arm/%s/anls|.4f}" % arm,
                "SD": "{anls:per_arm/%s/sd|.3f}" % arm,
                "img/q": "{anls:per_arm/%s/images_per_question|.2f}" % arm}
          for arm, k in _P3K.items()})]
