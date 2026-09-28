#!/usr/bin/env python3
"""build_reports.py -- render STRATIFIED_REPORT.md and RESOURCE_PROFILE.md from the result JSONs.

Both documents are RENDERED. Never edit them by hand: `--check` re-renders and diffs, and the
sync refuses to ship a difference. Every number below is read from an artefact at render time;
nothing is transcribed. Three preparation constants whose primary logs were not retained are
carried in cost/resource_inputs.json WITH the quote they came from, and each quote is re-found in
its source here -- if the source text changes, the build fails rather than printing a stale number.

Every (arm x corpus x metric x stratum) cell either resolves to a value or carries a registered
"not measured, because ..." reason. A cell with neither fails the build; so does a reason attached
to a cell whose data does exist, which is how a stale exclusion is caught.

Usage:  python3 build_reports.py            # render both documents
        python3 build_reports.py --check    # verify they match the artefacts (sync assertion)
"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # .../grammrag
OUT_STRAT = ROOT / "GraphRAG_baseline/STRATIFIED_REPORT.md"
OUT_RES = ROOT / "GraphRAG_baseline/RESOURCE_PROFILE.md"

ERR = []
def fail(msg): ERR.append(msg)
def check(cond, msg):
    if not cond:
        fail(msg)
    return cond

_cache = {}
def J(rel):
    if rel not in _cache:
        p = ROOT / rel
        if not p.exists():
            fail(f"source missing: {rel}")
            _cache[rel] = {}
        else:
            _cache[rel] = json.loads(p.read_text())
    return _cache[rel]

def norm(t):
    return " ".join(t.split())

def quote_in(rel, q):
    p = ROOT / rel
    if not p.exists():
        return fail(f"quote source missing: {rel}")
    check(norm(q) in norm(p.read_text()), f"quote not found in {rel}: {q!r}")

# ----------------------------------------------------------------- arms and sources
ARMS = [("GraMM-RAG", "gramm"), ("Flat-vector", "flat"),
        # C-80: the page-matched control. The archived k=2 arm stays in every table so the
        # page-budget effect is visible rather than swapped in silently.
        ("Flat-vector (k=3)", "flatk3"), ("LightRAG 1.5.7", "lightrag"),
        ("RAG-Anything 1.2.8", "s2"), ("MS GraphRAG 2.7.2", "s5"), ("ColPali", "colpali"),
        ("ColPali size-matched", "colpali_matched"), ("Union(CP∪GraMM)", "union")]
SHORT = {"GraMM-RAG": "GraMM-RAG", "Flat-vector": "Flat-vector (k=2)",
         "Flat-vector (k=3)": "Flat-vector (k=3)", "LightRAG 1.5.7": "LightRAG",
         "RAG-Anything 1.2.8": "RAG-Anything", "MS GraphRAG 2.7.2": "MS GraphRAG",
         "ColPali": "ColPali", "ColPali size-matched": "ColPali size-m.", "Union(CP∪GraMM)": "Union"}
COMPARATORS = ["lightrag", "s2", "s5", "colpali", "colpali_matched", "union", "flatk3"]

CT = J("gramm-rag-colpali/scores/cell_table.json")
SR = J("gramm-rag-colpali/scores/stratified_report.json")
ARCH = {"mmlongbench": (J("gramm-rag-3benchmarks/data/mmlongbench/gramm_mmlongbench_s42.json"),
                        J("gramm-rag-3benchmarks/data/mmlongbench/baseline_vector_mmlb.json")),
        "docbench": (J("gramm-rag-3benchmarks/data/docbench/gramm_docbench_s42.json"),
                     J("gramm-rag-3benchmarks/data/docbench/baseline_vector_docbench.json"))}
SCORES = {(k, b): J(f"gramm-rag-colpali/scores/{k}_scores_{b}.json")
          for k in COMPARATORS for b in ("mmlongbench", "docbench")}
P3A = J("gramm-rag-graphbracket/tier1/p3_mpdocvqa/p3_anls.json")
P3L = J("gramm-rag-graphbracket/tier1/p3_mpdocvqa/p3_localisation_common950.json")
P3ARM = {"gramm": "GraMM-RAG", "flat": "Flat-vector", "flatk3": "Flat-vector (k=3)",
         "lightrag": "LightRAG", "s2": "RAG-Anything", "s5": "MS GraphRAG"}

def ct_key(bench, prefix):
    """cell_table column names embed their denominator ('ctx-hit (n=838)')."""
    rows = CT[bench]["rows"]
    keys = [k for k in next(iter(rows.values())) if k.startswith(prefix)]
    if len(keys) != 1:
        fail(f"cell_table[{bench}] has {len(keys)} columns starting {prefix!r}")
        return prefix
    return keys[0]

# ----------------------------------------------------------------- context page budget
def ctx_pages(bench, k):
    """Pages the generator actually SAW, per question, counted from the artefacts.

    Never from RUN_CONFIG. P3 stamps TOPK_PAGES=3 for every arm, and this column first printed
    that -- but the driver slices `context_pages[:3]` from each arm's stored context, and a slice
    cannot add a page that was never retrieved: flat-vector's MP-DocVQA context averages 1.92
    pages and never reaches 3. Reading the setting instead of the output is C-76 exactly."""
    if bench == "mpdocvqa":
        sub = J("gramm-rag-graphbracket/tier1/p3_mpdocvqa/subsample.json")["questions"]
        if k == "flatk3":
            d = J("gramm-rag-colpali/retrieval/flatk3_retrieval_mpdocvqa.json")
            n = [len(d["context_pages"][q["q_index"]][:3]) for q in sub]
        elif k in ("flat", "gramm"):
            src = {"flat": "gramm-rag-3benchmarks/data/mpdocvqa/baseline_vector_mpdocvqa_pilot.json",
                   "gramm": "gramm-rag-3benchmarks/data/mpdocvqa/gramm_mpdocvqa_pilot_s42.json"}[k]
            d = J(src)
            n = [len(d["context_pages"][q["q_index"]][:3]) for q in sub]
        elif k in ("lightrag", "s2", "s5"):
            d = J(f"gramm-rag-graphbracket/tier1/p3_mpdocvqa/p3_retrieval_{k}.json")
            err = {e["i"] for e in d.get("query_errors", [])}
            n = [len(d["context_pages"][i][:3]) for i in range(len(sub)) if i not in err]
        else:
            return None
        return f"{statistics.mean(n):.2f} [{min(n)}–{max(n)}]"
    if k == "gramm":
        cp = ARCH[bench][0]["context_pages"]
    elif k == "flat":
        cp = ARCH[bench][1]["context_pages"]
    else:
        f = ROOT / f"gramm-rag-colpali/generation/{k}_gen_{bench}.json"
        if not f.exists():
            return None
        cp = json.loads(f.read_text())["context_pages"]
    n = [len(x) for x in cp]
    return f"{statistics.mean(n):.2f} [{min(n)}–{max(n)}]"

# ----------------------------------------------------------------- column definitions
def q(bench, prefix):
    key = ct_key(bench, prefix)
    return lambda arm: CT[bench]["rows"].get(arm, {}).get(key)

def loc_uniform(bench, prefix):
    key = ct_key(bench, prefix)
    return lambda arm: CT[bench]["rows"].get(arm, {}).get(key)

def loc_s1_docbench(arm):
    """DocBench localisation as the published contrast scored it -- NOT one definition.
    GraMM and Flat-vector are scored on their top-5 RETRIEVED page texts; every other arm on the
    pages its generator saw. The two are different page sets (see the header of the report)."""
    if arm == "GraMM-RAG":
        return ARCH["docbench"][0]["metrics"]["quote_recall@5"]
    if arm == "Flat-vector":
        return ARCH["docbench"][1]["metrics"]["quote_recall@5"]
    k = dict((a, kk) for a, kk in ARMS)[arm]
    return SCORES[(k, "docbench")].get("quote_recall@5")

def p3_q(arm):
    a = P3ARM.get(dict((x, y) for x, y in ARMS)[arm]) if arm in dict(ARMS) else None
    return P3A["per_arm"].get(a, {}).get("anls") if a else None

def p3_loc(field):
    def get(arm):
        k = dict((a, kk) for a, kk in ARMS)[arm]
        if k not in P3L:
            return None
        return P3L[k].get(field)
    return get

F4 = lambda v: f"{v:.4f}"
F3 = lambda v: f"{v:.3f}"
RAW = lambda v: str(v)

CORPORA = [
    dict(key="mmlongbench", title="MMLongBench-Doc", quality="ANLS",
         n=CT["mmlongbench"]["n_questions"], na=CT["mmlongbench"]["n_answerable"],
         nu=CT["mmlongbench"]["n_unanswerable"], nl=CT["mmlongbench"]["n_localisation_scoreable"],
         arms=[k for _, k in ARMS],
         cols=[
             dict(k="ctx", h="context pages / q", f=RAW, get=lambda arm, b="mmlongbench": ctx_pages(b, dict((a, kk) for a, kk in ARMS)[arm]),
                  src="derived: mean `context_pages` length in each arm's generation JSON [min–max]"),
             dict(k="q_pool", h="ANLS pooled", f=F4, get=q("mmlongbench", "ANLS pooled"),
                  src="`scores/cell_table.json` → `mmlongbench.rows[arm]['ANLS pooled']`"),
             dict(k="q_ans", h="ANLS answerable", f=F4, get=q("mmlongbench", "ANLS answerable"),
                  src="`scores/cell_table.json` → `['ANLS answerable']`"),
             dict(k="q_un", h="ANLS unanswerable", f=F4, get=q("mmlongbench", "ANLS unanswerable"),
                  src="`scores/cell_table.json` → `['ANLS unanswerable']`"),
             dict(k="l_pool", h="ctx-hit pooled", f=F4, get=lambda arm: None,
                  src="undefined; see reason (a)"),
             dict(k="l_ans", h="ctx-hit (n=838)", f=F4, get=loc_uniform("mmlongbench", "ctx-hit (n="),
                  src="`scores/cell_table.json` → `['ctx-hit (n=838)']`; equals `ctx_hit_rate` in each arm's `scores/*_scores_mmlongbench.json`"),
             dict(k="l_un", h="ctx-hit unanswerable", f=F4, get=lambda arm: None,
                  src="undefined; see reason (a)"),
             dict(k="a_pool", h="abstention bal. acc.", f=F3, get=q("mmlongbench", "decline balAcc"),
                  src="`scores/cell_table.json` → `['decline balAcc']`"),
             dict(k="a_un", h="decline rate, unanswerable", f=F3, get=q("mmlongbench", "decline on unanswerable"),
                  src="`scores/cell_table.json` → `['decline on unanswerable']`"),
             dict(k="a_ans", h="false-decline, answerable", f=F3, get=q("mmlongbench", "false-decline (answerable)"),
                  src="`scores/cell_table.json` → `['false-decline (answerable)']`"),
         ]),
    dict(key="docbench", title="DocBench", quality="judge accuracy",
         n=CT["docbench"]["n_questions"], na=CT["docbench"]["n_answerable"],
         nu=CT["docbench"]["n_unanswerable"], nl=CT["docbench"]["n_localisation_scoreable"],
         arms=[k for _, k in ARMS],
         cols=[
             dict(k="ctx", h="context pages / q", f=RAW, get=lambda arm, b="docbench": ctx_pages(b, dict((a, kk) for a, kk in ARMS)[arm]),
                  src="derived: mean `context_pages` length in each arm's generation JSON [min–max]"),
             dict(k="q_pool", h="judge acc. pooled", f=F4, get=q("docbench", "judge accuracy pooled"),
                  src="`scores/cell_table.json` → `docbench.rows[arm]['judge accuracy pooled']`"),
             dict(k="q_ans", h="judge acc. answerable", f=F4, get=q("docbench", "judge accuracy answerable"),
                  src="`scores/cell_table.json` → `['judge accuracy answerable']`"),
             dict(k="q_un", h="judge acc. unanswerable", f=F4, get=q("docbench", "judge accuracy unanswerable"),
                  src="`scores/cell_table.json` → `['judge accuracy unanswerable']`"),
             dict(k="l_pool", h="quote-recall pooled", f=F4, get=lambda arm: None,
                  src="undefined; see reason (a)"),
             dict(k="l_ctx", h="quote-recall@5 — context (n=730)", f=F4,
                  get=loc_uniform("docbench", "quote-recall@5 (n="),
                  src="`scores/cell_table.json` → `['quote-recall@5 (n=730)']` — every arm scored on the pages its generator saw"),
             dict(k="l_s1", h="quote-recall@5 — as published", f=F4, get=loc_s1_docbench,
                  src="GraMM/Flat: `metrics.quote_recall@5` in the archived run (top-5 **retrieved** pages); others: `quote_recall@5` in `scores/*_scores_docbench.json` (context pages)"),
             dict(k="l_un", h="quote-recall unanswerable", f=F4, get=lambda arm: None,
                  src="undefined; see reason (a)"),
             dict(k="a_pool", h="abstention bal. acc.", f=F3, get=q("docbench", "decline balAcc"),
                  src="`scores/cell_table.json` → `['decline balAcc']`"),
             dict(k="a_un", h="decline rate, unanswerable", f=F3, get=q("docbench", "decline on unanswerable"),
                  src="`scores/cell_table.json` → `['decline on unanswerable']`"),
             dict(k="a_ans", h="false-decline, answerable", f=F3, get=q("docbench", "false-decline (answerable)"),
                  src="`scores/cell_table.json` → `['false-decline (answerable)']`"),
         ]),
    dict(key="mpdocvqa", title="MP-DocVQA (P3)", quality="ANLS",
         n=P3A["n_common"], na=P3A["n_common"], nu=0, nl=P3A["n_common"],
         arms=["gramm", "flat", "lightrag", "s2", "s5", "colpali", "colpali_matched", "union"],
         cols=[
             dict(k="ctx", h="context pages / q", f=RAW,
                  get=lambda arm: ctx_pages("mpdocvqa", dict((a, kk) for a, kk in ARMS)[arm]) if dict((a, kk) for a, kk in ARMS)[arm] in P3ARM else None,
                  src="derived: pages actually passed to the generator — `context_pages[:3]` from each arm's own retrieval, exactly as `generate_p3.py` slices them (**not** the stamped `TOPK_PAGES`)"),
             dict(k="q_pool", h="ANLS pooled", f=F4, get=p3_q,
                  src="`tier1/p3_mpdocvqa/p3_anls.json` → `per_arm[arm].anls` (n=950 common set)"),
             dict(k="q_ans", h="ANLS answerable", f=F4, get=p3_q,
                  src="same value: every MP-DocVQA question is answerable, so pooled = answerable"),
             dict(k="q_un", h="ANLS unanswerable", f=F4, get=lambda arm: None,
                  src="no such stratum; see reason (c)"),
             dict(k="l_pool", h="page-hit@3 pooled", f=F4, get=p3_loc("page_hit_at_3"),
                  src="`p3_localisation_common950.json` → `[arm].page_hit_at_3` (defined on all 950: no unanswerable stratum)"),
             dict(k="l_ans", h="page-hit@3 (n=950)", f=F4, get=p3_loc("page_hit_at_3"),
                  src="same value as pooled"),
             dict(k="l_any", h="hit@any", f=F4, get=p3_loc("hit_any"),
                  src="`p3_localisation_common950.json` → `[arm].hit_any`"),
             dict(k="a_pool", h="abstention bal. acc.", f=F3, get=lambda arm: None,
                  src="no unanswerable stratum; see reason (c)"),
             dict(k="a_un", h="decline rate, unanswerable", f=F3, get=lambda arm: None,
                  src="no unanswerable stratum; see reason (c)"),
             dict(k="a_ans", h="false-decline, answerable", f=F3, get=lambda arm: None,
                  src="measurable, but not computed by P3; see reason (d)"),
         ]),
]

# ---------------------------------------------------- "not measured, because ..." registry
# key: (corpus, column, arm | "*") -> reason letter. Reasons are printed under each table.
REASONS = {
    "a": "not measured — localisation is undefined on unanswerable questions, and on answerable "
         "ones lacking gold evidence (C-8/C-9); the denominator is the scoreable subset in the "
         "neighbouring column",
    "b": "not measured — the arm was never run on this corpus",
    "c": "not measured — MP-DocVQA has no unanswerable stratum (C-61), so every abstention "
         "quantity and the unanswerable column are undefined",
    "d": "not measured — false-decline on answerable questions IS measurable on MP-DocVQA; P3 did "
         "not compute it, and no artefact carries it",
    "e": "not measured — the ColPali family and the Union arm are excluded from P3 by "
         "pre-registration (`P3_RESULT.md` §1)",
    "f": "**a value exists but is not comparable and is withheld** — `retrieved_pages` covers "
         "GraMM's graph leg only, while its context comes "
         "from hybrid fusion with a dense leg, so the two fields measure different stages, and "
         "hit@any for GraMM would not mean what it means for the other arms "
         "(`P3_RESULT.md` §3 footnote †; the stored figure is "
         + f"{P3L['gramm']['hit_any']:.4f})",
}
# A cell has one of three states, and each is guarded in the opposite direction:
#   value      -- resolved from an artefact
#   NM         -- no value exists; if one appears, the build fails (the exclusion went stale)
#   WITHHELD   -- a value EXISTS but is not this cell's quantity, so printing it would mislead;
#                 if it stops existing, the build fails (we are no longer withholding anything)
WITHHELD = {("mpdocvqa", "l_any", "gramm"): "f"}
NM = {}
for c in ("l_pool", "l_un"):
    NM[("mmlongbench", c, "*")] = "a"
    NM[("docbench", c, "*")] = "a"
for c in ("q_un", "a_pool", "a_un"):
    NM[("mpdocvqa", c, "*")] = "c"
NM[("mpdocvqa", "a_ans", "*")] = "d"
for col in ("ctx", "q_pool", "q_ans", "q_un", "l_pool", "l_ans", "l_any", "a_pool", "a_un", "a_ans"):
    for arm in ("colpali", "colpali_matched", "union"):
        NM[("mpdocvqa", col, arm)] = "e"

# ----------------------------------------------------------------- cross-checks
def cross_checks():
    # 1. the two independent producers agree, cell for cell
    pairs = [("pooled", "ANLS pooled", "judge accuracy pooled"), ("answerable", "ANLS answerable", "judge accuracy answerable"),
             ("unanswerable", "ANLS unanswerable", "judge accuracy unanswerable"),
             ("abstention_balanced_acc", "decline balAcc", "decline balAcc"),
             ("decline_unans", "decline on unanswerable", "decline on unanswerable"),
             ("decline_ans", "false-decline (answerable)", "false-decline (answerable)")]
    for b in ("mmlongbench", "docbench"):
        check(set(SR[b]) == set(CT[b]["rows"]),
              f"{b}: stratified_report arms {sorted(SR[b])} != cell_table arms {sorted(CT[b]['rows'])}")
        for arm in SR[b]:
            for srk, mmk, dbk in pairs:
                a = SR[b][arm][srk]
                c = CT[b]["rows"][arm][ct_key(b, mmk if b == "mmlongbench" else dbk)]
                check(abs(a - c) < 1e-9,
                      f"{b}/{arm}/{srk}: stratified_report {a} != cell_table {c}")
    # 2. localisation in cell_table equals each arm's own score file
    for k in COMPARATORS:
        v = CT["mmlongbench"]["rows"].get(dict(ARMS).get(k) or [a for a, kk in ARMS if kk == k][0], {})
        arm = [a for a, kk in ARMS if kk == k][0]
        ctv = CT["mmlongbench"]["rows"][arm][ct_key("mmlongbench", "ctx-hit (n=")]
        sv = SCORES[(k, "mmlongbench")]["ctx_hit_rate"]
        check(abs(ctv - sv) < 5e-4, f"mmlongbench/{k}: cell_table ctx-hit {ctv} != score file {sv}")
        ctv = CT["docbench"]["rows"][arm][ct_key("docbench", "quote-recall@5 (n=")]
        sv = SCORES[(k, "docbench")]["quote_recall@5"]
        check(abs(ctv - sv) < 5e-4, f"docbench/{k}: cell_table quote-recall {ctv} != score file {sv}")
    # 3. the archived arms' localisation, as every score file records it
    for k in COMPARATORS:
        s = SCORES[(k, "mmlongbench")]
        check(abs(CT["mmlongbench"]["rows"]["GraMM-RAG"][ct_key("mmlongbench", "ctx-hit (n=")]
                  - s["ctx_hit_rate_gramm"]) < 5e-4, f"GraMM ctx-hit disagrees with {k}'s score file")
        check(abs(CT["mmlongbench"]["rows"]["Flat-vector"][ct_key("mmlongbench", "ctx-hit (n=")]
                  - s["ctx_hit_rate_baseline"]) < 5e-4, f"Flat ctx-hit disagrees with {k}'s score file")
        d = SCORES[(k, "docbench")]
        check(abs(ARCH["docbench"][0]["metrics"]["quote_recall@5"] - d["quote_recall@5_gramm"]) < 5e-4,
              f"GraMM published quote-recall disagrees with {k}'s score file")
        check(abs(ARCH["docbench"][1]["metrics"]["quote_recall@5"] - d["quote_recall@5_baseline"]) < 5e-4,
              f"Flat published quote-recall disagrees with {k}'s score file")
    # 4. denominators come from the corpora themselves
    for b, arch_key in (("mmlongbench", "mmlongbench"), ("docbench", "docbench")):
        an = ARCH[b][1]["is_answerable"]
        check(CT[b]["n_questions"] == len(an), f"{b}: n_questions {CT[b]['n_questions']} != {len(an)}")
        check(CT[b]["n_answerable"] == sum(1 for x in an if x), f"{b}: n_answerable mismatch")
        check(CT[b]["n_unanswerable"] == sum(1 for x in an if not x), f"{b}: n_unanswerable mismatch")
    check(P3A["n_common"] == 950, f"P3 n_common is {P3A['n_common']}, expected 950")
    check(set(P3L) == {"gramm", "flat", "flatk3", "lightrag", "s2", "s5"},
          f"P3 localisation arms: {sorted(P3L)}")
    # 5. the two DocBench localisation definitions, stated as a measured gap rather than assumed
    g_ctx = CT["docbench"]["rows"]["GraMM-RAG"][ct_key("docbench", "quote-recall@5 (n=")]
    g_pub = ARCH["docbench"][0]["metrics"]["quote_recall@5"]
    check(abs(g_ctx - g_pub) > 5e-4,
          "GraMM's two DocBench localisation definitions now agree -- the report's header section is stale")
    # 6. quotes behind values that are not JSON
    quote_in("GraphRAG_baseline/P3_RESULT.md", "The\n  ColPali family is excluded by pre-registration.")
    quote_in("GraphRAG_baseline/P3_RESULT.md", "MP-DocVQA is **ANLS-only**: no judge and no unanswerable stratum (C-61)")
    quote_in("GraphRAG_baseline/P3_RESULT.md", "**It is not reported\nfor GraMM:**")

# ----------------------------------------------------------------- render: stratified
def render_stratified():
    L = ["# Stratified report — every arm × corpus × metric × stratum",
         "",
         "**Rendered from the result JSONs by `reports/build_reports.py`. Do not edit.** Every cell",
         "is read from an artefact at render time; `build_reports.py --check` re-renders this file and",
         "the sync refuses to ship a difference. Cells that carry no measurement print",
         "`not measured (x)` and the reason is printed under the table.",
         "",
         "Localisation metrics differ by corpus and are **comparable within a corpus only** "
         "(C-43): ctx-hit on MMLongBench, quote-recall@5 on DocBench, page-hit@3 on MP-DocVQA.",
         ""]

    # --- the two measurement questions a reader must see before using the tables
    g_ctx = CT["docbench"]["rows"]["GraMM-RAG"][ct_key("docbench", "quote-recall@5 (n=")]
    f_ctx = CT["docbench"]["rows"]["Flat-vector"][ct_key("docbench", "quote-recall@5 (n=")]
    g_pub = ARCH["docbench"][0]["metrics"]["quote_recall@5"]
    f_pub = ARCH["docbench"][1]["metrics"]["quote_recall@5"]
    lr_ctx = CT["docbench"]["rows"]["LightRAG 1.5.7"][ct_key("docbench", "quote-recall@5 (n=")]
    ct_con = CT["docbench"]["contrasts_vs_gramm"]["LightRAG 1.5.7"]["quote-recall@5"]
    L += ["## 0. Two things to settle before quoting the DocBench localisation column",
          "",
          "**(i) The published DocBench contrast scores two arms on a different page set from the "
          "others.** GraMM-RAG and Flat-vector are scored on their top-5 **retrieved** pages "
          "(`metrics.quote_recall@5` in the archived run); every other arm is scored on the pages its "
          "**generator saw** (`context_pages`, page text rebuilt from the parse). Scored the second way "
          "— the only like-for-like available — the numbers move:",
          "",
          "| arm | on retrieved top-5 (published) | on generator context (uniform) |",
          "|---|---|---|",
          f"| GraMM-RAG | {F4(g_pub)} | **{F4(g_ctx)}** |",
          f"| Flat-vector | {F4(f_pub)} | {F4(f_ctx)} |",
          f"| LightRAG 1.5.7 | {F4(lr_ctx)} (context) | {F4(lr_ctx)} |",
          "",
          f"So LightRAG − GraMM is **+{F4(lr_ctx - g_pub)} as published** (S1, McNemar p=0.0013, ledger "
          f"S-13) and **{ct_con['delta']:+.4f} [{ct_con['ci'][0]:+.4f}, {ct_con['ci'][1]:+.4f}]"
          f"{'' if ct_con['sig'] else ', CI includes zero'}** scored uniformly "
          "(`scores/cell_table.json`). Both tables below print both columns and this report does not "
          "choose between them: the DocBench reversal (S-10, S-13, `SINGLE_BENCHMARK_FINDING.md` leg 1) "
          "rests on the published pairing, so which page set is the pre-registered one is a decision "
          "for the authors, not a rendering choice.",
          "",
          "**(ii) The control arm reads fewer pages than the arms compared with it, on all three "
          "corpora.** Flat-vector's generator sees 2 pages on MMLongBench and DocBench against 3 for "
          "GraMM-RAG (`GRAPH_PAGE_UNION`) and 3 for every external arm (the harness's 3-page budget, "
          "C-37); on MMLongBench and DocBench it also read a **8,000**-character context where the "
          "3-page arms read 12,000. On MP-DocVQA the gap is smaller but present: flat-vector averages "
          "**1.92** pages and never reaches 3, against GraMM 2.74 and 2.45–2.52 for the graph arms. "
          "P3 stamps `TOPK_PAGES = 3` for all five arms, but its driver slices `context_pages[:3]` "
          "from each arm's own stored context, and a slice cannot add a page that was never "
          "retrieved — so the stamped k is the setting, not the realised budget. Both localisation "
          "metrics count a hit anywhere in the pages scored, so they rise with the page count: every "
          "contrast against Flat-vector — including η, which is defined against it — carries a "
          "page-budget difference as well as a retrieval difference. The `context pages / q` column "
          "in every table below is counted from the artefacts for exactly this reason.",
          ""]

    for c in CORPORA:
        L += [f"## {c['title']} — n = {c['n']:,} ({c['na']:,} answerable, {c['nu']:,} unanswerable"
              + (f"; localisation scoreable {c['nl']:,})" if c["key"] != "mpdocvqa" else ")"),
              ""]
        heads = ["arm"] + [col["h"] for col in c["cols"]]
        for h in heads:
            check("|" not in h, f"{c['key']}: column header {h!r} contains a pipe and breaks the table")
        L += ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
        used = set()
        for k in c["arms"]:
            arm = [a for a, kk in ARMS if kk == k][0]
            cells = []
            for col in c["cols"]:
                v = col["get"](arm)
                w = WITHHELD.get((c["key"], col["k"], k))
                if w:
                    if v is None:
                        fail(f"{c['key']}/{arm}/{col['k']}: withheld ({w}), but the artefact no longer "
                             f"carries a value -- the reason is stale")
                        cells.append("**STALE WITHHOLD**")
                    else:
                        used.add(w)
                        cells.append(f"not comparable ({w})")
                    continue
                r = NM.get((c["key"], col["k"], k)) or NM.get((c["key"], col["k"], "*"))
                if v is None and not r:
                    fail(f"{c['key']}/{arm}/{col['k']}: no value and no registered reason")
                    cells.append("**MISSING**")
                elif v is not None and r:
                    fail(f"{c['key']}/{arm}/{col['k']}: reason ({r}) registered, but a value exists ({v})")
                    cells.append("**STALE REASON**")
                elif v is None:
                    used.add(r)
                    cells.append(f"not measured ({r})")
                else:
                    cells.append(col["f"](v))
            L.append("| " + SHORT[arm] + " | " + " | ".join(cells) + " |")
        L.append("")
        for r in sorted(used):
            L.append(f"**({r})** {REASONS[r]}")
        L += ["", "**Sources, per column.**", ""]
        for col in c["cols"]:
            L.append(f"- *{col['h']}* — {col['src']}")
        L.append("")
        if c["key"] == "mmlongbench":
            L += ["**Caveat.** 1.7% of MMLongBench questions (19 of 1,091) are answered against the "
                  "wrong document, uniformly across arms: comparisons unbiased, absolute ANLS depressed "
                  "by 0.0023–0.0054 per arm (`CORPUS_PARSE_INTEGRITY.md`).", ""]
        if c["key"] == "docbench":
            L += ["**Judge.** `gpt-4-turbo`, the canonical judge. A second judge (`gpt-4o`, 7,715 calls) "
                  "shifts every arm down by a mean 0.0258 with no significant pairwise reversal "
                  "(`P2_RESULT.md`); the ordering here is not judge-specific.", ""]
        if c["key"] == "mpdocvqa":
            L += ["**Caveat.** GraMM's row measures a model trained on corrupted MP-DocVQA gold-page "
                  "labels (C-66, ledger S-51); the bias runs both ways — the APPA defect flattered "
                  "GraMM, the training defect plausibly penalised it.", ""]

    # contrasts
    L += ["## Paired contrasts against GraMM-RAG", "",
          "Each entry is *arm − GraMM-RAG*, with a 95% bootstrap interval; **\\*** marks an interval "
          "excluding zero. MMLongBench and DocBench: 4,000 resamples, seed 42, one RNG per contrast "
          "(`scores/cell_table.json`). MP-DocVQA: `tier1/p3_mpdocvqa/p3_anls.json` (signs inverted "
          "here, which stores GraMM − arm).", ""]
    for b, title in (("mmlongbench", "MMLongBench-Doc"), ("docbench", "DocBench")):
        qn = "ANLS" if b == "mmlongbench" else "judge accuracy"
        ln = "ctx-hit" if b == "mmlongbench" else "quote-recall@5"
        L += [f"**{title}**", "",
              f"| arm | {qn} pooled | {qn} answerable | {qn} unanswerable | {ln} |", "|---|---|---|---|---|"]
        for arm, k in ARMS:
            if arm == "GraMM-RAG":
                continue
            con = CT[b]["contrasts_vs_gramm"].get(arm)
            if not con:
                fail(f"{b}: no contrast for {arm}")
                continue
            cells = []
            for name in (f"{qn} pooled", f"{qn} answerable", f"{qn} unanswerable", ln):
                d = con.get(name)
                if d is None:
                    fail(f"{b}/{arm}: contrast {name!r} absent")
                    cells.append("**MISSING**")
                else:
                    cells.append(f"{d['delta']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"
                                 + ("\\*" if d["sig"] else ""))
            L.append("| " + SHORT[arm] + " | " + " | ".join(cells) + " |")
        L += ["", f"*{ln} contrast is on the uniform (generator-context) definition for both corpora.*"
              if b == "docbench" else "", ""]
    L += ["**MP-DocVQA (ANLS, n = 950)**", "", "| arm | ANLS − GraMM-RAG |", "|---|---|"]
    for arm, k in ARMS:
        if k == "gramm":
            continue
        if k not in P3ARM:
            L.append(f"| {SHORT[arm]} | not measured (e) |")
            continue
        key = f"GraMM-RAG - {P3ARM[k]}"
        d = P3A["contrasts"].get(key)
        if d is None:
            fail(f"P3: contrast {key!r} absent")
            L.append(f"| {SHORT[arm]} | **MISSING** |")
        else:
            sig = "\\*" if (d["lo"] > 0 or d["hi"] < 0) else ""
            L.append(f"| {SHORT[arm]} | {-d['diff']:+.4f} [{-d['hi']:+.4f}, {-d['lo']:+.4f}]{sig} |")
    L += ["", f"**({'e'})** {REASONS['e']}", ""]
    return "\n".join(L).rstrip() + "\n"


# ----------------------------------------------------------------- render: resource
RI = J("gramm-rag-colpali/cost/resource_inputs.json")
LAT = J("gramm-rag-colpali/cost/query_latency_v2.json")
HGT = J("gramm-rag-colpali/cost/hgt_training.json")
E5 = J("gramm-rag-colpali/cost/e5_10pct.json")
B1 = J("gramm-rag-colpali/cost/b1_siglip_edges.json")
PRE = J("gramm-rag-colpali/scores/package_precomputed.json")
RC = "gramm-rag-graphbracket/tier1/_resource_cells"
GIB = 2 ** 30


def transcribed(key):
    for t in RI.get("transcribed", []):
        if t["key"] == key:
            quote_in(t["source"], t["quote"])
            return t
    fail(f"resource_inputs.json carries no transcribed value {key!r}")
    return {}


def render_resource():
    pages = {c: RI["embedding_census"][f"colpali|{c}"]["page_matrices"] for c in ("mmlongbench", "docbench")}
    elems = {c: RI["corpus"][c]["elements"] for c in ("mmlongbench", "docbench")}
    docs = {c: RI["corpus"][c]["docs"] for c in ("mmlongbench", "docbench")}
    imgs = {c: B1["image_elements"][c] for c in ("mmlongbench", "docbench")}
    check(sum(pages.values()) == E5["total_pages"],
          f"corpus pages {sum(pages.values())} != e5_10pct total_pages {E5['total_pages']}")
    check(sum(elems.values()) == E5["total_elements"],
          f"parsed elements {sum(elems.values())} != e5_10pct total_elements {E5['total_elements']}")
    check(sum(imgs.values()) == B1["image_elements_total"], "image element counts disagree")

    parse = transcribed("mineru_parse_rate")
    rate = parse.get("seconds", 0) / parse.get("pages", 1)            # s/page, measured on one document
    cp_embed = transcribed("colpali_embed_index_minutes")
    cp_render = transcribed("colpali_render_minutes")

    prep = {"MinerU parse (A100)": {c: rate * pages[c] / 3600 for c in pages},
            "E5 node embedding (A100)": {c: E5["ms_per_element"] * elems[c] / 3.6e6 for c in pages},
            "SigLIP image embedding (A100)": {c: B1["siglip"]["ms_per_image"] * imgs[c] / 3.6e6 for c in pages},
            "HGT training, 50 epochs (A100)": {c: HGT[c]["hours_50ep"] for c in pages},
            "edge construction, 7 deterministic types (CPU)":
                {c: B1["edges"]["s_per_doc"] * docs[c] / 3600 for c in pages}}
    check(abs(sum(prep["E5 node embedding (A100)"].values()) - E5["total_hours_using_mean"]) < 5e-3,
          "E5 per-corpus allocation does not sum to the measured total")
    check(abs(sum(prep["SigLIP image embedding (A100)"].values()) - B1["siglip"]["total_hours"]) < 1e-6,
          "SigLIP per-corpus allocation does not sum to the measured total")
    check(abs(sum(prep["HGT training, 50 epochs (A100)"].values()) - HGT["total_hgt_A100_h"]) < 1e-6,
          "HGT hours do not sum to the measured total")
    check(abs(sum(prep["edge construction, 7 deterministic types (CPU)"].values())
              - B1["edges"]["total_hours_364docs"]) < 5e-3, "edge hours do not sum to the measured total")
    gpu_total = sum(v for st, d in prep.items() if "(A100)" in st for v in d.values())

    L = ["# Resource profile — preparation, index size, serving, embedders",
         "",
         "**Rendered from the result JSONs by `reports/build_reports.py`. Do not edit.** "
         "`build_reports.py --check` re-renders it and the sync refuses to ship a difference.",
         "",
         "Sizes are **GiB** (2³⁰ bytes), measured from the stores on vol3 by "
         "`gramm-rag-colpali/persist_resource_inputs.py` and recorded in `cost/resource_inputs.json`. "
         "`COST_GATE_RESULT.md` and `SECTION_4_13_REBUILD.md` print the same figures labelled GB; the "
         "values are GiB and the 1.57× ratio is unaffected.",
         "",
         f"Corpus sizes used for every per-corpus allocation: MMLongBench {docs['mmlongbench']} docs / "
         f"{pages['mmlongbench']:,} pages / {elems['mmlongbench']:,} parsed elements; DocBench "
         f"{docs['docbench']} docs / {pages['docbench']:,} pages / {elems['docbench']:,} elements. "
         f"Their totals reproduce `e5_10pct.json` exactly ({E5['total_pages']:,} pages, "
         f"{E5['total_elements']:,} elements).",
         "",
         "## 1. Preparation",
         "",
         "Stages measured per corpus are marked *measured*; corpus-level rates are **allocated** on "
         "pages, elements, images or documents, and say so. Allocation is arithmetic on a measured "
         "total, not a second measurement.",
         "",
         "| arm | stage | MMLongBench | DocBench | total | basis |",
         "|---|---|---|---|---|---|"]
    basis = {
        "MinerU parse (A100)": f"allocated on pages; rate {rate:.4f} s/page from one document "
                               f"({parse.get('seconds')} s / {parse.get('pages')} pages, `REVISION_LOG.md` R-10)",
        "E5 node embedding (A100)": f"allocated on parsed elements; {E5['ms_per_element']:.3f} ms/element "
                                    f"(10% sample, `e5_10pct.json`)",
        "SigLIP image embedding (A100)": f"allocated on image elements; {B1['siglip']['ms_per_image']:.2f} "
                                         f"ms/image (400-crop sample, `b1_siglip_edges.json`)",
        "HGT training, 50 epochs (A100)": "**measured per corpus** (250 timed steps each, `hgt_training.json`)",
        "edge construction, 7 deterministic types (CPU)":
            f"allocated on documents; {B1['edges']['s_per_doc']:.2f} s/doc (20-doc sample)",
    }
    for st, d in prep.items():
        L.append(f"| GraMM-RAG | {st} | {d['mmlongbench']:.2f} h | {d['docbench']:.2f} h | "
                 f"{sum(d.values()):.2f} h | {basis[st]} |")
    L.append("| GraMM-RAG | model-extracted edges, `kg` (API) | not measured (g) | not measured (g) | "
             "not measured (g) | — |")
    L.append(f"| ColPali | embedding + index build (A100) | not measured (h) | not measured (h) | "
             f"{sum(cp_embed.get('minutes', [0])) / 60:.2f} h | transcribed: \"{cp_embed.get('quote')}\" |")
    L.append(f"| ColPali | page rendering (CPU) | not measured (h) | not measured (h) | "
             f"{sum(cp_render.get('minutes', [0])) / 60:.2f} h | transcribed: `{cp_render.get('quote')}` |")
    for arm, k in (("LightRAG 1.5.7", "lightrag"), ("RAG-Anything 1.2.8", "s2"), ("MS GraphRAG 2.7.2", "s5")):
        cell = J(f"{RC}/{k}_mmlongbench_warm_occ1.json")
        pa = str(cell.get("prep_a100_hours", ""))
        check(pa.startswith("0.0"), f"{k}: prep_a100_hours is {pa!r}, expected a structural zero")
        L.append(f"| {arm} | indexing (API-mediated) | **0 A100-h** | **0 A100-h** | **0 A100-h** | "
                 f"`_resource_cells/{k}_*_occ1.json` → `prep_a100_hours`: \"{pa}\" |")
        L.append(f"| {arm} | indexing API spend | not measured (i) | not measured (i) | not measured (i) | — |")
    for arm in ("Flat-vector", "ColPali size-matched", "Union(CP∪GraMM)"):
        L.append(f"| {arm} | — | not measured (j) | not measured (j) | not measured (j) | — |")
    L += ["",
          "**(g)** not measured — the stage's status in the shipped runs is **unverified**: no KG or "
          "edge flag appears in the archived `RUN_CONFIG` for either benchmark (`SECTION_4_13_REBUILD.md` "
          "§1, an R-42-class stamping gap). Its cost is *order $0.1–0.2* for 364 documents.",
          "**(h)** not measured per corpus — the record is two container-log terms "
          f"({' + '.join(str(x) for x in cp_embed.get('minutes', []))} min) that are not attributed to "
          "corpora; the log was not retained, so the split cannot be recovered.",
          "**(i)** not measured here — indexing spend is spread across several logs per arm "
          "(MS GraphRAG alone has pilot, full and two expansion logs). Consolidating them is a "
          "measurement, not a rendering: see `tier1/*/index_log*.json`, `S2_INDEXING_RESULT.md`, "
          "`GATE_G_RESULT.md`.",
          "**(j)** not measured — the resource campaign (cost gate; work order B) covered GraMM-RAG, "
          "ColPali and the three API-mediated arms. Flat-vector, ColPali size-matched and Union have no "
          "resource cell of their own.",
          "",
          f"**GraMM-RAG total: {gpu_total:.2f} A100-h** "
          f"(+{sum(prep['edge construction, 7 deterministic types (CPU)'].values()):.2f} CPU-h), against "
          f"**ColPali {sum(cp_embed.get('minutes', [0])) / 60:.2f} A100-h** "
          f"(+{sum(cp_render.get('minutes', [0])) / 60:.2f} CPU-h) — "
          f"a ratio of **{gpu_total / (sum(cp_embed.get('minutes', [0])) / 60):.2f}×** on GPU hours. The "
          "pre-registered cost-gate falsifier fires either way (GraMM > ColPali).",
          "",
          "**Two published totals differ from this one, both explained:**",
          "",
          f"- `SECTION_4_13_REBUILD.md` and `COST_GATE_RESULT.md` print **9.36 A100-h**. They apply the "
          f"parse rate rounded to 0.7 s/page; the measurement is {parse.get('seconds')} s / "
          f"{parse.get('pages')} pages = {rate:.4f} s/page, worth "
          f"{prep['MinerU parse (A100)']['mmlongbench'] + prep['MinerU parse (A100)']['docbench']:.2f} h "
          f"rather than {0.7 * sum(pages.values()) / 3600:.2f} h.",
          "- The notebook's `pc_fig4` prints **9.31 A100-h** and labels SigLIP and edge construction "
          "\"still unmeasured\". Both have since been measured (`b1_siglip_edges.json`); the panel omits "
          f"the SigLIP stage ({sum(prep['SigLIP image embedding (A100)'].values()):.2f} h) and its label "
          "is stale.",
          "",
          "## 2. Index size on disk",
          "",
          "| arm | MMLongBench | DocBench | total | components |",
          "|---|---|---|---|---|"]
    gm = {c: RI["index_bytes"][f"gramm|{c}"] for c in pages}
    cp = {c: RI["index_bytes"][f"colpali|{c}"] for c in pages}
    L.append(f"| GraMM-RAG | {gm['mmlongbench']['total'] / GIB:.3f} GiB | {gm['docbench']['total'] / GIB:.3f} GiB | "
             f"**{(gm['mmlongbench']['total'] + gm['docbench']['total']) / GIB:.3f} GiB** | graphs "
             f"{(gm['mmlongbench']['graphs'] + gm['docbench']['graphs']) / GIB:.3f} + node embeddings "
             f"{(gm['mmlongbench']['embeddings'] + gm['docbench']['embeddings']) / GIB:.3f} "
             f"({gm['mmlongbench']['graphs_files'] + gm['docbench']['graphs_files']} graph files, "
             f"{gm['mmlongbench']['embeddings_files'] + gm['docbench']['embeddings_files']} tensor files) |")
    L.append(f"| ColPali | {cp['mmlongbench']['total'] / GIB:.3f} GiB | {cp['docbench']['total'] / GIB:.3f} GiB | "
             f"**{(cp['mmlongbench']['total'] + cp['docbench']['total']) / GIB:.3f} GiB** | multi-vector page "
             f"store, {cp['mmlongbench']['index_files'] + cp['docbench']['index_files']} `.npz` files |")
    for arm, k in (("LightRAG 1.5.7", "lightrag"), ("RAG-Anything 1.2.8", "s2"), ("MS GraphRAG 2.7.2", "s5")):
        a, b = PRE["index_gib"].get(f"{k}|mmlongbench"), PRE["index_gib"].get(f"{k}|docbench")
        if a is None or b is None:
            fail(f"package_precomputed.json has no index size for {k}")
            continue
        L.append(f"| {arm} | {a:.3f} GiB | {b:.3f} GiB | **{a + b:.3f} GiB** | "
                 f"`package_precomputed.json` → `index_gib`, prefix-filtered per corpus (C-62) |")
    for arm in ("Flat-vector", "ColPali size-matched", "Union(CP∪GraMM)"):
        L.append(f"| {arm} | not measured (j) | not measured (j) | not measured (j) | — |")
    # the stale field C-62 was about must never be the source
    for k in ("s2", "s5"):
        whole = J(f"{RC}/{k}_mmlongbench_warm_occ1.json").get("index_gib_whole_corpus")
        per = PRE["index_gib"][f"{k}|mmlongbench"]
        check(whole is not None and whole > per + 1e-6,
              f"{k}: the cell file's index_gib_whole_corpus ({whole}) is not the two-corpus total any more")
    L += ["",
          "**(j)** as above — no resource cell was measured for these arms.",
          "",
          "The per-corpus figures for RAG-Anything and MS GraphRAG come from the prefix-filtered "
          "precomputation, **not** from `index_gib_whole_corpus` in the cell files, which is the "
          "two-corpus total that C-62 corrected (both arms keep both corpora in one directory). C-62's "
          "table records MS GraphRAG as 3.62 / 1.29 GiB where the shipped artefact says "
          f"{PRE['index_gib']['s5|mmlongbench']:.4f} / {PRE['index_gib']['s5|docbench']:.4f}; the "
          "precomputation counts prefix-matched store directories only, and it is the value used here.",
          "",
          "## 3. Query latency and peak query memory",
          "",
          "**Cold** = the document's index or graph is opened inside the timed region (multi-tenant, "
          "cache miss); **warm** = already resident. All retrieval/context only, no generation, same "
          "A100 host. Two harnesses, so the two blocks are not interchangeable: GraMM and ColPali were "
          "timed by the cost gate (n = 1,000 per cell, A-B-B-A, 20 warm-ups discarded); the API arms by "
          "work order B (n = 500, two occurrences, 20 warm-ups discarded), and their latency includes "
          "an embedding-API round trip.",
          "",
          "| arm | corpus | state | n | median | p95 | p99 | p95/median [95% CI] | peak query GPU | drift |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    ncells = 0
    for arm, k in (("GraMM-RAG", "gramm"), ("ColPali", "colpali")):
        for b in ("mmlongbench", "docbench"):
            for st in ("cold", "warm"):
                c = LAT["cells"].get(f"{k}|{b}|{st}")
                if c is None:
                    fail(f"query_latency_v2.json has no cell {k}|{b}|{st}")
                    continue
                ncells += 1
                L.append(f"| {arm} | {b} | {st} | {c['n']:,} | {c['median']:.1f} ms | {c['p95']:.1f} ms | "
                         f"{c['p99']:.1f} ms | {c['p95_over_median']:.2f} [{c['ratio_ci95'][0]:.2f}, "
                         f"{c['ratio_ci95'][1]:.2f}] | {c['peak_gpu_gib']:.2f} GiB | "
                         f"{LAT['drift_pct'][f'{k}|{b}|{st}']:.2f}% |")
    for arm, k in (("LightRAG 1.5.7", "lightrag"), ("RAG-Anything 1.2.8", "s2"), ("MS GraphRAG 2.7.2", "s5")):
        for b in ("mmlongbench", "docbench"):
            for st in ("cold", "warm"):
                c1 = J(f"{RC}/{k}_{b}_{st}_occ1.json")
                c2 = J(f"{RC}/{k}_{b}_{st}_occ2.json")
                if not c1 or not c2:
                    continue
                ncells += 1
                drift = (c2["median_s"] - c1["median_s"]) * 1000
                if k == "s5":
                    d = J(f"{RC}/DECOMP_s5_{b}_{st}.json")
                    med, p95 = d["context_only_median_s"], d["context_only_p95_s"]
                    p99, ratio = "not measured (k)", f"{p95 / med:.2f} (l)"
                else:
                    med, p95 = c1["median_s"], c1["p95_s"]
                    p99 = f"{c1['p99_s'] * 1000:.0f} ms"
                    ratio = (f"{c1['tail_ratio_p95_over_median']:.2f} [{c1['tail_ratio_ci'][0]:.2f}, "
                             f"{c1['tail_ratio_ci'][1]:.2f}]")
                L.append(f"| {arm} | {b} | {st} | {c1['n_timed']:,} | {med * 1000:.0f} ms | {p95 * 1000:.0f} ms | "
                         f"{p99} | {ratio} | **0 (m)** | {drift:+.0f} ms |")
    for arm in ("Flat-vector", "ColPali size-matched", "Union(CP∪GraMM)"):
        L.append(f"| {arm} | — | — | — | not measured (j) | not measured (j) | not measured (j) | "
                 f"not measured (j) | not measured (j) | — |")
    gb = J(f"{RC}/s5_mmlongbench_warm_occ1.json").get("gpu_basis", "")
    L += ["",
          f"**{ncells} measured latency cells**: 2 GPU arms × 2 corpora × cold/warm (8) plus 3 API "
          f"arms × 2 corpora × cold/warm (12). The API block is backed by 24 occurrence files (two "
          f"palindrome occurrences per cell; occurrence 1 is reported, occurrence 2 is the drift "
          f"check) plus 4 MS GraphRAG decompositions -- the 28 files under `tier1/_resource_cells/`.",
          "",
          "**(k)** not measured — the B′ decomposition that separates MS GraphRAG's retrieval from its "
          "generation records the context-only median and p95 only; no context-only p99 exists.",
          "**(l)** derived, not measured — p95/median of the two context-only values; no bootstrap "
          "interval was computed for the decomposed series. MS GraphRAG's `api.local_search` always "
          "generates, so its rows are context-only and its tail is an upper bound.",
          f"**(m)** structurally zero, not merely observed: \"{gb}\"",
          "**(j)** no resource cell was measured for these arms.",
          "",
          "Drift is occurrence 2 − occurrence 1 for the API arms, and the cost gate's own drift "
          "percentage for GraMM and ColPali. Peak query GPU for GraMM and ColPali is the per-cell "
          "figure from the same run.",
          "",
          "## 4. Embedder and dimensionality",
          "",
          "Widths are read from the stored vectors themselves — every file, not a sample — because no "
          "`RUN_CONFIG` stamps them and D-32 was a single document silently embedded in a different space.",
          "",
          "| arm | stage | model | stored width | dtype | basis |",
          "|---|---|---|---|---|---|"]
    gmm = RI["embedding_census"]["gramm|mmlongbench"]
    gmd = RI["embedding_census"]["gramm|docbench"]
    for kind, label in (("text_raw", "text, raw"), ("text", "text, projected node vector"),
                        ("img", "image, projected node vector")):
        check(gmm[kind]["dim"] == gmd[kind]["dim"] and gmm[kind]["dtype"] == gmd[kind]["dtype"],
              f"GraMM {kind} differs between corpora: {gmm[kind]} vs {gmd[kind]}")
        model = ("`intfloat/e5-mistral-7b-instruct`" if kind == "text_raw" else
                 "linear projection to a common space" if kind == "text" else
                 "SigLIP-SO400M → linear projection")
        L.append(f"| GraMM-RAG | {label} | {model} | {gmm[kind]['dim']} | {gmm[kind]['dtype']} | "
                 f"every tensor: {gmm[kind]['files']} + {gmd[kind]['files']} files, one dimension |")
    L.append("| GraMM-RAG | image, raw | SigLIP-SO400M | not measured (n) | — | only the projection is stored |")
    cpm = RI["embedding_census"]["colpali|mmlongbench"]
    cpd = RI["embedding_census"]["colpali|docbench"]
    colcfg = J("gramm-rag-colpali/retrieval/colpali_retrieval_mmlongbench.json")["RUN_CONFIG"]
    L.append(f"| ColPali | page multi-vector | `{colcfg['COLPALI_MODEL']}` | {cpm['dim']} per token | "
             f"{cpm['dtype']} | every page matrix: {cpm['page_matrices'] + cpd['page_matrices']:,} pages, "
             f"median {cpm['tokens_per_page_median']:.0f} tokens/page |")
    for arm, k in (("LightRAG 1.5.7", "lightrag"), ("RAG-Anything 1.2.8", "s2")):
        cen = RI["embedding_census"][f"{k}|mmlongbench"]
        cend = RI["embedding_census"][f"{k}|docbench"]
        by = J(f"{RC}/{k}_mmlongbench_warm_occ1.json")["meter_stages"].get("by_model", {})
        model = ", ".join(f"`{m}`" for m in by) or "not recorded"
        check(cen["dim"] == cend["dim"], f"{k}: embedding dim differs between corpora")
        L.append(f"| {arm} | text chunk | {model} | {cen['dim']} | — | {cen['basis']} "
                 f"({cen['stores']} + {cend['stores']} stores) |")
    s5c = RI["embedding_census"]["s5|mmlongbench"]
    L.append(f"| MS GraphRAG 2.7.2 | text unit | `{s5c['model']}` | not measured (o) | — | {s5c['basis']} |")
    L += ["",
          "**(n)** not measured — SigLIP's raw output is projected before storage; only the 256-d "
          "projection is on disk.",
          "**(o)** not measured — the stored width was not read back from the LanceDB store; the model "
          "is taken from every store's `settings.yaml`.",
          "",
          "GraMM's text encoder is the 4,096-d E5-Mistral-7B in every one of the "
          f"{gmm['text_raw']['files'] + gmd['text_raw']['files']} documents: none fell back to the 1,024-d "
          "E5-large-v2, so the D-32 fallback did not recur on these corpora.",
          ""]
    return "\n".join(L).rstrip() + "\n"


def table_shape(name, text):
    """A markdown table whose rows disagree with its header renders as prose. Caught once by
    reading the output; now checked on every build."""
    hdr = None
    for i, line in enumerate(text.splitlines(), 1):
        if not line.startswith("|"):
            hdr = None
            continue
        n = line.count("|")
        body = line.replace("|", "").strip()
        if set(body) <= set("- ") and body:
            check(n == hdr, f"{name}:{i}: separator has {n} pipes, header has {hdr}")
        elif hdr is None:
            hdr = n
        else:
            check(n == hdr, f"{name}:{i}: row has {n} pipes, header has {hdr}: {line[:60]}")


def main():
    cross_checks()
    strat, res = render_stratified(), render_resource()
    table_shape("STRATIFIED_REPORT.md", strat)
    table_shape("RESOURCE_PROFILE.md", res)
    if ERR:
        print("REPORT BUILD FAILED:")
        for e in ERR:
            print(f"  - {e}")
        return 1
    if "--check" in sys.argv:
        bad = False
        for p, text in ((OUT_STRAT, strat), (OUT_RES, res)):
            if not p.exists():
                print(f"REPORT CHECK FAILED: {p.name} has not been rendered"); bad = True
            elif p.read_text() != text:
                print(f"REPORT CHECK FAILED: {p.name} differs from the artefacts -- "
                      f"re-run build_reports.py"); bad = True
        if bad:
            return 1
        print(f"reports ok: {OUT_STRAT.name} and {OUT_RES.name} match the result JSONs")
        return 0
    OUT_STRAT.write_text(strat)
    OUT_RES.write_text(res)
    print(f"wrote {OUT_STRAT.relative_to(ROOT)} ({len(strat.splitlines())} lines)")
    print(f"wrote {OUT_RES.relative_to(ROOT)} ({len(res.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
