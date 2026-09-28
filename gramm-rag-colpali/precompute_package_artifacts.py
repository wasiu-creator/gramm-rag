#!/usr/bin/env python3
"""H -- precompute every render-time value that needs data the handoff package EXCLUDES.

The cold-start test (regenerate the notebook from the package alone, vol3 unmounted) ran with
0 errors and 3 cells whose OUTPUT differed from vol3. Two were silent fallbacks to plausible
numbers:
  * index size -- walked tier1/*/stores (excluded as bulk data); an empty glob summed to 0.00 GiB
  * LightRAG DocBench quote-recall -- rebuilt page texts from the MinerU parse (excluded), and
    `f.read_text() if f.exists() else {'elements': []}` swapped each missing parse for an empty
    document: 0.5068 became 0.0000 and a PRE-REGISTERED clause flipped FALSIFIED -> HELD at
    p = 4.6e-100.
Neither raised. So: compute here, where the bulk data exists, and ship the small result. The
notebook reads these and RAISES if one is absent -- it never recomputes over missing inputs.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
T1 = ROOT / "gramm-rag-graphbracket/tier1"
OUT = ROOT / "gramm-rag-colpali/scores/package_precomputed.json"

# ---- 1. index size per arm x corpus, PREFIX-FILTERED ----
def gib(d, pre):
    return sum(f.stat().st_size for s in d.glob(f"{pre}*") if s.is_dir()
               for f in s.rglob("*") if f.is_file()) / 2**30
idx = {}
for b in ("mmlongbench", "docbench"):
    idx[f"lightrag|{b}"] = gib(T1 / f"lightrag_{b}/stores", "")
    idx[f"s2|{b}"] = gib(T1 / "s2_raganything/stores", f"{b}__")
    idx[f"s5|{b}"] = gib(T1 / "s5_graphrag/stores", f"{b}__")
for k, v in idx.items():
    assert v > 0, f"index size {k} measured as 0 -- stores missing HERE, refusing to ship a 0"

# ---- 2. LightRAG DocBench per-question quote hits, from the parse ----
sys.path.insert(0, str(ROOT / "gramm-rag-docbench"))
from src.evaluation.metrics import evidence_page_hit        # noqa: E402
DB = ROOT / "gramm-rag-3benchmarks/data/docbench"
dbase = json.loads((DB / "baseline_vector_docbench.json").read_text())
dqm = json.loads((DB / "qmeta_docbench.json").read_text())["rows"]
dgen = json.loads((ROOT / "gramm-rag-colpali/generation/lightrag_gen_docbench.json").read_text())
pdir = ROOT / "gramm-rag-docbench/cache/parsed"
# EXACT mirror of the notebook's _pt (section 16): single filename, elements SORTED BY
# reading_order within a page. A first version appended in file order and resolved several
# filename variants; it produced 0.494 against the notebook's 0.5068 and was not shipped. A
# precompute that disagrees with what it replaces would silently change the vol3 result too.
_c, missing = {}, set()
def _pt(doc_id, pg):
    if doc_id not in _c:
        f = pdir / f"{doc_id}.json"
        if not f.exists():
            missing.add(doc_id); _c[doc_id] = {}
        else:
            d = json.loads(f.read_text()); byp = {}
            for e in d.get("elements", []):
                try: k = int(e.get("page_no", -1))
                except (TypeError, ValueError): continue
                byp.setdefault(k, []).append((int(e.get("reading_order", 0) or 0),
                                              e.get("text", "") or ""))
            _c[doc_id] = {k: " ".join(x for _, x in sorted(v)) for k, v in byp.items()}
    return _c[doc_id].get(pg, "")
from src.evaluation.metrics import _quote_tokens            # noqa: E402
n = dbase["n_questions"]; dev = dbase["evidences"]; dan = dbase["is_answerable"]
lr_pt = [[_pt(dqm[i]["doc_id"], pg) for pg in dgen["context_pages"][i]] for i in range(n)]
assert not missing, f"{len(missing)} DocBench parses missing on vol3: {sorted(missing)[:3]}"
# per-question hit for EVERY question index, so the consumer applies its own denominator
hits = [1 if evidence_page_hit(lr_pt[i][:5], dev[i], .6) else 0 for i in range(n)]
sc = [i for i in range(n) if _quote_tokens(dev[i]) and dan[i]]     # the notebook's denominator
qr = sum(hits[i] for i in sc) / len(sc)
assert abs(qr - 0.5068) < 5e-4, (f"precompute gives {qr:.4f}, notebook reports 0.5068 -- "
                                 f"refusing to ship a value that changes the vol3 result")

# --- DocBench corpus content is EXCLUDED from the public package (an authors' decision on
# redistribution: page texts, questions, gold answers and evidence are DocBench's, not ours). The
# notebook computed three things directly from that content, so each is precomputed here against
# the same canonical predicate and shipped as a derived value. Absence raises downstream.
dgramm = json.loads((DB / "gramm_docbench_s42.json").read_text())
gramm_hits = [1 if evidence_page_hit(dgramm["page_texts"][i][:5], dev[i], .6) else 0 for i in range(n)]
flat_hits = [1 if evidence_page_hit(dbase["page_texts"][i][:5], dev[i], .6) else 0 for i in range(n)]
_qg = sum(gramm_hits[i] for i in sc) / len(sc)
_qf = sum(flat_hits[i] for i in sc) / len(sc)
assert abs(_qg - 0.4534) < 5e-4 and abs(_qf - 0.3959) < 5e-4, (
    f"precomputed quote-recall {_qg:.4f}/{_qf:.4f} against the published 0.4534/0.3959 -- "
    f"refusing to ship values that change the result")
_STRUCT = re.compile(r"does not (have|contain|include)|no such|"
                     r"not (have|contain) (page|section|appendix|chapter|table|figure)", re.I)
struct_unans = [i for i in range(n) if not dan[i] and _STRUCT.search(dbase["golds"][i] or "")]

out = dict(
    NOTE=("Precomputed on vol3, where bulk data exists; shipped so the notebook never recomputes "
          "over inputs the package excludes. Read by phaseC_evidence; absence RAISES."),
    index_gib={k: round(v, 4) for k, v in idx.items()},
    lightrag_docbench_quote_hits=hits,
    gramm_docbench_quote_hits=gramm_hits,
    flat_docbench_quote_hits=flat_hits,
    docbench_structural_unanswerable_idx=struct_unans,
    docbench_quote_scoreable_idx=sc,
    lightrag_docbench_quote_recall_scored=round(qr, 4),
    quote_recall_denominator=len(sc))
OUT.write_text(json.dumps(out, indent=1))
print(f"wrote {OUT.relative_to(ROOT)}")
print("index GiB:", {k: round(v, 2) for k, v in idx.items()})
print(f"LightRAG DocBench quote-recall (precomputed): {out['lightrag_docbench_quote_recall_scored']}"
      f"  over n={len(sc)}  (vol3 notebook reports 0.5068)")
