#!/usr/bin/env python3
"""Gate H §1 cell-level table: arm x benchmark x metric x stratum, every cell.

Every cell carries a value or an explicit "not measured, because ...". Localisation is
undefined on unanswerable questions by construction (C-8/C-9), and that is stated rather
than left blank. Paired bootstrap contrasts vs GraMM-RAG on BOTH corpora -- the MMLongBench
side was the gap on which the "beats the canonical implementation" claim rests.
"""
import json, random, sys
from pathlib import Path

CP = Path("/work/gramm-rag-colpali"); D3 = Path("/work/gramm-rag-3benchmarks/data")
sys.path.insert(0, "/work/gramm-rag-mmlongbench")
from src.evaluation.metrics import anls_score
sys.path.insert(0, "/work/gramm-rag-docbench")
from src.evaluation.metrics import evidence_page_hit, _quote_tokens
B, SEED = 4000, 42
ARMS = [("Flat-vector", None), ("GraMM-RAG", None), ("LightRAG 1.5.7", "lightrag"),
        ("RAG-Anything 1.2.8", "s2"), ("MS GraphRAG 2.7.2", "s5"), ("ColPali", "colpali"),
        ("ColPali size-matched", "colpali_matched"), ("Union(CP∪GraMM)", "union"),
        # C-77(ii): the page-matched control. The archived k=2 Flat-vector stays in the table so
        # the page-budget effect is visible rather than swapped in silently.
        ("Flat-vector (k=3)", "flatk3")]
DEC = __import__("re").compile(
    r"not\s*answerable|unanswerable|insufficient|cannot\s+(?:be\s+answered|provide|determine|answer)"
    r"|unable\s+to|no\s+(?:answer|information|mention)"
    r"|(?:do|does|did)\s+not\s+(?:include|contain|mention|provide|specify|have|appear|reference)"
    r"|(?:is|are)\s+not\s+(?:stated|provided|mentioned|specified|available|found|included|contained)"
    r"|not\s+(?:stated|provided|mentioned|specified|available|found)", __import__("re").I)


def load(bench):
    b = json.loads((D3 / bench / f"baseline_vector_{'mmlb' if bench=='mmlongbench' else bench}.json").read_text())
    g = json.loads((D3 / bench / f"gramm_{'mmlongbench' if bench=='mmlongbench' else 'docbench'}_s42.json").read_text())
    preds, pages = {}, {}
    preds["Flat-vector"], pages["Flat-vector"] = b["predictions"], b["context_pages"]
    preds["GraMM-RAG"], pages["GraMM-RAG"] = g["predictions"], g["context_pages"]
    for lab, k in ARMS:
        if k is None: continue
        f = CP / "generation" / f"{k}_gen_{bench}.json"
        if f.exists():
            d = json.loads(f.read_text()); preds[lab], pages[lab] = d["predictions"], d["context_pages"]
    return b, g, preds, pages


def quality(bench, b, g, preds):
    """Per-question quality score for every arm."""
    if bench == "mmlongbench":
        golds = b["golds"]
        return {a: [anls_score(p, t) for p, t in zip(P, golds)] for a, P in preds.items()}
    out = {"Flat-vector": b["judge"]["judge_scores"], "GraMM-RAG": g["judge"]["judge_scores"]}
    for lab, k in ARMS:
        if k is None: continue
        f = CP / "scores" / f"{k}_scores_docbench.json"
        if f.exists(): out[lab] = json.loads(f.read_text())["judge_scores"]
    return out


def localisation(bench, b, pages):
    """Per-question localisation hit, and the indices on which it is defined."""
    if bench == "mmlongbench":
        gold, an = b["gold_pages"], b["is_answerable"]
        idx = [i for i in range(len(gold)) if an[i] and gold[i]]
        return {a: [1 if any(x in P[i] for x in gold[i]) else 0 for i in idx] for a, P in pages.items()}, idx
    qm = json.loads((D3 / "docbench" / "qmeta_docbench.json").read_text())["rows"]
    ev, an = b["evidences"], b["is_answerable"]
    pdir = Path("/work/gramm-rag-docbench/cache/parsed"); cache = {}
    def ptext(doc, pg):
        if doc not in cache:
            f = pdir / f"{doc}.json"
            d = json.loads(f.read_text()) if f.exists() else {"elements": []}
            byp = {}
            for e in d.get("elements", []):
                try: k = int(e.get("page_no", -1))
                except (TypeError, ValueError): continue
                byp.setdefault(k, []).append((int(e.get("reading_order", 0) or 0), e.get("text", "") or ""))
            cache[doc] = {k: " ".join(t for _, t in sorted(v)) for k, v in byp.items()}
        return cache[doc].get(pg, "")
    idx = [i for i in range(len(ev)) if _quote_tokens(ev[i]) and an[i]]
    return ({a: [1 if evidence_page_hit([ptext(qm[i]["doc_id"], p) for p in P[i]][:5], ev[i], 0.6) else 0
                 for i in idx] for a, P in pages.items()}, idx)


def ci(v):
    v = sorted(v); return v[int(.025 * len(v))], v[int(.975 * len(v))]


def paired(x, y, n):
    rng = random.Random(SEED)
    d = [], 
    draws = []
    for _ in range(B):
        s = [rng.randrange(n) for _ in range(n)]
        draws.append(sum(x[k] for k in s) / n - sum(y[k] for k in s) / n)
    lo, hi = ci(draws)
    return sum(x) / n - sum(y) / n, lo, hi, (lo > 0 or hi < 0)


def main():
    out = {}
    for bench in ("mmlongbench", "docbench"):
        b, g, preds, pages = load(bench)
        an = b["is_answerable"]; n = len(an)
        Q = quality(bench, b, g, preds)
        L, lidx = localisation(bench, b, pages)
        A = [i for i in range(n) if an[i]]; U = [i for i in range(n) if not an[i]]
        qname = "ANLS" if bench == "mmlongbench" else "judge accuracy"
        lname = "ctx-hit" if bench == "mmlongbench" else "quote-recall@5"
        rows = {}
        for lab in Q:
            q = Q[lab]
            dec_a = sum(1 for i in A if DEC.search(str(preds[lab][i]))) / len(A)
            dec_u = sum(1 for i in U if DEC.search(str(preds[lab][i]))) / len(U)
            rows[lab] = {
                f"{qname} pooled": sum(q) / n,
                f"{qname} answerable": sum(q[i] for i in A) / len(A),
                f"{qname} unanswerable": sum(q[i] for i in U) / len(U),
                f"{lname} (n={len(lidx)})": sum(L[lab]) / len(L[lab]),
                f"{lname} pooled/unanswerable": "not measured — undefined on unanswerable "
                                                "questions and on answerable ones lacking gold "
                                                "evidence (C-8/C-9); denominator is the "
                                                f"{len(lidx)} scoreable questions",
                "decline balAcc": 0.5 * (dec_u + (1 - dec_a)),
                "false-decline (answerable)": dec_a,
                "decline on unanswerable": dec_u,
            }
        # paired contrasts vs GraMM-RAG, both metric families, both strata
        con = {}
        for lab in Q:
            if lab == "GraMM-RAG": continue
            c = {}
            for sname, idx in (("pooled", list(range(n))), ("answerable", A), ("unanswerable", U)):
                x = [Q[lab][i] for i in idx]; y = [Q["GraMM-RAG"][i] for i in idx]
                d, lo, hi, sig = paired(x, y, len(idx))
                c[f"{qname} {sname}"] = dict(delta=round(d, 4), ci=[round(lo, 4), round(hi, 4)], sig=sig)
            d, lo, hi, sig = paired(L[lab], L["GraMM-RAG"], len(lidx))
            c[lname] = dict(delta=round(d, 4), ci=[round(lo, 4), round(hi, 4)], sig=sig)
            con[lab] = c
        # eta = delta quality / delta localisation vs the flat-vector control.
        # Reported against BOTH controls: the archived k=2 arm for continuity with everything
        # published before C-77, and the page-matched k=3 arm, which is the one to quote --
        # eta's denominator is a localisation difference, and a page-budget gap inflates it.
        eta = {}
        for lab in Q:
            if lab == "Flat-vector": continue
            dq = sum(Q[lab]) / n - sum(Q["Flat-vector"]) / n
            dl = sum(L[lab]) / len(L[lab]) - sum(L["Flat-vector"]) / len(L[lab])
            eta[lab] = dict(d_quality=round(dq, 4), d_localisation=round(dl, 4),
                            eta=round(dq / dl, 4) if abs(dl) > 1e-9 else None)
        eta3 = {}
        if "Flat-vector (k=3)" in Q:
            for lab in Q:
                if lab == "Flat-vector (k=3)": continue
                dq = sum(Q[lab]) / n - sum(Q["Flat-vector (k=3)"]) / n
                dl = sum(L[lab]) / len(lidx) - sum(L["Flat-vector (k=3)"]) / len(lidx)
                eta3[lab] = dict(d_quality=round(dq, 4), d_localisation=round(dl, 4),
                                 eta=round(dq / dl, 4) if abs(dl) > 1e-9 else None)
        out[bench] = dict(n_questions=n, n_answerable=len(A), n_unanswerable=len(U),
                          n_localisation_scoreable=len(lidx), rows=rows,
                          contrasts_vs_gramm=con, eta_vs_flat=eta, eta_vs_flatk3=eta3)
    (CP / "scores" / "cell_table.json").write_text(json.dumps(out, indent=1))
    for bench in out:
        o = out[bench]
        print(f"\n=== {bench}  (n={o['n_questions']}, answerable {o['n_answerable']}, "
              f"unanswerable {o['n_unanswerable']}, localisation scoreable {o['n_localisation_scoreable']}) ===")
        print("  contrasts vs GraMM-RAG  (delta [95% CI], * = CI excludes zero)")
        for lab, c in o["contrasts_vs_gramm"].items():
            bits = []
            for k, v in c.items():
                bits.append(f"{k.split()[-1] if 'pooled' in k or 'answer' in k else k}: "
                            f"{v['delta']:+.4f}[{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]{'*' if v['sig'] else ' '}")
            print(f"    {lab:22s} " + "  ".join(bits))
    print("\n  wrote scores/cell_table.json")


if __name__ == "__main__":
    main()
