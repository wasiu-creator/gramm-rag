"""
Gate G projection: 5-document pilot -> full-corpus indexing wall-clock and API cost.

Decision rules were fixed in GATE_G_GRAPH_BRACKET.md BEFORE any install:
  authorised for Tier 1 iff  (API <= $10)  AND  (wall-clock lands Tier 1 before Sep 18).

Estimators, and why two of them:
  * per-document model applied to all 364 real documents, then summed. Correct weighting:
    corpus char mass sits in large documents, whose per-char rate is measurably LOWER.
  * pooled rate (pilot total / pilot chars x corpus chars), reported as a cross-check.
    If the two disagree materially the projection is not trustworthy and the gate should
    be decided on the more conservative of the pair.

The multimodal arm needs TWO cost drivers, not one: text-side cost scales with characters,
vision-side cost scales with IMAGE COUNT, and images also inflate the text side because each
generated description becomes extraction input. A chars-only model fitted to a multimodal arm
understates it by whatever share of the bill the images carry.
"""
import json, sys
import numpy as np
from pathlib import Path

GB = Path(__file__).resolve().parent
CORPUS = json.loads((GB / "data/corpus_sizes.json").read_text())
C_CHARS = sum(d["chars"] for d in CORPUS)
C_IMGS = sum(d["n_img"] for d in CORPUS)
C_DOCS, C_PAGES = len(CORPUS), sum(d["pages"] for d in CORPUS)
GATE_USD, GRAMM_ANCHOR_S_PER_DOC = 10.0, 37.6


def powerlaw(x, y):
    b, loga = np.polyfit(np.log(x), np.log(y), 1)
    return np.exp(loga), b


def project_text_only(log):
    rows = [r for r in log["docs"] if not r["error"]]
    x = np.array([r["text_chars"] for r in rows], float)
    s = np.array([r["seconds"] for r in rows], float)
    c = np.array([r["cost_usd"] for r in rows], float)
    a_s, b_s = powerlaw(x, s); a_c, b_c = powerlaw(x, c)
    tc = np.clip(np.array([d["chars"] for d in CORPUS], float), 1, None)
    return {"sec_fit": float((a_s * tc ** b_s).sum()), "usd_fit": float((a_c * tc ** b_c).sum()),
            "sec_pool": float(s.sum() / x.sum() * tc.sum()),
            "usd_pool": float(c.sum() / x.sum() * tc.sum()),
            "b_sec": b_s, "b_usd": b_c, "n": len(rows), "model": "power law in chars"}


def project_multimodal(log):
    """Non-negative least squares on two drivers: cost ~ k_char*chars + k_img*images."""
    rows = [r for r in log["docs"] if not r["error"]]
    ch = np.array([r["text_chars"] for r in rows], float)
    im = np.array([r["n_images"] for r in rows], float)
    s = np.array([r["seconds"] for r in rows], float)
    ct = np.array([r["text_side"]["cost_usd"] for r in rows], float)
    cv = np.array([r["vision_side"]["cost_usd"] for r in rows], float)
    A = np.column_stack([ch, im])
    from scipy.optimize import nnls
    k_txt, _ = nnls(A, ct); k_vis, _ = nnls(A, cv); k_sec, _ = nnls(A, s)
    usd_txt = k_txt[0] * C_CHARS + k_txt[1] * C_IMGS
    usd_vis = k_vis[0] * C_CHARS + k_vis[1] * C_IMGS
    sec = k_sec[0] * C_CHARS + k_sec[1] * C_IMGS
    # Pooled cross-check: per-image vision rate, per-char text rate, applied independently.
    usd_vis_pool = cv.sum() / max(im.sum(), 1) * C_IMGS
    usd_txt_pool = ct.sum() / ch.sum() * C_CHARS
    return {"sec_fit": float(sec), "usd_fit": float(usd_txt + usd_vis),
            "usd_text": float(usd_txt), "usd_vision": float(usd_vis),
            "sec_pool": float(s.sum() / ch.sum() * C_CHARS),
            "usd_pool": float(usd_txt_pool + usd_vis_pool),
            "usd_per_image": float(cv.sum() / max(im.sum(), 1)),
            "pilot_images": int(im.sum()), "n": len(rows),
            "model": "NNLS on (chars, images)"}


def report(name, p, parallel):
    seq_h, usd = p["sec_fit"] / 3600, p["usd_fit"]
    par_h = seq_h / parallel
    worst_usd = max(usd, p["usd_pool"]); worst_h = max(seq_h, p["sec_pool"] / 3600)
    print(f"\n=== {name} ===")
    print(f"  model: {p['model']}  (n={p['n']} pilot documents)")
    if "usd_vision" in p:
        print(f"  cost split: text ${p['usd_text']:.2f} + vision ${p['usd_vision']:.2f}"
              f"   (${p['usd_per_image']:.5f}/image x {C_IMGS:,} corpus images)")
    print(f"  {'fitted':<14}{seq_h:>8.2f} h seq {par_h:>8.2f} h par   ${usd:>9.2f}")
    print(f"  {'pooled (xchk)':<14}{p['sec_pool']/3600:>8.2f} h seq "
          f"{p['sec_pool']/3600/parallel:>8.2f} h par   ${p['usd_pool']:>9.2f}")
    ratio = worst_usd / GATE_USD
    print(f"  GATE (<= ${GATE_USD:.0f}): {'PASS' if worst_usd <= GATE_USD else 'FAIL'}"
          f"  -- conservative estimate ${worst_usd:.2f} = {ratio:.1f}x the ceiling")
    spd = p["sec_fit"] / C_DOCS
    print(f"  anchor check: {spd:.1f} s/doc vs GraMM {GRAMM_ANCHOR_S_PER_DOC} s/doc "
          f"({spd/GRAMM_ANCHOR_S_PER_DOC:.1f}x) -- "
          f"{'plausible' if spd > GRAMM_ANCHOR_S_PER_DOC else 'IMPLAUSIBLE, below anchor: re-check'}")
    return {"name": name, "seq_h": seq_h, "par_h": par_h, "usd": usd,
            "conservative_usd": worst_usd, "conservative_h": worst_h,
            "gate_pass": bool(worst_usd <= GATE_USD), **p}


def scope_table(log, kind):
    """Per-corpus split -- the obvious scoping lever if the full corpus fails the gate."""
    rows = [r for r in log["docs"] if not r["error"]]
    ch = np.array([r["text_chars"] for r in rows], float)
    if kind == "text":
        c = np.array([r["cost_usd"] for r in rows], float)
        a_c, b_c = powerlaw(ch, c)
        f = lambda sub: float(sum(a_c * max(d["chars"], 1) ** b_c for d in sub))
    else:
        im = np.array([r["n_images"] for r in rows], float)
        ct = np.array([r["text_side"]["cost_usd"] for r in rows], float)
        cv = np.array([r["vision_side"]["cost_usd"] for r in rows], float)
        from scipy.optimize import nnls
        A = np.column_stack([ch, im])
        kt, _ = nnls(A, ct); kv, _ = nnls(A, cv)
        f = lambda sub: float(sum((kt[0] + kv[0]) * d["chars"] + (kt[1] + kv[1]) * d["n_img"]
                                  for d in sub))
    print(f"  {'scope':<22}{'docs':>5}{'pages':>8}{'images':>8}{'cost $':>10}  gate")
    out = {}
    for nm, sub in [("MMLongBench only", [d for d in CORPUS if d["bench"] == "mmlongbench"]),
                    ("DocBench only", [d for d in CORPUS if d["bench"] == "docbench"]),
                    ("BOTH (full corpus)", CORPUS)]:
        v = f(sub); out[nm] = v
        print(f"  {nm:<22}{len(sub):>5}{sum(d['pages'] for d in sub):>8,}"
              f"{sum(d['n_img'] for d in sub):>8,}{v:>10.2f}  "
              f"{'PASS' if v <= GATE_USD else 'FAIL'}")
    return out


if __name__ == "__main__":
    print(f"corpus: {C_DOCS} docs / {C_PAGES:,} pages / {C_CHARS:,} chars / {C_IMGS:,} images")
    summary = {}
    lp = GB / "pilot/lightrag_current/pilot_log.json"
    if lp.exists():
        log = json.loads(lp.read_text())
        r = report("LightRAG 1.5.7 (text-only)", project_text_only(log),
                   log["run_config"]["MAX_PARALLEL_INSERT"])
        print()
        r["scope"] = scope_table(log, "text"); summary["lightrag_current"] = r
    rp = GB / "pilot/raganything_legacy/pilot_log.json"
    if rp.exists():
        log = json.loads(rp.read_text())
        r = report("RAG-Anything 1.2.8 + LightRAG 1.4.9.2 (multimodal)",
                   project_multimodal(log), 2)
        print()
        r["scope"] = scope_table(log, "mm"); summary["raganything_legacy"] = r
    else:
        print("\n(raganything pilot log not present yet)")
    (GB / "pilot/gate_g_projection.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nwrote {GB}/pilot/gate_g_projection.json")
