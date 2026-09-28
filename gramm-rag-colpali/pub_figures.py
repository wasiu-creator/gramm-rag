#!/usr/bin/env python3
"""Publication figures at IEEE print size -- RE-LAID-OUT, never scaled down.

The notebook figures are 12-15 in wide with 9-10 pt type; shrunk to a 3.5 in column that type prints
at ~2.5 pt. So every figure is rebuilt at its final width with print-sized type: 8 pt base, 7 pt ticks
and legends, a Times-compatible serif (STIX), fonts embedded as TrueType (IEEE rejects Type 3).

Widths: 3.5 in single column; 7.16 in double column only where a figure cannot fit a column.
Outputs: vector PDF (the submission file) + 600-dpi PNG (preview). Data come from the same JSONs as
phaseC_evidence.ipynb, with the same derivations, so a figure cannot drift from the notebook's tables.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

W = Path("/work")
SC, COST = W / "gramm-rag-colpali/scores", W / "gramm-rag-colpali/cost"
OUT = W / "GraphRAG_baseline/figures/publication"; OUT.mkdir(parents=True, exist_ok=True)
COL1, COL2 = 3.5, 7.16
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "grid.linewidth": 0.4, "grid.color": "#e2e1dc", "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# the validated palette (dataviz validator, both modes): colour follows the ARM, never its rank
ARM = {"GraMM-RAG": "#2a78d6", "LightRAG 1.5.7": "#eb6834", "RAG-Anything 1.2.8": "#1baf7a",
       "MS GraphRAG 2.7.2": "#eda100", "ColPali": "#e87ba4", "ColPali size-matched": "#008300",
       "Union(CP∪GraMM)": "#4a3aa7", "Flat-vector": "#8a8984",
       # Both controls share the validated neutral gray -- they are the same system at two context
       # budgets, not two systems. They are told apart by MARKER (secondary encoding), so the
       # palette keeps passing the validator with no ninth hue.
       "Flat-vector (k=3)": "#8a8984"}
MARK = {"GraMM-RAG": "o", "Flat-vector": "s", "LightRAG 1.5.7": "^", "RAG-Anything 1.2.8": "D",
        "MS GraphRAG 2.7.2": "v", "ColPali": "P", "ColPali size-matched": "X", "Union(CP∪GraMM)": "*",
        "Flat-vector (k=3)": "s"}
SHORT = {"GraMM-RAG": "GraMM-RAG", "LightRAG 1.5.7": "LightRAG", "RAG-Anything 1.2.8": "RAG-Anything",
         "MS GraphRAG 2.7.2": "MS GraphRAG", "ColPali": "ColPali", "ColPali size-matched": "ColPali (size-m.)",
         "Union(CP∪GraMM)": "Union", "Flat-vector": "Flat-vector (k=2)",
         "Flat-vector (k=3)": "Flat-vector (k=3, control)"}
INK, INK2 = "#1a1a19", "#52514e"
J = lambda p: json.loads(Path(p).read_text())
written = []

def ring_labels(ax, rs, theta_deg, fmt):
    """Ring values as free text ABOVE the data. Tick labels cannot do this: an axis draws all its
    tick labels as one artist beneath the plotted lines, so a knockout behind them is invisible --
    tried first, and the lines still ran through '.25' and '.50' on the DocBench radar."""
    ax.set_yticklabels([])
    for r in rs:
        ax.text(np.deg2rad(theta_deg), r, fmt(r), fontsize=6.5, color=INK2, ha="center", va="center", zorder=20,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.92))
def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    # Measure the WRITTEN file, not figsize: a tight bbox grows the page to hold labels that overhang
    # the figure, and two radars shipped at 3.626 in while figsize said 3.5.
    w = plt.imread(OUT / f"{name}.png").shape[1] / plt.rcParams["savefig.dpi"]
    if w > COL1 + 0.005:
        raise SystemExit(f"FIG FAIL: {name}.png is {w:.3f} in wide -- over the {COL1} in column")
    written.append(name)

sr = J(SC / "stratified_report.json")
# Localisation for every arm from ONE uniform source (C-77 i): cell_table.json scores each arm on
# the pages its own generator saw. The constants this replaced mixed two pipeline stages in one
# DocBench column. CONTROL is the page-matched k=3 arm (C-80); no η against the k=2 arm is quotable.
_CT = J(SC / "cell_table.json")
_LK = {"mmlongbench": "ctx-hit (n=838)", "docbench": "quote-recall@5 (n=730)"}
LOC = {b: {arm: row[_LK[b]] for arm, row in _CT[b]["rows"].items()} for b in _LK}
CONTROL = "Flat-vector (k=3)"
assert all(CONTROL in LOC[b] and CONTROL in sr[b] for b in _LK), "the page-matched control is absent"

# ---------------- 1. latency: cold vs warm, the figure the cost-gate withdrawal is about ----------------
lat = J(COST / "query_latency_v2.json")["cells"]
fig, axes = plt.subplots(2, 1, figsize=(COL1, 3.7), sharex=True)
for ax, (b, title) in zip(axes, (("mmlongbench", "MMLongBench-Doc"), ("docbench", "DocBench"))):
    for sysk, lab in (("gramm", "GraMM-RAG"), ("colpali", "ColPali")):
        med = [lat[f"{sysk}|{b}|{st}"]["median"] for st in ("cold", "warm")]
        p95 = [lat[f"{sysk}|{b}|{st}"]["p95"] for st in ("cold", "warm")]
        x = np.array([0, 1]) + (-0.06 if sysk == "gramm" else 0.06)
        ax.vlines(x, med, p95, color=ARM[lab], lw=1.0, alpha=0.55)
        ax.plot(x, med, color=ARM[lab], marker=MARK[lab], ms=4.5, mec="white", mew=0.6, label=lab)
        ax.plot(x, p95, ls="none", marker="_", ms=7, color=ARM[lab], mew=1.2)
        ax.annotate(f"{med[1]:.0f}", (x[1], med[1]), xytext=(5, -2), textcoords="offset points",
                    fontsize=6.5, color=INK2, va="center")
    ax.set_ylabel("latency (ms)"); ax.set_title(title, loc="left", pad=2)
    ax.grid(axis="y"); ax.set_axisbelow(True); ax.set_ylim(0, None)
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["cold start", "warm"]); axes[1].set_xlim(-0.35, 1.35)
axes[0].legend(loc="upper right", ncol=2, handlelength=1.2, columnspacing=0.8)
fig.text(0.01, -0.035, "Marker: median; bar top: p95 (n = 1,000 per cell). ColPali is faster warm on both corpora;\n"
         "GraMM-RAG is faster and far more predictable only at cold start.", fontsize=6.5, color=INK2, ha="left", va="top")
fig.tight_layout(h_pad=0.6); save(fig, "fig_latency_cold_warm")

# ---------------- 2. resource radar (same derivation as notebook 10.2) ----------------
def cell(s, b, r): return lat[f"{s}|{b}|{r}"]
AX = [("Prep.\ncost", 9.36, 6.32, False), ("Index\nsize", 4.4, 2.8, False),
      ("Warm\nlatency", cell("gramm", "mmlongbench", "warm")["median"], cell("colpali", "mmlongbench", "warm")["median"], False),
      ("Cold-tail\nlatency", cell("gramm", "mmlongbench", "cold")["p95"], cell("colpali", "mmlongbench", "cold")["p95"], False),
      ("Peak query\nmemory", cell("gramm", "mmlongbench", "cold")["peak_gpu_gib"], cell("colpali", "mmlongbench", "cold")["peak_gpu_gib"], False),
      ("Tail\npredictability", 1 / cell("gramm", "mmlongbench", "cold")["p95_over_median"], 1 / cell("colpali", "mmlongbench", "cold")["p95_over_median"], True)]
g, c, labs = [], [], []
for name, gv, cv, hb in AX:
    best = max(gv, cv) if hb else min(gv, cv)
    sc = (lambda v: v / best) if hb else (lambda v: best / v)
    g.append(sc(gv)); c.append(sc(cv)); labs.append(name)
ang = np.linspace(0, 2 * np.pi, len(labs), endpoint=False).tolist(); ang += ang[:1]
fig, ax = plt.subplots(figsize=(COL1, 3.25), subplot_kw=dict(polar=True))
for vals, lab in ((g, "GraMM-RAG"), (c, "ColPali")):
    v = vals + vals[:1]; ax.plot(ang, v, color=ARM[lab], lw=1.3, label=lab); ax.fill(ang, v, color=ARM[lab], alpha=0.10)
ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(labs, fontsize=7); ax.tick_params(axis="x", pad=7)
# values of exactly 1.0 sat ON the ring where the labels are, and vertices ran through them
ax.set_ylim(0, 1.12); ax.set_yticks([.5, 1.0])
ring_labels(ax, [.5, 1.0], 90, lambda r: ".5" if r == .5 else "1")
ax.grid(color="#d6d5d0", lw=0.4); ax.spines["polar"].set_linewidth(0.5)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2)
fig.text(0.5, -0.1, "Each axis: ratio to the better system (1 = better). MMLongBench, one A100.",
         fontsize=6.5, color=INK2, ha="center")
save(fig, "fig_radar_resource")

# ---------------- 3. quality radars, one column each (two side by side are illegible at 3.5 in) ----------------
PLOT = ["Flat-vector", "LightRAG 1.5.7", "GraMM-RAG", "RAG-Anything 1.2.8", "MS GraphRAG 2.7.2", "ColPali"]
for b, title, labels in (("mmlongbench", "MMLongBench-Doc", ["Overall\nANLS", "Answerable\nANLS", "Localisation\n(ctx-hit)", "Abstention\n(bal. acc.)"]),
                         ("docbench", "DocBench", ["Overall\njudge acc.", "Answerable\njudge acc.", "Localisation\n(quote-rec.@5)", "Abstention\n(bal. acc.)"])):
    fig, ax = plt.subplots(figsize=(COL1, 3.45), subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.19, right=0.81)         # the side-axis labels overhang the polar disc
    ang = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist(); ang += ang[:1]
    for s in PLOT:
        if s not in sr[b] or LOC[b].get(s) is None:
            continue                                   # absent arm OMITTED, never imputed
        d = sr[b][s]; v = [d["pooled"], d["answerable"], LOC[b][s], d["abstention_balanced_acc"]]; v += v[:1]
        ax.plot(ang, v, color=ARM[s], lw=1.1, label=SHORT[s], ls=(0, (3, 1.5)) if s == "Flat-vector" else "-",
                marker=MARK[s], ms=3, mec="white", mew=0.4)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=7); ax.tick_params(axis="x", pad=8)
    ax.set_ylim(0, 1.0); ax.set_yticks([.25, .5, .75])
    # A 4-axis radar has NO empty sector (every between-axis sector is crossed by a polygon edge), so
    # the ring values are drawn above the data with a knockout rather than moved to a "clear" angle.
    ring_labels(ax, [.25, .5, .75], 45, lambda r: f"{r:.2f}".lstrip("0"))
    ax.grid(color="#d6d5d0", lw=0.4); ax.spines["polar"].set_linewidth(0.5)
    ax.set_title(title, pad=14, fontsize=8, fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.40), ncol=3, handlelength=1.4, columnspacing=0.7)
    save(fig, f"fig_radar_quality_{b}")

# ---------------- 4. conversion frontier (notebook section 11) -- eta in the legend, never on the plot ----------------
fig, axes = plt.subplots(2, 1, figsize=(COL1, 5.0))
_eta_stable, _eta_unstable, _trend_note = {}, {}, {}
SYS = ["Flat-vector", "Flat-vector (k=3)", "LightRAG 1.5.7", "GraMM-RAG", "RAG-Anything 1.2.8",
       "MS GraphRAG 2.7.2", "ColPali", "Union(CP∪GraMM)", "ColPali size-matched"]
for ax, (b, title, loc_l, q_l) in zip(axes, (("mmlongbench", "MMLongBench-Doc", "localisation (ctx-hit)", "pooled ANLS"),
                                             ("docbench", "DocBench", "localisation (quote-recall@5)", "pooled judge accuracy"))):
    bl_x, bl_y = LOC[b][CONTROL], sr[b][CONTROL]["pooled"]
    # Two trends, because the best-localising arm is one WE built. Union(CP∪GraMM) and ColPali
    # size-matched exist only inside this project, so a conversion rate defined by them is a rate
    # no reader can ship. The primary (dashed) line is anchored on the best arm that exists
    # outside this project; the derived arm's line is drawn faintly beside it and named.
    DERIVED = {"Union(CP∪GraMM)", "ColPali size-matched"}
    _pres = [a for a in SYS if a in sr[b] and LOC[b].get(a) is not None and a != CONTROL]
    _real = max((a for a in _pres if a not in DERIVED), key=lambda a: LOC[b][a])
    _any = max(_pres, key=lambda a: LOC[b][a])
    xs = np.linspace(min(LOC[b].values()) - 0.02, max(LOC[b].values()) + 0.02, 2)
    for _anchor, _ls, _lw, _al in ((_real, (0, (4, 2)), 0.9, 1.0),) + (
            ((_any, (0, (1, 2)), 0.7, 0.75),) if _any != _real else ()):
        _sl = (sr[b][_anchor]["pooled"] - bl_y) / (LOC[b][_anchor] - bl_x)
        ax.plot(xs, bl_y + _sl * (xs - bl_x), ls=_ls, color="#9b9a95", lw=_lw, alpha=_al, zorder=1)
    _trend_note[b] = (SHORT[_real], SHORT[_any] if _any != _real else None)
    for s in SYS:
        if s not in sr[b] or LOC[b].get(s) is None: continue
        x, y = LOC[b][s], sr[b][s]["pooled"]
        # An eta whose DENOMINATOR is near zero is not a small number, it is an undefined one:
        # LightRAG's ctx-hit sits 0.004 from the page-matched control and its ratio reads 4.09.
        # Section 15 flags these as UNSTABLE rather than giving them a spuriously precise value,
        # and the figure must not say something the analysis refuses to.
        dl = x - bl_x
        eta = (y - bl_y) / dl if abs(dl) > 1e-9 else float("nan")
        # The legend is SHARED by both panels, so it carries names only: eta differs per corpus
        # (GraMM is 0.19 on MMLongBench and 0.30 on DocBench), and one legend cannot state both.
        # Eta goes in a per-panel line instead.
        lab = SHORT[s]
        if s != CONTROL:
            # Not split(" (")[0]: that turned "ColPali (size-m.)" into a second "ColPali",
            # so the annotation listed two different arms under one name.
            _nm = SHORT[s].replace(" (size-m.)", " size-m.").replace(" (k=2)", " k=2")
            (_eta_unstable if abs(dl) < 0.05 else _eta_stable).setdefault(b, []).append(
                _nm if abs(dl) < 0.05 else f"{_nm} {eta:.2f}")
        hollow = (s == "Flat-vector")          # the retired k=2 control: same gray, hollow square
        ax.scatter([x], [y], s=(62 if MARK[s] == "*" else 26),
                   facecolor=("none" if hollow else ARM[s]), color=None if hollow else ARM[s],
                   edgecolor=(ARM[s] if hollow else "white"), marker=MARK[s],
                   linewidth=(0.9 if hollow else 0.5), zorder=3, label=lab)
    ax.set_xlabel(loc_l); ax.set_ylabel(q_l); ax.set_title(title, loc="left", pad=2)
    ax.grid(); ax.set_axisbelow(True)
    # Wrapped by hand at the axes width: the 6.5 pt floor from C-71 is not negotiable for a
    # figure that has to be read at 3.5 in.
    _lines, _cur = [], "η: "
    for _e in _eta_stable.get(b, []):
        if len(_cur) + len(_e) > 44:
            _lines.append(_cur.rstrip(" ·")); _cur = "    "
        _cur += _e + " · "
    _lines.append(_cur.rstrip(" ·"))
    if _eta_unstable.get(b):
        _lines.append("unstable (Δloc≈0): " + ", ".join(_eta_unstable[b]))
    ax.text(0.02, 0.97, "\n".join(_lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=6.5, color=INK2, linespacing=1.35)
# No per-axes legend: nine entries inside the axes covered RAG-Anything's point on the MMLongBench
# panel. One shared legend below both panels instead.
# The caption runs to three lines now that both trends are named, so the reserved strip grew with
# it -- at the previous height it landed on the legend's last entry.
fig.tight_layout(h_pad=0.6, rect=(0, 0.25, 1, 1))
_h, _l = axes[0].get_legend_handles_labels()
fig.legend(_h, _l, loc="lower center", ncol=2, fontsize=6.0, frameon=False, handletextpad=0.3,
           columnspacing=0.8, labelspacing=0.22, bbox_to_anchor=(0.5, 0.095))
fig.text(0.5, 0.008, "η = Δquality / Δlocalisation against the PAGE-MATCHED control (k=3, C-80). Descriptive:\n"
                     "no pairwise difference is significant (bootstrap CIs). Dashed: control → best\n"
                     "SHIPPABLE system. Dotted: control → best arm overall, which is one built here.",
         fontsize=6.0, color=INK2, ha="center", va="bottom")
save(fig, "fig_conversion_frontier")

# ---------------- 4b. length-bin conversion (notebook section 13), against the page-matched control ----
# C-81: the published "101+ collapse" was measured against the k=2 control and does not survive
# matching. Both panels are drawn from scores/length_bins_k3.json so the figure cannot drift from
# the notebook that computed it.
_LB = J(SC / "length_bins_k3.json")
assert _LB["control"] == CONTROL, f"length bins use {_LB['control']}, the frontier uses {CONTROL}"
fig, axes = plt.subplots(2, 1, figsize=(COL1, 4.4), sharex=True)
_x = np.arange(len(_LB["bins"]))
_ORDER = [a for a in SYS if a in _LB["d_quality"]]
for ax, key, ylab in ((axes[0], "d_localisation", "Δ localisation"),
                      (axes[1], "d_quality", "Δ answer quality")):
    for a in _ORDER:
        ax.plot(_x, _LB[key][a], marker=MARK[a], ms=3.2, lw=1.0, color=ARM[a],
                ls=("--" if a == "Flat-vector" else "-"),
                mfc=("none" if a == "Flat-vector" else ARM[a]), mew=0.8,
                label=SHORT[a] if key == "d_localisation" else None)
    ax.axhline(0, color=INK, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_ylabel(ylab); ax.grid(); ax.set_axisbelow(True)
axes[0].set_title("Against the page-matched control, by document length", loc="left", pad=2)
axes[1].set_xticks(_x); axes[1].set_xticklabels(_LB["bins"])
axes[1].set_xlabel("document length (pages)")
# The legend goes BELOW both panels: inside the axes it covered the vision lines at 21-60 pages,
# which is where the panels are read. The notebook figure hit the same collision.
fig.tight_layout(h_pad=0.5, rect=(0, 0.14, 1, 1))
_h, _l = axes[0].get_legend_handles_labels()
fig.legend(_h, _l, loc="lower center", ncol=3, fontsize=6.0, frameon=False,
           handletextpad=0.3, columnspacing=1.0, labelspacing=0.25, bbox_to_anchor=(0.5, 0.045))
fig.text(0.5, 0.005, "MMLongBench, ANLS and ctx-hit. No arm's answer advantage is negative at 101+ once the\n"
                     "control reads the same three pages (C-81); the 101+ bin is n=115.",
         fontsize=6.0, color=INK2, ha="center")
save(fig, "fig_length_conversion")

# ---------------- 5. utility-attribute bars: does any retrieval-side attribute detect answerability? ----------------
ad = {b: J(SC / f"attr_decomp_{b}.json") for b in ("mmlongbench", "docbench")}
ATTR = [("auc_coverage", "coverage"), ("auc_similarity", "similarity"), ("auc_temporal", "temporal"),
        ("auc_U", "U\n(weighted)"), ("joint_cv_auc", "joint\n(CV logit)")]
CEIL = {"mmlongbench": 0.806, "docbench": 0.750}          # generator's decline decision, identical context (C-28)
CORP = {"mmlongbench": ("MMLongBench-Doc", "#3a3a38"), "docbench": ("DocBench", "#a9a8a2")}   # neutral: not arms
fig, ax = plt.subplots(figsize=(COL1, 2.5))
x = np.arange(len(ATTR)); w = 0.36; RX = len(ATTR) - 0.45      # right margin, clear of every bar
for i, b in enumerate(("mmlongbench", "docbench")):
    vals = [ad[b].get(k, np.nan) for k, _ in ATTR]
    ax.bar(x + (i - 0.5) * w, vals, w * 0.94, color=CORP[b][1], label=CORP[b][0], zorder=2)
    for xi, v in zip(x + (i - 0.5) * w, vals):     # labels INSIDE bar tops: nothing above can collide
        if v == v:
            ax.text(xi, v - 0.008, f"{v:.2f}", ha="center", va="top", fontsize=6.5,
                    color=("white" if b == "mmlongbench" else INK), rotation=90)
    ax.hlines(CEIL[b], -0.5, RX, colors=CORP[b][1], linestyles=(0, (4, 2)), lw=0.9, zorder=1)
    ax.text(RX + 0.05, CEIL[b], f"{'MMLB' if b == 'mmlongbench' else 'DocBench'} gen.",
            fontsize=6.5, color=(INK if b == "mmlongbench" else INK2), ha="left", va="center")
ax.hlines(0.5, -0.5, RX, colors=INK, lw=0.6, zorder=1)
ax.text(RX + 0.05, 0.5, "chance", fontsize=6.5, color=INK, ha="left", va="center")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in ATTR]); ax.set_ylabel("AUC, answerability")
ax.set_xlim(-0.55, RX + 0.95); ax.set_ylim(0.4, 0.92)
ax.grid(axis="y"); ax.set_axisbelow(True)
ax.legend(loc="upper left", ncol=2, handlelength=1.0, bbox_to_anchor=(0.0, 1.02))
save(fig, "fig_utility_attributes")

print("wrote:", ", ".join(written))
