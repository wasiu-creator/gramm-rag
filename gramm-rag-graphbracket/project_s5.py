#!/usr/bin/env python3
"""S5 full-corpus projection, exactly as pre-registered in S5_PILOT_PREREGISTRATION.md 1a/1c.

Per document:
  extraction (+ description summarisation) and embeddings   ~ linear in ingested chars
  unassigned calls                                           ~ linear in ingested chars (kept, never dropped)
  community reports = r * communities,  communities = k * chars^beta   (beta is what the pilot measures)
Projected by SUMMING over all 364 documents' ingested_chars -- with beta > 1 the total depends on the
size distribution, not only its sum (Jensen). Operative total = upper bound of the 95% bootstrap CI.
Pure python: the host interpreter has no numpy.
"""
import json, math, random, sys
from pathlib import Path

GB = Path(__file__).resolve().parent
OUT = GB / "tier1" / "s5_graphrag"
B, SEED = 4000, 42


def stage(r, name):
    return r["stages"].get(name, {}).get("cost_usd", 0.0)


def fit(rows):
    """Returns params dict, or None if the log-log community fit is degenerate."""
    C = sum(r["ingested_chars"] for r in rows)
    if C <= 0:
        return None
    pts = [(math.log(r["ingested_chars"]), math.log(r["n_communities"]))
           for r in rows if r["n_communities"] > 0 and r["ingested_chars"] > 0]
    if len({x for x, _ in pts}) < 3:
        return None
    mx = sum(x for x, _ in pts) / len(pts); my = sum(y for _, y in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None
    beta = sum((x - mx) * (y - my) for x, y in pts) / sxx
    k = math.exp(my - beta * mx)
    ncomm = sum(r["n_communities"] for r in rows)
    lp = [(math.log(r["ingested_chars"]), math.log(r["seconds"])) for r in rows
          if r["seconds"] > 0 and r["ingested_chars"] > 0]
    tx = sum(x for x, _ in lp) / len(lp); ty = sum(y for _, y in lp) / len(lp)
    txx = sum((x - tx) ** 2 for x, _ in lp)
    gamma = sum((x - tx) * (y - ty) for x, y in lp) / txx if txx else 1.0
    return dict(
        a_ext=sum(stage(r, "extraction") + stage(r, "summarize") for r in rows) / C,
        a_emb=sum(stage(r, "embedding") for r in rows) / C,
        a_una=sum(stage(r, "unassigned") for r in rows) / C,
        r_rep=sum(stage(r, "community_reports") for r in rows) / ncomm if ncomm else 0.0,
        k=k, beta=beta, s=math.exp(ty - gamma * tx), gamma=gamma)


def project(p, corpus):
    lin = (p["a_ext"] + p["a_emb"] + p["a_una"]) * sum(corpus)
    rep = sum(p["r_rep"] * p["k"] * c ** p["beta"] for c in corpus)
    secs = sum(p["s"] * c ** p["gamma"] for c in corpus)
    return lin + rep, lin, rep, secs / 3600


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    rows = [r for r in json.loads((OUT / f"index_log_{tag}.json").read_text())["docs"]]
    corpus = [d["ingested_chars"] for d in json.loads((OUT / "corpus_chars.json").read_text())["docs"]]
    print(f"  pilot {len(rows)} docs, {sum(r['ingested_chars'] for r in rows):,} chars, "
          f"${sum(r['cost_usd'] for r in rows):.4f};  corpus {len(corpus)} docs, {sum(corpus):,} chars")
    p = fit(rows)
    if p is None:
        raise SystemExit("  FIT DEGENERATE: fewer than 3 distinct documents with communities")
    tot, lin, rep, hrs = project(p, corpus)
    lin_rep = p["r_rep"] * p["k"] * (sum(corpus) / len(corpus)) ** p["beta"] * len(corpus)
    print(f"\n  point fit: beta {p['beta']:.3f}  k {p['k']:.3e}  $/community {p['r_rep']:.5f}  "
          f"extraction ${p['a_ext']*1e6:.3f}/1M  embedding ${p['a_emb']*1e6:.4f}/1M  gamma(time) {p['gamma']:.3f}")
    print(f"  point projection ${tot:.2f} = per-char ${lin:.2f} + community reports ${rep:.2f}   "
          f"wall-clock {hrs:.1f} h")
    print(f"  (reports if every doc were average-sized: ${lin_rep:.2f} -- the size-distribution effect is "
          f"${rep - lin_rep:+.2f})")

    rng = random.Random(SEED); draws, degen = [], 0
    for _ in range(B):
        q = fit([rows[rng.randrange(len(rows))] for _ in rows])
        if q is None:
            degen += 1; continue
        draws.append((project(q, corpus), q["beta"]))
    tots = sorted(d[0][0] for d in draws); betas = sorted(d[1] for d in draws)
    hrs_s = sorted(d[0][3] for d in draws)
    ci = lambda v: (v[int(.025 * len(v))], v[int(.975 * len(v))])
    (tlo, thi), (blo, bhi), (hlo, hhi) = ci(tots), ci(betas), ci(hrs_s)
    print(f"\n  bootstrap {len(draws):,}/{B:,} usable ({degen} degenerate, {degen/B:.1%})")
    print(f"    total   95% CI ${tlo:.2f} - ${thi:.2f}")
    print(f"    beta    95% CI {blo:.3f} - {bhi:.3f}")
    print(f"    wall-clock 95% CI {hlo:.1f} - {hhi:.1f} h")

    cost = sum(r["cost_usd"] for r in rows)
    una = sum(stage(r, "unassigned") for r in rows)
    print("\n  === PRE-REGISTERED NOT-IDENTIFIED TESTS (1c) ===")
    i_fail = cost > 0 and una / cost > 0.05
    ii_fail = blo <= 1.0 <= bhi and blo <= 1.2 <= bhi
    print(f"    (i)  unassigned cost share {una/cost if cost else 0:.2%} (limit 5%): {'NOT SEPARABLE' if i_fail else 'separable'}")
    print(f"    (ii) beta CI contains both 1.0 and 1.2: {'YES -> NOT IDENTIFIED' if ii_fail else 'no'}")
    print(f"    -> {'NOT IDENTIFIED' if (i_fail or ii_fail) else 'IDENTIFIED'}")

    op = thi
    print(f"\n  === OPERATIVE TOTAL (upper 95% bound, fixed before data): ${op:.2f} ===")
    if op > 250 or hhi > 72:
        print("  -> > $250 or wall-clock > 72 h: SCOPE DOWN to MMLongBench only (mine)")
    elif op > 150:
        print("  -> $150-250: STOP AND ESCALATE to Prince as a cap decision")
    else:
        print("  -> <= $150: PROCEED on both benchmarks (automatic)")


if __name__ == "__main__":
    main()
