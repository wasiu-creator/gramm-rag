#!/usr/bin/env python3
"""P2 analysis -- judge robustness, evaluated exactly as pre-registered.

Primary quantity: Kendall's tau-b between the two judges' rank orders of the 7 arms, the same
statistic the corpus-dependence analysis used between corpora (localisation +0.828, answer
quality +0.600, abstention -0.828). Pure python: the host interpreter has no numpy.
"""
import json, math, random
from pathlib import Path

CP = Path(__file__).resolve().parent
D3 = CP.parent / "gramm-rag-3benchmarks" / "data" / "docbench"
B, SEED = 4000, 42


def tau_b(x, y):
    n = len(x); C = D = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = x[i] - x[j], y[i] - y[j]
            s = (a > 0) - (a < 0), (b > 0) - (b < 0)
            if s[0] * s[1] > 0: C += 1
            elif s[0] * s[1] < 0: D += 1
    def ties(v):
        c = {}
        for t in v: c[t] = c.get(t, 0) + 1
        return sum(k * (k - 1) / 2 for k in c.values())
    n0 = n * (n - 1) / 2
    den = math.sqrt((n0 - ties(x)) * (n0 - ties(y)))
    return (C - D) / den if den else float("nan")


def kappa(a, b):
    n = len(a); po = sum(1 for i in range(n) if a[i] == b[i]) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def main():
    d = json.loads((CP / "scores" / "p2_judge_sweep.json").read_text())
    arms = list(d["arms"])
    an = json.loads((D3 / "baseline_vector_docbench.json").read_text())["is_answerable"]
    n = len(an)
    S2 = {a: d["arms"][a]["scores_second_judge"] for a in arms}
    S1 = {a: d["arms"][a]["scores_canonical"] for a in arms}
    strata = {"pooled": list(range(n)),
              "answerable": [i for i in range(n) if an[i]],
              "unanswerable": [i for i in range(n) if not an[i]]}
    acc = lambda s, idx: sum(s[i] for i in idx) / len(idx)

    print(f"{'arm':24s}{'canonical':>11s}{'second':>9s}{'delta':>9s}{'agree':>8s}{'kappa':>8s}")
    for a in arms:
        print(f"{a:24s}{acc(S1[a], strata['pooled']):>11.4f}{acc(S2[a], strata['pooled']):>9.4f}"
              f"{acc(S2[a], strata['pooled']) - acc(S1[a], strata['pooled']):>+9.4f}"
              f"{d['arms'][a]['raw_agreement']:>8.4f}{kappa(S1[a], S2[a]):>8.3f}")

    print("\n  === PRIMARY: tau-b between judges, by stratum ===")
    taus = {}
    for name, idx in strata.items():
        x = [acc(S1[a], idx) for a in arms]; y = [acc(S2[a], idx) for a in arms]
        taus[name] = tau_b(x, y)
        o1 = [a for _, a in sorted(zip(x, arms), reverse=True)]
        o2 = [a for _, a in sorted(zip(y, arms), reverse=True)]
        print(f"    {name:14s} tau-b {taus[name]:+.3f}   order changed: {'yes' if o1 != o2 else 'no'}")

    rng = random.Random(SEED); draws = {k: [] for k in strata}
    for _ in range(B):
        for name, idx in strata.items():
            r = [idx[rng.randrange(len(idx))] for _ in idx]
            x = [sum(S1[a][i] for i in r) / len(r) for a in arms]
            y = [sum(S2[a][i] for i in r) / len(r) for a in arms]
            t = tau_b(x, y)
            if t == t: draws[name].append(t)
    ci = lambda v: (sorted(v)[int(.025 * len(v))], sorted(v)[int(.975 * len(v))])
    print(f"\n  bootstrap ({B:,} paired resamples of questions, seed {SEED})")
    for name in strata:
        lo, hi = ci(draws[name])
        print(f"    {name:14s} tau-b 95% CI [{lo:+.3f}, {hi:+.3f}]")

    print("\n  === pairwise reversals (both judges confident, opposite directions) ===")
    rev = []
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            A, Bm = arms[i], arms[j]
            d1 = acc(S1[A], strata['pooled']) - acc(S1[Bm], strata['pooled'])
            d2 = acc(S2[A], strata['pooled']) - acc(S2[Bm], strata['pooled'])
            if (d1 > 0) == (d2 > 0):
                continue
            r1, r2 = [], []
            rr = random.Random(SEED)
            for _ in range(1000):
                s = [rr.randrange(n) for _ in range(n)]
                r1.append(sum(S1[A][k] - S1[Bm][k] for k in s) / n)
                r2.append(sum(S2[A][k] - S2[Bm][k] for k in s) / n)
            c1, c2 = ci(r1), ci(r2)
            if (c1[0] > 0 or c1[1] < 0) and (c2[0] > 0 or c2[1] < 0):
                rev.append((A, Bm, d1, d2, c1, c2))
                print(f"    {A} vs {Bm}: canonical {d1:+.4f} {c1}, second {d2:+.4f} {c2}")
    if not rev:
        print("    none")

    t = taus["pooled"]
    print("\n  === PRE-REGISTERED DISPOSITION ===")
    if t >= 0.90 and not rev:
        print(f"    tau-b {t:+.3f} >= 0.90, no significant reversal -> RANKINGS ARE JUDGE-ROBUST.")
        print("    Evaluation instability is localised to corpus choice; SINGLE_BENCHMARK_FINDING sharpened.")
    elif t <= 0.60 or rev:
        print(f"    tau-b {t:+.3f} <= 0.60 or {len(rev)} significant reversal(s) -> EVALUATION FRAGILITY")
        print("    IS BROADER THAN CORPUS CHOICE. Weakens any single-judge ranking, this project's included.")
    else:
        print(f"    0.60 < tau-b {t:+.3f} < 0.90, no significant reversal -> PARTIAL; NO CLAIM EITHER WAY.")
    sa, su = taus["answerable"], taus["unanswerable"]
    if (sa >= 0.90) != (su >= 0.90):
        print(f"    MIXED ACROSS STRATA (answerable {sa:+.3f} vs unanswerable {su:+.3f}) -> per the fixed")
        print("    rule this confirms NEITHER branch; reported as stratum-dependent with the split stated.")
    print(f"\n  vs corpus-dependence reference: localisation +0.828, quality +0.600, abstention -0.828")


if __name__ == "__main__":
    main()
