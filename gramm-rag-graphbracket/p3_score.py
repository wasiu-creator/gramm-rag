#!/usr/bin/env python3
"""D4 -- P3 answer scoring: ANLS over the FIVE-ARM COMMON SET.

MP-DocVQA is ANLS-only, pre-registered: it ships no judge and no unanswerable stratum, so
no judge accuracy and no decline-decision analysis are computable here. That was stated in
P3_PREREGISTRATION.md before any result existed, not discovered now.

Every cross-arm contrast is computed on the intersection of all five arms' available
questions (n=950), not on each arm's own count. The 28 excluded questions failed retrieval
for RAG-Anything alone and are significantly EASIER than chance (doc pages p=0.0196,
question chars p<0.0001, gold depth p=0.0411), so their removal makes RAG-Anything's row
conservative rather than flattering.

Score dispersion is reported per arm because P3's headline may be "rankings agree on the
third corpus", and a reviewer can ask whether agreement is genuine or an artifact of scores
compressing toward a ceiling -- MP-DocVQA's native page images are 2.82x the pixel area of
the 120-dpi renders used on the other two corpora. Dispersion makes that testable rather
than assertable.
"""
import json, random, statistics, sys
from pathlib import Path

GB = Path(__file__).resolve().parent
ROOT = GB.parent
sys.path.insert(0, str(ROOT / "gramm-rag-mpdocvqa"))
from src.evaluation.metrics import anls_score        # noqa: E402  (canonical scorer)
sys.path.insert(0, str(GB))
from p3_postproc import parse_answer, EXTRACTED_MD5  # noqa: E402  (canonical POSTPROC_V=2)

# ALL FIVE arms were generated in this session (D3 regenerated flat and gramm so every arm
# shares k=3 / 12000 chars / T=0.0), so all five hold RAW generator output and all five need
# POSTPROC_V=2.
#
# The first version of this restricted post-processing to {lightrag, s2, s5} on the grounds
# that flat and gramm were "archived". That conflated two different things: their RETRIEVAL
# is archived, their PREDICTIONS are not. The label travelled from the source of one input
# to a decision about a different one, and the symptom was both arms scoring 0.0012 with
# byte-identical SD -- two distinct systems cannot produce identical score distributions.
RAW_ARMS = {"lightrag", "s2", "s5", "flat", "gramm", "flatk3"}

P3 = GB / "tier1/p3_mpdocvqa"
GEN = P3 / "generation"
# flatk3 is the page-matched control (C-77(ii)); flat (k=2) stays for provenance.
ARMS = ["gramm", "flat", "flatk3", "lightrag", "s2", "s5"]
NAME = {"lightrag": "LightRAG", "s2": "RAG-Anything", "s5": "MS GraphRAG",
        "flat": "Flat-vector", "gramm": "GraMM-RAG", "flatk3": "Flat-vector (k=3)"}
SEED, B = 42, 4000

sub = json.loads((P3 / "subsample.json").read_text())
qs = sub["questions"]
common = sorted(json.loads((P3 / "common_set.json").read_text())["excluded_indices"])
common = [i for i in range(len(qs)) if i not in set(common)]
print(f"five-arm common set: n = {len(common)}")
print(f"POSTPROC_V=2 applied to {sorted(RAW_ARMS)} (raw this session); archived arms as stored")
print(f"p3_postproc extracted md5 {EXTRACTED_MD5[:12]}\n")

preds, imgs = {}, {}
for a in ARMS:
    d = json.loads((GEN / f"p3_generation_{a}.json").read_text())
    preds[a] = d["predictions"]
    imgs[a] = d["page_images_attached"] / max(d["n_generated"], 1)

scores = {}
for a in ARMS:
    s = []
    for i in common:
        p = preds[a][i] or ""
        if a in RAW_ARMS:
            p = parse_answer(p)
        golds = list(dict.fromkeys(qs[i]["answers"])) or [""]   # distinct variants only
        s.append(max(anls_score(p, g) for g in golds))
    scores[a] = s
    assert len(s) == len(common)

print(f"{'arm':14s}{'ANLS':>8s}{'SD':>8s}{'IQR':>8s}{'=0':>7s}{'=1':>7s}{'img/q':>8s}")
for a in ARMS:
    s = scores[a]
    q = sorted(s)
    iqr = q[int(.75*len(q))] - q[int(.25*len(q))]
    print(f"{NAME[a]:14s}{statistics.mean(s):>8.4f}{statistics.pstdev(s):>8.4f}{iqr:>8.4f}"
          f"{sum(1 for x in s if x == 0)/len(s):>7.3f}{sum(1 for x in s if x >= .999)/len(s):>7.3f}"
          f"{imgs[a]:>8.2f}")

def paired(a, b):
    da, db = scores[a], scores[b]
    diff = statistics.mean(da) - statistics.mean(db)
    rng = random.Random(SEED); n = len(da)
    boot = sorted(statistics.mean([da[j] for j in idx]) - statistics.mean([db[j] for j in idx])
                  for idx in ([rng.randrange(n) for _ in range(n)] for _ in range(B)))
    return diff, boot[int(.025*B)], boot[int(.975*B)]

print(f"\npaired contrasts on n={len(common)} (bootstrap {B}, seed {SEED})")
print(f"{'contrast':34s}{'diff':>9s}{'95% CI':>22s}")
pairs = [("gramm","flat"),("gramm","lightrag"),("gramm","s2"),("gramm","s5"),
         ("flat","lightrag"),("flat","s2"),("flat","s5"),
         ("s2","lightrag"),("lightrag","s5"),("s2","s5"),
         # C-77(ii): the page-matched control. "gramm - flatk3" is the contrast the dissertation
         # cites for this corpus; "flatk3 - flat" is the page-budget effect in isolation.
         ("gramm","flatk3"),("flatk3","flat"),("flatk3","lightrag"),("flatk3","s2"),("flatk3","s5")]
for a,b in pairs:
    d,lo,hi = paired(a,b)
    sig = "*" if (lo>0 or hi<0) else " "
    print(f"{NAME[a]+' - '+NAME[b]:34s}{d:>+9.4f}  [{lo:+.4f}, {hi:+.4f}]{sig}")

out = dict(n_common=len(common), benchmark="mpdocvqa", metric="ANLS",
           per_arm={NAME[a]: dict(anls=round(statistics.mean(scores[a]),4),
                                  sd=round(statistics.pstdev(scores[a]),4),
                                  frac_zero=round(sum(1 for x in scores[a] if x==0)/len(scores[a]),4),
                                  frac_one=round(sum(1 for x in scores[a] if x>=.999)/len(scores[a]),4),
                                  images_per_question=round(imgs[a],3)) for a in ARMS},
           contrasts={f"{NAME[a]} - {NAME[b]}": dict(zip(("diff","lo","hi"),
                      [round(x,4) for x in paired(a,b)])) for a,b in pairs})
(P3 / "p3_anls.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {P3/'p3_anls.json'}")
