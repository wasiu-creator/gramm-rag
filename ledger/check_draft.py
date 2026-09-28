#!/usr/bin/env python3
"""Scan a draft for claims the ledger has retracted or superseded.

  python3 check_draft.py DRAFT.md [DRAFT2.docx.txt ...]

For every claim whose status is retracted / superseded / not established, searches the draft for
each phrase of its ORIGINAL wording (case-insensitive, whitespace- and typography-normalised) and
reports the line, the claim id, its status, and what replaced it. Caveated claims are flagged when
their wording appears so the writer can confirm the caveat travels with them.

Exit status is the number of retracted/superseded hits (0 = clean). This is the check for the
failure mode the ledger exists to prevent: a writer reaching for a claim remembered as true.
"""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Compile from SOURCE every run -- never through importlib's bytecode cache. The cache revalidates
# on (mtime to 1 s, size): a same-size edit inside one second was rendered from STALE bytecode
# during this file's own negative controls. A verifier must not verify a cached copy of its input.
import types
src = types.ModuleType("claims_source")
_p = HERE / "claims_source.py"
exec(compile(_p.read_text(), str(_p), "exec"), src.__dict__)
byid = {c["id"]: c for c in src.CLAIMS}
N = lambda t: re.sub(r"\s+", " ", t.replace("−", "-").replace("’", "'")).lower()

BLOCK = {"retracted", "superseded", "not established"}
if len(sys.argv) < 2:
    sys.exit(__doc__)
total = 0
for path in sys.argv[1:]:
    lines = Path(path).read_text(errors="replace").splitlines()
    print(f"=== {path} ({len(lines)} lines)")
    found = 0
    for c in src.CLAIMS:
        if c["status"] not in BLOCK and c["status"] != "caveated":
            continue
        for phrase in c.get("original", []):
            p = N(phrase)
            if len(p) < 6:                                    # too short to search safely
                continue
            for ln, line in enumerate(lines, 1):
                if p in N(line):
                    rep = c.get("superseded_by")
                    if c["status"] == "caveated":
                        tgt = f"-> may stand ONLY with its caveat: {c.get('caveat','')[:110]}"
                    elif rep in byid:
                        tgt = f"-> use {rep}: {byid[rep]['claim'][:90]}"
                    else:
                        tgt = "-> do not state"
                    tag = "CAVEAT" if c["status"] == "caveated" else c["status"].upper()
                    print(f"  line {ln:>4}  [{c['id']} {tag}]  «{phrase}»\n              {tgt}")
                    if c["status"] in BLOCK:
                        found += 1
    print(f"  {found} retracted/superseded/unestablished wording(s) found\n")
    total += found
sys.exit(min(total, 125))
