#!/usr/bin/env python3
"""Render CLAIMS_LEDGER.md from claims_source.py, and REFUSE to render anything unverifiable.

  python3 build_ledger.py          write CLAIMS_LEDGER.md
  python3 build_ledger.py --check  exit 1 if the committed ledger differs from a fresh render, or
                                   if any provenance check fails (the sync assertion)

Fails on: a missing source document; a quote not present verbatim (whitespace-normalised) in its
source; a JSON path that does not resolve; a supersedes/superseded_by pointer to a nonexistent id;
a docs_must_show document that does not print the rendered number. Pure stdlib, runs on the host.
"""
import json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # .../grammrag
OUT = HERE.parent / "CLAIMS_LEDGER.md"
# Compile from SOURCE every run -- never through importlib's bytecode cache. The cache revalidates
# on (mtime to 1 s, size): a same-size edit inside one second was rendered from STALE bytecode
# during this file's own negative controls. A verifier must not verify a cached copy of its input.
import types
src = types.ModuleType("claims_source")
# LEDGER_SOURCE lets a guard be tested against a deliberately broken copy without editing the
# live source -- a guard that has never been seen to fail is not evidence (rule 2).
_p = Path(os.environ.get("LEDGER_SOURCE") or (HERE / "claims_source.py"))
exec(compile(_p.read_text(), str(_p), "exec"), src.__dict__)

errors = []
norm = lambda t: re.sub(r"\s+", " ", t).strip()
# typography-insensitive form for NUMBER checks only: Unicode minus -> "-", "5,187" -> "5187"
numform = lambda t: re.sub(r"(?<=\d),(?=\d{3}\b)", "", t.replace("\u2212", "-"))
_docs, _json = {}, {}

def doc(path):
    if path not in _docs:
        p = ROOT / path
        if not p.exists():
            errors.append(f"missing source document: {path}"); _docs[path] = None
        else:
            t = p.read_text(errors="replace")
            if p.suffix == ".ipynb":                       # search source AND outputs
                nb = json.loads(t); parts = []
                for c in nb["cells"]:
                    parts.append("".join(c.get("source", "")))
                    for o in c.get("outputs", []):
                        parts.append("".join(o.get("text", "")))
                        parts.append("".join(o.get("data", {}).get("text/plain", "")))
                t = "\n".join(parts)
            _docs[path] = norm(t)
    return _docs[path]

def jget(alias, path):
    f = src.ROOT_ALIASES[alias]
    if f not in _json:
        p = ROOT / f
        if not p.exists():
            raise KeyError(f"JSON alias '{alias}' -> {f} does not exist")
        _json[f] = json.loads(p.read_text())
    v = _json[f]
    for k in path.split("/"):
        v = v[int(k)] if isinstance(v, list) else v[k]
    return v

TOK = re.compile(r"\{(\w+):([^|}]+)(?:\|([^}]+))?\}")
def render_value(cid, kind, text):
    if kind == "Q":
        return text, [text]
    nums = []
    def sub(m):
        alias, path, fmt = m.group(1), m.group(2), m.group(3)
        try:
            v = jget(alias, path)
        except Exception as e:
            errors.append(f"{cid}: cannot resolve {{{alias}:{path}}} ({type(e).__name__}: {e})"); return "??"
        s = format(v, fmt) if fmt else str(v)
        nums.append(s); return s
    return TOK.sub(sub, text), nums

ids = {c["id"] for c in src.CLAIMS}
rows = []
for c in src.CLAIMS:
    cid = c["id"]
    kind, text = c["value"]
    val, checkables = render_value(cid, kind, text)
    sdoc, ssec = c["src"]
    d = doc(sdoc)
    if d is not None and kind == "Q" and norm(text) not in d:
        errors.append(f"{cid}: quote not found verbatim in {sdoc}: «{text[:70]}»")
    for other in c.get("docs_must_show", []):
        od = doc(other)
        if od is None: continue
        odn = numform(od)
        for n in checkables:
            n2 = numform(n); bare = n2.lstrip("+")          # "+0.0315" may print as "0.0315"
            if n2 not in odn and bare not in odn:
                errors.append(f"{cid}: {other} does not show the data value {n} -- document disagrees with the data")
    for key in ("supersedes", "superseded_by"):
        for ref in [x.strip() for x in (c.get(key) or "").split(",") if x.strip()]:
            if ref not in ids:
                errors.append(f"{cid}: {key} points at unknown id {ref}")
    rows.append((c, val))

# ---- tables, cell by cell ----
table_report = []
for T in getattr(src, "TABLES", []):
    raw = (ROOT / T["doc"]).read_text() if (ROOT / T["doc"]).exists() else None
    if raw is None:
        errors.append(f"{T['id']}: missing {T['doc']}"); continue
    if T["header"] not in raw:
        errors.append(f"{T['id']}: table header «{T['header']}» not found in {T['doc']}"); continue
    i = raw.index(T["header"]); block = raw[i:].split("\n\n")[0].splitlines()[2:]
    got = {}
    for line in block:
        # Footnote markers are presentation, not identity: strip any of them from the row label.
        cells = [c.strip().strip("*").translate({ord(m): None for m in "‡§†¶"}).strip()
                 for c in line.strip().strip("|").split("|")]
        if cells and cells[0] in T["rows"]:
            got[cells[0]] = dict(zip(T["cols"], [c.strip("*").strip() for c in cells[1:1 + len(T["cols"])]]))
    ncell = 0
    for arm, spec_ in T["rows"].items():
        if arm not in got:
            errors.append(f"{T['id']}: row «{arm}» missing from {T['doc']}"); continue
        for col, tmpl in spec_.items():
            want, _ = render_value(T["id"], "J", tmpl) if "{" in tmpl else (tmpl, [])
            ncell += 1
            if numform(got[arm].get(col, "")) != numform(want):
                errors.append(f"{T['id']}: {T['doc']} [{arm} / {col}] = «{got[arm].get(col)}», data says «{want}»")
            for other in T.get("also_show", []):
                od = doc(other)
                if od is not None and want not in ("—",) and numform(want) not in numform(od):
                    errors.append(f"{T['id']}: {other} does not show [{arm} / {col}] = {want}")
    table_report.append(f"{T['id']}: {ncell} cells verified in {T['doc'].split('/')[-1]}"
                        + (f" + shown in {', '.join(x.split('/')[-1] for x in T.get('also_show', []))}" if T.get("also_show") else ""))

# ---- control-contrast guard (C-80/C-81) -------------------------------------------------
# Every flat-vector contrast in this project was computed against a control reading 2 context
# pages, until C-77(ii) found the asymmetry and C-80 re-ran the control at k=3. A claim that
# still rests on the k=2 control, with nothing on its face saying so, is exactly the failure this
# ledger exists to catch -- and S-04 was sitting in it. So: any LIVE claim naming the flat-vector
# baseline must say which control it is measured against.
# Naming EITHER control satisfies the guard: the requirement is that the face says which one,
# not that every claim be page-matched. "Stands against the shipped k=2 control, with the
# page-matched value beside it" is an honest claim; silence is not.
_PM = ("page-matched", "k=3", "k=2", "retrieval ranking", "random-page")
for c, _ in rows:
    if c["status"] in ("retracted", "superseded", "not established"):
        continue                                    # already marked; the wording is indexed
    face = " ".join(str(c.get(k, "")) for k in ("claim", "scope", "caveat", "value")).lower()
    named = ("flat-vector" in face) or ("flat vector" in face)
    if not named and c["id"] not in getattr(src, "CONTROL_DEPENDENT", set()):
        continue
    if not any(t in face for t in _PM):
        errors.append(f"{c['id']}: rests on a flat-vector contrast but its face does not say which "
                      f"control (k=2 or page-matched k=3) produced it -- C-80")

STATUS_ORDER = ["retracted", "superseded", "not established", "caveated", "stands"]
counts = {s: sum(1 for c, _ in rows if c["status"] == s) for s in STATUS_ORDER}
esc = lambda t: str(t).replace("|", "\\|").replace("\n", " ")

L = ["# Claims ledger — GraMM-RAG",
     "",
     "**Generated by `ledger/build_ledger.py` from `ledger/claims_source.py`. Do not edit by hand.**",
     "The sync refuses to ship if this file differs from a fresh render, or if any value below fails",
     "its provenance check. Quoted values must appear verbatim in the cited source; values marked `J`",
     "are read live from result JSONs; `docs_must_show` documents must print the same number.",
     "",
     f"**{len(rows)} claims** — " + " · ".join(f"{counts[s]} {s}" for s in STATUS_ORDER),
     "",
     "## How to use this when writing",
     "",
     "**Before stating any claim you remember as true, search this file for its wording.** Retracted",
     "claims are indexed under the words they were **first published** in, not the words that replaced",
     "them, because a writer reaches for the remembered sentence. `ledger/check_draft.py <draft>`",
     "does the search for a whole draft and flags every retracted or superseded wording it finds.",
     "",
     "| status | may be written? |",
     "|---|---|",
     "| **stands** | yes, as worded |",
     "| **caveated** | yes, **only with its caveat** in the same sentence or table |",
     "| **not established** | no — state it as untested or not supported |",
     "| **superseded** | no — use the claim it points to |",
     "| **retracted** | no — and say so if a reader would expect it |",
     ""]
for s in STATUS_ORDER:
    grp = [(c, v) for c, v in rows if c["status"] == s]
    if not grp: continue
    L += [f"## {s.capitalize()} ({len(grp)})", "",
          "| id | claim | value + interval | source | scope | supersedes / superseded by |",
          "|---|---|---|---|---|---|"]
    for c, v in grp:
        claim = esc(c["claim"])
        if c.get("original"):
            claim += "<br>*originally worded:* " + "; ".join(f"“{esc(o)}”" for o in c["original"])
        if c.get("why"):
            claim += f"<br>**why:** {esc(c['why'])}"
        if c.get("caveat"):
            claim += f"<br>**caveat (must accompany):** {esc(c['caveat'])}"
        kind = c["value"][0]
        vcell = f"{esc(v)}" + (" `J`" if kind == "J" else "")
        sdoc, ssec = c["src"]
        scell = f"`{sdoc.split('/')[-1]}` {esc(ssec)}" + (f" · {c['register']}" if c.get("register") else "")
        rel = []
        if c.get("supersedes"): rel.append(f"supersedes {c['supersedes']}")
        if c.get("superseded_by"): rel.append(f"→ **{c['superseded_by']}**")
        L.append(f"| **{c['id']}** | {claim} | {vcell} | {scell} | {esc(c.get('scope',''))} | {'; '.join(rel) or '—'} |")
    L.append("")
L += ["## Not in this ledger", "",
      "**Dissertation v4 was not available on the instance** (2026-09-21); `PHASE_B_DISSERTATION.md` is",
      "v1.0, a plan. Run `ledger/check_draft.py` against v4 to flag any retracted wording it still uses.",
      ""]
rendered = "\n".join(L)

if errors:
    print("LEDGER BUILD FAILED -- provenance checks:")
    for e in errors: print("  -", e)
    sys.exit(1)
if "--check" in sys.argv:
    cur = OUT.read_text() if OUT.exists() else ""
    if cur != rendered:
        print("LEDGER DRIFT: CLAIMS_LEDGER.md differs from a fresh render -- rerun build_ledger.py"); sys.exit(1)
    print(f"ledger ok: {len(rows)} claims, every value verified against its source; " + "; ".join(table_report)); sys.exit(0)
OUT.write_text(rendered)
print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} claims -- " + ", ".join(f"{counts[s]} {s}" for s in STATUS_ORDER))
