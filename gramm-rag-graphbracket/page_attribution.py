"""Canonical page attribution, extracted VERBATIM from retrieve_lightrag.py.

S5 runs in a Python 3.11 container with graphrag installed; retrieve_lightrag.py imports
lightrag and cannot be imported there, so the shared logic lives here instead of being
rewritten -- a second implementation would diverge silently, which is the D-32 class.

Provenance: copied from retrieve_lightrag.py, functions `_norm`, `build_shingle_index`,
`attribute_pages`. md5 of the extracted block at extraction time: d908479711d62cd1d4bd5e3457b03fd0
Ranking semantics are C-37's: pages rank by the retriever's OWN order, character mass breaks
ties only. The image route (C-49) is retained but inert for S5, which retrieves no images.
"""
import re
from collections import defaultdict

SHINGLE = 50
PROBE_STRIDE = 50
INDEX_STRIDE = 1
IMG_PATH_RE = re.compile(r"Image Path:\s*(\S+)")


def _norm(s: str) -> str:
    """Whitespace-insensitive, case-insensitive view for matching."""
    return re.sub(r"\s+", " ", s).strip().lower()


SHINGLE = 50          # characters per probe
INDEX_STRIDE = 1      # element side: EVERY offset -- see below
PROBE_STRIDE = 50     # chunk side: non-overlapping, so each hit accounts for 50 chars


def build_shingle_index(parsed: dict):
    """Index the FULL ingested text at stride 1, mapping each offset to its page.

    Three attempts, recorded because the differences are instructive and the number is
    reported rather than assumed:

      1. whole-element substring match -> 0.418, but over-credits: a matched element is
         credited its entire length, and an element appearing in two chunks counts twice.
      2. 50-char shingles, stride 50 on BOTH sides -> 0.027. Two non-overlapping griddings
         coincide only when their offsets agree modulo 50; that is an artefact of grid
         alignment, not a property of the retrieval.
      3. stride 1 on the element side -> 0.325, honest but still floored: elements shorter
         than 50 characters were skipped entirely, so their characters could never match
         however faithfully they were returned.

    This version indexes the exact string that was ingested -- `mapper.elements_to_text`
    output, headings and all -- so every ingested character is eligible, and carries a
    parallel offset->page map so attribution survives element boundaries.
    """
    import bisect
    order = sorted(parsed["elements"], key=lambda x: x.get("reading_order", 0))
    parts, starts, pages, pos = [], [], [], 0
    for el in order:
        raw = (el.get("text") or "").strip()
        if not raw:
            continue
        rendered = f"## {raw}" if el.get("type") == "section" else raw
        norm = _norm(rendered)
        if not norm:
            continue
        starts.append(pos); pages.append(int(el.get("page_no", 0) or 0))
        parts.append(norm); pos += len(norm) + 1          # +1 for the joining space
    joined = " ".join(parts)
    idx = {}
    for k in range(0, len(joined) - SHINGLE + 1, INDEX_STRIDE):
        h = hash(joined[k:k + SHINGLE])
        if h not in idx:
            j = bisect.bisect_right(starts, k) - 1
            idx[h] = pages[j] if j >= 0 else 0
    return idx


def attribute_pages(chunks: list[str], shingles: dict, img_pages: dict | None = None):
    """Map returned chunk text back to pages via shingle lookup.

    Returns (pages ranked by attributed character mass, alignment_rate, chunk chars).
    alignment_rate = attributed characters / returned characters. Probes are
    non-overlapping, so the rate is a genuine coverage fraction bounded by 1.
    """
    mass, best_rank, matched, total = defaultdict(int), {}, 0, 0
    img_matched, img_chunks = 0, 0
    for ci, c in enumerate(chunks):
        cn = _norm(c)
        total += len(cn)
        # ROUTE 2 -- multimodal chunks, attributed by exact image path, not by text.
        # Probes are non-overlapping (PROBE_STRIDE == SHINGLE), so a fully-attributed
        # text chunk accrues exactly len(cn) mass; giving an image chunk len(cn) puts
        # both routes on the SAME mass scale, so ranking stays comparable.
        mp = IMG_PATH_RE.search(c) if img_pages else None
        if mp and mp.group(1) in img_pages:
            pg = img_pages[mp.group(1)]
            mass[pg] += len(cn)
            matched += len(cn)
            img_matched += len(cn)
            img_chunks += 1
            best_rank.setdefault(pg, ci)
            continue
        for k in range(0, max(len(cn) - SHINGLE + 1, 0), PROBE_STRIDE):
            pg = shingles.get(hash(cn[k:k + SHINGLE]))
            if pg:
                mass[pg] += SHINGLE
                matched += SHINGLE
                if pg not in best_rank:
                    best_rank[pg] = ci          # LightRAG's OWN relevance order
    rate = (matched / total) if total else 0.0
    img_rate = (img_matched / total) if total else 0.0
    # Rank pages by the retriever's ordering, NOT by how much text they happen to hold.
    # LightRAG returns Document Chunks in relevance order, so a page's rank is the rank of
    # the best chunk that maps to it; character mass breaks ties only.
    # Ranking by mass alone (the first implementation) discarded the retriever's ranking
    # entirely and measured "which pages carry the most text among those LightRAG touched"
    # -- a different quantity, and one no other arm is scored on. ColPali ranks by MaxSim,
    # GraMM by graph score; both are relevance. See C-37.
    ranked = [pg for pg, _ in sorted(mass.items(),
                                     key=lambda kv: (best_rank.get(kv[0], 10**6), -kv[1]))]
    return (ranked, round(min(rate, 1.0), 4), total,
            round(min(img_rate, 1.0), 4), img_chunks)
