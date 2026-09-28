#!/usr/bin/env python3
"""P3 -- verbatim extraction of `parse_answer` (POSTPROC_V=2) from
gramm-rag-mpdocvqa/gen_notebook.py.

Extracted rather than imported for the same reason page_attribution.py was: the canonical
copy lives inside a notebook-generator string literal and cannot be imported. Extraction is
therefore the only way to share it -- but a COPY can drift from its source, so the md5 of
the extracted block is stamped here and re-verified by p3_score.py on every run.

Why this matters: p3_score.py originally reused the canonical SCORER without this step, and
every arm scored ANLS 0.001 with 99.8% exact zeros. anls_score is edit-distance with a hard
0.5 threshold, so raw generator output ("EVIDENCE: ...\nANSWER: 7.5") against a 3-character
gold exceeds the threshold and returns 0. Matching the POSTPROC_V=2 config stamp is not the
same as running the code that stamp names.

md5 of the extracted block at extraction time: 480e36a2cf636c47ce03eda41d58b831
"""
import re


def parse_answer(raw):
    """Extract the value from the quote-then-answer (EVIDENCE/ANSWER)
    response format, then canonicalise the span for ANLS (POSTPROC_V=2).

    ANLS is edit-distance-based with a 0.5 hard threshold: every leaked
    label token, lead-in phrase, or wrapper character inflates distance
    and can zero an otherwise-correct answer. The metric itself already
    lowercases and strips punctuation, so only token-level cleanup that
    the metric cannot do belongs here."""
    if not isinstance(raw, str):
        return raw
    if 'ANSWER:' in raw:
        tail = raw.rsplit('ANSWER:', 1)[-1].strip()
        a = tail.splitlines()[0].strip() if tail else raw.strip()
    else:
        a = raw.strip()
    # -- POSTPROC v2: ANLS-aware span canonicalisation ---------------
    for _pre in ('the answer is', 'answer is', 'the answer:', 'answer:'):
        if a.lower().startswith(_pre):
            a = a[len(_pre):].strip()
    a = a.strip('\'"` ').strip()
    if a.startswith('(') and a.endswith(')'):
        a = a[1:-1].strip()
    # trailing sentence period (keep decimals like '3.' -> rare; and
    # abbreviations like 'Inc.' stay because gold spans include them
    # inconsistently -- the metric strips punctuation anyway)
    if a.endswith('.') and not re.search(r'\d\.$', a):
        a = a[:-1].strip()
    a = re.sub(r'\s+', ' ', a)
    return a


EXTRACTED_MD5 = "480e36a2cf636c47ce03eda41d58b831"
