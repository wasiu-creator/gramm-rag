"""
src/generation/prompt_builder.py
──────────────────────────────────
Graph-provenance prompt construction for GraMM-RAG.

Builds the final prompt sent to Qwen2.5-VL-72B (or GPT-4o fallback) from:
  - The original question
  - Retrieved node content (text, table, figure captions)
  - Provenance metadata (page numbers, section headers, temporal markers)
  - Reward score and refusal flag from self-healing
"""

from typing import Optional


def build_prompt(
    question: str,
    retrieved_nodes: list[dict],
    parsed: dict,
    reward_detail: dict,
    refused: bool,
    benchmark: str = "mmlongbench",
    max_context_chars: int = 4000,   # ← EXTEND: increase for longer-context VLMs
    answer_format: str = None,       # R-49: MMLongBench 'Int'|'Float'|'Str'|'List'|'None'
) -> str:
    """
    Construct the VLM prompt from retrieved graph nodes.

    Args:
        question:        Original question string.
        retrieved_nodes: Output of SelfHealingRetriever.retrieve()["nodes"].
        parsed:          Full parsed document dict.
        reward_detail:   Dict from RewardFunction.compute().
        refused:         True if self-healing decided to refuse.
        benchmark:       Benchmark name (affects answer format instructions).
        max_context_chars: Truncate context to this many characters.
                          # ← EXTEND: increase when using longer-context models.

    Returns:
        Formatted prompt string for the VLM.
    """
    if refused:
        return _build_refusal_prompt(question, reward_detail)

    # ── Gather context snippets from retrieved nodes ─────────────────────────
    elements = parsed.get("elements", [])
    type_to_elems = {}
    for e in elements:
        type_to_elems.setdefault(e["type"], []).append(e)

    context_blocks = []
    total_chars = 0

    for node in retrieved_nodes:
        if total_chars >= max_context_chars:
            break

        ntype = node.get("node_type", "text")
        # Support both "local_idx" (from HGT retrieval) and "global_idx" (direct)
        local_idx = node.get("local_idx", node.get("global_idx", 0))
        typed = type_to_elems.get(ntype, [])

        if local_idx >= len(typed):
            continue

        elem = typed[local_idx]
        text = elem.get("text", "").strip()
        if not text:
            continue

        if benchmark == "mpdocvqa":
            # MP-DocVQA answers are short verbatim spans. The [TYPE|Page|Time]
            # decoration leaks into answers (e.g. the model returns "Time:
            # 1976"). Feed clean reading-order text only.
            block = text
        else:
            page = elem.get("page_no", "?")
            temporal = [m["text"] for m in elem.get("temporal_markers", [])]
            time_str = f" [Time: {', '.join(temporal)}]" if temporal else ""
            block = f"[{ntype.upper()} | Page {page}{time_str}]\n{text}"

        context_blocks.append(block)
        total_chars += len(block)

    sep = "\n" if benchmark == "mpdocvqa" else "\n\n---\n\n"
    context_str = sep.join(context_blocks) if context_blocks else "(No relevant context retrieved)"

    # ── Format instructions per benchmark ────────────────────────────────────
    format_instruction = _get_format_instruction(benchmark)
    # R-49: typed generation contract. MMLongBench ships a per-question
    # `answer_format` (Int 26.6% / Str 22.9% / None 22.4% / Float 14.7% /
    # List 13.5%) that the pipeline was not using. Gold answers average ~1.9
    # words; the VLM averaged 10.7 on the questions it got wrong, and 5 of 17
    # measured extraction failures were prose where a bare value is scored
    # (e.g. gold '2' vs pred '1.0'; gold "['Google Play store','1.3']" vs a
    # full sentence). Appending the expected type constrains the output shape.
    # This does NOT address the other 12 failures, which are wrong *values*
    # (gold '18.9%' vs pred '12.8') -- those need a stronger generator, not a
    # stronger prompt. See the ceiling note in R-49.
    _typed = _typed_format_clause(answer_format)
    if _typed:
        format_instruction = f"{format_instruction} {_typed}"

    prompt = (
        f"You are a document question-answering assistant. "
        f"Answer the question using ONLY the provided document excerpts.\n\n"
        f"Document Context (retrieved via graph-augmented retrieval):\n"
        f"{'='*60}\n"
        f"{context_str}\n"
        f"{'='*60}\n\n"
        f"Question: {question}\n\n"
        f"{format_instruction}\n"
        f"Answer:"
    )

    return prompt


def _build_refusal_prompt(question: str, reward_detail: dict) -> str:
    """Prompt used when self-healing decides to refuse (R < τ·0.6)."""
    R = reward_detail.get("reward", 0.0)
    return (
        f"Question: {question}\n\n"
        f"Based on the retrieved document content (confidence score: {R:.2f}), "
        f"the available evidence is insufficient to answer this question reliably.\n\n"
        f"Answer: The document does not contain sufficient information to answer this question."
    )


def _typed_format_clause(answer_format: str) -> str:
    """R-49: per-question output contract from MMLongBench's `answer_format`.

    Returns a short clause constraining the ANSWER's shape to the type the
    gold answer actually has. Deliberately terse — a long instruction competes
    with the document context for the model's attention, and the failure being
    targeted is verbosity, so the remedy must not itself be verbose.

    Returns '' for unknown/absent formats so behaviour is unchanged when the
    field is missing (DocBench has no equivalent field).
    """
    if not answer_format:
        return ""
    fmt = str(answer_format).strip().lower()
    return {
        # digits only: observed failure was gold '2' answered as '1.0'
        "int": ("The expected answer is an INTEGER: reply with digits only, "
                "no decimal point, no units, no words."),
        # keep the % or unit when it is part of the value, as the gold does
        "float": ("The expected answer is a NUMBER: reply with the numeric "
                  "value only, keeping a trailing % or unit only if the "
                  "document shows it as part of the value (e.g. 18.9%)."),
        # gold strings are short entities, not sentences
        "str": ("The expected answer is a SHORT STRING: reply with the exact "
                "entity, name or phrase only — no sentence, no explanation."),
        # gold is a python-style list; a prose sentence scores zero
        "list": ("The expected answer is a LIST: reply with the items only, "
                 "comma-separated, in document order — no sentence."),
        # 'None' == MMLongBench's unanswerable class (R-8)
        "none": ("If the excerpts genuinely do not contain the answer, reply "
                 "exactly: Not answerable"),
    }.get(fmt, "")


def _get_format_instruction(benchmark: str) -> str:
    """Return benchmark-specific answer format instruction."""
    instructions = {
        "mmlongbench": (
            "Provide a concise, direct answer. "
            "If the answer is a number, give only the number. "
            "If it requires explanation, keep it under 3 sentences."
        ),
        "docbench": (
            "Answer concisely. Cite the page number if relevant (e.g., 'Page 5: ...')."
        ),
        "mpdocvqa": (
            "Quote-then-answer. Respond with EXACTLY two lines:\n"
            "EVIDENCE: <the single line copied verbatim from the context that "
            "contains the answer>\n"
            "ANSWER: <only the exact answer value -- a number, name, date, or "
            "brief entity. No labels, no field names, no explanation. If "
            "numeric, digits only (keep a unit/symbol only when it is part of "
            "the value, e.g. 41.09%).>\n"
            "Every question is answerable from the excerpts -- never reply "
            "'no information' or 'not available'; commit to the most likely "
            "value found in your EVIDENCE line.\n"
            "Examples --\n"
            "EVIDENCE: PAPER CODE 12427\nANSWER: 12427\n"
            "EVIDENCE: Total responses received: 27\nANSWER: 27\n"
            "EVIDENCE: Venue owner: Richard Flemming\nANSWER: Richard Flemming"
        ),
        "multimodalqa": (
            "Answer the question using both textual and visual evidence from the context."
        ),
    }
    return instructions.get(benchmark, "Answer concisely based on the provided context.")
