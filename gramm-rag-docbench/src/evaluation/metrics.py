"""
src/evaluation/metrics.py
──────────────────────────
Evaluation metrics for GraMM-RAG (Step 14).

Implements:
  - ANLS  (Average Normalised Levenshtein Similarity) — DocVQA standard
  - F1    (token-level F1, SQuAD-style)
  - Accuracy (exact match after normalisation)
  - Abstention Precision/Recall (unanswerable-question detection)
  - APPA  (Answer-Page Prediction Accuracy) — MP-DocVQA retrieval localisation:
           did retrieval surface the page that actually holds the answer?
  - Quote-Recall@k — DocBench retrieval localisation. DocBench has no page-level
           evidence, only a free-text `evidence` quote, so APPA is impossible;
           instead we check whether the top-k retrieved pages' text contains the
           gold evidence quote (`compute_quote_recall`).
  - GPT-4 judge accuracy — DocBench's official metric (`compute_judge_accuracy`,
           faithful port of Anni-Zou/DocBench evaluate.py): a GPT-4 judge scores
           each answer 0/1; accuracy reported overall and per question-type.

  NOTE: `compute_appa()` (abstention) and `compute_answer_page_accuracy()`
  (the proposal's APPA) are DIFFERENT metrics that historically shared the
  "APPA" acronym. They are kept distinct here on purpose.

All metrics operate on lists of (prediction, gold_answer) pairs.
"""

import logging
import re
import string
from collections import Counter
from typing import Union

logger = logging.getLogger(__name__)


# ─── Text normalisation ────────────────────────────────────────────────────────

def normalise_answer(s: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ─── Levenshtein distance (for ANLS) ──────────────────────────────────────────

def levenshtein(s1: str, s2: str) -> int:
    """Standard dynamic programming Levenshtein distance."""
    if s1 == s2:
        return 0
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def anls_score(prediction: str, gold: str, threshold: float = 0.5) -> float:
    """
    ANLS for a single prediction/gold pair.
    Score = max(0, 1 - NLS) if NLS < threshold, else 0.
    NLS = edit_distance / max(len(pred), len(gold)).
    """
    pred = normalise_answer(prediction)
    g = normalise_answer(gold)
    if not pred and not g:
        return 1.0
    if not pred or not g:
        return 0.0
    dist = levenshtein(pred, g)
    nls = dist / max(len(pred), len(g))
    return max(0.0, 1.0 - nls) if nls < threshold else 0.0


def compute_anls(
    predictions: list[str],
    gold_answers: list[Union[str, list[str]]],
    threshold: float = 0.5,
) -> float:
    """
    ANLS averaged over all examples.
    gold_answers may be a list of strings (multiple valid answers).
    """
    assert len(predictions) == len(gold_answers), "Length mismatch"
    scores = []
    for pred, gold in zip(predictions, gold_answers):
        if isinstance(gold, list):
            score = max(anls_score(pred, g, threshold) for g in gold)
        else:
            score = anls_score(pred, gold, threshold)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


# ─── Token-level F1 ───────────────────────────────────────────────────────────

def token_f1(prediction: str, gold: str) -> float:
    """SQuAD-style token-level F1 for a single pair."""
    pred_tokens = normalise_answer(prediction).split()
    gold_tokens = normalise_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1(
    predictions: list[str],
    gold_answers: list[Union[str, list[str]]],
) -> float:
    """Macro-averaged token F1 over all examples."""
    assert len(predictions) == len(gold_answers)
    scores = []
    for pred, gold in zip(predictions, gold_answers):
        if isinstance(gold, list):
            score = max(token_f1(pred, g) for g in gold)
        else:
            score = token_f1(pred, gold)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


# ─── Exact match accuracy ──────────────────────────────────────────────────────

def compute_accuracy(
    predictions: list[str],
    gold_answers: list[Union[str, list[str]]],
) -> float:
    """Exact match accuracy after normalisation."""
    assert len(predictions) == len(gold_answers)
    correct = 0
    for pred, gold in zip(predictions, gold_answers):
        pred_n = normalise_answer(pred)
        if isinstance(gold, list):
            correct += int(any(pred_n == normalise_answer(g) for g in gold))
        else:
            correct += int(pred_n == normalise_answer(gold))
    return correct / len(predictions) if predictions else 0.0


# ─── APPA: Abstention Precision & Recall ──────────────────────────────────────

def compute_appa(
    predictions: list[str],
    is_answerable: list[bool],
    refused_flags: list[bool],
) -> dict:
    """
    APPA (Abstention Precision / Recall) for unanswerable question detection.

    Args:
        predictions:   Model answer strings.
        is_answerable: True if the question has a ground-truth answer.
        refused_flags: True if the model refused to answer (R < τ).

    Returns:
        Dict with keys: abstention_precision, abstention_recall, abstention_f1,
                        overall_accuracy (answerable only).
    """
    assert len(predictions) == len(is_answerable) == len(refused_flags)

    # Abstention = model said "I don't know" (refused=True)
    # Unanswerable = ground truth has no answer (is_answerable=False)
    tp = sum(1 for a, r in zip(is_answerable, refused_flags) if not a and r)
    fp = sum(1 for a, r in zip(is_answerable, refused_flags) if a and r)
    fn = sum(1 for a, r in zip(is_answerable, refused_flags) if not a and not r)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "abstention_precision": precision,
        "abstention_recall":    recall,
        "abstention_f1":        f1,
        "abstention_tp":        tp,
        "abstention_fp":        fp,
        "abstention_fn":        fn,
    }


# ─── APPA: Answer-Page Prediction Accuracy (retrieval localisation) ───────────
# Distinct from compute_appa() above (abstention). This is the proposal's
# MP-DocVQA metric: of the top-k retrieved nodes, do any sit on the page
# that actually contains the gold answer?

def compute_answer_page_accuracy(
    retrieved_pages: list,
    gold_pages: list,
    k_values: tuple = (1, 5, 10),
) -> dict:
    """
    APPA — Answer-Page Prediction Accuracy.

    Args:
        retrieved_pages: list (len = n questions). Element i is an ordered
                         list of page numbers for that question's retrieved
                         nodes, ranked by retrieval score (index 0 = top node).
        gold_pages:      list (len = n questions). Element i is the gold
                         answer page (int), or a list/set of acceptable pages.
        k_values:        cut-offs to report success@k at.

    Returns:
        {'appa@1':.., 'appa@5':.., 'appa@10':.., 'appa':..,
         'n_appa_scoreable':..}
        where 'appa' aliases the largest-k value (full retrieved set).

    Scoreable denominator (C-1). Only questions carrying at least one annotated
    gold page are scoreable: a question with no gold page can never register a
    hit, so including it adds a guaranteed miss to the denominator. On
    MMLongBench 246/1,091 (22.5%) of questions have empty `evidence_pages`
    (the unanswerable set), which uniformly deflated every reported APPA by
    ~22.5% — a true 0.50 was reported as 0.380.

    This deflation scaled both arms equally, so the GraMM-vs-baseline direction
    and the significance verdict were unaffected; what it broke was (a) the
    absolute values and (b) cross-benchmark comparability, since MP-DocVQA's
    anchor APPA was computed where every question has a gold page. The Phase-10
    conversion funnel already excluded non-scoreable questions, so this also
    removes an internal inconsistency between the funnel and the APPA table.

    `n_appa_scoreable` is returned so the denominator is always explicit in the
    reported artefacts.
    """
    assert len(retrieved_pages) == len(gold_pages), "Length mismatch"

    def _gold_set(gold):
        if isinstance(gold, (list, set, tuple)):
            return {g for g in gold if g is not None}
        return {gold} if gold is not None else set()

    # Keep only questions with at least one annotated gold page.
    scoreable = [(pages, _gold_set(gold))
                 for pages, gold in zip(retrieved_pages, gold_pages)
                 if _gold_set(gold)]

    n = len(scoreable)
    if n == 0:
        out = {f"appa@{k}": 0.0 for k in k_values}
        out["appa"] = 0.0
        out["n_appa_scoreable"] = 0
        return out

    out = {}
    for k in k_values:
        hits = 0
        for pages, gold_set in scoreable:
            if any(p in gold_set for p in list(pages)[:k]):
                hits += 1
        out[f"appa@{k}"] = hits / n
    out["appa"] = out[f"appa@{max(k_values)}"]
    out["n_appa_scoreable"] = n
    return out


# ─── DocBench: evidence-quote retrieval recall (APPA-analog) ──────────────────
# DocBench provides NO page-level evidence (only a free-text `evidence` quote),
# so MP-DocVQA/MMLongBench-style answer-page APPA cannot be computed. Instead we
# measure whether retrieval surfaced the page(s) whose text contains the gold
# evidence quote. This is the deterministic retrieval-localisation signal (the
# dissertation's most defensible "GraMM beats baseline" claim on DocBench).


def _quote_tokens(s: str) -> set:
    """Content tokens (len>2, de-punctuated, lower-cased) of an evidence quote."""
    return {t for t in normalise_answer(s or "").split() if len(t) > 2}


def evidence_page_hit(page_texts: list, evidence: str, min_overlap: float = 0.6) -> bool:
    """
    True if any single page's text covers >= `min_overlap` of the evidence quote's
    content tokens. `page_texts` is the ordered list of concatenated page texts
    for the retrieved pages (index 0 = top-ranked page).
    """
    ev = _quote_tokens(evidence)
    if not ev:
        return False
    for pt in page_texts:
        pt_tokens = _quote_tokens(pt)
        if not pt_tokens:
            continue
        overlap = len(ev & pt_tokens) / len(ev)
        if overlap >= min_overlap:
            return True
    return False


def compute_quote_recall(
    retrieved_page_texts: list,
    evidences: list,
    is_answerable: list = None,
    k_values: tuple = (1, 3, 5),
    min_overlap: float = 0.6,
) -> dict:
    """
    Quote-Recall@k for DocBench — did the top-k retrieved pages contain the gold
    evidence quote?

    Args:
        retrieved_page_texts: list (len = n questions). Element i is an ORDERED
                              list of page-text strings (top page first) for
                              question i.
        evidences:            list of gold evidence quote strings (may be empty
                              for meta-data / unanswerable questions).
        is_answerable:        optional list[bool]; if given, questions with no
                              usable evidence quote are excluded from the denom.
        k_values:             recall cut-offs.
        min_overlap:          fraction of evidence tokens a page must cover.

    Returns:
        {'quote_recall@1':.., 'quote_recall@3':.., 'quote_recall@5':..,
         'quote_recall':.. (largest k), 'quote_recall_n': #scored questions}
    """
    assert len(retrieved_page_texts) == len(evidences), "Length mismatch"
    # Only questions that HAVE a non-trivial evidence quote are scoreable.
    scoreable = [i for i, ev in enumerate(evidences) if _quote_tokens(ev)]
    if is_answerable is not None:
        scoreable = [i for i in scoreable if is_answerable[i]]
    out = {}
    n = len(scoreable)
    for k in k_values:
        if n == 0:
            out[f"quote_recall@{k}"] = 0.0
            continue
        hits = sum(
            1 for i in scoreable
            if evidence_page_hit(retrieved_page_texts[i][:k], evidences[i], min_overlap)
        )
        out[f"quote_recall@{k}"] = hits / n
    out["quote_recall"] = out.get(f"quote_recall@{max(k_values)}", 0.0)
    out["quote_recall_n"] = n
    return out


# ─── DocBench: GPT-4 LLM-judge accuracy (official protocol) ───────────────────
# Faithful port of Anni-Zou/DocBench evaluate.py + evaluation_prompt.txt:
# a GPT-4 judge scores each answer 0/1 given question, system answer, reference
# answer, and reference evidence text. Accuracy is the mean, reported overall
# and per question-type.

DOCBENCH_JUDGE_PROMPT = """Task Overview:
You are tasked with evaluating user answers based on a given question, reference answer, and additional reference text. Your goal is to assess the correctness of the user answer using a specific metric.

Evaluation Criteria:
1. Yes/No Questions: Verify if the user's answer aligns with the reference answer in terms of a "yes" or "no" response.
2. Short Answers/Directives: Ensure key details such as numbers, specific nouns/verbs, and dates match those in the reference answer.
3. Abstractive/Long Answers: The user's answer can differ in wording but must convey the same meaning and contain the same key information as the reference answer to be considered correct.

Evaluation Process:
1. Identify the type of question presented.
2. Apply the relevant criteria from the Evaluation Criteria.
3. Compare the user's answer against the reference answer accordingly.
4. Consult the reference text for clarification when needed.
5. Score the answer with a binary label 0 or 1, where 0 denotes wrong and 1 denotes correct.
NOTE that if the user answer is 0 or an empty string, it should get a 0 score.

Question: {question}
User Answer: {sys_ans}
Reference Answer: {ref_ans}
Reference Text: {ref_text}

Evaluation Form (score ONLY):
- Correctness: """


def _parse_judge_score(response: str) -> int:
    """Extract the binary 0/1 verdict from the judge's free-text response."""
    if not response:
        return 0
    # First standalone digit that is 0 or 1 wins (prompt asks for score only).
    m = re.search(r"[01]", response)
    return int(m.group()) if m else 0


def judge_answer_gpt4(
    question: str,
    sys_ans: str,
    ref_ans: str,
    ref_text: str,
    client=None,
    model: str = "gpt-4-turbo",
) -> int:
    """
    Score a single DocBench answer with the GPT-4 judge. Returns 0 or 1.

    `client` is an openai.OpenAI instance (created by the caller so the key is
    only read once). If None, or on API failure, returns 0 (conservative —
    matches DocBench treating unscoreable/empty answers as wrong).
    """
    sys_ans = (sys_ans or "").strip()
    if not sys_ans or sys_ans == "0" or sys_ans == "[GENERATION_ERROR]":
        return 0
    if client is None:
        return 0
    prompt = DOCBENCH_JUDGE_PROMPT.format(
        question=question, sys_ans=sys_ans,
        ref_ans=ref_ans or "", ref_text=ref_text or "",
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=8,
        )
        return _parse_judge_score(resp.choices[0].message.content)
    except Exception as e:
        # D-19: this except used to swallow everything and return 0, which is
        # indistinguishable from "the judge scored this answer wrong". A dead
        # model name (gpt-4-0125-preview now 404s on this account) therefore
        # produced a confident judge_accuracy of 0.000 over the whole run.
        # Re-raise the errors that mean the judge itself is broken -- a wrong
        # model, a bad key, no quota -- so the run stops instead of reporting a
        # fabricated score. Transient per-call failures still return 0.
        name = type(e).__name__
        if name in ("NotFoundError", "AuthenticationError", "PermissionDeniedError"):
            raise RuntimeError(
                f"GPT-4 judge is unusable ({name}: {str(e)[:200]}). "
                f"Model '{model}' — fix JUDGE_MODEL or the API key. Refusing to "
                f"report a judge accuracy computed from failed calls."
            ) from e
        # D-34: RateLimitError covers BOTH transient throttling (retryable) and
        # permanent credit exhaustion (fatal). The D-19 fix listed "no quota" as
        # a re-raise case but could not catch it, because OpenAI reports an empty
        # balance as 429/RateLimitError rather than a distinct class. On the first
        # DocBench full run the account ran dry mid-judging: 342 of 1,102 calls
        # were scored 0 by this handler, and the run reported judge_accuracy
        # 0.3167 when the true figure on genuinely-judged questions was 0.4592.
        # Distinguish the two by message: an exhausted balance never recovers, so
        # continuing only manufactures zeros.
        if name == "RateLimitError":
            msg = str(e).lower()
            if any(s in msg for s in ("no credits", "insufficient_quota",
                                      "exceeded your current quota", "billing")):
                raise RuntimeError(
                    f"GPT-4 judge is out of credits ({str(e)[:200]}). Refusing to "
                    f"score the remaining questions 0 — that would understate "
                    f"judge accuracy by conflating API failures with wrong answers. "
                    f"Add credits and re-run; generation results are cached."
                ) from e
        logger.warning(f"Judge call failed ({name}), scoring 0: {str(e)[:150]}")
        return 0


def compute_judge_accuracy(
    predictions: list,
    questions: list,
    gold_answers: list,
    evidences: list,
    types: list = None,
    client=None,
    model: str = "gpt-4-turbo",
    progress_every: int = 0,
) -> dict:
    """
    Official DocBench accuracy: mean GPT-4 binary verdict, overall + per type.

    Returns {'judge_accuracy':.., 'judge_scores':[0/1,...], 'per_type':{type:acc},
             'judge_n':N}. If `client` is None every score is 0 (call is skipped
             upstream in that case).
    """
    n = len(predictions)
    scores = []
    for i in range(n):
        s = judge_answer_gpt4(
            questions[i], predictions[i], gold_answers[i],
            evidences[i] if evidences else "", client=client, model=model,
        )
        scores.append(s)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    judge [{i+1}/{n}] running acc="
                  f"{sum(scores)/len(scores):.3f}")
    out = {
        "judge_accuracy": sum(scores) / n if n else 0.0,
        "judge_scores": scores,
        "judge_n": n,
    }
    if types is not None:
        per_type = {}
        for t in sorted(set(types)):
            idx = [i for i, tt in enumerate(types) if tt == t]
            per_type[t] = (sum(scores[i] for i in idx) / len(idx)) if idx else 0.0
        out["per_type"] = per_type
    return out


# ─── Consolidated metric report ───────────────────────────────────────────────

def compute_all_metrics(
    predictions: list[str],
    gold_answers: list[Union[str, list[str]]],
    is_answerable: list[bool],
    refused_flags: list[bool],
    retrieved_pages: list = None,
    gold_pages: list = None,
) -> dict:
    """
    Run all metrics and return a consolidated results dict.

    If `retrieved_pages` and `gold_pages` are supplied, APPA (Answer-Page
    Prediction Accuracy) is included. Omitting them keeps the call
    backward-compatible with callers that only have answer-level outputs.
    """
    result = {
        "anls":  compute_anls(predictions, gold_answers),
        "f1":    compute_f1(predictions, gold_answers),
        "accuracy": compute_accuracy(predictions, gold_answers),
        **compute_appa(predictions, is_answerable, refused_flags),
        "n_total":       len(predictions),
        "n_answerable":  sum(is_answerable),
        "n_unanswerable": sum(not a for a in is_answerable),
        "n_refused":      sum(refused_flags),
    }
    if retrieved_pages is not None and gold_pages is not None:
        result.update(
            compute_answer_page_accuracy(retrieved_pages, gold_pages))
    return result
