# Reproducing the GraMM-RAG Phase C evidence from this package

**Verified 2026-09-21:** the evidence notebook regenerates from this package alone — no access to the
machine it was produced on — with **52 code cells, 0 errors, and 0 cells whose output differs** from the
original run. Re-verify at any time with `verify_coldstart.sh`.

## What "reproducible" means here, and what it does not

This package reproduces the **analysis layer**: every table, contrast, bootstrap CI, figure and
verdict in `phaseC_evidence.ipynb`, from the per-question result files it ships. It does **not**
ship the bulk data those results were computed from, and so does not re-run retrieval,
indexing or generation.

| layer | in package? | to regenerate it |
|---|---|---|
| per-question results, scores, contrasts | **yes** | shipped |
| notebook + every figure | **yes** — regenerated from the above | `verify_coldstart.sh` |
| graph stores (LightRAG / RAG-Anything / GraphRAG) | no — tens of GB | re-index from the parse; ~$62 of API |
| MinerU parses, page images | no — bulk data | re-parse from the source PDFs |
| generator / judge API calls | no | re-run against OpenAI; costs recorded per arm |

Values that need excluded data are **precomputed** on the full instance by
`gramm-rag-colpali/precompute_package_artifacts.py` and shipped in
`scores/package_precomputed.json`. The notebook reads them and **raises** if one is missing; it
never recomputes over absent inputs.

## Why the verification compares outputs rather than counting errors

The first cold-start run reported **0 errors** and was wrong in three cells. Two were silent
fallbacks to plausible numbers:

- **Index sizes** read **0.00 GiB** for every arm: they were measured by walking store
  directories the package excludes, and an empty directory sums to zero without raising.
- **LightRAG's DocBench quote-recall** read **0.0000** instead of **0.5068**, because a missing
  parse was silently replaced with an empty document. That reversed pre-registered **clause 1**
  from **FALSIFIED** to **HELD**, at p = 4.6e-100.

A reader regenerating from the package would have received the opposite scientific conclusion,
stated with near-certainty, and no error. Only comparing outputs catches that, so that is the
test.

## Running it

```bash
./verify_coldstart.sh [path-to-package]   # default: the directory the script is in
```

Requires Docker and the `grammrag-analysis` image (its Dockerfile is in `_analysis_env/`). Exit
status 0 means the package reproduces the evidence exactly.

## Secrets

This package carries **no credentials**. The build that produces it refuses to emit any file or
string shaped like a secret.
