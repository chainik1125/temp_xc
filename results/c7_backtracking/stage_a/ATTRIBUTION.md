---
title: Stage A artifacts — attribution + provenance
component: c7
status: read-only
ported_by: [pipeline]
ported_on: 2026-05-03
source_branch: origin/case-backtracking
source_commit: a62175ee7e99528fe833ce45e62255103bb2bac5
source_path: results/ward_backtracking/
---

These six files are **read-only inputs** to the C7 (Ward Stage B
backtracking) pipeline, ported once with attribution per PROTOCOL.md
§ 2 (wasteland boundary). Do not regenerate from this directory; if
upstream changes, re-port via `git show`.

## Files

| File | Bytes | Description |
|---|---:|---|
| `prompts.json` | 115,246 | 300 reasoning prompts across 10 categories (basic_logic, geometry, probability, arithmetic, counting, number_theory, set_theory, sequences, inequalities, algebra_word_problems — 30 each). 280 prompts in `split=dom` (used to build direction-of-meaning vectors), 20 in `split=eval`. |
| `traces.json` | 3,616,858 | 300 reasoning traces from `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, one per prompt. Fields: `question_id`, `category`, `prompt`, `full_response`, `thinking_process`, `answer`, `answer_index`. |
| `sentence_labels.json` | 4,904,727 | Per-trace sentence segmentation + Sonnet 4.6 backtracking labels. List of 300 dicts with `question_id`, `trace_idx`, `sentences` (per-sentence label list). |
| `dom_vectors.pt` | 264,901 | Direction-of-meaning vectors (PyTorch dict). Keys: `base` (Llama-3.1-8B BASE residual averages), `reasoning` (R1-Distill-Llama equivalents), `meta` (provenance). |
| `steering_results.json` | 2,240,102 | Stage A steering experiments (`rows`, `meta`). For reference only — not used in our Stage B pipeline. |
| `validation.json` | 1,122 | Stage A keyword vs. LLM-judge agreement at 4 steering strengths (0/4/8/12). Precision/recall/F1 per strength. |

## Reproduction

To re-port from upstream:

```bash
mkdir -p results/c7_backtracking/stage_a
for f in dom_vectors.pt prompts.json sentence_labels.json \
         steering_results.json traces.json validation.json; do
    git show "origin/case-backtracking:results/ward_backtracking/$f" \
        > "results/c7_backtracking/stage_a/$f"
done
```

## Notes

- The wasteland path is `results/ward_backtracking/` (not
  `results/ward_backtracking_txc/stage_a/` as referenced in some
  briefings — the `_txc` directory holds Stage B outputs only).
- The prior author's headline cohort is 31 truly-wrong + 30 originally-correct
  questions = 61 of the 300; filtering happens at run time, not in
  the data.
- All Stage A artifacts were generated under the prior wasteland code
  (`origin/case-backtracking:experiments/ward_backtracking/`). Our
  Stage B port reads them as opaque inputs.
