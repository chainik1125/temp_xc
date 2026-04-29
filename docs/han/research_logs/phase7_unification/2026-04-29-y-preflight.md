---
author: Agent Y (Aniket pod)
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Y pre-flight — pipeline + ckpt + API checks pass

First-action gate from `agent_y_brief.md` and `.claude/y_pod_kickoff.md`.
Three assumptions verified on the A40 pod before any Q1/Q2 science work.

### #1 — Steering pipeline runs end-to-end

Smoke-test (note: CLI flag is `--archs`, not `--arch` as in the
kickoff prompt):

```bash
TQDM_DISABLE=1 .venv/bin/python -m \
  experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_window \
  --archs agentic_txc_02 --limit-concepts 1 --force
```

Result: `generations.jsonl` written with **9 rows** (1 concept × 9
strengths) under
`experiments/phase7_unification/results/case_studies/steering_paper/agentic_txc_02/`.
Pipeline loads the matryoshka TXC ckpt, builds the T=5 window-clamp
hook with `use_cache=False`, runs Gemma-2-2b base in bf16, and emits
non-empty generated text per (concept, strength) cell. No NaNs, no
crashes.

Note on data state: this pod is freshly cloned, so Dmitry's 270-row
`steering_paper/<arch>/{generations,grades}.jsonl` files are NOT
present locally — they're on his pod (gitignored per
`agent_y_brief.md` "Outputs (gitignored, on a40_txc_1)"). Q1.2 will
need to either re-generate the per-arch grids on this pod or pull
them from Dmitry's pod separately. Q1.1 is independent of those
artefacts.

### #2 — `tsae_paper.py` faithful port loads cleanly

Loaded `tsae_paper_k20__seed42.pt` from
`han1823123123/txcdr-base/ckpts/`. Confirmed:

- `sae.k = 20` buffer.
- `sae.threshold = 8.227` post-training inference threshold.
- `d_in = 2304`, `d_sae = 18432` (matches paper-faithful 8× expansion).
- **BatchTopK mode** (`use_threshold=False`): random-batch encode with
  B=64 yields **exactly B × k = 1280 active features** flat across
  the batch — paper-faithful BatchTopK behaviour.
- **Threshold mode** (default at inference): per-row activation count
  varies (~4 active per row on this random batch); this is the
  paper's documented variable-per-token sparsity at inference, not
  a regression. Source comment at `tsae_paper.py:16-29` already
  flags this.

### #3 — Anthropic Sonnet 4.6 access + prompt caching + ThreadPool=8

Model ID `claude-sonnet-4-6` (matches
`_paths.ANTHROPIC_GRADER_MODEL`). Prompt caching engages once the
system prompt clears the 1024-token minimum: a 2433-token system
block was cached on the 1st call (`cache_creation_input_tokens=2433`)
and read on all 8 subsequent parallel calls (`cache_read=2433` for
each). 8 parallel grader calls in 2.14s = 0.27s/call avg under
`ThreadPoolExecutor(max_workers=8)`. Graded sample inputs returned
expected scores (3 for medical content, 0 for unrelated text).

### Env hygiene check

- Pod user: `aniket`.
- Repo at `/workspace/aniket/temp_xc`.
- Branch: `aniket-phase7-y` (no commits to `han-phase7-unification`).
- `ANTHROPIC_API_KEY` and `HF_TOKEN` sourced from `.env` at repo root
  (gitignored). No Han-side keys in the shell.
- A40 GPU available, 46 GB free.

### Next

Begin Q1.1: `<z[j]_orig>` distributions across the 6 shortlisted archs
on the existing 30-concept × 5-example probe set. Output to
`results/case_studies/steering_magnitude/q1_1_z_orig_distributions.{png,json}`.
