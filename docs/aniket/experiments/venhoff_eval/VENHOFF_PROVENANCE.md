---
author: Aniket Deshpande
date: 2026-04-18
tags:
  - reference
  - venhoff-eval
---

## Venhoff code provenance

Attribution + change tracking for every file ported from
`cvenhoff/thinking-llms-interp` into this repo.

**Upstream repo**: https://github.com/cvenhoff/thinking-llms-interp
**Upstream paper**: Venhoff et al., "Base Models Know How to Reason,
Thinking Models Learn When", arXiv:2510.07364
**Upstream commit pinned**: `49a7f731ce693d813b9ae9a414f1739b992dbcef`
**Upstream license**: no `LICENSE` / `COPYING` file shipped at the
pinned commit. README is silent on licensing. TODO: email authors to
clarify terms before any redistribution; internal research use is
consistent with typical academic-code norms but not explicitly
permitted. Local clone lives at `vendor/thinking-llms-interp/` (not
modified).

## Ported files

Populated as each integration step completes. Entries are:
- upstream path
- our local path
- commit hash of upstream source at time of port
- what we changed + why

| upstream | ours | upstream-commit | changes |
|---|---|---|---|
| `utils/autograder_prompts.py` | `src/bench/venhoff/autograder_prompts.py` | `49a7f73` | Byte-for-byte port. Only change: added a module docstring flagging the pre-registered-invariant status. All 5 `categories_examples`, 3 judge prompts, and helper formatters preserved. |
| `utils/utils.py::split_into_sentences` | `src/bench/venhoff/tokenization.py` | `49a7f73` | Port with reformatted docstring (behavior identical). Parity tests in `tests/bench/venhoff/test_tokenization.py` against Venhoff's `test_split_into_sentences.py` cases. |
| `utils/utils.py::get_char_to_token_map` | `src/bench/venhoff/tokenization.py` | `49a7f73` | Verbatim port. |
| _new — not from Venhoff_ | `src/bench/venhoff/tokenization.py::sentence_token_span` | — | New helper lifting the sentence-find / offset lookup out of `process_saved_responses` (`utils.py:426-436`) as a reusable fn. Returns `(token_start, token_end)` *without* the `-1` offset that Venhoff's `process_saved_responses` applies on line 438 — callers that need the Venhoff contract subtract 1 themselves. |
| _new — not from Venhoff_ | `src/bench/venhoff/judge_client.py` | — | Lift of `ClaudeAPIBackend` from `temporal_crosscoders/NLP/autointerp.py:440` into a standalone module. Default model is Haiku 4.5; includes an `OpenAIJudge` parallel for the bridge-drift check only. |
| _new — not from Venhoff_ | `src/bench/venhoff/dataset.py` | — | Thin MMLU-Pro loader wrapping `TIGER-Lab/MMLU-Pro` with Venhoff's expected prompt template (copied from `generate-responses/generate_responses.py`). |
| `utils/responses.py::extract_thinking_process` | `src/bench/venhoff/responses.py` | `49a7f73` | Verbatim port. Selection policy for single/multi-`<think>` tags preserved. |
| `generate-responses/generate_responses.py` | `src/bench/venhoff/generate_traces.py` | `49a7f73` | Rewrote the entry-point around our `model_registry.py`; kept the vLLM path (preferred) and added a transformers fallback. Prompt-formatting via `dataset.py`'s MMLU-Pro template. Sidecars metadata for resume semantics. |
| `utils/utils.py::process_saved_responses` (Path 1 contract) | `src/bench/venhoff/activation_collection.py::collect_path1` | `49a7f73` | Preserves Venhoff's `[:, token_start - 1 : token_end, :]` offset + running-mean definition. Adapted to our HF model loader + atomic-write sidecar metadata for resume. |
| _new — not from Venhoff_ | `src/bench/venhoff/activation_collection.py::collect_path3` | — | Path 3 T-window collector: per-sentence `(T, d_model)` slice centered on the sentence. TempXC-only contract per Q1 lock-in; no Venhoff equivalent because their per-sentence-mean assumes no temporal axis. |
| _new — not from Venhoff_ | `src/bench/venhoff/sae_shim.py` | — | Duck-type adapter exposing `.encoder`, `.W_dec`, `.b_dec`, `.activation_mean` around our `TopKSAE`/`TempXC`/`MLC`. Path 3 shim injects aggregation (last/mean/max/full_window) into `.encoder`. |
| `utils/clustering_methods.py::clustering_sae_topk` | `src/bench/venhoff/train_small_sae.py` | `49a7f73` | Small-k training loop built on our existing `ArchSpec.train` so we inherit plateau + W&B logging + gradient clipping. 10 000-step cap + `plateau_pct=0.005` locked per Q5 `integration_plan § 5`. Ckpt serialization is ours, not theirs (they use plain `state_dict`, we add config + train_log for resume + debug). |
| `generate-responses/annotate_thinking.py` | `src/bench/venhoff/annotate.py` | `49a7f73` | Argmax-over-latents preserved. Path 3 path replaces the direct `.encoder` call with the shim's aggregation-aware variant. Skips when `assignments_json` already exists with a matching config hash. |
| _new — not from Venhoff_ | `src/bench/venhoff/paths.py` | — | Artifact path registry + `write_with_metadata` / `can_resume` sidecar helpers. Shared by every stage so resume semantics are consistent. |
| _new — not from Venhoff_ | `src/bench/venhoff/smoke.py` | — | End-to-end orchestrator with `--force` / `--force-stage` / `--skip-stage` flags, mirrors the SAEBench launcher's resume pattern. |
| _new — not from Venhoff_ | `scripts/runpod_venhoff_launch.sh` | — | Pod-side single-entry launcher. `MODE=smoke|full` toggles 1k-trace smoke vs 5k-trace full Phase 1b sweep. Every stage resumable via `FORCE_STAGE=` / `SKIP_STAGE=` env vars. |

## Files we deliberately did not port

- `train-vectors/` — Phase 2, stub for now. Would duplicate their
  steering vector training verbatim.
- `hybrid/` — Phase 3. Only needed if Phase 2 shows steering works.
- `human_eval/` — not in the default pipeline per their README.

## Files we wrapped, not ported

Listed here are files we call as-is via subprocess or import, without
local copies. Subject to their upstream license.

| upstream | how we use it | why not ported |
|---|---|---|
| _(tbd)_ | | |

## Upstream configs we pre-register as invariants

When we swap architectures, these MUST stay fixed to keep the metric
comparable:

- The 5 hand-written reasoning-category examples in
  `utils/autograder_prompts.py::categories_examples` (Generating
  Hypotheses, Expressing Uncertainty, Planning Future Steps,
  Stating Assumptions, Recalling Mathematical Definitions).
  Changing these changes the benchmark. Held fixed across SAE /
  TempXC / MLC comparisons.
- Judge prompts in `utils/autograder_prompts.py` — all three
  (accuracy, completeness, semantic-orthogonality) used as-shipped.
- n_autograder_examples = 100, accuracy_target_cluster_percentage =
  0.2 — their defaults. If we tune these, we're not running their
  benchmark anymore.

Any deviation from these requires a note in this file with a reason.

## Deliberate deviations from Venhoff's setup

| invariant | their value | ours | reason |
|---|---|---|---|
| judge model | gpt-4o (paper) / gpt-4.1-mini (released code default at pinned commit — see note below) | **claude-haiku-4-5-20251001** (Haiku 4.5) | Cost (~4-10× cheaper) + existing Anthropic credits. Venhoff's code exposes `anthropic` as a first-class fallback, so the swap is supported. **Bridge plan**: during the smoke run, re-judge the same 100 sentences with GPT-4o and report the drift. If mean per-metric drift > 0.5 points on the 0-10 rubric, revisit. Until then, report only within-judge deltas (TempXC vs SAE under the same judge) rather than absolute scores. |

**Note on Venhoff's current judge default.** At the pinned commit,
`train-saes/evaluate_trained_clustering.py` argparse default for
`--evaluator_model` is `gpt-4.1-mini`, not `gpt-4o`. The paper
(arXiv:2510.07364) reports numbers with `gpt-4o`. The released code's
default drifted after publication. Our bridge run uses `gpt-4o` (the
paper number), not `gpt-4.1-mini` (the current code default), so a
positive bridge pass means we reproduce the paper's judge contract,
not the code-as-shipped. Flagged so this discrepancy doesn't quietly
become "Venhoff uses gpt-4o" in downstream summaries.
