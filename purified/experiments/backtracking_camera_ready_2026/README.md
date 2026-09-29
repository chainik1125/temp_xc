# Backtracking camera-ready campaign, September 29, 2026

This directory implements the authorized rerun on four H100 80 GB GPUs.
TXC-pro is excluded. Paid judging is explicitly deferred to the user.

## Frozen training comparison

- TXC-base, shared TopK SAE, T-SAE 32K, and independent-position Stacked:
  seeds **1, 2, 42**, each at **300,000 completed optimizer steps**.
- T-SAE 16K: the same three seeds and 300K steps, separately labeled width
  sensitivity. This bounded comparison does not establish a globally best
  T-SAE configuration.
- Historical source `284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3`, checked by
  individual file hashes; original architecture objectives are preserved.
- Logical batch 1024 five-token windows, Adam, LR 0.0003, warmup 1000, and
  k=20. T-SAE retains its consecutive-pair objective, so its reconstructed
  token exposure differs from models reconstructing all five positions;
  receipts record both quantities rather than calling them equal.
- Python, NumPy, Torch and CUDA are seeded before initialization. Resume
  snapshots include optimizer and all RNG state. A short pilot is marked
  incomplete and cannot enter the publication comparison.

The 4.24 GB training cache and 1.14 GB detection cache are allowlisted,
revision-pinned and hash-verified. Large inputs and weights stay on the pod.
The old 300K checkpoint bytes could not be recovered; these are fresh seeded
trainings, not exact replays of the paper's unidentified old initialization.

## Detection corrections

All primary readouts see offsets -12 through -8 before the sentence onset.
One shared SAE supplies last-token, mean-pooled, max-pooled and position-aware
readouts. Stacked preserves `(position, feature)` identities instead of
pooling unrelated equal indices across independent dictionaries.

T-SAE now uses its frozen inference threshold via `model.eval()`. The July
detector omitted this call and used batch-dependent training-mode encoding.
Separate diagnostics retain that old mode and run the pinned original probe.

The 300 question IDs contain 213 distinct normalized problem texts. Primary
five-fold evaluation groups identical problem texts together. The original
question-ID split remains a diagnostic. Feature selection and feature scaling
use training folds only; raw C=1 and scaled C=1 probe variants remain separate.
The predeclared primary is scaled C=1 at S=8; raw C=1 is a sensitivity,
and S={1,2,4,8,16,32} is always saved. OOF predictions, selected
features, fitted coefficients, supports, and fold identities are retained.
`paired_detection.py` computes paired prompt-cluster bootstrap intervals at
S=8 and S=32 from these fixed out-of-fold predictions, resampling problems
within their original folds. These intervals condition on the trained models
and fitted probes; paired differences across dictionary seeds have their own
sample SD. Neither quantity is substituted for the other.

## Steering and deferred judging

The fixed cohort has 20 validation and 100 test questions from pinned MATH-500,
excluding the historical 61 steering questions and exact normalized mining
prompt matches. No exact matches were found between the 300 synthetic mining
prompts and MATH-500; this does not exclude semantic overlap or pretraining
exposure. The unsupervised cache's metadata identifies `ward_backtracking_math500`
but does not map its 4044 cached sequences back to question IDs.

Primary steering uses all eight 32K readout arms across three seeds, plus three
norm-matched random directions: **27 arms, 22,680 continuations** at the fixed
signed magnitudes {-12,-8,-4,0,4,8,12}. Generation is greedy, with a 1024-token
unsteered reference cap and cut25 continuation budgets. Truncation and actual
token counts are saved. T-SAE16K is a detection width sensitivity in this
campaign, not an additional steering family.

Both unsteered and continued paths now use identical explicit single-BOS
prompt IDs. The historical implementation doubled BOS in the first path.
The layer-10 hook affects prompt/prefix prefill and continuation, as recorded
in the generation identity; it is not a continuation-only intervention.

Generations are persisted before judging. Test candidate panels remain
ungraded and cannot be exported for judging until validation labels select
one signed magnitude. Placeholder validation exports live in `judge_template/`;
they are **not submission-ready**. Later choose an available OpenAI model and
check its total cost against the user's **$9.88 API balance** before submitting.
This code makes no API calls. `export-judge`, `import-judge`, `select`, `unlock`,
and `summarize` prepare and analyze the later user-operated judging stage.

## Execution and artifacts

Pod workspace: `/workspace/backtracking`; Python: `venv/bin/python`.
`campaign.py` runs training and detection workers. `steering_campaign.py` waits
for its GPU's training worker to finish, then generates and exports the saved
panels. `summarize_detection.py` produces Nord previews from actual completed
results; it never labels partial seed sets as a three-seed aggregate.

`pack_results.py` bundles compact JSON/JSONL, OOF predictions, CSVs and figures,
excluding weights, activations and sparse code matrices. Checkpoints remain in
`results/cells/<cell>/checkpoint/`; each has a SHA-256 and an actual completion
receipt. Copy them to durable storage before stopping or deleting this pod:
the inspected pod has container disk and no attached volume mount.

The current queue and failures are recorded in `results/queue_state.json` and
`results/steering/queue_state.json`. Failed cells are never silently replaced
with a short run. Preview figures before updating the manuscript.

`status.py` prints progress without reading generated answers. If a worker
dies, `campaign.py --reconcile-stale` and
`steering_campaign.py --reconcile-stale` release dead-owner claims only after
checking for active child work; restart the appropriate worker afterward.
Scientifically failed jobs require inspection and are not automatically retried.
`finalize_campaign.py` waits for the finite queues, renders summaries, and writes
`completion.json` plus `/workspace/backtracking/backtracking_compact_results.zip`.
`ready_for_deferred_judging` means the GPU work and exports completed; it does
not mean the steering effects have been judged or established.

The protocol follows the [focused audit](../../docs/aniket/camera-ready-2026/backtracking-focus-plan.md),
the [NeurIPS discussion](https://openreview.net/forum?id=Z27xj38Fta), and the
[ICML workshop discussion](https://openreview.net/forum?id=JfN7nRdBxA).
