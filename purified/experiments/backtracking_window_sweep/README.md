# Backtracking window-size sweep

This is the current-branch-native implementation of Aniket's July 23
backtracking deliverable. It trains TXC-base and a matched TopK SAE at
\(T=1,\ldots,6\) for seeds \(1,2,42\), then evaluates question-grouped
sentence-level detection on Ward's event-aligned activation artifact. It does
not use code or data from another branch.

## What the six points mean

The available artifact contains only layer-10 activations at offsets
\(-13,-12,-11,-10,-9,-8\) relative to the first token of each labeled
backtracking sentence. The sweep therefore uses trailing subsets:

| T | Physical offsets |
|---:|---|
| 1 | -8 |
| 2 | -9…-8 |
| 3 | -10…-8 |
| 4 | -11…-8 |
| 5 | -12…-8 |
| 6 | -13…-8 |

This directly tests whether covering more of the layer-10 region isolated by
Ward et al. improves detection. It cannot test \(T>6\), a window ending at
offset \(-1\), or the meeting's loose recollection of an approximately
18-token lag. Those require a new, wider event-aligned extraction.

## Comparators and controls

Each \((T,s)\) cell trains both dictionaries from the same 4,044-by-128
activation cache. A counter-based schedule gives them the identical
\(B\times T\) raw activation values at every update; the SAE flattens those
values to \(B T\) token examples while the TXC sees the ordered window. This is
stricter exposure matching than reusing one SAE checkpoint across all T.
The schedule is nested across T as well: for each seed and step it samples the
same sequence IDs and endpoints of a \(T=6\) window, and every shorter T gets
the trailing subset. Window-size differences therefore cannot be attributed
to different sampled training cohorts.

The held-out evaluation reports:

- TXC codes with a question-grouped sparse probe;
- the same fixed probe after per-example shuffle, reversal, and non-zero
  circular shift;
- a shared SAE whose codes are concatenated into explicit positional blocks,
  which is the strongest multi-token SAE baseline;
- an exact order-invariant max pool of those SAE codes and a last-token SAE;
- a train-fold-only covariance-whitened residual detector, including its best
  single offset and invariant mean.

All held-out probabilities, labels, test indices, and question IDs are
persisted. Each cell reports a deterministic paired question-group bootstrap
for TXC minus the SAE positional stack, the strongest learned control, and the
strongest control after also admitting the supervised residual upper bound.
Smoke uses 50 replicates; full uses 2,000.

An improvement with T but no damage from order controls supports additional
context, not ordered temporal structure. A TXC that loses to the SAE positional
stack does not isolate a benefit from the TXC reconstruction objective.

## Required artifacts

The full protocol refuses to run unless it finds:

1. `purified/artifacts/c7/sentence_acts_L10.npz`, SHA-256
   `1656f6be…e27810`, with `X=(25204,6,4096)`, `is_bt`, and grouped keys;
2. `acts.npy` for activation cache `fb2a74be884e512a`, with shape
   `(4044,128,4096)`.

The default cache path is the exact recovered RunPod file
`purified/artifacts/hf_temp_bench_data/act_cache/fb2a74be884e512a/resid_post_L10.npy`.
Set
`BACKTRACKING_ACTIVATION_CACHE` if the Hugging Face download is placed
elsewhere. No language-model weights or HF permission are needed once these
two activation artifacts are present.

The Python environment needs PyTorch, NumPy, SciPy, scikit-learn,
safetensors, and Matplotlib. The run wrapper defaults to
`$TXC_RUNPOD_ROOT/.venv-e0-extract/bin/python`.

## Dry run and launch

The worker is the orchestration hook:

```bash
export BACKTRACKING_T_SWEEP_RUNNER="$TXC_RUNPOD_ROOT/purified/experiments/backtracking_window_sweep/run_runpod.sh"
bash "$BACKTRACKING_T_SWEEP_RUNNER"
```

Set `BACKTRACKING_DRY_RUN=1` to inventory artifacts and print the exact cell
queue without writing results or starting training. The standalone named-tmux
launcher can split whole seeds across available GPUs:

```bash
BACKTRACKING_GPU_LIST=0,1 \
  bash "$TXC_RUNPOD_ROOT/purified/experiments/backtracking_window_sweep/launch_tmux.sh"
```

Every architecture checkpoint has an atomic model and optimizer state, every
cell has its own `result.json`, and `summary.md`, `summary.json`, and
`window_curve.png` are regenerated after each completion. A vanished process
can be relaunched with the same command; completed steps and cells are skipped.
Smoke and full runs use separate `smoke/` and `full/` result/checkpoint roots
and mode-specific tmux session names, so a completed smoke cell can never make
the full runner skip or reject a cell.

## Compute queue

The dispatch order is seed 42 first, with \(T=1\) and \(T=6\) before the
interior points, then seeds 1 and 2. This makes the canonical endpoint gate
available before spending on the full curve. With BF16 weights, all 18 cells
need about 44 GB of final model files; retaining the current Adam resume state
raises the persistent estimate to roughly 130 GB, with additional transient
space during atomic saves.

The compute is large because the matched SAE receives the same \(B T\) values
as each TXC. Run the smoke profile first, then benchmark a short full-width
\(T=6\) segment before estimating wall time. If the seed-42 \(T=1\) versus
\(T=6\) gate is null against the positional SAE and residual controls, the
meeting's fail-fast rule says to stop rather than fill all three seeds.

## Isolated T16 extension

The reviewer-stage extension is a new protocol rather than an in-place change
to the completed six-offset sweep:

- protocol `2026-07-26.t16.1`;
- grid \(T\in\{1,2,4,6,8,10,12,14,16\}\);
- offsets \(-23,\ldots,-8\);
- results under
  `purified/results/neurips_rebuttal/backtracking_window_sweep_t16/`;
- checkpoints under `checkpoints/backtracking_window_sweep_t16/`.

The wider extraction necessarily drops sentence events whose start position is
too early to support offset \(-23\). Every T is therefore evaluated on the
same ordered keyed subset in `sentence_acts_L10_T16.npz`; it is invalid to
combine the old T<=6 cohort with the new T>=8 points. The T16 runner checks
that keys form an order-preserving subset of the official six-offset artifact,
labels match on the join, the artifact contains exact offsets `-23..-8`, its
SHA matches the builder manifest, and the manifest records bit-exact agreement
between the trailing six activations and the official artifact.

The intended artifact is teacher-forced from the pinned labeled
`full_response` traces. The public 3,300-by-256 Stage-B dictionary-training
cache has no proven mapping to the 300 labeled evaluation traces and is not an
acceptable extraction source. A `ward-c7-wide-teacher-force.v1` manifest must
pin the response file path, SHA, and commit; model and tokenizer IDs and
revisions; layer 10 `resid_post`; the common-cohort hash and key order; and an
exact keyed comparison against the official trailing six offsets. The older
coordinate-map manifest remains accepted only when it supplies an explicit
event map and its own exact-tail proof; the irrelevant 6.9 GB residual-cache
provenance is never required for a teacher-forced artifact.

The teacher-force builder never fetches traces or reads another branch. Point
it at one already-supplied file and pin that file explicitly:

```bash
export BACKTRACKING_TEACHER_TRACES=/workspace/inputs/traces.json
export BACKTRACKING_TEACHER_TRACES_SHA256=<64-hex-sha256>
export BACKTRACKING_TEACHER_SOURCE_PATH=<repository-relative-path>/traces.json
export BACKTRACKING_TEACHER_SOURCE_COMMIT=<40-hex-commit>

BACKTRACKING_TEACHER_PHASE=preflight \
  bash "$TXC_RUNPOD_ROOT/purified/experiments/backtracking_window_sweep/run_teacher_force_runpod.sh"
```

The preflight validates all 300 raw records against the pinned prompts,
sentence labels, and official key/label order without loading a model. A
bounded GPU smoke adds
`BACKTRACKING_TEACHER_MAX_TRACES=1`; remove that limit and launch one
modulo-partitioned trace worker per available GPU:

```bash
export BACKTRACKING_TEACHER_GPU_LIST=0,1
bash "$TXC_RUNPOD_ROOT/purified/experiments/backtracking_window_sweep/launch_teacher_force_tmux.sh"
```

Each trace shard is committed only after its extracted `-13..-8` values equal
the official values bit-for-bit. Once all 300 shards exist, assemble and
repeat the complete keyed proof:

```bash
BACKTRACKING_TEACHER_PHASE=assemble \
  bash "$TXC_RUNPOD_ROOT/purified/experiments/backtracking_window_sweep/run_teacher_force_runpod.sh"
```

Dry-run the full plan without launching compute:

```bash
BACKTRACKING_T16_DRY_RUN=1 \
  bash "$TXC_RUNPOD_ROOT/purified/experiments/backtracking_window_sweep/run_t16_runpod.sh"
```

Before a full run, the real-width memory smoke instantiates the T=16,
32,768-feature TXC and matched SAE, performs one forward/backward/Adam step,
reports peak CUDA memory, and writes no checkpoint:

```bash
BACKTRACKING_T16_MODE=memory-smoke \
  bash "$TXC_RUNPOD_ROOT/purified/experiments/backtracking_window_sweep/run_t16_runpod.sh"
```

The T16 checkpoints and held-out code caches report actual effective L0 after
the implementation's TopK-then-ReLU composition. This does not alter the
architecture: it measures how often selected negative preactivations are
zeroed, so the nominal \(k_{\mathrm{pos}}T\) budget underfills.

Publication plots accept an explicit arbitrary grid and emit both absolute
ordered/shuffled curves and their difference:

```bash
python -m experiments.backtracking_window_sweep.plot_publication \
  purified/results/neurips_rebuttal/backtracking_window_sweep_t16/full \
  purified/results/neurips_rebuttal/backtracking_window_sweep_t16/full/publication \
  --windows 1,2,4,6,8,10,12,14,16 --seeds 1,2,42
```

An increase shared by ordered and shuffled curves is consistent with
order-invariant denoising or a DC-like component. The
`txc_ordered_minus_shuffled` plot isolates the smaller fixed-probe residual
that actually depends on local order.
