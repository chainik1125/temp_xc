---
author: Aniket
date: 2026-04-17
tags:
  - proposal
  - in-progress
---

## Sparse Probing Benchmark — Pre-Registration

Three-way comparison of regular SAE, layer-wise crosscoder (MLC), and
TempXC on the [SAEBench](https://github.com/adamkarvonen/SAEBench)
sparse probing suite (Karvonen et al., ICLR 2025). Designed with
Dmitry + David Chanin.

SAEBench interface + data flow constraints: see
[[experiments/sparse_probing/saebench_notes|SAEBench notes]].
Sprint context: [[experiments/sprint_feature_geometry/summary|feature-geometry sprint]],
[[experiments/sprint_coding_dataset/plan|coding-dataset plan]].

## 1. Core question

Does TempXC produce representations more useful for downstream sparse
probing than alternative architectures? Specifically: is the
**temporal crosscoding axis** responsible for any advantage, as
opposed to "multi-position reconstruction in general"? We sit three
architectures side-by-side on SAEBench's sparse probing suite and
report which wins under which conditions.

## 2. Three architectures

All trained on **Gemma 2B activations at layer 12**
(`blocks.12.hook_resid_post`, `d_model = 2304`), cached from the same
activation dataset, TopK sparsity, same optimizer, same LR schedule,
same token budget, same `d_sae = 18,432` (8× expansion).

- **Regular SAE.** Single-token. Input `(d_model,)`, latent
  `(d_sae,)`. No multi-position structure. Baseline "does temporal
  matter at all" control.
- **Layer-wise crosscoder (MLC), 5-layer window middle-out around
  layer 12** — layers `{10, 11, 12, 13, 14}`. Input `(5, d_model)`,
  latent `(d_sae,)` shared across layers via TopK on summed
  pre-activation. "Multi-position crosscoding but not temporal"
  control.
- **TempXC, T=5.** Input `(T, d_model)` at layer 12, latent
  `(d_sae,)` shared across positions via TopK on position-summed
  pre-activation.

The MLC-vs-TempXC comparison is the cleanest isolation in the whole
experiment: **same param count, same FLOPs, same d_sae** — only
difference is whether crosscoding happens across layers or across
time.

## 3. Matched compute — what's fixed, what varies

**Fixed across architectures:**

- `d_sae` (18,432).
- Training tokens seen.
- Optimizer, LR schedule, batch size measured in tokens-per-step.
- Subject model, layer 12 anchor, activation cache.

**Varies (reported as columns):** parameter count, training FLOPs.
TempXC and MLC have ~5× the encoder params of regular SAE at T=5,
~5× the FLOPs per step. This follows Lindsey et al. and the
Anthropic crosscoder convention — deviating from "match d_sae +
tokens, let params/FLOPs vary" makes our numbers incomparable to
existing literature.

Dmitry acknowledged: *"ultimately equivalent FLOPs is the right
metric, but that may be hard to benchmark cleanly"* — we use matched
d_sae + tokens as the practical proxy and report FLOPs as a column.

## 4. Matched sparsity — two protocols, both reported

SAE at TopK=k activates k features per token → over a T-token window
it spends k×T total activations. TempXC at TopK=k activates k features
per window total. "Matched k" and "matched total budget" are
different sparsity constraints.

| protocol | SAE k | MLC k | TempXC k (T=5) | tests |
|---|---|---|---|---|
| **A — per-token k matched** | 100 | 100 | 100 | are TempXC's *individual* features more probing-useful? |
| **B — total-window budget matched** | 100 | 100 | 500 | is TempXC's representation *as a whole* more probing-useful? |

Running both yields 3 architectures × 2 protocols = **6 trained
checkpoints at T=5**. The pattern across protocols is itself part
of the finding.

**Protocol-B direction note.** Dmitry's Slack framing was
"SAE at k, TempXC at k/T" — i.e. keep SAE anchored at k=100 and
reduce TempXC to 100/T. Our table above does the opposite direction
(keep TempXC's k in the familiar 100–500 range, scale up if needed).
The two framings are mathematically equivalent — both test "match
total-window activation budget" — but anchoring TempXC's k to its
training-time value (100 at T=5) may be cleaner for the paper
write-up. If we flip direction before training, the downstream
matching_protocols.protocol_k() values change but nothing else in
the infrastructure does.

## 5. Four aggregation strategies

TempXC produces `(T, d_sae)` per window. SAEBench's
`get_sae_meaned_activations` calls `sae.encode(acts: (B, L, d_in)) →
(B, L, d_sae)` and then mean-pools across non-BOS/pad positions to
`(B, d_sae)`. Our aggregation axis lives **inside `encode()`** — it
determines how per-position encoder values are derived from T-window
inputs:

- **Last-position-only.** For each token at position `t`, return the
  last-position encoding of the window ending at `t` — i.e.
  `z[B, t-1, :]` from the window covering `[t-T+1..t]`. Matches
  single-token SAE semantics. Most conservative. Edge positions
  `t < T-1` fall back to the closest valid window.
- **Mean-over-T.** For each `t`, mean the T positions of the window
  centered at `t`. Closest to TempXC's training objective.
- **Max-over-T.** `max(dim=T)` within the centered window. "Did this
  concept appear anywhere in the window."
- **Full-window probe** (Dmitry's cheeky option). Don't collapse the
  T axis: return shape `(B, L, T × d_sae)` per position, letting the
  k-sparse probe pick `(feature, position)` pairs directly. Most
  generous to TempXC — lets position-specificity be exploited.

For regular SAE, all four are identical (no T axis). For MLC the
analog collapses layers rather than time; full-window flattens over
layers.

Predicted shape of the headline curve: **last-position → mean → max →
full-window** in order of increasing TempXC-favorability. If TempXC
only wins under full-window, the paper claim becomes "TempXC features
carry position-specific information" rather than "TempXC features are
individually better."

**Full-window semantics caveat.** SAEBench's
`get_sae_meaned_activations` mean-pools the encoded output across the
sequence axis `L` regardless of aggregation. For `full_window`,
`encode` returns `(B, L, T × d_sae)` — one window's worth of activations
per sequence position — and SAEBench's subsequent mean-pool across `L`
reduces that to `(B, T × d_sae)`. Each feature axis is therefore
"feature `i` at window-relative position `j`, averaged across all `L`
sequence positions where window-position `j` appeared." The probe
picks `(feature, window-position)` pairs, but the "window-position"
signal is aggregated over all stride-1 windows in the sequence rather
than a single window. Paper framing must phrase this as "features fire
in characteristic patterns across window-relative positions" rather
than "position-specificity within a single window matters." The
weaker interpretation is still informative and publishable; just don't
over-claim.

## 6. T-sweep extension (H100 bonus run)

Dmitry's addition: *"spin up an H100, push T as far as can fit."* On
the 80 GB H100 SXM pod, target `T ∈ {5, 10, 20, 40}`. Profile VRAM at
T=5 first, extrapolate to set max T — at 80 GB we expect T=40 to fit
at normal batch size. If T=40 OOMs in probing (where the Gemma
forward is the dominant consumer at ~22 GB), fall back to T=32.

**Default launch caps T-sweep at T=20** to keep the full sweep
overnight-feasible (~12–14 h total). T=40 doubles the training
budget and adds ~5× probing cost (stride-1 × 124 windows per
seq_len=128 sequence). Once the T ∈ {5, 10, 20} sweep finishes and
the headline direction is clear, a second-session T=40 run is a
~6-hour add-on. The orchestrator's `T_SWEEP_MAX=40` override
re-enables it without touching any other config.

Purpose: tests whether probing advantage scales with window size.
MLC and SAE unchanged (no T axis) — the sweep is **TempXC-only**.
Three possible patterns, all publishable:

- Advantage grows with T → temporal crosscoding benefits from longer
  context.
- Advantage plateaus at small T → T=5 captures most of what's there.
- Advantage shrinks at large T → shared-z saturates or dilutes
  (most scientifically interesting — reveals inductive-bias edge).

**Runtime warning at large T.** Stride-1 sliding-window encoding at
`T=40, L=128` yields `L - T + 1 = 89` overlapping windows per
sequence. Our `_encode_window` chunks through these in batches of
1024, but the total compute is ~`5–10×` the `T=5` cost per probing
eval. The 10-min-per-probing-eval estimate in § 12 is for T=5; at T=40
expect 50-100 min per eval. **Benchmark one probing eval at T=40
before committing to the full sweep** (via
`runpod_saebench_run_eval.sh --t 40 --aggregation last` on one
checkpoint). If prohibitive, two fallbacks:

- **Stride > 1.** Change the sliding-window step from 1 to
  `max(1, T // 4)`. Documented loss of per-token resolution — some
  tokens get encoding from a different-centered window than their
  neighbours. Reduces compute by ~4× at T=40.
- **Cap the sweep at T=20.** Halves T-sweep compute at the cost of
  less extrapolation range. T=20 still distinguishes "advantage grows
  monotonically" from "plateau at small T" for most scaling shapes.

## 7. The full experimental grid

**Per data condition at T=5:**

- 3 architectures × 2 matching protocols × 4 aggregations × 4 k-levels
  `{1, 2, 5, 20}` × ~10 probing tasks
- = **960 probing evaluations** per data condition.

SAEBench's default `k_values = [1, 2, 5]`; extending to include `20`
is a config change — the output schema already has a slot for it.

**Plus T-sweep (TempXC-only, T ∈ {5, 10, 20, 40}):**

- 1 architecture × 4 T × 2 protocols × 4 aggregations × 4 k-levels ×
  ~10 tasks = **1,280 additional evaluations**.

Each evaluation is an sklearn logistic regression fit — cheap once
checkpoints and encoded activations exist. The bottleneck is training
the 6 base checkpoints + 3 additional TempXC checkpoints for the
T-sweep = **9 total checkpoints**.

## 8. Pre-registered predictions

**Architecture comparison (protocol A, T=5):**

- **H1 (temporal crosscoding works):** TempXC ≥ MLC > SAE. Advantage
  largest under full-window; smallest (possibly none) under
  last-position.
- **H2 (multi-position works but not temporally-specific):**
  MLC ≈ TempXC > SAE. TempXC-vs-MLC gap is small.
- **H3 (shared-z is architectural overhead):** SAE ≥ MLC ≈ TempXC.
  Baseline wins or ties.

**Aggregation axis:** TempXC-favorability increases monotonically
last-position → mean → max → full-window.

**Matching axis:** TempXC performs better under protocol B
(budget-matched). The gap between A and B quantifies how much of any
advantage is feature-quality vs. representation-capacity.

**T-sweep:** uncommitted. Genuine open question. Monotonic growth,
plateau, and non-monotonic decline are all publishable; the last is
the most informative about TempXC's inductive-bias edge.

## 9. What counts as a win

Pre-register before looking at numbers:

- **Headline cell:** protocol A + full-window aggregation + k=5 +
  averaged across tasks. This is Figure 1.
- **Robust win:** TempXC > both baselines on ≥ 6 of 8
  (aggregation × matching) cells per task, averaged across tasks.
- **Weak win:** TempXC > SAE on headline but loses to MLC, or wins
  only under full-window.
- **Tie / loss:** TempXC ≤ SAE on headline.

Cleanest win is H1 with largest margin under full-window + low k.
Acceptable second-best is H2. Negative result (H3) is a genuine
finding — we pivot to the feature-geometry / reasoning-trace story
and keep sparse probing as a methods-section "negative control."

## 10. Figures

- **Figure 1** — bar chart, one panel per task, grouped bars for
  `{SAE, MLC, TempXC}` at the headline cell. Top-line result.
- **Figure 2** — 3-arch × 4-aggregation heatmap, averaged across
  tasks.
- **Figure 3** — protocol A vs B, same layout as Figure 2.
- **Figure 4** — probing accuracy vs k, one line per architecture,
  averaged across tasks.
- **Figure 5** — T-sweep curve, TempXC only, one line per
  aggregation (appendix unless results are dramatic).
- **Appendix table** — full 960-cell grid + per-task breakdowns +
  param counts + FLOPs.

## 11. Infrastructure

Code lives at `src/bench/saebench/`:

- `configs.py` — single source of truth: protocols A/B, k-values,
  T-values, aggregation names, Gemma-2-2B L12 spec.
- `aggregation.py` — the four aggregation strategies, one function
  per strategy.
- `matching_protocols.py` — dataclass for per-protocol k values,
  parameterized by T.
- `saebench_wrapper.py` — `SAEBenchAdapter(BaseSAE)`, wraps our
  `ArchSpec` instances to match SAEBench's `BaseSAE` contract.
- `probing_runner.py` — thin wrapper around SAEBench's `run_eval`,
  flattens its JSON into our cross-run JSONL schema.

Scripts at `scripts/runpod_saebench_*.sh` follow the existing
`runpod_*.sh` pattern. No nested `scripts/saebench/` subdir,
no separate `_common.sh` — `runpod_activate.sh` already does the env
setup all these scripts need.

Configs live in Python (`configs.py`), not YAML, matching
`src/bench/config.py` conventions. The Python-dataclass approach
gives type-checking + single-source-of-truth without adding a
YAML-parsing dependency.

### MLC integration roadmap

MLC is **the critical path for the headline claim**. Without it the
comparison collapses to "TempXC vs SAE," which cannot rule out H2
(multi-position crosscoding wins, temporal-specificity irrelevant).

Status:

- **Architecture** — landed in `src/bench/architectures/mlc.py`.
  `LayerCrosscoder` inherits from `TemporalCrosscoder` with identical
  encode/decode math; `LayerCrosscoderSpec` sets
  `data_format="multi_layer"` and a distinct `name` for logging.
  Registered in `REGISTRY`. Trainable once the data pipeline below
  lands.
- **Training data pipeline** — TODO. Needs multi-hook activation
  caching: simultaneous hooks on Gemma layers `{10, 11, 12, 13, 14}`
  writing `(n_seq, seq_len, 5, d_model)` tensors to disk. Path:
  extend `temporal_crosscoders/NLP/cache_activations.py` to accept
  multiple `--layer_indices` and stack into a 4-D output tensor, OR
  write a new `cache_mlc_activations.py` that hooks all 5 layers in
  one Gemma forward pass and stacks. Expect ~5× disk vs a
  single-layer cache.
- **Sweep runner data dispatch** — TODO. `src/bench/sweep.py`'s
  `build_pipeline()` currently dispatches on `data_format` ∈
  {`flat`, `seq`, `window`}. Add `multi_layer` → iterator yields
  batches of `(B, n_layers, d_model)` from the multi-hook cache.
- **SAEBench probing eval integration** — TODO. SAEBench's
  `run_eval_single_dataset` hooks a single layer and hands the
  adapter `(B, L, d_in)`. For MLC we need either: (a) a ~200 LoC
  fork that hooks `{10..14}`, collects `(B, L, 5, d_model)`, and
  passes that to a dedicated `SAEBenchAdapter` encode path; or (b) a
  pre-compute-and-inject path where we pre-encode our MLC on cached
  multi-layer activations, write `(B, L, d_sae)` to disk, and use a
  minimal probing script that skips SAEBench's data collection. (a) is
  more faithful; (b) is cheaper to implement.
- **Currently guarded paths.** `saebench_wrapper.SAEBenchAdapter.encode`
  raises `NotImplementedError` for `arch="mlc"` with a pointer to this
  section. `scripts/runpod_saebench_train.sh --arch mlc` fails fast
  with the same pointer. Architecture imports + `protocol_k("mlc", ...)`
  are fully functional — integration only blocks the runtime path.

Scoping these four items as a separate block of work rather than
inside this pre-registration; estimate 2-3 days of focused work.

**Output schema** (one JSONL record per
arch × T × protocol × aggregation × task × k):

```json
{
  "architecture": "tempxc",
  "t": 5,
  "matching_protocol": "A",
  "aggregation": "full_window",
  "task": "ag_news",
  "k": 5,
  "accuracy": 0.87,
  "param_count": 170000000,
  "training_flops": 1.2e18,
  "checkpoint_path": "results/saebench/ckpts/tempxc__A__t5.pt"
}
```

Results directory: `results/saebench/<run_id>/results.jsonl`.

## 12. Runtime budget

Checkpoint training (on H100 80GB):

| arch | training time | notes |
|---|---|---|
| SAE | ~20 min | d_sae=18432, TopK=100, 10k steps |
| MLC | ~40 min | 5× encoder params vs SAE |
| TempXC T=5 | ~40 min | same as MLC |
| TempXC T=10 | ~60 min | T-sweep |
| TempXC T=20 | ~90 min | T-sweep |
| TempXC T=40 | ~150 min | T-sweep; may need batch-size reduction |

Total: ~5 hours of H100 time for all 10 checkpoints (2 SAE + 2 MLC +
6 TempXC spanning T ∈ {5, 10, 20} × 2 protocols), at 5000 training
steps each (half the sprint's 10k default — chosen to fit the full
sweep in one overnight session without sacrificing convergence).

Probing eval per checkpoint × aggregation: ~10 min (Gemma forward
amortized across runs via SAEBench's activation cache, SAE encode +
sklearn probes per run). 10 checkpoints × 4 aggregations = 40
probing invocations × ~10 min = **~6-7 hours**.

**Grand total: ~13-14 hours on H100.** Fits the "launch + leave
overnight" profile.

## 13. Launch procedure

**Pre-flight.** Before committing to the overnight sweep, run the
30-minute pre-flight validation — trains one SAE for 500 steps, runs
probing on one task (`ag_news`), verifies the JSONL output has the
expected schema fields. Saves a 14-hour run on a silent schema bug.

```bash
# On the H100 pod, from repo root:
bash scripts/runpod_saebench_preflight.sh
```

Exits non-zero if anything's off. If it passes, launch the full
orchestrator:

```bash
bash scripts/runpod_saebench_orchestrator.sh full 2>&1 | \
    tee logs/saebench-$(date +%Y%m%d-%H%M).log
```

Or phase-by-phase (handy if you want to re-run a single stage without
the others):

```bash
bash scripts/runpod_saebench_orchestrator.sh cache       # ~1 h
bash scripts/runpod_saebench_orchestrator.sh preflight   # ~30 min
bash scripts/runpod_saebench_orchestrator.sh train       # ~5 h
bash scripts/runpod_saebench_orchestrator.sh eval        # ~7 h
bash scripts/runpod_saebench_orchestrator.sh summary     # seconds
```

**Pull results locally when done** (from the laptop):

```bash
bash scripts/fetch_saebench_results.sh
```

SSH target (`c2o4it0x73x88e-64412168@ssh.runpod.io`) is baked into
the script; override with `POD_SSH=... bash scripts/fetch_saebench_results.sh`
when the pod rotates.

## 14. Related docs

## 15. Related docs

- [[experiments/sparse_probing/saebench_notes|SAEBench exploration notes]]
- [[experiments/sprint_feature_geometry/summary|feature-geometry results]]
- [[experiments/sprint_coding_dataset/plan|coding-dataset plan]]
- [[experiments/sprint_5k_autointerp/plan|5k autointerp plan]]
- [[SPRINT_PIPELINE|sprint pipeline overview]]
