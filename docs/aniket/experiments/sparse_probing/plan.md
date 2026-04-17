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

## 6. T-sweep extension (H100 bonus run)

Dmitry's addition: *"spin up an H100, push T as far as can fit."* On
the 80 GB H100 SXM pod, target `T ∈ {5, 10, 20, 40}`. Profile VRAM at
T=5 first, extrapolate to set max T — at 80 GB we expect T=40 to fit
at normal batch size. If T=40 OOMs in probing (where the Gemma
forward is the dominant consumer at ~22 GB), fall back to T=32.

Purpose: tests whether probing advantage scales with window size.
MLC and SAE unchanged (no T axis) — the sweep is **TempXC-only**.
Three possible patterns, all publishable:

- Advantage grows with T → temporal crosscoding benefits from longer
  context.
- Advantage plateaus at small T → T=5 captures most of what's there.
- Advantage shrinks at large T → shared-z saturates or dilutes
  (most scientifically interesting — reveals inductive-bias edge).

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

Total: ~6-8 hours of H100 time for all 9 checkpoints.

Probing eval per checkpoint: ~10 min (Gemma forward on ~32k labeled
examples, SAE encode, sklearn probes). 9 checkpoints × 4 aggregations
= 36 probing-run invocations × ~10 min = **~6 hours**.

**Grand total: ~14 hours on H100 ≈ \$40-60** depending on pod pricing.

## 13. Related docs

- [[experiments/sparse_probing/saebench_notes|SAEBench exploration notes]]
- [[experiments/sprint_feature_geometry/summary|feature-geometry results]]
- [[experiments/sprint_coding_dataset/plan|coding-dataset plan]]
- [[experiments/sprint_5k_autointerp/plan|5k autointerp plan]]
- [[SPRINT_PIPELINE|sprint pipeline overview]]
