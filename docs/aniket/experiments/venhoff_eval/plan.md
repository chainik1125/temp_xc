---
author: Aniket Deshpande
date: 2026-04-18
tags:
  - proposal
  - venhoff-eval
  - temporal-crosscoder
---

## Venhoff reasoning-eval — experiment plan

**Purpose**: test whether Temporal Crosscoder (TempXC) features produce
a more coherent taxonomy of reasoning behavior than per-token SAE
features, using Venhoff et al.'s reasoning-trace pipeline as the
evaluation harness.

**Paper deadline**: NeurIPS abstract May 4 (16 days). ICML workshop May 8.

See [[integration_plan|integration_plan.md]] for the code-level
decisions and axis-collapse strategy. This document covers the
experiment-level hypothesis, metric, and predictions.

## 1. Hypothesis

> On reasoning traces from DeepSeek-R1-Distill-Llama-8B, TempXC features
> cluster into more coherent and temporally-structured reasoning categories
> than per-token SAE features, as measured by Venhoff et al.'s taxonomy-
> quality metrics (accuracy, completeness, semantic orthogonality) with
> **Claude Haiku 4.5** as judge (deviation from their GPT-4o default; see
> [[integration_plan#5. Configuration choices|integration_plan § 5]]).

Why we'd expect this: reasoning steps (backtracking, verification,
case analysis) span multiple sentences or multi-token phrases — they're
not token-local. A per-token SAE sees one token's residual stream at a
time, so multi-step patterns can only be reconstructed by
superposition of single-token features. TempXC's shared-latent
window explicitly represents multi-position structure as a single
feature.

## 2. What we're measuring

Venhoff's pipeline produces three scalar quality metrics per trained
dictionary, averaged across cluster-sizes and repetitions:

- **accuracy** — does each cluster's (title, description) actually
  apply to the sentences assigned to it? Judge Yes/No over 100 sampled
  sentences. 0–1.
- **completeness** — does each sentence match its cluster's description?
  Judge 0–10 rubric.
- **semantic orthogonality** — how non-redundant are cluster descriptions?
  Pairwise judge 0–10, inverted.

Judge is **Claude Haiku 4.5** (model id `claude-haiku-4-5-20251001`),
substituted for Venhoff's GPT-4o. Bridge run on 100 sentences during
smoke test quantifies judge-drift vs GPT-4o before committing to full
sweep.

Composite: `avg_final_score` rolled up per cluster size, reported for
each (architecture, T, layer, aggregation) combo.

Plus our harness-native metrics:
- **Reconstruction NMSE** on reasoning activations (SAEBench-parallel)
- **Training loss curves** with plateau-early-stop (same plateau infra
  we built in commit `2fae76c`)
- **Ordered + shuffled pair** for each eval cell — advantage must
  survive shuffling to count as "temporal"

## 3. Architectures compared

Per Dmitry's 4/18 simplification: `{SAE, TempXC, MLC}` only. No TFA.

- **SAE**: standard TopK per-token. Baseline. Venhoff's default slot.
- **TempXC**: `T ∈ {5, 10, 20}`. Trained either on per-sentence-mean
  activations (Path 1, matches Venhoff's contract) or on per-token
  activations with aggregation at annotation time (Path 3, preserves
  temporal axis). Plan defaults to Path 3 for TempXC (see
  integration_plan § 3).
- **MLC**: `n_layers = 5` around the anchor layer. Venhoff uses layer
  6 for 8B, so MLC window is `{4, 5, 6, 7, 8}`.

## 4. Grid

Phase 1 full run:

- 3 architectures
- 10 cluster sizes per arch (Venhoff's full sweep: `5..50`)
- For TempXC: × 3 T values × 4 aggregation strategies = 12 TempXC
  variants per cluster size
- Fixed layer 6 (Venhoff's chosen anchor for Llama-8B)
- 5000 traces generated from MMLU-Pro test
- Shuffled-control pair for every cell

**Total cells**: 10 + 120 + 10 = 140 trained small-k dictionaries per
cluster size × 10 cluster sizes = 1400 fits, but each tiny SAE takes
<5 min so it's ~3 H100-days of small-k training.

**Plus** the Path 3 re-encode of our existing wide TempXC ckpts from
SAEBench (fast, hours not days).

## 5. Predictions

Pre-registered. If results match prediction, the narrative is clean.
If they don't, the plan doc has to update to reflect what actually
happened.

| prediction | composite score delta | interpretation if observed |
|---|---|---|
| **P1 (null)**: TempXC ≤ SAE on all metrics, all cluster sizes, all aggregations | Δ ≤ 0 | TempXC doesn't help for reasoning. Combined with the SAEBench null, the architecture is dead for this domain. Paper becomes a careful dual-null writeup. |
| **P2 (weak)**: TempXC wins under `full_window` aggregation at small cluster sizes (5-20) but flattens at large (30-50) | Δ = +0.5–1 pt at k≤20, ≈0 at k≥30 | Temporal features matter when the taxonomy is small enough that individual categories have to compress; at larger cluster sizes each cluster gets to be token-local anyway. Medium-strength paper finding. |
| **P3 (medium)**: TempXC wins monotonically across cluster sizes under the right aggregation | Δ = +1–2 pt consistently | Strong positive signal. Steering-vector Phase 2 almost certainly worth doing. Publishable as NeurIPS abstract. |
| **P4 (strong)**: TempXC wins monotonically AND MLC loses to TempXC in the same window | TempXC > MLC on composite | Specifically *temporal* structure matters, not just multi-position. Best-case paper finding. |

**Win criteria for the paper**:
- **NeurIPS abstract**: Any of P2, P3, or P4. P1 triggers the careful-null
  writeup instead.
- **ICML workshop fallback**: P1 with diagnostic clarity (e.g. training
  curves show TempXC converged but still lost) is still publishable as a
  careful negative.

## 6. Runtime expectations

Per `integration_plan § 6`: ~40 H100-hours + ~$15 in Haiku 4.5 judge
fees for full Phase 1 (4-10× cheaper than Venhoff's GPT-4o default).
Smoke test at 1000 traces, one arch, one cluster size: ~2-3 H100-hours
+ <$1 judge fee.

## 7. Relationship to SAEBench result

SAEBench (our previous eval) found TempXC ≈ SAE at T=5 with degradation
at larger T. That was on single-token probing — not the domain TempXC
was designed for.

Reasoning traces are a strictly friendlier domain for TempXC:
- Multi-sentence patterns are the *norm*, not rare
- The discovery task (cluster/label) is more permissive than single-
  token classification
- The clustering + taxonomy metric weights structural coherence, which
  is TempXC's strongest axis

So a null result here is meaningfully stronger evidence against
temporal crosscoding than the SAEBench null was. Conversely, a positive
result here outweighs the SAEBench negative because this is the
better-matched benchmark.

## 8. Out-of-scope for Phase 1

- No steering-vector training (Phase 2)
- No hybrid-model experiments (Phase 3)
- No non-{Llama-8B} models (Phase 4)
- No comparison to TFA (Dmitry's 4/18 simplification)
- No auto-interp-heavy feature analysis beyond the judge-generated cluster
  titles Venhoff's pipeline already emits

## 9. Fallback plan

If smoke test (Phase 1a) reveals that Venhoff's released pipeline
can't be reproduced cleanly in our env (dep mismatches, missing data,
broken scripts), escalate: we either (a) email the authors for help
or (b) drop to a scaled-down variant where we do the clustering + LLM
labeling ourselves from scratch, without their code. Budget 2 days
for Phase 1a; if it's still not running by then, escalate.

## 10. What produces the headline figure

One plot, saved at `results/venhoff_eval/plots/fig1_taxonomy_quality.png`:

- x-axis: cluster size (5..50)
- y-axis: composite taxonomy quality score (0–10)
- lines: SAE (blue), TempXC-T5 (light green), TempXC-T10 (mid green),
  TempXC-T20 (dark green), MLC (orange)
- error bars: from 3 repetitions per cluster size
- shaded region: ordered-vs-shuffled delta — the portion of advantage
  attributable to temporal structure specifically

This is the figure Dmitry + team react to at check-in. If it shows
TempXC above SAE, paper is alive.
