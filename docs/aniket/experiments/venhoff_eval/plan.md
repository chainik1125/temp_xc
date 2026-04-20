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

### Decisions locked (2026-04-18)

| # | decision | status |
|---|---|---|
| Q1 | Path 1 (SAE/MLC) + Path 3 (TempXC) hybrid | **locked** |
| Q2 | Layer 6 fixed; no layer sweep | **locked** |
| Q3 | All 4 aggregations run; `full_window` is the headline | **locked** |
| Q4 | Smoke 1k traces, full 5k traces | **locked** |
| Q5 | Haiku 4.5 judge; GPT-4o bridge on 100 sentences at smoke; drift threshold 0.5/10 | **locked** |
| — | P2 vs P3 as NeurIPS-abstract bar | **open**, not blocking Phase 1a |

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
each (architecture, T, layer, aggregation) combo. **Headline composite
uses TempXC-`full_window`** (locked Q3); other three aggregations
reported as supplement rows.

Plus our harness-native metrics:
- **Reconstruction NMSE** on reasoning activations (SAEBench-parallel)
- **Training loss curves** with plateau-early-stop (same plateau infra
  we built in commit `2fae76c`)
- **Ordered + shuffled pair** for each eval cell — advantage must
  survive shuffling to count as "temporal"

## 3. Architectures compared

Per Dmitry's 4/18 simplification: `{SAE, TempXC, MLC}` only. No TFA.

- **SAE**: standard TopK per-token. Baseline. Venhoff's default slot.
- **TempXC**: `T = 5` only (single value, not swept). Trained on
  per-token activations with aggregation at annotation time (Path 3,
  preserves temporal axis) — see [[integration_plan#3. Axis-collapse decision (load-bearing)|integration_plan § 3]].
  Rationale for collapsing the T sweep: (a) SAEBench showed T=5 was
  the least-degraded of our T values, so it's the setting most likely
  to produce a positive signal at all — if T=5 nulls, T=10/20 is very
  unlikely to rescue; (b) we already have T=5 checkpoints from the
  SAEBench rerun, avoiding a full retrain; (c) 3× compute + judge
  savings, which keeps Phase 1 inside one pod-week. If T=5 shows
  signal, expanding to T ∈ {10, 20} moves to Phase 1c instead of the
  main run.
- **MLC**: `n_layers = 5` around the anchor layer. Venhoff uses layer
  6 for 8B, so MLC window is `{4, 5, 6, 7, 8}`.

## 4. Grid

Phase 1 full run:

- 3 architectures (SAE, TempXC-T5, MLC)
- 10 cluster sizes per arch (Venhoff's full sweep: `5..50`)
- **Fixed layer 6 only** (not a sweep — locked in the Slack proposal
  as Q2 answered "anchor, not sweep")
- For TempXC: × 4 aggregation strategies at *annotation time* (Path 3
  re-encodes the same wide T=5 ckpt 4× — not 4 separate trainings)
- 5000 traces generated from MMLU-Pro test
- All fits capped at **10k training steps** with plateau-based
  early stop (see [[integration_plan#5. Configuration choices|integration_plan § 5]])
- Shuffled-control pair for every cell

**Training fit count** (what actually consumes GPU):
- SAE (Path 1, per-sentence-mean at layer 6): 10 cluster sizes = **10 fits**
- MLC (Path MLC, per-sentence-mean over layers {4,5,6,7,8}): 10 cluster
  sizes = **10 fits**
- TempXC (Path 3, T=5 token window per sentence): 10 cluster sizes = **10 fits**
- **Total: 30 fits**, each <5 min at the 10k-step cap → **~2.5 H100-hours**
  of dictionary training.

**Evaluation cell count** (what the judge scores):
- SAE: 10 sizes = 10 cells
- MLC: 10 sizes = 10 cells
- TempXC: 10 sizes × 4 aggregations = 40 cells
- **Total: 60 cells**, each gets a (accuracy, completeness, orthogonality)
  triple from Haiku 4.5.

## 5. Predictions

Pre-registered. If results match prediction, the narrative is clean.
If they don't, the plan doc has to update to reflect what actually
happened.

### Reframing after Han's 2026-04-19 sparse-probing rerun

On 2026-04-19 Han reran the sparse-probing benchmark with proper
convergence and expanded the task count from 8 → 27. The headline
number flipped from "MLC wins decisively (0.941 vs 0.862 TXC-T5)" to
"MLC and TXCDR-T5 at parity (0.807 vs 0.797, overlapping error bars)".
This was **before the Venhoff eval ran**, so it's not a post-hoc
re-interpretation of Venhoff results — it's an update to what the
Venhoff eval is *for*. The relevant comparison is no longer
TempXC-vs-SAE; it's TempXC-vs-MLC, with SAE held as a per-token
baseline that neither outperforms on probing.

If sparse probing says MLC ≈ TempXC-T5, then Venhoff's taxonomy-quality
metric is the differentiator between "the axes surface different
features" and "MLC is strictly better." That's the load-bearing
question now.

Predictions below are against MLC (primary) with SAE as a sanity floor.

| prediction | composite score delta | interpretation if observed |
|---|---|---|
| **P1 (null)**: TempXC ≤ MLC on all metrics, all cluster sizes, all aggregations; TempXC also ≤ SAE | Δ ≤ 0 vs both | TempXC doesn't help for reasoning taxonomies and doesn't even beat the per-token baseline. Combined with the sparse-probing parity, TempXC's case is weak. Paper pivots to "MLC is a strong crosscoder; temporal axis doesn't generalize." |
| **P2 (weak)**: TempXC > SAE at small cluster sizes (5-20) under `full_window`, but MLC matches or beats TempXC at all sizes | TempXC − SAE > 0 for k≤20; TempXC − MLC ≤ 0 | Temporal axis offers some structure beyond per-token but MLC captures at least as much. Supports "any non-trivial crosscoding axis helps; which axis matters less." Medium paper finding; autointerp/feature-geometry becomes the main differentiator. |
| **P3 (medium)**: TempXC and MLC at parity on the headline `full_window` metric, but diverge on *which* clusters they surface (qualitative) | \|TempXC − MLC\| < 0.5 pt on composite | This is the outcome most consistent with Han's sparse-probing parity. Paper headline becomes "temporal and layer crosscoding find complementary but non-overlapping reasoning categories," motivating the feature-geometry contribution. Strong NeurIPS story. |
| **P4 (strong)**: TempXC beats MLC on composite under `full_window` at 5k traces | TempXC − MLC > +0.5 pt | Temporal axis genuinely better for reasoning than layer axis. Strongest possible result; motivates Phase 2 steering-vector work immediately. |

**Win criteria for the paper** (post-2026-04-19 reframing):
- **NeurIPS abstract**: P3 or P4 both support the abstract cleanly. P2
  supports a weaker version ("temporal axis helps over SAE but not over
  MLC"). P1 pivots to a negative-result writeup.
- **Open with Dmitry**: whether P2 clears the bar. Not blocking the run
  — the same pipeline produces the evidence to decide either way.
- **ICML workshop fallback**: P1 with diagnostic clarity (training
  curves, SVD spectrum per Han's T20 finding) is still a careful
  negative worth publishing.

## 6. Runtime expectations

Per `integration_plan § 6`: ~40 H100-hours + ~$15 in Haiku 4.5 judge
fees for full Phase 1 (4-10× cheaper than Venhoff's GPT-4o default).
Smoke test at 1000 traces, one arch, one cluster size: ~2-3 H100-hours
+ <$1 judge fee.

## 7. Relationship to sparse-probing result

**2026-04-19 update (Han's 27-task rerun with proper convergence):**
MLC: 0.807. TXCDR-T5: 0.797. Overlapping error bars — parity.
TXCDR-T20: 0.751 (regression, confirming under-regularization per
Han's SVD-spectrum diagnostic). SAE: 0.745. Last-token logistic
regression baseline: 0.934 (well above every dictionary — consistent
with "SAEs don't recover the probe's signal cleanly at this scale").

Implications for Venhoff:
- TempXC-vs-SAE is no longer the core comparison. TempXC does beat SAE
  by ~5 pp on the new numbers, but that's not the story.
- TempXC-vs-MLC is the load-bearing question. Venhoff's clustering +
  taxonomy metric rewards structural coherence, which is the axis
  where "different features" could surface even if benchmark accuracy
  ties. If MLC and TempXC select different reasoning categories, the
  Venhoff eval exposes it.
- T=5 is confirmed as the right TempXC operating point. T=20's
  regression is a separate story (act_fn / regularization) Han is
  investigating on a different track.

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
- headline lines: **MLC (orange, bold — primary comparison post-2026-04-19)**,
  **TempXC-T5 `full_window` (green, bold)**, SAE (blue, dashed — per-token
  baseline, not the story)
- supplement lines (same plot, lighter/dashed so they're visible but
  non-dominant): TempXC-T5 `last`, `mean`, `max`
- error bars: from 3 repetitions per cluster size
- **shuffled-control shaded region per cell**: for each (arch, cluster
  size) cell, shade the band between the ordered-trace composite and
  the shuffled-trace composite. The shaded height quantifies how much
  of the score is attributable to temporal structure specifically
  (the shuffle control removes within-trace ordering). A narrow band
  means the advantage is not temporal; a wide band means it is. This
  is the figure-level evidence for the "temporal" claim.
- The MLC line is the thing Dmitry reads first. If MLC ≥ TempXC at
  every cluster size, the paper story becomes qualitative (§ 5 P2/P3);
  if TempXC ≥ MLC, it's P4 — the strongest outcome.

This is the figure Dmitry + team react to at check-in. If it shows
TempXC above SAE, paper is alive.
