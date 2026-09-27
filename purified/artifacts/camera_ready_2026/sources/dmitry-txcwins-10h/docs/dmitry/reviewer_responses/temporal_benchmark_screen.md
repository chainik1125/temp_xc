---
author: Dmitry
date: 2026-07-23
tags:
  - design
  - in-progress
---

## A screen for whether a task is a good temporal-steering benchmark

A cheap, task-agnostic diagnostic — in the style of the backtracking experiment
(`experiments/ward_backtracking_txc/`) — that decides *before* the full compute
whether a temporal crosscoder will actually beat a per-token SAE on a candidate
safety task. Motivated by the refusal post-mortem: the criterion for a good
benchmark is not "the behavior is multi-token," it is **headroom that is
specifically closed by temporal aggregation**, and refusal fails it because a
single-direction/single-position baseline (Arditi) already near-saturates. See
[[temporal_safety_tasks_litreview]], [[refusal_experiment_plan]],
[[window_length_theory]].

### The one fairness insight that makes the screen honest

The per-token baseline must be **the best single position, not a default one.** A
window that only beats a per-token SAE read at the *terminal* token has won nothing
temporal — it has just found a better position, which a per-token method could also
do. The window must beat the **position-oracle** (best single position) *and* a
**stacked all-positions linear read** (aggregation without cross-position binding).
Only then does the win require *combining information across positions* — which is
the sole thing a temporal dictionary can do that a per-token one cannot. This
directly operationalizes the spatial-vs-temporal distinction: relocation of a
signal across positions is beatable by position selection; *integration* across
positions is not.

### The measurement ladder (the synthetic theory, applied to a real task)

Every rung is a read-out of the **anticipation window** (offsets around the
behavior's trigger). Rungs are cheap: linear/MLP probes on raw activations, or on a
small quickly-trained dictionary. This is the P1/P2 conversion ladder from
[[window_length_theory]] instantiated on a real task, so the screen inherits the
theory's interpretation.

| rung | read-out | isolates |
|---|---|---|
| R0 | chance | floor |
| R1 | best single position, linear probe (position-oracle) | honest per-token ceiling |
| R1′ | per-token SAE code at best position | per-token *dictionary* baseline |
| R2 | mean-pooled over window, linear | aggregation, order destroyed |
| R3 | stacked per-position codes + linear (**Stacked SAE**) | aggregation, order kept, no cross-position binding |
| R4 | windowed dictionary (TXC) code + linear | cross-position binding *before* the nonlinearity |
| R5 | MLP on stacked window / task oracle | information ceiling (denominator) |

The decisive quantities, normalized as $S(\cdot) = (\cdot - R1)/(R5 - R1)$ so they
compare across tasks:

- **Anticipation profile** — R1 evaluated at each offset (the per-offset firing
  curve, `b2`/`per_offset_firing`). Its *shape* classifies the task: rising toward
  the trigger = build-up (Shape A, temporal-friendly); flat plateau = persistent
  state (Shape B, weaker lead-time); spike only at/after the trigger = local (per-
  token, reject); flat at chance = no signal / wrong hookpoint.
- **Aggregation headroom** $H_{\text{agg}} = \max(R2, R3) - R1$. Does combining
  positions beat the best single one at all? If $\approx 0$, the task is per-token-
  solvable — **reject** (this is the refusal/sparse-probing failure mode, and the
  exact weakness the reviewers flagged).
- **Order dependence** $R3 - R2$. If stacked $\gg$ pooled, *order* matters — the
  signal is genuinely temporal, not bag-of-positions.
- **Temporal-crosscoder headroom** $H_{\text{txc}} = R4 - R3$. Does binding-before-
  the-nonlinearity beat a Stacked SAE? **This is the decisive rung for us**: if
  $\approx 0$, a Stacked SAE ties the TXC and the crosscoder is not motivated — the
  reviewers' central objection, realized. If $> 0$, cross-position weight sharing is
  doing work no per-token-plus-readout can.
- **Ceiling residual** $R5 - R4$ — information present but not linearly extracted
  (context, not headline).

### Window-length and steering rungs

- **Window-length dependence.** Recompute R4 for $T = 1, 3, 5, \dots$. A monotone
  rise (to a plateau) is the positive signal *and* is itself the reviewer-response
  figure on "window length has no effect"; the plateau length estimates the
  behavior's timescale.
- **Steering headroom (the part that killed refusal).** Two interventions under the
  task's judge, magnitude-swept, scored as genuine-event counts (Δevents):
  - T1 — single-direction difference-of-means at the **best single position**
    (the Arditi-style baseline).
  - T2 — windowed feature (TXC decoder row across the window).
  - $H_{\text{steer}} = \Delta\text{events}(T2) - \Delta\text{events}(T1)$, normalized
    by the *flippable ceiling* (fraction of instances that can be flipped at all by
    the strongest available intervention). If $H_{\text{steer}} \approx 0$, the
    single direction already saturates — a bad *steering* benchmark even if detection
    headroom exists. **This is exactly the refusal diagnosis, made a number.**

### The task-adapter contract

To run the screen on any task, supply five things (everything task-specific lives
here; the rungs are fixed):

1. **Model + hookpoint(s)** and an activation cache.
2. **Positives + matched negatives** (behavior occurs vs not).
3. **Trigger**: the position marking the surface behavior (first "Wait"; first
   refusal token; defection token; first reward-hack action).
4. **Anticipation window**: offsets before the trigger (D+) and matched control
   offsets (D−).
5. **A judge** that scores genuine events in generations (for the steering rungs).

Backtracking already implements this contract (`mine_features` D+/D− over
`offset_window`, `b2` per-offset firing, `b1_steer_eval` magnitude+judge). The
screen is that pipeline with (a) the fixed rung ladder added and (b) a thin adapter
per task.

### Decision rule

A task is a good **temporal-steering** benchmark iff:

1. Anticipation profile is non-degenerate (not spike-only, not flat).
2. $H_{\text{agg}} > \delta$ **and** order matters ($R3 > R2$).
3. $H_{\text{txc}} > \delta$ (TXC beats Stacked SAE).
4. R4 rises with $T$.
5. $H_{\text{steer}} > \delta$ (window beats single-direction steering).
6. Controls pass: per-window shuffle collapses the advantage; chat-template/
   generation correct; D− at chance.

A task can pass for **detection** (1–4, 6) but fail for **steering** (5) — that is
a real and useful verdict, not a null result. Report per-metric.

### What the screen predicts on known tasks (its own validation)

- **Polynomial clock (synthetic):** R1 = chance, R4 = oracle — the canonical pass.
  Anchors the screen against ground truth.
- **Backtracking:** build-up profile ✓; $H_{\text{steer}} > 0$ (paper: Δgc 0.541
  TXC vs 0.400 per-token) ✓. $H_{\text{txc}}$ is *measurable from the existing runs*
  (`stacked_sae` is already in `config.yaml`'s `arch_list`) but not yet reported —
  computing it hands the reviewers the Stacked SAE number they asked for.
- **Refusal (clean):** predicted to fail at (3) and (5) — Arditi saturates T1, so
  $H_{\text{steer}} \approx 0$; may pass detection only, and only in the jailbreak-
  collapse regime where the terminal-token signal drops (R1-at-terminal collapses
  but R1-at-best-position may not — the fairness point decides it).
- **Sleeper armed-state:** predicted *split* — detection $H_{\text{agg}}$ may be
  *small* (a single prompt-final probe already reads defection at AUROC ~99%, so R1
  is high), but $H_{\text{steer}}$ is potentially *large* because no steering/defuse
  baseline exists (T1 undefined/weak). High-steering-headroom, low-detection-
  headroom — the screen surfaces exactly this nuance.

### Why the screen is itself a contribution (not just internal tooling)

It operationalizes the reviewers' meta-question — "is temporal aggregation
*responsible* for the gains?" — as $H_{\text{txc}}$ and the R1→R3→R4 ladder, and its
byproducts *are* the requested figures: the Stacked SAE control ($R3$), the window-
length curve, and the position-oracle honest baseline ($R1$). Running it on
backtracking + one new task simultaneously (a) selects the task and (b) generates
the rebuttal. It also gives the paper a principled "when do temporal dictionaries
help?" instrument, which is more defensible than any single task result.

### Cost & implementation

Screens 1–3 (the detection ladder + window-length) need **no full dictionary
training** — probes on raw activations + one small dictionary, ~1 GPU-day/task.
Only the steering rung needs a mined feature (single-direction T1 is free = DoM;
windowed T2 needs a quick small TXC). So the whole screen is ~1–2 GPU-days/task vs
pod-days for the full experiment. Implementation: a `temporal_screen/` module
forking `ward_backtracking_txc/` with the fixed rung ladder + a per-task adapter;
each candidate (backtracking as the calibrated positive control, refusal, sleeper,
restraint) implements the adapter.

### Open design questions

- Threshold $\delta$: set it from the backtracking positive control and the
  refusal-clean predicted-negative (calibrate so backtracking passes and refusal-
  clean fails on steering).
- For R1′/R4 do we train per-task small dictionaries, or reuse a general-cache
  dictionary (cheaper, tests transfer like Ward's base→distill)?
- Steering ceiling estimation (flippable fraction) — needs a strongest-available
  intervention per task; for some tasks (sleeper defuse) that oracle is itself
  unknown.
