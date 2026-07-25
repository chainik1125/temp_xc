---
author: Claude (Fable 5), for Dmitry Manning-Coe
date: 2026-07-24
tags:
  - design
  - complete
---

## Sprint kickoff — semisynthetic settings where performance improves with window size

10h unsupervised sprint, **wall-clock**: 2026-07-24 21:58 PDT → 2026-07-25 07:58 PDT.
Hourly time checks against `date`. Last hour reserved for `summary.md`.

### Context

Today's session established (see
[[semisynthetic_language_tasks]], `results/temporal_screen/trajectory_full.json`):

- Naturalistic language behaviors are *mode-dominated*: per-token SAE broadcast matches
  or beats a windowed template on ordered-sequence generation (negative result, k≥3).
- Trajectory-profile tasks with multiset-matched foils fix this **by construction**:
  template steering grows ~linearly in k (lang_profile +76→+219, alt_phase +21→+81 over
  k=2→10) while broadcast is pinned at ~0 and a single-segment write decays as 1/k.
- Generation-mode confirmation: 81% per-slot language-profile accuracy (template) vs 44%
  (broadcast), chance 50%.

Main open objections: per-slot additivity makes the teacher-forced k-sweep look
mechanical; the "scheduled SAE" objection (what does the *dictionary* add over an
external schedule); and no bridge yet from these constructed tasks to behaviors anyone
cares about steering in real models.

### Goal

Construct the most compelling semisynthetic setting(s) for temporal-crosscoder steering,
where the headline curve is **performance improving with window size W** — the span of
the steering handle — at fixed control budget, ideally with the required W tracking an
intrinsic task timescale, and with an explicit mapping to real steerable model behaviors.

### Key files

- Branch: `dmitry-semisynth-10h` (this sprint). Prior work on `dmitry-spectral-sprint2`.
- Harness: `experiments/temporal_screen/trajectory_steering/{sol,full,gen2}_modal.py`
- Results so far: `results/temporal_screen/trajectory_{sol,full,gen2}.json`
- Theory inputs: `docs/dmitry/reviewer_responses/window_length_theory.md`,
  `experiments/temporal_screen/synthetic.py` (the clock)
- Real-behavior inputs: `docs/dmitry/reviewer_responses/temporal_safety_tasks_litreview.md`

### Plan (agent team)

- **theory** (subagent): formal conditions; knob-budget theory of the W-sweep
  (prediction: Δ ∝ min(mW, k)); the ℓ-timescale task family and the (W, ℓ) phase
  diagram; entrainment predictions. → `theory.md`
- **realmodel** (subagent): enumerate known steerable behaviors (refusal, sycophancy,
  backtracking, EM, …), classify DC-mode vs trajectory, rank best semisynthetic matches,
  nominate cheap tests. → `real_behaviors.md`
- **review** (subagent): adversarial audit of today's results + sprint designs;
  objection → severity → cheapest killing control. → `review_audit.md`
- **experiment** (main loop): W-sweep (centerpiece), entrainment generation sweep,
  ℓ-timescale family, controls, real-behavior match test.

### Milestones (wall-clock)

- H1: kickoff done; agents launched; W-sweep v1 running on Modal.
- H3: W-sweep + entrainment results; review controls triaged.
- H5: ℓ-timescale phase diagram running; real-behavior match chosen and designed.
- H7: real-behavior match result; hardening (seeds/model size) as budget allows.
- H8–9: figures consolidated; controls closed or honestly logged.
- H9–10: `summary.md` only (red-team/blue-team pass).

### Compute & budget

Modal serverless (auth in `~/.modal.toml`, works — ~$5 spent today). Cap **$50** for the
sprint, burn ≤ $10/h. A10G default; anything larger only for a decisive experiment.

### Judging

Per sprint skill: `summary.md` is the deliverable — 2–5 findings, each carried by one
self-explanatory graph; positive writing; honest limitations.
