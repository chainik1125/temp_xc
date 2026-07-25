---
author: Claude (Fable 5), for Dmitry Manning-Coe
date: 2026-07-25
tags:
  - design
  - in-progress
---

## Sprint kickoff — does a temporal crosscoder beat a TopK SAE at steering a trajectory?

10h unsupervised sprint, **wall-clock**: 2026-07-25 16:19 PDT → 2026-07-26 02:19 PDT.
Hourly checks against `date`. Last hour reserved for `summary.md`.

### The question, and why it is the right one

The previous sprint ([[2026-07-24_semisynth_10h/summary|semisynth 10h]]) established that
steering a *trajectory* needs a schedule rather than a level, quantified how a
resolution-limited handle degrades, and closed the adjacency question. It could not make
a claim about **dictionaries**, and said so as its largest limitation:

> The per-position template is rank-1 (σ₁ = 89% of the energy) — one direction with an
> externally supplied sign schedule. Everything here is about the *form of the control
> signal*: a schedule beats a level. Whether a temporal dictionary beats a per-token one
> needs trained dictionaries and is the main thing this sprint does not answer.

That is this sprint. The gap is specific: last time the *schedule came from ground truth*.
A per-token SAE user could hand-schedule coefficients and match us exactly. The claim only
becomes about dictionaries when the schedule is **read off a decoder** rather than
supplied.

### Goal

Train a temporal crosscoder and a TopK SAE on one activation cache at pre-registered
matched sparsity, steer the trajectory tasks **from decoder rows**, and measure how much
of a target trajectory each dictionary can write at a matched budget of active latents.

### What already exists (checked before scoping)

The bench harness is built and this is why 10h is enough:

- `src/bench/architectures/crosscoder.py` — `TemporalCrosscoder(d_in, d_sae, T, k)`,
  encodes a T-token window into a shared latent, and critically exposes
  **`decoder_directions_at(pos)`** — the per-position rows of a latent's decoder. That is
  exactly the W×d pattern a steering write needs.
- `src/bench/architectures/topk_sae.py` — `TopKSAE(d_in, d_sae, k)` with
  `decoder_directions`.
- `src/bench/saebench/matching_protocols.py` — **pre-registered** sparsity matching:
  **Protocol A** (per-token k matched: SAE k=100, TempXC k=100 — asks "are TempXC's
  individual features better?") and **Protocol B** (total window budget matched: TempXC
  k = 100·T — asks "is TempXC's representation as a whole better?"). Both will be run;
  reporting only the flattering one would be the obvious way to fool ourselves.
- `src/bench/data.py::build_cached_activations_pipeline` — trains on cached real LM
  activations.
- `experiments/temporal_screen/trajectory_steering/` — the eval: teacher-forced log-prob
  margin against a **multiset-matched foil**, plus the arms and controls from last sprint.

### Design decisions, locked now

- **Subject model** Qwen-2.5-1.5B-Instruct, layer 14 — identical to all prior work, so the
  difference-of-means (DoM) result is directly comparable rather than a re-run.
- **Task** the intensity axis (tense/calm) at k=12 as primary, since `alt_phase` and the
  (W, ℓ) family are the best-characterised; language (EN/FR) secondary.
- **Training corpus** a *mixture*: the task distribution (random tense/calm profiles) plus
  general text. Training only on the task distribution would manufacture a feature that
  is the task, which proves nothing. The mix ratio is a reported parameter.
- **Three arms, not two.** TXC and SAE, **plus the DoM direction as the incumbent**. If
  trained dictionaries lose to a difference-of-means vector, that is the finding, and it
  would matter more than a TXC-vs-SAE ordering.
- **Feature selection must be matched.** Whichever rule picks the steering latent
  (e.g. highest activation difference between tense and calm segments) is applied
  identically to both dictionaries, and stated before results.

### The measurement

At a budget of **m active latents**, how much of the target trajectory can each dictionary
write? A TXC latent spans W positions for one coefficient; an SAE feature is one direction
and costs one coefficient per position, or must be broadcast. So:

    fidelity(m) = Δmargin(write built from m latents) / Δmargin(full ground-truth schedule)

swept over m. Registered prediction: **TXC > SAE at small m for structured trajectories,
converging as m → k**. Registered falsifier: **if SAE features with per-position
coefficients match TXC at matched m, the dictionary claim fails** and only last sprint's
"a schedule beats a level" survives.

### Milestones

- H1: smoke — cache a few hundred activations, train tiny dicts (d_sae small, few hundred
  steps), steer from one decoder row, confirm the plumbing end to end. **No long run until
  this passes.** (Last sprint's clearest lesson: three multi-hour designs were confounded
  and a 3-minute smoke would have caught two of them.)
- H2–H4: real activation cache; train SAE + TXC under Protocols A and B.
- H4–H7: steering eval from decoder rows, m-sweep, DoM incumbent arm.
- H7–H8: controls the review agent demands; robustness.
- H8–H9: figures.
- H9–H10: `summary.md` only.

### Compute and budget

Modal serverless. **Known risk from last sprint: another project on the same account held
all 10 containers for hours.** Mitigations: check `modal app list` before long jobs, keep
jobs short and resumable, cache activations once and reuse, run detached so a harness
timeout cannot kill a queued job. Cap **$60**, ≤ $12/h.

### How this will be judged

Same standard as last sprint: `summary.md` is the deliverable — 2–5 findings each carried
by one self-explanatory graph, honest limitations, and every number verified against the
raw JSONs. Adversarial review and a blind fresh-eyes figure test before finalising.
