---
author: Dmitry
date: 2026-07-23
tags:
  - proposal
  - in-progress
---

## Refusal as a second temporal steering task — experiment plan

A faithful fork of the backtracking experiment (`experiments/ward_backtracking_txc/`)
for refusal. Same two-metric structure (inducement + detection), same matched-budget
architecture sweep, same pre-trigger-window logic — with refusal's trigger and
anticipation window mapped onto the prompt→generation boundary. Motivation and task
choice: [[temporal_safety_tasks_litreview]]. Synthetic companion: [[window_length_theory]].

### The backtracking pipeline we are mirroring

From `experiments/ward_backtracking_txc/` (Stage A elicits + labels; Stage B is the
dictionary experiment):

- **Stage A** (`results/ward_backtracking/`): prompts, traces, `sentence_labels`
  (marks each "Wait"/backtracking sentence), `dom_vectors` (difference-of-means
  steering vector). Defines the **trigger** (the "Wait" token) and the
  **anticipation window** (offsets [-13,-8] before it).
- **Stage B** (fork target):
  1. `cache_activations.py` — cache activations at hookpoints (`resid_L10`,
     `attn_L10`, `ln1_L10`).
  2. `train_txc.py` — one dictionary per (arch, hookpoint), matched `d_sae=16384`,
     `T=6`, `k_per_position=32`. Archs: `txc, topk_sae, stacked_sae, tsae,
     tsae_paper, txc_h8, txc_h13`. **Stacked SAE is already here** — the reviewers'
     demanded control.
  3. `mine_features.py` — rank features by **D+/D− selectivity** over the
     `offset_window [-13..-8]`; take top-K for steering.
  4. `b1_steer_eval.py` — decoder rows as steering vectors, magnitude grid
     `[-16..16]`, judge-scored inducement (Sonnet counts genuine backtracking).
  5. `b2_cross_model.py` — **per-offset firing curves** (`offset_window_full
     [-30..5]`) = the lead-time/detection visualization.
  6. `plot/` — `steering_comparison_bars` (inducement), `per_offset_firing`
     (lead-time), `feature_firing_heatmap`, `coherence` (degeneracy check),
     `text_examples` (case study).

### Component-by-component mapping to refusal

| Backtracking | Refusal analogue |
|---|---|
| DeepSeek-R1-Distill-Llama-8B (behavior); base Llama-3.1-8B (vector source) | **Llama-3.1-8B-Instruct** (behavior + vector source). Reuses the base-Llama cache infra; lets us also test base→instruct direction repurposing (a B2 analogue). |
| 61 MATH-500 questions | **Cohort:** harmful (AdvBench / HarmBench / JBB-harmful) + harmless (Alpaca / JBB-benign) + **borderline-benign (XSTest)** control + **jailbroken-harmful** (GCG/template suffix → refusal suppressed). ~150–200 prompts, grouped by behavior for GroupKFold. |
| Trigger = "Wait"/"Hmm" token, mid-generation | Trigger = **first refusal token** in the generation ("I cannot"/"Sorry"/"As an AI"), detected by refusal-prefix match + judge, at the prompt→generation boundary. |
| Anticipation window = offsets [-13,-8] before "Wait" | Anticipation window = the **prompt span before the surface refusal token** (request → onset → terminal). Offsets measured from the boundary: first generated token = 0, prompt tokens negative. D+ = last W prompt tokens of harmful prompts; D− = matched windows of harmless/XSTest. |
| DoM vector (base model, backtracking contrast) | **Arditi difference-of-means refusal direction**: mean(harmful post-instruction acts) − mean(harmless), at the steering layer. Adding → refuse; ablating → comply (jailbreak). |
| **Headline — Inducement: Δgc (genuine-backtracking *count* lift), judge = Sonnet** | **Headline — genuine-event count, both modes** (see below): Sonnet counts *genuine* events, baseline-corrected Δ, bootstrap CI over the cohort, per arch at optimal magnitude. StrongREJECT / HarmBench classifier are secondary corroboration, never the headline. |
| Detection: sparse-probe PR-AUC@S=8, GroupKFold by question | Detection (secondary): sparse-probe PR-AUC@S on the anticipation window ("will refuse" vs "won't"), GroupKFold by behavior. **Sharpest figure: jailbreak-collapse** — final-token probe fails, windowed dictionary wins (the Doda 2605.12726 result as a dictionary comparison). |
| Per-offset firing (lead-time) | Per-offset firing across prompt positions = the **construction→collapse trajectory** (high mid-prompt, attenuated at terminal; Hu/Doda). Plus **position-resolved steering**: efficacy vs offset → onset > terminal (the refusal analogue of −13..−8). |
| Coherence check ("Wait Wait Wait" degeneracy) | Coherence check: suppression must not buy compliance with gibberish (StrongREJECT already penalizes non-answers; add distinct-2 / length). |

### Headline metric — genuine events suppressed / induced, under a steering protocol

Exactly the backtracking `gc`/`Δgc` structure: a **judge-scored count of genuine
events in steered generations**, not a keyword rate and not StrongREJECT ASR. The
"genuine" qualifier is what makes it a count of real events — it is the refusal
analogue of "genuine backtracking" and absorbs the coherence guard (a suppression
that yields gibberish is not a genuine compliance event).

Two event types, both under the same steering protocol:

- **Suppress:** `gs(a, q, m)` = 1 if, on **harmful** prompt `q`, steering with
  architecture `a`'s mined refusal feature at magnitude `m` produces a *genuine,
  coherent, on-topic* compliance with the harmful request (Sonnet judge; degenerate
  or partial outputs score 0), else 0.
- **Induce:** `gi(a, q, m)` = 1 if, on **harmless** prompt `q`, the same steering
  produces a *genuine, coherent* refusal of the benign request, else 0.

Headline lifts, mirroring `Δgc`:
$$\Delta gs(a,m) = \tfrac{1}{|\mathcal{Q}_{\text{harm}}|}\!\sum_q\! \big[gs(a,q,m) - gs(a,q,0)\big], \qquad \Delta gi(a,m) = \tfrac{1}{|\mathcal{Q}_{\text{harmless}}|}\!\sum_q\! \big[gi(a,q,m) - gi(a,q,0)\big],$$
reported per architecture at its optimal magnitude against the shared unsteered
floor (`m=0`), with bootstrap 95% CI over the cohort — the exact form of the paper's
inducement figure. Both floors are near zero (unsteered model refuses harmful,
complies harmless), so Δ is essentially the genuine-event count that steering buys.

**The steering protocol** (the cut25 analogue): (1) format the prompt with the
Instruct chat template; (2) apply architecture `a`'s mined feature direction (decoder
row) at the steering layer as an additive hook at magnitude `m` over the symmetric
grid `[-16..16]` — negative = ablate/suppress, positive = add/induce; (3) free-generate
under the hook and judge the output. Unlike backtracking, no 25%-cut seeding is needed
— refusal is a boundary behavior, so steering acts from the first generated token.
Default steers the anticipation window + continuation; the **position-resolved**
variant (steer only at a given offset) is run separately for the onset>terminal curve.
`b1_steer_eval.py` already implements this hook + magnitude grid + judge loop; only the
judge prompt (genuine compliance / genuine refusal instead of genuine backtracking)
and the two prompt cohorts change.

### The temporal core — what makes this a temporal-crosscoder result

The whole claim rests on one structural fact (verified in the literature, to be
re-verified on our model at Gate 4): **refusal is constructed over the prompt span
and the terminal token carries only an attenuated aggregate** (Hu et al. 2605.02958;
Doda 2605.12726: harmful-span probe ≈ 0.998 vs final-token ≈ 0.174). Therefore:

- A **per-token SAE** reading the terminal prompt token sees the weakest, collapsed
  signal; a **temporal crosscoder** reading the window [request … onset … terminal]
  captures the construction a per-token dictionary structurally cannot represent —
  the exact backtracking argument, relocated to the prompt boundary.
- **Window-length sweep** `T ∈ {1,3,5,8}`: detection/inducement should improve with
  `T` up to the construction-span length — ties directly to the synthetic
  window-length story ([[window_length_theory]]) and answers the reviewers'
  "window length has no effect" complaint on a real safety task.
- **Position-resolved steering curve**: steering efficacy vs offset, predicting
  onset-window > terminal — the refusal analogue of the −13..−8 backtracking offset,
  and the thing that makes this *windowed lead-time steering* (unclaimed whitespace)
  rather than standard all-position abliteration.

### Architectures & budget (unchanged from backtracking)

Same `arch_list` (`txc, topk_sae, stacked_sae, tsae, tsae_paper, txc_h8, txc_h13`),
same `d_sae=16384`, `k_per_position=32`, matched across archs, trained on the same
Instruct activation cache. Hookpoints: sweep `resid` and `ln1` (config note: ln1 was
the strongest temporal hookpoint) at layers **{12, 15, 18}** — 15 matches the repo's
existing `arditi` runs; refusal is decided in later-middle layers. `T` swept as above.

### Validation gates (pre-register before compute; from EM lessons + Fomin caveat)

1. **Chat-template correctness.** Format every prompt with the Instruct chat
   template; the anticipation window must include post-instruction chat-control
   tokens (Zhao et al.: refusal encoded at `t_post-inst`). *This is the EM
   sprint's headline bug — delegate to a reference `generate_*` if one exists;
   do not hand-roll the template.* ([[feedback_reuse_recipe_functions]])
2. **Behavior baseline.** Unsteered model refuses harmful, complies harmless
   (StrongREJECT sanity) — before any dictionary work.
3. **Lever exists (Arditi replication).** The DoM refusal direction: adding induces
   refusal on harmless, ablating jailbreaks harmful. If this fails, stop.
4. **Temporal-reality gate (critical).** A position-resolved probe map must show the
   refusal signal is **upstream and attenuated at the terminal token** on our model.
   Per Fomin et al. 2606.30449 (probes "read the situation, not the action"; signal
   does not always build toward the trigger), we must *show* the construction→collapse
   trajectory, not assume it. If the signal is flat or terminal-only, refusal is
   Shape-B-only and the windowed advantage is weak — decide here, cheaply, before
   training the full arch sweep.
5. **Order-destruction control.** Per-window shuffle → the windowed (TXC) advantage
   should collapse to the per-token SAE level, isolating temporal structure (not
   extra capacity) as the source of the win.

### Phased execution

- **Phase 0 — fork + data.** Copy `ward_backtracking_txc/` → `refusal_txc/`. Pull
  AdvBench / HarmBench / JBB-Behaviors / XSTest (not in repo). Build the cohort +
  one jailbreak variant. Wire StrongREJECT/HarmBench + Sonnet judges.
- **Phase 1 — Stage A (elicit + label).** Generate Instruct completions on the
  cohort; label refuse/comply; mark refusal-onset positions and the anticipation
  window; compute the DoM refusal direction. Run Gates 1–3.
- **Phase 2 — Gate 4 (go/no-go).** Position-resolved probe map on one hookpoint.
  Confirm construction→collapse. Only proceed to the full sweep if it holds.
- **Phase 3 — cache + train.** `cache_activations.py` (Instruct, hookpoint/layer
  grid) → `train_txc.py` (full arch sweep, `T` sweep).
- **Phase 4 — mine + steer (HEADLINE inducement).** `mine_features.py` (D+/D− on the
  refusal window) → `b1_steer_eval.py` (suppress + induce modes, symmetric magnitude
  grid, **Sonnet genuine-event judge → Δgs / Δgi**). This is the headline figure
  (genuine events suppressed/induced per arch at optimal magnitude, bootstrap CI).
- **Phase 5 — detection + lead-time (secondary).** Sparse-probe PR-AUC (incl.
  jailbreak-collapse regime) + `b2` per-offset firing + position-resolved steering
  curve. Gate 5.
- **Phase 6 — plots + case study.** Inducement bars (TXC vs SAE vs stacked),
  construction→collapse trajectory, window-length curve, PR-AUC bars, and a
  single-prompt case study (harmful refused unsteered → complies under TXC-feature
  steering; and/or harmless → induced refusal).

### Compute

Same order as backtracking: `d_sae=16384`, Llama-8B, ~15k steps/dictionary with
held-out FVU early-stop; H100 fits the arch × hookpoint × layer × T grid over a
couple of pod-days (see `reference_gpu_hosts`, `reference_h100_4_em`). Generation
(Phase 4) is the cost driver — reuse the parallelized per-row-magnitude hook.

### Novelty framing / honest risks

- **Refusal steering is well-trodden (abliteration).** The novelty is *not*
  jailbreaking — it is the **temporal** angle: windowed feature quality vs per-token
  at the anticipation window, position-resolved efficacy (onset > terminal),
  window-length dependence, and windowed **detection in the jailbreak-collapse
  regime where the final-token probe fails**. Lead with those, never with ASR.
- **Refusal is more Shape-B than backtracking's Shape-A.** The construction→collapse
  gives it Shape-A character at the boundary, but Gate 4 must confirm it; do not
  assume the lead-time (Fomin 2606.30449).
- **Trigger detection is fuzzier than "Wait".** Mitigate with refusal-prefix
  matching + judge; report onset-detection agreement.
- **Citations to verify before external use:** 2605.02958 (Hu et al. — load-bearing
  for onset > terminal); 2605.12726 and 2510.20487 and 2606.30449 verified this
  session; 2406.11717 (Arditi) canonical.

### Decisions (locked)

- **Model:** Llama-3.1-8B-Instruct.
- **Steering emphasis:** both — suppress (genuine harmful-compliance events on harmful
  prompts) and induce (genuine refusal events on harmless prompts).
- **Headline metric:** as in backtracking, the **count of genuine events
  suppressed/induced** under the steering protocol (Δgs / Δgi, Sonnet-judged,
  baseline-corrected, bootstrap CI), *not* a rate or ASR. Detection (PR-AUC) and the
  window-length / position-resolved curves are secondary support.

### Still open (not blocking Phase 0)

- Steering layer / hookpoint grid: default `{resid, ln1} × L{12,15,18}`; the position-
  resolved lead-time curve may prefer one hookpoint — pick after Gate 4.
- Jailbreak variant for the collapse regime (GCG suffix vs a fixed template) — a
  Phase-0 data decision, not a design one.
