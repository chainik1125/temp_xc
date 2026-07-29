---
author: Dmitry
date: 2026-07-23
tags:
  - reference
  - in-progress
---

## Safety-relevant tasks with plausibly temporal steering structure — literature review

Purpose: backtracking is the paper's one real-model demonstration that a *windowed*
dictionary beats a per-token SAE. We want a second such task — a safety-relevant
behavior we can *steer*, where the signal is genuinely temporal so a temporal
crosscoder has a principled edge. This note is a synthesis of a four-cluster
literature sweep (reasoning/CoT, refusal/jailbreak, deception/scheming,
hallucination/persona). Companion: [[window_length_theory]] (the synthetic side),
[[reviewer_responses]] (the reports this feeds).

### The template we are matching (from the paper, §Backtracking)

1. **Multi-token signature** — a *process* spanning positions, not a per-token event.
2. **Per-token SAE provably can't represent the span** — the paper's exact
   justification ("Because the signature spans positions, a per-token sparse
   dictionary cannot represent it").
3. **A causal steering lever exists** — a direction/feature that induces or
   suppresses the behavior.
4. **Lead-time / low-frequency anticipation** — intervening *before* the surface
   trigger works best; in our own FreqBench sprint the backtracking-anticipation
   signal was low-frequency (a slowly-varying mode, not a spike).

The published anchor for our own template: Ward, Lin, Venhoff, Nanda, *Reasoning-
Finetuning Repurposes Latent Representations in Base Models* (arXiv:2507.12638,
verified) — a base-Llama direction induces backtracking in DeepSeek-R1-Distill at
layer 10; the paper reports the effective handle is a window ~8–13 tokens *before*
the "Wait" token. (Our specific ±8–13 number is corroborated internally by the
Llama-Scope feat_71839 finding; cite the offset as ours, the direction as theirs.)

### The one distinction that governs everything: two temporal shapes

The sweep makes clear that "temporal" splits into two shapes, and conflating them
is the trap:

- **Shape A — anticipatory build-up before a discrete trigger.** The signal ramps
  over several tokens and *then* a surface token fires; intervening early beats
  intervening at the trigger. This is backtracking's shape. It is the strong form
  and the rarer one.
- **Shape B — a persistent, slowly-varying state/mode.** A latent (armed /
  evaluation-aware / refusal-under-construction / persona) read out roughly
  constant across a span, with no sharp trigger token. A window reads it *cleaner,
  earlier, and more stably* than a per-token SAE, but the "intervene N tokens
  before the trigger" framing does not apply.

**Caution flagged by three of four clusters:** several "stages of X" / "iterative
refinement" papers describe structure across *layers* (depth), **not** across
*tokens* (time). Do not cite layer-depth stages as temporal lead-time.

### The whitespace, and the honest counter-evidence

- **Whitespace.** Across all four clusters: *every published lead-time result is
  detection; every published steering result is time-agnostic* (a single vector
  added at all positions after the prompt). Backtracking is the sole task that
  pairs a token-level lead-time with a demonstrated steering intervention. So
  "windowed / lead-time *steering* on a new safety task" is genuinely unclaimed.
- **Counter-evidence (must be stated up front in any writeup).** Fomin, David,
  LeVi, *Internal-State Probes Read the Situation, Not the Action* (arXiv:2606.30449,
  verified) is a direct negative result for Shape A in *agentic misalignment*:
  probes "read the situation, not the action," the predictive signal "does not
  strengthen as the model approaches the action token," and prompt-domain is
  decoded at AUC 0.999 while the best future-behavior probe reaches only 0.801.
  Read as: the *persistent-state* framing (Shape B) is well-supported; the
  *build-up-toward-a-trigger* framing is contradicted **for misaligned actions
  specifically**. This tells us which tasks to avoid as headline (scheming /
  agentic harm) and which to trust (refusal, reasoning, eval-awareness state).

### Ranked shortlist

**Tier 1 — best fits (genuine Shape A + mature steering + safety-relevant).**

1. **Refusal / jailbreak "refusal onset" — top pick.** Shape A, and the closest
   published cousin of backtracking. Refusal is *constructed over positions*: the
   harmful-request span carries a high probe score that **collapses at the final
   token** (Doda, *Before the Last Token*, arXiv:2605.12726, verified: harmful-span
   probe ≈ 0.998 vs final-token ≈ 0.174; a PCA-HMM *trajectory* model recovers
   ~94% of missed jailbreaks while **naive max-pooling fails**, flagging benign
   prompts too — a direct argument for a *structured* window over pooling). Hu et
   al., *Tracing the Dynamics of Refusal* (arXiv:2605.02958, agent-fetched, verify)
   report **steering at the "Refusal Onset Token" significantly outperforms the
   final token**, with the causally-effective layer sliding deeper as position
   advances. Steering is mature and off-the-shelf (Arditi et al., arXiv:2406.11717,
   canonical), and **this repo already has refusal steering infra** (`plots/2026-05-13_arditi*`).
   Why it wins: multi-token construction→collapse trajectory a per-token SAE
   structurally misses + mature lever + live infra + we get to claim the
   whitespace (existing refusal work is detection or all-position steering, not
   windowed lead-time steering).
   *Caveat:* the published lead-time results are detection; steering-with-lead-time
   on refusal is our contribution to make.

2. **Reasoning "future-behavior / safety-recovery" family — strong, but adjacent
   to what we already did.** Shape A. Kortukov et al., *Predicting Future Behaviors
   in Reasoning Models Enables Better Steering* (arXiv:2606.11172, agent-verified)
   define *behavior-prediction features* that encode what the model will do before
   it surfaces, with anticipatory steering (FPCG). Ghosal et al., *Safety Recovery…
   Only a Few Early Steering Steps Away* (arXiv:2602.11096) — steering the first
   1–3 reasoning steps drops attack-success 63%→6%. Shen et al. (arXiv:2605.16874)
   — causal levers are early, sparse, planning-related tokens. Safety-relevant
   (restraint / safety-recovery), steerable, anticipatory. *Risk:* same domain and
   author-orbit as our backtracking result — reviewers may see it as incremental.

**Tier 2 — strong steering, but Shape B (persistent state, not build-up).**

3. **Evaluation awareness.** Steering *demonstrated and strong*: Hua, Qin, Marks,
   Nanda, *Steering Evaluation-Aware Models to Act Like They Are Deployed*
   (arXiv:2510.20487, verified) — a contrastive vector suppresses eval-awareness
   and flips proxy behavior (agent: ~80%→~10%); Nguyen et al. (arXiv:2507.01786)
   — SAE-feature steering, probe AUROC 0.829 at Llama-3.3-70B layer 24. Hottest
   2025–26 subarea, very safety-relevant. *Frame honestly:* a persistent context
   state, so the claim is "window reads a stable latent earlier/cleaner," not
   lead-time. No repo infra.

4. **Sleeper-agent "armed state."** Shape B **plus** a genuine trigger→action
   latency. Detection confirmed (MacDiarmid et al., Anthropic *Simple probes can
   catch sleeper agents*, 2024; model: Hubinger et al., arXiv:2401.05566,
   canonical — AUROC ≈ 99% from the prompt's final token, before any harmful
   token). **Steering / defusing the armed state is entirely untried** — the
   biggest open-novelty slot in the whole sweep — and this repo has a sleepers
   stream already. *Cost:* needs backdoored models (may exist in the sleepers
   stream already).

**Tier 3 — weaker temporal fit or shakier evidence.**

5. **Emergent misalignment / persona.** Shape B slow mode, steerable (persona
   vectors, arXiv:2507.21509; EM persona features, arXiv:2506.19823; origin
   Betley et al., arXiv:2502.17424 — all canonical), and **repo has an EM stream**.
   But timescale is turns/hundreds of tokens, no sharp trigger, no token-level
   temporal analysis published. Cheap for us given infra; weak on the temporal
   claim.
6. **Strategic deception / lying.** Steering confirmed (arXiv:2509.18058, 5%↔96%;
   probes Apollo arXiv:2502.03407; MASK arXiv:2503.03750). But its "stages" are
   layer-depth, not token-time — windowing is a *denoising* argument here, not a
   lead-time one.
7. **Hallucination onset.** Shape A with a published *negative detection delay*
   (~11 tokens before onset, "text-novelty drift not surprisal") and steering
   handles (ACT, arXiv:2406.00034; semantic entropy, Farquhar et al. Nature 2024).
   But the sharpest lead-time paper is an unrefereed Research Square preprint —
   replicate before building.
8. **Scheming / alignment-faking / sandbagging.** Highest stakes, weakest evidence,
   and partly *contradicted* for latent build-up by arXiv:2606.30449. Do not
   headline; alignment-faking's scratchpad build-up (Greenblatt et al.,
   arXiv:2412.14093) is token-visible CoT, and *The Refusal Residue*
   (arXiv:2607.13346) explicitly asks for multi-token extraction — a windowed
   read is the natural instrument, but no steering on the faking decision exists.

### Recommendation

Make **refusal** the second temporal safety task. It is the only candidate that
hits all four template criteria with published support *and* has live
infrastructure in this repo: a multi-token construction→collapse trajectory
(2605.12726) that a per-token SAE provably misses, an onset > final-token
steering asymmetry (2605.02958), a mature off-the-shelf lever (2406.11717), and
the `arditi` steering harness already here. It also lets us plant a flag in the
whitespace — *windowed, lead-time refusal steering* — since all prior refusal
work is either detection or all-position steering.

Keep two hedges: (i) the eval-awareness (2510.20487) or sleeper-armed-state
routes are the higher-novelty Shape-B bets if we want to steer a *persistent*
safety state rather than a build-up; (ii) whatever we pick, cite Fomin et al.
(2606.30449) as the honest boundary — Shape A does not generalize to agentic
misaligned actions, so we must show our chosen task genuinely has the build-up,
not assume it.

### Citation ledger (confidence flags)

- **Verified this session (fetched, title+authors confirmed):** 2507.12638
  (Ward et al.), 2605.12726 (Doda), 2510.20487 (Hua et al.), 2606.30449
  (Fomin et al.).
- **High-confidence canonical (well-established, not re-fetched):** 2406.11717
  (Arditi refusal), 2401.05566 (Hubinger sleeper agents), 2502.17424 (Betley EM),
  2507.21509 (Anthropic persona vectors), 2506.19823 (OpenAI persona features),
  2312.06681 (Rimsky CAA), 2412.04984 & 2502.03407 (Apollo scheming / deception
  probes), 2503.03750 (MASK), 2304.13734 (Azaria–Mitchell), 2412.14093
  (Greenblatt alignment faking), Farquhar et al. semantic entropy (Nature 2024).
- **Agent-surfaced, VERIFY arXiv id before any external use:** 2605.02958 (Hu et
  al. refusal dynamics — load-bearing for the onset claim, verify first),
  2606.11172, 2602.11096, 2605.16874, 2507.01786, 2509.18058, 2607.13346,
  2406.00034, and the hallucination-onset Research Square preprint (unrefereed).

### Open questions for the pause

- Refusal vs eval-awareness vs sleeper-armed-state as the second task — steering
  maturity + repo infra favor refusal; novelty favors sleeper-defuse.
- Do we need to *first* establish the lead-time empirically on our chosen task
  (a position-resolved steering-strength curve, the refusal analogue of the
  backtracking −13..−8 offset), before claiming the windowed advantage?
- Is the goal a *steering* win (induce/suppress) or a *detection* win (probe
  PR-AUC), or both, matching the paper's two backtracking metrics?
