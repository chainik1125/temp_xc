---
author: Agent Y (via Aniket)
date: 2026-04-28
tags:
  - design
  - in-progress
---

## Y orientation log — picking up from agent_y_brief

> Drop-in note from Agent Y on `aniket-phase7-y` (branched off
> `han-phase7-unification` at `a05b596`, no edits to Han's branch).
> Confirms inheritance, locks in the plan, flags assumptions to
> verify before any pod work starts.

### Context absorbed

Read in full:

- `agent_y_brief.md` — the Y-specific brief, including Dmitry's
  protocol controversy, Q1/Q2 agenda, and 50%-time TXC-win mandate.
- `brief.md` (paper-wide) — twelve leaderboard archs, two subject
  models (gemma-2-2b base L12, IT L13), `paper_archs.json` source-
  of-truth, A40+H200 split, three concurrent agents on this branch.
- `2026-04-26-c1-hh-rlhf-stage1.md` — Agent C's HH-RLHF pass.
- `2026-04-26-c2-steering-stage1.md` — Agent C's AxBench-additive
  steering pass (six archs, the v1 Pareto plot).
- `2026-04-26-agent-c-stage1-synthesis.md` — C's TL;DR claim that
  TXC family Pareto-dominates AxBench-style steering.
- `dmitry-rlhf:docs/dmitry/case_studies/rlhf/summary.md` — Dmitry's
  paper-clamp reproduction, headline result: T-SAE k=20 wins peak
  success on **both** protocols (1.93 paper / 2.00 AxBench), window
  archs lag 0.5–0.8 under paper-clamp.
- `dmitry-rlhf:docs/dmitry/case_studies/rlhf/notes/methodology.md`
  — formal definition of paper-clamp vs AxBench-additive,
  window-arch generalisation, the 5× activation-magnitude argument.
- The three intervention scripts on this branch:
  `intervene_paper_clamp.py`, `intervene_paper_clamp_window.py`,
  `intervene_axbench_extended.py`.

### One-line restatement

**The architectural ranking depends on the steering protocol.** Under
AxBench-additive (decoder-direction × strength), TXC family
clusters in the Pareto upper-right at moderate strength. Under
paper-clamp (z-clamp + error-preserve), T-SAE k=20 wins peak
success by 0.5–0.8. Dmitry's analytical answer: window-encoder z[j]
magnitudes are O(T × per-token) ≈ 5× larger, so the paper's strength
schedule that's "10× typical" for per-token archs is only "2× typical"
for window archs. Y's job is to verify this empirically, find a
defensible TXC protocol, and (50% of bandwidth) hunt for a case
study where TXC genuinely wins.

### Plan — Q1 (WHY) magnitude-scale verification

Pre-registered before any pod execution so the negative outcome (no
catch-up) is also publishable.

**Q1.1 — `<z[j]_orig>` distributions.** For each of the six archs
already with feature_selection.json under `results/case_studies/steering/<arch>/`:

  - Pull the picked feature index `j*` per concept.
  - Re-run encoder over the 30 concepts × 5 examples probe set.
  - Record `z[j*]_orig` at every right-edge token (window archs) or
    every token (per-token archs).
  - Plot KDE / histogram per arch on a shared axis. Compute median
    and IQR.
  - Hypothesis pass condition: window-arch medians are ≈ 5× per-
    token medians (with at most 2× scatter).

**Q1.2 — success-vs-strength curves.** Already partially done by
Dmitry (his per_arch_breakdown.md). My re-cut: for each arch, plot
success(strength) on a *log* axis and fit peak strength
`s*_arch`. Test whether `s*_window / s*_per-token ≈ T`. If
linear-in-T, the magnitude story is the *full* story; if
non-linear (e.g. window archs need >> 5× strength for the same
relative push), there's a second factor — likely error-preserve
interaction with magnitude.

**Q1.3 — family-normalised paper-clamp re-run.** Re-run paper-
clamp on the three window archs at strengths

  ```
  s_norm = s_paper × <z[j]_orig>_arch / <z[j]_orig>_T-SAE-k20
  ```

over the same 30-concept set. Sonnet 4.6 grader, same prompts. If
TXC catches up to T-SAE k=20 within concept-variance noise, the
hypothesis is empirically confirmed. Output goes to
`results/case_studies/steering_paper_normalised/<arch>/`.

Single seed, single pass. Cost: ≈ 2700 grader calls × 3 archs (no
need to re-run per-token archs). Same Sonnet 4.6 path Dmitry used.
Budget under $10 incremental.

### Plan — Q2 (HOW) defensible TXC protocol candidates

Y will judge between four candidates after Q1 lands:

| candidate | rationale | when to recommend |
|---|---|---|
| (A) AxBench-additive as canonical | already runs uniformly across families | if Q1.3 fails AND no per-position variant rescues |
| (B) Per-family strength scaling on top of paper-clamp | preserves the paper's protocol | if Q1.3 succeeds (closes the gap to within noise) |
| (C) Per-position window clamp (vs current right-edge only) | tests whether "where in the window" matters | as a fallback if (B) doesn't close the gap |
| (D) Train TXCs differently | this is Z's territory; flag if Y finds something promising | only if (A)–(C) all fail |

The choice goes into the synthesis log
`2026-04-29-y-tx-steering-final.md`. Han makes the final call.

### Plan — 50% TXC-win hunt

Methodology: many ideas, fast smoke-tests on 1–2 archs (TXC T=5 +
T-SAE k=20), kill candidates that don't show decisive lead, document
each candidate (win or loss) in
`docs/han/research_logs/phase7_unification/2026-04-2*-y-csN-<topic>.md`.

Top of my candidate stack, in order of estimated TXC-prior fit:

1. **Persistent / slow features over context.** TFA / autoencoding-
   slow paper framing — features that should be on for many
   consecutive positions, not just one. Measure: dwell-time
   distribution per active feature across long contexts. TXC's T-
   window encodes this natively; per-token has to relearn it at
   every position. Cheap to prototype on existing FineWeb cache.

2. **Multi-token concept extraction probe.** Predict span-level
   labels (named entities, multi-word idioms, syntactic spans).
   AUC at fixed `k_feat` across archs. Hypothesis: TXC needs fewer
   features to hit a given AUC. Caveat: SAEBench-side workstream
   already plans this for sparse probing; coordinate to avoid
   duplicating Chanin's probing infrastructure.

3. **Anaphora / coreference probing.** Predict from feature subset
   whether two tokens corefer. Phase 5's `wsc_coreference` is the
   small-scale precedent. Honest expectation: TXC should win if
   T ≥ 3 includes both anaphor and antecedent in the receptive
   field — but on natural text the antecedent is often > T tokens
   away, so this might not pan out without longer T.

4. **Activation reconstruction at masked positions.** Mask 1
   token, reconstruct from neighbour activations using SAE features
   only. TXC's encoder has the unmasked context built in; per-
   token has to bridge. MSE per arch.

5. **Multi-token concept counterfactual editing.** "Replace the
   named entity in this sentence via SAE-feature manipulation".
   Per-token has to manipulate every token of the span; TXC can
   do it in one feature. This is the *steering-specific*
   structural argument Y already plans to surface in Q2.

I'll spawn one log per candidate. After 3–5 candidates I'll write
`2026-04-2*-y-cs-synthesis.md` ranking by TXC margin and propose
≤ 2 to keep as paper-grade case studies.

#### What I'll NOT spend time on first

- Anything that requires retraining TXCs (that's Z, per Han's split).
- Probing-pipeline extensions beyond what already exists in
  `experiments/phase7_unification/` (X owns that).
- Re-running Dmitry's paper-clamp grids — they're done, I'll reuse
  the rows from `results/case_studies/steering_paper/<arch>/grades.jsonl`.

### Assumptions I will verify before pod work

Per the brief's three explicit assumptions:

1. **Steering pipeline runs end-to-end.** First action on the pod:
   `python -m experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_window --arch agentic_txc_02 --limit-concepts 1 --force` and confirm a non-empty `generations.jsonl`. If broken, the
   blocker log goes up before any science.

2. **`tsae_paper.py` faithful port loads cleanly.** Smoke-test:
   load `tsae_paper_k20__seed42.pt` from HF, encode a batch, sanity-
   check active-feature count is k=20. The faithful-port comment in
   the source claims it; I'll verify rather than trust.

3. **Anthropic API access (Sonnet 4.6) is available** with prompt
   caching and ThreadPool=8. If the API key is rotated, fall back
   to a local grader.

If any of these fails, `2026-04-28-y-blockers.md` goes up first.

### Branch + worktree mechanics (my side, not paper-relevant)

Local Mac is disk-pressured (92% full) and Spotlight is in an
"unknown indexing state" — full worktree checkout hangs in
`mmap(loose-object)` syscalls. Workaround for this session: I'll
work via git plumbing on `aniket-phase7-y`. Real work happens on
the A40 pod where checkout is unconstrained. No commits go to
`han-phase7-unification`; everything is on `aniket-phase7-y` until
Y's deliverables land back on the unification branch via PR or
direct merge (Han's call).

### Things I want Han's input on before executing

1. **Can I write Q1 outputs to `results/case_studies/` paths even
   though they're under his unification branch?** The brief says
   `case_studies/` is Y's territory but Y outputs originally went
   to `results/case_studies/steering_paper/<arch>/`. I'll mirror
   that under `steering_paper_normalised/<arch>/` rather than
   adding a `_y_*` prefix unless Han prefers the prefix for
   provenance.

2. **For the 50% TXC-win hunt, is the persistent/slow-feature
   probe (#1 in my list) high enough priority to start there
   before Q1.3 finishes?** The Q1 work and the slow-feature probe
   don't share data products, so they could run in parallel. My
   default: start Q1.1 + Q1.2 immediately (they reuse Dmitry's
   existing grades), kick off Q1.3 generation as a long-running
   job, and use the gap to draft the slow-feature probe.

3. **30 concepts: keep, or expand?** Han's caveat lists "30
   concepts is small" as a stage-1 limitation. If Q1.3 doesn't
   close the gap, expanding to 60–100 concepts on the survivor
   protocol would tighten the Pareto-noise band. Cost is ≈ 3–5×
   the current grader budget. Opting in only if the small-set
   result is ambiguous.

### What "absorbed" means here

Y now has the full chain of: paper protocol → C's apples-to-
apples choice (AxBench-additive) → C's TXC-Pareto-dominates
claim → Dmitry's paper-clamp reproduction overturning that claim
under the original protocol → Dmitry's magnitude-scale
explanation. Y's job is the empirical verification (Q1) +
defensible-protocol recommendation (Q2) + an independent search
for a TXC-win case study to balance the steering result.
