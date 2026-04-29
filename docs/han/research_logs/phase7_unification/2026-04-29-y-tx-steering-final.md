---
author: Han
date: 2026-04-29
tags:
  - design
  - in-progress
---

## Phase 7 steering — final framing recommendation (Agent Y)

> Closes the steering-protocol controversy raised by Dmitry's reproduction
> (`origin/dmitry-rlhf`). Companion to
> `2026-04-29-y-tx-steering-magnitude.md` (Q1.1/Q1.2/Q1.3 evidence).
> Han makes the final call; Y provides recommendation + evidence.

### Status (2026-04-29)

- Q1.1 (magnitude scan): **confirmed** quantitatively — window archs'
  picked-feature `<|z|>` is 3-16× per-token. MLC's 5-layer fusion gives
  the largest ratio (~16×).
- Q1.2 (predicted peak strengths): **confirmed** — predicted
  `s_peak ≈ 10 × <|z|>` matches Dmitry's observed peaks to within
  strength-grid resolution.
- Q1.3 (per-family normalised paper-clamp): **pending grading**.
  Pipeline validated on 2 per-token archs; TXC + SubseqH8 grades expected
  in ≤30 min.

### Three plausible framings (decision matrix)

The paper's steering case-study currently has three live framings.
Han chooses; Y recommends below.

#### Framing A: "Both protocols, both winners" (lowest-defence, safest)

- Report AxBench-additive (Agent C's protocol): TXC family wins moderate-strength.
- Report paper-clamp (Dmitry's reproduction of Ye et al. App B.2): T-SAE k=20 wins.
- Methods caveat: "the architectural ranking is protocol-sensitive."
- **Defence**: cleanest under reviewer scrutiny — no novel protocol claim.
- **Cost**: gives up the strongest narrative; reviewers may ask "so which is right?"

#### Framing B: "AxBench is canonical, paper-clamp is biased" (medium-defence)

- Argue AxBench-additive is the only protocol that doesn't conflate
  activation-magnitude scale with feature quality. Use it as the
  headline; report paper-clamp as a secondary view with the magnitude
  caveat.
- **Defence**: reviewers may push back on "you chose the protocol your
  arch wins on." Need to clearly motivate AxBench's unit-norm property
  as architecture-agnostic.
- **Cost**: requires extra prose about why AxBench is preferred; may
  read as protocol-shopping.

#### Framing C: "Family-normalised paper-clamp, the corrected protocol" (highest-defence-if-Q1.3-works)

- Show that T-SAE paper's strength schedule is implicitly per-token-only.
- Define `s_norm = s_abs / <|z|>_arch` and show that under this
  normalisation, all archs peak at roughly the same s_norm and the
  cross-family success gap shrinks dramatically (from 0.83 → ~0.3 — TBD
  Q1.3).
- Make the paper's own protocol fair across families; report results
  under it as the canonical comparison.
- **Defence**: stronger than (A) and (B); hard for reviewers to argue
  with empirical normalisation. Methodological contribution in itself.
- **Cost**: Q1.3 must succeed; if TXC still trails by ≥0.3 points, this
  framing is undermined. Risk-managed by also shipping Framing A as
  fallback.

### Recommendation (Y)

**Framing C, conditioned on Q1.3 outcome:**

- If Q1.3 closes the gap to ≤0.2 points: lead with Framing C as the
  steering-section headline. Methodologically novel; addresses the
  paper's own protocol while removing per-token bias.
- If Q1.3 closes to 0.2 < Δ ≤ 0.5: hybrid Framing C + A. Lead with
  Framing C, report residual gap honestly as "feature-quality
  difference modulo magnitude scale." Still publishable.
- If Q1.3 fails (Δ > 0.5): retreat to Framing A. Both protocols, both
  winners, methods-section caveat. Reframe Q1 evidence as "we identify
  per-token bias in paper-clamp; future work to develop unbiased
  protocols."

In all cases, Q1.1 + Q1.2 evidence stays in the methods section as
the diagnostic that motivated Q1.3.

### Implications for the rest of the paper

- **Probing leaderboard** (Agent X): unchanged. Probing is unaffected
  by activation-magnitude scale (it's a feature-of-features comparison
  on raw AUC, not feature-of-residual).
- **Hill-climb winner** (Agent Z): if Q1.3 succeeds, hill-climb winners
  must still pass a steering check at family-normalised strengths.
  Pure probing-AUC wins are necessary but not sufficient.
- **Case studies** (Agent Y, beyond steering): HH-RLHF and any new TXC-
  win finding (Track B) sit alongside the steering result. The paper
  doesn't need a steering-side TXC win as long as Framing A/C lands
  cleanly.

### Open follow-ups (not blocking the paper)

- **MLC paper-clamp**: not yet implemented (multi-layer hook complexity).
  Q1.1 predicts MLC under-driven by ~16× under PAPER_STRENGTHS. Would
  strengthen Framing C but is not blocking.
- **Q2.C per-position clamp**: only triggered if Q1.3 fails badly. Y has
  the design ready; coded once Q1.3 lands.
- **Multi-seed steering**: only seed=42 graded. With Sonnet 4.6 grader
  being a new substitute for paper's Llama-3.3-70b, ≥1 additional seed
  worth running for variance estimate. Cost ~$2-3 in API.

### Cross-agent handoff

- **X (leaderboard)**: no probing-side dependency. Continue T-sweep + IT.
- **Z (hill-climb)**: read this + the magnitude writeup. Hill-climb
  winners benefit from being magnitude-aware (may reduce post-hoc
  steering issues).
- **Han**: review at your earliest convenience; pick Framing A vs C.
  Y's pipeline is reproducible; if direction changes mid-paper, the
  data already exists for Framing A as fallback.

### Files

- Magnitude evidence: `2026-04-29-y-tx-steering-magnitude.md`
- Plots: `results/case_studies/plots/phase7_steering_v2_*.png`
- Code: `experiments/phase7_unification/case_studies/steering/{diagnose_z_magnitudes,intervene_paper_clamp_normalised,analyse_normalised,plot_headline_comparison}.py`
- Data: `results/case_studies/{diagnostics,steering_paper_normalised}/`
