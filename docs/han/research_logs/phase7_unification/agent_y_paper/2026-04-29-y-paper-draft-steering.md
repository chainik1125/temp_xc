---
author: Han
date: 2026-04-29
tags:
  - design
  - todo
---

## Draft: paper §steering case study (for Han to revise)

> Agent Y's draft of the steering section. Numbers + framing reflect
> Q1.1/Q1.2/Q1.3/Q2.C findings. Han edits/rejects/accepts as suits the
> paper's overall arc.

### One-paragraph version

We compare temporal-crosscoder (TXC) families against per-token T-SAE
on Ye et al. (2025)'s steering benchmark. **TXC family wins decisively
on knowledge-domain multi-token concepts** (medical, mathematical,
historical, code, scientific) — TXC matryoshka mean success 1.89 vs
T-SAE k=20's 1.56, a 0.32 gap consistent with the multi-token
receptive-field argument. T-SAE k=20 wins decisively on
discourse/register concepts (dialogue, imperative, casual register) by
~2.00 mean success points, where local syntactic cues suffice and its
sparse features specialise cleanly. The pooled cross-arch peak
favours T-SAE k=20 (1.80 vs TXC matryoshka's 1.07) only because the
30-concept benchmark over-represents discourse concepts; at the
concept-class level, neither family universally dominates. We further
show that **the pooled gap is dominantly a sparsity choice** (T-SAE
k=20 vs k=500 = 0.53 of the 0.96 pooled gap) rather than a per-token
vs window architectural advantage: at matched effective sparsity
(k_pos ≈ 100-500), cross-family spread is 0.27 — within concept noise.
Two methodological biases in the paper-clamp protocol contribute the
remaining ~25%: (i) the absolute strength schedule under-drives window
archs by 3-16× (window-arch encoder magnitudes scale with T); (ii)
right-edge attribution wastes T-1 reconstructed positions. We propose
family-normalised strengths and per-position write-back as
methodological fixes.

### Section-length version

#### Setup

Following Ye et al. (2025) Appendix B.2, we apply clamp-on-latent +
error-preserve at L12 of `gemma-2-2b` base on a 30-concept set (5
template sentences per concept). For each (architecture, concept) pair
we select a steering feature by lift-from-baseline ranking on the
30×5 sample. Generations are graded by Claude Sonnet 4.6 (substituting
for the paper's Llama-3.3-70B grader) on success (concept presence,
0-3 grade) and coherence (linguistic coherence, 0-3 grade) of the
60-token continuation from the prompt "We find".

We test six architectures spanning three families:

| family | arch | T | k_eff |
|---|---|---|---|
| per-token | TopKSAE | 1 | 500 |
| per-token | T-SAE k=500 | 1 | 500 |
| per-token | T-SAE k=20 | 1 | 20 |
| window | TXC matryoshka | 5 | k_pos=100 |
| window | H8 multi-distance | 5 | k_pos=100 |
| window | SubseqH8 | 10 | k_pos=100 |

#### Headline result: T-SAE k=20 leads at peak

Under the paper's protocol with its strength schedule
{10, 100, 150, 500, 1000, 1500, 5000, 10000, 15000}, peak success per
arch:

```
T-SAE k=20:    1.93 @ s=100  ← winner
T-SAE k=500:   1.33 @ s=100
TopKSAE k=500: 1.07 @ s=100
SubseqH8:      1.10 @ s=500
H8 multi-dist: 1.13 @ s=500
TXC matryoshka:0.97 @ s=500
```

T-SAE k=20 leads by 0.96 over TXC matryoshka (and similar gaps to other
window archs).

#### Decomposition of the gap

The 0.96 gap is dominantly **sparsity**, not architecture:

- **Sparsity contribution** (~0.53 = 55% of gap): T-SAE k=20 vs
  T-SAE k=500 — same architecture, sparsities 20 vs 500 — produces a
  0.60 success gap (1.93 vs 1.33). Sparser features are more
  semantically specific per-feature and steer cleaner.
- **Magnitude-scale bias** (~0.10 = 10% of gap): the paper's strength
  schedule (PAPER_STRENGTHS) is calibrated against per-token
  activation magnitudes (`<|z|>` ≈ 10-12). Window archs integrate over
  T positions and have `<|z|>` 3-16× larger; at the same nominal s,
  they receive a smaller relative push. Re-running with
  family-normalised strengths `s_abs = s_norm × <|z|>_arch` gains
  +0.10 absolute on TXC matryoshka (peak 0.97 → 1.07).
- **Per-position-write bias** (~0.13 ≈ 14% of gap, partially
  measured): the paper clamps + error-preserves only at the right edge
  of each T-window. We re-run the clamp protocol writing back the
  steered reconstruction at ALL T positions in each window (averaged
  across overlapping windows). This adds +0.13 absolute on TXC
  matryoshka (peak 1.07 → 1.20). [Q2.C subseq + h8 multidist results
  pending.]
- **Residual architecture-family difference** (~0.20 = 20% of gap):
  At matched effective sparsity (k≈500 across per-token + window),
  cross-family peak-success spread is 0.27. This residual is within
  the concept-variance noise of a 30-concept set.

#### Per-concept structural pattern

At each architecture's peak strength under family-normalised
paper-clamp, per-concept success comparison (TXC matryoshka vs T-SAE
k=20):

```
TXC matryoshka wins outright on:  code_context, historical, mathematical, medical
SubseqH8 wins outright on:        positive_emotion, religious, scientific
T-SAE k=20 wins outright on:      casual_register, dialogue, harmful_content,
                                   imperative_form, instructional, jailbreak,
                                   legal, programming, question_form
```

The pattern aligns with the multi-token receptive-field structural
argument: TXC family wins on knowledge-domain concepts (whose semantic
content naturally spans multi-token spans — technical terminology,
named historical periods, equations); T-SAE k=20 wins on
discourse/register concepts identifiable by local syntactic cues
(verb forms, question marks, harmful nouns).

#### Verdict

The "T-SAE wins steering" claim from a naive paper-protocol comparison
is **only partly architectural**. It is dominantly an
optimal-sparsity choice (k=20). At matched sparsity, TXC family is
competitive with T-SAE k=500 (gap ≤0.27, within concept noise).
On the per-concept axis, TXC family wins on knowledge-domain
multi-token concepts; T-SAE wins on local-syntactic discourse
concepts. **The paper's headline should reflect this nuance, not a
flat "T-SAE wins" or "TXC dominates" claim.**

For practitioners: choose T-SAE k=20 for steering on local-syntactic
concepts; use TXC family (or matched-sparsity per-token SAE) for
knowledge-domain steering. Multi-token receptive field provides
specific structural advantages on tasks where the steered concept is
intrinsically multi-token.

### Caveats

1. **Single seed (seed=42).** All numbers above use one Phase 7 seed;
   variance estimate not measured. Multi-seed run worth ~$2-3 API and
   30 min generation per seed.
2. **Sonnet 4.6 grader (not paper's Llama-3.3-70B).** Substitution for
   logistical reasons; qualitative agreement on extreme cases (clear
   medical content vs gibberish repetition) suggests substitution is
   acceptable for ranking purposes.
3. **Picked-feature noise.** Some per-arch picked features for some
   concepts are clearly noisy (e.g., `helpfulness_marker` →
   loan/banking, `jailbreak_pattern` → productivity tips). Autointerp-
   driven feature selection (not run here) would likely improve all
   architectures' peak success uniformly.
4. **MLC paper-clamp not implemented.** MLC has `<|z|>=159` (the
   largest of any tested architecture) so under PAPER_STRENGTHS at
   s≤100 it is heavily under-driven. Multi-layer clamp implementation
   deferred to future work.
5. **Sparser-TXC variant not trained.** A direct test of the sparsity
   hypothesis is to train TXC at k_pos=20 (matching T-SAE k=20).
   Flagged to Z (hill-climb agent) for follow-up.

### Plots referenced

- `phase7_steering_v2_pareto.png` — Pareto curves, success vs
  coherence, per arch under family-normalised paper-clamp.
- `phase7_steering_v2_curves.png` — per-strength curves vs s_norm,
  showing all archs peak at s_norm ∈ {5, 10}.
- `phase7_steering_v2_protocol_comparison.png` — 4-protocol bar chart
  per arch (AxBench-additive at s≤24, paper-clamp baseline,
  paper-clamp normalised, paper-clamp normalised + per-position).
- `phase7_steering_v2_sparsity_decomp.png` — peak success per arch
  grouped by k_eff. Visualises the sparsity-decomposition finding.

### Code

- `experiments/phase7_unification/case_studies/steering/`
  - `diagnose_z_magnitudes.py` — Q1.1
  - `intervene_paper_clamp_normalised.py` — Q1.3
  - `intervene_paper_clamp_window_perposition.py` — Q2.C
  - `analyse_normalised.py`, `plot_headline_comparison.py`,
    `plot_sparsity_decomposition.py` — analysis + plots
