---
author: Agent Y (Aniket pod)
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Y — TXC steering magnitude verification (Q1.1 first; Q1.2 + Q1.3 to follow)

> Tests Dmitry's magnitude-scale hypothesis: window-arch encoder
> activations are O(T x per-token magnitude), so the paper's clamp
> schedule (designed against per-token archs) is "smaller relative
> push" for window archs and explains the 0.5-0.8 success gap under
> paper-clamp.
>
> Pre-registered plan in
> [`2026-04-28-y-orientation.md`](2026-04-28-y-orientation.md).

### Q1.1 — z[j*]_orig distributions across the 6 shortlisted archs

**Method.** For each arch, take the lift-selected best feature `j*`
per concept from Agent C's existing
`results/case_studies/steering/<arch>/feature_selection.json` and
re-run the encoder over the same 30-concept x 5-example probe set
(150 sentences, max_len=64). Record `z[j*]` at every content token
(right-edge only for window archs, all content tokens for per-token
+ MLC archs). Pool per arch over all (concept, example, token)
triples. Compute median + IQR over **active** values (`z > 1e-6`)
since post-TopK / post-threshold inactive positions would otherwise
dominate the distribution and hide the magnitude story.

**Code.**
[`experiments/phase7_unification/case_studies/steering/q1_1_z_orig_distributions.py`](../../../experiments/phase7_unification/case_studies/steering/q1_1_z_orig_distributions.py).
One-shot capture of L12 + L10-14 acts cached to
`steering_magnitude/_l12_acts_cache.npz` so Q1.2 / Q1.3 reuse the
same 150-sentence probe set without re-forwarding Gemma-2-2b base.

**Outputs.**
- [`results/case_studies/steering_magnitude/q1_1_z_orig_distributions.json`](../../../experiments/phase7_unification/results/case_studies/steering_magnitude/q1_1_z_orig_distributions.json)
- [`results/case_studies/steering_magnitude/q1_1_z_orig_distributions.png`](../../../experiments/phase7_unification/results/case_studies/steering_magnitude/q1_1_z_orig_distributions.png)

**Results.**

| arch | T or n_lay | median z[j*] | IQR | n_active_tokens | ratio to T-SAE k=20 |
|---|---|---|---|---|---|
| `topk_sae` (per-token, k=500) | 1 | 10.21 | 4.25-16.50 | 2413 | 1.12 |
| `tsae_paper_k500` (per-token, k=500) | 1 | 11.30 | 5.39-17.28 | 2413 | 1.24 |
| **`tsae_paper_k20`** (per-token, k=20) | 1 | **9.11** | 0.00-17.50 | 2413 | **1.00 (ref)** |
| `mlc_contrastive_alpha100_batchtopk` | 5 layers | 86.89 | 12.79-265.0 | 2413 | **9.53** |
| `agentic_txc_02` (TXC matryoshka) | T=5 | 21.57 | 11.95-31.90 | 1813 | **2.37** |
| `phase5b_subseq_h8` (SubseqH8) | T_max=10 | 63.11 | 38.30-91.52 | 1063 | **6.93** |

![Q1.1 per-arch distribution](../../../experiments/phase7_unification/results/case_studies/steering_magnitude/q1_1_z_orig_distributions.thumb.png)

**Read.**

- **Per-token archs cluster tight** (medians 9-11; ratios 1.0-1.24).
  Topology of this cluster is independent of `k` (k=20 vs k=500). So
  the steering-strength operating point should be ~constant within
  the per-token family.
- **TXC T=5 is 2.37x T-SAE k=20** — below the linear-in-T prediction
  of ~5x. Either the matryoshka multiscale decomposition damps the
  per-feature magnitude, or T=5 averaging is sub-linear. The
  brief's hypothesis pass condition was "window-arch medians ~5x
  per-token medians with at most 2x scatter"; 2.37 is at the edge of
  that band, not in its centre.
- **SubseqH8 T_max=10 is 6.93x T-SAE k=20** — closer to a linear-in-T
  reading (T_max=10 -> 10x prediction; observed 6.93x is ~70% of
  prediction). Subseq sampling integrates over a longer window than
  TXC's T=5 and the magnitude tracks that.
- **MLC 5-layer is the biggest, 9.53x** — Dmitry's analysis only
  considered temporal aggregation; layer aggregation also amplifies,
  even more strongly per axis. This is novel and worth flagging:
  the magnitude story generalises to ANY aggregation, not just T.

**Open question.** Does the **peak steering strength** under
paper-clamp scale with these ratios? If yes (Q1.2), the magnitude
story is the FULL story for the protocol mismatch and family-
normalised paper-clamp (Q1.3) closes the gap. If the peak strengths
ratio is non-linear — e.g. TXC needs a 5x push despite only 2.4x
typical magnitude — then there's a second factor (likely the
error-preserve term interacting with how much of `(s - z[j]_orig)`
is "in-distribution" for the encoder).

### Note on n_active_tokens drop for window archs

| arch | T effective | n_active_tokens / 2413 |
|---|---|---|
| per-token | 1 | 100% |
| TXC T=5 | 5 | 75% |
| SubseqH8 T=10 | 10 | 44% |

Window archs lose tokens at sequence-start positions where no full
window exists (positions `t < T-1` skipped per
`encode_per_position`'s right-edge attribution). Plus,
post-threshold sparsity differs across archs — the active fraction
isn't uniform. This is why I aggregate over active tokens only;
including zero positions would show window archs as "less active",
which is an artefact, not a magnitude story.

### Q1.2 — peak-strength scaling test

**Method.** Dmitry's `per_arch_breakdown.md` on
`origin/dmitry-rlhf` already tabulates mean (success, coherence)
over 30 concepts at 9 strengths × 6 archs under paper-clamp,
single seed=42, Sonnet 4.6 grader. Re-running the grids on this
pod would cost ~$5 + 30 min compute and produce the same numbers
to the same significant figures, so I parsed his tables verbatim
into JSON and fit per-arch peaks via parabolic interpolation in
log10(s) on the top-3 success cells.

**Code.**
[`q1_2_strength_curves.py`](../../../experiments/phase7_unification/case_studies/steering/q1_2_strength_curves.py).

**Outputs.**
- [`q1_2_strength_curves.json`](../../../experiments/phase7_unification/results/case_studies/steering_magnitude/q1_2_strength_curves.json)
- [`q1_2_strength_curves.png`](../../../experiments/phase7_unification/results/case_studies/steering_magnitude/q1_2_strength_curves.png)

![Q1.2 paper-clamp success vs strength](../../../experiments/phase7_unification/results/case_studies/steering_magnitude/q1_2_strength_curves.thumb.png)

**Results.**

| arch | T | grid peak_s | log-fit peak_s | peak suc | peak_s ratio to ref | Q1.1 magnitude ratio |
|---|---|---|---|---|---|---|
| `topk_sae` | 1 | 100 | 60.7 | 1.07 | 1.00 (ref) | 1.12 |
| `tsae_paper_k500` | 1 | 100 | 70.2 | 1.33 | 1.00 (ref) | 1.24 |
| `tsae_paper_k20` | 1 | 100 | 107.2 | 1.93 | 1.00 (ref) | 1.00 |
| `agentic_txc_02` | 5 | 500 | 430.8 | 0.97 | 5.00 | **2.37** |
| `phase5b_subseq_h8` | 10 | 500 | 387.3 | 1.10 | 5.00 | **6.93** |
| `phase57_partB_h8_bare_multidistance_t5` | 5 | 500 | 387.3 | 1.13 | 5.00 | n/a |

**Read.** All three window archs show **the same peak-strength
ratio (~5x ref)** despite having very different Q1.1 magnitude
ratios (TXC 2.37x, SubseqH8 6.93x, H8 multidist not measured).
Per-token peaks all land at the same operating point
(parabolic fits 60-107 around the s=100 grid point). This is
**inconsistent with the magnitude story being the full story**:

- If peak shift were proportional to magnitude shift, TXC's peak
  should land near `100 x 2.37 = 237`, not 430.
- SubseqH8's peak should land near `100 x 6.93 = 693`, not 387.

Both window archs land at almost the same fitted peak (~390-430)
**regardless of T or measured magnitude**. This points at a
second factor — possibly:

1. **Strength-grid resolution.** Dmitry only sampled
   `{10, 100, 150, 500, 1000, 1500, ...}` — a 5x jump from 150 to
   500. The fitted parabola is anchored on the (150, 500, 1000)
   triple for window archs, so it can't resolve a true peak
   between, say, 200 and 400. **Q1.3 needs a finer grid in this
   range to distinguish.**
2. **Error-preserve interaction.** At `s = z[j]_orig`, the
   intervention is a no-op. Window archs have higher per-token
   `z[j]_orig`, so the no-op point is *farther* from zero. If the
   "useful steering" range is roughly `[z[j]_orig + delta,
   z[j]_orig + delta_max]` with arch-independent delta, the peak
   shift would be additive in z, not multiplicative — which would
   fit "peak ≈ z + ~100" better than "peak ≈ z × 5".
3. **Encoder out-of-distribution at high s.** Clamping z[j] to
   500 with all other features at their normal values yields a
   reconstruction whose decoder output is far outside the
   distribution the model was trained on. Coherence collapse
   (the second factor on the y-axis) might gate which strengths
   are "usable" before any concept-pull happens.

**Caveat.** Single seed, n=30 concepts. The ~5x universality
across window archs could be a single-seed artefact; with seed
variance on 30 concepts the operating window could easily move
±50%. Q1.3 fixes this by sampling additional strengths around the
predicted optimum and explicitly testing the magnitude-normalised
hypothesis.

### Q1.3 design (revised after Q1.2)

The original plan was: "re-run paper-clamp on window archs at
`s_norm = s_paper × magnitude_ratio_arch`". Q1.2's universal-5x
finding makes the original Q1.3 not the most informative move.
The revised design:

- **Sample a finer strength grid for the window archs**:
  `{50, 100, 150, 200, 300, 400, 500, 700, 1000}`. This brackets
  the magnitude-normalised predictions (TXC 237, SubseqH8 693)
  AND the universal-5x prediction (~500), so we can disambiguate.
- For TXC, SubseqH8, H8 multidist: 9 strengths × 30 concepts × 1
  generation each = 270 generations × 3 archs = 810 total. Plus 2
  grader prompts each (success + coherence) = 1620 grader calls.
  Sonnet 4.6 + ThreadPool=8 + prompt caching, ~3-5 min wall time
  + ~$3-5 in API spend.
- Also re-grade the per-token archs at the same finer grid for
  the cross-arch comparison (so the apples-to-apples baseline
  isn't anchored on Dmitry's coarser grid). 3 per-token archs ×
  9 strengths × 30 concepts = 810 generations + 1620 grader
  calls. Same cost order.
- Total Q1.3: ~3000-3500 grader calls, ~$10-15.

**Hypothesis discrimination.**

- **If the new finer grid still shows window archs peaking at
  ~500 regardless of magnitude ratio** -> the universal-5x is
  real, NOT magnitude-driven. Dmitry's "magnitude is the full
  story" claim is rejected.
- **If TXC's fine-grid peak lands at 237 ± 50 and SubseqH8's at
  693 ± 100** -> magnitude is the full story; the original
  Q1.3 normalisation works.
- **If both archs peak at the same fitted value (e.g. ~400)** ->
  consistent with universal-5x; reject magnitude story.

Cost so far: 0 grader calls. Q1.3 will be the first grader-heavy
step.
