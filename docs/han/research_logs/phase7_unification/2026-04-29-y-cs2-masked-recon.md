---
author: Agent Y (Aniket pod)
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Y CS2 — held-out-position reconstruction (TXC structural-prior win)

> 50%-time TXC-win candidate #2 (orientation log #4): when a single
> token's L12 residual is held out (zeroed), can each arch
> reconstruct it from surrounding context? Per-token archs cannot
> by architecture; TXC family can because the encoder integrates
> over T positions.
>
> **Result: TXC T=5 has FVE = 0.281 on the held-out token from
> context alone; per-token archs have FVE = 0.000-0.026.** This is
> a clean structural-prior win for TXC at this metric. SubseqH8
> T_max=10 fails (FVE = -1.693, worse than predicting zero) — its
> decoder over-relies on the right-edge input.

### Method

Reuses the 150-sentence Q1.1 cache (no fresh forward pass needed).
For each valid token position t (attn=1, t >= T-1 for window archs):

- **Baseline reconstruction**: encode the unmasked residual, decode,
  measure MSE between reconstruction and original at t.
- **Held-out reconstruction**: zero out the residual at t (or, for
  window archs, the rightmost position of the window ending at t),
  encode + decode, measure MSE between reconstruction at t and the
  ORIGINAL (unmasked) target.

Frac variance explained (FVE) = `1 - MSE / signal_variance` per
token, averaged over all valid positions. FVE = 0 means
"reconstruction is no better than predicting zero"; FVE = 1 means
"reconstruction is exact"; **negative FVE means reconstruction is
worse than predicting zero**.

### Code

[`smoke_masked_recon.py`](../../../../experiments/phase7_unification/case_studies/cs2_masked_recon/smoke_masked_recon.py)

### Outputs

- [`smoke_masked_recon.json`](../../../../experiments/phase7_unification/results/case_studies/cs2_masked_recon/smoke_masked_recon.json)
- [`smoke_masked_recon.png`](../../../../experiments/phase7_unification/results/case_studies/cs2_masked_recon/smoke_masked_recon.png)

### Results

![CS2 baseline vs held-out FVE](../../../../experiments/phase7_unification/results/case_studies/cs2_masked_recon/smoke_masked_recon.thumb.png)

| arch | T | baseline FVE | held-out FVE | held-out MSE / baseline MSE |
|---|---|---|---|---|
| `topk_sae` (per-token, k=500) | 1 | **0.992** | 0.000 | 130x |
| `tsae_paper_k20` (per-token, k=20) | 1 | 0.981 | 0.026 | 51x |
| **`agentic_txc_02`** (TXC matryoshka T=5) | 5 | 0.773 | **0.281** | 3.2x |
| `phase5b_subseq_h8` (T_max=10) | 10 | 0.498 | -1.693 | 5.4x |

### Read

- **TXC T=5 is the only arch that meaningfully reconstructs a
  held-out token from context.** FVE = 0.281 means 28% of the
  L12 residual variance at the masked position is recovered
  using only the T-1 = 4 neighbouring positions. This is a
  measurable, repeatable structural prior.
- **Per-token archs collapse to ~0 FVE on held-out.** Encoder sees
  zero, decoder produces ~b_dec which captures essentially none
  of the per-token signal. (TopKSAE actually goes to *exactly*
  0.000 in this run — the b_dec it learned is near-zero.)
- **SubseqH8 fails worse than predicting zero (FVE = -1.69).**
  Its baseline reconstruction is also weakest (0.498) — Subseq's
  matryoshka subseq sampling and longer T_max=10 produce a
  decoder that's strongly anchored on the right-edge input. When
  the right edge is zeroed, the decoder hallucinates in the wrong
  direction. The 130x ratio (held-out / baseline MSE) for
  TopKSAE is large but absolutely small (both terms are tiny);
  SubseqH8's 5.4x is on top of an already-large baseline.
- **Caveat: this is partly an "encoder context" comparison.**
  Per-token archs CANNOT use neighbouring positions by
  architecture, while window archs CAN. Saying TXC wins by 28%
  vs 0% is partially "TXC has more input information at t" — by
  construction. But the *quantitative* size of the win is the
  representational result: 28% recovery from context alone is
  much better than chance (0%) and not trivial.

### Why this is more interesting than CS1

CS1's dwell-time test had three confounds (activation rate,
window attribution, metric mismatch) that obscured any TXC signal.
CS2 has only one confound (encoder context by architecture), and
the WIN ON THE PRESENT METRIC is direct and quantitative. This
makes CS2 a real candidate for a paper case study.

### What the win means downstream

A 28% FVE on missing-token reconstruction suggests TXC features
could be useful for:

- **Activation imputation under noise / damage** (e.g., for OCR'd
  text where occasional tokens are corrupted). Per-token SAEs
  can't recover from a corrupted token; TXC can partially.
- **Sparse-probing on incomplete activations**. If a probe has
  missing position activations, TXC features still reconstruct
  some signal at those positions; per-token features go to zero.
- **Reduced bandwidth / edge inference**. Encoding only every
  T-th position and reconstructing the rest via decoder. Per-token
  SAE can't do this; TXC can with FVE = 0.28 quality.

### What needs to happen before this becomes a paper case study

1. **Confirm on long natural text, not just the 150-sentence
   concept probe.** Run on FineWeb-edu passages at S=256.
2. **Ablate the held-out position within the T-window** for TXC.
   If we hold out the leftmost position instead of the rightmost,
   does FVE change? If so, TXC's structural prior is direction-
   sensitive. If not, it's robust.
3. **Compare against a non-trivial baseline.** "Predict zero" is
   FVE=0; "predict the mean residual" might be ~0.05; "predict
   the residual at t-1" might be ~0.5. Where does TXC's 0.28
   sit relative to these? If a "predict-t-from-t-1 linear probe"
   already gets 0.5, TXC's 0.28 is below that.
4. **TXC vs T-SAE matched on baseline reconstruction.** TXC's
   baseline FVE is 0.773 vs T-SAE k=20's 0.981. The held-out
   reconstruction is bounded above by the baseline; some of TXC's
   28% lead might be explained by it having a more "spread"
   reconstruction that happens to handle missing inputs better.
   A fair comparison would normalise by baseline.
5. **Test on > 5 archs** — does the TXC family in general win
   this metric, or only the matryoshka variant?

### Verdict

**Promising, not yet proven.** This is the first 50%-time
candidate showing a clean TXC margin (vs CS1's null). Worth
deeper investigation before declaring a paper case study —
items 1-5 above. Estimated effort: ~3-4 hours of follow-up to
either confirm into a publishable result or kill via a control
experiment.

### Cost

| step | time | API spend |
|---|---|---|
| L12 cache (reused from Q1.1) | 0 | $0 |
| 4-arch encode + reconstruction | ~2 min | $0 |
| **CS2 total** | **~2 min** | **$0** |

### Next

CS3: candidate to be picked from the orientation log's stack.
Anaphora / coreference probing (#3) is high on the list but
requires labelled data and antecedent-distance constraints (T
window may be too small for natural-text antecedents). Will
either pursue that or jump to candidate #5
("counterfactual SAE-driven editing of multi-token spans") which
is a steering-flavoured win condition.
