---
author: Aniket
date: 2026-04-16
tags:
  - proposal
  - in-progress
---

## Sprint Coding Dataset Plan — Pre-Registration

Pre-registers the Gemma+Stack-Python cell that extends the current 2-cell
shuffle-vs-unshuffle sprint ([[experiments/sprint_feature_geometry/summary|feature-geometry results]]) to a
2×3 matrix. Purpose: commit to predictions before results land so
post-hoc rationalization isn't possible.

Go/no-go on actually running the experiment depends on (a) the TopKSAE
control result and (b) sign-off on this document.

## Hypothesis space

Three live hypotheses the 2×3 design is trying to disambiguate:

- **H1 (data richness scaling).** TXCDR shuffle sensitivity is a monotonic function of the temporal structure present in the data, holding model and inference mode fixed. Prediction: FineWeb < Stack < GSM8K on every shuffle-sensitivity metric.
- **H2 (architectural null).** TXCDR feature geometry is fundamentally driven by the shared-z window; shuffle effects are near-zero everywhere and the apparent DeepSeek+GSM8K difference is noise / UMAP seed variance. The existing step 2 result probably rules H2 out already; Gemma+Stack would nail it down if it also shows a shuffle effect.
- **H3 (model/mode confound).** The DeepSeek+GSM8K result is driven by DeepSeek being a reasoning-tuned model in generate mode with `<think>` traces, not by reasoning data being temporally richer than web text. Under H3, Gemma+Stack ≈ Gemma+FineWeb regardless of how much structure code has.

## Why H3 is the target this experiment rules out

The current 2-cell comparison (FineWeb vs GSM8K) varies model, data, and
inference mode *simultaneously*. A reviewer will write: "the comparison
confounds data modality with model architecture and inference mode."
The Gemma+Stack cell answers this by varying only the data while holding
model (Gemma 2B) and mode (forward) fixed. If Stack shows a clearly
larger shuffle effect than FineWeb, we've demonstrated data-driven
temporal sensitivity within a single model — a much cleaner claim. If
Stack ≈ FineWeb, the story collapses to "DeepSeek+GSM8K differs from
Gemma+FineWeb for some reason" and we'd need to also run Gemma+GSM8K or
DeepSeek+FineWeb to disentangle.

## 2×3 experimental matrix

| | unshuffled | shuffled |
|---|---|---|
| Gemma+FineWeb | existing (step 1) | existing (step 1) |
| Gemma+Stack | TODO (step 3) | TODO (step 3) |
| DeepSeek+GSM8K | existing (step 2) | existing (step 2) |

## Metric definitions

Two of the three metrics are geometric over decoder directions; the
third requires forward-passing cached activations through the
architecture's `encode()`. Different pipeline inputs, different cost
profiles — splitting them matters for "what do I actually need to run
to get the number." All three are backfilled on the four existing
step 1 / step 2 checkpoints before any new cell is launched, so the
table is apples-to-apples from day one.

**From decoder geometry** (TXCDR decoder directions, T-averaged,
L2-normalized, projected via the existing PCA → UMAP pipeline):

- **Silhouette score** over KMeans cluster labels (k=20, matching the existing pipeline), using cosine distance in the pre-UMAP PCA-50 space. **Δsilhouette = silhouette(unshuffled) − silhouette(shuffled)**. Positive = shuffling degraded cluster separation.
- **Cluster size entropy** H = −Σ pᵢ log pᵢ where pᵢ is the fraction of features in cluster i. Larger entropy = more uniform cluster sizes = less dominant islands. **Δentropy = entropy(shuffled) − entropy(unshuffled)**. Positive = shuffling smeared out dominant clusters (predicted direction).

**From encoded activations** (forward-pass the cached activation tensor
through `encode()` on a held-out sample; produces per-feature firing
patterns of shape `(B, T, d_sae)`):

- **Mean auto-MI across lags.** For each feature, compute MI between its binarized activation at position `t` and at position `t+k` for `k ∈ {1, 2, 4, 8}` using `src/shared/temporal_metrics.py::temporal_mi`. Report `mean_mi_per_lag` (length-4 vector) and `frac_features_above_threshold` in the output JSON for transparency, but the **pre-registered headline scalar is `mean(mean_mi_per_lag)` — one number per checkpoint, averaged across lags {1,2,4,8} and features, unambiguous.** **ΔTemporalMI = TemporalMI(unshuffled) − TemporalMI(shuffled)**. Positive = shuffling destroyed temporal coherence in feature firing patterns (predicted). TopKSAE is included in the metric as a secondary sanity check — see caveat below.

## Encode contract

`encode(x: (B, T, d_in)) → (B, T, d_sae)` is uniform across architectures:

- **TopKSAE** applies its per-token encoder independently at each position.
- **Stacked SAE T=5** uses its per-position encoders directly.
- **TXCDRv2** returns per-position feature contributions with the shared-z TopK mask applied: for each active feature in the window, its activation at position `t` is the position-`t` pre-activation contribution to that feature (zero elsewhere).

Defined in `src/bench/architectures/base.py` as a method on `ArchSpec`;
implemented in `topk_sae.py`, `stacked_sae.py`, and `crosscoder.py`. The
uniform shape is load-bearing because `src/shared/temporal_metrics.py`
asserts `ndim == 3` on input.

We deliberately do *not* use TXCDRv2's native `(B, h)` shared-z output.
The native encode sums pre-activations across `T` before TopK
(`einsum("btd,tds->bs", ...)` + TopK), so the output is
permutation-invariant under within-window shuffling: `z_unshuf ==
z_shuf` bitwise, and auto-MI is constant under the treatment variable.
That makes the native output mathematically non-functional as a
shuffle-sensitivity metric — not merely "insensitive," literally
invariant. The per-position contribution formulation preserves
position-dependent signal while respecting the shared-z TopK feature
selection. Implementation note: after masking by the shared-z TopK
support, verify that masked-out features are *exactly* zero — einsum
floating-point noise can show up as spurious auto-MI otherwise.

Cross-architecture comparability caveat: for TopKSAE and StackedSAE,
`encode(x)[b, t, f]` is "feature f's activation magnitude at position
t." For TXCDRv2 under the per-position formulation, it's "position t's
contribution to shared feature f." These are related but not identical
quantities — a TXCDRv2 feature that fires because position 3 did all
the work has large `f[3]` and small `f[0..4]`, while a TopKSAE feature
firing at every position has large `f[t]` everywhere. Consequence:
**absolute auto-MI magnitudes across architectures are only
approximately comparable**. The pre-registered clean signal is the
within-architecture Δ (unshuf − shuf); cross-architecture absolute
comparisons are qualitative tie-breakers, not the headline. The same
caveat tightens the "TopKSAE as null baseline" framing — it's a
secondary sanity check, not the rigorous zero-point. See the matching
ugly case below.

## Pre-registered predictions

Under H1:

| metric | FineWeb | Stack | GSM8K |
|---|---|---|---|
| Δsilhouette (unshuf−shuf) | small | medium | large |
| Δcluster entropy | small | medium | large |
| ΔTemporalMI (unshuf−shuf) | small | medium | large |

Paper-draft claim, verbatim: *"We predict that if TXCDR's temporal
sensitivity is a property of the data's temporal structure rather than
the model or inference mode, the shuffle-induced geometric degradation
will be smallest on FineWeb, intermediate on code, and largest on
reasoning traces."*

## Autointerp character predictions

Qualitative counterpart to the quantitative metrics. Under H1:

- **FineWeb clusters:** dominated by single-token concepts — entities, articles, topic keywords, surface lexical patterns.
- **Stack clusters:** mix of single-token syntax (identifiers, operators, keywords) and function-spanning features (scope blocks, control-flow patterns, indentation-aware features, variable-definition-to-use chains).
- **GSM8K clusters:** heavily multi-position — backtracking phrases, arithmetic chains, case analysis, reasoning-trace-specific tokens (`Wait,`, `Hmm`, `So,`, `Actually`).

Under H3, Stack clusters look like FineWeb clusters in *character* even
though the tokens are different — mostly single-token semantic features,
no function-spanning emergence.

## Seed variance

All runs use a single seed (`seed=42`). UMAP, KMeans, and TopK-training
are seed-dependent, so the three metrics above have unquantified
seed-variance. **Commitment:** before treating any single-number Δ as
evidence, we re-run one condition (step2-unshuffled, TXCDR) at 3 seeds
and report the per-metric seed-stddev. Any observed Δ must exceed ~2×
that stddev to count as signal.

## Ugly-case planning

- **Stack > GSM8K on shuffle sensitivity.** Possible: Python's positional semantics are brutal (bracket matching, scope via indentation, variable binding). Story shifts from "reasoning is specially temporal" to "structured modalities generally produce temporal features" — arguably a stronger claim for NeurIPS, but the ICML workshop framing (reasoning-focused) would need to adjust.
- **Stack ≈ FineWeb despite code's obvious structure.** Likely means T=5 is too short to capture code's structural patterns (function bodies span 20–100 tokens, not 5). Before concluding anything, run a T-sweep on Stack. Han's T-sweep experiments become critical-path rather than nice-to-have.
- **Stack between FineWeb and GSM8K, but FineWeb-to-Stack gap tiny and Stack-to-GSM8K huge.** Weak H1 support; H3 partially alive. Would need autointerp character evidence (function-spanning features appearing in Stack but not FineWeb) to shore up the claim. Another reason to scale autointerp on all six cells, not just DeepSeek.
- **TopKSAE auto-MI substantially different from TXCDR auto-MI across all three datasets.** Means the null baseline is contaminated by feature-population differences between architectures rather than reflecting a clean "no temporal binding" floor. Fallback: treat the within-architecture ΔTemporalMI (TXCDR shuf vs unshuf) as the primary signal and demote TopKSAE to a sanity check. The pre-registered headline already uses within-TXCDR, so this is a framing risk — don't over-claim TopKSAE as "the null" in the paper draft.

## Related

- [[experiments/sprint_feature_geometry/summary|feature-geometry results]] — existing 2-cell results (step 1 + step 2)
- [[experiments/sprint_5k_autointerp/plan|5k autointerp plan]] — 5k autointerp scale-up, gated on TopKSAE control
- [[SPRINT_PIPELINE]] — overall sprint infrastructure
