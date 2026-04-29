---
author: Dmitry
date: 2026-04-28
tags:
  - results
  - in-progress
---

## Three rankings of "the misalignment feature" pick disjoint feature sets

For v2 SAE arditi 100k base-trained, we now have three different ways of identifying "which SAE feature is most relevant for misalignment":

1. **Δz̄ ranking** (encoder-side, Wang et al. 2025): which feature fires most differently between base Qwen and bad-medical Qwen on EM prompts.
2. **Causal screen** (Wang stage 2): for each top-100 Δz̄ feature, steer it at α=±1, see how much alignment shifts. Rank by `align(α=−1) − align(α=+1)`.
3. **Single-feature classifier AUROC** (this writeup): generate responses, judge with Gemini, fresh-forward each response, encode through SAE; for each feature compute AUROC of "z[i] predicts misaligned label."

These three rankings are conceptually distinct:
- Δz̄ measures *firing-shift between models*
- Wang screen measures *causal effect of single-feature steering on the misaligned model*
- Classifier AUROC measures *how informative one feature's activation is for response-level alignment*

In principle they should correlate. They do not.

### Top-10 by each metric on v2 SAE arditi 100k

| rank | classifier AUROC | Wang screen score | Δz̄ |
|---:|---|---|---|
| 1 | feat 30659 (AUC 0.10) | feat 29650 (+26.98) | feat 7613 (+1.185) |
| 2 | feat 12756 (AUC 0.12) | feat 16095 (+23.67) | feat 8330 (+1.128) |
| 3 | feat 30286 (AUC 0.86) | feat 16663 (+20.33) | feat 5261 (+0.806) |
| 4 | feat 7467 (AUC 0.14) | feat 7303 (+18.67) | feat 25782 (+0.464) |
| 5 | feat 1764 (AUC 0.84) | feat 14988 (+17.24) | feat 17914 (+0.433) |
| 6 | feat 19556 (AUC 0.83) | feat 2136 (+16.33) | feat 10651 (+0.424) |
| 7 | feat 14640 (AUC 0.17) | feat 30316 (+16.00) | feat 22025 (+0.422) |
| 8 | feat 10634 (AUC 0.18) | feat 21012 (+14.29) | feat 31131 (+0.403) |
| 9 | feat 29574 (AUC 0.18) | feat 11467 (+14.17) | feat 17750 (+0.319) |
| 10 | feat 29691 (AUC 0.82) | feat 863 (+13.79) | feat 30286 (+0.305) |

(Classifier AUROC < 0.5 means high z[i] predicts *aligned*; > 0.5 means high z[i] predicts *misaligned*.)

### Pairwise overlap of top-10 sets

| metric pair | overlap |
|---|---|
| classifier ∩ Wang | **0/10** |
| classifier ∩ Δz̄ | 1/10 (feat 30286) |
| Wang ∩ Δz̄ | **0/10** |

The three rankings pick essentially **disjoint** feature sets. None of the Wang causal champions are predictive in the classifier sense. None of the Δz̄ champions are causally pruned by the Wang screen. The single feature that appears in two rankings (30286) is the SAE Wang causal champion bundle's #3 by AUROC — a coincidence, not a structural pattern.

### Specifically: the Wang #1 causal champion (feat 30316)

This is the feature we built our [SAE Wang bundle k=30 result](../wang/causal_screen_finding.md) around: peak align 57.4 at α=−10 with coh 35.8, our SAE 100k headline. Its classifier AUROC is **0.527** — chance level. **Knowing whether feature 30316 fires on a response gives you essentially zero information about whether the response is aligned or misaligned**, even though steering it dramatically shifts the *generation-time* alignment.

This decouples two concepts that get conflated in SAE interpretability discourse:
- **"Feature represents misalignment"** (as in: high activation = the model is in a misaligned state) — a representational claim
- **"Feature mediates misalignment"** (as in: steering it changes generation behavior) — a causal claim

Feat 30316 satisfies the second but not the first. Steering it controls whether the model produces misaligned output; reading its activation off the resulting text doesn't tell you whether that text is misaligned. The reverse holds for high-AUROC features like feat 30286: their activation is highly informative about alignment of response text, but they don't show up as causal champions when we steer them individually.

### Implication for SAE feature interpretation

This sharpens the [sleeper-agent / trigger-circuit framing](../robustness/sleeper_agent_connection.md) we developed earlier. If our PEFT-LoRA EM organism has a trigger circuit gated on the chat-template scaffold, we'd predict:

- The features that **mediate** the trigger (i.e., the LoRA's prompt-conditional contribution) are the ones whose steering directly amplifies / suppresses misalignment → **Wang causal champions**, like feat 30316
- The features that **encode the misaligned output content** (e.g., features that fire on tokens describing harmful actions) are the ones whose post-hoc activation predicts the response label → **classifier champions**, like feat 30286, 1764, 19556

These are different sets of features because they live at different points in the computation. Steering inputs to a downstream computation (the trigger circuit) changes what gets produced; reading SAE activations on the produced text reflects what was produced.

A clean test of this hypothesis: if we steer a classifier-champion (high-AUROC) feature like 30659, do we get the alignment shifts we get from steering Wang causal champions? Prediction: probably not, because it represents output content rather than mediating the LoRA's contribution. Easy to run as a follow-up.

### Files

- `single_feat_classifier_sae_v2_100k.json` — full per-feature AUROC + correlation
- `single_feat_classifier_tsae_30k.json` — same on T-SAE 30k (different feature IDs, similar pattern)
- `tsae_30k_encoder_top200.json` — T-SAE Δz̄ ranking
- (existing) `../wang/sae_stage2_screen.json`, `../wang/sae_stage4_final_frontier.json` — SAE Wang procedure
