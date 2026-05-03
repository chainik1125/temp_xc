# TSAE paper-param audit (NeurIPS final push, 2026-05-02)

## What the meeting said

Dmitry, May 2 meeting, ~14:00 transcript:

> "I think they used K equals 20, and their contrastive loss is 0.1, just to make sure that you rerun the TSAE at paper parameters."

## What the source paper actually says

Paper: Bhalla et al., "Temporal Sparse Autoencoders: Leveraging the Sequential Nature of Language for Interpretability", ICLR 2026. `https://openreview.net/pdf?id=bojVI4l9Kn`.

Page 5 (Section 4.1, Hyperparameters), direct quotes:

> "All SAEs are trained with the **BatchTopK activation (k=20), 16k features**, and the auxiliary loss from (Bussmann et al., 2025; Gao et al., 2024). These models, layers, and hyperparameters are chosen to allow for comparability with pretrained and evaluated SAEs on Neuronpedia (Lin, 2023). Temporal and Matryoshka SAEs are trained with **20%-80% feature splits**, where for Temporal SAEs the 20% are the high-level features. We use **a regularization parameter of 1.0 on the temporal loss** for all Temporal SAEs."

So the paper-faithful spec is:

- BatchTopK, k=20, 16k features, layer 8 (Pythia-160m) / layer 12 (Gemma2-2b).
- 20% high-level / 80% low-level feature split.
- Temporal contrastive loss applied to the 20% high-level features only, between adjacent tokens t and t−1.
- Temporal-loss regularization coefficient = **1.0** (NOT 0.1; Dmitry mis-remembered).
- BatchTopK auxiliary loss (Bussmann 2025 / Gao 2024 style).

## What our codebase actually does

Both `tsae` and `tsae_paper` registry entries in `experiments/ward_backtracking_txc/architectures.py` use Han's `TemporalSAE` class from `temporal_crosscoders/han_tsae/saeTemporal.py`. That class implements an attention-based predicted+novel-code architecture — fundamentally different from Bhalla's high/low feature split + adjacent-token contrastive design:

- `tsae`: Han's TemporalSAE w/ `sae_diff_type='topk'`. No high/low split, no adjacent-token contrastive loss.
- `tsae_paper`: Same Han TemporalSAE class, just `sae_diff_type='relu'` + `l1_coef=1e-3`. Still NOT Bhalla's architecture; the name is misleading.

`_is_contrastive(...)` in `architectures.py` only triggers for `txc_h8` / `txc_h13`. Our TSAE variants have no contrastive loss term wired anywhere in the training pipeline.

## Decision for the NeurIPS push (Option A)

Use `tsae` (Han attention TSAE, TopK) with `kval_topk=20` as our "TSAE-paper" line in figures and document this deviation explicitly. We do not implement Bhalla's 20/80 split or adjacent-token contrastive loss for this submission.

Rationale: implementing Bhalla faithfully (Option B) is ~1–1.5 days of work (new SAE class with masked encoder, contrastive forward pass, training-loop hook). With a Sunday EOD experiment freeze, that eats the entire detection probe build (§3 of `experiments/ward_backtracking_txc/NEURIPS_PUSH.md`) — which is a much higher-leverage contribution. Given Dmitry mis-remembered the contrastive coefficient anyway, paper faithfulness is already not the top concern.

## Code changes shipped

1. `experiments/ward_backtracking_txc/config.yaml`: added `kval_topk: 20` to `txc.arch_kwargs.tsae` block.
2. `experiments/ward_backtracking_txc/architectures.py:106`: replaced hardcoded `kval_topk=k * T` with `kval_topk=int(kwargs.get("kval_topk", k * T))` so the config override actually flows through. Verified by direct call: `build_arch('tsae', ..., kval_topk=20).kval_topk == 20`.

## What we will say in the paper

The "Baselines" subsection of the case study will state, verbatim:

> "Our `TSAE-paper` baseline approximates Bhalla et al. (2026) with k=20 BatchTopK sparsity, but uses the attention-based predicted+novel-code architecture of Wang et al. (2025) rather than Bhalla's high/low feature split and adjacent-token contrastive loss. We adopt this approximation due to compute constraints. The model, layer, and feature dimensionality (DeepSeek-R1-Distill-Llama-8B, layer 10, 16k features) also differ from Bhalla's Pythia-160m / Gemma2-2b setup. A faithful re-implementation is left to future work."

## Open question for Dmitry

If Dmitry pushes back on Option A, Option B (faithful Bhalla TSAE) takes ~1–1.5 days and would need to start before EOD May 03 to finish before the freeze. Default to Option A unless explicitly approved otherwise on Slack.
