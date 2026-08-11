---
author: Claude (with Dmitry)
date: 2026-08-10
tags:
  - results
---

## Denoising vs reconstruction for TopK SAEs on Gemma-2-2B: a signs-of-life study

**TL;DR.** Sparse autoencoders (SAEs) — the standard tool for decomposing
language-model activations into interpretable features — are almost
universally trained to *reconstruct* activations. We trained an otherwise
identical TopK SAE to instead *denoise* them (reconstruct the clean
activation from a noise-corrupted copy, over a ladder of noise levels — a
one-line change to the loss), on Gemma-2-2B layer-12 activations at small
scale (16,384 features, 10M training tokens, two seeds per objective).
The denoising-trained SAE has substantially more atomic features —
feature absorption drops from 0.31 to 0.18 (−42%) — and far more robust
ones — under input perturbations at half the activation scale, its active
feature set retains 76% overlap versus 63% for the standard SAE — at the
same sparsity and near-identical concept decodability. The costs are on
the reconstruction side, exactly where theory predicts: 11% worse
reconstruction error, downstream loss-recovered 0.84 vs 0.91, and 3.6×
more dead features. On sparse probing (decoding concepts from few
features), this first pass is **competitive** — wins and losses mixed
across datasets and k, with large seed spread — rather than the clear
advantage we preregistered; at 2% of a standard training budget, with the
denoising run paying a 25% dead-feature capacity tax it should not have
to pay, we read this as unresolved rather than negative. On the standard
SAE scoreboard, which leads with reconstruction metrics, this objective
would be rejected without its feature-quality gains ever being measured —
which may explain why, as far as we can find, no one trains
interpretability SAEs this way.

### Summary table

| metric (2 seeds/objective, arm means) | reconstruction | denoising | better |
| --- | --- | --- | --- |
| feature absorption rate (↓) | 0.306 | **0.179** | denoising, −42% |
| active-set overlap under ε=0.5·RMS perturbation (↑) | 0.627 | **0.762** | denoising, at every ε tested |
| LLM-judged feature explainability (balanced acc) | 0.652 | 0.672 | denoising, ≈1 SE |
| first-letter decodability (probe acc) | 0.928 | 0.918 | ≈ tie |
| sparse probing, k=1 / k=5 (3 datasets) | mixed | mixed | competitive; unresolved at this budget |
| clean-reconstruction NMSE (↓) | **0.299** | 0.331 | reconstruction |
| downstream loss recovered (↑) | **0.908** | 0.841 | reconstruction |
| dead features at end of training (↓) | **7.5%** | 25–28%, rising | reconstruction |

### What we did

Both SAEs are Gao-et-al-style TopK autoencoders (16,384 latents, k=40
active per token, AuxK anti-dead loss, tied-transpose initialization),
trained for 6,000 steps on the same 10M-token cache of Gemma-2-2B
layer-12 residual activations, differing *only* in the loss:
reconstruction minimizes ‖f(x) − x‖²; denoising minimizes
‖f(x + σε) − x‖² with σ drawn log-uniformly from 0.05–1.0× the layer's
RMS each batch. TopK makes the comparison sparsity-matched by
construction. Evaluations: SAEBench-style sparse probing (three
classification datasets), a first-letter feature-absorption measurement,
a perturbation-robustness test (re-encode the same activations with
Gaussian noise added; measure active-set overlap and probe flips), and
LLM-judged explanation-then-detection over each SAE's top-activating
contexts. All predictions were preregistered in `README.md` before any
evaluation ran; the probing prediction failed and is reported as failed.

### Why the pattern makes sense

Denoising's optimum is a Bayes posterior mean rather than the identity:
theory predicts it sacrifices clean reconstruction (a σ-blind denoiser
behaves like a Wiener filter, which under-reproduces clean inputs) and
noise-fitting capacity in exchange for features that survive corruption —
which is what absorption (features must stay separable when co-occurrence
statistics are disturbed) and the robustness test measure directly. The
same trade appeared first in our synthetic program (`experiments/
bird_clock/`, results in `docs/dmitry/proposals/2026-08-10_bird_clock_results.md`),
where ground truth made it measurable exactly: denoising-trained
dictionaries had worse reconstruction and cleaner, ground-truth-aligned
atoms in every setting.

### Limitations, honestly

- 2% of a standard training budget (10M tokens vs SAEBench's 500M), two
  seeds, one layer, one model, one sparsity level. Nothing here is
  converged; every number is a lower bound on both objectives.
- The sparse-probing result is unresolved, not negative: the denoising
  arm is competitive while paying a 25–28% dead-feature capacity tax at
  2% budget with 2 seeds. Three testable ways the advantage could emerge:
  fix the dead features (a direct capacity recovery); harder/sparser
  concepts than topic and sentiment; and temporally-bound structure,
  which per-token evaluations cannot see and where the synthetic program
  located denoising's detection advantage.
- Judged explainability used 48 features per SAE and one judge model.
- Our absorption and probing implementations are SAEBench-style
  reimplementations at reduced scale, not the SAEBench harness itself.

### What's next

1. Fix the dead-feature tax (anneal σ over training, or drive the AuxK
   dead-feature loss from corrupted preactivations) and rerun at 5–10×
   budget on the SAEBench harness proper.
2. The actual goal: the *temporal* version (window dictionaries trained
   with denoising) evaluated on behavioural detection — backtracking — where
   both the synthetic theory and the earlier frequency-band evidence say
   the advantage should be largest.

Artifacts: four checkpoints, training curves, eval JSONs, and
top-activating-context dumps in the Modal volume `diffusion-txc`; code in
this directory (`sae.py`, `train_arms.py`, `run_evals.py`, `modal_*.py`).
