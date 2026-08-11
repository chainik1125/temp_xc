---
author: Claude (with Dmitry)
date: 2026-08-10
tags:
  - design
  - in-progress
---

## diffusion-txc: denoising objectives for temporal dictionaries

Workstream: replace the reconstruction objective of temporal dictionaries
(TXC / SAE / crosscoders) with denoising score matching (DSM) over a
σ-ladder, motivated by the BIRD/ELS correspondence. Ultimate goal: improved
detection/steering on behavioural evals for language models; the synthetic
program de-risks the method where ground truth is available.

### Status (2026-08-10)

The synthetic program lives in `experiments/bird_clock/` (A1–A4, B1–B3)
with results in `docs/dmitry/proposals/2026-08-10_bird_clock_results.md`
and theory in `docs/dmitry/proposals/2026-08-10_bird_temporal_codes.md`.
Headlines:

- DSM ≥ reconstruction for the TXC across all four synthetic settings;
  the gain concentrates in sub-Rayleigh/slow features — the band where
  behavioural signals (backtracking, persona) live.
- Preregistered atom-recovery test (B3): no interpretability-for-fidelity
  trade; DSM's dominant atom-level effect is eliminating the junk/noise
  tail. One caveat: Gaussian corruption slightly degrades atom purity on
  quasi-discrete (binary-event) generative structure.
- The Bayes-form posterior head dominates both axes on discrete-template
  data but is a specialist; TXC+DSM is the general transfer candidate.

### Motivations for the denoising objective (ranked 2026-08-11, post-Matryoshka comparison)

1. **Robust steering**: (a) robust features as causal handles (measured:
   perturbation support-overlap 0.74 vs 0.60); (b) **manifold-projected
   steering** — a DSM dictionary is a score model of its activation site,
   so one denoiser application after the steering hook re-projects
   off-distribution activations toward the manifold, potentially
   extending the coherent α range (the EM steer-finely-over-a-large-range
   fragility). Denoise-after-steer variant added to steering wave 2;
   reconstruction SAEs cannot offer this (no defined off-manifold
   behaviour).
2. **Temporal binding** — reconstruction provably cannot force
   cross-position binding; denoising makes it loss-bearing. Being tested
   by the TXC trio.
3. **Bayes semantics** — the objective derives its own encoder (posterior
   codes, principled activations, σ-conditioning) instead of needing
   patches; death-free training under weak sparsity pressure (0–2% vs
   31–57% TopK; caveat: strong prior-rate penalties reintroduce ~24%
   gate-death, though non-absorbing in principle).
4. **Density/anomaly monitoring (unexploited)** — the dictionary doubles
   as an activation-space OOD detector via the learned score.
5. Absorption reduction (−29 to −42%) — real but dominated by Matryoshka's
   architectural fix (−90%, same model/layer/L0); the two compose and the
   {standard, matryoshka} × {recon, dsm} 2×2 is an open cell.

### Planned next

- LLM transfer: add the σ-ladder corruption (σ scaled to per-layer RMS)
  to the c7 `train_llama_txc.py` recipe; evaluate with the paper's
  backtracking detection/steering protocol.
- DSM per-token SAE arms (the minimal drop-in variant) for the 2×2
  objective × temporality ablation.
- See `LITERATURE.md` for prior-work positioning.
