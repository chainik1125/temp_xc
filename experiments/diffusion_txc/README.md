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
   perturbation support-overlap 0.74 vs 0.60); (b) ~~manifold-projected
   steering~~ — **falsified 2026-08-13**: even a density-matched
   (distill-captured, on-domain) DSM projector destroys generation at
   α=0 (Sonnet 0.25, 0/20 above floor), and the recon-twin control is
   less bad on every metric — substituting a k=96 reconstruction at
   every position kills computation-carrying precision regardless of
   objective. The claim that reconstruction SAEs cannot offer projection
   is moot: neither objective can, in this form.
2. **Temporal binding** — reconstruction provably cannot force
   cross-position binding; denoising makes it loss-bearing. Being tested
   by the TXC trio.
3. **Bayes semantics** — the objective derives its own encoder (posterior
   codes, principled activations, σ-conditioning) instead of needing
   patches; death-free training under weak sparsity pressure (0–2% vs
   31–57% TopK; caveat: strong prior-rate penalties reintroduce ~24%
   gate-death, though non-absorbing in principle).
4. **Density/anomaly monitoring (unexploited)** — the dictionary doubles
   as an activation-space OOD detector via the learned score. Flip side,
   now measured (2026-08-11 recalibration probe, volume
   `ooc_recal/results.json`): DSM dictionaries collapse *direction-deep*
   off-distribution — per-latent rate recalibration on distill windows
   revives 98% of the recon pool but 214→215 of 16,384 for dsm
   (preactivations almost all negative OOD). Portability requires
   training-side coverage of the deployment activation distribution;
   post-hoc calibration cannot rescue it.
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
