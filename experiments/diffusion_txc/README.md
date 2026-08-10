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

### Planned next

- LLM transfer: add the σ-ladder corruption (σ scaled to per-layer RMS)
  to the c7 `train_llama_txc.py` recipe; evaluate with the paper's
  backtracking detection/steering protocol.
- DSM per-token SAE arms (the minimal drop-in variant) for the 2×2
  objective × temporality ablation.
- See `LITERATURE.md` for prior-work positioning.
