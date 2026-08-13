---
author: Claude (with Dmitry)
date: 2026-08-13
tags:
  - results
  - complete
---

## DSM program post-mortem: verdict and surviving assets

**Verdict.** For the program's stated goal — improved behavioural
detection and steering handles in LLMs — swapping reconstruction for
denoising score matching (DSM) is not worth it, as instantiated
(isotropic-Gaussian corruption, signs-of-life budgets, 1 seed, Llama/
Gemma sites). After fair on-distribution tests, DSM adds no information
beyond reconstruction, and it carries a unique cost: provable fragility
to deployment-distribution mismatch. The arc ran ~72 hours from
synthetic promise to this verdict, with every major claim either
measured or falsified under pre-registered readings.

### Claim ledger

| claim | outcome | where |
| --- | --- | --- |
| DSM ≥ recon, synthetic (4 settings) | held; matched-distribution only | [[2026-08-10_bird_clock_results]] |
| Absorption −42%, robustness ↑ (per-token LLM) | held; dominated by Matryoshka's −90% (compose-cell never run) | topk_vs_topkdiff/README |
| Sparse probing advantage | failed in-distribution, wash at 100M | topk_vs_topkdiff/README |
| Detection advantage (temporal, on-domain) | dissolved: recon's mass-selected core 0.223 > dsm survivors 0.208 | [[2026-08-11_backtracking_detection_dsm]] |
| "18× per-latent informativeness" | artifact of random-subsample null; superseded | same, final section |
| Steering handle | null vs random control (all our dicts) | [[2026-08-11_backtracking_steering_dsm]] |
| Manifold-projected steering (motivation 1b) | falsified on-domain; recon control less bad | same + [[2026-08-12_bird_transfer_theory]] |
| OOD collapse direction-deep | confirmed 3 ways; derived from the framework | [[2026-08-12_bird_transfer_theory]] |
| Covering-law pool predictions | 3 pre-registrations hit (incl. P1 dead-centre) | same |
| Tweedie-projection intervention story | falsified at k=96 fidelity | same, P2 section |

### The one open ablation

Corruption model. Everything tested was isotropic-Gaussian corruption;
the synthetic B3 anomaly (purity degradation on quasi-discrete
structure) flagged corruption mismatch from the start, and the theory
note lists it as the lead suspect surviving K1. An interference-shaped
corruption arm would distinguish "DSM is wrong for LLM activations"
from "isotropic Gaussian is the wrong corruption". Cheap (one sol-scale
pair). Not scheduled; run it before ever reviving the objective.

### Surviving assets (independent of the verdict)

- **Steering methodology + result**: the norm-matched random-direction
  control and symmetric/antisymmetric decomposition nullified
  conventional DoM steering (+0.015 [−0.13, +0.15]) while the paper's
  trace-trained crosscoder retains **+0.42 [+0.31, +0.52]** — the
  strongest causal result of the arc, and it strengthens the paper.
- **Domain decomposition of detection**: temporal-alone ≈ +0.006;
  domain match carries the stage-B edge; behavioural handles are
  domain-born for both objectives.
- **Covering-law theory**: [[2026-08-12_bird_transfer_theory]] — the
  ratio law fit two dictionaries at 1%; three pre-registrations hit;
  encoder story validated, intervention story falsified. Publishable
  core with the clock-model numerics.
- **JumpReLU as MMSE limit**: [[2026-08-11_jumprelu_mmse_note]] +
  bayes_gate's death-free rate-KL training (alive 1.0 at three L0
  points) and its rate-matched autointerp edge (0.77 vs 0.64).
- **Mass-selection of recon cores**: a few hundred top-mass recon
  latents match the full dictionary on detection — a free compact
  auditable unit set, discovered via the controls.
- **Instrument discipline**: capacity-matched *and selected* nulls;
  live-pool as the convention-free cross-instrument metric; five Modal
  launch traps documented in module docstrings and project memory.

### Artifact map

Checkpoints: volume `diffusion-txc` (`logs*`, `bayes_gate/`, `txc_w6*`)
mirrored to HF `dmanningcoe/diffusion-topk-saes`. Results JSONs:
`backtracking_eval/*` (detection, controls, survivor battery),
`backtracking_eval/steering/*` (waves 1–2, projector cells),
`ooc_recal/*` (collapse probes, pullback, P1/P3 scorecard). Docs: the
wikilinked notes above plus [[2026-08-12_arc_review]].
