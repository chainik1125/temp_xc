---
author: Claude (10h unsupervised sprint, literature agent)
date: 2026-06-10
tags:
  - reference
---

## Literature search results (sprint, condensed)

Headline: **no prior work measures frequency-response curves for
dictionary-learning / SAE / crosscoder architectures** (searched "frequency
response" + SAE/dictionary/crosscoder, "spectral crosscoder", "temporal
crosscoder"). Nearest neighbours to cite and differentiate:

### Key (top 5)

- **Lindsey et al. 2024, "Sparse Crosscoders"** (Transformer Circuits). The
  architecture our spectral crosscoder extends; original spans layers, not
  timesteps.
- **Bhalla et al. 2025, "Temporal Sparse Autoencoders"** (arXiv:2511.05541).
  Closest prior: contrastive smoothness loss separates slow/fast features —
  *imposes* slowness, never *measures* a frequency response. The "temporal SAE"
  name is taken; position against it.
- **Grosse et al. 2007, "Shift-Invariance Sparse Coding"** (UAI). Canonical
  precedent for our conv dictionary (localized temporal templates + sparse
  codes).
- **Nanda et al. 2023, "Progress measures for grokking"** (ICLR). Transformers
  solve mod-P arithmetic with Fourier "clock" features — predicts what atoms our
  cyclic task should induce.
- **Rife & Boorstyn 1974** (IEEE IT). ML single-tone frequency estimation =
  periodogram peak; our circle-task oracle. Harris 1978 for Rayleigh resolution
  Δf ≈ 1/W.

### Also relevant

- Power et al. 2022 (grokking origin); Chughtai et al. 2023 (group-theoretic
  generalization; which irreps a network learns is seed-dependent).
- **Engels et al. 2024, "Not all features are one-dimensionally linear"**:
  real LLMs represent days-of-week/months as *circles* used for modular
  arithmetic — the circle embedding is the realistic case for cyclic concepts.
- Minsky & Papert 1969: modular difference from concatenated one-hots is the
  M-ary XOR — canonical non-linear-separability citation (our P2).
- Elhage et al. 2022 (toy models of superposition — evaluation philosophy);
  SynthSAEBench (arXiv:2602.14687) — iid synthetic SAE benchmark, no temporal
  dimension (we are the temporal counterpart); Menon et al. 2024
  (arXiv:2410.11767) formal-language SAE testbed, no HMM/frequency analysis.
- S4/HiPPO (Gu et al.) — "sequence layer = learned filter bank" framing;
  Solozabal et al. 2025 (arXiv:2508.20441) spectral bias in SSM kernels.
- **Rahaman et al. 2019, "Spectral bias of neural networks"** — low-frequency-
  first training bias. CONFOUND: a poor high-f response could be optimization
  bias, not architectural capacity. Mitigation: matched training budgets +
  oracle normalization + capacity sweep; flag in limitations.
