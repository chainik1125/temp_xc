---
author: Han
date: 2026-04-22
tags:
  - results
  - in-progress
---

## Phase 6 summary (draft)

**Status (2026-04-22)**: brief + plan + encoding protocol landed on
`han-phase6`. `tsae_ours` training in progress; `tfa_big` deprioritised
after its 9 hr projected runtime proved incompatible with the single-
GPU Phase 6 timeline (see [[plan]] scope adjustments). Core three-arch
comparison targeted: `agentic_txc_02` × `agentic_mlc_08` × `tsae_ours`.

This writeup will be filled in as results land. Expected sections:

### Setup

- Gemma-2-2b-IT, layer 13, A40, d_sae=18432, seed=42
- Concat-set A (752 tok), B (1067 tok), C (160 × 20 tok across 8 MMLU
  subjects) — see [[build_concat_corpora.py]]
- Per-arch encoding conventions — see
  [[2026-04-22-encoding-protocol]]
- Seed + settings: `seed=42` for training and UMAP; TopK k=100 per
  token for all archs.

### Results to land

- UMAP subject/context silhouette table across the 3 core archs,
  high-level prefix vs low-level prefix
- Top-8 feature activation curves on concat-set A + B
- Autointerp labels (Claude Haiku)

### Hypothesis verdicts (TBD)

- H1 (semantic clustering on high-level prefix) — pending
- H2 (passage smoothness on top features) — pending
- H3 (concept-level autointerp labels) — pending

### Caveats

- `tfa_big` dropped from this phase's core deliverable set due to
  wall-clock budget (9 hr projected on A40 for full-size training).
  Adding it would require either a longer session or a lower-fidelity
  compromise (shorter seq_len) that defeats the "proper-size TFA"
  framing motivating Phase 5.7 handover experiment (i).
- spaCy not available on the pod's CPython 3.14 venv (no wheel for
  3.14 yet); POS labels degrade to a char-class heuristic. The paper's
  "low-level split → syntactic clusters" point is therefore assessed
  qualitatively, not against the same POS taxonomy the paper used.
- The paper's T-SAE uses BatchTopK(k=20) and h = 0.2 · d_sae; our
  `tsae_ours` uses plain TopK(k=100) and h = d_sae/2 for consistency
  with the Phase 5 bench. Noted at [[tsae_ours.py]] top.
