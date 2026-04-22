## Phase 6: Qualitative latent comparison

Scope marker for Phase 6 scripts + artefacts. The canonical brief lives at:

`docs/han/research_logs/phase6_qualitative_latents/brief.md`

### What goes here

- Concat-corpus builders (Newton + MMLU + Gita concats, MMLU TSNE
  windows, etc.)
- Encoded `z` tensor caches per arch per concat-set (under `z_cache/`)
- Autointerp scripts + per-feature label tables (under `autointerp/`)
- UMAP + silhouette analysis scripts (under `analysis/`)
- Per-experiment writeups + figures (`results/`)

### What doesn't go here

- Model classes — those belong in `src/architectures/` (e.g.
  `tsae_ours.py` for the Ye et al. T-SAE port).
- Ckpts — kept under
  `experiments/phase5_downstream_utility/results/ckpts/` since they
  share the Phase 5.7 training infrastructure.
