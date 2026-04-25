---
author: Han
date: 2026-04-22
tags:
  - proposal
  - in-progress
---

## Phase 6 plan: qualitative comparison of TXC / MLC / T-SAE (± TFA) latents

Pre-registered execution plan for the phase scoped in [[brief]]. Written
before encoding any latents.

### Hypotheses

1. **H1 (semantic clustering).** On concat-set C UMAP plots coloured
   by MMLU subject, the *high-level* prefix of `tsae_ours`,
   `agentic_txc_02`, and `agentic_mlc_08` will show visibly tighter
   subject clusters than their *low-level* prefix. That is, multi-scale
   InfoNCE generalises T-SAE's "semantic-at-top, syntactic-at-bottom"
   property.
2. **H2 (passage smoothness).** On concat-set A and B, the top-8
   features by variance in `tsae_ours` will show clear phase
   transitions at passage boundaries (mirroring the paper's Figure 1 /
   Figure 4). The same features in `agentic_txc_02` and
   `agentic_mlc_08` will either match or look noticeably noisier.
3. **H3 (semantic autointerp labels).** Claude Haiku labels for the
   top-8 features of each arch will produce concept-level phrases
   ("discussion of plant biology") for `tsae_ours` / `agentic_txc_02` /
   `agentic_mlc_08`, but syntactic-style labels ("capitalized first
   word") more often for `tfa_big`.

If H1 and H3 both hold on the multi-scale contrastive winners, that is
direct evidence that the Phase 5.7 recipe generalises the paper's
qualitative claim.

### Success criteria

- **Minimum success.** All chosen archs successfully encode A + B + C.
  UMAPs are produced. Silhouette scores are tabulated. Autointerp
  labels are written. A 1-page qualitative writeup is committed under
  `docs/han/research_logs/phase6_qualitative_latents/` and pushed to
  GitHub.
- **Strong success.** Silhouette scores confirm the high-level-prefix
  UMAPs cluster by MMLU subject better than the low-level prefix
  across all multi-scale archs. Feature labels read as concept-level
  phrases on `tsae_ours`, `agentic_txc_02`, `agentic_mlc_08` (say, 4+
  of 8 per arch). Top-feature activations on A/B show visible
  passage-level phase transitions on at least 3 of the 4 archs.
- **Stretch.** A phase-5.7-style table comparing *smoothness score*
  $S$ (per paper §4.3) across archs, showing our multi-scale winners
  match or beat `tsae_ours`.

### Figures to produce (pre-registered)

1. `umap_high_semantic.png` — 2×2 grid (arch rows) of UMAP coords on
   high-level prefix, coloured by MMLU subject. Silhouette score
   annotated on each panel.
2. `umap_high_context.png` — same layout, coloured by question ID.
3. `umap_low_semantic.png` — same, low-level prefix.
4. `top8_features_concat_B.png` — for each arch, activation curves of
   top-8 features across the 1067-token concat-set B sequence, with
   passage-boundary dashed vertical lines.
5. Table: `autointerp/summary.md` — per-arch top-8 feature labels.
6. Table: `silhouette_scores.csv` — per (arch, prefix, label) triple.

### Scope adjustments

Per user feedback (2026-04-22 mid-phase):

- `tfa_big` is **optional**. If its training exceeds ~1 hr wall-clock
  before plateau, the phase ships with just TXC / MLC / TSAE and
  `tfa_big` moves to future work. The hypotheses above are still
  discriminating with 3 archs (H1 and H3 compare multi-scale winners
  against the paper's own recipe).

### Protocol

1. Download `agentic_txc_02__seed42`, `agentic_mlc_08__seed42` from HF
   and the L13 activation cache (L13 only; no L11/L12/L14/L15 needed
   since we don't retrain MLC).
2. Add `tfa_big` and `tsae_ours` dispatcher branches in
   `train_primary_archs.py`; port a paper-faithful `TSAEOurs`
   (`src/architectures/tsae_ours.py`).
3. Train `tfa_big__seed42` if time allows; train `tsae_ours__seed42`.
4. Build concat corpora A, B, C under
   `experiments/phase6_qualitative_latents/concat_corpora/`.
5. Encode each arch on each concat-set via `encode_archs.py`; cache
   z tensors under `z_cache/`.
6. Run UMAP + silhouette via `run_umap.py`.
7. Run autointerp via `run_autointerp.py` with Claude Haiku.
8. Write `summary.md` with figure embeddings + hypothesis verdicts.
9. Upload all new ckpts to HF model repo; optionally upload concat
   corpora + z_cache to HF dataset repo.

### Resource budget

Disk: ≤ 30 GB new on top of the `uv sync` baseline (pod has 100 GB).
RAM: ≤ 40 GB (pod ceiling is 46 GB; Gemma + SAE + activation buffer
comfortably fits). GPU: A40 48 GB — tfa_big training uses ~25 GB.

### Reproducibility

Seed 42 for all training and UMAP. The build_concat_corpora.py
download step is idempotent and cached. All figures go to
`experiments/phase6_qualitative_latents/results/` with `.thumb.png`
companions per the CLAUDE.md plotting convention.
