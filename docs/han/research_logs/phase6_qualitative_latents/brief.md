---
author: Han
date: 2026-04-22
tags:
  - proposal
  - in-progress
---

## Phase 6: Qualitative comparison of TXC, MLC, TFA, TSAE latents

**Audience**: a parallel agent running this phase while the Phase 5.7
follow-on handover (full-size TFA → BatchTopK → T-sweep) runs on a
separate GPU / worker. Assume you are picking this up cold.

**Why this phase exists**: Phase 5.7 settled the *quantitative* claim
(multi-scale InfoNCE with γ=0.5 decay is a family-agnostic recipe
that wins on sparse probing across TXC and MLC families). It does
NOT settle the *qualitative* claim — "do the features the recipe
discovers actually look semantic, or just statistically separable?"
The Ye et al. 2025 T-SAE paper's strongest narrative asset is its
qualitative argument that Matryoshka features are noisy and
token-level, while T-SAE features are smooth and passage-level
(see their Figure 1 + Figure 4 + TSNE visualizations in Figure 2).

If our multi-scale recipe really is a generalization of their
temporal contrastive idea, our features should *inherit* the same
smooth passage-level behavior. If they don't, we'll discover that
honestly. Either outcome is paper-worthy.

**Phase 6 Scope**: qualitative latent comparison across four archs on
engineered-concatenation probe sequences, with two headline deliverables:

1. **Auto-interpretation** of the top features from each arch.
2. **UMAP / TSNE visualization** comparing the latent spaces.

### Paper reference

Ye et al. 2025, "Temporal Sparse Autoencoders", arxiv 2511.05541.
Paper summary at [`papers/temporal_sae.md`](../../../../papers/temporal_sae.md).
Key sections: 4.2 (probing + TSNE), 4.5 (alignment case study), Figures 1, 2, 4.

Vendored code at [`references/TemporalFeatureAnalysis/`](../../../../references/TemporalFeatureAnalysis/).
Read `sae/saeTemporal.py` and `train_temporal_saes.py` to port their recipe.

### The 4 archs to compare

All trained on **the same setup as Phase 5.7**: Gemma-2-2b-IT, layer 13
residual stream, FineWeb-Edu subset, batch_size=1024, d_sae=18432,
max_steps=25000 with plateau-stop. Same activation cache — see
[`2026-04-21-reproduction-brief.md`](../phase5_downstream_utility/2026-04-21-reproduction-brief.md)
for details.

| arch | ckpt path (local + HF) | notes |
|---|---|---|
| `agentic_txc_02` | `results/ckpts/agentic_txc_02__seed42.pt` | TXC winner (matryoshka + multi-scale InfoNCE n=3 γ=0.5 α=1.0) |
| `agentic_mlc_08` | `results/ckpts/agentic_mlc_08__seed42.pt` | MLC winner (multi-scale InfoNCE on d_sae/4, d_sae/2, d_sae prefixes γ=0.5 α=1.0) |
| `tfa_big` | `results/ckpts/tfa_big__seed42.pt` (Phase 5.7 handover item (i)) | Proper-sized TFA; n_attn_layers=1, d_sae=18432, seq_len=128, Ye et al.'s two-component (novel + pred) |
| `tsae_ours` | **NEW — Phase 6 to train** | Ye et al.'s exact T-SAE recipe ported to our setup (this brief) |

**Blocker**: `tfa_big` comes from Phase 5.7 handover experiment (i).
Either wait for that to finish or train `tfa_big` yourself at the
start of Phase 6. Instructions in
[`2026-04-22-handover-batchtopk-tsweep.md`](../phase5_downstream_utility/2026-04-22-handover-batchtopk-tsweep.md).

HF ckpts are on `han1823123123/txcdr` for the three Phase-5.7 archs.

### What "TSAE ported to our setup" means

The Ye et al. T-SAE architecture:
- Standard TopK-SAE encoder + decoder (width 16384 in their Gemma paper).
- Latent split into High-level (`h = d_sae // 2`) and Low-level prefix.
- Temporal contrastive loss: InfoNCE on L2-normalised `z_cur[:, :h]` vs
  `z_prev[:, :h]`, i.e. on the high-level prefix of adjacent tokens.
- Train on next-token adjacent pairs, same pair-generation as our
  `txcdr_contrastive_t5` but with T=1 (single token, not window).

Our MLC-contrastive already implements the per-token version of this
recipe on a multi-layer-stack encoder. The missing piece is the
**single-layer single-token version** that matches Ye et al. literally:

- Inputs: single-token residual-stream activations at layer 13
  (our existing anchor buffer, not multi-layer).
- Encoder: one `nn.Linear(d_in=2304, d_sae=18432)` + bias, then TopK
  (k=100 per token, matching Phase 5 convention).
- Decoder: one `nn.Linear(d_sae, d_in)`.
- Forward: returns (loss, x_hat, z) with `loss = recon_mse + α · InfoNCE(z_cur_H, z_prev_H)`
  where α=1.0 (matching cycle 02's winning weight) and H is the first
  `d_sae // 2` latents.

This is NOT the same as `temporal_contrastive` in our bench (which is
a slightly different training loop Aniket ported earlier). Verify
equivalence with Ye's `sae/saeTemporal.py` before training.

**File to create**: `src/architectures/tsae_ours.py` with class
`TSAEOurs` subclassing `nn.Module`. Mirror the API of
`MLCContrastive` but with single-layer input.

### The "engineered dataset" protocol

Not a pre-built dataset — a concatenation recipe over public corpora.
Match the paper's Figure 1 + Figure 4 exactly for direct comparability:

1. **Concat-set A** (Figure 1 analogue): Newton's *Principia* excerpt
   (Project Gutenberg #28233) + MMLU genetics question + Bhagavat
   Gita excerpt (Gutenberg #2388). Target ~1024 tokens total.
2. **Concat-set B** (Figure 4 analogue): MMLU biology question +
   Charles Darwin letter (Gutenberg #2087) + Animal Farm Wikipedia
   article + MMLU math question. Target ~1024 tokens.
3. **Concat-set C** (extra for TSNE): 20-token windows from 20 MMLU
   questions across 8 categories. Use HF `cais/mmlu`
   (`all` split, sample balanced across categories).

Each concat-set gets encoded through all 4 archs at layer 13. Store
the encoded `z` tensors + token-level labels (source-passage ID for
A and B, question category for C) in a reproducible cache.

### Deliverable (i): auto-interpretation of top features

**Goal**: for each of the 4 archs, pick the top-N features by
activation variance across concat-sets A+B, and assign a
human-readable label to each. Report a table comparing the labels.

**Method** (follow the paper's convention):

1. For each feature in each arch, compute its activation pattern
   over concat-sets A+B (shape `(n_tokens,)`).
2. Pick the top-N features by activation *variance* (the features
   that most differentiate passages). Paper uses N=8 top features for
   Fig 4.
3. For each top feature, collect the top-k *activating contexts*:
   pull ~10 windows of 20 tokens surrounding that feature's top
   activations (across concat-sets A+B or a larger FineWeb-Edu slice
   if sparsity is low).
4. Send those contexts + the feature activation values to an
   autointerp LLM (Claude Haiku or similar via Anthropic API) with a
   prompt like:

   ```
   You are a feature explanation assistant for sparse autoencoders.
   Below are 10 text excerpts where feature #F of a sparse autoencoder
   activated strongly. The activation magnitude is shown at the start
   of each excerpt. Identify the common concept or pattern that
   triggers this feature. Reply with a single phrase (5-10 words),
   e.g. "discussion of plant biology" or "capitalized sentence starts".
   ```

5. Collect labels into a table: arch × feature_idx → label.

**Tip**: use [`src/utils/autointerp.py`](../../../../src/utils/) if it exists; otherwise
create it. A reference implementation can be derived from
Neuronpedia's feature-labelling pipeline (cited in the paper) or
from Bills et al. 2023 (the OpenAI LM-explanations recipe).

**Deliverable format**: a table in the writeup showing top-8 features
for each arch, with (feature_idx, label, top-3 activating contexts,
mean activation on concat-set A vs B). This directly mirrors the
annotations in the paper's Figure 1 + Figure 4.

**Claude API note**: use `claude-haiku-4-5` for speed; cost is
~$0.01-0.05 per feature at 10 × 20-token contexts. For 4 archs × 8
features = 32 calls ≈ $1 total. See `docs/claude-api-usage.md` if it
exists, or `.env.example` for API key conventions.

### Deliverable (ii): UMAP comparison

**Goal**: visualize the latent spaces of the 4 archs on the same
data, labeled by source-passage (for concat-sets A+B) or MMLU
category (for concat-set C), to argue that our winners recover
similar passage-level structure to TSAE (or don't).

**Method** (match the paper's Figure 2):

1. Encode concat-set C (20-token MMLU windows × 20 questions × 8
   categories = 160 sequences) with each arch.
2. For each arch, extract the *high-level* prefix
   (scale-1 for matryoshka, first d_sae/2 for MLC-contrastive, first
   d_sae/2 for TSAE). For the non-matryoshka TXC arch pick the top-k
   features by variance (paper did this).
3. Run UMAP with n_components=2 (2D plot). Color by MMLU category
   (semantic label). Compute silhouette score as quantitative
   complement.
4. Repeat with color = question ID (contextual label — tokens from
   the same question should cluster).
5. Repeat with color = POS tag (syntactic label — use spaCy). If
   matryoshka / MLC-contrastive have a *low-level* prefix, run
   analogous plots on that prefix too (paper shows this in their
   second row of Figure 2).

**Plot layout**: 4 archs × {high-level, low-level} × {semantic, context,
syntax} = 4 × 2 × 3 = 24 subplots. Use a single figure grid with
rows=arch, cols=(semantic, context, syntax), and two figures total
(one for high-level, one for low-level). Save as
`results/umap_high.png` and `results/umap_low.png`.

**Match the paper's visual language** so reviewers can eyeball the
comparison against their Figure 2. Include a silhouette-score table
as quantitative anchor.

**Libraries**: `umap-learn` is already installed via pyproject.toml
(check `uv.lock`). Use scikit-learn's `silhouette_score` for the
metric.

### Estimated timing

| step | time |
|---|---|
| 1. Port Ye et al. T-SAE recipe into `src/architectures/tsae_ours.py` | 1-2 hr |
| 2. Train `tsae_ours__seed42` on our setup (25k steps, plateau-stop) | ~30 min |
| 3. Wait on or train `tfa_big` (Phase 5.7 handover item (i)) | 0-3 hr depending |
| 4. Build concat-set A/B/C corpus (download + tokenize + store) | 1-2 hr |
| 5. Encode all 4 archs on concat-set A/B/C, cache the `z` tensors | 1 hr |
| 6. Autointerp pipeline + labeling (Deliverable (i)) | 3-4 hr |
| 7. UMAP / silhouette analysis (Deliverable (ii)) | 2-3 hr |
| 8. Writeup + figures + commit | 2 hr |
| **Total** | **10-16 hr wall-clock** |

Easy one-overnight session if you have `tfa_big` already; one plus a
morning if you train `tfa_big` yourself.

### Directory layout

Create Phase 6 artifacts under:

- `experiments/phase6_qualitative_latents/` — scripts, concat corpora,
  caches, encoded z tensors. README to be added.
- `docs/han/research_logs/phase6_qualitative_latents/` — this brief,
  plus your experiment writeups, plus `summary.md` at phase end.

Per the [CLAUDE.md](../../../../CLAUDE.md) phase convention. Shortname
`phase6_qualitative_latents` is used in both paths.

### Resume checklist (first 15 min)

1. `git log --oneline | head -5` — confirm at `8a72a62` or later.
2. Read this brief and `papers/temporal_sae.md` (~30 min).
3. Glance at `references/TemporalFeatureAnalysis/sae/saeTemporal.py`
   (~15 min) — confirm you understand the TSAE loss.
4. `ls experiments/phase5_downstream_utility/results/ckpts/` — confirm
   `agentic_txc_02`, `agentic_mlc_08`, `tfa_big` (if trained) exist.
5. Check the Phase 5.7 handover progress:
   `cat docs/han/research_logs/phase5_downstream_utility/2026-04-22-handover-batchtopk-tsweep.md | head -80`.
6. If `tfa_big` not done yet, decide: (a) wait or (b) train it here
   yourself (30-90 min).
7. Start by porting the TSAE recipe as the lowest-risk step.

### What success looks like

- **Minimum success**: all 4 archs encode concat-set A+B+C; UMAPs are
  produced; autointerp labels are written; a 1-page qualitative
  comparison with 1-2 figures is committed.
- **Strong success**: the visual comparison clearly shows whether our
  multi-scale winners preserve TSAE's passage-level smoothness or
  don't; silhouette scores quantify the visual impression;
  autointerp labels read like semantic concepts ("discussion of
  plant biology") rather than syntactic patterns ("capitalized first
  word").
- **Stretch**: alignment case study on HH-RLHF (Section 4.5 / App B.1)
  on our winners — different phase-level lift but would mirror the
  paper's biggest qualitative claim.

### Known gotchas

1. **TFA probing novelty convention**: `tfa_big` has dual probing
   (`z_novel` vs `z_novel + z_pred`) per Phase 5.7 finding. For Phase 6
   qualitative analysis, use `z_novel` alone (it's the sparse
   TopK-selected per-token component, comparable to other archs'
   TopK latents). The `z_pred` component is dense / context-predicted
   and doesn't fit the "top-k feature activation" mental model.
2. **Matryoshka scale-1 vs full**: `agentic_txc_02` has nested
   matryoshka scales. The "high-level" analogue is the scale-1
   prefix (d_sae/T = 3686 features when T=5). This is where the
   contrastive signal acts. Use scale-1 for the "high-level" UMAP
   panel, scale-T-inclusive-remainder for "low-level".
3. **Autointerp cost runaway**: Claude API calls add up. Cap at 8
   features × 4 archs = 32 calls. If you try to label all d_sae
   features you'll spend $$.
4. **UMAP is stochastic**: set `random_state=42` for reproducibility.
5. **Don't pollute Phase 5 caches**: your encoded z-tensors for concat
   sets go in `experiments/phase6_qualitative_latents/z_cache/`, not
   `experiments/phase5_downstream_utility/results/`.
6. **Gemma-2-2b-IT vs base**: the Ye et al. paper used Gemma-2-2b
   base. Our setup is -IT. When you write up, note this: our
   comparison is "TSAE recipe ported to Gemma-2-2b-IT L13", not "the
   paper's pre-trained Gemma TSAE as-published". Helps reviewers
   understand which claim is ours vs theirs.

### End of Phase 6 brief.
