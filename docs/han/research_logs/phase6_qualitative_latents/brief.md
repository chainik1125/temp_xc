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

### Storage sizing (for pod config)

Approximate disk footprint for Phase 6, based on Phase 5.7 numbers:

| item | size |
|---|---|
| `.venv` + git repo + code | ~15 GB |
| HF cache (Gemma-2-2b-IT model) | ~6 GB |
| Pre-trained ckpts from HF: `agentic_txc_02` 1.3 GB + `agentic_mlc_08` 0.8 GB + `tfa_big` ~1.3 GB | ~3.4 GB |
| `tsae_ours` ckpt trained locally | ~1.3 GB |
| Activation cache (required if training `tsae_ours` / `tfa_big` locally) | 17 GB |
| Phase 6 z-caches (4 archs × 3 concat-sets × fp32) | ~1 GB |
| Autointerp contexts + labels | ~2 GB |

**Tiered pod sizing**:

- **~30 GB**: inference-only, all ckpts pre-trained elsewhere. Unusual.
- **~50 GB**: realistic — train `tsae_ours` locally, includes activation
  cache. Most likely path.
- **80 GB**: comfortable buffer for logs + autointerp API caches +
  intermediate UMAP tensors. **Recommended default.**
- **150 GB**: if you also want `probe_cache` for the stretch-goal
  HH-RLHF alignment study (Appendix B.1 mirror), add 66 GB.
- **170 GB**: full Phase 5.7 + 6 cold reproduction with everything
  downloaded from HF.

### Hugging Face: access check, backup convention, Phase 6 uploads

The project uses two HF repos as the canonical backup for anything
gitignored (ckpts, probe-caches, large activation buffers). The
canonical doc is [`docs/huggingface-artifacts.md`](../../../huggingface-artifacts.md) —
read it first if you haven't set up HF before. Phase-6-specific notes
follow.

**Verify HF access (do this first before any HF work)**:

```bash
# 1. Confirm the token file exists. Write-scoped token lives here.
ls -la /workspace/hf_cache/token
cat /workspace/hf_cache/token   # single line, starts with 'hf_'

# 2. Probe read access on the model repo (no auth needed for public repos)
.venv/bin/python -c "
from huggingface_hub import HfApi
api = HfApi()
print('model repo commits:', [c.commit_id[:8] for c in api.list_repo_commits('han1823123123/txcdr')][:3])
print('dataset repo commits:', [c.commit_id[:8] for c in api.list_repo_commits('han1823123123/txcdr-data', repo_type='dataset')][:3])
"

# 3. Probe write access (uploads a 1-byte canary then deletes it)
HF_HOME=/workspace/hf_cache .venv/bin/python -c "
import os, tempfile
from huggingface_hub import HfApi
api = HfApi(token=open('/workspace/hf_cache/token').read().strip())
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write('phase6-write-probe'); p = f.name
try:
    r = api.upload_file(path_or_fileobj=p, path_in_repo='.phase6_write_probe.txt',
                        repo_id='han1823123123/txcdr', commit_message='phase6 write probe')
    print('WRITE OK:', r.commit_url)
    api.delete_file('.phase6_write_probe.txt', 'han1823123123/txcdr',
                    commit_message='remove phase6 write probe')
    print('CLEANUP OK')
except Exception as e:
    print('WRITE FAILED:', e)
finally:
    os.unlink(p)
"
```

If write access fails with 403, ask Han to regenerate a write-scoped
token and update `/workspace/hf_cache/token`. The token is explicitly
**separate** from the `HF_TOKEN` environment variable — use the
file path, not the env var.

**Backup convention** (same as Phase 5.7, from `docs/huggingface-artifacts.md`):

- **Two repos**, purpose-separated:
  - `han1823123123/txcdr` (model repo) → all `.pt` ckpts, mirroring
    `experiments/*/results/ckpts/<name>.pt` → `experiments/*/results/ckpts/<name>.pt`
    in the HF repo. Exact mirror paths so a reproducer using
    `huggingface-cli download --local-dir .` lands everything in place.
  - `han1823123123/txcdr-data` (dataset repo) → activation buffers
    and probing caches (gitignored large `.npz` files), same
    mirror-path convention.
- **Upload scripts** (always prefer these over ad-hoc `upload_folder`):
  - `scripts/hf_upload_ckpts.py` — idempotent; `upload_folder` skips
    unchanged files by hash.
  - `scripts/hf_upload_data.py` — same, for dataset repo.
- **Neither repo has a README** with research narrative — Han's
  explicit instruction is to keep them empty-ish (license + github
  link only). Don't push writeups there.

**What Phase 6 should upload at end**:

1. **New ckpts** to the model repo:
   - `tsae_ours__seed42.pt` (the Ye et al. recipe port — new to Phase 6).
   - `tfa_big__seed42.pt` + symlinks `tfa_big_full__seed42.pt` etc. (if
     Phase 6 trains them ahead of Phase 5.7 handover (i) finishing).

2. **New dataset-repo artefacts** (if you want reproducibility):
   - `experiments/phase6_qualitative_latents/concat_corpora/` — the
     concat-set A/B/C token IDs + provenance JSON.
   - `experiments/phase6_qualitative_latents/z_cache/` — encoded
     latents per arch per concat-set. These are expensive to
     recompute and help reviewers reproduce the UMAPs fast.

3. **Don't upload**:
   - Autointerp labels / writeups / figures (those go to git via the
     research log).
   - Per-feature dump tensors if they're easy to recompute from ckpt
     + concat corpora.

**Running the uploads at end of Phase 6**:

```bash
# Model repo — picks up the new tsae_ours ckpt + any tfa_big you trained
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/hf_upload_ckpts.py

# Dataset repo — add Phase 6 corpora + z_caches BEFORE running this.
# You may need to update the script to include the phase6 subdirs; check
# before running. Current script uploads data/cached_activations +
# probe_cache. Extend LOCAL_PATHS if Phase 6 adds a new subtree.
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/hf_upload_data.py
```

If you extend `hf_upload_data.py`, commit that change to git — don't
leave a fork-only helper.

**Downloading for a cold reproduction** (useful reference for your
writeup's "reproduction" section):

```bash
# Full model + data mirror in one shot
cd /workspace/temp_xc   # repo root — mirror paths land things correctly
HF_HOME=/workspace/hf_cache \
  huggingface-cli download han1823123123/txcdr --local-dir .
HF_HOME=/workspace/hf_cache \
  huggingface-cli download --repo-type dataset han1823123123/txcdr-data --local-dir .
```

This restores ckpts + probe_cache in one pass, matching local paths.

### Resume checklist (first 15 min)

1. `git log --oneline | head -5` — confirm at `8a72a62` or later.
2. Read this brief and `papers/temporal_sae.md` (~30 min).
3. Glance at `references/TemporalFeatureAnalysis/sae/saeTemporal.py`
   (~15 min) — confirm you understand the TSAE loss.
4. **Run the HF access check** at the top of the previous section.
   If write access fails, escalate before doing any other work.
5. `ls experiments/phase5_downstream_utility/results/ckpts/` — confirm
   `agentic_txc_02`, `agentic_mlc_08`, `tfa_big` (if trained) exist.
   If not, download from HF per the cold-reproduction command above.
6. Check the Phase 5.7 handover progress:
   `cat docs/han/research_logs/phase5_downstream_utility/2026-04-22-handover-batchtopk-tsweep.md | head -80`.
7. If `tfa_big` not done yet, decide: (a) wait or (b) train it here
   yourself (30-90 min).
8. Start by porting the TSAE recipe as the lowest-risk step.

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
