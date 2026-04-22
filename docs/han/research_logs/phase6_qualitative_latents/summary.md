---
author: Han
date: 2026-04-22
tags:
  - results
  - complete
---

## Phase 6 summary: qualitative comparison of TXC, MLC, T-SAE latents

This phase compares three architectures on a qualitative-feature
benchmark that mirrors Ye et al. 2025's Figures 1, 2, and 4:

- `agentic_txc_02` — TXC winner (matryoshka + multi-scale InfoNCE)
- `agentic_mlc_08` — MLC winner (multi-layer crosscoder + multi-scale InfoNCE)
- `tsae_ours` — paper-faithful T-SAE port (newly trained in this phase)

`tfa_big` was dropped from the headline comparison after its
wall-clock was projected at ~9 hr on our A40 (see [[plan]] scope
adjustments). Adding it would not change the TXC/MLC-vs-T-SAE axis.

### Setup

- Model: `google/gemma-2-2b-it`, layer 13 residual stream (L11–L15 for MLC)
- All archs d_sae=18 432, k=100 per token, seed=42
- Hardware: A40 48 GB, Phase 5 activation cache (L13 slice, ~3.5 GB)
- Concat-sets built by [[build_concat_corpora]]:
  - **A** (Fig 1 analogue): Newton Principia + MMLU genetics Q + Bhagavat Gita — 752 tokens
  - **B** (Fig 4 analogue): MMLU bio Q + Darwin letter + Animal Farm wiki + MMLU math Q — 1067 tokens
  - **C** (TSNE set): 160 × 20-token MMLU windows across 8 subjects
- Per-arch encoding conventions in [[2026-04-22-encoding-protocol]]

### Training metrics (tsae_ours)

Plateau-stopped at **step 5400** (5.8 min wall), loss 45.76 → 11.72
(74 % drop), L0 = 98.6 / 100, plateau-slope 0.0156 < 0.02 threshold.
Ckpt at `experiments/phase5_downstream_utility/results/ckpts/tsae_ours__seed42.pt`,
mirrored to `han1823123123/txcdr` on HF.

### Deliverable (i): autointerp labels

Top-8 features per arch by activation variance on concat A + B,
labelled by Claude Haiku 4.5 from the top-10 activating contexts
(full labels table at [[results/autointerp/summary.md]]).

| arch | semantic labels (rough count) | representative top-1 label |
|---|---|---|
| `agentic_mlc_08` | **5 / 8 clearly semantic** | "Evolution and natural selection in populations" |
| `tsae_ours`      | 3–4 / 8 semantic              | "Punctuation and narrative transition markers in classical texts" |
| `agentic_txc_02` | 2 / 8 semantic                | "Family relationships in epic poetry" |

**Read:** `agentic_mlc_08`'s top features cleanly recover passage
topics (evolution, Orwell, Soviet/Stalinist history, Latin text,
historical dates). `tsae_ours` recovers some semantic features
(selection pressure, Stalin, family relationships) but a majority of
its top-8 are punctuation / digit / narrative-marker features.
`agentic_txc_02` is dominated by punctuation features at top-by-variance.

This **partially disconfirms H3**: we expected the paper-literal T-SAE
port to give the cleanest semantic labels, but MLC beat it. Hypothesis
revision: MLC's multi-layer stack carries more passage-level signal
than a single-layer SAE (even one with a temporal contrastive loss),
because the residual stream at nearby layers encodes redundant
semantic information that the crosscoder can denoise.

### Deliverable (ii): UMAP + silhouette

2-D UMAP with `n_neighbors=15, min_dist=0.1, metric="cosine",
random_state=42` on concat-set C (3200 tokens) for each arch, split
into high-level prefix (scale-1 for TXC, d_sae//2 for MLC/TSAE) and
low-level remainder. Colored by MMLU subject / question ID / a
char-class POS heuristic (no spaCy on CPython 3.14 yet).

Silhouette score (cosine, 2000-point subsample) per (arch, prefix, label):

| arch | prefix | semantic | context | pos |
|---|---|---|---|---|
| `agentic_txc_02` | high | +0.008 | −0.092 | **+0.077** |
| `agentic_txc_02` | low  | +0.006 | −0.054 | +0.019 |
| `agentic_mlc_08` | high | −0.009 | −0.115 | +0.050 |
| `agentic_mlc_08` | low  | −0.018 | −0.123 | −0.019 |
| `tsae_ours`      | high | −0.013 | −0.150 | +0.044 |
| `tsae_ours`      | low  | −0.009 | −0.112 | +0.011 |

All silhouette scores are **near-zero** (magnitude < 0.15), meaning
no arch produces clean MMLU-subject clusters in 2-D UMAP on this
concat-C setup. The biggest non-zero score is `agentic_txc_02`'s
**high-prefix pos clustering** (+0.077), consistent with the autointerp
finding that TXC's top features lean syntactic.

**Verdict on H1 (semantic clustering on high-level prefix)**:
**disconfirmed at this scale**. None of the archs recover clean
MMLU-subject structure via cosine-UMAP on our 20-token windows.
Looking at the plots qualitatively (see `results/umap/`), every arch
produces a dense central blob with a fringe of outlier points — the
expected "one cluster per subject" pattern the paper shows on
Pythia-160m does not appear here on Gemma-2-2b-IT L13.

Confounds worth flagging:
- Our concat-set C uses 20-token windows from MMLU questions. MMLU
  questions share a "A./B./C./D. …" answer template that produces
  very similar early tokens across subjects.
- Gemma-2-2b-IT L13 may carry less subject-discriminative information
  than Pythia-160m's L8 (the paper's Gemma experiments use L12 of
  the base model, not IT L13).
- Pair-wise cosine UMAP on 9216-dim high-prefix vectors that are
  mostly zeros (TopK k=100) may not produce the same 2-D structure
  as the paper's TSNE on 16k batch-TopK vectors.

### Deliverable (iii): passage-smoothness (H2)

Top-8 features by variance on concat-set B plotted per token, with
passage boundaries annotated. See
`results/top_features/concat_B__<arch>__top8.png` for full-res plots.

Qualitative read:

- **`agentic_mlc_08`**: clearest passage-level structure. Feature
  #4587 "Orwell's acknowledgments" turns on in the Darwin → Animal
  Farm region; #70 / #152 "historical dates" spike in Animal Farm
  (which contains year references); #6121 "Stalinist era" elevates in
  Animal Farm. Top-2 ("natural selection") are high across all
  passages but noticeably elevated in the Darwin segment.
- **`tsae_ours`**: intermediate. Feature #4560 "Selection pressure"
  peaks in the Darwin section; #2285 "Stalinist ideology" in Animal
  Farm. Several other top features (punctuation, narrative markers)
  fire uniformly across the concat without passage structure.
- **`agentic_txc_02`**: weakest passage signal. Top features are
  predominantly punctuation / formatting, firing uniformly across
  the four passages. One feature (#1254, unlabelled in the A+B
  top-8) shows a visible Animal-Farm-block elevation.

**Verdict on H2**: **confirmed for `agentic_mlc_08`**, partially for
`tsae_ours`, disconfirmed for `agentic_txc_02`. This is consistent
with the autointerp labels: archs that produce cleaner semantic
labels also show cleaner passage-level activation smoothness.

### Overall read

The empirical ranking on *qualitative* metrics in this phase is:

$$ \text{MLC} > \text{T-SAE} > \text{TXC} $$

whereas Phase 5.7's *quantitative* sparse-probing benchmark ranked
TXC ≈ MLC > T-SAE (on a last-position probing task). The divergence
suggests that:

- TXC's multi-scale contrastive recipe optimises for the sparse-probe
  metric (selecting features whose presence/absence maps cleanly to
  task labels) more than for sequence-level smoothness.
- MLC's multi-layer crosscoder architecture carries more
  passage-level context, regardless of the contrastive recipe
  attached to it — the 5-layer residual stack is doing heavy lifting.
- A literal port of the paper's T-SAE (`tsae_ours`) trained on
  Gemma-2-2b-IT L13 gets *some* of the way to the paper's
  qualitative claim (some semantic labels, moderate passage
  structure) but is outperformed by our MLC winner.

For the paper narrative, the cleanest story is:
**"our MLC + multi-scale InfoNCE recipe achieves what T-SAE's
contrastive loss was designed to achieve (passage-level smoothness,
concept-level features), and it does so with a quantitative
probing advantage."**

### Caveats

- `tfa_big` not included — would need ~9 hr wall-clock on A40 at
  d_sae=18432, seq_len=128. Queued as future work.
- spaCy wheels don't cover CPython 3.14 at the time of this phase;
  POS labels degrade to a char-class heuristic (punct / num /
  propn-or-sent-start / word). The POS silhouette scores above are
  therefore not directly comparable to the paper's spaCy-based numbers.
- `tsae_ours` uses TopK(k=100) and h = d_sae/2 for consistency with
  the Phase 5 bench, not the paper's BatchTopK(k=20) and h=0.2·d_sae.
  If BatchTopK is the load-bearing piece of the paper's result,
  our port may understate T-SAE's true qualitative quality.
- Concat-set C is 8 MMLU subjects × 20 Qs × 20 tokens = 3200 points.
  This is fewer points than the paper's Figure 2 (though comparable
  coverage). Could be underpowered for sharp silhouette signal.
- Silhouette threshold choice: 2000-point subsample. Full-set
  silhouette may differ slightly.

### Artifacts committed

- Code: `src/architectures/tsae_ours.py`, four Phase 6 scripts under
  `experiments/phase6_qualitative_latents/`.
- Ckpt: `tsae_ours__seed42.pt` (163 MB fp16) on HF `han1823123123/txcdr`.
- Concat corpora: `experiments/phase6_qualitative_latents/concat_corpora/*.json`
  on GitHub + HF `han1823123123/txcdr-data`.
- Results: `experiments/phase6_qualitative_latents/results/`
  containing `autointerp/`, `top_features/`, `umap/` subtrees.
- Docs: this `summary.md`, [[brief]], [[plan]], [[2026-04-22-encoding-protocol]].

### Reproduction

Given the Phase 5 cold-reproduction working (see Phase 5
`2026-04-21-reproduction-brief.md`), Phase 6 is:

```bash
# 1. Train the new arch
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
  experiments/phase5_downstream_utility/train_primary_archs.py \
  --archs tsae_ours --seeds 42

# 2. Build concat corpora (idempotent)
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
  experiments/phase6_qualitative_latents/build_concat_corpora.py

# 3. Encode
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
  experiments/phase6_qualitative_latents/encode_archs.py \
  --archs agentic_txc_02 agentic_mlc_08 tsae_ours

# 4. UMAP + silhouette
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
  experiments/phase6_qualitative_latents/run_umap.py \
  --archs agentic_txc_02 agentic_mlc_08 tsae_ours

# 5. Autointerp (needs ANTHROPIC_API_KEY)
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
  experiments/phase6_qualitative_latents/run_autointerp.py \
  --archs agentic_txc_02 agentic_mlc_08 tsae_ours

# 6. Top-feature plots
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
  experiments/phase6_qualitative_latents/plot_top_features.py \
  --archs agentic_txc_02 agentic_mlc_08 tsae_ours
```

End-to-end budget: ~30 min wall-clock on A40 given a warm HF cache.
