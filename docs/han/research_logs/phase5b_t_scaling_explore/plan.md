---
author: Han
date: 2026-04-25
tags:
  - design
  - in-progress
---

## Phase 5B plan — pre-registered T-scaling exploration

### Hypotheses (pre-registered)

- **H_A — per-(position, scale) decoder**: a fully-decoupled
  `W_decs[s, t]` decoder gives no probing gain over H8. (Null
  hypothesis — testing whether the param-tying in H8 is load-bearing.)
- **H_B — subsequence sampling at training, full window at probe**:
  the encoder learns a position-redundant latent. Probing AUC matches
  vanilla TXCDR T=t_sample at last_position; matches T=T_max at
  mean_pool (since mean_pool already aggregates many slides).
- **H_C — token-level encoder + sparse subsum**: probing AUC drops
  ≥ 0.02 vs vanilla TXCDR T=5. Position embeddings (C2 vs C1) recover
  ≤ 0.01.
- **H_D — strided window with span 10**: T_eff=5 stride=2 beats
  vanilla T=5 by ≥ +0.005 on at least one of (lp, mp). Stride > 2
  starts to hurt.

### Success criteria for *paper-relevant* findings

1. **New TXC champion**: any candidate with mp AUC ≥ 0.8126 (H8 3-seed
   ref) at some T, with seed variance ≤ 0.01 over 3 seeds.
2. **T-scaling**: monotonicity ≥ 0.8 over T values tested
   (definition: weighted Spearman > 0.8 between T values and AUC).
3. **Pareto improvement over vanilla**: both lp AND mp simultaneously
   beat vanilla TXCDR at the same T value, by ≥ 1σ each.

A null result is paper-relevant if it isolates a load-bearing
mechanism cleanly.

### Experimental matrix (seed=42 sweep)

Training cost (estimated, RTX 5090, d_sae=18432, batch=1024,
plateau=2% per 1k steps, max_steps=25k):

| ID | T | extra | est wall-clock | memory |
|---|---|---|---|---|
| A1 | 5 | per-(pos,scale) n=2 | ~25 min | 8 GB |
| A2 | 5 | per-(pos,scale) n=3 | ~30 min | 12 GB |
| B1 | 10 | t_sample=5 contiguous | ~30 min | 8 GB |
| B2 | 10 | t_sample=5 non-contig | ~30 min | 8 GB |
| B3 | 20 | t_sample=5 non-contig | ~45 min | 14 GB |
| B3' | 30 | t_sample=5 non-contig (if B3 promising) | ~60 min | 18 GB |
| C1 | — | token-enc + 5-of-128 sum | ~25 min | 5 GB |
| C2 | — | C1 + sin pos emb | ~25 min | 5 GB |
| D1 | 5 | stride=2 span=10 | ~25 min | 4 GB |
| D2 | 5 | variable stride {1,2,3} | ~25 min | 4 GB |
| D3 | 5 | H8 + stride=2 | ~30 min | 8 GB |
| E1 | T_paired={3,5,8} | Track 2 on pair-sum | ~30 min × 3 | 5 GB |
| E2 | T_paired={3,5,8} | H8 on pair-sum | ~30 min × 3 | 8 GB |
| E3 | n/a | topk_sae(x0+x1) | ~15 min | 3 GB |

**Total seed=42 training**: ~9 hours wall-clock if serial, ~3 hr if
3 trained in parallel (only when memory permits, e.g., D1+D2+C1).

**Methodology — apply each trick to ≥ 2 baselines**: per user feedback,
each architectural innovation must be tested on at least 2 backbones
to confirm it's a real recipe and not a baseline-specific artefact.
The matrix above already does this: A1+A2 (Track2 + H8 base), B1-B4
(Track2 + H8), D1-D3 (Track2 + H8), E1-E3 (Track2 + H8 + topk).
Candidate C is excluded from this rule because it's an entirely
different encoder shape, not a recipe layered on a backbone.

**Probing cost**: ~5 min per arch per aggregation. 9 archs × 2 aggs ×
~5 min = 1.5 hours.

### Followup expansion gates

After seed=42 sweep, expand only candidates that pass:

- **Tier A (3-seed)**: any candidate beating H8 mp at T=5 by ≥ 0.005,
  OR Pareto-dominating vanilla at any T. Add seeds 1, 2.
- **Tier B (T-sweep)**: any candidate with seed=42 AUC ≥ 0.80 at lp
  OR mp. Train at additional T values (6, 8, 15) where memory permits.
- **Tier C (deep follow-up)**: a candidate showing monotonicity ≥ 0.6
  on a 3-T mini-sweep. Push to T ∈ {30, 50} if memory allows; consider
  alternate sparsity mechanisms (BatchTopK with dial-in JumpReLU
  threshold).

### Figures committed

- `phase5b_headline_bar_lp_mp.png` — paired bar chart, all candidates
  vs H8 / vanilla T=5 / vanilla T=10 baselines, both aggregations.
- `phase5b_t_sweep.png` — T-sweep line plot for B/D candidates that
  reach Tier B.
- `phase5b_subseq_curve.png` — for B candidates: train-time t_sample
  vs probe-time aggregation AUC (heatmap or line).
- `phase5b_param_efficiency.png` — AUC vs total parameter count for
  all candidates + benchmarks.

### Probing protocol — apples-to-apples constraint (critical)

Phase 5 leaderboard is the canonical reference. Phase 5B candidates
MUST be probed using:

- The same probe cache (`results/probe_cache/<task>/{acts_anchor,acts_mlc}.npz`)
- The same 36 binary tasks, same train/test split (`n_train=3040`, `n_test=760`)
- The same top-k feature selector + L1 LR fitter (Kantamneni Eq. 1)
- The same two aggregations: `last_position` and `mean_pool` (tail-20 slides)
- The same k_feat values (1, 2, 5, 20); headline at k=5
- The same d_sae=18432
- The same training hyperparams (Adam lr=3e-4, batch=1024,
  plateau=2%/1k, max_steps=25k)

For non-trivial input transformations (pair-sum, strided, subsequence,
token-encoder) we must explicitly pick a probe-time mapping that
preserves the meaning of `last_position` and `mean_pool`. Below.

#### Per-candidate probe-time encoding

| candidate | last_position encoding | mean_pool encoding |
|---|---|---|
| A (per-(pos,scale)) | identical to vanilla TXC: encode last T tokens → z | identical to vanilla TXC: K=20−T+1 slides, mean of z |
| B1 (subseq contig, T_max=10, t_sample=5) | take last 10, sample contiguous t_sample=5 ending at last token | K=20−10+1=11 slides; per slide, contiguous t_sample=5 from end of slide |
| B2/B3 (non-contig, T_max=10/20) | take last T_max tokens, sum ALL T_max positions (= full encoder, not sampled) | per slide of T_max, full-T_max sum of z |
| C1/C2 (token-enc + sparse sum) | take last 5 tokens, encode each, sum z's | K=20−5+1=16 slides, sum-of-5 per slide, mean across slides |
| D1 (T_eff=5 stride=2) | take last 10 raw tokens, sample even-stride to 5, encode | K=20−10+1=11 slides; per slide, even-stride to 5 |
| D3 (H8 + stride=2) | same as D1 | same as D1 |
| E1/E2 (Track2/H8 paired, T_paired) | take last 2·T_paired tokens, pair-sum → encode | K=20−2·T_paired+1 slides; per slide, pair-sum then encode |
| E3 (topk on x0+x1) | take last 2 tokens, sum, encode | tail-20 → 10 paired tokens; 10 z's; mean |

For B2/B3 the probe-time encoder uses the FULL T_max-window (no
subsampling). The training-time random sampling is a regularizer that
should produce a "subset-redundant" latent; at probe time we always use
all available positions. This is the cleanest apples-to-apples
mapping.

#### Probe pipeline plumbing

Need to add a custom encoder adapter for each new arch shape.
`run_probing.py`'s `encode_for_probing()` dispatches on arch type;
Phase 5B will add Phase 5B branches but write outputs to the same
`predictions/` and `probing_results.jsonl` schemas (with `run_id`
prefixed by `phase5b_`).

A separate analysis script will pull only `phase5b_*` rows from
`probing_results.jsonl` and produce Phase 5B-specific tables/plots,
to keep Phase 5 leaderboard reproduction trivial (read all rows; same
layout).

### Files I MUST NOT modify

To preserve Phase 5 reproducibility:

- `experiments/phase5_downstream_utility/results/ckpts/` (Phase 5 agent's ckpt set)
- `experiments/phase5_downstream_utility/results/training_index.jsonl`
- `experiments/phase5_downstream_utility/results/probing_results.jsonl`
- The 36-task probe cache (read-only)
- `src/architectures/__init__.py` REGISTRY (don't shadow Phase 5 archs;
  Phase 5B archs use new class names only)

Phase 5B writes to:
- `experiments/phase5b_t_scaling_explore/results/ckpts/`
- `experiments/phase5b_t_scaling_explore/results/training_logs/`
- `experiments/phase5b_t_scaling_explore/results/training_index.jsonl`
- `experiments/phase5b_t_scaling_explore/results/probing_results.jsonl`
- `experiments/phase5b_t_scaling_explore/results/predictions/`
- `experiments/phase5b_t_scaling_explore/results/plots/`

### Branch hygiene

- Commit each completed candidate (arch class + train fn + result row)
  as a single commit. Message format:
  `Phase 5B {ID}: {arch_name} {key result}`
- After each Tier A pass, regenerate `phase5b_headline_bar_lp_mp.png`
  and update `summary.md` "headline" section.
- Rebase from `origin/han` periodically — at minimum once per Tier A
  promotion — to stay aware of Phase 5 agent's queue.
- **Do not** push to `origin/han-phase5b` until at least one Tier A
  result lands (avoids broadcasting a half-finished phase).

### What this phase will NOT do

- Re-train H8 / H7 / H9 (Phase 5 agent owns those).
- Train on layer ≠ 13.
- Modify probing protocol (k, aggregation set, task list).
- Run autointerp / Haiku qualitative scoring (Phase 6 territory).
- Mix candidates inside one training run (each arch is independent).
