---
author: Han
date: 2026-04-25
tags:
  - design
  - in-progress
---

## Phase 7 plan — pre-registered protocol for the unified leaderboard

See `brief.md` for the *why*. This doc pre-registers the *what* and *how*.

### Subject model

`google/gemma-2-2b` (base, NOT instruct).

- Layer 13 anchor (matches Phase 5's L=12 0-indexed → L13 1-indexed
  in HuggingFace; verify which convention the paper compares to).
- L11–L15 for MLC.

### Activation cache (rebuild)

- Source: 24 000 sequences from FineWeb (same as Phase 5; consistent
  with prior pipeline). Document the FineWeb-vs-Pile difference vs
  T-SAE/TFA in the limitations section.
- Tokenizer: gemma-2-2b base tokenizer.
- Context length: 128 tokens (full Gemma-2 max for our purposes).
- Storage: fp16, layer-major. ~3 GB per layer × 5 layers = ~15 GB.
  Path: `data/cached_activations/gemma-2-2b/fineweb/resid_L<n>.npy`.

### Probe cache (rebuild)

- Tasks: same 36-task set as Phase 5 (8 dataset families).
- Per-task storage:
  - `acts_anchor.npz`: train_acts, test_acts at L13, **shape
    (N, 128, d)** (full 128-token tail).
  - `acts_mlc.npz`: train_acts, test_acts at L11-L15, **shape
    (N, 128, L=5, d)**.
  - `meta.json`: dataset_key, task_name, n_train, n_test, etc.
- Splits: same as Phase 5 (n_train=3040, n_test=760).
- Storage: ~3 GB per task × 36 = ~110 GB. Plus mlc tail ~5× that =
  ~550 GB. **TOO BIG**. Compromise:
  - L13 anchor at full 128 tokens, fp16: 36 × ~3 GB = ~110 GB.
  - L11–L15 stack at 128 tokens, fp16: 36 × ~15 GB = ~540 GB. Skip;
    instead, recompute MLC tail only when probing MLC (slower but
    feasible).
  - OR: store L13 at S=128, but L11–L15 only at S=20 (matches Phase 5
    convention for MLC, which we don't expect to need >20 for the
    main probing claim). Re-evaluate after first cache build.

### Sparsity convention

`k_win = 100` across all archs and all T values.

Per-arch translation:

| arch | k_win | k_pos (per active position) | notes |
|---|---|---|---|
| topk_sae | 100 | 100 | per-token (one position) |
| tsae_paper (k=100) | 100 | 100 | per-token |
| tsae_paper_k20 | 20 | 20 | paper-faithful reference |
| mlc | 100 | 100 / 5 = 20 | per-layer effective |
| mlc_contrastive_alpha100_batchtopk | 100 | 20 | same as mlc |
| agentic_mlc_08 | 100 | 20 | same as mlc |
| txcdr_t5 | 100 | 20 | per-position-slab |
| agentic_txc_02 | 100 | 20 | per-position-slab |
| phase57_partB_h8_bare_multidistance (T=5) | 100 | 20 | per-position-slab |
| phase57_partB_h8_bare_multidistance_t6 (T=6) | 100 | ~17 | |
| phase5b_subseq_track2 T_max=10 t_sample=5 | 100 | ~20 (per active sampled position) | |
| phase5b_subseq_h8 T_max=10 t_sample=5 | 100 | ~20 | |
| tfa_big | 100 (novel head) | n/a | TFA-specific |

### Probing protocol — pre-registered

Single S-parameterized aggregation:

```
def probe_aggregate(model, anchor_acts, T, S=128):
    # anchor_acts: (N, S, d)
    if T == 1:
        # per-token SAE
        z_per = encode_per_token(model, anchor_acts)  # (N, S, d_sae)
        z = z_per[:, T-1:].mean(axis=1)               # drop first T-1, mean
    else:
        # window arch
        K = S - T + 1
        windows = slide_windows(anchor_acts, T)        # (N, K, T, d)
        z_per_win = encode(model, windows)             # (N, K, d_sae)
        z = z_per_win[:, T-1:].mean(axis=1)            # drop first T-1, mean
    return z
```

For window archs (T > 1), the first T − 1 windows include tokens
that bleed into earlier positions; we drop them so all averaging
units cover tokens entirely within the considered tail.

For per-token archs we also drop the first T − 1 tokens to keep
coverage strictly aligned. Choose T_drop = 5 (matching the smallest
window arch's T) for the per-token arch's drop; this keeps the
comparison fair with the smallest window.

Probe: SAEBench protocol. Top-`k_feat`-by-`|mean_pos − mean_neg|` +
L1 LR (C=1.0, max_iter=2000, `with_mean=False` standardization).
Headline `k_feat = 5`; ablation at `k_feat ∈ {1, 2, 20}`.

#### Reported S values

- **S = 128 (headline)**: long-tail; minimal boundary asymmetry.
- **S = 20 (continuity)**: matches Phase 5's tail length, lets us
  validate against existing numbers.

NOT reported: S = T (per-window comparison) — the user explicitly
opted for one definitive leaderboard, no per-window comparison
across T values.

### Hypotheses (pre-registered)

- **H1 (Gemma-IT vs base)**: per-task AUCs change by ~0.01-0.02 on
  most tasks, but ARCH RANKINGS preserve. If rankings flip, this is
  a finding worth reporting.
- **H2 (k_win=100 vs k_win=500)**: the absolute AUC drops for TXC
  family by ~0.03-0.05 (sparser regime is harder). MLC family
  unchanged (already at k=100). RANKING within TXC family should
  preserve (B4 > H8 > vanilla TXC > strided), but we genuinely
  don't know.
- **H3 (long-tail S=128 vs S=20)**: AUC increases for all archs
  (more context). Spread between archs may compress (everyone
  gets the same boost) or expand (better archs use the extra
  context more efficiently). Open question.
- **H4 (TXC vs SAE at fix k_win=100)**: TXC family wins by
  ≥ 0.005 mp. If not, the entire "TXC > SAE" claim weakens.

### Architectures × seeds matrix

12 archs × 3 seeds = 36 ckpts. At ~25 min each on the local 5090,
~15 hr serial. Acceptable.

### Figures committed

- `phase7_headline_bar_mp_S128.png` — paired bar chart at headline S.
- `phase7_S_sweep.png` — line plot, AUC vs S for top archs.
- `phase7_seed_variance.png` — error bars on top archs.
- `phase7_qualitative_pareto.png` — TXC-vs-T-SAE Pareto from Phase 6.
  Re-rendered with Phase 7 ckpts.

### Ablations / appendix tables

- **k_win=500 ablation**: same archs at k_win=500, S=128. Shows
  effect of sparsity convention. ~6 archs × 1 seed = ~3 hr extra.
- **k_feat ∈ {1, 2, 20}**: probe-budget sweep. Same ckpts, just
  different probe `k_feat` (cheap re-probing).
- **S sweep**: AUC vs S for the headline archs at fix k_win.
- **Negative results**: D1 strided, C-family token-level, F subset-
  encoder. Numbers from Phase 5B at IT regime; note "not retrained
  for Phase 7 due to clean negative result on the prior setup."

### Branch hygiene

- All Phase 7 work on `han-phase7-unification`, branched off
  `han` AFTER the 5b + phase6 merges.
- New ckpts in
  `experiments/phase7_unification/results/ckpts/` (gitignored, sync
  to HF at `han1823123123/txcdr/phase7_ckpts/`).
- Probing results jsonl at
  `experiments/phase7_unification/results/probing_results.jsonl`.
- One commit per arch family (+ ablation, + writeup).
- No force-push to `han`. PR or fast-forward only after manual
  verification.

### Files I MUST NOT modify

- Phase 5's `experiments/phase5_downstream_utility/results/ckpts/`
  (Phase 5 agent's untouched ckpt set).
- Phase 5's `probing_results.jsonl`, `training_index.jsonl`.
- Phase 6's `experiments/phase6_qualitative_latents/results/`.
- Phase 5B's `experiments/phase5b_t_scaling_explore/results/`.
- The 36-task probe cache for Gemma-IT (read-only reference).

Phase 7 writes ONLY to `experiments/phase7_unification/`.

### What this phase will NOT do

- Train new architectures beyond the canonical 12.
- Modify the SAEBench-style probing protocol (top-k-by-class-sep +
  L1 LR is unchanged).
- Run autointerp / Haiku qualitative scoring beyond Phase 6's
  protocol (rerun on new ckpts, but no protocol changes).
- Cross-token tasks (winogrande, wsc) get the FLIP convention as
  before — no change to FLIP.
- Train on layer ≠ 13 (anchor) or != L11-L15 (MLC).
- Use `last_position` as a separate metric in the headline. Reported
  only as a caveat-laden footnote in the paper.
