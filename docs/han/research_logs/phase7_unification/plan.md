---
author: Han
date: 2026-04-25
tags:
  - design
  - in-progress
---

## Phase 7 plan — pre-registered protocol for the unified leaderboard

See `brief.md` for the *why* and the agent-execution model. This doc
pre-registers the *what* and *how*, addressed to the two execution
agents (sparse-probing agent + autointerp agent).

### Subject model

`google/gemma-2-2b` (base, NOT instruct).

- Layer 13 anchor.
- L11–L15 for MLC.
- HF model id: `google/gemma-2-2b`.
- Tokenizer: `google/gemma-2-2b` (matches base model).

### Activation cache (rebuild) — owned by Agent A

- Source: 24 000 sequences from FineWeb (consistent with prior Phase
  5/5b pipeline; document the FineWeb-vs-Pile difference vs T-SAE/TFA
  in the limitations section).
- Context length: 128 tokens (full Gemma-2 max for our purposes).
- Storage: fp16, layer-major. ~3 GB per layer × 5 layers ≈ 15 GB.
  Path: `data/cached_activations/gemma-2-2b/fineweb/resid_L<n>.npy`.
- Sync to HF for cross-agent access:
  `han1823123123/txcdr/phase7_activation_cache/`.

### Probe cache (rebuild) — owned by Agent A

- Tasks: same 36-task set as Phase 5 (8 dataset families).
- Per-task storage:
  - `acts_anchor.npz`: train_acts, test_acts at L13, **shape
    (N, 128, d)** — full 128-token tail.
  - `acts_mlc.npz`: train_acts, test_acts at L11-L15. Decision: store
    L11–L15 only at S=20 (matches Phase 5 MLC convention; recompute
    longer tails on demand if needed). Reduces storage 5×.
  - `meta.json`: dataset_key, task_name, n_train, n_test, etc.
- Splits: same as Phase 5 (n_train=3040, n_test=760).
- Storage: ~3 GB per task × 36 = ~110 GB for L13 anchor + ~30 GB for
  L11-L15 stack at S=20 = **~140 GB**. Verify RunPod volume.
- Sync to HF for cross-agent access:
  `han1823123123/txcdr/phase7_probe_cache/`.

### Sparsity convention

`k_win = 500` across all archs and all T values.

Per-arch translation (TopK count applied to the `d_sae`-dim
pre-activation, by arch):

| arch | k_win | k_pos (per active position) | notes |
|---|---|---|---|
| topk_sae | 500 | 500 | per-token (one position) |
| tsae_paper (Phase 7) | 500 | 500 | per-token; matryoshka BatchTopK at k=500 |
| tsae_paper_k20 | 20 | 20 | paper-faithful native baseline |
| mlc | 500 | 100/layer | matches "k_pos=100 × L=5" |
| mlc_contrastive_alpha100_batchtopk | 500 | 100/layer | retrain at k=500 |
| agentic_mlc_08 | 500 | 100/layer | retrain at k=500 |
| txcdr_t5 | 500 | 100/slab | k_pos × T = 100 × 5 = 500 (Phase 5 default, no change in convention) |
| agentic_txc_02 | 500 | 100/slab | same |
| H8 (T=5) | 500 | 100/slab | same |
| H8 (T=6) | 500 | ~83/slab | k_win = 500 with T=6 → k_pos=83 (slight departure from Phase 5 H8 T=6's k=600; we use 500 for consistency) |
| H8 (T=7) | 500 | ~71/slab | similar |
| B2 subseq_track2 (T_max=10, t_sample=5) | 500 | 100/active-slab | matches Phase 5B B2 convention |
| B4 subseq_h8 (T_max=10, t_sample=5) | 500 | 100/active-slab | matches Phase 5B B4 convention |
| tfa_big | 500 | n/a | TFA novel-head L0 = 500; verify training stack supports it |

### Probing protocol — pre-registered (owned by Agent A)

Single S-parameterized aggregation:

```
def probe_aggregate(model, anchor_acts, T, S=128):
    """anchor_acts: (N, S, d_in) — the S-token tail per example.
    Returns (N, d_sae) per-example representations."""
    if T == 1:
        # per-token SAE
        z_per = encode_per_token(model, anchor_acts)   # (N, S, d_sae)
        z = z_per[:, T-1:].mean(axis=1)                # drop first T-1 (=0), mean
    else:
        # window arch
        K = S - T + 1
        windows = slide_windows(anchor_acts, T)        # (N, K, T, d_in)
        z_per_win = encode(model, windows)             # (N, K, d_sae)
        # Drop windows whose left edge < T-1: those bleed beyond the tail boundary.
        # Equivalently: keep windows starting at position >= T-1.
        z = z_per_win[:, T-1:].mean(axis=1)            # (N, d_sae)
    return z
```

For per-token archs (T=1), drop is degenerate (drop first 0). For
window archs at T>1, drop the first T−1 windows so all averaging
units cover tokens entirely within the considered tail. (Per-token
archs do NOT additionally drop tokens — the tail is the same for
both, so the only asymmetry is that window archs start their first
"valid" averaging unit T-1 tokens later. For S>>T this is negligible.)

Probe: SAEBench protocol. Top-`k_feat`-by-`|mean_pos − mean_neg|` +
L1 LR (C=1.0, max_iter=2000, `with_mean=False` standardization).
Headline `k_feat = 5`; ablation at `k_feat ∈ {1, 2, 20}`.

#### Reported S values

- **S = 128 (headline)**: long-tail; minimal boundary asymmetry.
- **S = 20 (continuity)**: matches Phase 5's tail length; lets us
  validate that Phase 7 numbers match Phase 5 in the limit S=20.

NOT reported: S = T (per-window comparison) — explicitly dropped
because comparing across architectures with different T at S=T is
structurally confusing.

#### FLIP convention

Apply `max(AUC, 1−AUC)` per-task on `winogrande_correct_completion`
and `wsc_coreference` (cross-token tasks with arbitrary label
polarity). Same as Phase 5.

### Hypotheses (pre-registered)

- **H1 (Gemma-IT vs base)**: per-task AUCs may change by ~0.01-0.02
  on most tasks; ARCH RANKINGS likely preserve. If rankings flip
  significantly, this is itself a finding worth reporting.
- **H2 (k_win=500 fixed across archs)**: bumping MLC family from k=100
  to k=500 should improve their absolute AUC moderately (more candidate
  features). Whether the lp/mp ranking of MLC vs TXC families
  preserves is the key open question.
- **H3 (long-tail S=128 vs S=20)**: AUC increases for all archs
  with S=128 (more context). Spread between archs may compress or
  expand. Open question.
- **H4 (TXC vs SAE at fix k_win=500)**: TXC family wins by ≥ 0.005
  mp at S=128. If not, the entire "TXC > SAE" claim weakens.

### Architectures × seeds matrix — owned by Agent A

12 archs (or 13 including `tsae_paper_k20` baseline) × 3 seeds = 36-39
ckpts. At ~5-10 min each on H100, ~6-7 hr serial. Acceptable.

### Figures committed (Agent A produces)

- `phase7_headline_bar_S128.png` — paired bar chart at headline S.
- `phase7_S_sweep.png` — line plot, AUC vs S for top archs.
- `phase7_seed_variance.png` — error bars on top archs.

### Figures committed (Agent B produces)

- `phase7_qualitative_pareto.png` — TXC-vs-T-SAE Pareto from Phase 6.
  Re-rendered with Phase 7 ckpts.
- `phase7_top8_panel.png` — top-8-by-variance feature visualization
  for each canonical arch on concat-A and concat-B.

### Ablations / appendix tables

- **k_win=100 ablation** (Agent A, optional if time permits): same
  archs at k_win=100, S=128. Shows effect of sparsity convention.
  ~6 archs × 1 seed = ~3 hr extra.
- **k_feat ∈ {1, 2, 20}**: probe-budget sweep. Same ckpts, just
  different probe `k_feat` (cheap re-probing).
- **S sweep**: AUC vs S for the headline archs at fix k_win.
- **Negative results**: D1 strided, C-family token-level, F subset-
  encoder. Numbers from Phase 5B at IT regime; note "not retrained
  for Phase 7 due to clean negative result on the prior setup."

### Branch hygiene — both agents

- All Phase 7 work on `han-phase7-unification`, branched off
  `origin/han`.
- Cherry-picked arch files only (NOT phase5b/6 experiment infra).
- New ckpts in `experiments/phase7_unification/results/ckpts/`
  (gitignored, sync to HF at `han1823123123/txcdr/phase7_ckpts/`).
- Probing results jsonl at
  `experiments/phase7_unification/results/probing_results.jsonl`.
- Autointerp results jsonl at
  `experiments/phase7_unification/results/autointerp/<arch>/<concat>_labels.json`.
- Commits: one per arch family + ablation + each writeup
  (`<DATE>-<topic>.md`).
- No force-push to `han`. PR or fast-forward only after manual
  human verification.

### Files Phase 7 MUST NOT modify

To preserve historical reproducibility:

- Phase 5's `experiments/phase5_downstream_utility/results/ckpts/`
  (Phase 5 agent's untouched ckpt set).
- Phase 5's `probing_results.jsonl`, `training_index.jsonl`.
- Phase 6's `experiments/phase6_qualitative_latents/results/`.
- Phase 5B's `experiments/phase5b_t_scaling_explore/results/`.
- The 36-task probe cache for Gemma-IT (read-only reference; new
  Gemma-base cache lives at a separate path).

Phase 7 writes ONLY to `experiments/phase7_unification/`.

### What this phase will NOT do

- Train new architectures beyond the canonical 12.
- Modify the SAEBench-style probing protocol (top-k-by-class-sep +
  L1 LR is unchanged).
- Run autointerp protocol changes beyond Phase 6's protocol (rerun
  on new ckpts, but no protocol changes).
- Cross-token tasks (winogrande, wsc) get the FLIP convention as
  before — no change.
- Train on layer ≠ 13 (anchor) or != L11-L15 (MLC).
- Use `last_position` as a separate metric in the headline. Reported
  only as a caveat-laden footnote in the paper, if at all.

### Coordination protocol between Agent A and Agent B

- **Day 1**: both agents branch `han-phase7-unification` independently.
  Both pull arch files via cherry-pick. Agent A starts cache build;
  Agent B starts passage build + Phase 6 pipeline port.
- **Day 2-3**: Agent A pushes activation cache + probe cache to HF.
  Agent B pulls activation cache (or builds locally — passages may
  need different sequences than the 24k FineWeb cache).
- **Day 3-4**: Agent A starts pushing trained ckpts to HF as they
  complete. Agent B polls HF for new ckpts; for each new ckpt, runs
  the autointerp pipeline immediately.
- **Day 5-6**: Agent A finishes all ckpts. Agent B catches up on
  any remaining autointerp.
- **Day 6+**: both agents have all data they need. Run final
  probing/autointerp passes. Generate figures.
- **Day 7-9**: write-up phase. Both agents draft their respective
  sections of the unified summary; merge into one
  `phase7_unification/summary.md` near deadline.
