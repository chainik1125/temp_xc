---
author: Han
date: 2026-04-26
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

- Source: 24 000 sequences from FineWeb (consistent with prior
  Phase 5/5b pipeline; document the FineWeb-vs-Pile difference vs
  T-SAE/TFA in the limitations section).
- Context length: 128 tokens.
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
- Storage: ~3 GB per task × 36 ≈ 110 GB for L13 anchor + ~30 GB for
  L11-L15 stack at S=20 ≈ **~140 GB**. Verify RunPod volume.
- Sync to HF for cross-agent access:
  `han1823123123/txcdr/phase7_probe_cache/`.

### Sparsity convention

`k_win = 500` across all archs and all T values, fixed.

### Probing protocol — pre-registered (owned by Agent A)

Single S-parameterized aggregation:

```python
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
        # Drop windows whose left edge < T-1 (would bleed beyond tail).
        z = z_per_win[:, T-1:].mean(axis=1)            # (N, d_sae)
    return z
```

Probe: SAEBench protocol. Top-`k_feat`-by-`|mean_pos − mean_neg|` +
L1 LR (C=1.0, max_iter=2000, `with_mean=False` standardization).
Headline `k_feat ∈ {5, 20}`; ablation at `k_feat ∈ {1, 2}`.

#### Reported S values

- **S = 128 (headline)**: long-tail; minimal boundary asymmetry.
- **S = 20 (continuity)**: matches Phase 5's tail length; lets us
  sanity-check Phase 7 numbers vs Phase 5 in the limit S=20.

NOT reported: S = T (per-window comparison) — explicitly dropped
because comparing across architectures with different T at S=T is
structurally confusing.

#### FLIP convention

Apply `max(AUC, 1−AUC)` per-task on `winogrande_correct_completion`
and `wsc_coreference` (cross-token tasks with arbitrary label
polarity). Same as Phase 5.

---

## Agent A — sparse-probing leaderboard

### Deliverable A.i — definitive leaderboard

For all canonical archs (table below), 3 seeds, headline at S=128 and
k_feat ∈ {5, 20}. Per-arch: mean ± σ over 3 seeds. Output:

- `experiments/phase7_unification/results/probing_results.jsonl` —
  full per-task per-seed per-k_feat per-S rows.
- `experiments/phase7_unification/results/headline_S128_k5.json`
  + `headline_S128_k20.json` — aggregated mean±σ per arch.
- `experiments/phase7_unification/results/plots/phase7_headline_bar_S128_k5.png`
  + `..._k20.png` — paired bar charts.

### Deliverable A.ii — T-sweep at fix k_win=500

For arch ∈ {`txcdr_t<T>`, `H8_t<T>`}, train at T ∈
{3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32} with
`k_win = 500` (so per-slab `k_pos = 500/T`), 3 seeds each.
Anchor cell: at T=20 ALSO train at fix `k_pos = 100` (so k_win =
2000) for both archs, 3 seeds — to disentangle "context limit"
from "per-slab sparsity collapse" interpretations.

Per-cell metrics in the JSON output:

- `test_auc` at S=128, k_feat=5
- `test_auc` at S=128, k_feat=20
- `test_auc` at S=20, k_feat=5 (continuity check)
- `alive_fraction`: fraction of d_sae features that fired ≥ once on
  a 5K-token held-out batch
- `final_recon_loss`: last logged reconstruction loss
- `final_step`, `converged`, `plateau_last`: convergence sanity

Output:

- `experiments/phase7_unification/results/t_sweep_results.jsonl` —
  per-cell rows.
- `experiments/phase7_unification/results/plots/phase7_t_sweep_S128_k5.png`
  — line plot AUC vs T for {txcdr, H8}, with anchor cell.
- `experiments/phase7_unification/results/plots/phase7_t_sweep_alive_fraction.png`
  — alive_fraction vs T (sanity-check plot to flag sparsity
  collapse at large T).

**Caveat documented in the writeup**: at fix k_win=500, per-slab
k_pos = 500/T shrinks with T. By T=32, k_pos ≈ 16 per slab. If
T=32's AUC regression coincides with alive_fraction collapse,
the regression is partly attributable to per-slab under-training,
not architecture-intrinsic context limit. The anchor cell at
T=20 (fix k_pos=100) tests this disentanglement directly.

### Canonical arch set — all in the leaderboard, 3 seeds

Because T-sweep entries also serve as leaderboard entries (same
seeds), the "leaderboard" and "T-sweep" share rows. Distinct archs:

**Per-token / non-TXC family (7 archs):**

| arch | family | notes |
|---|---|---|
| `topk_sae` | per-token TopK | k_win=500 (5× Phase 5 default) |
| `tsae_paper_k500` | per-token Matryoshka BatchTopK + InfoNCE | T-SAE port at our k convention |
| `tsae_paper_k20` | same, native k=20 | paper-faithful reference |
| `mlc` | per-token TopK over 5 layers | k_win=500 globally (5× Phase 5 default) |
| `mlc_contrastive_alpha100_batchtopk` | MLC + matryoshka + InfoNCE + BatchTopK | retrain at k=500 |
| `agentic_mlc_08` | MLC + multi-scale InfoNCE | retrain at k=500 |
| `tfa_big` | TFA full | predictive-coding reference; verify TFA's training stack supports k_win=500 |

**TXC variants — fixed-T archs (8 archs):**

| arch | T | notes |
|---|---|---|
| `agentic_txc_02` | 5 | multi-scale matryoshka TXC; Phase 5 multi-scale winner |
| `txc_bare_antidead_t5` (Track 2) | 5 | bare TXC + anti-dead stack only |
| `txc_bare_antidead_t10` (Track 2 T=10) | 10 | same recipe, T=10 |
| `txc_bare_antidead_t20` (Track 2 T=20) | 20 | same recipe, T=20 |
| `phase5b_subseq_track2` (B2) | T_max=10, t_sample=5 | subseq sampling on Track 2 base |
| `phase5b_subseq_h8` (B4) | T_max=10, t_sample=5 | subseq sampling on H8 stack |

**TXC T-sweep — txcdr (16 archs):**

`txcdr_t<T>` for T ∈ {3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32}.
Each at k_win=500 fixed.

**TXC T-sweep — H8 (16 archs):**

`phase57_partB_h8_bare_multidistance_t<T>` for T ∈ {3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32}.
Each at k_win=500 fixed. H8 multi-distance shifts auto-scale per
Phase 5's convention: `(1, max(1, T//4), max(1, T//2))` deduped.

**Anchor cells (fix k_pos=100):**

| arch | T | k_win | notes |
|---|---|---|---|
| `txcdr_t20_kpos100` | 20 | 2000 | anchor at fix k_pos=100 |
| `phase57_partB_h8_bare_multidistance_t20_kpos100` | 20 | 2000 | same |

**Total**: 7 + 6 + 16 + 16 + 2 = **47 archs** × 3 seeds = **141 trainings**
on H100. At 5-10 min each → 12-24 hours of compute.

(Note: leaderboard entries that also appear in T-sweep — e.g.
`txcdr_t5` is both a leaderboard arch and a T-sweep cell — are
counted once. The counts above already de-duplicate.)

### Hypotheses (pre-registered)

- **H1 (Gemma-IT vs base)**: per-task AUCs may change by ~0.01-0.02
  on most tasks; ARCH RANKINGS likely preserve. If rankings flip
  significantly, this is itself a finding worth reporting.
- **H2 (k_win=500 fixed across families)**: bumping per-token SAE
  and MLC families up to k_win=500 from Phase 5's k=100 should
  improve their absolute AUC moderately (more candidate features).
  Whether the cross-family ranking (TXC vs MLC vs SAE) preserves is
  the key question.
- **H3 (T-sweep at fix k_win)**: Phase 5's T=5 peak was at fix
  k_pos=100 (k_win = 100·T). At fix k_win=500 the peak may shift,
  flatten, or persist. Three sub-hypotheses:
    - **H3a (sparsity-dilution)**: peak shifts to larger T, T-scaling
      becomes near-monotone.
    - **H3b (context-mismatch)**: peak stays near T=5–6, regardless
      of k convention.
    - **H3c (sparsity-collapse at large T)**: T ≥ 24 underperforms
      due to per-slab k_pos < 20 being too sparse. Anchor cell at
      T=20 fix-k_pos=100 disentangles vs H3b.
- **H4 (TXC vs SAE at fix k_win=500 + S=128)**: TXC family wins by
  ≥ 0.005 mp at headline. If not, the entire "TXC > SAE" claim
  weakens significantly.

### Figures Agent A produces

- `phase7_headline_bar_S128_k5.png` — paired bar chart at headline.
- `phase7_headline_bar_S128_k20.png` — same at k_feat=20.
- `phase7_t_sweep_S128_k5.png` — line plot (txcdr, H8) AUC vs T.
- `phase7_t_sweep_alive_fraction.png` — alive_fraction vs T.
- `phase7_seed_variance.png` — error bars.
- (Optional) `phase7_S_sweep.png` — AUC vs S for top archs.

---

## Agent B — qualitative autointerp

### Deliverable B.i — Top-256 cumulative semantic Pareto

Single Pareto plot, replicating
`experiments/phase6_qualitative_latents/results/phase63_pareto_top256.png`
from `han-phase6` but with Phase 7 archs. **Skip the other 3 Pareto
plots from Phase 6** (top-N=32, top-N=64, top-N=128 cumulative are
not part of Phase 7's deliverable).

For each arch in the Phase 7 leaderboard:

1. Take the top-256 features by per-token activation variance over
   `concat_A + concat_B + concat_random` (Phase 6 protocol).
2. For each of those 256 features, send the top-10 activating
   20-token contexts to Claude Haiku 4.5 with the Bills-et-al-style
   labelling prompt. Get a one-line label.
3. Hand-classify each label as **semantic** or **non-semantic**
   (Phase 6's protocol; semantic = names a concept/topic/theme, not
   punctuation/whitespace/syntax/format pattern).
4. Plot **cumulative semantic count** vs **rank** (rank = position
   in variance-sorted top-256), giving a curve per arch.
5. The headline figure is the FINAL cumulative count at rank 256
   (i.e., total semantic features in top-256), plotted on the y-axis
   against the arch's sparse-probing AUC at S=128, k_feat=5 on the
   x-axis. Arches in the upper-right are Pareto-better.

Output:

- `experiments/phase7_unification/results/autointerp/<run_id>/concat_<A|B|random>_labels.json`
  — per-arch per-feature labels.
- `experiments/phase7_unification/results/autointerp/<run_id>/cumulative_semantic.json`
  — per-arch cumulative-count curves.
- `experiments/phase7_unification/results/plots/phase7_qualitative_pareto_top256.png`
  — the headline Pareto figure.

### Cost-saving: autointerp at seed=42 only

To keep Claude Haiku spend reasonable, **autointerp scoring uses
seed=42 only** for each arch (not 3 seeds). Sparse probing on the
x-axis still uses 3-seed mean. This is a documented compromise.

Scale check at seed=42 only:

- 47 archs × 256 features × 10 contexts × 1 seed ≈ 120K Haiku calls.
- At ~5K input tokens / call with prompt caching, expected cost
  ~$50-150. Tractable.

### Coordination dependency

Agent B's Pareto plot REQUIRES Agent A's `probing_results.jsonl` to
populate the x-axis (the sparse-probing AUC). Coordination:

- Agent A pushes ckpts to HF as they complete (incremental, per arch).
- Agent A pushes a partial `probing_results.jsonl` to HF after
  probing each batch of archs.
- Agent B polls HF for ckpts; for each new ckpt, runs autointerp.
- Agent B reads Agent A's `probing_results.jsonl` (latest snapshot)
  to compose the Pareto plot.
- Final Pareto plot is generated AFTER Agent A finishes the full
  3-seed sparse-probing pass.

### Figures Agent B produces

- `phase7_qualitative_pareto_top256.png` — the single deliverable.
- (Optional) `phase7_top8_panel.png` — top-8-by-variance feature
  activation panel for each canonical arch on concat-A and concat-B.
  Phase 6.1's standard qualitative figure, useful for paper but not
  required.

---

## Branch hygiene — both agents

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

- Train new architectures beyond the canonical set above.
- Modify the SAEBench-style probing protocol (top-k-by-class-sep +
  L1 LR is unchanged).
- Run autointerp protocol changes beyond Phase 6's protocol (rerun
  on new ckpts, but no protocol changes).
- Cross-token tasks (winogrande, wsc) get the FLIP convention as
  before — no change.
- Train on layer ≠ 13 (anchor) or != L11-L15 (MLC).
- Use `last_position` as a separate metric in the headline. Reported
  only as a caveat-laden footnote in the paper, if at all.
- Reproduce Phase 6's other 3 Pareto plots (top-32, top-64, top-128).
  Top-256 is the deliverable.

### Coordination protocol between Agent A and Agent B

- **Day 1**: both agents branch `han-phase7-unification` independently.
  Both pull arch files via cherry-pick. Agent A starts cache build;
  Agent B starts passage build + Phase 6 pipeline port.
- **Day 2-3**: Agent A pushes activation cache + probe cache to HF.
  Agent B pulls activation cache (for passage encoding).
- **Day 3-5**: Agent A starts pushing trained ckpts to HF as they
  complete. Agent B polls HF for new ckpts; for each new ckpt at
  seed=42, runs the autointerp pipeline immediately.
- **Day 5-6**: Agent A finishes all 3-seed trainings + sparse-
  probing pass. Agent A pushes final `probing_results.jsonl`.
- **Day 6**: Agent B catches up on autointerp for any seed=42 ckpts
  not yet processed. Pulls Agent A's final probing jsonl. Generates
  Pareto plot.
- **Day 7-9**: write-up phase. Both agents draft their respective
  sections of the unified summary; merge into one
  `phase7_unification/summary.md` near deadline.
