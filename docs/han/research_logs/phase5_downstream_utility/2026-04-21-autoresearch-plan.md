---
author: Han
date: 2026-04-21
tags:
  - proposal
  - in-progress
---

## Phase 5.7 autoresearch — agent-driven arch + hyperparam exploration

Handover document for the post-compact agent. This replaces T17 (seed
variance) with **agent-driven autoresearch** on the top-3 SAEs
(TXCDR-T5, MLC, time_layer_crosscoder_t5). Goal: push these
architectures as far as possible against the strong-baseline wall
before the NeurIPS deadline (**~14 days out from 2026-04-21**).

### Why scrap T17

T17 was a 3-seed variance study on 5 existing archs — useful for
publishing "we ran multiple seeds" but did not test new ideas. Phase
5.7 instead uses the compute budget to *discover* architectural
improvements. Seed variance can be added as a final sanity step on
whatever winners emerge.

### Current state (2026-04-21 ~13:30)

- **Committed up to `883faa7`** on branch `han`.
- **T17 seed 1 fully complete** (5 archs trained + probed, committed
  at `9b429ec`). Ckpts retained for reference.
- **T17 seed 2 training**: all 5 archs' state dicts saved to disk
  (`{time_layer,mlc,txcdr_rank_k,txcdr_t5,txcdr_tied}_crosscoder_t5__seed2.pt`),
  but probing rows not yet emitted. Ckpts retained as extra seed-1/2
  data in case we want to revisit variance.
- **Memory**: cgroup sits at ~45 GB (page cache at limit; 7–8 GB
  actual Python RSS; failcnt still 0 — not a concern).

### Step 0 — kill T17 before starting Wave 1

```bash
pkill -f "run_overnight_phase5.sh"
pkill -f "train_primary_archs"
pkill -f "run_probing.py"
```

Verify nothing is left running before starting Wave 1 (`ps -ef | grep
run_ | grep -v grep`).

### CRITICAL: fairness rules

If we only bump d_sae (or any other "capacity" variable) for one arch,
that arch gets more features and could win not because of the arch
but because of the extra capacity. That breaks Kantamneni/SAEBench
convention. Rules:

**Scope**: these rules apply to the **Phase-5.7 autoresearch cohort
only** (TXCDR-T5, MLC, time_layer_crosscoder_t5). The existing 25-arch
canonical benchmark (summary.md) stays FROZEN at matched defaults —
we do NOT re-train the other 22 archs at any Wave-1 bumped setting.
The paper reports two tables side-by-side:

- **Canonical benchmark** (25 archs × matched defaults) — apples-to-
  apples architecture comparison, unchanged from summary.md.
- **Tuned leaders** (3 archs × their individually-best configs after
  autoresearch) — shows how far each top arch can be pushed. Each
  row in this table discloses its full hyperparameter block
  (d_sae, k, steps, lr, special losses).

Rules within the 3-arch autoresearch cohort:

1. **d_sae is FIXED at per-arch defaults** during Wave 1:
   - TXCDR-T5: 18432
   - MLC: 18432
   - time_layer_crosscoder_t5: 8192

   The only place d_sae moves is **Wave 2b** (a dedicated capacity
   study where all 3 archs bump together at a common multiplier).
2. **Sparsity `k` and training length CAN be bumped per-arch in Wave
   1**, but the orchestrator ships each "axis variant" as a triple
   (e.g., `bigk` is run for all three archs). That way, if a bump
   helps one arch but not another, we still have a matched-
   hyperparam comparison on record. If a triple-member is clearly
   non-promising (val AUC < base − 2 pp at plateau), we can abandon
   that one arm early — the leader table reports "each arch at its
   own best" with hyperparams disclosed.
3. **Arch-internal hyperparams have no fairness issue** — they only
   exist in that arch. Examples: `mlc_contrastive`'s `α` contrastive
   weight; Matryoshka H/L partition sizes; `txcdr_lowrank_dec`'s
   rank-K. These are legitimate single-arch tuning.
4. **A candidate that changes multiple dimensions at once is fine**
   (e.g., `txcdr_contrastive_t5` adds Ye et al. InfoNCE + Matryoshka
   H/L to TXCDR-T5). Record the full hyperparameter block.
5. **Disclose everything in the final leader table**: each row's
   d_sae, k, training_steps, lr, special losses. Make it easy for a
   reviewer to see what's arch vs hyperparam.
6. **Do NOT re-run the 22 non-cohort archs** at any bumped setting.
   The canonical 25-arch bench is the arch comparison; the tuned
   leaders is the push-the-leaders result. Keep them separate in
   the writeup.

### Val / test split

The existing probing protocol: per task, 3040 train + 760 test
examples. During autoresearch we must NOT peek at test too much.

Add a **train → train' + val** split:

- Per task, split the 3040 train examples deterministically into
  **train' = 2432** + **val = 608** (80 / 20).
- Seed the split from `dataset_key` (not a global seed) so all archs
  on the same task see the same split.
- Keep the 760 test examples untouched.

Protocol during autoresearch:
- Fit the probe on train' (2432).
- Report **val AUC** in the autoresearch_index.jsonl.
- Agent iterates on val numbers.

Protocol for finalists:
- Fit the probe on the **full train (3040)**.
- Evaluate on the **held-out test (760)** at BOTH `last_position` and
  `mean_pool`.
- Report this as the final number.

Implementation: extend `run_probing.py` with
`--aggregation last_position_val` (and `mean_pool_val`) that
internally splits. Emit JSONL rows tagged with the new aggregation
values so they don't collide with existing rows. Full-train rows stay
tagged `last_position` / `mean_pool` as today — existing headline
numbers are unchanged.

### Iteration strategy — three waves

Iteration budget: ~14 days × ~16 h/day ≈ 220 GPU-hours.

#### Wave 1 — hyperparam sweep on existing archs (~12 h, no new code)

All 8 candidates: no new arch classes, just parameter changes
threaded through `train_primary_archs.py`.

| # | name | change vs default | est. train |
|---|---|---|---|
| 1 | `txcdr_t5_bigk` | k_win 500 → 1000 (k_pos 100 → 200), uniform 2× k across all archs ✓ | ~30 min |
| 2 | `mlc_bigk` | k 100 → 200 (matches 2× multiplier) ✓ | ~12 min |
| 3 | `time_layer_bigk` | k_pos 100 → 200 (matches 2× multiplier) ✓ | ~45 min |
| 4 | `txcdr_t5_long` | 25k → 50k steps, cosine lr decay (fair ✓) | ~30 min |
| 5 | `mlc_long` | 25k → 50k steps, cosine lr decay (fair ✓) | ~12 min |
| 6 | `time_layer_long` | 25k → 50k steps, cosine lr decay (fair ✓) | ~45 min |
| 7 | `mlc_contrastive_alpha5` | α 0.1 → 0.5 (arch-internal) ✓ | ~15 min |
| 8 | `mlc_contrastive_bigh` | H head 0.5 → 0.75 of d_sae (arch-internal) ✓ | ~15 min |

Each is probed at `last_position_val` only (~8 min). Wave 1 total:
~5–6 hours.

#### Wave 2 — new arch classes (~2–3 days)

Each needs real code (~100–300 lines in `src/architectures/`).

| # | name | idea | est. |
|---|---|---|---|
| 1 | `txcdr_contrastive_t5` | TXCDR-T5 + Ye et al. InfoNCE on (t−1, t) paired windows + Matryoshka H/L. Combines the two top findings (TXCDR-T5 best mean_pool + mlc_contrastive's +0.8 pp contrastive lift). | ~35 min |
| 2 | `mlc_temporal_t3` | MLC extended across 3 adjacent tokens: input (B, 3, L, d), encode per token with shared weights, joint decode. Genuinely new. | ~45 min |
| 3 | `txcdr_jumprelu_t5` | JumpReLU instead of TopK — continuous sparsity with per-feature learnable threshold. Fairer than TopK when d_sae is small. | ~30 min |
| 4 | `txcdr_warmstart_t5` | init TXCDR-T5 decoder columns from a trained `topk_sae`; train only temporal-delta params. Tests if the training budget is the bottleneck. | ~50 min |
| 5 | `mlc_aux_ortho` | MLC with an auxiliary decoder-orthogonality regularizer (encourage feature diversity). | ~15 min |
| 6 | `time_layer_contrastive_t5` | time_layer + InfoNCE on (t−1, t) layer-slabs. Tests whether contrastive helps time_layer like it helps MLC. | ~50 min |

Branch Wave 2 based on Wave 1 findings: if contrastive adds to TXCDR
the way it added to MLC (+0.8 pp), prioritize all contrastive
variants. If training-length helped, prioritize warmstart. Etc.

#### Wave 2b — capacity study (fair d_sae bump, one shot)

To answer "does d_sae matter for the best archs?" fairly, bump d_sae
for ALL three archs together:

- `txcdr_t5_d36k`   — d_sae 18432 → 36864
- `mlc_d36k`        — d_sae 18432 → 36864
- `time_layer_d16k` — d_sae 8192 → 16384 (2× multiplier)

Same training_steps / k as current defaults. Report as a 3-row
capacity comparison table. ~2–3 hours total.

#### Wave 3 — hyperparam micro-grids on Wave 1/2 winners (~1 week)

For each finalist, run 3×3 grid over (lr, training_steps) or (k,
alpha). This avoids dismissing a good arch for a bad hyperparam
choice. Plus **final test-set evaluation** on held-out test at BOTH
`last_position` + `mean_pool`.

**"Good arch looks bad" safeguard**: if a Wave 1/2 candidate lands
within 2 pp of its base arch at plateau, rerun at 50k steps without
plateau-stop before dismissing. Budget: 2 retries per candidate max.

### Infrastructure

Files to touch:

- `experiments/phase5_downstream_utility/probing/run_probing.py`:
  add val/test split support. New `--aggregation` choices
  `last_position_val`, `mean_pool_val`. Implementation: after
  loading the task cache, split `train_acts` / `train_labels` /
  `train_last_idx` deterministically seeded from the dataset key.
  Use train' to fit the probe, evaluate on val (for `*_val`) or
  reassemble full train for finalist `last_position` / `mean_pool`
  (existing behaviour).
- `experiments/phase5_downstream_utility/train_primary_archs.py`:
  add argparse flags `--k-multiplier`, `--max-steps`, `--lr`,
  `--cosine-decay`. Thread into `TrainCfg`. Add dispatcher entries
  for `mlc_contrastive_alpha5`, `mlc_contrastive_bigh`, etc.
  (parameterized variants of existing archs).
- `src/architectures/<new_arch>.py`: one file per Wave 2 class.
- `experiments/phase5_downstream_utility/run_autoresearch.sh`:
  new orchestrator that sequentially trains + probes + commits each
  candidate. Logs to `logs/overnight/autoresearch_*.log`. One commit
  per candidate.
- `experiments/phase5_downstream_utility/results/autoresearch_index.jsonl`:
  one row per candidate with `(name, hyperparams, val_auc_mean,
  val_auc_std, test_auc_final, elapsed_s, notes)`. Human-readable
  source of truth for what's been tried.
- `experiments/phase5_downstream_utility/plots/plot_autoresearch.py`:
  generates a plot of val AUC over candidates ordered by run time
  — quick visual of what worked.

### Tracking / output files

Each Wave 1 candidate emits:
- `results/ckpts/<cand>__seed42.pt`  (~850 MB per ckpt — budget ~8
  × 0.85 = 7 GB for Wave 1, ~20 GB total through Wave 3).
- `results/training_logs/<cand>__seed42.json`.
- `probing_results.jsonl` rows tagged `aggregation=last_position_val`
  (during autoresearch) and later `last_position` + `mean_pool`
  (for finalists on test set).
- `autoresearch_index.jsonl` summary row.
- `logs/overnight/autoresearch_<cand>.log`.

### Wave 1 candidate specs (concrete configs)

Defaults (all fair):
- seed = 42 (single-seed during autoresearch; add 3-seed variance
  on finalists in Wave 3 if time permits).
- d_sae: 18432 (TXCDR/MLC), 8192 (time_layer) — UNCHANGED.
- optimizer: Adam, lr 3e-4, batch 1024, grad_clip 1.0.
- plateau: 2 % loss drop threshold over 1k steps, min_steps 3000.

Candidate-specific overrides (all else = defaults):

```python
CANDIDATES = [
    # name,                         base arch,     overrides
    ("txcdr_t5_bigk",               "txcdr_t5",   {"k_pos": 200, "k_win": 1000}),
    ("mlc_bigk",                    "mlc",        {"k_pos": 200}),
    ("time_layer_bigk",             "time_layer", {"k_pos": 200}),
    ("txcdr_t5_long",               "txcdr_t5",   {"max_steps": 50_000, "cosine_decay": True}),
    ("mlc_long",                    "mlc",        {"max_steps": 50_000, "cosine_decay": True}),
    ("time_layer_long",             "time_layer", {"max_steps": 50_000, "cosine_decay": True}),
    ("mlc_contrastive_alpha5",      "mlc_contrastive", {"alpha": 0.5}),
    ("mlc_contrastive_bigh",        "mlc_contrastive", {"h_frac": 0.75}),
]
```

### Open questions / decisions for the next agent

1. **cosine lr decay vs constant**: cosine usually helps at long
   training. Reuse the decay schedule from `src/training/lr.py`
   if it exists; else constant is fine for Wave 1.
2. **3-seed variance on finalists**: probably yes in Wave 3, but
   only on the 2-3 winners (not all 8 Wave-1 candidates).
3. **Stopping criteria for a candidate**: val AUC < (base arch val
   AUC − 2 pp) at plateau ⇒ abandon; val AUC within ±1 pp ⇒ retry
   at 50k steps; val AUC > (base + 1 pp) ⇒ mark as finalist.
4. **Budget allocation**: start Wave 1 → Wave 2 → Wave 2b → Wave 3.
   Between waves, review autoresearch_index.jsonl and re-prioritize.
5. **Running in parallel**: GPU is a single A40; one train or one
   probe at a time. Orchestrator must serialize. Existing pgrep-wait
   pattern works.

### Resume checklist (for the post-compact agent)

1. `git log --oneline -n 10` — confirm at `883faa7` or later.
2. Read this doc, then read
   [`summary.md`](summary.md) (status of the 25-arch bench).
3. Run Step 0 above to kill T17.
4. Implement val/test split in `run_probing.py` (~30 min).
5. Extend `train_primary_archs.py` with the 4 parameterizable
   overrides (`k_multiplier`, `max_steps`, `lr`, `cosine_decay`).
   Add dispatcher branches for the 8 Wave-1 candidate names.
6. Write `run_autoresearch.sh` — simple sequential orchestrator.
7. Launch Wave 1. Commit + push after each candidate.
8. Review Wave 1 results; plan Wave 2 candidates based on what
   worked.

### Hard resource constraints (unchanged from prior handoff)

- **Pod volume `/workspace`**: check `df -h /workspace`; don't let
  free drop under 5 GB. Each Wave 1 ckpt ~850 MB; 8 × 0.85 = 7 GB.
- **Container disk** (`/tmp`, `/home`): 150 GB free; fine for scratch.
- **cgroup memory limit: 46 GB.** Page cache will hug the limit
  (`cat /sys/fs/cgroup/memory/memory.stat | head`); failcnt=0 means
  safe. Python RSS typically 7–9 GB during probing.
- **Claude Code state**: /home/appuser/.claude symlinked to
  /workspace/claude_home. If pod is rebuilt, run
  `bash scripts/bootstrap_claude.sh` to restore the symlink before
  launching Claude.
