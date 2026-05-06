---
author: Han
date: 2026-04-20
tags:
  - in-progress
  - proposal
---

## Phase 5 overnight handoff — 2026-04-20

Handoff document for the next agent picking up after compact. Summarises
(a) what's currently running unattended, (b) the state of the phase,
(c) the queued overnight task list, and (d) concrete commands / files
to pick up.

**Current time at writing: ~21:45 UTC on 2026-04-20.**

### ⚠️ Hard resource constraints (read first)

- **Pod volume (`/workspace`)**: **~60 GB free** — this is where all
  `experiments/phase5_downstream_utility/results/` data lives (probe
  caches, checkpoints, JSONL, plots). **Every new checkpoint, cache,
  or large intermediate eats into this budget.** Check with
  `df -h /workspace` before big writes; abort writes when free space
  drops below 5 GB (consistent with the guard in
  `run_mlc_contrastive.sh`).
- **Container disk (`/tmp`, `/home`, everything not under `/workspace`)**:
  **~150 GB free, barely used.** Use this for large scratch intermediates
  that don't need to survive the pod (tempfiles, caches for one-off
  computations). Persistent artefacts still belong on the pod volume so
  they survive a restart.
- **Memory cgroup limit: 46 GB.** Breached via OOM kill; watch
  `cat /proc/<pid>/status | grep VmHWM` and
  `cat /sys/fs/cgroup/memory.current`. Our
  highest observed peak was **39.5 GB VmHWM** during `time_layer`
  full_window probing (now streaming-patched). Any new code that
  materialises large `(N, K, T, L, d)` tensors can blow this — prefer
  slide-by-slide streaming loops.
- **Claude Code state — RELOCATED to pod volume (2026-04-20 21:53 UTC).**
  `/home/appuser/.claude` is a symlink to `/workspace/claude_home`
  (6.66 MB on moosefs; previously 7.4 MB on container disk). Data
  persists across pod restart; the **symlink itself does not** (it
  lives on container disk). After any pod-settings change that wipes
  `/home/appuser/`, the next agent must recreate the symlink before
  launching Claude Code:

  ```bash
  bash /workspace/temp_xc/scripts/bootstrap_claude.sh
  ```

  The script is idempotent (no-op when symlink is already correct).
  Safety backup of original state: `/home/appuser/.claude.bak.20260420-215347`
  (delete when comfortable).

  **Other things a pod restart also kills** that pod-volume persistence
  does not rescue:
    1. All running background processes (the three live orchestrators
       at PIDs 1102, 2593, 4038 — would need to be re-launched from
       where they left off by inspecting `probing_results.jsonl` and
       `training_index.jsonl` to see which runs completed).
    2. Scheduled wakeups (the daemon that fires them dies with the
       pod; the wakeup records are data files in `/workspace/claude_home/`
       but have to be re-scheduled by a live agent).
    3. The Claude Code CLI binary if it's installed under `/home/appuser/`
       or `/usr/local/`. Reinstall if `which claude` returns nothing
       after the rebuild.

Planned writes for the overnight tasks (T15–T17):

| task | size |
|---|---|
| T15 predictions (7 archs × 36 tasks × 4 k × ~4 KB per task) | ~4 MB total |
| T17 × 15 new checkpoints × ~850 MB fp16 | **~13 GB** |
| Plots (T16 + regenerations) | < 100 MB |

T17 is the one that eats real pod-volume space. **Monitor during the run.**
After T17, we'll be at ~47 GB free on pod volume — tight for further
experiments without cleanup. Consider moving older / superseded
checkpoints (e.g. `txcdr_t3`, `txcdr_t8`, `txcdr_t15`, earlier weight-
sharing ablations) off to container disk once their probe rows are in
`probing_results.jsonl`.

### TL;DR for the next agent

Three orchestrator shell scripts are running unattended. All write to
`logs/overnight/*.log`. They each `pgrep` on the prior orchestrator's
script filename and sleep until it exits, so the chain is self-serialising.

1. **T-sweep orchestrator** — probing 5 new TXCDR T variants; ETA ~22:30.
2. **Mean-pool orchestrator** — waits for (1); probes all 24 archs at
   `mean_pool` aggregation; ETA ~23:30.
3. **mlc_contrastive orchestrator** — waits for (2); trains + probes
   the new `mlc_contrastive` arch; ETA ~00:30.

After (3) finishes, a **fourth orchestrator (not yet written)** is
planned to run the overnight collaborator-feedback tasks listed in
"Queued overnight" below: deprecate full_window in plots, patch
`--save-predictions`, re-probe top archs for confusion matrices, run
error-overlap analysis, and 3-seed autoresearch on top-5 archs. The
next agent should write and launch this fourth orchestrator.

### Currently running (do not kill)

| PID | Script | Log | Purpose | ETA |
|---|---|---|---|---|
| 1102 | `experiments/phase5_downstream_utility/run_fw_tsweep.sh` | `logs/overnight/fw_tsweep_orchestrator.log` | T-sweep probing (STEP C) + plots (STEP D) | ~22:30 |
| 2593 | `experiments/phase5_downstream_utility/run_mean_pool_probing.sh` | `logs/overnight/mean_pool_orchestrator.log` | Waits for (1); probes 24 archs at mean_pool | ~23:30 |
| 4038 | `experiments/phase5_downstream_utility/run_mlc_contrastive.sh` | `logs/overnight/mlc_contrastive_orchestrator.log` | Waits for (2); trains + probes `mlc_contrastive` | ~00:30 |

Check health via `ps -ef | grep -E "run_fw_tsweep|run_mean_pool|run_mlc_contrastive|run_probing|train_primary"`.

### State snapshot

#### Architectures trained (seed 42, all converged)

24 total:

- **Baselines**: `topk_sae`, `mlc`
- **TXCDR family**: `txcdr_t{2, 3, 5, 8, 10, 15, 20}` (5 new today — t2, t3, t8, t10, t15)
- **Stacked**: `stacked_t{5, 20}`
- **Matryoshka**: `matryoshka_t5`
- **Weight-sharing ablations**: `txcdr_{shared_dec, shared_enc, tied, pos, causal}_t5`
- **Time-sparsity novel**: `txcdr_block_sparse_t5`
- **Decoder-rank novel**: `txcdr_{lowrank_dec, rank_k_dec}_t5`
- **Time-contrastive (Ye et al.)**: `temporal_contrastive`
- **Time × Layer novel**: `time_layer_crosscoder_t5` (d_sae=8192, others 18432)
- **TFA**: `tfa_small`, `tfa_pos_small` (d_sae=4096)

**1 queued to train**: `mlc_contrastive` (MLC base + Matryoshka H/L + InfoNCE on adjacent-token high-level latents). New code at `src/architectures/mlc_contrastive.py`. Trainer: `train_mlc_contrastive` in `train_primary_archs.py`. Queued to run by orchestrator (3) above.

#### Probing caches — `experiments/phase5_downstream_utility/results/probe_cache/`

36 tasks (34 SAEBench-derived + `winogrande_correct_completion` + `wsc_coreference`). Each task dir contains:

- `acts_anchor.npz` — `(N, 20, 2304)` fp16, L13 tail-20.
- `acts_mlc.npz` — `(N, 5, 2304)` fp16, L11–L15 at each prompt's last real token.
- `acts_mlc_tail.npz` — `(N, 20, 5, 2304)` fp16, **rebuilt today** with `TAIL_MLC_N=20` (previously quota-limited at 5). ~66 GB total across 36 tasks.
- `meta.json` — dataset-level info + positive class frac.

Disk quota check: user flagged 60 GB remaining at the time of writing. One `mlc_contrastive` checkpoint (~850 MB fp16) is the only additional write queued; 3-seed autoresearch adds 15 × ~1 GB = ~15 GB. Orchestrator (3) aborts if `<5 GB` free.

#### Aggregations

| aggregation | formula | status |
|---|---|---|
| `last_position` | single-slot encoding at each prompt's last real token; for windowed archs, the T-window ending there, slot T−1 | **primary** |
| `full_window` | slide stride-1 T-windows across tail-20; concatenate K = 20−T+1 slide outputs into `(N, K·d_sae)` | **deprecating** (per user request) |
| `mean_pool` | same K slides as full_window but **average** them → `(N, d_sae)`; matches SAEBench's `get_sae_meaned_activations` convention | **in flight** |

`mean_pool` is implemented as a post-processing step on `full_window`'s output — one `np.reshape` + `.mean(axis=1)` — so it reuses the existing slide-encode GPU path with no extra forward passes. See `_encode_for_probe` prologue in `experiments/phase5_downstream_utility/probing/run_probing.py`.

#### Baselines

- `baseline_last_token_lr` — L2 logistic regression on raw last-token L13 activations.
- `baseline_attn_pool` — Kantamneni Eq. 2 attention-pooled probe.
- **36/36 coverage** at `last_position` and `full_window` (fixed today — previously 9/36 at full_window).
- Baselines at `mean_pool` will be added by the currently-running mean-pool orchestrator.

#### Key results — last_position × AUC × k=5 × 36 tasks

| arch | mean AUC |
|---|---|
| **baseline_attn_pool** | **0.9290** |
| **baseline_last_token_lr** | **0.9264** |
| mlc | 0.7943 |
| time_layer_crosscoder_t5 | 0.7928 |
| txcdr_rank_k_dec_t5 | 0.7852 |
| txcdr_t5 | 0.7822 |
| ... | ... |
| tfa_pos_small | 0.6390 |

**No SAE beats either baseline.** Outcome B (nuanced positive) with a Time×Layer wrinkle — matches the existing `summary.md`.

#### Collaborator feedback (source of the overnight tasks)

Collaborator reviewed the last_position figure and said:

1. "TXCDR-T5 is roughly on parity with MLC — slight indictment of the temporal-SAE field, but good enough to publish."
2. Wants **non-overlapping confusion matrices** between TXCDR-T5 and MLC (i.e., do they get *different* things right?).
3. Wants **autoresearch for T=5, 10, 20 TXCDR + MLC** (interpretation: multi-seed / hyper-param sweep to see if we can do better).
4. Wants individual latents compared to the T-SAE (Ye et al.) paper — **deferred to Phase 6** per user.

### Queued overnight tasks

Ready to kick off once the three live orchestrators finish (~00:30).

#### T13. Deprecate `full_window` from plots and summary

- `experiments/phase5_downstream_utility/plots/make_headline_plot.py:221`:
  change `for aggregation in ("last_position", "full_window", "mean_pool"):`
  to `for aggregation in ("last_position", "mean_pool"):`. Also update the
  module docstring at line 6.
- `experiments/phase5_downstream_utility/plots/make_headline_plot.py:86`:
  update the assertion to drop `"full_window"` if desired (optional; safe to leave).
- `docs/han/research_logs/phase5_downstream_utility/summary.md`: add a "deprecated" note to the full-window aggregation comparison table and flag that the canonical aggregations going forward are `last_position` + `mean_pool`. Do **not delete the table** — it's useful for reproducibility and the "why we changed our minds" narrative.
- **Don't delete** full_window rows from `probing_results.jsonl`.
- Delete the 8 `*_full_window_*.png` plot files (optional, for hygiene).

No GPU.

#### T14. Patch `--save-predictions` into the probing runner

- `experiments/phase5_downstream_utility/probing/run_probing.py`: add argparse flag `--save-predictions`. Default `False` (existing behaviour unchanged).
- Inside `sae_probe_metrics`, when flag set, capture `clf.decision_function(Xte_s)` + `clf.predict(Xte_s)` and emit a `.npz` to `results/predictions/<run_id>__<aggregation>__<task>__k<k>.npz` with keys: `example_id` (arange), `y_true`, `decision_score`, `y_pred`.
- Thread a `save_predictions: bool` argument through `run_probing(...)` and to `sae_probe_metrics`.
- Add `predictions` subdirectory to `results/` (gitignored — add to `.gitignore` if needed).

No GPU.

#### T15. Re-probe top archs for confusion matrices (last_position only)

Archs to re-probe:
- `mlc__seed42`
- `time_layer_crosscoder_t5__seed42`
- `txcdr_rank_k_dec_t5__seed42`
- `txcdr_t5__seed42`
- `txcdr_tied_t5__seed42`
- `topk_sae__seed42`
- `mlc_contrastive__seed42` (once trained)

Single probing command per arch:

```bash
.venv/bin/python experiments/phase5_downstream_utility/probing/run_probing.py \
    --aggregation last_position --skip-baselines --save-predictions \
    --run-ids <run_id>
```

GPU cost ≈ 45 min total for 7 archs (each ~5–7 min on last_position).

**Skip mean_pool for this analysis** — user explicitly agreed last_position is primary; mean_pool is secondary and can be revisited if findings are interesting.

#### T16. Error-overlap analysis + plot

Write a new analysis script at `experiments/phase5_downstream_utility/analyze_error_overlap.py`. For each of C(7, 2) = 21 pairs of archs × 36 tasks:

- **McNemar's χ² test**: from (wins_A_loses_B, wins_B_loses_A) 2×2 table, report p-value.
- **Jaccard of error sets**: |err_A ∩ err_B| / |err_A ∪ err_B|.
- **Per-task "A wins, B loses" fraction**: `mean(clf_A correct AND clf_B wrong)`.

Plots to produce (in `results/plots/`):

- `error_overlap_jaccard_k5_last_position.png` — 7×7 heatmap, Jaccard values. Diagonal = 1.0; low values = archs make different errors.
- `error_overlap_winsloss_k5_last_position.png` — 7×7 heatmap, "fraction A wins, B loses". Upper triangle = A's wins, lower = B's. High asymmetry = archs have complementary strengths.
- `error_overlap_per_task_mlc_vs_txcdr_t5.png` — per-task version of the "wins/loses" heatmap specifically for the mlc vs txcdr_t5 comparison (collaborator's explicit ask).

Emit a JSON summary at `results/error_overlap_summary.json` with per-pair statistics.

No GPU (sklearn + matplotlib, ~5 min).

#### Design requirement — autoresearch must be easy to extend

Autoresearch is **high-priority**. The collaborator explicitly asked for
it on T=5, 10, 20 TXCDR and MLC, and it's the highest-impact remaining
lever for improving the headline numbers. For it to actually pay off,
**adding a new architecture to the benchmark must not require massive
changes.**

Current friction (all the places you must touch to add one arch):

1. `src/architectures/<new_arch>.py` — the nn.Module class itself.
2. `experiments/phase5_downstream_utility/train_primary_archs.py` —
   (a) a new `train_<arch>` function, (b) an `elif arch == "<name>":`
   branch in `run_all`'s dispatcher (~10 lines).
3. `experiments/phase5_downstream_utility/probing/run_probing.py` —
   (a) an `elif arch == "<name>":` branch in `_load_model_for_run` to
   re-construct the class from `state_dict`, (b) if the encoder API
   doesn't match one of the existing dispatch branches (`topk_sae`,
   `mlc`, `txcdr_t*`, `stacked_t*`, TFA, …) in `_encode_for_probe`, a
   new branch there too.
4. `experiments/phase5_downstream_utility/plots/make_headline_plot.py`
   — append the new arch name to `ORDERED_ARCHS` so it renders in
   headline bars.
5. (optionally) `src/architectures/__init__.py` — add the `ArchSpec`
   export so the toy pipeline can use it.

For archs that reuse an existing encoder API (e.g. `mlc_contrastive`
subclasses `MultiLayerCrosscoder` and matches MLC's encode signature
exactly), steps 3(b) and 5 degenerate to a one-liner tuple membership.

**Refactor opportunity** (Phase 6 or bonus overnight work if compute
permits): introduce an `ARCH_REGISTRY` dict in
`experiments/phase5_downstream_utility/` that maps arch name →
`ArchEntry(spec_class, trainer_fn, encoder_dispatch, plot_order,
default_meta)`. `run_all`, `_load_model_for_run`, `_encode_for_probe`,
and `ORDERED_ARCHS` would all derive from this single source of
truth — adding a new arch then means (a) writing its module, (b)
registering one dict entry. Not strictly required for the overnight
tasks below, but the next agent should consider doing this *before*
T17 if they're going to iterate further, because each new arch the
collaborator proposes otherwise adds friction.

Alternatively, if the refactor is too invasive, at minimum the next
agent should **document the 5-step checklist above in a prominent
place** (e.g. a new `experiments/phase5_downstream_utility/ADDING_ARCHS.md`)
so proposing a new arch is mechanical, not archaeological.

#### T17. 3-seed autoresearch on top-5 archs

Train the top-5 archs at seeds {1, 2, 3} (we already have seed 42). 15 new runs:

- `mlc__seed{1,2,3}`
- `time_layer_crosscoder_t5__seed{1,2,3}`
- `txcdr_rank_k_dec_t5__seed{1,2,3}`
- `txcdr_t5__seed{1,2,3}`
- `txcdr_tied_t5__seed{1,2,3}`

Estimated GPU time (scaling from seed-42 elapsed):

| arch | per-seed | × 3 seeds |
|---|---|---|
| mlc | ~11 min | 33 min |
| time_layer_crosscoder_t5 | ~40 min | 2h |
| txcdr_rank_k_dec_t5 | ~34 min | 1h42m |
| txcdr_t5 | ~15 min | 46 min |
| txcdr_tied_t5 | ~14 min | 42 min |
| **total training** | | **~5h40m** |

Then probe each new ckpt at last_position only: 15 × ~7 min = ~1h45m.

Grand total: **~7h30m GPU**. Should run unattended.

After completion: extend `plots/make_headline_plot.py` (or a new script) to compute mean ± std across seeds for each arch, and re-render the last_position × AUC headline bar chart with error bars. Update summary.md to replace the "single seed (42) on every row" caveat.

#### Orchestrator plan (T13–T17)

Write `experiments/phase5_downstream_utility/run_overnight_phase5.sh`:

```
1. Wait for mlc_contrastive orchestrator to finish (pgrep loop).
2. Run T13 (plot/summary patch) — no GPU.
3. Run T14 (code patch) — no GPU.
4. Run T15 (re-probe 7 archs) — GPU.
5. Run T16 (error-overlap analysis) — CPU.
6. Run T17 (3-seed training + probing) — GPU.
7. Regenerate all plots with seed variance + updated aggregations.
8. Commit + push.
```

The orchestrator should emit `logs/overnight/overnight_phase5_orchestrator.log` following the pattern of the existing three.

#### T6 (still pending, deferred)

`tfa_big` (full d_sae=18432, seq_len=128) — estimated ~50–80 min training. Low priority relative to T13–T17; can slot in after T17 if compute permits.

#### Phase 6 material (don't start in Phase 5)

- T-SAE paper latent comparison: qualitative comparison of `temporal_contrastive__seed42` latents against Ye et al.'s engineered dataset. User explicitly said "leave this for next phase".

### Pointers for the next agent

**Code:**
- `src/architectures/mlc_contrastive.py` — new today.
- `experiments/phase5_downstream_utility/train_primary_archs.py` — extended with `txcdr_t{2,3,8,10,15}`, `mlc_contrastive`, `make_pair_multilayer_gen_gpu`.
- `experiments/phase5_downstream_utility/probing/run_probing.py` — extended with `mean_pool` aggregation, `mlc_contrastive` loader, streaming patch for time_layer full_window.
- `experiments/phase5_downstream_utility/probing/build_probe_cache.py` — `TAIL_MLC_N` bumped 5→20.
- `experiments/phase5_downstream_utility/plots/make_headline_plot.py` — iterates over `("last_position", "full_window", "mean_pool")`; extended ORDERED_ARCHS with `mlc_contrastive`.
- `experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep.py` — new today; produces `txcdr_t_sweep_{auc,acc}.png` for the 7-point T sweep.
- `experiments/phase5_downstream_utility/run_{fw_tsweep,mean_pool_probing,mlc_contrastive}.sh` — the three live orchestrators.

**Data:**
- `experiments/phase5_downstream_utility/results/probing_results.jsonl` — single source of truth for probe AUC/acc per (run, task, aggregation, k).
- `experiments/phase5_downstream_utility/results/training_index.jsonl` — one row per trained checkpoint.
- `experiments/phase5_downstream_utility/results/ckpts/*.pt` — fp16 state dicts, gitignored.
- `experiments/phase5_downstream_utility/results/probe_cache/<task>/` — activation caches, gitignored.

**Docs:**
- `docs/han/research_logs/phase5_downstream_utility/brief.md` — pre-phase context.
- `docs/han/research_logs/phase5_downstream_utility/plan.md` — pre-registered experimental plan.
- `docs/han/research_logs/phase5_downstream_utility/summary.md` — end-of-phase synthesis (updated today with 36/36 baselines + MLC/time_layer full_window rows).
- **This file** — overnight handoff.

### Checklist for the next agent on resume

1. `tail /workspace/temp_xc/logs/overnight/{fw_tsweep,mean_pool,mlc_contrastive}_orchestrator.log` to confirm all three live orchestrators finished cleanly.
2. Confirm `results/plots/headline_bar_k5_mean_pool_auc_full.png` exists (mean_pool orchestrator output).
3. Confirm `results/ckpts/mlc_contrastive__seed42.pt` exists (mlc_contrastive orchestrator output).
4. Confirm `results/probing_results.jsonl` has rows for `mlc_contrastive__seed42` at both `last_position` and `mean_pool`.
5. Write and launch `run_overnight_phase5.sh` for tasks T13–T17.
6. When T15 + T16 finish (first few hours), post collaborator-feedback results to `summary.md`.
7. When T17 finishes (~morning), update headline table with seed variance and commit.
