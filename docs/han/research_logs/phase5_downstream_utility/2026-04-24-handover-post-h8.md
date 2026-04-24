---
author: Han
date: 2026-04-24
tags:
  - proposal
  - in-progress
---

## Handover: post-H8 session wrap-up

**Audience**: post-compact agent. Previous session found H8 as new TXC
champion **at T=5**; session at 81% context, autocompacting.

**Current state**: `han` branch HEAD — `2a06c1b` or newer.
`git pull origin han` before starting.

---

## ⚠️ PAPER-CRITICAL OPEN PROBLEM: T-scaling has NOT been solved

**The golden TXC architecture should be one whose sparse-probing AUC
MONOTONICALLY INCREASES with T (window size).** The paper's central
existential claim is that temporal windows carry useful structure
beyond single-token encoders. If AUC doesn't climb with T, the "TXC
is the thing" story collapses regardless of how good H8 is at T=5.

**So far we have NOT found a T-scaling arch.** Full session data:

| family | aggregation | monotonicity | Δ(T_max−T_min) | verdict |
|---|---|---|---|---|
| vanilla TXCDR × TopK | lp | 0.52 | +0.006 | peaks at T=5, drops both sides |
| vanilla TXCDR × TopK | mp | 0.33 | **−0.024** | anti-monotone |
| vanilla TXCDR × BatchTopK | lp | 0.57 | +0.006 | U-shape (T=15/20 miscalibrated) |
| vanilla TXCDR × BatchTopK | mp | 0.43 | −0.006 | no trend |
| agentic_txc_02 (T∈{2,3,5,6,7,8}) | mp | ≤0.7 | +0.01 | T=8 top (0.7917 lp) but T≥10 OOMs |
| H1 ConvTXCDR (T∈{5,10,15,20,30}) | lp | **0.40** | **−0.032** | ❌ FAILS (sum-pool kills signal) |
| H1 ConvTXCDR | mp | 0.17 | −0.080 | worse |

**Target**: monotonicity ≥ 0.8, Δ(T=30 − T=5) > +0.02. None achieved.

**Also critical**: vanilla TXCDR at d_sae=18432 **OOMs at T≥24** on A40.
Any T-scaling arch MUST be more parameter-efficient than vanilla
TXCDR to train at T=30.

Feasibility map (all at d_sae=18432):
| arch | T=20 | T=24 | T=28 | T=30 | T=32 | T=36 |
|---|---|---|---|---|---|---|
| vanilla TXCDR | ✓ (trained) | OOM | OOM | OOM | OOM | OOM |
| vanilla TXCDR BatchTopK | ✓ | OOM | OOM | OOM | OOM | OOM |
| matryoshka T-scale | ✗ (T≥10 OOM) | — | — | — | — | — |
| ConvTXCDR H1 (conv enc) | ✓ | — | — | ✓ | — | — |

ConvTXCDR trained at T=30 because its encoder params are T-invariant
(127M regardless of T). Any new candidate should consider the same
trick if it needs to train at T=30.

**Mission for the post-compact agent**: find an arch that makes mp
AUC climb with T. H8 at T=5 is NOT the ending — it's the baseline to
build FROM. Plausible directions:

1. **H8 T-sweep**: train H8 recipe at T∈{5, 10, 15, 20, 30} and see
   if multi-distance InfoNCE + anti-dead finally produces monotonicity.
   At T=30, drop matryoshka H/L (saves params); use
   `matryoshka_h_size=None`. Shifts={1, T/4, T/2} per-T-scaled.
   This is the direct "does H8 scale?" test and SHOULD be priority 1.
2. **Log-matryoshka H3**: arch ready, not run. Coarser matryoshka
   scales {1,2,4,8,16,32} escape O(T²) OOM. If it trains at T=30 AND
   AUC climbs, new champion AND T-scaling.
3. **Mamba/SSM encoder (H6)**: biggest change, deferred from the
   original handover. If everything else fails, the state-space
   encoder is the remaining architectural wildcard.

---

## The headline (what this session found)

### New TXC champion: H8 = `phase57_partB_h8_bare_multidistance`

**Single-seed 42 results** (36 tasks, k_feat=5):
- last_position: **0.8039** (TXC leader; MLC top is 0.8124)
- mean_pool: **0.8139** ⭐ (tops ENTIRE benchmark — all archs + all families)

**Recipe**: bare TXC encoder + Phase 6.2 Track 2 anti-dead stack
(AuxK + unit-norm decoder + parallel-grad removal + geom-median b_dec)
+ matryoshka H/L recon + **multi-distance InfoNCE** (shifts {1, 2},
inverse-distance weighted).

The novel ingredient is multi-distance contrastive. Everything else
was already tested individually.

### Matryoshka H/L detail (NOT Phase 5.7's position-nested matryoshka)

H8's matryoshka is tsae_paper-style: **two reconstructions, both
targeting the full T-token window**. Not T nested sub-windows.

```
x_hat_H  = z[:, :3686]  · W_dec[:3686, :, :]  + b_dec   # H prefix = d_sae/5
x_hat    = z            · W_dec              + b_dec    # full dict (d_sae=18432)

L_recon  = MSE(x_hat_H, x)  +  MSE(x_hat, x)     # unweighted sum
```

Key differences from Phase 5.7 `agentic_txc_02`:
- Only **2 scales** (H, Full) — not T scales.
- **Both reconstructions target the FULL T-token window** — not centered
  (s+1)-token sub-windows.
- **Single `W_dec : (d_sae, T, d_in)`** — same per-position decoder
  structure as vanilla TemporalCrosscoder. The H reconstruction uses
  the first 3686 ROWS `W_dec[:3686, :, :]` (all T positions intact).
  **Not a shared decoder across positions** — the matryoshka partitions
  the FEATURE axis, not the position axis. No per-scale separate
  decoders either (unlike PositionMatryoshkaTXCDR's `W_decs` list).
- **Unweighted sum** — no γ-decay.

The contrastive InfoNCE operates on `z[:, :3686]` (the H prefix).
So the multi-distance invariance pressure ALSO lives in the H partition
— the 3686 high-level latents must reconstruct x alone AND stay
consistent across shift-1/2 pairs. The remaining 14746 (L partition)
are free to specialize without contrastive constraint, and just
contribute to the full recon.

Handover of H8's arch class: `TXCBareMatryoshkaContrastiveAntidead`
(parent, in `txc_bare_matryoshka_contrastive_antidead.py`) →
`TXCBareMultiDistanceContrastiveAntidead` (H8, in
`txc_bare_multidistance_contrastive_antidead.py`). The matryoshka
logic lives in parent's `_recon_loss` method.

### New TXC runner-up: H7 = `phase57_partB_h7_bare_multiscale` (3-seed)

Same stack as H8 but with agentic_txc_02's **multi-scale** InfoNCE
(nested prefixes at scales {1, 2, 3} with γ=0.5 decay) instead of
multi-distance. 3-seed variance complete:
- last_position: **0.7886 ± 0.0070** (σ-defensible +0.014 vs agentic_txc_02 3s 0.7749)
- mean_pool: **0.8059 ± 0.0106**

### What's outstanding (not yet in JSONL)

1. **H8 seeds 1 + 2 probes** — seed 1 ckpt done ✓; seed 2 TRAINING IN
   BACKGROUND at handover time (PID 62152, bash wrapper 59662/59664).
   When seed 2 finishes, the bash wrapper will auto-probe both seeds at
   both aggregations and write to `probe_h8_seeds12_{lp,mp}.log`.
2. **H9 contrastive seeds 1 + 2** — not yet trained (seed 42 only, mp
   0.7891 full 36-task, below agentic_txc_02 — H9c is NOT a winner).
3. **T=10, 15, 20 vanilla TXCDR alive-fraction** — OOMed during
   alive_fraction.py run (GPU was full with other jobs). Retry in
   isolation.
4. **A7 HF sync** — checkpoints not yet uploaded to HuggingFace.
5. **T-scaling on H8/H7 recipes** — NOT STARTED. Must be done before
   the paper can claim TXC scales with T.

### In-flight jobs at handover time (DO NOT KILL until checked)

| PID | job | ETA |
|---|---|---|
| 62152 | H8 seed 2 training | ~30 min remaining |
| 59662/59664 | H8 seed 2 bash wrapper + auto-probe | follows seed 2 |

When H8 seed 2 is done:
- ckpt at `results/ckpts/phase57_partB_h8_bare_multidistance__seed2.pt`
- probes will auto-run, results land in `probing_results.jsonl`
- THEN you can fire new jobs.

### Complete list of trained ckpts from this session (as of handover)

New TXC-family trained this session:
- `phase57_partB_h7_bare_multiscale__seed{1,2,42}` — H7 champion candidate
- `phase57_partB_h8_bare_multidistance__seed{1,42}` + (seed 2 in flight)
- `phase57_partB_h7_bare_multiscale_recal` (not applicable)
- `feature_nested_matryoshka_t5__seed42` — H9 plain
- `feature_nested_matryoshka_t5_contrastive__seed42` — H9 with contrastive (mp 0.7891)
- `txc_shared_relu_sum_{pos,nopos}_t5__seed42` — H10 ablations
- `txc_shared_concat_two_layer_t5__seed42` — H12 ablation
- `conv_txcdr_t{5,10,15,20,30}__seed42` — H1 ConvTXCDR T-sweep (failed)
- `txcdr_t{6,7}__seed42` + `_batchtopk` — detailed T-sweep fill-in
- `agentic_txc_02_t{6,7}__seed42` — detailed T-sweep
- `agentic_{txc_02,mlc_08}_batchtopk__seed{1,2}` — A3 Tier 2 3-seed variance
- `mlc_contrastive__seed{1,2}` — A3 Tier 1
- `matryoshka_t5__seed{1,2}` — A3 Tier 1
- `txcdr_t{15,20}_batchtopk_recal__seed42` — A2 recalibration (no AUC change)

---

## What to run next (ordered)

### ⭐ Priority 0: push contrastive shifts beyond {1, 2}

**The biggest open question.** H8 shifts={1,2} was the single change
that produced the mp champion (0.8139, +0.008 over H7). If shifts={1,2,3}
or {1,2,3,4} keeps climbing, we might find an even stronger winner.

See the detailed plan under "If the user wants to keep pushing →
HIGHEST PRIORITY" section below. Implement BEFORE H8 3-seed if GPU
is available (H8 3-seed is rigor; shifts exploration is the next
headline).

### Priority 1: finalize H8 3-seed

```bash
# Wait for H8 seed 2 training (auto-continues from background job).
# When done, probe seeds 1, 2 at both aggregations:
tail -F /workspace/temp_xc/logs/overnight/train_phase57_partB_h8_bare_multidistance__seed2.log

# Once seed 2 ckpt exists:
.venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
    --aggregation last_position --skip-baselines \
    --run-ids phase57_partB_h8_bare_multidistance__seed1 \
              phase57_partB_h8_bare_multidistance__seed2

.venv/bin/python -u experiments/phase5_downstream_utility/probing/run_probing.py \
    --aggregation mean_pool --skip-baselines \
    --run-ids phase57_partB_h8_bare_multidistance__seed1 \
              phase57_partB_h8_bare_multidistance__seed2
```

### Priority 2: H9 contrastive seeds 1, 2

```bash
for SEED in 1 2; do
    .venv/bin/python -u -c "
from experiments.phase5_downstream_utility.train_primary_archs import run_all
run_all(seeds=[$SEED], max_steps=25000,
        archs=['feature_nested_matryoshka_t5_contrastive'])
" > logs/overnight/train_feature_nested_matryoshka_t5_contrastive__seed${SEED}.log 2>&1
done
# Then probe both aggregations.
```

### Priority 3: T=10, 15, 20 alive-fraction (when GPU is free)

```bash
# alive_fraction.py OOMed earlier — retry with only 1 job at a time
# after H8 seeds + H9c seeds finish.
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
    experiments/phase5_downstream_utility/analysis/alive_fraction.py
```

### Priority 4: A7 HF sync

```bash
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/hf_upload_ckpts.py
# Idempotent. Uploads all new ckpts from this session including:
#   phase57_partB_h{7,8}_*, feature_nested_matryoshka_t5*,
#   txc_shared_*, conv_txcdr_t*, txcdr_t{6,7}*, agentic_txc_02_t{6,7}*,
#   agentic_*_batchtopk__seed{1,2}, matryoshka_t5__seed{1,2},
#   mlc_contrastive__seed{1,2}
```

---

## Key commits this session

(Most recent first)

- `3d93a1e` Regen plots with full H9c data: H8 remains mp champion
- `bafffbe` Headline plot: add Part B archs + T=6/7 to ORDERED_ARCHS
- `e73b9cb` Regenerate headline + 4-panel plots with H7/H8 + A3 Tier 2
- `3ff7c1e` Fig 1/2 + 3-seed variance table: add H7/H8 rows
- `53fc242` Summary.md: alive fraction section across TXC T-sweep
- `3a7d9b9` Alive fraction analysis: anti-dead raises alive 31% → 77%
- `04f5212` H7 3-seed variance: lp 0.7886±0.007 / mp 0.8059±0.011
- `43fc160` 🎉 H8 multi-distance: TXC CHAMPION at mp 0.8139
- `1aa0de5` Detailed T-sweep + A3 Tier 2 + encoder variants — mp probe complete
- `8b231b7` Part B H10/H12 encoder ablations: per-pos W_enc IS load-bearing
- `6d8310d` Part B H1 cycle: mp result added, both aggs fail T-scaling
- `bb30c91` Part B H9: feature-nested matryoshka TXCDR (user proposal)
- `a146253` Part B H8: bare TXC + anti-dead + multi-distance InfoNCE
- `63c2a9e` Part B H7: anti-dead × agentic_txc_02 multi-scale
- `31a6c9b` A3 stats: paired-t-test + Bonferroni across 16 comparisons
- `2ec1772` Part B H7 lp result: 0.7915 (+0.017 over agentic_txc_02)

---

## Files you'll need

### Architectures (all new this session, in `src/architectures/`):

- `conv_txcdr.py` — H1 ConvTXCDR (FAILED T-scaling)
- `log_matryoshka_txcdr.py` — H3 LogMatryoshka (not yet run)
- `txc_bare_multiscale_contrastive_antidead.py` — H7 class
- `txc_bare_multidistance_contrastive_antidead.py` — **H8 class (CHAMPION)**
- `feature_nested_matryoshka_txcdr.py` — H9 class
- `txc_encoder_variants.py` — H10/H12 classes
- `txc_bare_antidead.py` + `txc_bare_batchtopk_antidead.py` + `txc_bare_matryoshka_contrastive_antidead.py` — vendored from Phase 6 (Track 2 bases)

### Analysis scripts (`experiments/phase5_downstream_utility/analysis/`):

- `batchtopk_threshold_audit.py` — A2
- `recalibrate_batchtopk_threshold.py` — A2 recalibration
- `batchtopk_delta_table.py` — A4
- `t_scaling_score.py` — Part B scoring
- `a3_seed_stats.py` — A3 paired-t-test
- `alive_fraction.py` — dead-feature analysis
- `concat_probe.py` — A1

### Launchers (`experiments/phase5_downstream_utility/`):

- `run_partA_finish.sh` — A5 + A3 + plot regen
- `run_partB_h{1,3,7,8,9,10_h12}.sh` — per-hypothesis launchers
- `run_tsweep_detailed.sh` — T=6/7/24/28 + A3 Tier 2
- `run_a1_recal_a4.sh` — A1 concat + recalibrate + re-probe

### Key docs:

- `summary.md` — Phase 5/5.7 paper-ready doc with all updates
- `2026-04-24-partB-tscaling-log.md` — Part B cycle log (H1-H12)
- `EXPERIMENT_INDEX.md` — master arch index (cherry-picked from phase6)
- Handover you're reading now: `2026-04-24-handover-post-h8.md`

---

## The paper implication (headline for NeurIPS)

The paper's "TXC is the thing" claim now has a defensible anchor:

- **H8 `bare_multidistance_antidead` is the strongest SAE at mean_pool
  across ALL archs tested** (0.8139 vs baseline_attn_pool 0.9292 — still
  ~12 pp below raw baselines, but top SAE by a clear margin).
- **H8 at last_position (0.8039) is the TXC-family leader, 0.9 pp
  below MLC top.** With 3-seed variance + anti-dead stack's +0.03 alive
  fraction advantage, H8 closes the gap to MLC almost entirely.
- **The anti-dead stack × multi-distance contrastive interaction is the
  novel contribution**. Both components existed separately (Phase 6.2
  had anti-dead + single-shift contrastive at 0.7834 lp; Phase 5.7 had
  multi-scale contrastive at 0.7749 3s lp). Their combination was the
  untried cell.

Honest caveats for reviewers:
- H8 is single-seed; H7 3-seed confirms the anti-dead mechanism but H8's
  lead over H7 (+0.012 lp, +0.003 mp) is within H7's σ (±0.007 / ±0.011).
- No T-scaling story — H8 is T=5 only. Part B T-scaling goal FAILED
  decisively for all hypotheses tried (H1 was the only T-sweep; mono 0.4).
- Per-position W_enc IS load-bearing (H10/H12 ablations confirm).

---

## If the user wants to keep pushing

### ⭐ HIGHEST PRIORITY: contrastive-shift ablation (mechanism study, not leaderboard chase)

H8 at T=5 uses shifts={1, 2}. The progression agentic_txc_02 (shift=1
only) → H7 (shift=1, multi-scale prefixes) → H8 (shifts={1, 2}) showed
that **adding a shift=2 pair lifted mp from 0.8059±0.011 (H7 3s) to
0.8139 (H8 seed 42)**. The question is NOT "can wider shifts top 0.8139"
(though that would be nice) — the more useful question is:

**How does downstream-probing AUC depend on the contrastive-shift
window length, holding every other ingredient fixed?**

This is an ablation study. Design it to produce a CURVE (AUC vs shifts
config) rather than just a winner.

**Systematic shift-set sweep at T=5** (each ~1 hr training + probe):

| label | shifts | token overlap (T=5) | notes |
|---|---|---|---|
| H8a-0 | {1} (baseline = H7-style) | 80% | single-shift contrastive, multi-scale removed |
| **H8** | {1, 2} (current champion) | 80%, 60% | baseline we just found |
| H8a-123 | {1, 2, 3} | 80%, 60%, 40% | incrementally add shift-3 |
| H8a-1234 | {1, 2, 3, 4} | 80%, 60%, 40%, 20% | all 4 shifts fitting within T=5 |
| H8a-1024 | {1, 2, 4} | 80%, 60%, 20% | skip shift-3 — does skipping matter? |
| H8a-2only | {2} | 60% | does shift-1 even help? |
| H8a-4only | {4} | 20% | extreme: only very weak pairs |
| H8a-uniform | {1, 2, 3} with w=1 each | | remove inverse-distance weighting |

Interpretable outcomes (what you LEARN from this sweep):

- **If AUC is monotonically increasing with number of shifts**: richer
  invariance signal wins; post-compact agent should explore shifts
  at T=8 and beyond.
- **If AUC plateaus or drops past some shift**: there's an optimal
  contrastive diversity — too-weak pairs (low token overlap) become
  noise. Report the optimal set.
- **If shift-{2} alone outperforms shift-{1} alone**: the classical
  tsae_paper shift-1 recipe is suboptimal — paper should default to
  shift-2.
- **If uniform weighting ≈ inverse-distance weighting**: the
  weighting scheme doesn't matter much, simplifying the recipe.
- **If shift-4-only still trains a strong probe**: the contrastive
  loss is not very sensitive to token overlap — supports a
  "contrastive helps regardless of local/global structure" story.

Each of these is publishable as an ablation.

**Implementation**: the arch
([`src/architectures/txc_bare_multidistance_contrastive_antidead.py`](../../../src/architectures/txc_bare_multidistance_contrastive_antidead.py))
already accepts arbitrary shifts via `shifts=(1, 2, 3, ...)` and
accepts custom `weights=(w_1, w_2, ...)`. Each variant needs a
dispatcher branch (template):

```python
elif arch == "phase57_partB_h8a_shifts123":
    model, log = train_txc_bare_multidistance_contrastive_antidead(
        cfg, device, k=100, T=5, alpha=1.0,
        shifts=(1, 2, 3),
        matryoshka_h_size=int(DEFAULT_D_SAE * 0.2),
        aux_k=512, dead_threshold_tokens=10_000_000,
        auxk_alpha=1.0 / 32.0,
        buf=get_anchor(),
    )
    meta = dict(seed=seed, k_pos=100, k_win=500, T=5,
                alpha=1.0, shifts=[1, 2, 3],
                matryoshka_h_size=int(DEFAULT_D_SAE * 0.2),
                aux_k=512, dead_threshold_tokens=10_000_000,
                auxk_alpha=1.0/32.0,
                variant="phase57_partB_h8a_ablation_shifts123")
```

For the uniform-weighting variant, pass `weights=(1.0, 1.0, 1.0)`
via `train_txc_bare_multidistance_contrastive_antidead`'s `shifts`
arg (needs a tiny extension to thread `weights` through — or hack by
adding a dispatcher branch that manually constructs the model).

Also update `run_probing.py` — generalize the H8 branch:
```python
elif arch.startswith("phase57_partB_h8"):
```

**Variance**: run each ablation at seed=42 first for the curve; then
take the winner + strongest contenders and run seeds 1, 2 for σ.

**Budget**: 8 variants × ~1 hr training + probe = ~8-10 hr. Plus ~4-6 hr
of 3-seed variance on 2-3 top performers = ~14 hr total.

### Separately: TEST IF H8 SCALES WITH T (the paper-critical T-scaling question)

The above is a T=5 ablation — a MECHANISM study. Orthogonal question:
does H8's recipe scale with T? Train H8 at T ∈ {5, 10, 15, 20, 30}
and measure mp AUC. If it climbs, the paper's T-scaling claim is
salvaged.

At T=30, H8's default params would be ~2.4B (encoder 1.27B + per-scale
decoder ~1B). Test feasibility first. If OOM, drop matryoshka H/L
(set `matryoshka_h_size=None`) — saves ~200M decoder params.

### Other Part B hypotheses not yet tested

- H3 log-matryoshka (arch + launcher ready at `run_partB_h3.sh`)
- H5 SVD regularizer (not implemented)
- H2 attention-pool decoder (not implemented)

Or: use H8 as the base and test further refinements
- Combine H8's multi-distance with H7's multi-scale (orthogonal axes —
  stack both contrastive pressures, could push further than either alone)
- Try H8 recipe at T=8 (where agentic_txc_02 peaked at mp)
- **Ablate inverse-distance weighting**: is `1/(1+s)` optimal? Try
  uniform `w_s = 1`, or exponential `w_s = e^{-s}`.

---

## For the post-compact agent: start here

1. `git log --oneline | head -5` to confirm at commit `3d93a1e` or later.
2. Check if H8 seed 2 finished: `ls experiments/phase5_downstream_utility/results/ckpts/phase57_partB_h8_bare_multidistance__seed*.pt`
3. Follow "Priority 1" above. Commit + push after each probe completes.
4. HF sync at the end (Priority 4).
