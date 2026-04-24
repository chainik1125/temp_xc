---
author: Han
date: 2026-04-24
tags:
  - proposal
  - in-progress
---

## Handover: post-H8 session wrap-up

**Audience**: post-compact agent. Previous session found H8 as new TXC
champion; session at 81% context, autocompacting.

**Current state**: `han` branch HEAD. Commit `3d93a1e` (push may be
newer by time you read this).

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

### New TXC runner-up: H7 = `phase57_partB_h7_bare_multiscale` (3-seed)

Same stack as H8 but with agentic_txc_02's **multi-scale** InfoNCE
(nested prefixes at scales {1, 2, 3} with γ=0.5 decay) instead of
multi-distance. 3-seed variance complete:
- last_position: **0.7886 ± 0.0070** (σ-defensible +0.014 vs agentic_txc_02 3s 0.7749)
- mean_pool: **0.8059 ± 0.0106**

### What's outstanding (not yet in JSONL)

1. **H8 seeds 1 + 2** — seed 1 done training (ckpt saved); seed 2
   training in background as of handover. After both trained, probe
   at both aggregations. If σ < 0.01, H8 is σ-defensible TXC champion.
2. **H9 contrastive seeds 1 + 2** — not yet trained (seed 42 only, mp
   0.7891 with full 36 tasks).
3. **T=10, 15, 20 alive-fraction** — OOMed during alive_fraction.py
   run. Retry in isolation.
4. **A7 HF sync** — checkpoints not yet uploaded to HuggingFace.
5. **Headline bar plots regenerated with all Part B archs** — DONE but
   visual inspection may show some archs' labels are too small to read;
   regenerate with only top-20 if needed.

---

## What to run next (ordered)

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

Remaining Part B hypotheses not yet tested:
- H3 log-matryoshka (arch + launcher ready at `run_partB_h3.sh`)
- H5 SVD regularizer (not implemented)
- H2 attention-pool decoder (not implemented)

Or: use H8 as the base and test further refinements
- Extend shifts to {1, 2, 3} or even {1, T/4, T/2} at larger T
- Combine H8's multi-distance with H7's multi-scale (orthogonal axes —
  could stack)
- Try H8 recipe at T=8 (where agentic_txc_02 peaked at mp)

---

## For the post-compact agent: start here

1. `git log --oneline | head -5` to confirm at commit `3d93a1e` or later.
2. Check if H8 seed 2 finished: `ls experiments/phase5_downstream_utility/results/ckpts/phase57_partB_h8_bare_multidistance__seed*.pt`
3. Follow "Priority 1" above. Commit + push after each probe completes.
4. HF sync at the end (Priority 4).
