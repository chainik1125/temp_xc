---
author: Han
date: 2026-04-29
tags:
  - design
  - todo
---

## Y → Z handoff (post-Q1.3 hill-climb directions)

> Agent Y → Agent Z. Z owns hill-climbing for `barebones-TXC variant
> that beats current leaderboard`. Y's Q1.3 + sparsity-decomposition
> findings change what "winner" should mean.

### Key finding from Y's shift

T-SAE k=20's apparent steering advantage (peak success 1.80 vs TXC
matryoshka's 1.07) is **dominantly driven by sparsity, not architecture
family**. At matched effective sparsity (k_eff ≈ 100-500 across
families), cross-family peak-success spread is 0.27 — within concept
noise on a 30-concept set.

See `2026-04-29-y-tx-steering-magnitude.md` and
`results/case_studies/plots/phase7_steering_v2_sparsity_decomp.png`.

### What this changes for hill-climb

**Direct hypothesis test for Z**: train a TXC variant at `k_pos=20`
(matching T-SAE k=20 sparsity) and run paper-clamp + AxBench on it.
If a TXC at k_pos=20 matches T-SAE k=20's peak success (~1.80), the
"sparsity is the dominant driver" hypothesis is fully confirmed and
the paper has a clean reversal: **at matched sparsity, TXC family is
competitive with T-SAE k=20**.

This should be priority-1 for Z's hill-climb if probe-AUC isn't yet
the bottleneck.

### Concrete spec

Train a barebones TXC variant on Gemma-2-2b base, L12, with:

| param | value | notes |
|---|---|---|
| arch_id (suggested) | `txc_bare_antidead_t5_kpos20` | follow Phase 7 naming |
| src_class | `TXCBareAntidead` | barebones, no contrastive |
| T | 5 | matches agentic_txc_02 / h8_multidist |
| k_pos | **20** | the new sparsity target — MATCHES T-SAE k=20 |
| k_win | 100 (= k_pos × T) | window k = 5 × per-position k |
| d_sae | 18432 | paper-wide |
| training_constants | from `paper_archs.json` | batch_size 4096, lr 3e-4, max_steps 25k |
| seed | 42 | primary |

Compute: A40-feasible (smaller than agentic_txc_02 which is 1.3 GB).
Runtime estimate: 40-60 min on A40.

### Validation pipeline

After training:

```bash
# 1. Pull ckpt to local (or already on HF after train_phase7 upload)
TQDM_DISABLE=1 .venv/bin/python -m \
  experiments.phase7_unification.case_studies._download_ckpts txc_bare_antidead_t5_kpos20

# 2. Compute z magnitudes (Q1.1) — should be SMALLER than k_pos=100
TQDM_DISABLE=1 .venv/bin/python -m \
  experiments.phase7_unification.case_studies.steering.diagnose_z_magnitudes \
  --archs txc_bare_antidead_t5_kpos20

# 3. Pick steering features
TQDM_DISABLE=1 .venv/bin/python -m \
  experiments.phase7_unification.case_studies.steering.select_features \
  --archs txc_bare_antidead_t5_kpos20

# 4. Run normalised paper-clamp
TQDM_DISABLE=1 .venv/bin/python -m \
  experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_normalised \
  --archs txc_bare_antidead_t5_kpos20

# 5. Grade
.venv/bin/python -m \
  experiments.phase7_unification.case_studies.steering.grade_with_sonnet \
  --archs txc_bare_antidead_t5_kpos20 --subdir steering_paper_normalised \
  --n-workers 2

# 6. Plot vs other archs
.venv/bin/python -m \
  experiments.phase7_unification.case_studies.steering.plot_sparsity_decomposition
```

### Predicted outcome

Under H1 ("sparsity is the dominant driver"):
- TXC at k_pos=20 should peak at success ≈ 1.6-1.8 under normalised
  paper-clamp.
- This would close the headline gap entirely.
- **Paper headline reversal**: "At matched sparsity, TXC family is
  competitive with — possibly exceeds — T-SAE k=20."

Under H2 ("architecture matters at low k"):
- TXC at k_pos=20 still peaks at <1.4.
- Then there's a true architecture-family difference at low k.

Either outcome is informative.

### Warning on probe-AUC trade-off

Sparser TXC may TRAIL on sparse-probing AUC (which Z's hill-climb
optimises for). If a k_pos=20 TXC has ~0.85 AUC vs the leaderboard's
0.94, we're trading off probe-AUC for steering. The right paper
decision depends on which downstream task matters more.

If probe-AUC is the gold standard for the leaderboard plot, train the
sparse-TXC variant SEPARATELY for case-study purposes only — it's a
dedicated experiment, not a leaderboard contender.

### Cross-link to my findings

- Q1.1+Q1.2: `2026-04-29-y-tx-steering-magnitude.md`
- Q1.3 + sparsity decomposition: same file, "Refined narrative" section
- Per-concept TXC-favourable pattern (B.6): `2026-04-29-y-cs-synthesis.md`
- Headline plot: `phase7_steering_v2_sparsity_decomp.png`
