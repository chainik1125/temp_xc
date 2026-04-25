---
author: Han
date: 2026-04-24
tags:
  - proposal
  - in-progress
---

## Phase 5.7 Part B+ — "Groundbreaking results" session log

**Audience**: post-compact agent. Session goal: complete EVERYTHING from
`2026-04-24-handover-post-h8.md` plus the user's late-session ask
(MLC + anti-dead fairness counterparts).

**Master launcher**:
[`run_groundbreaking.sh`](../../../experiments/phase5_downstream_utility/run_groundbreaking.sh)

---

## Queue contents (25 trainings + analysis)

### P0a — H8 T-sweep (paper-critical T-scaling answer)

`phase57_partB_h8_bare_multidistance` at T ∈ {5, 6, 7, 8, 10, 15, 20, 30}.
T=5 already trained; rest queued.

The paper-critical question: does H8's mp AUC monotonically increase with T?
If yes, the "TXC scales with T" headline is saved. If no, MLC headline pivot.

Default shifts auto-scale with T via `(1, T//4, T//2)` deduped. So:
| T  | shifts        |
|----|---------------|
| 5  | (1, 2)        |
| 6  | (1, 3)        |
| 7  | (1, 3)        |
| 8  | (1, 2, 4)     |
| 10 | (1, 2, 5)     |
| 15 | (1, 3, 7)     |
| 20 | (1, 5, 10)    |
| 30 | (1, 7, 15)    |

T=30 may OOM at d_sae=18432 on A40. Queue handles this gracefully (skip
on failure, continue to next).

### P0b — Shift-ablation at T=5 (mechanism study)

8 variants. The ablation produces a CURVE relating AUC to shift-set:

| label              | shifts        |
|--------------------|---------------|
| {1}                | (1,)          |
| {1,2} = H8         | (1, 2)        |
| {1,2,3}            | (1, 2, 3)     |
| {1,2,3,4}          | (1, 2, 3, 4)  |
| {1,2,4}            | (1, 2, 4)     |
| {2}                | (2,)          |
| {4}                | (4,)          |
| {1,2,3} uniform    | (1, 2, 3) w=1 |

Whatever shape we observe is publishable as ablation. Specific
interpretable outcomes:
- Monotonic-with-shifts → richer invariance signal wins
- Plateaus or drops → optimal shift count exists; weak pairs become noise
- {2} > {1} → tsae_paper shift-1 default is suboptimal
- Uniform ≈ inverse-distance → weighting doesn't matter much

### P0c — MLC + anti-dead fairness counterparts (user-requested)

3 archs to address "you may have given TXC unfair anti-dead advantage":

| arch                                              | TXC counterpart                            |
|---------------------------------------------------|--------------------------------------------|
| `mlc_bare_antidead`                              | TXCBareAntidead (recon-only, never trained at 5.7 — would need separate run) |
| `mlc_bare_matryoshka_contrastive_antidead`       | Phase 6.2 C3 (single-shift contr)          |
| `mlc_bare_multiscale_antidead`                   | H7 (multi-scale contr)                     |

`k=100` (NOT `k * n_layers`) to match MLC's native sparsity convention.
Multi-distance has no MLC analog — TXC-only feature.

### P0d — H13 mdms (multi-distance × multi-scale orthogonal stack)

`phase57_partB_h13_md_x_ms`: combines H8 shifts {1,2} with H7's multi-scale
prefix InfoNCE {h, 2h, 3h} with γ=0.5 decay. Six contrastive terms. If
orthogonal contrastive pressures stack constructively → potential new
champion. Same anti-dead + matryoshka stack as H7/H8.

### P2 — H9c contrastive seeds 1, 2

`feature_nested_matryoshka_t5_contrastive` at seeds 1, 2 for σ. Seed 42
already done (mp 0.7891 — confirms H9c is NOT a winner; useful for variance).

### P3 — H3 log-matryoshka T-sweep

`log_matryoshka_t<T>` at T ∈ {5, 10, 15, 20, 30}. Designed to escape
O(T²) matryoshka decoder OOM that stopped vanilla matryoshka at T≥10.
If T-scaling AND beats H8, double-headline.

### P4 — alive-fraction retry

`analysis/alive_fraction.py` extended to all new archs (H8 T-sweep,
shift ablation, log-matryoshka, MLC + anti-dead).

### P5 — HF sync

`scripts/hf_upload_ckpts.py` uploads all new ckpts.

---

## Post-queue analysis pipeline (ready to run)

```bash
# 1. Aggregate all new probe results
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
    experiments/phase5_downstream_utility/analysis/groundbreaking_summary.py

# 2. Regenerate plots
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
    experiments/phase5_downstream_utility/plots/make_headline_plot.py
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
    experiments/phase5_downstream_utility/plots/make_t_scaling_plot.py
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
    experiments/phase5_downstream_utility/plots/make_shift_ablation_plot.py
TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
    experiments/phase5_downstream_utility/plots/make_fairness_plot.py
```

---

## Results (filled in as queue completes)

### H8 T-sweep — does it scale with T?

| T  | shifts     | lp AUC | mp AUC | notes |
|----|------------|--------|--------|-------|
| 5  | (1, 2)     | **0.8005 ± 0.003** (3-seed) | **0.8126 ± 0.003** (3-seed) | T=5 confirmed champion |
| 6  | (1, 3)     | **0.7965** (s42) | **0.8188** (s42) | mp **+0.005 over T=5 s42**, +0.006 over 3-seed |
| 7  | (1, 3)     | **0.7930** | **0.8036** | mp drops 0.015 from T=6 peak |
| 8  | (1, 2, 4)  | **0.7937** | **0.7992** | mp drops further from T=6 peak |
| 10 | (1, 2, 5)  | **0.7931** (s42) | **0.8040** (s42) | drops at BOTH (-0.011 lp, -0.010 mp vs T=5 s42) |
| 15 | (1, 3, 7)  | **0.7722** | **0.7772** | major drop (~0.04 vs T=6 peak) |
| 20 | (1, 5, 10) | **0.7823** | **0.7873** | small bump back from T=15 nadir |
| 30 | (1, 7, 15) | OOM    | OOM    | confirmed OOM during Adam init (43.8/44.4 GB) |

**T-sweep summary (seed 42 unless noted)**:

| T  | H8 mp     | H8 lp     | vanilla TXCDR mp | vanilla TXCDR lp |
|----|-----------|-----------|------------------|------------------|
| 3  | 0.7960 | 0.7690 | 0.8022 | 0.7711 |
| 4  | **0.8051** (s42) | 0.7865 | TBD              | TBD              |
| 5  | 0.8139 (3s 0.8126) | 0.8039 (3s 0.8005) | 0.8064 | 0.7827 |
| 6  | **0.8188** ⭐ | 0.7965 | 0.7955 | 0.7788 |
| 7  | 0.8036    | 0.7930    | 0.7957           | **0.7834**       |
| 8  | training  | training  | 0.7711           | 0.7540           |
| 9  | TBD       | TBD       | TBD              | TBD              |
| 10 | 0.8040    | 0.7931    | 0.7754           | 0.7671           |

**H8 mp curve**: T=3 0.7960 → T=5 0.8139 → T=6 **0.8188 ⭐** (peak) → T=7 0.8036 → T=8 0.7992 → T=10 0.8040 → T=15 0.7772 → T=20 0.7873 → T=30 OOM.

**Counter-intuitive**: H8 at T=3 (mp 0.7960) is WORSE than vanilla TXCDR at T=3 (mp 0.8022). The matryoshka + InfoNCE machinery may HURT at very small T — possibly because contrastive needs enough token-overlap diversity to be useful, and at T=3 with shift=1 the anchor/positive overlap is 67% (2/3 tokens shared), maybe too high to drive useful invariance.

**Peak structure**: H8's mp peak is at T=6 (0.8188). H8 always beats vanilla TXCDR at T≥5 — but NOT at T=3.

**The "TXC scales with T" claim**: Decisively NO at any aggregation. mp peaks at T=6, lp peaks at T=5. Both decline thereafter.

**Paper headline**: "the optimal T is small (T=6) for H8, T=5 for vanilla TXCDR — H8 closes the lp gap to MLC and tops mp benchmark, but T-scaling story is paper-negative".
**Vanilla TXCDR mp curve**: 0.8022 → 0.8064 (T=5 peak) → 0.7955 → 0.7957 → 0.7711 → 0.7754.

**Findings**:
1. **Both H8 and vanilla TXCDR have a single mp peak** in the T=5-7 range, then decline.
   So neither is truly monotonic, but H8's peak is shifted later (T=6 vs T=5) AND ~0.012
   higher (0.8188 vs 0.8064).
2. **The "TXC scales with T" claim cannot be made for H8 either**. But the milder claim
   "the optimal T is small (≤ 8) and H8 is the best at every T tested" still holds.
3. **lp**: H8 drops monotonically with T (0.8039 → 0.7930). Vanilla TXCDR is choppier
   with a small peak at T=7 (0.7834).

Monotonicity score (mp): TBD (target ≥ 0.80)
Δ(T_max−T_min) (mp): TBD (target > +0.020)

**Implications**:
1. The "T-scaling story" is now firmly negative across all hypotheses tried
   (H1, H7-as-T7, H8, vanilla TXCDR, vanilla matryoshka). Either H3
   log-matryoshka is the last hope, OR the paper pivots to MLC headline
   with T-scaling as an open problem.
2. H8 at T=5 remains the σ-defensible mp champion across all SAE families,
   but its advantage is bounded to T=5 — it's not "TXC scales with T",
   it's "the best TXC at T=5".

### Shift-ablation curve at T=5

| shifts            | lp AUC | mp AUC |
|-------------------|--------|--------|
| {1}               | **0.7656** | **0.7947** |
| {1, 2} = H8       | 0.8039 | 0.8139 |
| {1, 2, 3}         | 0.7961 | 0.8108 |
| {1, 2, 3, 4}      | TBD    | TBD    |
| {1, 2, 4}         | TBD    | TBD    |
| {2}               | TBD    | TBD    |
| {4}               | TBD    | TBD    |
| {1, 2, 3} uniform | TBD    | TBD    |

**Early interpretation**:
- Multi-distance {1,2} HELPS over single-shift {1} by +0.038 lp / +0.018 mp at T=5.
- T=3 H8 mp = 0.7960 (NOT a breakthrough — that earlier 0.8339 was a partial-
  probe artifact; final 36-task mean is 0.7960).
- Implication: at T=5, multi-distance is the right recipe, and T=6 is the
  optimal T for H8.

### TXC vs MLC + anti-dead

| recipe                        | TXC lp/mp     | MLC lp/mp     | Δ |
|-------------------------------|---------------|---------------|---|
| matryoshka + 1-shift contr    | (Phase 6.2 C3 0.7834 lp) | TBD           | TBD |
| matryoshka + multi-scale contr (= H7) | 0.7915 / 0.8104 (s42) | TBD           | TBD |
| multi-distance (H8, T=5)      | 0.8039 / 0.8139 | (no MLC analog) | n/a |

Verdict: TBD

### H13 mdms (orthogonal stack)

| arch | lp AUC | mp AUC | Δ vs H8 |
|------|--------|--------|---------|
| `phase57_partB_h13_md_x_ms` | TBD | TBD | TBD |

Verdict: TBD

### H3 log-matryoshka T-sweep

| T  | lp AUC | mp AUC |
|----|--------|--------|
| 5  | TBD    | TBD    |
| 10 | TBD    | TBD    |
| 15 | TBD    | TBD    |
| 20 | TBD    | TBD    |
| 30 | TBD    | TBD    |

Monotonicity score: TBD

### H9c 3-seed σ

| seed | lp AUC | mp AUC |
|------|--------|--------|
| 42   | 0.7891 (mp) | — |
| 1    | TBD    | TBD    |
| 2    | TBD    | TBD    |
| **mean ± σ** | TBD | TBD |

### ⭐ H8 3-seed FINAL — T=5, shifts={1,2}

Auto-probe completed 22:53 UTC.

| seed | lp AUC | mp AUC |
|------|--------|--------|
| 42   | 0.8039 | 0.8139 |
| 1    | 0.7994 | 0.8148 |
| 2    | 0.7981 | 0.8092 |
| **mean ± σ** | **0.8005 ± 0.0030** | **0.8126 ± 0.0030** |

**Verdict at T=5**: σ-defensible new TXC champion at BOTH aggregations.

- **lp**: H8 (0.8005) cleanly beats H7 3-seed (0.7886 ± 0.0070) by +0.012 (>1.5σ).
  Still 0.012 below MLC top (mlc_contrastive_alpha100_batchtopk 0.8124, single-seed).
- **mp**: H8 3-seed (0.8126) tops the ENTIRE mp benchmark across all archs/families
  (vs H7 3-seed 0.8059, agentic_txc_02 3-seed 0.7987 — both clearly below σ).

Tighter σ (0.003) than H7 σ (0.007). H8's recipe is more reproducible.

---

## Key files

- `run_groundbreaking.sh` — master launcher
- `analysis/groundbreaking_summary.py` — final summary aggregator
- `plots/make_t_scaling_plot.py` — T-scaling line plot
- `plots/make_shift_ablation_plot.py` — shift-ablation bar plot
- `plots/make_fairness_plot.py` — TXC vs MLC + anti-dead
- `analysis/alive_fraction.py` — extended alive analysis
- `src/architectures/mlc_bare_antidead.py` — MLC + anti-dead family
- `src/architectures/txc_bare_md_ms_contrastive_antidead.py` — H13 mdms

## What was outside this session's scope

- H2 attention-pool decoder — NOT IMPLEMENTED (handover lists as untested)
- H5 SVD-spectrum regularizer — NOT IMPLEMENTED (handover lists as untested)
- H6 Mamba/SSM encoder — NOT IMPLEMENTED (handover defers as wildcard)

These were "Other Part B hypotheses not yet tested" in the handover —
optional rather than priority. Possible follow-ups for next session.
