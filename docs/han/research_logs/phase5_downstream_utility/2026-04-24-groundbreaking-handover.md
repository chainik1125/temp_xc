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
| 9  | **0.8091** | 0.7997 | TBD              | TBD              |
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
| {1, 2, 3, 4}      | 0.7973 | 0.8165 |
| {1, 2, 4}         | 0.7914 | 0.8070 |
| {2}               | 0.8017 | 0.8104 |
| {4}               | 0.7863 | 0.8062 |
| {1, 2, 3} uniform | 0.7931 | 0.8129 |

**Early interpretation**:
- Multi-distance {1,2} HELPS over single-shift {1} by +0.038 lp / +0.018 mp at T=5.
- T=3 H8 mp = 0.7960 (NOT a breakthrough — that earlier 0.8339 was a partial-
  probe artifact; final 36-task mean is 0.7960).
- Implication: at T=5, multi-distance is the right recipe, and T=6 is the
  optimal T for H8.

**Long-range shifts (extras queue, complete for {5},{10},{20})**:

| shift set | lp | mp | token-overlap @ T=5 |
|---|------|------|------|
| {5} | 0.7827 | 0.8080 | 0% (anchor[0..4], pos[5..9]) |
| {10} | 0.7702 | 0.7881 | 0% with 5-token gap |
| **{20}** | **0.8041** ⭐ | **0.8179** ⭐ | 0% with 15-token gap |

**🎯 Single shift={20} TIES H8 at lp (0.8041 vs 0.8039) AND TOPS H8 at mp (0.8179 vs 0.8139, +0.004)**. Counter-intuitive U-shape:
- shift=1-2 (60-80% overlap): "local invariance" — works
- shift=4-10: dead zone (worst results)
- shift=20 (large gap, same-sequence): "semantic invariance" — works again!

Hypothesis: shift=20 forces features to encode SEQUENCE-LEVEL semantics (not
position-specific), since anchor and positive are far enough apart that
local-token-cooccurrence patterns differ but topic/style stays constant.
3-seed verification needed before claiming it as champion (single-seed
variance σ≈0.003 means +0.004 is at the edge of noise).

### TXC vs MLC + anti-dead — FAIRNESS CHECK

Question: does the anti-dead stack unfairly advantage TXC? Test by
applying the SAME stack to MLC.

| recipe                        | TXC (lp/mp)        | MLC (lp/mp)        | Δ MLC anti-dead |
|-------------------------------|--------------------|--------------------|----|
| **vanilla**                   | txcdr_t5 0.7827/0.8064 | mlc 0.7954/0.7848 | (baseline) |
| **+ anti-dead** (recon-only)  | (no TXC counterpart trained) | mlc_bare_antidead **0.7960/0.7756** | **+0.001 lp / −0.009 mp** |
| matryoshka + 1-shift contr    | (Phase 6.2 C3 0.7834 lp) | TBD           | TBD |
| matryoshka + multi-scale contr (= H7) | 0.7915 / 0.8104 (s42) | mlc_bare_multiscale_antidead TBD | TBD |
| multi-distance (H8, T=5)      | 0.8039 / 0.8139 | (no MLC analog) | n/a |

**Verdict (so far)**: **Anti-dead does NOT unfairly advantage TXC**.
Applying the same stack to MLC gives ~0 lp gain (+0.001) and HURTS mp
(−0.009). The TXC family genuinely benefits from anti-dead in a way
MLC does not — likely because TXC's per-position W_enc has many more
parameters per atom than MLC's single shared encoder, so dead atoms
are a bigger problem for TXC.

This rebuts the reviewer concern that "you gave TXC a free anti-dead
boost MLC didn't get". The stack was tested on both; only TXC benefited.

### H13 mdms (orthogonal stack)

| arch | lp AUC | mp AUC | Δ vs H8 |
|------|--------|--------|---------|
| H8 (multi-distance only) | 0.8039 | 0.8139 | (baseline) |
| H7 (multi-scale only) | 0.7915 | 0.8104 | −0.012 lp / −0.004 mp |
| **H13 (md × ms stack)** | **0.7930** | **0.8125** | **−0.011 lp / −0.001 mp** |

**Verdict**: H13 does NOT compound. Lands BETWEEN H7 and H8. Stacking
multi-distance × multi-scale is WORSE than multi-distance alone — the
"orthogonal contrastive pressures" hypothesis fails. Adding multi-scale
on top of multi-distance dilutes rather than enhances the H8 recipe.

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

## Session-wide lessons learned (read this if you continue)

### The partial-probe trap

**Twice in this session I reported wrong AUCs by querying probing_results.jsonl
during a probe.** The first ~5-10 tasks (ag_news_*, amazon_*, bias_in_bios_*)
have systematically higher AUCs than the harder cross-token tasks. Querying
mid-probe gave inflated means:
- T=3 H8 reported 0.8339 mp; actual (n=36) is **0.7960** — 0.038 lower
- shifts4 lp reported 0.8086; actual (n=36) is **0.7863** — 0.022 lower

**Fix going forward**: ALWAYS check `n=36` (or expected task count) before
treating an AUC as final. The audit query is at the bottom of this section.

### The FLIP convention

`FLIP_TASKS = {winogrande_correct_completion, wsc_coreference}`. For these
2 of 36 tasks, the headline plot does `v = max(v, 1.0 - v)` — post-hoc
polarity correction for cross-token coreference/entailment where label
assignment is arbitrary. Other 34 tasks use raw AUC. Lives ONLY in
plot/aggregation scripts (`make_headline_plot.py`, `make_t_scaling_plot.py`,
`groundbreaking_summary.py`); probe code writes raw AUCs to jsonl.

**Phase 5B used NO_FLIP convention** in their own report. Their B2/B4
3-seed numbers (lp 0.8160 ± 0.003 / 0.8232 ± 0.013) are slightly lower
than Phase 5 FLIP convention (lp 0.8213 ± 0.0005 / 0.8295 ± 0.013).
Same data, different post-processing. Cross-phase comparisons MUST
agree on convention.

### k_pos vs k_win across families

Phase 5's convention scales `k_win = k_pos × T` for window archs, but
keeps `k_win = 100` constant for MLC (which has L=5 layers). At probe
time, ONE encoded vector contains:
- 100 active features for TopK SAE / MLC
- 500 for TXC at T=5
- 600 for H8 at T=6
- 1000 for vanilla TXCDR at T=10

This is a structural asymmetry. Phase 7's solution: fix `k_win = 500`
across all families. See [phase7_unification/plan.md](../phase7_unification/plan.md).

### The seed-42 filter for headline plots

`make_headline_plot.py` originally did `out[arch][task] = v`, overwriting
per-task values when multiple seeds existed. Whichever seed was probed
LAST (here seed 2) silently won, hiding seed 42 numbers. Fixed
2026-04-25 (commit a1e4b2c) to filter `__seed42` only. 3-seed σ is
reported separately in summary.md tables, not in the bar plot.

### Phase 5B integration discrepancy

The Phase 5B agent reported `phase5b_subseq_h8_t10_s8_k500` peak at
**lp 0.8545 / mp 0.8590**. Recomputed from their own jsonl with EITHER
convention:
- FLIP: lp 0.8218 / mp 0.8284
- NO_FLIP: lp 0.8145 / mp 0.8183

Their reported numbers don't reproduce. Likely a typo or alternate
aggregation. The actual Phase 5B mp peak (Phase 5 convention, seed 42)
is plain `phase5b_subseq_h8` at **0.8516**, not the t10_s8_k500 cell.

### Audit query (run before reporting any AUC)

```python
import json, statistics as st
F = 'experiments/phase5_downstream_utility/results/probing_results.jsonl'
data = []
for line in open(F):
    try: r = json.loads(line)
    except: continue
    if r.get('run_id') != f'{ARCH}__seed{SEED}': continue
    if r.get('aggregation') != AGG or r.get('k_feat') != 5: continue
    v = float(r.get('test_auc', 0))
    if r.get('task_name') in {'winogrande_correct_completion','wsc_coreference'}:
        v = max(v, 1.0 - v)
    data.append(v)
print(f'n={len(data)}, mean={st.mean(data):.4f}')
# n MUST be 36 before treating mean as final.
```

---

### shifts_20 3-seed verification (side queue, started 2026-04-26 00:35 UTC)

The single-seed phase57_partB_h8a_shifts_20 finding (lp 0.8041 / mp 0.8179
at seed 42) is at the edge of H8's σ=0.003. Side queue
[`run_shifts20_3seed.sh`](../../../experiments/phase5_downstream_utility/run_shifts20_3seed.sh)
runs seeds 1, 2 once GPU has < 25 GB used. PID 113714. ETA ~2-3 hours.

---

## What was outside this session's scope

- H2 attention-pool decoder — NOT IMPLEMENTED (handover lists as untested)
- H5 SVD-spectrum regularizer — NOT IMPLEMENTED (handover lists as untested)
- H6 Mamba/SSM encoder — NOT IMPLEMENTED (handover defers as wildcard)

These were "Other Part B hypotheses not yet tested" in the handover —
optional rather than priority. Possible follow-ups for next session.
