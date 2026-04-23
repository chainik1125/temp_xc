---
author: Han
date: 2026-04-23
tags:
  - proposal
  - in-progress
---

## Handover: T-sweep consolidation + extension + T-scaling autoresearch

**Audience**: post-compact agent. Previous agent (this one) ran out of
context mid-stream; this doc captures the outstanding work so you can
pick up cold.

**State at handover**: `han` branch, last commit `e5c39d0`. Extended
BatchTopK pipeline is **still probing mean_pool** (the 17-arch
extension). Monitor: `ls -la logs/overnight/probe_batchtopk_extend_mean_pool.log`
should be growing; `ps -ef | grep run_probing` should show PID 16715
alive (or a successor).

---

## Part A — Immediate wrap-up gaps

Items from the **original** `2026-04-22-handover-batchtopk-tsweep.md`
that were missed:

### A1. Concat-latent probing (experiment iv step 5) — script ready, not run

Script exists at [`analysis/concat_probe.py`](../../../experiments/phase5_downstream_utility/analysis/concat_probe.py)
and is end-to-end complete (encodes TXC + MLC per example, concatenates
to `(N, 2·d_sae=36864)`, runs the standard top-k-by-class-sep + L1 LR
on the concat, at both aggregations, seed=42, all 36 tasks).

Just needs to run:

```bash
PYTHONPATH=. TQDM_DISABLE=1 .venv/bin/python \
  experiments/phase5_downstream_utility/analysis/concat_probe.py
```

Takes ~5-10 min on A40 (encode + CPU probe). Coordinate with the
running extended-BatchTopK mean_pool probe — they'd share GPU briefly.
Output lands at `results/concat_probe_results.json` with `mean_auc`
+ `per_task` breakdown for both aggregations.

**Deliverable**: update `summary.md` section "Complementarity: TXC/MLC
routing" to include concat-probe row. Rename heading to
"Complementarity: TXC/MLC routing **and concat probing**" (per the
original handover's deliverable spec).

Expected result: if concat-probe AUC > best individual, "use both
archs at once" is the stronger claim. If < best individual, the
concat dilutes signal (too many features in the top-k-by-class-sep
pool), and routing-by-task is the better granularity.

### A2. BatchTopK inference-threshold sanity check

From original handover Known Risk (experiment ii): *"Sanity check: run
forward on a training batch in eval mode and confirm the sparsity rate
is close to `k/d_sae`. If not, bump the momentum up or use a
quantile-based threshold."*

Some BatchTopK archs regressed −0.02 AUC vs TopK counterparts
(agentic_txc_02, mlc). Plausible cause: EMA threshold miscalibration.
Quick check:

```python
from src.architectures._batchtopk_variants import MatryoshkaTXCDRContrastiveMultiscaleBatchTopK
import torch
ckpt = torch.load("experiments/phase5_downstream_utility/results/ckpts/agentic_txc_02_batchtopk__seed42.pt", map_location="cuda", weights_only=False)
m = MatryoshkaTXCDRContrastiveMultiscaleBatchTopK(2304, 18432, T=5, k=500, n_contr_scales=3, gamma=0.5).cuda()
m.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
m.eval()
# Forward on a batch of real activations, check nonzero rate per sample.
# Expected: ~k/d_sae = 500/18432 ≈ 0.027 on average.
# If actual is << that, threshold is too high (model shuts down at eval);
# if >> that, threshold is too low.
```

If miscalibrated, post-hoc fix: recalibrate threshold by forward on a
large unlabeled batch and set `m.sparsity.threshold` to the (d_sae − k)
quantile of pre-activation magnitudes. Then re-save the ckpt and
re-probe the affected archs.

### A3. 3-seed variance on BatchTopK winners

Original handover recommended 3-seed variance on agentic_*_batchtopk.
Current state: only seed=42. Train seeds 1, 2 for:

- `agentic_txc_02_batchtopk`
- `agentic_mlc_08_batchtopk`
- optionally `mlc_contrastive_alpha100_batchtopk` (strong last_position
  finding — 0.8027, only 0.002 below its TopK counterpart)

Pattern: `experiments/phase5_downstream_utility/agentic/seed_variance.sh`
already exists for the original 4-arch set; extend the CANDS list to
include `_batchtopk` suffixes and the new extended-scope archs.
~2 hr wall-clock for 3 archs × 2 seeds.

### A4. Minor formatting items

- `summary.md` BatchTopK section needs a **full 21-arch TopK-vs-BatchTopK
  Δ table** once mean_pool probing completes. Currently it shows only
  the original 4 archs in the Δ table and mentions extended partial
  results in prose.
- Run `plots/make_batchtopk_plot.py` and `plots/make_headline_plot.py`
  again after mean_pool finishes so all plots reflect the full data.
- HF sync: `HF_HOME=/workspace/hf_cache .venv/bin/python scripts/hf_upload_ckpts.py`
  after any new ckpts land.

---

## Part B — Extend T-sweep to large T (T ∈ {24, 28, 32, 36})

### Motivation

Inspect `results/plots/txcdr_t_sweep_batchtopk_comparison_last_position.png`:
BatchTopK AUC **rises from T=15 (0.7577) to T=20 (0.7672)**, a +0.010
jump back toward the T=5 peak. This suggests the curve may still be
climbing — we haven't found the BatchTopK peak at large T. Possible
outcomes at T ∈ {24, 28, 32, 36}:

1. **Monotone ascent**: AUC keeps climbing past T=20 — big finding,
   BatchTopK + large T is a genuine improvement. The current "T=5 is
   the sweet spot" narrative is wrong for this sparsity mechanism.
2. **Plateau**: AUC stabilises around 0.77 at T≥20 — confirms BatchTopK
   has an *extended* optimum rather than a second peak at T=20.
3. **Descent**: T=20 was a blip — descent at T≥24 brings BatchTopK
   back to the U-shape story.

### Scope

Train 4 new TXCDR archs at each T value, both sparsity mechanisms:

- TopK: `txcdr_t{24,28,32,36}` (4 archs)
- BatchTopK: `txcdr_t{24,28,32,36}_batchtopk` (4 archs)

**Skip matryoshka (agentic) variants at T≥24** — briefing from the
completed (iii) T-sweep documented OOM at T≥10 for the matryoshka
decoder (O(T³·d_in) parameter scaling). Large T is vanilla-TXCDR-only.

### Implementation

1. **Dispatcher branches**: extend the existing `elif arch in ("txcdr_t2", ...)`
   tuple in `train_primary_archs.py` (line ~1096) to include the new
   T values. Same for the BatchTopK variant dispatcher at line ~1430.
2. **Probe routing**: `arch.startswith("txcdr_t")` already matches —
   no change needed in `run_probing.py`.
3. **BASE_OF entries** in `run_autoresearch.sh`:
   ```
   BASE_OF[txcdr_t24]=txcdr_t20
   BASE_OF[txcdr_t28]=txcdr_t24
   ...
   BASE_OF[txcdr_t24_batchtopk]=txcdr_t24
   ...
   ```
4. **Probing caveat — T > 20 at mean_pool**: current probe_cache has
   `acts_anchor` shape `(N, 20, d)` and `acts_mlc_tail` shape
   `(N, 20, L, d)`. At T > 20 the mean_pool slider `K = LN − T + 1 ≤ 0`
   — **mean_pool probing for T > 20 won't work on the current cache**.
   Options:
   - **(Recommended)** Only probe at last_position for T > 20; the
     T-scaling story needs single-aggregation evidence first.
     `_window_at_last` pads cleanly if `T > LN`.
   - Rebuild probe_cache with `TAIL_LEN = 40` (slow, recomputes 36
     tasks × 2 splits — ~1 hr).

   Start with last_position only; decide on mean_pool based on what
   last_position shows.

### Time estimate

8 archs × ~20 min each = ~2.5 hr train. Probing (last_position only):
8 archs × ~5 min = 40 min. **Total ~3.5 hr**.

---

## Part C — Consolidate T-sweep results into a single section

Currently `summary.md` has T-sweep info scattered across four places:

- Line 382: `#### T-sweep on agentic_txc_02 (Phase 5.7 experiment iii)`
- Line 449: `#### T-sweep ladder (TXCDR at T ∈ {2, 3, 5, 8, 10, 15, 20})`
- Line ~499 (just added): `#### T-sweep under BatchTopK sparsity`
- Figure 1 / Figure 2 tables each contain T-sweep rows inline

**Task**: merge into a single section `#### T-sweep matrix — {TopK, BatchTopK}
× {TXCDR, agentic_txc_02} × {last_position, mean_pool}`. Target: a
single table that reads the full T-sweep matrix at a glance.

Proposed layout:

```md
#### T-sweep matrix

**TXCDR vanilla (matryoshka off):**

| T | TopK lp | TopK mp | BatchTopK lp | BatchTopK mp |
|---|---|---|---|---|
| 2 | 0.7380 | 0.7713 | 0.7623 | ... |
| 3 | 0.7659 | 0.7931 | 0.7547 | ... |
| 5 | 0.7752 | 0.7991 | 0.7678 | 0.7832 |
| 8 | 0.7498 | 0.7635 | 0.7434 | ... |
| 10 | 0.7573 | 0.7670 | 0.7492 | ... |
| 15 | 0.7695 | 0.7766 | 0.7577 | ... |
| 20 | 0.7425 | 0.7464 | 0.7672 | ... |
| **24** | new | n/a (T>20) | new | n/a |
| **28** | new | n/a | new | n/a |
| **32** | new | n/a | new | n/a |
| **36** | new | n/a | new | n/a |

**agentic_txc_02 multi-scale (matryoshka on, γ=0.5, n_contr_scales=min(3,T)):**

| T | TopK lp | TopK mp | BatchTopK lp | BatchTopK mp |
|---|---|---|---|---|
| 2 | 0.7518 | 0.7725 | 0.7495 | ... |
| 3 | 0.7469 | 0.7870 | 0.7531 | ... |
| 5 | 0.7775 | 0.8007 | 0.7543 | 0.7772 |
| 8 | 0.7626 | 0.7839 | 0.7539 | ... |
| 10 | OOM | OOM | OOM | OOM |
| 15 | OOM | OOM | OOM | OOM |
| 20 | OOM | OOM | OOM | OOM |
```

Plus a single consolidated plot: 4-panel grid
`{TXCDR, agentic} × {last_position, mean_pool}`, each panel plotting
TopK + BatchTopK curves together. Template: extend
[`plots/plot_txcdr_t_sweep_batchtopk.py`](../../../experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep_batchtopk.py).

Keep the existing narrative paragraphs (U-shape for BatchTopK, T=5 peak
for TopK, mechanism hypotheses), but move them after the consolidated
table.

Delete the three scattered T-sweep subsections and reference the
consolidated one from the relevant earlier sections.

---

## Part D — Autoresearch for a T-scaling architecture

### The research question

Our current TXC variants (TopK and BatchTopK) **fluctuate with T** — no
monotone improvement with context window size. This is the central
weakness of the paper's "temporal SAE" claim: if AUC doesn't scale
with T, we haven't shown that longer contexts help.

**Objective**: find an architectural change (or probing trick) such
that **AUC increases monotonically with T** on our 36-task bench.
Even a weakly monotone curve (no T=x peak, gradual +0.005/T rise)
would be paper gold — it would let us write "our temporal SAE
genuinely uses longer context".

Pattern: Karpathy-style agentic autoresearch. Reference:
[`2026-04-21-agentic-log.md`](2026-04-21-agentic-log.md) for the
8-cycle loop that discovered cycle 02's multi-scale contrastive recipe.
Same format: **hypothesis → architectural change → evaluate (T-sweep
at last_position, 3-5 T values) → takeaway**. Budget 8-12 cycles.

### Evaluation protocol

For each candidate arch (a new `.py` file under `src/architectures/`):

1. Train at **5 T values: T ∈ {5, 10, 15, 20, 30}** at seed=42, 25k
   max steps, plateau-stop. (Skip T=2, T=3 — too short to test
   scaling.) If T=30 OOMs, substitute T=24.
2. Probe at **last_position only** (mean_pool constrained by tail=20).
3. Compute **monotonicity score**: fraction of T-pairs where
   `auc(T_i) ≤ auc(T_j)` for i < j. Range 0 (fully anti-monotone)
   to 1 (strictly monotone). Current vanilla TXCDR BatchTopK: 0.6
   (6 of 10 pairs monotone). Target: ≥ 0.8.
4. **Secondary: Δ(T=30 − T=5)** — actual AUC gain from scaling up.
   Current vanilla TXCDR BatchTopK: ~0.0. Target: > +0.02.

### Candidate hypotheses (seed the agentic loop)

Ordered by expected impact × ease:

#### H1 — Attention-pooling decoder

Replace per-position decoder `W_dec^(t) : (d_sae, d_in)` with a
cross-attention read-out: the latent `z : (d_sae,)` queries T
position-specific keys and produces a single attention-weighted
decoded vector. Probing reads `z` unchanged, but the **reconstruction
loss** is now position-shared. The claim: latents that work across
all T positions are concept-coded (not position-coded), which should
benefit larger T.

#### H2 — Convolutional encoder

Replace `W_enc : (T, d_in, d_sae)` with a 1-D conv of kernel size ≥ 3
across the T positions, sharing weights. The encoder becomes genuinely
translation-invariant — features don't have position-specific
imprinting. Large T then gets more "data" to compute each feature
over, should help scaling.

#### H3 — Hierarchical scale decoder

Instead of matryoshka's T nested scales (scale-s reconstructs
s-token sub-windows), use **log-scale** sub-windows: scale-1 = 1 token,
scale-2 = 2 tokens, scale-3 = 4 tokens, scale-4 = 8 tokens, scale-5 = 16
tokens, scale-6 = full window. Captures coarse-grain temporal structure
while keeping decoder param count O(T · d_in · d_sae), not O(T² · d_in · d_sae).

#### H4 — Multi-distance contrastive

Instead of contrasting only shift-1 pairs, contrast **multiple shift
distances simultaneously**: shift-1 (local), shift-⌊T/4⌋ (medium),
shift-⌊T/2⌋ (long). Weight inversely with distance. Forces latents to
be consistent across both short-range and long-range context.

#### H5 — Positional-probe with SAE features

**Not an arch change, but a probe change.** Current probe uses
top-k-by-class-sep on the d_sae feature vector at last-position. At
large T, the SAE has features that fire at earlier positions that
contribute to the answer but the last-position probe can't see them.

Extension: for each feature f, compute per-position activation
`z[t, f]` across the T-window. Use **attention-pooled SAE features**
(like the baseline_attn_pool but on d_sae instead of d_in) to
aggregate. Let the probe learn which positions of which features
matter. Should naturally scale with T.

#### H6 — State-space encoder (Mamba-style)

Replace the per-position `W_enc` with a recurrent state-space model
that processes T positions sequentially with fixed per-step compute.
Features become genuinely sequential. Compute cost O(T · d_sae) not
O(T² · d_sae) like attention.

#### H7 — SVD-spectrum regularizer

The existing summary.md §"Per-feature decoder SVD" documents that
vanilla TXCDR at T=20 has a 7.5% flatter singular-value spectrum than
T=5. Add an explicit penalty on the flatness of the per-feature
decoder SVD. This directly targets the "over-regularization at large
T" failure mode identified in the Phase 5.7 analysis.

#### H8 — Stack of H1 + H3 winners

After running H1-H7, stack the two best into `agentic_t_scale_02`. If
cumulative monotonicity score ≥ 0.9 and Δ(30-5) > 0.02, this is the
new headline TXC.

### How to run each cycle

Same as Phase 5.7 agentic loop:

1. Write new arch class in `src/architectures/`.
2. Add dispatcher branch in `train_primary_archs.py` with naming
   `agentic_tscale_{01..08}` or similar.
3. Add probe routing in `run_probing.py`.
4. Train at T ∈ {5, 10, 15, 20, 30} (~2 hr per cycle = 25 min/T × 5).
5. Probe at last_position.
6. Compute monotonicity score + Δ(30-5).
7. Log to new `2026-04-24-t-scaling-agentic-log.md` with
   hypothesis → change → result → takeaway.
8. Commit per cycle.

### Expected trajectory

| cycle | expected monotonicity | expected Δ(30-5) |
|---|---|---|
| H1: attention decoder | 0.7 | +0.005 |
| H2: conv encoder | 0.8 | +0.010 |
| H3: log-scale matryoshka | 0.75 | +0.005 |
| H4: multi-distance | 0.7 | +0.010 |
| H5: positional probe | 0.9 | +0.020 |
| H6: state-space | 0.85 | +0.015 |
| H7: SVD regularizer | 0.7 | +0.005 |
| H8: stack | **0.9** | **+0.025** |

H5 (positional probe) is the highest-expected-value *single* change
because it changes the probe rather than retraining the arch — you can
test it on existing ckpts in hours, not days. If it works, the paper
pivots to "TXC features ARE T-scaling; the probe was the bottleneck".

---

## Resume checklist (first 15 min)

1. `git log --oneline | head -10` — confirm at `e5c39d0` or later.
2. Check extended-probe pipeline status: `ps -ef | grep run_probing`.
   If still running, wait; if done, grep for mean_pool BatchTopK
   coverage via the partB_summary pattern.
3. Read this doc + the original
   [`2026-04-22-handover-batchtopk-tsweep.md`](2026-04-22-handover-batchtopk-tsweep.md)
   for missed items it lists.
4. Check `analysis/concat_probe.py` — it's ready to run.
5. Current summary.md state — the BatchTopK extended partial results
   are there but the **full 21-arch Δ table** is not yet written.
6. Start with Part A wrap-up (concat-probe + BatchTopK sanity check) —
   fast, closes existing loops. Then Part B (extend T-sweep) in
   parallel with Part C (consolidate section). Part D (autoresearch)
   is the paper-level priority but can run after A+B+C land.

## Key files (cheat sheet)

- **Module**: [`src/architectures/_batchtopk.py`](../../../src/architectures/_batchtopk.py),
  [`src/architectures/_batchtopk_variants.py`](../../../src/architectures/_batchtopk_variants.py)
- **Dispatchers**: [`experiments/phase5_downstream_utility/train_primary_archs.py`](../../../experiments/phase5_downstream_utility/train_primary_archs.py)
- **Probe routing**: [`experiments/phase5_downstream_utility/probing/run_probing.py`](../../../experiments/phase5_downstream_utility/probing/run_probing.py)
- **Scripts for extended runs**: [`experiments/phase5_downstream_utility/run_batchtopk_extend.sh`](../../../experiments/phase5_downstream_utility/run_batchtopk_extend.sh),
  [`analysis/concat_probe.py`](../../../experiments/phase5_downstream_utility/analysis/concat_probe.py),
  [`analysis/router.py`](../../../experiments/phase5_downstream_utility/analysis/router.py)
- **T-sweep plots**: [`plots/plot_txcdr_t_sweep.py`](../../../experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep.py),
  [`plots/plot_txcdr_t_sweep_batchtopk.py`](../../../experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep_batchtopk.py)
- **Phase 5.7 agentic log (template)**: [`2026-04-21-agentic-log.md`](2026-04-21-agentic-log.md)
- **Summary**: [`summary.md`](summary.md) — current structure at line 449 has the consolidation target
