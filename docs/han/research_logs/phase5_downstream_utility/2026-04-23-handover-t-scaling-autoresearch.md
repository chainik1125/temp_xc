---
author: Han
date: 2026-04-23
tags:
  - proposal
  - in-progress
---

## Handover: finish Phase 5.7 session items, then find a T-scaling TXC arch

**Audience**: post-compact agent. Previous agent ran out of context
mid-stream.

**Scope**: one continuous session. Finish Part A first (foundation),
then Part B (the research push). Both before you hand back — don't
stop half-done.

**State at handover**: `han` branch at commit `75d90ad`. Extended
BatchTopK pipeline is **still probing mean_pool** (17-arch extension).
Monitor via `tail logs/overnight/probe_batchtopk_extend_mean_pool.log`.

---

## Overview — two parts, in order

**Part A**: finish the open session items that leave us with a
complete, reviewer-defensible Phase 5.7 bench. These are not research
— they're mechanical follow-throughs on experiments (i)-(iv) that
were started but not closed.

**Part B**: find a TXC architecture whose downstream-probing AUC
scales monotonically with T under the **fixed** evaluation protocol.
Paper's central claim hinges on this. If no arch scales with T, the
paper pivots to headlining MLC instead of TXC.

**Do Part A first.** Part A is what makes the paper submittable at
all; Part B is what makes the paper's TXC headline defensible.
Without Part A the submission is incomplete; without Part B the
submission is weak-but-complete (MLC pivot).

---

## Evaluation protocol is FIXED. Do not modify it.

The probing protocol is the benchmark. It cannot be changed to favor
a particular arch:

1. **Fixed aggregation**: `last_position` and `mean_pool`, defined as
   they are today in
   [`probing/run_probing.py`](../../../experiments/phase5_downstream_utility/probing/run_probing.py).
2. **Fixed feature selection**: top-k-by-class-separation on train
   (Kantamneni Eq. 1), `k_feat = 5`.
3. **Fixed classifier**: L1 logistic regression (`penalty="l1",
   solver="liblinear", max_iter=2000, C=1.0`) on the 5 selected features.
4. **Fixed test split**: the existing `probe_cache/*/acts_*.npz` splits.
5. **Fixed seed**: 42 (plus 1, 2 for variance reporting).
6. **Fixed task set**: the 36 binary tasks in `results/probe_cache/`.

**Any change to the evaluation protocol is reward-hacking unless it
applies uniformly to ALL archs in the bench.** If you find yourself
wanting to add an attention-pooling step to the probe so your TXC
features look better, stop. Either:

- (a) Keep the arch under the fixed probe — if AUC scales with T here,
  you've won honestly.
- (b) Propose a new probing protocol *in a separate paper*, apply it
  uniformly to the 30-arch bench, and report whatever it shows.

Do **not** mix these. A reviewer will notice and reject.

### Things that are *not* reward-hacking (OK to do)

- Training a new architecture that outputs the same d_sae latent at the
  same last position — fits the standard probe.
- Changing the training objective (contrastive loss, sparsity
  mechanism, decoder structure) — arch changes are fair.
- Running the standard probe at k_feat ∈ {1, 2, 5, 20} as a sensitivity
  check — this is already in the bench. k_feat=5 remains the headline.
- Sweeping T at the architecture level and reporting all T values
  including those that don't look great.

### Things that ARE reward-hacking (forbidden)

- Changing k_feat only for your candidate arch.
- Introducing an attention-over-positions step in the probe for TXC
  but not for TLC / MLC / TFA.
- Dropping tasks where your arch does poorly.
- Using different train/test splits.
- Training on the test set or on val data leaked from test.
- Tuning probe hyperparameters (C, solver, scaler) per arch.

---

## Part A — finish outstanding session items

### A1. Concat-latent probing (experiment iv step 5)

Script ready at
[`analysis/concat_probe.py`](../../../experiments/phase5_downstream_utility/analysis/concat_probe.py).
Encodes every task's train + test activations through both
`agentic_txc_02` and `agentic_mlc_08`, concatenates to `(N, 2·d_sae)`,
runs the standard top-k-by-class-sep + L1 LR at k=5.

```bash
PYTHONPATH=. TQDM_DISABLE=1 .venv/bin/python \
  experiments/phase5_downstream_utility/analysis/concat_probe.py
```

Takes ~5–10 min on A40. Coordinate with the still-running mean_pool
probe — they share GPU briefly. Writes
`results/concat_probe_results.json`.

**Deliverables**:
- New column in `summary.md` §"Complementarity: TXC/MLC routing" —
  adds `concat probe` to the 4-column table (best individual / oracle /
  learned / **concat**).
- Rename section heading to "Complementarity: TXC/MLC routing **and
  concat probing**" (per original handover deliverable spec).
- 1-paragraph interpretation: if concat > best individual, the "use
  both archs at once" story is the stronger claim than routing.

### A2. BatchTopK inference-threshold sanity check

Origin: original handover Known Risk (experiment ii). Some BatchTopK
archs regressed −0.02 AUC vs TopK counterparts — plausibly because the
EMA-tracked inference threshold is miscalibrated.

```python
import torch
from src.architectures._batchtopk_variants import MatryoshkaTXCDRContrastiveMultiscaleBatchTopK
ckpt = torch.load("experiments/phase5_downstream_utility/results/ckpts/agentic_txc_02_batchtopk__seed42.pt",
                  map_location="cuda", weights_only=False)
m = MatryoshkaTXCDRContrastiveMultiscaleBatchTopK(
    2304, 18432, T=5, k=500, n_contr_scales=3, gamma=0.5).cuda()
m.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()})
m.eval()
# Forward a batch of real activations, check per-sample nonzero rate.
# Expected: ~k/d_sae = 500/18432 ≈ 0.027.
# If actual << that → threshold too high → model shuts down at eval.
# If actual >> that → threshold too low.
```

Do the check on all 21 BatchTopK archs. If any are miscalibrated (ratio
< 0.5× or > 2× expected), **re-calibrate threshold** from the
(d_sae − k) quantile of pre-activations on unlabeled fineweb data, save
new ckpt, re-probe. Document finding in `summary.md` Caveats.

### A3. 3-seed variance on baselines and headline winners

**Critical for headline defense.** Currently only the 2 agentic
winners have seeds {1, 2}; their baselines don't. Our Δ is
`mean(agentic)_3seeds − single_seed(baseline)` — a reviewer will ask
"what is the baseline's σ?" and we can't answer.

Minimum scope (run seeds 1 and 2 for these 6 archs):

- `txcdr_t5`, `mlc`, `matryoshka_t5`, `mlc_contrastive`
- `agentic_txc_02_batchtopk`, `agentic_mlc_08_batchtopk`

Pattern: `experiments/phase5_downstream_utility/agentic/seed_variance.sh`.
~6 hr wall-clock. After, update the Δ columns in `summary.md` Figure 1/2
tables with σ or with 95% CI, and add paired-t-test p-values with
Bonferroni correction across the 29-arch bench.

### A4. Full 21-arch TopK-vs-BatchTopK Δ table

Extended BatchTopK probing is still running as of handover. Once
complete, rewrite `summary.md` §"BatchTopK apples-to-apples" to include:

- Full Δ table: all 21 archs with (base, batchtopk, Δ) at both
  aggregations. Flag Δ < 0.01 as "within noise" honestly.
- Section title fix: per original handover, the Complementarity
  section should be "TXC/MLC routing and concat probing".
- Caveats updates: note the mixed picture (7/21 regressions were the
  headline; with 21 archs, pattern is more nuanced — some archs
  actually gain with BatchTopK: `topk_sae` +0.016, `txcdr_t2` +0.024).

### A5. Extended T-sweep to T ∈ {24, 28, 32, 36}

Motivation: BatchTopK T-sweep at last_position jumps from T=15 (0.7577)
to T=20 (0.7672), suggesting curve may still be climbing past T=20. We
need T > 20 data to know if this is a second peak, ascent, or plateau.

Scope: 8 new archs — `txcdr_t{24,28,32,36}` (TopK) and matching
`_batchtopk` variants. Skip matryoshka variants at T ≥ 24 (they OOM
at T ≥ 10 already — documented).

**Critical caveat**: mean_pool probing is blocked at T > 20 because
`acts_anchor` has LAST_N = 20; `K = LN − T + 1 ≤ 0`. Probe
**last_position only** at T > 20. `_window_at_last` pads cleanly.

Time: 8 archs × ~20 min train + ~5 min probe each = ~3.5 hr.

### A6. Consolidate T-sweep into one `summary.md` section

Currently scattered across 3 subsections (lines 382, 449, 499) plus
inline rows in Figure 1/2 tables. Merge into a single `#### T-sweep
matrix — {TopK, BatchTopK} × {TXCDR, agentic_txc_02} × {last_position,
mean_pool}` with 4-panel plot. Template provided in prior commit's
briefing; extend
[`plots/plot_txcdr_t_sweep_batchtopk.py`](../../../experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep_batchtopk.py)
for the 4-panel grid.

Do this **last** in Part A — one consolidation pass after all T data
lands (including A5's T > 20 results).

### A7. HF sync + final plot regeneration

```bash
PYTHONPATH=. .venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py
PYTHONPATH=. .venv/bin/python experiments/phase5_downstream_utility/plots/make_batchtopk_plot.py
PYTHONPATH=. .venv/bin/python experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep_batchtopk.py
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/hf_upload_ckpts.py
```

Commit per step. HF sync is idempotent (skip-by-hash).

---

## Part B — T-scaling architecture autoresearch

### The mission

Find a TXC architecture such that **sparse-probing AUC increases with T
under the fixed evaluation protocol** (see "Evaluation protocol is
FIXED" above).

### Success criteria

Train each candidate at T ∈ {5, 10, 15, 20, 30} seed=42. Probe at
last_position (mean_pool is tail-20-limited). Report:

1. **Monotonicity score**: fraction of T-pairs (i<j) where
   auc(T_j) ≥ auc(T_i). Range [0, 1], random ~0.5.
   - Current best TXC: ~0.6.
   - **Target: ≥ 0.8.**
   - Strict monotone: 1.0.
2. **Δ(T=30 − T=5)**: actual AUC gain from scaling up.
   - Current: ~0.0.
   - **Target: > +0.02.**

If both hit, that's the submission headline.

### Cycle budget

~2 hr per candidate (5 T × 25 min + 5 min probe). Run 8-10 cycles in
sequence. Each cycle is self-contained (write arch → train → probe →
score → log → commit). Don't parallelize cycles — serialize so each
result informs the next.

### Hypotheses (architectural changes only, NOT probe changes)

Ordered by ease × expected value. All use the standard probe.

#### H1 — Convolutional encoder

Replace `W_enc : (T, d_in, d_sae)` with a 1-D conv of kernel ≥ 3
across the T positions, weights shared. Encoder becomes
translation-invariant — features don't get position-imprinted. At
larger T, each feature is computed from more "data". Intuition: a
trans-invariant encoder should benefit more from longer context than
the current per-position encoder.

#### H2 — Attention-pooling decoder

Replace per-position `W_dec^(t) : (d_sae, d_in)` with a single shared
decoder accessed via learned cross-attention over T positions. Latent
queries → attention over T position keys → decoded via shared V. Forces
latents to be position-invariant; shared-decoder param count is
constant in T. (Note: this is a **decoder** change, not a probe change.
Latent is still a single d_sae vector at the last position.)

#### H3 — Log-scale matryoshka

Instead of T nested scales (scale-s reconstructs s-token sub-window),
use log₂-spaced scales: 1, 2, 4, 8, 16, 32 token sub-windows.
- Captures coarse temporal structure
- Escapes the O(T²) matryoshka decoder OOM that stopped our matryoshka
  variants at T=10
- Enables matryoshka-based arches at T ≥ 30

#### H4 — Multi-distance contrastive

InfoNCE at multiple shift distances simultaneously: shift-1 (local),
shift-⌊T/4⌋, shift-⌊T/2⌋ (long-range), inverse-distance-weighted.
Forces latents to be consistent across both local and long-range
contexts. Original 2026-04-22 handover queued this as Cycle E; run
here explicitly.

#### H5 — SVD-spectrum regularizer

Phase 5.7 §"Per-feature decoder SVD" documented that vanilla TXCDR at
T=20 has 7.5% flatter singular-value spectrum than T=5. Add penalty
pushing spectrum toward Zipfian. Directly targets the
"over-regularization at large T" failure mode identified empirically.

#### H6 — State-space / Mamba-style encoder

Replace per-position `W_enc` with a recurrent state-space model. Compute
O(T) instead of O(T²). Features become genuinely sequential. High
effort; try only if H1-H5 don't yield a hit.

#### H7 — Hybrid MLC-encoder + T-decoder

MLC's multi-layer sharing (the Phase 5 last_position winner) + a
per-position (or attention-pool) decoder over T positions. Attempts to
fuse the two winning ideas.

#### H8 — Stack winners from H1-H7

If any 2-3 of H1-H7 individually improve monotonicity score, stack them
into one arch and test. Stack candidates from `src/architectures/`
subclass-hierarchy; aim for <200 lines of new code.

### Per-cycle protocol (mirrors Phase 5.7 agentic loop)

1. Write new arch class in `src/architectures/`.
2. Dispatcher branch in `train_primary_archs.py`.
3. Probe routing in `run_probing.py`.
4. Train at T ∈ {5, 10, 15, 20, 30} (~2 hr).
5. Probe at last_position.
6. Compute (monotonicity, Δ(30-5)) — write `analysis/t_scaling_score.py`
   on cycle 1, ~30 lines.
7. Log to new `2026-04-24-t-scaling-agentic-log.md` with hypothesis →
   change → result → takeaway.
8. Commit per cycle.

### T-sweep constraints for large T

- **T > 20 at mean_pool blocked** by probe_cache tail size = 20.
  Probe last_position only.
- **Matryoshka at T ≥ 10 OOMs** at d_sae=18432. Only H3
  (log-matryoshka) escapes this; H2/H7 OK if they don't use matryoshka.
- **Training buffer supports T up to 128** — `resid_L13.npy` is
  (6000, 128, 2304).

### If no hypothesis wins

**The paper pivots to MLC.** `agentic_mlc_08` beats vanilla MLC by
+0.016 at last_position (likely p < 0.05 Bonferroni-corrected, needs
verification), 5/8 semantic autointerp labels in Phase 6. That's an
MLC paper with a temporal-crosscoder complementarity argument, not a
TXC paper. Submit it honestly rather than overclaim TXC.

Don't fake T-scaling by cherry-picking T values, tasks, or the probe.

---

## Methodological context (things reviewers will attack)

Background for decision-making; these are fixable in Part A or are
acknowledged limitations. None require Part B to resolve.

1. **Effect sizes within probing noise**: TXC headline Δ is +0.0023
   (within ~0.005 noise). 3-seed σ needed on baselines (Part A3).
   Without σ, the TXC Δ is not a defensible claim.
2. **T-scaling absence**: the central existential threat. Part B.
3. **Plateau-stop bias**: models converge at different step counts.
   Reviewer will ask "is the weak arch just undertrained?" → report
   (steps, final_loss, converged) per arch in a new summary table.
4. **winogrande + wsc polarity flip**: `max(AUC, 1-AUC)` inflates 2/36
   tasks. Remove or move to supplementary.
5. **12 pp gap to baselines**: SAEs lose to `baseline_attn_pool` by 12
   pp. Standard SAE community answer (interpretability) depends on
   Phase 6 — and TXC currently loses Phase 6. Reframe the paper's
   downstream-utility claim as "SAE features preserve X% of raw probe
   signal while enabling interpretability".
6. **"Family-agnostic" claim weakness**: γ=0.5 tested on MLC only at
   one value. Sweep γ ∈ {0.3, 0.5, 1.0} on MLC to verify — ~2 hr.
7. **Top-k-by-class-sep stability**: no train-subsample bootstrap
   reported. Easy to add as a sensitivity table.
8. **TLC d_sae=8192** — half the feature-pool vs others' 18432.
   Either acknowledge or retrain at matched d_sae.

---

## Resume checklist (first 15 min)

1. `git log --oneline | head -10` — confirm at `1458df8` or later.
2. **HF token**: `export HF_HOME=/workspace/hf_cache`. Token lives at
   `/workspace/hf_cache/token`, user `han1823123123`. Two repos:
   `han1823123123/txcdr` (ckpts), `han1823123123/txcdr-data` (probe
   caches + activations).
3. Check extended-probe pipeline: `ps -ef | grep run_probing`. If still
   running (PID was 16715 at handover), wait for mean_pool to finish
   before A4 / A6. Monitor:
   `tail -f logs/overnight/probe_batchtopk_extend_mean_pool.log`.
4. Read this doc + `summary.md` §T-sweep (lines 382, 449, 499) +
   original handover
   [`2026-04-22-handover-batchtopk-tsweep.md`](2026-04-22-handover-batchtopk-tsweep.md).
5. **Phase 6 is running in parallel** on branch `han-phase6`
   (commit `e40c0b6` at handover). Don't touch it. The Phase 6 agent
   has its own briefing
   [`docs/han/research_logs/phase6_qualitative_latents/2026-04-23-handover-txc-qualitative.md`](../phase6_qualitative_latents/)
   pursuing TXC qualitative improvements (AuxK port, decoder norm,
   etc.). If the Phase 5 T-scaling work here succeeds, the Phase 6
   agent may want to rebase the improved arch.
6. **Start with A1** (concat-probe, ~5-10 min). Fast, closes a loop.
7. A2 (BatchTopK sanity check, ~30 min) in parallel — CPU-ish.
8. A3 (3-seed variance) launched as a background batch while doing A4/A5.
9. Only after Part A completes, begin Part B. **Don't skip Part A for
   Part B** — the submission needs A to be complete.

## Key files (cheat sheet)

- **Architectures**: [`src/architectures/`](../../../src/architectures/) —
  existing classes to subclass for Part B:
  - `crosscoder.py::TemporalCrosscoder` — vanilla TXCDR (no matryoshka)
  - `matryoshka_txcdr.py::PositionMatryoshkaTXCDR` — position-nested matryoshka
  - `matryoshka_txcdr_contrastive.py::MatryoshkaTXCDRContrastive` — + InfoNCE
  - `matryoshka_txcdr_contrastive_multiscale.py` — agentic_txc_02 base
  - `_batchtopk.py` + `_batchtopk_variants.py` — BatchTopK sparsity layer
- **Training dispatchers**: [`train_primary_archs.py`](../../../experiments/phase5_downstream_utility/train_primary_archs.py)
- **Probe routing**: [`probing/run_probing.py`](../../../experiments/phase5_downstream_utility/probing/run_probing.py)
  — **DO NOT MODIFY** for Part B (that would be reward-hacking)
- **Phase 5.7 agentic log template**: [`2026-04-21-agentic-log.md`](2026-04-21-agentic-log.md)
- **Phase 5.7 architecture reference**: [`2026-04-21-phase5_7-architectures.md`](2026-04-21-phase5_7-architectures.md)
- **Seed variance runner**: `experiments/phase5_downstream_utility/agentic/seed_variance.sh`
- **Analysis helpers**: [`analysis/router.py`](../../../experiments/phase5_downstream_utility/analysis/router.py),
  [`analysis/concat_probe.py`](../../../experiments/phase5_downstream_utility/analysis/concat_probe.py)
  (ready), `analysis/t_scaling_score.py` (write in Part B cycle 1)
- **Current summary.md state** (commit `1458df8`): BatchTopK section
  updated with 3 new extended-scope last_position rows
  (topk_sae/mlc_contrastive/mlc_contrastive_alpha100 batchtopk). Full
  21-arch Δ table + mean_pool rows still need a pass (A4) after probe
  pipeline finishes. BatchTopK T-sweep table + plot added at the
  bottom of §T-sweep section.
- **HF sync script**: `scripts/hf_upload_ckpts.py` (idempotent —
  skip-by-hash; safe to re-run)

## Bottom line

Finish Part A first — the bench becomes reviewer-defensible (seed
variance, full BatchTopK Δ table, concat-probe, consolidated
T-sweep). Then Part B autoresearch until monotonicity ≥ 0.8 with
Δ(30-5) > 0.02 hits, or the hypothesis pool is exhausted.

- **If Part B finds a T-scaling arch**: paper headlines TXC.
- **If Part B exhausts without a hit**: paper pivots to the MLC story
  and submits honestly.

**The evaluation protocol is the protocol. Architectural changes only.**

One continuous session. Don't stop between parts.
