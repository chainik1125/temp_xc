---
author: Han
date: 2026-04-23
tags:
  - proposal
  - in-progress
---

## Handover: find a T-scaling TXC architecture for the NeurIPS submission

**Audience**: post-compact agent. Previous agent ran out of context
mid-stream.

**Timeline**: **NeurIPS submission in ~2 weeks.** This deadline is load-
bearing for every decision below.

**State at handover**: `han` branch at commit `1c6cd44`. Extended
BatchTopK pipeline is **still probing mean_pool** (17-arch extension).
Monitor via `tail logs/overnight/probe_batchtopk_extend_mean_pool.log`.

---

## The mission

**Find a TXC architecture (or probing protocol) such that sparse-probing
AUC increases with T.** Everything else is secondary.

### Why this is the mission

Our current paper framing — "temporal SAE with multi-scale contrastive" —
implicitly claims longer windows help. What the data shows (see
`summary.md` §T-sweep):

- TopK TXCDR peaks at T=5, **fluctuates** for T ∈ {2, 3, 5, 8, 10, 15, 20}
- BatchTopK TXCDR has a **U-shape** (T=2 and T=20 both good)
- Multi-scale agentic_txc_02 is **flat** across T

**Monotonicity score of our best curve ≈ 0.6** (random = 0.5). Reviewers
will ask: "if AUC doesn't scale with T, why is your arch *temporal*?"
Without a refutation, the paper's central claim collapses.

The other paper-level weaknesses (§"Methodological risks" below) are
fixable in the last few days. **The T-scaling story is the one thing
that needs a research breakthrough, not just more runs.**

### Success criteria

For each candidate arch, train at T ∈ {5, 10, 15, 20, 30} seed=42.
Probe at last_position only (mean_pool is tail=20-limited; see Part B
note below). Report:

1. **Monotonicity score**: fraction of T-pairs (i<j) where auc(T_j) ≥ auc(T_i).
   - Current best: ~0.6 (vanilla TXCDR BatchTopK)
   - **Target: ≥ 0.8**
   - Stretch: strict monotone (1.0)
2. **Δ(T=30 − T=5)**: actual AUC gain from scaling up.
   - Current: ~0.0
   - **Target: > +0.02**
3. **Does it hold at mean_pool** (for T ≤ 20)? Desirable but not required.

If a candidate hits both targets, that's the submission headline.

### If nothing works, pivot

If after 8-10 cycles no arch hits both targets, the **paper pivots to MLC**:

- `agentic_mlc_08` beats vanilla MLC by +0.016 at last_position
  (well above noise; likely p < 0.05 after Bonferroni — needs verification)
- 5/8 semantic autointerp labels in Phase 6 (competitive)
- Honest framing: "multi-scale contrastive on the MLC family is our
  contribution; temporal scaling remains an open question"

This is a worse paper but defensible. **Don't fake T-scaling by
cherry-picking T values or tasks.**

---

## Context the next agent needs

### Methodological risks uncovered 2026-04-23

These are things a NeurIPS reviewer will attack. Most are fixable in
the last few days; one is existential (T-scaling). Ranked by severity:

1. **Current TXC headline Δ is +0.0023 at last_position, within probing
   noise (~0.005).** Every arch in the bench needs 3-seed σ before the
   paper can claim any Δ < 0.01. Only `agentic_txc_02` and
   `agentic_mlc_08` have 3-seed — baselines have seed=42 only.
   → Fix by running `agentic/seed_variance.sh` on txcdr_t5, mlc,
   matryoshka_t5, mlc_contrastive at seeds 1, 2. ~6 hr.
2. **T-scaling is absent** — the existential threat, central to this
   handover. No currently-trained arch has monotonicity > 0.6.
3. **Plateau-stop bias**: models converge at different step counts,
   different total compute. Reviewer will ask "are you sure the
   weaker arch isn't just undertrained?" → either retrain short
   runs to 25k steps or report (steps, final_loss, converged) per arch.
4. **WinoGrande + WSC `max(AUC, 1-AUC)` flip** inflates 2/36 tasks.
   → Remove the flip or move those tasks to supplementary.
5. **SAEs lose to baselines by 12 pp**. Standard SAE-community answer
   (interpretability) needs Phase 6 numbers. Currently Phase 6 says
   TXC qualitatively loses; MLC is mid.
6. **"Family-agnostic" recipe** is based on one cross-family transfer
   (TXC→MLC) with γ=0.5. Should sweep γ on MLC to verify.
7. **Top-k-by-class-sep uses train labels**, is sensitive to train-
   subset. No bootstrap stability check.
8. **TLC has d_sae=8192 vs others' 18432** — half the feature-pool.
   Cannot be directly ranked.

**What a brutal reviewer reads as the current paper:**
> "You invented a TXC arch that barely beats its baseline at sparse
> probing (within noise), loses the interpretability comparison, and
> doesn't scale with context length. Meanwhile your MLC arch is the
> clear winner but you don't headline it because MLC is layer-wise,
> not temporal."

The T-scaling breakthrough is what turns this into a real paper.

---

## Part A — T-scaling autoresearch (primary mission)

Pattern: Karpathy-style agentic loop — hypothesis → arch change →
evaluate → takeaway. Template:
[`2026-04-21-agentic-log.md`](2026-04-21-agentic-log.md).

Log new cycles to `2026-04-24-t-scaling-agentic-log.md`.

### Cycle budget

Target 8-10 cycles in **7-10 days**. Each cycle:

- Engineering: 30-60 min (subclass + dispatcher + probe routing)
- Training: **~2 hr per cycle** (5 T values × ~25 min each)
- Probing: ~5 min (last_position only)
- Writeup: 15 min

**Total per cycle: ~3 hr.** 10 cycles ≈ 30 hr.

### Hypotheses to seed the loop

Ordered by **expected value × ease**. Do H5 first — it's a probing-only
change, testable on existing ckpts in hours.

#### H5 — Positional-probe on existing TXC latents (RUN FIRST)

**Why first:** no retraining. Tests whether the SAE features are
already T-scaling but our single-position probe can't see it.

**Method:** current probe takes latent at position T-1 (last) →
top-k-by-class-sep → LR. New probe: for each feature f, compute
activation at ALL T positions → `z[:, f] : (T,)` per example →
flatten to `(N, T × d_sae)` → top-k across the T × d_sae pool → LR.
Or: attention-pool the T positions per feature via a learned attention
layer. Both are minor modifications to `run_probing.py`'s
`_encode_for_probe`.

**If H5 shows T-scaling on existing ckpts**: the paper headline becomes
"TXC features encode T-context; standard last-position probing
undersells them. With our T-aware probe, AUC scales with T by Δ."
Much stronger paper than any single arch change.

#### H1 — Attention-pooling decoder

Replace per-position `W_dec^(t) : (d_sae, d_in)` with a single shared
decoder accessed via learned cross-attention over T positions. Latent
queries → attention over T position keys → decoded via shared V.
Decoder params: `(d_sae, d_in)` + `(d_sae, d_attn)` keys +
`(d_in, d_attn)` Vs = same size regardless of T. Forces latents to be
position-invariant.

#### H2 — Convolutional encoder

Replace `W_enc : (T, d_in, d_sae)` with a 1-D conv (kernel 3, shared
across T) producing the pre-activation. Makes the encoder
translation-invariant — features don't get position-imprinted. Larger
T → more "data" to compute each feature over.

#### H3 — Log-scale matryoshka

Instead of T nested scales (scale-s reconstructs s-token sub-window),
use **log₂-spaced** scales: 1, 2, 4, 8, 16, 32 token sub-windows. For
T=32: 6 scales total. Keeps decoder param count O(log T · d_in · d_sae)
instead of O(T² · d_in · d_sae) → escapes the OOM ceiling that stopped
our matryoshka at T=10.

#### H4 — Multi-distance contrastive

InfoNCE at **multiple shift distances** simultaneously: shift-1 (local),
shift-⌊T/4⌋, shift-⌊T/2⌋ (long-range). Inverse-distance weighted.
Forces latents to be consistent across both local and long-range
contexts. Handover of 2026-04-22 already queued as Cycle E; run
here explicitly.

#### H6 — State-space / Mamba-style encoder

Replace per-position encoder with a recurrent state-space model
producing features sequentially. Compute O(T) not O(T²). Features
become genuinely sequential — analogous to how LSTMs beat plain
MLPs on long-sequence tasks in pre-transformer NLP.

#### H7 — SVD-spectrum regularizer

Phase 5.7 §"Per-feature decoder SVD" found that vanilla TXCDR at T=20
has 7.5% flatter singular-value spectrum than T=5. Add a penalty that
pushes the spectrum toward Zipfian (not flat). Directly targets the
"over-regularization at large T" failure mode.

#### H8 — Hybrid MLC + temporal read-out

Build TXC with MLC's layer-axis sharing as the encoder + temporal
decoder that takes position-weighted combinations. MLC's multi-layer
structure gave the Phase 5 win at last_position; bringing that into
TXC might capture the "TXC feature use across layers and positions".

#### H9 — Stack winners

After H1-H8, stack the top 2-3 into `agentic_tscale_stack`. If
cumulative monotonicity ≥ 0.9 + Δ(30-5) > 0.03 → paper headline.

### Expected outcomes

Not all hypotheses will work. Likely outcomes:

- H5 (positional-probe): **highest a priori probability**. If TXC
  features do encode T-context, a T-aware probe should reveal it.
- H1, H2: attention/conv priors are well-motivated. Moderate chance.
- H3 (log-matryoshka): low implementation cost, moderate chance —
  at minimum it escapes OOM so we can probe T≥10 with matryoshka.
- H6 (state-space): high effort, uncertain payoff. Skip unless others fail.

### Evaluation per cycle

Immediately after training:

```bash
# Compute monotonicity score + Δ
PYTHONPATH=. .venv/bin/python -c "
from analysis.t_scaling_score import score_arch  # new helper
score_arch('agentic_tscale_01', T_values=[5,10,15,20,30])
"
```

Write `analysis/t_scaling_score.py` on first cycle — ~30 lines,
reads probing_results.jsonl, computes the 2 metrics.

**Commit per cycle.** Don't batch.

### T-sweep caveats for large T

- **T > 20 at mean_pool: blocked.** `acts_anchor` shape is `(N, 20, d)`;
  slide-windowing with T > 20 gives K ≤ 0. Probe at **last_position
  only** for T ∈ {24, 28, 30, 32, 36}. The `_window_at_last` pads
  cleanly.
- **Matryoshka at T ≥ 10 OOMs at d_sae=18432** due to O(T³) decoder
  param scaling. Only H3 (log-matryoshka) resolves this at large T;
  other matryoshka-based hypotheses are capped at T=8.
- **Training buffer is fine** — `resid_L13.npy` has (6000, 128, 2304),
  supports T up to 128.

---

## Part B — Secondary / supporting work (if time permits)

Only do these if Part A finds a T-scaling arch early (Week 1), or
after the paper's headline is locked in. **Do not prioritize these
over Part A.**

### B1. Concat-latent probing (wrap up experiment iv)

Script ready at [`analysis/concat_probe.py`](../../../experiments/phase5_downstream_utility/analysis/concat_probe.py).
Run ~5 min. Adds the missing "concat probe" column to the router
table. Deliverable is the rename
"Complementarity: TXC/MLC routing and concat probing" in summary.md.

### B2. 3-seed variance on baselines + BatchTopK winners

**Critical for headline defense** but listed as secondary here because
(i) it's mechanical (no research), and (ii) the paper story depends on
Part A first — if no T-scaling arch, headline changes and variance
priorities change. Run at the end of Week 1 once headline is set.

Pattern: `experiments/phase5_downstream_utility/agentic/seed_variance.sh`.
Archs that need seeds {1, 2}: `txcdr_t5`, `mlc`, `matryoshka_t5`,
`mlc_contrastive`, `agentic_txc_02_batchtopk`, `agentic_mlc_08_batchtopk`.
~6 hr total.

### B3. BatchTopK inference-threshold sanity check

Some BatchTopK archs regressed −0.02 AUC vs TopK; may be EMA
threshold miscalibration. Quick check on `agentic_txc_02_batchtopk`:
forward in eval mode, confirm per-sample nonzero rate ≈ k/d_sae.
If miscalibrated, recalibrate threshold from unlabeled batch's
`(d_sae − k)`-quantile and re-save ckpt.

### B4. Extended T-sweep to T ∈ {24, 28, 32, 36}

**Only do if Part A reveals an arch worth extending.** For the current
T=2-20 vanilla sweep, T=30 is useful but don't waste compute on
T=24, 28, 32 if the trend is clear from T=5, 10, 15, 20, 30.

### B5. Consolidate T-sweep section in `summary.md`

See previous handover commit `1c6cd44` Part C for the matrix-table
template. Do this **last**, once the final T-sweep data is in — one
consolidation pass, not two.

### B6. Full 21-arch TopK-vs-BatchTopK Δ table + paired plots

Once extended BatchTopK mean_pool probing completes (still running
as of this handover), regenerate:

```bash
PYTHONPATH=. .venv/bin/python experiments/phase5_downstream_utility/plots/make_batchtopk_plot.py
PYTHONPATH=. .venv/bin/python experiments/phase5_downstream_utility/plots/make_headline_plot.py
PYTHONPATH=. .venv/bin/python experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep_batchtopk.py
```

HF sync: `HF_HOME=/workspace/hf_cache .venv/bin/python scripts/hf_upload_ckpts.py`.

---

## Resume checklist

1. `git log --oneline | head -10` — confirm at `1c6cd44` or later.
2. Confirm extended BatchTopK probe pipeline status:
   `ps -ef | grep run_probing` — may still be running mean_pool.
3. Read this doc + `summary.md` §T-sweep (lines 382, 449, 499).
4. Read Phase 5.7 agentic log pattern:
   [`2026-04-21-agentic-log.md`](2026-04-21-agentic-log.md).
5. **Start with H5 (positional-probe)** — it's a no-retrain experiment
   that could answer the paper's central question in a day.
6. If H5 works: build the paper around it. If not: start H1, H2, H3 in
   parallel.

## Key files (cheat sheet)

- **Architectures**: [`src/architectures/`](../../../src/architectures/)
  — existing matryoshka variants live here; subclass for H1-H9
- **Training dispatchers**: [`experiments/phase5_downstream_utility/train_primary_archs.py`](../../../experiments/phase5_downstream_utility/train_primary_archs.py)
- **Probe routing**: [`experiments/phase5_downstream_utility/probing/run_probing.py`](../../../experiments/phase5_downstream_utility/probing/run_probing.py)
  — H5 lives here: new path in `_encode_for_probe` for position-aware encoding
- **Phase 5.7 agentic log (template)**: [`2026-04-21-agentic-log.md`](2026-04-21-agentic-log.md)
- **T-sweep plot script**: [`plots/plot_txcdr_t_sweep_batchtopk.py`](../../../experiments/phase5_downstream_utility/plots/plot_txcdr_t_sweep_batchtopk.py)
  — extend to show new candidates' T-sweep curves
- **Analysis helpers**: [`analysis/`](../../../experiments/phase5_downstream_utility/analysis/)
  — `router.py`, `concat_probe.py` exist; add `t_scaling_score.py`

## Bottom line

**You have 2 weeks. T-scaling is the research breakthrough we need;
everything else is mechanical. Try H5 first (cheap, high-EV). If
nothing in H1-H9 works, the paper pivots to MLC — the MLC story is
already defensible and you should ship that rather than overclaim TXC.**
