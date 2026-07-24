# Amendment card — Stage 2, candidate 1: the **budget-matched TXC-post** re-run

**Status: FROZEN at commit (commit-then-run; no matched cell has been
executed when this card is committed — git order is the evidence).**
Agent: runpod-d. Briefing: `briefings/task-hunt-r2-d.md` § 1.
Amends: `card.md` (Stage-1 screen) → the Stage-2 panel run by
`run_stage2.py`, whose results live in
`results/stage2_ward_real_lambda_base_l12.json`. This card changes
**one arch's sparsity budget** and nothing else.

## 1. The defect being repaired

Round-1 Stage 2 ran the whole panel at nominal `k_pos = 8`. For
`txc_batchtopk_post` that budget is spent **per window**, not per
token: `_TXCBatchTopKBase._batchtopk` pools over `B` window rows for
the post variant (`_compute_post` → `(B, d_sae)`), so the shared code
carries ≤ `k_pos` nonzeros for the whole `T`-tile, and the eval's
`l0_per_token = l0_per_window / T` collapses as `T` grows. Verified
from the round-1 rows (`results/stage2_ward_real_lambda_base_l12.json`,
mean over seeds {1, 2, 42}):

| T | nominal k | untrained l0/token | trained l0/token | trained l0/window | realized/nominal | recovery (trained) |
|---|---|---|---|---|---|---|
| 2 | 8 | 4.000 | 2.848 | 5.70 | 0.712 | 0.130 |
| 4 | 8 | 2.000 | 1.603 | 6.41 | 0.802 | 0.161 |
| 8 | 8 | 1.000 | 0.877 | 7.02 | 0.877 | 0.185 |
| 16 | 8 | 0.500 | 0.484 | 7.74 | 0.968 | **0.255** |

The untrained column is **exactly** `8/T` — untrained cells never set
the JumpReLU threshold, so inference still runs the BatchTopK path and
realizes the nominal budget exactly. That exactness is the proof of the
mechanism, not an inference from it. Trained cells sit below nominal
because the inference-time JumpReLU threshold prunes atoms that
BatchTopK would have kept; the shortfall shrinks with `T`.

So round-1's monotone post rise to 0.255 is **budget-confounded**: post
at `T = 16` was spending 0.48 code-atoms per token position while every
other arch in the panel spent 4.5–7.9. The rise is measured along a
sparsity ramp, not at fixed budget.

## 2. What "matched" means here (the convention, stated because it is contested)

Two defensible readings of a shared window code's budget:

- **Code-rate (adopted).** The shared code transmits `k` (index, value)
  pairs to describe `T` positions; the per-position decoder rows are
  learned parameters, not data. Amortized cost = `k/T` per token. This
  is what `_realized_sparsity` measures as `l0_per_token`, and it is
  the convention round 1 already adopted for the whole panel (LOG
  review note 1, the "code-readout convention" + code-rate defense).
- **Activation-count (rejected, but recorded).** `txc_batchtopk.py`'s
  own docstring argues that because each squashed atom is reused at all
  `T` positions, "`k_pos` shared atoms ≈ `k_pos·T` token-activations —
  parity with the per-token archs". Under that reading round 1 was
  already matched and this re-run over-budgets post by `T×`.

We adopt code-rate because parameter-side reuse is not transmitted
information: the decoder is fixed at eval time, so reusing an atom at
`T` positions costs nothing extra to describe. The arch docstring's
parity claim double-counts. **This is a disclosed disagreement with the
arch's own documentation, not a silent reinterpretation** — and it is
why both post columns stay on the record (§ 5).

## 3. The amendment

Set the nominal budget **per T** so the realized code rate matches the
rest of the panel's ≈ 8 atoms per token position:

> **nominal `k_pos` = 8·T → k = 16 / 32 / 64 / 128 at T = 2 / 4 / 8 / 16.**

This **deliberately deviates from the program's equal-nominal-`k_pos`
fairness rule** (`explorations.synthetic.design`, locked uniform grid)
in favour of the briefing's matched-**realized**-`l0` requirement. The
deviation is confined to `txc_batchtopk_post`; every other axis
(d_sae = 2048, eval_window_L = 32, n_steps = 8000,
buffer_tokens = 524288, seeds {1, 2, 42}, datasource
`ward_real_lambda_base_l12`) is byte-identical to round 1. Dictionary
constraint is satisfied: post is not in `design._POOLED`, so it needs
only `d_sae ≥ k_pos`, and 2048 ≫ 128.

**Predicted realized l0 (pre-registered, so the check is falsifiable):**
untrained cells will realize **exactly 8.00** per token at every `T`.
Trained cells will land at `8 ×` the realized/nominal ratio; if that
ratio is preserved from the table above, **5.7 / 6.4 / 7.0 / 7.7** at
T = 2 / 4 / 8 / 16 — the same band TXC-pre occupies (5.81 / 6.81 /
7.79 / 7.84). Actuals are read off the rows and reported; any trained
cell outside **[5.0, 8.0]** is recorded as a residual mismatch and
carried into the reading, not smoothed over.

**Cells:** post × T ∈ {2, 4, 8, 16} × seeds {1, 2, 42} × {trained,
untrained} = **24**. Written to a **separate results file**
`results/stage2_postmatched_ward_real_lambda_base_l12.json` so matched
cells can never silently mix with the round-1 nominal-k = 8 cells; the
renderer merges explicitly and labels by realized l0.

## 4. Pre-registered readings

- **(a) The rise survives matching** — matched post is monotone in `T`
  and its `T = 16` cell is within noise of, or above, 0.255 ⇒ the money
  plot upgrades from "TXC-pre peaks at T = 8" to a monotone
  matched-budget line through T = 16; a materially stronger rebuttal
  figure.
- **(b) The rise does not survive** — matched post falls at large `T`
  (in particular `T = 16` drops toward the pre/stacked level) ⇒ the
  0.255 was **sparsity-starvation behaviour**, recorded as such, and
  **TXC-pre remains the headline**. No claim is made that post is bad;
  the claim is that its round-1 column was not budget-matched and is
  not usable as a win.
- **(c) The drop is probe capacity, not representation** — the
  confound that makes (b) ambiguous, pre-registered here so the
  diagnostic is not chosen after seeing the sign. `lambda_recovery`
  fits an **unregularized** `LinearRegression` on the single-tile code
  (`p = d_sae = 2048` features) with `n = 1024 · (32/T)` rows: **32768 /
  16384 / 8192 / 4096 / 2048 at T = 1 / 2 / 4 / 8 / 16**. At `T = 16`,
  `n = p` exactly. A code with ~8 nonzeros per row survives that
  regime; a code with ~100 nonzeros per row need not. Raising post's
  nominal `k` by `T×` therefore moves it toward the same overfitting
  regime that already suspects the round-1 `T = 16` drops of TXC-pre
  (0.206 → 0.138) and Stacked (0.125 → 0.094), whose realized
  `l0_per_window` at T = 16 are ≈ 125 each.

  **Diagnostic (run either way, labelled post-hoc, kept OUT of the
  leaderboard):** refit the λ probe on the *same* trained checkpoints
  with more probe data (`n_windows` 1024 → 8192, giving 16384 rows at
  T = 16) and with ridge, and report held-out r. If the extra probe
  data lifts the matched `T = 16` cell back toward the round-1 value,
  the drop is probe-limited and the honest statement is that the
  `T = 16` column is probe-limited for **every dense arch in the
  panel**, not that post's representation degrades. This diagnostic
  cannot change the leaderboard cells; it can only change what the
  record is allowed to claim about them.

## 5. What stays on the record either way

Both post columns are reported and labelled — nominal-k = 8 (round 1,
code rate 8/T per token) and matched (this card, code rate ≈ 8 per
token). Deleting either would hide the confound rather than resolve it.
The Stage-2 figure gains the realized-l0 annotation on every post point
(LOG review note 3, mandatory before any external use).

## 6. Falsifier for the amendment itself

If the untrained matched cells do **not** realize l0/token = 8.00
(± 0.02) at every `T`, the `l0 ≈ k/T` mechanism stated in § 1 is wrong,
the per-T `k` above is not the right correction, and the run is void —
reported as a failed amendment, not reinterpreted.
