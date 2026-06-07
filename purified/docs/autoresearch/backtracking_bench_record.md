# Research record — backtracking synthetic benchmark (architecture test)

**Verdict: POSITIVE.** On the self-exciting (Hawkes-style) backtracking mirror,
**window/temporal architectures recover the hidden order-sensitive intensity
`λ` far better than per-token SAEs — robustly across the scarce capacity regime
— while per-token archs sit exactly at their provable information floor.** This
is stage 6 of autoresearch investigation #1; it closes the loop
*measured real property → validated synthetic mirror → architecture test*.

Preregistration / spec (frozen before running):
[`backtracking_bench_spec.md`](backtracking_bench_spec.md). Gating:
[`backtracking_gating_stats.json`](backtracking_gating_stats.json). Built to
[`synthetic_benchmark_guidance.md`](../synthetic_benchmark_guidance.md) and the
loop's [`autoresearch_spec.md`](../autoresearch_spec.md).

> **Headline.** A per-token SAE *provably* cannot read the self-exciting
> intensity beyond `corr = √(Var λ/Var b) ≈ 0.41` (data-processing inequality);
> trained per-token archs land at **0.40**, flat across every dictionary size.
> A window crosscoder, which sees the event *history* that sets `λ`, recovers it
> at **0.87 (T=2) → 0.95 (T≥4)** — and this holds even at `d_sae = 8 < F = 20`,
> so it is not an over-completeness artifact. The same window archs *trade away*
> local-feature recovery (eAUC) and tight reconstruction to do it: a clean
> global(temporal)-vs-local(per-token) architectural specialization.

## 1. Setup

- **Data:** `toy_backtracking_selfexcite_d64` — the validated self-exciting
  mirror (`synthetic.py:self_exciting`). `λ_i = σ(a + Σ_{l=1..K} w_l·b_{i-l})`,
  `b_i ~ Bernoulli(λ_i)`; `K = 2` (held-out model-selected), `w = α·κ`,
  `κ = exp(-l/τ)`, `τ = 2`, `α = 3.06`, intercept tuned to base rate 0.12,
  trend = 0. Emission over `F = 20` orthonormal directions
  (`u_bt` dominant + 19 content, `n_c = 3`, `σ = 0`). `seq_len = 64`,
  `n_seqs = 4096`. The intrinsic ceilings of the *generated* data match the
  gating exactly (per-token 0.408, window full-history 0.985).
- **Grid (120 cells, 0 failures):** archs `{topk_sae, tsae}` (per-token, `T=1`)
  vs `{txc_base, stacked_sae}` (window, `T ∈ {2,4,8}`); `d_sae ∈ {8,16,20,40}`
  anchored on `F = 20` (scarce `{8,16,20}` + over-complete `{40}`); `k_pos = 1`;
  common tiled eval window `L = 32`; seeds `{1,2,42}`; 30k steps. Plus an
  **untrained-encoder control** (`n_steps = 0`) at `d_sae = 20`, all archs/`T`,
  3 seeds (24 cells). Everything through the canonical runner, code-version
  stamped, flock-safe parallel append.
- **Metrics:** `lambda_recovery` (headline) — held-out Pearson `corr` of a
  **linear** regression probe on per-tile codes predicting `λ` at each tile's
  leading edge (memorization-free: features = one tile's `d_sae` code, never
  concatenated); `eauc` (local feature-direction cosine recovery); `nmse`
  (tiled windowed reconstruction); `lambda_chance` (shuffle-label floor).

## 2. Headline result — `λ` recovery vs capacity

Trained `lambda_recovery` (mean over 3 seeds), by `d_sae`:

| arch / T | d_sae=8 | 16 | 20 | 40 |
|---|---|---|---|---|
| topk_sae (per-token) | 0.401 | 0.399 | 0.398 | 0.397 |
| tsae (per-token) | 0.416 | 0.412 | 0.409 | 0.417 |
| **txc_base T=2** | 0.870 | 0.830 | 0.868 | 0.858 |
| **txc_base T=4** | **0.951** | 0.949 | 0.948 | 0.940 |
| **txc_base T=8** | 0.950 | 0.949 | 0.949 | 0.947 |
| stacked_sae T=2 | 0.869 | 0.865 | 0.863 | 0.862 |
| stacked_sae T=4 | 0.947 | 0.943 | 0.942 | 0.941 |
| stacked_sae T=8 | 0.947 | 0.941 | 0.941 | 0.938 |

- **Per-token is pinned at the DPI floor** (~0.40 ≈ 0.408) and **flat across all
  `d_sae`** — exactly the prediction: no dictionary size lets a per-token
  encoder see the history, so capacity is irrelevant.
- **Window recovers `λ` at 0.86–0.95**, ~2.3× the per-token floor, and the win
  **holds in the scarce regime** (`d_sae = 8`: txc T4 = 0.951). The realistic-
  regime gate passes — the advantage is architectural, not over-completeness.
- `lambda_chance` ≈ 0 for every cell (worst −0.13, sampling noise), so the
  recovery is real, not an inflated floor.

Figure: [`figs/backtracking_lambda_frontier.png`](figs/backtracking_lambda_frontier.png).

## 3. `λ` recovery rises with `T`, saturates by `T = 4`

At `d_sae = 20`: per-token (T=1) 0.40 → T=2 0.86 → T=4 0.94 → T=8 0.94. The
curve tracks the gating linear ceilings (0.41 / 0.91 / 0.99 / 0.99) and
**saturates by `T = 4`** — because `K = 2`, a tile of `T = 4` already contains
both relevant lags. `T = 8` confirms the plateau (no further gain), as the spec
anticipated. Figure: [`figs/backtracking_lambda_vs_T.png`](figs/backtracking_lambda_vs_T.png).

## 4. The global-vs-local trade-off (eAUC, NMSE)

Window archs buy `λ` recovery by **spending capacity on temporal structure** at
the cost of local-feature recovery and reconstruction:

| eAUC (trained) | d8 | 16 | 20 | 40 |
|---|---|---|---|---|
| tsae (per-token) | 0.404 | 0.768 | **0.976** | **0.990** |
| topk_sae (per-token) | 0.426 | 0.523 | 0.471 | 0.444 |
| txc_base T=4 | 0.334 | 0.513 | 0.590 | 0.925 |
| txc_base T=8 | **0.080** | 0.462 | 0.582 | 0.870 |

- **tsae recovers the local feature directions almost perfectly** (eAUC 0.98 at
  `d_sae = F`) while being `λ`-blind — the per-token "local" specialist.
- The **window crosscoder trails on local recovery in the scarce regime** (txc
  T8 at `d_sae = 8`: eAUC 0.08) yet still gets `λ = 0.95` — a striking
  dissociation: the *code* linearly carries the event history even when no
  single *decoder atom* aligns to `u_bt`. eAUC recovers as `d_sae` grows (0.93
  at `d_sae = 40`), i.e. once it can afford both.
- **NMSE:** per-token reconstructs tighter (≈0.39 at `d_sae = 20`) than window
  (txc T8 ≈ 0.63) — the reconstruction cost of temporal coding. All archs *do*
  reconstruct (NMSE 0.38–0.69), so the capability-vs-artifact gate passes: the
  window recovers `λ` while still representing the data, not instead of it.

Figure: [`figs/backtracking_eauc_nmse.png`](figs/backtracking_eauc_nmse.png).

## 5. Untrained-encoder control — access vs learning

At `d_sae = 20`, trained vs random-init (`n_steps = 0`) `lambda_recovery`:

| arch / T | untrained (access) | trained (access+learning) |
|---|---|---|
| topk_sae (per-token) | 0.304 | 0.398 |
| tsae (per-token) | 0.336 | 0.409 |
| txc_base T=2 | 0.622 | 0.868 |
| txc_base T=4 | 0.728 | 0.948 |
| stacked_sae T=4 | 0.772 | 0.942 |

The window advantage **does not vanish at random init** — a random window
projection already exposes the history linearly (access ≈ 0.62–0.77, well above
the per-token floor), because the architecture genuinely *has* the history. The
honest decomposition: the per-token→window gap is **architectural access**
(provable: per-token can't see history, window can), and **training sharpens the
linear exposure** by a further ~0.2 (0.73 → 0.95 at T=4). Per-token, by
contrast, barely moves with training (0.30 → 0.40) because it is capped at the
DPI floor regardless. Figure:
[`figs/backtracking_untrained_control.png`](figs/backtracking_untrained_control.png).

## 6. Preregistered predictions (spec § 7)

- **P1 — per-token ≈ information ceiling:** ✅ 0.40 vs ceiling 0.408, flat in
  `d_sae`.
- **P2 — window rises with `T`, ~oracle by `T ≈ K`:** ✅ 0.86→0.95, saturates by
  `T = 4` (the code does linearly expose the history, as the logit-linear latent
  predicted).
- **P3 (headline) — window > per-token by the history margin:** ✅ gap ≈ 0.5,
  robust across `d_sae` and both window families.
- **P4 — per-token recovers `F` well, window trails locally:** ✅ tsae eAUC 0.98;
  window eAUC collapses in the scarce/large-`T` corner.
- **Possible negative (window fails to *linearly* expose history):** did not
  occur — but note the *eAUC dissociation* (§4): the window exposes `λ` in its
  code without atom-level `u_bt` alignment, a refinement of "represents the
  events."

## 7. Validity controls passed (spec § 3 / § 6)

- **Provable floor (not pattern-count budget):** per-token DPI ceiling 0.41,
  hit exactly — independent of `d_sae`/probe. The discriminating baseline is
  floored by information, not memorization (`K = 2` ⇒ only 4 history patterns,
  so the old `2^K` budget is moot; the DPI floor is the correct, stronger gate).
- **Memorization-free probe:** per-tile-as-example, features = one tile's
  `d_sae` code; `lambda_chance ≈ 0`.
- **Realistic regime:** the win holds at `d_sae ≤ F` (incl. `d_sae = 8`).
- **Untrained-encoder control:** run and reported (§5) — the advantage is part
  access (genuine, by architecture) + part learning; per-token gains nothing
  from training beyond its floor.
- **Capability-vs-artifact:** window archs reconstruct (NMSE finite) and recover
  features at adequate `d_sae` — they recover `λ` *and* represent the data.
- **Apples-to-apples:** identical `L = 32` tiling, equal `d_sae`/`k_pos` across
  archs, leak-free per-sequence split.

## 8. Caveats (honest scope)

- **Synthetic mirror, weak-validated.** The bench tests architectures on a
  generator fit to the *measured* self-excitation signature
  ([`backtracking_record.md`](backtracking_record.md)), not on real CoT
  activations. Fidelity is the matched-statistic level (spec § 2.5 weak), not a
  trained-on-real-vs-synthetic equivalence.
- **Access ≠ pure learning.** The window's edge is substantially architectural
  access (untrained window already > per-token). This is the *point* (a window
  can see history, a per-token provably can't), but it means the headline is
  "architecture that can see history wins," not "the SAE learned something a
  per-token couldn't be handed."
- **`u_bt` made dominant by design** so both arch families recover the event at
  `k_pos = 1`; the gap is therefore purely about history access, not feature
  saliency. A weaker `u_bt` would confound recovery with detection.
- **Small discrete latent.** `K = 2` ⇒ `λ` takes 4 values; recovery is a coarse
  (but provable-floor-anchored) corr. A longer kernel was rejected by held-out
  NLL (it overfit), so this is the faithful regime, not a convenience.

## 9. What this buys + next

This is the **first positive architecture result** in the autoresearch loop (the
signed-motion AC bench was a documented negative): a real-language property
(self-exciting backtracking), mirrored synthetically with a provable per-token
floor, on which window/temporal dictionaries demonstrably recover the
order-sensitive latent and per-token dictionaries provably cannot. It is the
order-sensitive (AC) companion to the existing aggregation (DC) benches, and the
clean substrate for the global-vs-local specialization narrative. Next: the
change-point/sticky-dwell class via topic-switching
([`changepoint_bench_spec.md`](changepoint_bench_spec.md)).

## 10. Reproduction

```bash
cd purified/
# gating (per-token vs window ceilings) + kernel-length selection
.venv/bin/python -m experiments.autoresearch.backtracking_gating
.venv/bin/python -m experiments.autoresearch.backtracking_kernel_order
# the 120-cell grid (parallel) + figures
.venv/bin/python -m experiments.autoresearch.run_backtracking_grid 6
.venv/bin/python -m experiments.autoresearch.render_backtracking_figs
```
Generator `temp_bench.data.synthetic:self_exciting`; evaluator add-on
`temp_bench.evals.lambda_recovery` (dispatched from `SyntheticRecovery` when
`extra['lambda_labels']` is present; protocol unchanged at 1.2.0). Outputs:
`backtracking_grid_results.json`, `backtracking_bench_stats.json`,
`figs/backtracking_*`. Deterministic per cell (training seed re-materializes the
ground truth). No `core/` edits.
