# Research record — hedging-drift synthetic benchmark (architecture test)

**Benchmark:** hierarchical-AR(1) confidence stream (`toy_hedging_drift_d64`),
the DC / slow-drift dynamics class — the grounded-expansion loop's first
**DC-axis** benchmark to face the architectures.
**Spec:** [`bench_spec.md`](bench_spec.md) (frozen expansion C1; `hier_ar1`
amendment 2026-07-19; canonical mirror
[`mirror_params_hier.json`](mirror_params_hier.json)). **Gating:**
[`gating.py`](gating.py) →
[`results/hedging_gating_stats.json`](results/hedging_gating_stats.json)
(PASS). **Grid:** [`run_grid.py`](run_grid.py), the uniform fair-backbone
design through the canonical runner. **This record + figures are
auto-generated** from the canonical leaderboard by
[`render_figs.py`](render_figs.py) — re-run it to rebuild every number, table,
and figure; nothing is hand-typed.

**The design in one line:** the mirror of a *measured* property of real
R1-Distill reasoning traces (expressed confidence persists and drifts
hedged→committed; ACF(1) 0.316 ≫ nulls with a long-memory *plateau* only the
hierarchical-AR(1) mirror reproduces) — a continuous confidence state `c_i`
carried by one feature's *magnitude*, with per-token folded-normal
multiplicative noise; the probe asks how much of `c_i` each code makes
linearly available at the tile's leading edge.

## Headline

<!-- BEGIN AUTO:headline -->
(run render_figs)
<!-- END AUTO:headline -->

![Main result: confidence-recovery frontier](figs/hedging_main.png)

![Recovery vs window size](figs/hedging_T.png)

## 1. Setup

- **Substrate** (spec § 2 + the hier_ar1 amendment): Layer 1 = the canonical
  C3 hierarchical AR(1) — `c_i = μ + β·(i/L) + l_j + r_i`, `r` AR(1) with
  ρ = 0.248, the 210 *empirical* per-trace levels `l_j` (heavy tails
  preserved); gate-8 PASS on ACF(2)+ACF(4) — the generated stream holds the
  plateau (pooled ACF lags 1–8: 0.333, 0.168, 0.123, 0.113, 0.110, 0.108,
  0.108, 0.108). Layer 2 = continuous-loading emission: `u_conf`'s magnitude
  carries `c_i` (per-token folded-normal `m` ⇒ irreducible multiplicative
  noise), + 19 content dirs (`F = 20`). `d_in = 64`, `seq_len = 64`,
  `n_seqs = 4096`, `σ = 0`.
- **Latent + ceilings** (gating, committed): `c_i` continuous; chance = 0
  (pooled mean), spec oracle = 1 (**not reachable** — the multiplicative
  noise bounds any linear reader). Measured raw-linear access ceilings:
  per-token **R² 0.770** (= the `u_conf`-projection ceiling 0.769); raw
  windows **0.774 / 0.778 / 0.776** at T = 2/4/8 — the temporal-denoising
  headroom in raw space is only ≈ +0.005.
- **The structural fact recorded BEFORE the grid** (gating): unlike
  changepoint's AC latent, `c_i` is *linearly present* in the raw window —
  a window gain here can be plain linear access; the untrained-encoder
  control is the access-vs-learning arbiter.
- **Archs:** the BatchTopK fair-backbone family — `batchtopk_sae`, `tsae`
  (per-token), `stacked_batchtopk`, `txc_batchtopk_pre`, `txc_batchtopk_post`,
  `spectral_txc` (windows, `T ∈ {2,4,8}`); equal tokens/step
  (`batch = 1024/T`), equal `B·T = 1024` BatchTopK pool, eval window `L = 32`,
  seeds {1, 2, 42}.
- **Grid:** the locked uniform design — `d_sae ∈ {10, 20, 40}` anchored on
  `F = 20`, `k_pos ∈ {1,2,4,8,16}` (dict-feasible), untrained control per
  `(arch, T)`; 495 cells through the canonical runner.
- **Metric** (per-tile leading-edge ridge probe, memorization-free,
  sequence-split): `conf_recovery` (held-out R², normalized by construction
  to [chance 0, oracle 1]), `conf_corr` companion, + `gauc` (conf dir),
  `eauc` (content dirs), `nmse`.

## 2. Confidence recovery vs capacity

<!-- BEGIN AUTO:conf_frontier -->
(run render_figs)
<!-- END AUTO:conf_frontier -->

## 3. Untrained-encoder control (access vs learning)

![Untrained control](figs/hedging_untrained_control.png)

<!-- BEGIN AUTO:untrained -->
(run render_figs)
<!-- END AUTO:untrained -->

## 4. Sparsity robustness (k_pos)

<!-- BEGIN AUTO:kpos -->
(run render_figs)
<!-- END AUTO:kpos -->

## 5. Capability gate — feature recovery + reconstruction

![Local tradeoff](figs/hedging_local_tradeoff.png)

<!-- BEGIN AUTO:feature_recovery -->
(run render_figs)
<!-- END AUTO:feature_recovery -->

## 6. Frozen predictions vs actual (the blind check)

*(hand-written after the blind grid — nothing here was tuned for)*

| frozen prediction (spec § 5, before any run) | actual | verdict |
|---|---|---|
| per-token SAE captures hedge/commit lexicon per token | — | — |
| per-token SAE misses the slow drift (no cross-sentence state) | — | — |
| medium windows (T = 8+) best capture the persistence | — | — |
| very short windows lose the drift | — | — |

**Verdict:** —
