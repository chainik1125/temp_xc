# Research record — assumption→consequence synthetic benchmark (architecture test)

**Benchmark:** directed-grammar 3-state discourse chain
(`toy_assumption_consequence_d64`), the AC / directed-transition dynamics
class — **the first benchmark discovered by the grounded-expansion loop to
face the architectures** (measure→mirror→**bench** closed).
**Spec:** [`bench_spec.md`](bench_spec.md) (frozen expansion C1; g7 amendment
2026-07-14; canonical mirror
[`mirror_params_g7.json`](mirror_params_g7.json)). **Gating:**
[`gating.py`](gating.py) →
[`results/assumption_gating_stats.json`](results/assumption_gating_stats.json)
(PASS). **Grid:** [`run_grid.py`](run_grid.py), the uniform fair-backbone
design through the canonical runner. **This record + figures are
auto-generated** from the canonical leaderboard by
[`render_figs.py`](render_figs.py) — re-run it to rebuild every number, table,
and figure; nothing is hand-typed.

**The design in one line:** the mirror of a *measured* property of real
R1-Distill reasoning traces (assume-before-derive; strict-labeler directed
asymmetry 0.297 ≫ nulls) — a 3-state {N, A, C} Markov chain whose directed
A→C edge is the latent under test: the DC probe reads the *current* discourse
state, the AC probe asks whether the code supports *next-state* prediction
above the marginal (the direction of the grammar).

## Headline

<!-- BEGIN AUTO:headline -->
(run render_figs)
<!-- END AUTO:headline -->

![Main result: state and next-state frontiers](figs/assumption_main.png)

![Recovery vs window size](figs/assumption_T.png)

## 1. Setup

- **Substrate** (spec § 2 + the g7 amendment): Layer 1 = the canonical g7
  Markov mirror — `P[A→C] = 0.363` vs unconditional C rate 0.294, fit on 207
  strict-labeled (ctx=0) traces, held-out validated, gate-8 PASS. Layer 2 =
  the standard emission over `F = 20` orthonormal directions: 3 dominant
  state-signatures (mag 2.5 → `hidden_features`/gAUC) + 17 content (mag 1.0,
  `n_c = 3`, state-independent → `emission_features`/eAUC). `d_in = 64`,
  `seq_len = 64`, `n_seqs = 4096`, `σ = 0`.
- **Latents + ceilings** (gating, committed):
  | latent | type | chance | oracle | per-token readout | raw window readout |
  |---|---|---|---|---|---|
  | state `s_i` | DC | 1/3 (balanced) | 1.0 | **1.000** (noiseless probe) | 0.999 |
  | next state `s_{i+1}` | AC-directed | 1/3 (balanced) | 0.544 (Bayes-balanced of the one-step conditional) | **0.464** | 0.466–0.467 (T=2/4/8) |
- **The structural fact recorded BEFORE the grid** (gating): the mirror is
  order-1, so `s_i` is a *sufficient statistic* for `s_{i+1}` — per-token and
  raw-linear window readouts are identical (0.464 vs 0.466 balacc). Unlike
  backtracking (DPI floor) or changepoint (equality-pattern blindness), this
  substrate has **no information-theoretic per-token/window separation**; the
  grid adjudicates what *trained scarce codes* expose linearly at the tile's
  leading edge. (The gap between the 0.466 readout and the 0.544 oracle is
  the class-unweighted logistic probe convention, uniform across archs.)
- **Archs:** the BatchTopK fair-backbone family — `batchtopk_sae`, `tsae`
  (per-token), `stacked_batchtopk`, `txc_batchtopk_pre`, `txc_batchtopk_post`,
  `spectral_txc` (windows, `T ∈ {2,4,8}`); equal tokens/step
  (`batch = 1024/T`), equal `B·T = 1024` BatchTopK pool, eval window `L = 32`,
  seeds {1, 2, 42}.
- **Grid:** the locked uniform design — `d_sae ∈ {10, 20, 40}` anchored on
  `F = 20`, `k_pos ∈ {1,2,4,8,16}` (dict-feasible), untrained control per
  `(arch, T)`; 495 cells through the canonical runner.
- **Metrics** (per-tile leading-edge linear probes, memorization-free,
  sequence-split): `state_recovery` (multinomial, normalized to [1/3, 1]),
  `nextstate_recovery` (multinomial → `s_{i+1}`, normalized to [1/3, the
  sample-matched Bayes-balanced oracle]), + `gauc` (state dirs), `eauc`
  (content dirs), `nmse` — direction sets never pooled.

## 2. DC half — state recovery vs capacity

<!-- BEGIN AUTO:state_frontier -->
(run render_figs)
<!-- END AUTO:state_frontier -->

## 3. AC half — next-state (directed) recovery vs capacity

<!-- BEGIN AUTO:nextstate_frontier -->
(run render_figs)
<!-- END AUTO:nextstate_frontier -->

## 4. Untrained-encoder control (access vs learning)

![Untrained control](figs/assumption_untrained_control.png)

<!-- BEGIN AUTO:untrained -->
(run render_figs)
<!-- END AUTO:untrained -->

## 5. Sparsity robustness (k_pos)

<!-- BEGIN AUTO:kpos -->
(run render_figs)
<!-- END AUTO:kpos -->

## 6. Capability gate — feature recovery + reconstruction

![Local tradeoff](figs/assumption_local_tradeoff.png)

<!-- BEGIN AUTO:feature_recovery -->
(run render_figs)
<!-- END AUTO:feature_recovery -->

## 7. Frozen predictions vs actual (the blind check)

*(hand-written after the blind grid — nothing here was tuned for)*

| frozen prediction (spec § 5, before any run) | actual | verdict |
|---|---|---|
| per-token SAE captures the A/C connective features per token | — | — |
| per-token SAE is blind to the directed A→C dependency across sentences | — | — |
| window families (TXC-pre/-post / Stacked / Spectral) expose the order-sensitivity | — | — |
| additive (pre-squash) families weaker on the directed latent | — | — |

**Verdict:** —
