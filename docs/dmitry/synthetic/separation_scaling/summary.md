---
author: Dmitry Manning-Coe
date: 2026-05-26
tags:
  - results
  - complete
---

## Separation-scaling: MatTXC as a non-ergodic component detector vs the Bayes ceiling

Consolidated summary of the `experiments/separation_scaling/` study, which
benchmarks SAE/crosscoder-family architectures on their ability to recover the
hidden non-ergodic *component identity* from a transformer's residual stream,
measured against a closed-form Bayes-optimal ceiling. The headline finding is
that the **MatryoshkaTXC (MatTXC) recovers ~100% of the Bayes ceiling** at the
most-separated cell — with important nuance, documented below, once the metric
definition and seed variance are pinned down.

This doc pulls together five source artifacts that live on other branches:

- `tables/separation_scaling.md` — headline 7-arch × 5-δ table (`origin/dmitry`, commit `16452d5`, 2026-04-20)
- `README.md` — design + terse "Headline" prose (`origin/dmitry`)
- `NOTES_r2_ceiling.md` — derivation of the Bayes ceiling R²_max (`origin/dmitry`)
- `tables/probe_window_sweep.md` — dense-probe R² vs window width (`origin/dmitry-rlhf`)
- `tables/separation_scaling_tsae_paper.md` — protocol sweep, head-to-head, multi-seed (`origin/dmitry-rlhf`)

Plus the independent replication: `docs/aniket/experiments/mess3_mat_ablation/summary.md` (2026-04-22).

## README framing (`README.md`)

The experiment's own one-paragraph framing, for orientation before the data:

> How do SAE/crosscoder-family architectures scale as we grow the generator's
> temporal-structure demand? Benchmarks 7 architectures across
> δ ∈ {0, 0.05, 0.1, 0.15, 0.2} at r=0 (shared mess3 vocab, no tag tokens),
> using a YAML-driven runner.

- **Archs benchmarked**: TopK SAE, TXC, MatryoshkaTXC, MultiLayerCrosscoder,
  TFA (Han), TFA-pos, Temporal BatchTopK SAE — plus dense linear and
  (early-stopped) dense MLP probes on the residual stream.
- **Provenance**: a copy-paste port from `sae_day`; the full dependency graph
  (`NonergodicGenerator`, SAE classes, TFA, driver) is frozen under `vendor/`.
- **Reproduce**: `uv sync --extra separation-scaling`, then
  `python -m sae_day.run_driver --config config.yaml` from
  `experiments/separation_scaling/`. ~95 min on an A40 (5 cells × ~19 min);
  `transformer.load_if_exists: true` reuses cached transformers on reruns.
- **Ceiling tooling**: `compute_r2_ceiling.py` (Bayes-optimal R² from the
  forward filter), `run_window_probe.py` (dense linear + logistic probes at
  window W), `run_ridge_sweep.py` (ridge λ sweep for W ∈ {20, 30, 60}).

The README's own **Headline** prose (verbatim):

> At δ=0.20 (τ=0.60), MatTXC recovers best_single=0.87 on its best component vs
> TopK's 0.14 — a 0.73 gap. TFA and TFA-pos come close to TXC at high δ
> (≈0.38 vs 0.42) but under-perform crosscoders at intermediate δ=0.15 (0.10,
> 0.16 vs TXC 0.49).

Note the **0.87** quoted here is the *protocol-tuned* MatTXC best-single number,
not the **0.421** main-config value in the headline table below (see Variant 1
and the reconciliation table). The README prose and the headline table are
quoting different MatTXC configurations.

## Shared setup (all variants)

- **Generator**: `NonergodicGenerator`, `mess3_shared`, `r=0` (shared vocab, no
  tag tokens). Each sequence commits at start to one of 3 components
  `C ∈ {0, 1, 2}` (uniform prior); given `C` it emits tokens from the
  corresponding mess3 HMM. The component label is *never observed* — it must be
  inferred from the token stream.
- **Transformer**: σ=1e-3 init, `d_model=64`, `n_ctx=128`, 20k next-token-predict
  steps. Activations read at `blocks.1.hook_resid_post`.
- **Sweep axis**: temporal separation `δ ∈ {0, 0.05, 0.10, 0.15, 0.20}`. Larger δ
  pushes the three components apart, raising the temporal-structure demand
  `τ(δ)` and the information about `C` that the sequence carries.
- **Probe / metric**: a probe maps an activation to `g(X) ∈ R³`, trained against
  the `one-hot(C)` target under MSE; recovery reported as `R²`. "best single
  feature" picks the single SAE/crosscoder latent whose linear probe best
  recovers a component; "all" uses a linear probe on the full latent vector.
- **Seed**: 42 for all headline tables (single seed — see the multi-seed caveat).

## The Bayes ceiling (R²_max) — what "100%" is measured against

From `NOTES_r2_ceiling.md`. Because the `one-hot(C)` target is a *stochastic
draw* from the posterior `P(C | X)`, even a probe that reproduces the exact
posterior cannot reach R²=1 — there is irreducible Bernoulli variance. The
closed-form ceiling is

```text
R²_max = 1 − E_X[Var(one-hot(C) | X)] / Var(one-hot(C))
```

reached exactly when `g(X) = P(C | X)`. Computed from the exact Bayesian forward
filter on each δ-cell generator (no transformer involved):

| δ | τ (bits) | R²_max (mean over t) | R²_max (t = T−1) |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.00 | 0.00 | 0.00 |
| 0.05 | 0.12 | 0.07 | 0.12 |
| 0.10 | 0.37 | 0.22 | 0.36 |
| 0.15 | 0.56 | 0.36 | 0.51 |
| 0.20 | 0.60 | 0.45 | 0.54 |

Two subtleties that matter for reading every "% of ceiling" number below:

- **The ceiling is per-component asymmetric at δ=0.20.** comp-0 (sticky:
  x=0.05, a=0.8) has R²_max = **0.866 mean / 0.998 final**; comp-1 and comp-2
  are only ~0.31. A `best_single_r2` metric tracks comp-0 in practice, so
  "gap recovered" for best-single features is computed against **0.866**, while
  the aggregate `all`-features probe is compared against the **mean ceiling 0.45**.
- **τ is not R²_max.** τ=0.60 at δ=0.20 is the *entropy* ceiling; the *variance*
  (R²) ceiling is 0.45. The earlier τ=0.60 reference line overstated the gap.

## Variant 1 — Headline 7-arch sweep (`tables/separation_scaling.md`)

The canonical 5-δ × 7-arch table. Archs: TopK SAE, TXC (window 30, 3000
temporal steps), MatTXC (matryoshka_widths [8,16,32,64,128], `fixed_k_total=10`,
window 60, 1500 temporal steps), MultiLayerCrosscoder (MLxC), TFA, TFA-pos,
Temporal BatchTopK SAE; plus dense linear and dense MLP probes on the raw
residual. Values are best-single-feature R² (`best`) and full-latent R² (`all`):

| δ | R²_max | dense-lin | dense-MLP | TopK best/all | TXC best/all | MatTXC best/all |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.00 | −0.015 | −0.031 | −0.001/−0.009 | −0.001/−0.021 | −0.010/−0.027 |
| 0.05 | 0.07 | −0.007 | −0.018 | −0.001/−0.002 | −0.000/−0.002 | −0.012/−0.033 |
| 0.10 | 0.21 | +0.026 | +0.042 | 0.002/0.016 | 0.030/0.047 | −0.000/0.039 |
| 0.15 | 0.36 | +0.261 | +0.315 | 0.050/0.245 | 0.244/0.356 | 0.263/0.350 |
| 0.20 | 0.45 | +0.361 | +0.416 | 0.069/0.332 | 0.209/0.434 | **0.421/0.449** |

The **`all`-features reading of "100%"**: MatTXC's full-latent probe reaches
**0.449 against the mean ceiling 0.45 ≈ 99.8%**. `NOTES_r2_ceiling.md` confirms
the MatTXC linear-on-latents probe "hits 0.45 precisely" — because its window-60
latents integrate evidence across all positions, exactly what the mean-over-t
ceiling integrates over. By comparison the best *position-agnostic* dense probe
(window-30 linear) reaches 0.42 = 94% of 0.45.

Caveat on this table's **best-single** MatTXC number (0.421): this is the
*main-config* MatTXC, which is **not** the config that maximises single-feature
R² — see Variant 4. The README "Headline" prose quotes best_single ≈ 0.87 at
δ=0.20, which corresponds to the *protocol-tuned* MatTXC, not the 0.421 in this
table. The two are different configurations; do not treat 0.87 and 0.421 as the
same measurement.

## Variant 2 — Probe window sweep (`tables/probe_window_sweep.md`)

Dense linear probe on `flat(x_{t:t+W})` as a function of window width W — an
architecture-free reference for how much component information a plain linear
read-off can extract once it sees multiple positions. R² per (δ, W):

| δ | W=1 | W=5 | W=10 | W=20 | W=30 | W=60 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | +0.026 | +0.060 | +0.076 | +0.080 | +0.075 | +0.008 |
| 0.15 | +0.261 | +0.295 | +0.306 | +0.305 | +0.284 | +0.196 |
| 0.20 | +0.361 | +0.399 | +0.413 | **+0.420** | +0.417 | +0.399 |

At δ=0.20 the dense linear probe peaks at W=20 (0.420 ≈ 93% of the 0.45 mean
ceiling) and falls off by W=60 — i.e. a naive flatten-the-window probe cannot by
itself saturate the ceiling; the crosscoder latents do better at W=60 because
they compress rather than concatenate positions.

## Variant 3 — Protocol sweep + head-to-head + multi-seed (`tables/separation_scaling_tsae_paper.md`)

The richest findings narrative, on `origin/dmitry-rlhf`. Three layers:

**(a) Protocol sweep — MatTXC single-feature R² is config-dependent.** The
headline table's main-config MatTXC (0.421) under-sells the architecture. Tuning
the sparsity method/budget recovers far more of the comp-0 ceiling (0.866):

| MatTXC config | δ=0.15 single | δ=0.20 single | gap recovered (δ=0.20) |
| --- | ---: | ---: | ---: |
| batchtopk k=10 | 0.43 | 0.81 | 93.5% |
| batchtopk k=20 | 0.51 | 0.80 | 92.4% |
| topk baseline | 0.57 | 0.81 | 93.5% |
| batchtopk k=4 | 0.59 | 0.73 | 84.3% |
| main-config (headline table) | — | 0.421 | 48.6% |

**(b) T-SAE paper-faithful + head-to-head (single seed 42).** The original
"Temporal BatchTopK SAE" entry used `group_fractions=[0.5, 0.5]` + 2k steps,
which makes the matryoshka pressure trivial — effectively BatchTopK + a temporal
contrastive term. The paper-faithful arch uses `[0.2, 0.8]` + 25k steps. On
identical activations:

| δ | MatTXC bk10/topk | T-SAE Paper | comp-0 ceiling | MatTXC gap | T-SAE gap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15 | 0.564 | 0.285 | 0.730 | 77% | 39% |
| 0.20 | **0.851** | 0.611 | 0.866 | **98%** | 71% |

At seed=1 MatTXC hit **0.881 = 102% of the comp-0 ceiling**. This is the
**best-single reading of "100%"**: protocol-tuned MatTXC saturates the comp-0
Bayes ceiling at the most-separated cell.

**(c) Multi-seed variance (seeds 42 / 1 / 2) — the honest caveat.** Retraining
the transformer from scratch per seed shows seed-to-seed variance dwarfs the
architectural gap. Best single-feature R² (and % of comp-0 ceiling):

| arch | δ | seed=42 | seed=1 | seed=2 | mean ± std | mean % ceil |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TXC | 0.15 | 0.549 | 0.111 | 0.435 | 0.365 ± 0.227 | 50% |
| TXC | 0.20 | 0.453 | 0.270 | 0.135 | 0.286 ± 0.160 | 33% |
| T-SAE Paper | 0.15 | 0.285 | 0.049 | 0.280 | 0.205 ± 0.135 | 28% |
| T-SAE Paper | 0.20 | 0.611 | 0.159 | 0.083 | 0.284 ± 0.285 | 33% |
| **MatTXC** | 0.15 | 0.564 | 0.386 | 0.576 | **0.509 ± 0.106** | **70%** |
| **MatTXC** | 0.20 | 0.851 | 0.881 | 0.162 | **0.632 ± 0.407** | **73%** |

Takeaways from this layer:

- **MatTXC is the clear winner** — ~70–73% of comp-0 ceiling on average vs
  28–50% for TXC and T-SAE Paper.
- **MatTXC is bimodal at δ=0.20**: 2/3 seeds hit ~98–102% of ceiling, but seed=2
  collapses to 0.162 (19%). So the flat "100%" is its best-case, not its mean.
- **The single-seed T-SAE Paper 0.611 was a fluke** (mean across 3 seeds = 0.284);
  TXC and T-SAE Paper are statistically tied at δ=0.20.
- Open follow-up: understand the seed=2 δ=0.20 MatTXC collapse (TXC is also low
  there at 0.135 — the *transformer* representation may be poorly extractable at
  that init).

## Variant 4 — Independent ablation replication (Aniket, `mess3_mat_ablation`)

A reduced-compute replication (2026-04-22) that fills the missing
no-window × matryoshka cell (single-position `MatryoshkaSAE`) to disambiguate
whether MatTXC's advantage comes from the matryoshka penalty or the temporal
window. No `r2_ceiling.json` was regenerated, so it normalizes against the
**dense-linear probe** (summed 1.046) as a proxy ceiling rather than the Bayes
R²_max. Summed best-feature R² across the 3 components at δ=0.20:

| Architecture | matryoshka | window | summed best-feature R² |
| --- | :---: | :---: | ---: |
| TopK SAE | ✗ | ✗ | 0.206 |
| MatryoshkaSAE (new cell) | ✓ | ✗ | 0.211 |
| TXC | ✗ | ✓ | 0.669 |
| MatTXC | ✓ | ✓ | 0.582 |

Effect decomposition vs TopK SAE baseline: matryoshka alone +0.005 (1.02×),
**window alone +0.463 (3.25×)**, both +0.376 (2.83×). **Verdict: H1 confirmed —
the temporal window, not the matryoshka penalty, drives the gap.** Translated to
the proxy-ceiling axis, MatTXC = 0.582/1.046 = **0.56**.

Note this 0.56 is *not* in tension with the ~100% headline: it uses the
main-config (un-tuned) MatTXC, a dense-linear proxy ceiling rather than the
Bayes ceiling, and the *summed* metric rather than best-single-vs-comp-0. It is
a clean answer to a different question (which ingredient matters), not a
contradiction of the ceiling-saturation result.

## Reconciling the MatTXC δ=0.20 numbers

These all appear in the sources and refer to genuinely different things:

| number | metric | denominator | config |
| ---: | --- | --- | --- |
| 0.449 | all-features linear probe | mean ceiling 0.45 (→ ~100%) | main-config |
| 0.421 | best single feature | comp-0 ceiling 0.866 (→ 49%) | main-config |
| 0.81 | best single feature | comp-0 ceiling 0.866 (→ 94%) | protocol-tuned |
| 0.851 / 0.881 | best single feature | comp-0 ceiling 0.866 (→ 98% / 102%) | protocol-tuned head-to-head |
| 0.632 ± 0.407 | best single feature, 3-seed | comp-0 ceiling 0.866 (→ 73% mean) | protocol-tuned, multi-seed |
| 0.56 | summed best-feature | dense-linear proxy 1.046 | main-config (Aniket) |

The defensible one-line claim: **a protocol-tuned MatTXC saturates the Bayes
ceiling for the recoverable (comp-0) non-ergodic component at the most-separated
cell on its best seeds (~98–102%), and averages ~73% across three transformer
seeds — clearly ahead of TXC and paper-faithful T-SAE (~30–50%).** The "~100% of
ceiling" memory is real; it is a best-seed / best-config / right-metric statement,
not a flat mean.

## Source links

- Headline table — [separation_scaling.md (commit 16452d5)](https://github.com/chainik1125/temp_xc/blob/16452d53cf9ab2bd95cdc2b0b89a266bdeefb659/experiments/separation_scaling/tables/separation_scaling.md)
- Ceiling derivation — [NOTES_r2_ceiling.md](https://github.com/chainik1125/temp_xc/blob/dmitry/experiments/separation_scaling/NOTES_r2_ceiling.md)
- Findings narrative — [separation_scaling_tsae_paper.md](https://github.com/chainik1125/temp_xc/blob/dmitry-rlhf/experiments/separation_scaling/tables/separation_scaling_tsae_paper.md)
- Window sweep — [probe_window_sweep.md](https://github.com/chainik1125/temp_xc/blob/dmitry-rlhf/experiments/separation_scaling/tables/probe_window_sweep.md)
- Replication — `docs/aniket/experiments/mess3_mat_ablation/summary.md`
