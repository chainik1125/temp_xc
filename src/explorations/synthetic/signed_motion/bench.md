# An AC-only synthetic benchmark for temporal feature recovery

*A clean test for order-sensitive structure — which current sparse
dictionaries (per-token and crosscoder alike) fail at realistic capacity.*

**Bench:** `toy_signed_motion_M19_d40`. **Conventions:**
[`README.md`](../README.md).
**Code map:** [§ 8](#8-reproduction).

---

## Abstract

The paper's two synthetic benchmarks (`coupled_hmm`, `markov_chain_support`)
are **DC-only**: a model wins by *averaging* repeated noisy evidence for a
time-stable latent, so a pure temporal smoother passes. The signed-motion
bench adds the strictly orthogonal axis — **order-sensitive (AC) recovery** —
where the only recoverable latent is the *direction* of a symbol's motion,
carried entirely by the step between consecutive tokens. It has a hard
impossibility result (per-token encoders are barred by the data-processing
inequality) and a clean ground truth (`F = 19` feature directions).

Evaluated under the framework's [synthetic-benchmark
conventions](../README.md) — `d_sae` anchored on `F` and
swept into the scarce regime, a common `L = 32` tiled eval window, and a
**memorization-free** linear probe — the honest result is a **negative**:

> In the realistic scarce regime (`d_sae ≤ F`), **no architecture recovers
> the order-sensitive sign.** Per-token SAEs are pinned at exactly chance
> (`s_temp = −0.00 ± 0.03` at every `d_sae`, by the DPI). The window
> crosscoder is *also* at chance there — it does not factor the order at
> realistic capacity. Its only above-chance cell is a single over-complete
> point (`txc T=4, d_sae = 2F`, `s_temp = 0.81 ± 0.33`) — high-variance, and
> in exactly the regime where the probe is confounded by tabulation.

So the bench is best read as a **discriminator that today's sparse
dictionaries fail**: it shows that representing order *compactly* is beyond
both per-token SAEs (provably) and the linear crosscoder (empirically). A
genuine positive would require an architecture that *factors* order from a
feature-matched dictionary.

---

## 1. Motivation

A temporal dictionary model is meant to exploit *time*, but that has two
distinct modes:

- **DC / aggregation.** A latent is roughly constant over a window and the
  observations are noisy; the model wins by averaging. The coupling and
  denoising benches live here — a low-pass smoother passes, never needing to
  know which event came first.
- **AC / order.** The recoverable latent lives in the *changes* between
  consecutive observations. Averaging destroys it; the model must represent
  a relation across positions.

The existing suite only probes the first axis. Signed motion isolates the
second, and pairs it with a proof that the per-token baselines must fail —
so any positive result is architectural, not a tuning artifact.

---

## 2. The signed-motion process

Fix an odd prime alphabet size `M = 19`, a step `v = 9` (coprime to `M`),
`d_in = 40`, `seq_len = 64`. Draw `M` orthonormal "alphabet" directions
`u_0, …, u_{M-1} ∈ R^{d_in}`. **These `F = M = 19` directions are the entire
ground-truth feature dictionary.**

Per sequence, sample a **sign** `S ~ Unif({-1,+1})` (the latent of interest)
and a **phase** `B ~ Unif(Z_M)` (a nuisance). The symbol walks the cycle at
speed `S·v`, and the activation is its embedding:

```
Q_t = (B + S·v·t) mod M ,        x_t = u_{Q_t}  (noiseless, σ = 0)
```

The sign is a **dynamical latent, not a feature direction** — it sets the
direction of motion, recoverable only from the step `Q_{t+1} − Q_t = S·v`.

---

## 3. Why per-token encoders provably fail

**(a) A single token carries zero information about the sign.** Because `B`
is uniform, `Q_t | S = s ~ Unif(Z_M)` for *both* signs, so `I(S; Q_t) = 0`.
For any per-token encoder `Z = φ(x_t)`, the chain `S → Q_t → x_t → Z` and the
data-processing inequality give `I(S; Z) = 0`. No per-token SAE can read the
sign off a single token at any width, sparsity, or nonlinearity.

**(b) A *linear* read of the whole per-token code stream also fails.** A
per-token encoder gives `z_t = φ(u_{Q_t})`; a linear probe forms the additive
score `Σ_t g_t(Q_t)`. Summed over the `M` phases of an orbit, the `+v` and
`−v` classes have identical totals (`Σ_t Σ_q g_t(q)` either way), so additive
scores cannot separate them. The sign is the *interaction* `Q_{t+1} − Q_t`,
which a reader linear in per-token codes cannot represent. (Unchanged if each
position has its own encoder — a stacked per-position SAE.)

**(c) A window encoder *could*.** A `T`-window latent is a joint function of
`(x_0, …, x_{T-1})` and so can depend on the step. Whether a
reconstruction-trained crosscoder actually learns such a code — at realistic
capacity — is the empirical question § 5 answers (it does not).

---

## 4. Method (per the synthetic-benchmark conventions)

- **Ground truth.** `F = 19` feature directions; one dynamical latent (the
  sign). `d_sae` is budgeted against `F`, not against the `2M = 38` distinct
  *window patterns* (a derived quantity, not a feature count).
- **Capacity.** `d_sae ∈ {4, 8, 16, 19}` (≤ F, the scarce/realistic regime)
  plus one over-complete reference `d_sae = 38 = 2F`; matched across all
  archs; `k_pos = 1`. (In the scarce regime `k_pos > 1` mostly clips
  `k_win = k_pos·T` to `d_sae`.)
- **Windowing.** A common `L = 32` eval window, tiled non-overlapping into
  `L/T` sub-windows, so archs of any power-of-two `T` are scored on identical
  positions. Window archs (`txc_base`, `stacked_sae`) are trained at
  `T ∈ {2,4,8}`; per-token archs (`topk_sae`, `tsae`) at `T = 1`.
- **Metrics.** `s_temp = 2·(probe_acc − 0.5)` (sign recovery; **linear**
  logistic-regression probe), `eAUC` (local alphabet-direction recovery),
  windowed `NMSE` — all over the `L = 32` tiling. Three seeds.
- **The memorization-free probe (load-bearing).** The probe scores **each
  tile-code as a separate example**, so its feature count is one tile's code
  (`d_sae`), *not* the concatenation over tiles. There are only `2M = 38`
  distinct windows; a probe with `≥ 38` features could memorize them and
  "generalize" because train/eval share the same window set. Single-tile
  probing keeps features `< 38` in the scarce regime, so any separation is
  genuine. For a per-token arch a tile is one token — the cleanest possible
  DPI control.

**The structural confound (why a clean *positive* is impossible here).** The
number of distinct windows is always `2M = 2F`. The probe is memorization-
free only for `d_sae < 2F`; but the crosscoder can only *reconstruct /
tabulate* the windows once `d_sae ≥ 2F`. The two regimes don't overlap — at
`d_sae < 2F` the probe is clean but the crosscoder can't reconstruct, and at
`d_sae ≥ 2F` the crosscoder reconstructs but the `38`-feature probe can
memorize. Scaling `M` doesn't help (`#windows = 2F` always). So this bench
can cleanly demonstrate the **negative** but not a positive.

---

## 5. Results

![AC signed-motion frontier: in the scarce regime no architecture recovers
the sign; the crosscoder's only above-chance cell is the over-complete 2F
reference.](figs/fig_ac_signed_motion.png)

*Left: `s_temp` vs `d_sae`, one line per (arch, T); the scarce /
memorization-free regime (`d_sae < 2F = 38`) is shaded; `F = 19` and `2F`
marked. Middle: `eAUC` (local recovery). Right: `NMSE`.*

**Sign recovery is a clean negative in the scarce regime.**
- **Per-token SAEs sit at exactly chance everywhere:** `topk_sae` and `tsae`
  both `s_temp = −0.00 ± 0.03` at all five `d_sae` — the DPI floor, with no
  variance.
- **The window crosscoder is also at chance for `d_sae ≤ F`:** every
  `txc_base` / `stacked_sae` cell in the scarce regime is within seed-noise
  of zero (largest is `txc T=2, d_sae=16` at `0.14 ± 0.16`, overlapping 0).
- **The only above-chance cell is over-complete and confounded:** `txc_base
  T=4` at `d_sae = 38 = 2F` reaches `0.81 ± 0.33` — huge variance, and exactly
  where the probe can tabulate. `txc T=2/T=8` don't even do that.

**Local feature recovery is the per-token archs' regime.** `eAUC` rises with
`d_sae`; at `d_sae = 19`, `tsae = 0.84 ± 0.06` and `topk = 0.68 ± 0.04`,
while `txc_base` trails (`0.40–0.54`) and gets *worse* at larger `T`. The
crosscoder is the weakest local recoverer at every capacity.

**Reconstruction is capacity-limited for everyone.** `NMSE` falls from
`≈ 0.74` at `d_sae = 4` to `≈ 0` at `d_sae = 38`; the crosscoder reconstructs
worse at larger `T` (one shared code must cover more tokens).

Full per-(arch, T) × `d_sae` tables (mean ± std across 3 seeds):

<!-- BEGIN AUTO-RESULTS ac_signed_motion -->
F = 19 feature directions. Probe is memorization-free for d_sae < 2F = 38 (d_sae=38 is the over-complete reference; its s_temp is confounded by tabulation). Mean ± std across ≤3 seeds.

**s_temp (sign recovery: 0=chance, 1=oracle)**

| arch (T) | d_sae=4 | d_sae=8 | d_sae=16 | d_sae=19 | d_sae=38 |
|---|---|---|---|---|---|
| `txc_base` (T=2) | 0.035±0.003 | -0.022±0.102 | 0.142±0.160 | 0.052±0.049 | 0.085±0.131 |
| `txc_base` (T=4) | 0.038±0.007 | -0.019±0.017 | 0.035±0.167 | 0.089±0.032 | 0.807±0.334 |
| `txc_base` (T=8) | 0.003±0.024 | 0.013±0.037 | 0.089±0.051 | 0.059±0.107 | 0.073±0.117 |
| `stacked_sae` (T=2) | -0.001±0.028 | -0.001±0.028 | 0.000±0.030 | 0.034±0.096 | 0.034±0.059 |
| `stacked_sae` (T=4) | -0.051±0.031 | 0.140±0.155 | 0.049±0.258 | 0.066±0.244 | 0.101±0.234 |
| `stacked_sae` (T=8) | 0.036±0.056 | 0.055±0.167 | 0.075±0.136 | 0.091±0.111 | 0.106±0.113 |
| `topk_sae` (T=1) | -0.001±0.028 | -0.001±0.028 | -0.001±0.028 | -0.001±0.028 | -0.001±0.028 |
| `tsae` (T=1) | -0.001±0.028 | -0.001±0.028 | -0.001±0.028 | -0.001±0.028 | -0.001±0.028 |

**sign_probe_acc (raw linear-probe accuracy)**

| arch (T) | d_sae=4 | d_sae=8 | d_sae=16 | d_sae=19 | d_sae=38 |
|---|---|---|---|---|---|
| `txc_base` (T=2) | 0.517±0.002 | 0.489±0.051 | 0.571±0.080 | 0.526±0.025 | 0.543±0.066 |
| `txc_base` (T=4) | 0.519±0.004 | 0.490±0.009 | 0.518±0.084 | 0.545±0.016 | 0.904±0.167 |
| `txc_base` (T=8) | 0.502±0.012 | 0.507±0.018 | 0.545±0.026 | 0.529±0.053 | 0.536±0.058 |
| `stacked_sae` (T=2) | 0.500±0.014 | 0.500±0.014 | 0.500±0.015 | 0.517±0.048 | 0.517±0.030 |
| `stacked_sae` (T=4) | 0.474±0.015 | 0.570±0.077 | 0.524±0.129 | 0.533±0.122 | 0.551±0.117 |
| `stacked_sae` (T=8) | 0.518±0.028 | 0.528±0.083 | 0.538±0.068 | 0.546±0.056 | 0.553±0.057 |
| `topk_sae` (T=1) | 0.500±0.014 | 0.500±0.014 | 0.500±0.014 | 0.500±0.014 | 0.500±0.014 |
| `tsae` (T=1) | 0.500±0.014 | 0.500±0.014 | 0.500±0.014 | 0.500±0.014 | 0.500±0.014 |

**eAUC (local: alphabet-direction recovery)**

| arch (T) | d_sae=4 | d_sae=8 | d_sae=16 | d_sae=19 | d_sae=38 |
|---|---|---|---|---|---|
| `txc_base` (T=2) | 0.332±0.004 | 0.375±0.020 | 0.506±0.026 | 0.539±0.020 | 0.708±0.064 |
| `txc_base` (T=4) | 0.305±0.007 | 0.364±0.006 | 0.424±0.027 | 0.448±0.021 | 0.537±0.034 |
| `txc_base` (T=8) | 0.297±0.023 | 0.344±0.008 | 0.382±0.009 | 0.403±0.004 | 0.481±0.016 |
| `stacked_sae` (T=2) | 0.342±0.003 | 0.498±0.011 | 0.589±0.010 | 0.542±0.040 | 0.648±0.008 |
| `stacked_sae` (T=4) | 0.333±0.001 | 0.421±0.027 | 0.482±0.010 | 0.483±0.034 | 0.564±0.021 |
| `stacked_sae` (T=8) | 0.340±0.011 | 0.401±0.005 | 0.455±0.055 | 0.477±0.015 | 0.496±0.028 |
| `topk_sae` (T=1) | 0.352±0.002 | 0.490±0.014 | 0.736±0.025 | 0.678±0.039 | 0.855±0.025 |
| `tsae` (T=1) | 0.298±0.009 | 0.450±0.014 | 0.696±0.078 | 0.837±0.063 | 0.914±0.039 |

**NMSE (windowed reconstruction; lower=better)**

| arch (T) | d_sae=4 | d_sae=8 | d_sae=16 | d_sae=19 | d_sae=38 |
|---|---|---|---|---|---|
| `txc_base` (T=2) | 0.744±0.001 | 0.558±0.004 | 0.253±0.004 | 0.173±0.005 | 0.018±0.007 |
| `txc_base` (T=4) | 0.754±0.000 | 0.614±0.000 | 0.376±0.004 | 0.295±0.005 | 0.000±0.000 |
| `txc_base` (T=8) | 0.793±0.000 | 0.670±0.001 | 0.440±0.001 | 0.361±0.004 | 0.000±0.000 |
| `stacked_sae` (T=2) | 0.737±0.000 | 0.544±0.015 | 0.157±0.045 | 0.061±0.016 | 0.000±0.000 |
| `stacked_sae` (T=4) | 0.737±0.001 | 0.537±0.008 | 0.145±0.040 | 0.053±0.013 | 0.000±0.000 |
| `stacked_sae` (T=8) | 0.737±0.001 | 0.532±0.005 | 0.158±0.025 | 0.039±0.006 | 0.000±0.000 |
| `topk_sae` (T=1) | 0.737±0.001 | 0.561±0.031 | 0.158±0.053 | 0.070±0.061 | 0.000±0.000 |
| `tsae` (T=1) | 0.752±0.005 | 0.551±0.012 | 0.236±0.087 | 0.132±0.080 | 0.054±0.051 |

<!-- END AUTO-RESULTS ac_signed_motion -->

---

## 6. What this bench is

A **discriminator that current sparse dictionaries fail.** It cleanly
establishes:

1. **Per-token encoders cannot represent order** — a theorem (DPI), confirmed
   to the third decimal (`−0.00 ± 0.03` everywhere).
2. **A linear crosscoder does not *factor* order either** — at realistic
   (scarce) capacity it is no better than chance; it only exposes the sign
   when handed enough atoms to *tabulate* the windows, where the measurement
   is itself confounded.

A genuine positive — recovering the sign at `d_sae < 2F` with a clean probe —
would require an architecture with explicit relational / dynamical structure
(e.g. one that represents "the `F` features" plus a small "motion" code),
which the linear crosscoder is not. That is the natural follow-up: use this
bench to test *factoring* architectures, where the negative leaves clear room
for a positive.

This is the honest complement to the DC benches: there the crosscoder wins by
aggregating; here, on order, neither family wins at realistic capacity.

---

## 7. Caveats

- **Noiseless, single frequency** (`σ = 0`, one step `v`). The DPI argument is
  exact at `σ = 0`. Noise and multi-frequency variants are future work (see
  [`frequency_lens.md`](../../../../docs/ideas/frequency_lens.md)).
- **`d_sae = 38` is confounded** (probe can memorize the 38 windows); it is
  included only as the over-complete reference, never as a headline.
- **`k_pos = 1` only.** In the scarce regime larger `k_pos` clips `k_win` to
  `d_sae`; a `k_pos` sweep would mostly probe the dense corner.

---

## 8. Reproduction

```bash
# (run from the repo root; TEMP_BENCH_ALLOW_DIRTY=1)

# one cell (txc T=4, scarce d_sae, L=32 tiling)
.venv/bin/python run.py synthetic --arch txc_base --seed 1 \
  --datasource toy_signed_motion_M19_d40 \
  --d-sae 16 --k-pos 1 --T 4 --eval-window-l 32 --n-steps 10000 --batch-size 1024

# full 3-seed sweep (8 arch×T × 5 d_sae)
SEED=1 bash synthetic/signed_motion/minisweep.sh &
SEED=2 bash synthetic/signed_motion/minisweep.sh &
SEED=42 bash synthetic/signed_motion/minisweep.sh &

# tables (fills the AUTO-RESULTS block) + frontier figure
.venv/bin/python synthetic/signed_motion/populate.py results/leaderboard.jsonl \
 synthetic/signed_motion/bench.md
.venv/bin/python -m explorations.synthetic.signed_motion.render_figs
```

| Component | Path |
|---|---|
| Generator (`signed_motion`) | `src/temp_bench/data/synthetic.py` |
| Datasource | `configs/data.yaml` → `toy_signed_motion_M19_d40` |
| Tiled probe + metrics | `src/temp_bench/evals/signed_motion_recovery.py` |
| Tiled recon + window sampling | `src/temp_bench/evals/synthetic_recovery.py` |
| Sweep / populate / figure | `synthetic/signed_motion/minisweep.sh`, `synthetic/signed_motion/populate.py`, `synthetic/signed_motion/render_figs.py` |
| Tests | `tests/test_ac_bench.py` |

**Plumbing note.** The bench runs through the existing `synthetic` experiment
and `SyntheticRecovery` evaluator (which computes `s_temp` only when the
datasource carries sign labels), honoring hard rule #3 (no edits to
`temp_bench/core/`). The `L = 32` tiled metrics are protocol `1.2.0`.
