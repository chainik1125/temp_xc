# Colored sources (FB-3) — bench record

**Status: DONE — verdict POSITIVE, weak realization, with an ordering
INVERSION against the tone benches (2026-07-23, runpod-b, FB-C1).**
Frozen card:
[`../freqbench/cards/FB-3.md`](../freqbench/cards/FB-3.md) (commit
`f0e6778f`, BEFORE construction). Gates: T1/§8 PASS
([`results/colored_gating_stats.json`](results/colored_gating_stats.json)),
T2 PASS ([`results/colored_t2_stats.json`](results/colored_t2_stats.json)),
skeptic PROCEED 5/5
([`../freqbench/results/skeptic_verdict_FB-3.json`](../freqbench/results/skeptic_verdict_FB-3.json)).
Provenance `theorem-first`. **The suite's first
feature-direction-recovery-primary bench.**

Grid outcome: **582/582 cells, 0 failures, 49 min** (28 workers). Stats:
[`results/colored_bench_stats.json`](results/colored_bench_stats.json),
figure [`figs/colored_bench.png`](figs/colored_bench.png).

## 1. The task (frozen)

`N = d_in = 32` sources on a Haar orthonormal basis `F`; per-coordinate
AR(1) **at lag D=2** with `ρ_i = linspace(0.1, 0.9, 32)`; `x_t = Fᵀz_t +
0.1·ε`. `F = 32` anchors `d_sae ∈ {16, 32, 64}`. Primary metric
`colored_rec_adj` (chance-adjusted max-cos² of F rows vs per-position
decoder slices; weight-space, no probe).

## 2. Proof obligations — discharged

- **CS-1 floor (local impossibility):** analytic (d=N ⇒ marginal exactly
  `N(0,(1+σ²)I)`; C_ℓ=0 for 0<ℓ<D + joint Gaussianity ⇒ windows ≤ D are
  iid isotropic). Verified on the built data: marginal off-diag ≤ 0.008,
  lag-1 cov ≤ 0.009; W ≤ 2 estimators inside the (orthonormal-null) floor
  band (+0.023); trained-token control +0.025.
- **CS-2 ceiling (lag-D eigen-recovery):** verified at the exact frozen
  budget — full-sequence oracle **rec_adj +0.962**; W-resolved oracle
  +0.961 (W=4), +0.960 (W=8); the **W = D+1 transition is 0.03 → 0.96**
  between W=2 and W=4.
- **Memorization:** continuous Gaussian data — no template set exists;
  primary metric is probe-free.

**Documented amendments (skeptic-examined):** (1) floor checks use the
orthonormal null (eigenbases score 0.181 vs Gaussian 0.170 by geometry);
(2) measured SYSTEMATIC stream-leakage of the marginal estimator
+0.011 rec_sq over 8 seeds (CS-1 is iid-strict; the correlated stream
tilts Ĉ₀'s fluctuations) — operational floor bar |rec_adj| ≤ 0.05 ≈ 20×
below the ceiling; (3) card § 3's bag line corrected: pooling/shuffling
only DILUTES C_D (pooled eig rec_adj +0.69) — **the true null is window
truncation (W ≤ D)**, so FB-3 is a *depth* bench, not a permutation bench;
(4) untrained spectral scores −0.07..−0.09 adj (correlated band-limited
time-slices ⇒ effective candidates < d_sae·T) — the Gaussian chance
reference is conservative AGAINST spectral; remember when reading small
spectral lifts.

## 3. Frozen predictions under test (card § 6, verbatim summary)

Context recorded pre-freeze: the sprint's own trained TXC never left the
floor at any (D, W) while its oracle passed — trained realization is the
open question; the all-floor outcome is pre-registered as a strong citable
NEGATIVE, not a failed bench.

1. Token archs (T=1) and ALL window archs at T=2: floor — **provable**.
2. stacked at T ∈ {4,8}: floor (per-position marginal inputs).
3. txc-pre: floor to weak (≤ 0.15), lean floor.
4. txc-post: weak lift 0.05–0.35 at T ∈ {4,8}, ρ-ordered if present.
5. spectral ≈ txc-post (no band advantage expected at D=2).
6. Any lift confined to mixing archs × T ∈ {4,8}; ρ-quartile curve
   increasing where a lift exists.
7. Falsifiers: any arch above the floor band at T ≤ 2 (CS-1 bug); § 8
   oracle < 0.75 (sample starvation — did not fire: 0.96); untrained
   POSITIVE artifact (metric bug).

## 4. Blind verdict vs the frozen predictions

Written against the card § 6, falsifiers first. `colored_rec_adj`, 3-seed
means at the canonical slice (d_sae = F = 32, k_pos = 2) unless noted; the
oracle ceiling at this exact budget is **+0.96**.

**4.0 Falsifiers — none fired.** All 261 trained cells at T ≤ D: max
**+0.0369**, mean −0.005 (floor band 0.05 + 0.02 leakage) — **the CS-1
impossibility holds in the trained grid, wholesale**. No untrained cell
shows a positive artifact (spectral's negative offsets as documented). The
§ 8 oracle passed at +0.96 pre-grid.

**4.1 Token archs + all T=2 window cells at floor — HELD (provable,
measured).** batchtopk_sae +0.006, tsae +0.004; T=2 row: −0.033 … +0.013.

**4.2 stacked at floor for ALL T — HELD.** Max anywhere **+0.037**
(canonical: +0.007 / +0.002 at T=4/8) — the per-position-marginal argument
is airtight in practice.

**4.3 The W = D+1 transition IS REALIZED — the headline, and it is carried
by the arch the card bet AGAINST.** txc-pre: T2 **−0.007** → T4 **+0.109**
→ T8 +0.095 (canonical), up to **+0.205** (T8, d=2F, k=1) — and its
per-source recovery is cleanly **ρ-ordered** (quartile means 0.29 → 0.35 →
0.47 → 0.65 at the best cell): the temporal route, used exactly as CS-2
describes, strongest sources first. The frozen prediction said "floor to
weak (≤ 0.15), lean floor" — **missed low** (several cells exceed 0.15).

**4.4 txc-post 0.05–0.35 — MISSED, opposite direction.** Canonical cells
at or BELOW floor (−0.059 / −0.065 at T=4/8; the negative values are a
trained-atom-clustering geometry effect, recorded); best cell +0.073. The
coincidence family (ReLU after position-mixing) does NOT realize the lag
covariance.

**4.5 spectral ≈ post — MISSED: spectral is clearly above.** +0.093
canonical T8, **+0.202** at (T8, d=2F, k=1), on a metric whose chance
reference is conservative AGAINST it (−0.05..−0.08 at init; § 2 amendment
4) — the true lift is larger. The "no band advantage at D=2" bet was
wrong: band-limited slow kernels align naturally with high-ρ AR structure.
(Its best-cell quartile curve is non-monotone with an unexplained q1
excess — single-cell observation, flagged, not interpreted.)

**4.6 Lift confined to mixing archs × T ∈ {4, 8} — HELD.**

**Verdict: POSITIVE — the bench separates architectures, and it separates
them DIFFERENTLY from every other bench in the suite.** Two headline
facts, both citable:

1. **Provably-present temporal dictionary information is realized only
   weakly by the current panel:** best arch ≈ **21 % of the provable
   ceiling** (+0.205 vs +0.96). The sprint's all-floor result is
   overturned in kind (BatchTopK + 30k steps + N=32 lift off the floor)
   but confirmed in magnitude — dictionary training barely touches what
   the eigen-oracle proves extractable. The gap is the finding.
2. **The panel ordering INVERTS the tone benches:** here txc-pre ≥
   spectral > post ≈ stacked ≈ token, while frequency/multilane give
   spectral > post ≫ pre. Tone latents reward nonlinearity-after-mixing
   (coincidence); covariance-eigenstructure rewards additive gating with
   a T-spanning decoder. No single window architecture dominates the
   axis-1/axis-2 plane — exactly the kind of dissociation the two-generator
   program exists to expose.

## 5. Coordinates (axis 1, FreqFrac at bench time)

`freqfrac_report colored_sources` (seed 1; stats under
`../freqbench/results/freqfrac_stats_colored_sources_s1_T{4,8}.json`),
firing-weighted dc_frac / concentration, trained (init):

| arch | T=4 dc | T=4 conc | T=8 dc | T=8 conc |
|---|---|---|---|---|
| token archs | 1.000 | 1.000 | 1.000 | 1.000 |
| stacked | 0.264 (0.256) | 0.561 (0.553) | 0.138 (0.125) | 0.307 (0.301) |
| txc-pre | 0.292 (0.254) | 0.672 (0.551) | 0.144 (0.127) | 0.360 (0.311) |
| txc-post | **0.503 (0.256)** | 0.606 (0.550) | **0.489 (0.127)** | 0.576 (0.311) |
| spectral | 0.314 (0.314) | 1.000† | 0.329 (0.363) | 0.974 (0.927) |

† T=4 band degeneracy (§ G caveat); read the curve.

A telling decoupling: **txc-post develops the strongest DC-shifted taps in
the suite outside hedging (0.13 → 0.49)** — its atoms DO respond to the
slow AR structure — yet its decoder never aligns with F (§ 4.4). Slow
temporal response ≠ dictionary identification: the coincidence
nonlinearity integrates the drift without factoring it. txc-pre, whose
FreqFrac shift is modest (conc 0.31 → 0.36), is the one that converts the
structure into F-aligned decoder taps. Axis-1 coordinate: the ρ-ladder
spans DC→AC by construction; the recoverable object is 2nd-moment at lag
D (order-2), stationary.

## 6. Review (2026-07-23, mac-local) — APPROVED

Verdict stands, including the inverted ordering. The orthonormal-null +
stream-leakage amendments verified genuine (correct reference class; the
iid-premise leakage measured and bounded, not hidden); the § 2 amendment 3
(bag-dilution) rightly reframes this as a *depth* bench — true null =
window truncation — which makes txc-pre's win coherent with an
additive-over-window second-moment route and feeds the order-2 subtype
rule (README coordinates). 582 grid + 1 smoke cell reconcile; 0 dup keys;
three prediction misses labeled as misses. Audit: `../freqbench/PORT.md` § H.
