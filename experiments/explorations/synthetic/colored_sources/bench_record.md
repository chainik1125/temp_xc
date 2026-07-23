# Colored sources (FB-3) — bench record

**Status: AWAITING GRID (2026-07-23, runpod-b, FB-C1).** Frozen card:
[`../freqbench/cards/FB-3.md`](../freqbench/cards/FB-3.md) (commit
`f0e6778f`, BEFORE construction). Gates: T1/§8 PASS
([`results/colored_gating_stats.json`](results/colored_gating_stats.json)),
T2 PASS ([`results/colored_t2_stats.json`](results/colored_t2_stats.json)),
skeptic PROCEED 5/5
([`../freqbench/results/skeptic_verdict_FB-3.json`](../freqbench/results/skeptic_verdict_FB-3.json)).
Provenance `theorem-first`. **The suite's first
feature-direction-recovery-primary bench.**

> ⚠ THIS FILE IS A SKELETON until the blind-verdict section is written from
> `results/colored_bench_stats.json`. No claim below the fold is final.

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

*(TO FILL from `colored_bench_stats.json` — check falsifiers FIRST.)*

## 5. Coordinates (axis 1, FreqFrac at bench time)

*(TO FILL — `freqfrac_report` on the canonical cells once registered.)*
