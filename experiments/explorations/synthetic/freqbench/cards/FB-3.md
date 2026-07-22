# FB-3 — colored sources (per-coordinate AR(1) at lag D)

**Axis-point card, frozen per LOOP.md before any construction.**
Status: FROZEN (2026-07-22, runpod-b, cycle FB-C1). Provenance:
`theorem-first` (`origin/dmitry-synthetic:src/v6_colored_sources/` — README
math + CS-1/CS-2 in PORT.md § B; sprint empirics in
`results/v6_colored_sources/stage{1,2}.json`).

**This is a *feature-direction-recovery* bench** (cosine primary), the first
in the suite where direction recovery is the headline rather than the
capability companion — it fills the "feature recovery vs latent recovery"
gap in the coordinate system (LOOP.md seed list).

---

## 1. Target coordinates + gap claim

- **Axis 1 (spectral):** a *ladder*, not a point — each coordinate `i` is an
  AR(1) at lag `D` with autocorrelation `ρ_i ∈ [0.1, 0.9]`, so the sources
  span DC-leaning (high ρ) to near-white (low ρ) within one task. The
  per-ρ recovery curve is an axis-1 spectrum sweep inside a single bench.
  FreqFrac coordinates measured at bench time (T4).
- **Axis 2 (interaction order):** order-2 — the recoverable structure lives
  exclusively in the lag-D *second moment* `C_D = Fᵀ diag(ρ) F`; every lower
  moment and every shorter lag is exactly F-blind (§ 3 floor).
- **Axis 3 (localization):** stationary / spread.

**Gap claim.** Every evaluated bench scores *latent* recovery; `eauc` is only
a companion gate. No bench tests whether temporal architectures recover the
**dictionary itself** from temporal structure that is provably invisible to
any iid/per-token learner. FB-3 is that bench, with the sharpest floor in the
registry (CS-1: a *local impossibility*, not just a chance line) and a sharp
window-depth phase transition (CS-2: `W ≥ D+1`). Not redundant: no existing
row has a provable per-token impossibility for its PRIMARY metric that is
information-theoretic on the training distribution itself.

## 2. Constructed task

```
F ∈ R^{N × d_in}, N = d_in: Haar-random orthonormal basis (rows)
per coordinate i, per residue class r mod D: independent AR(1) at lag D —
    z_{t,i} = ρ_i · z_{t−D,i} + sqrt(1 − ρ_i²) · η_{t,i},   η ~ N(0,1)
    (stationary init on t < D: z ~ N(0,1))
x_t = Fᵀ z_t + σ ε_t = Σ_i z_{t,i} f_i + σ ε_t,   ε_t ~ N(0, I_d)
```

**Exact parameterization (frozen):** `N = d_in = 32` (d = N is load-bearing:
it makes the one-token marginal exactly isotropic, § 3), `D = 2`, `σ = 0.1`,
`ρ_i = linspace(0.1, 0.9, 32)` (distinct; eigengap `γ = 0.8/31 ≈ 0.0258`),
`seq_len = 64`, `n_seqs = 4096`, eval `L = 32`. Datasource:
`toy_colored_sources_N32_D2_d32` (generator `colored_sources`, append-only).
Sizing vs the sprint: the sprint's N=128 gave `γ ≈ 0.0063` and a sample-
starved oracle (s_adj 0.58–0.64); at N=32 the CS-2 sample condition
`T_eff ≫ (1+σ²)² N / γ² ≈ 49k` is met ~5× by our 254k lag-D pairs, so the
ceiling should be decisively high — verified numerically at T1, gated at § 8.

**Ground truth (Part II § 1).** Feature directions: the `N = 32` rows of `F`
— **`F = 32` in the strict Part II sense**, and it directly anchors capacity:
`d_sae ∈ {16, 32, 64}`. Hidden latents (not directions): the continuous
states `z_{t,i}` (Gaussian, dense — every source active at every token; no
sparse support exists) and the ρ-ladder assignment. Nothing is conflated
with a pattern count: the data is continuous, there are no templates.

**Primary metric:** `colored_recovery` add-on —
`rec_sq = (1/N) Σ_i max_{j,t} cos²(f_i, W_dec[j,t,:])` (max over atoms AND
decoder positions — a window atom's direction content may live at any tap;
per-token archs reduce to the standard max over atoms), chance-adjusted:
`rec_adj = (rec_sq − chance)/(1 − chance)` with the chance floor computed
empirically from seeded random dictionaries of the same (d_sae, T) shape
(analytic scale `≈ log(H·T)/N`, stated alongside). Per-ρ-quantile recovery
curve reported (recovery should be ordered by ρ). Justification for the
add-on: `eauc` stores only threshold-AUC/|cos| scalars — the CS-form
statistic (cos², chance-adjusted, per-ρ curve) cannot be derived from them,
and the proofs are stated in Rec form. Additive dispatch on
`extra['rho_schedule']`; protocol stays 1.3.0. `eauc`/`nmse` still reported
(companions). Optional secondary (diagnostic only, not gating): ridge probe
on codes → `z_t` at the tile leading edge.

## 3. Proof obligations (PORT.md § B format)

- **Ceiling (CS-2 — lag-D recoverability).** `C_D := E[x_{t+D} x_tᵀ] =
  Fᵀ diag(ρ) F`; with distinct ρ_i (eigengap γ) the eigenvectors of the
  symmetrized empirical `Ĉ_D` recover the rows of F with angular error
  ~ ε/γ once the effective sample count clears `(1+σ²)² N / γ²`. The oracle
  is *window-local with W = D+1*: it needs only (x_t, x_{t+D}) pairs, so the
  information is present in every length-(D+1) window. **Discharge:**
  numerical (the `verify_theory` pattern): compute the oracle's `rec_adj` on
  the ACTUAL generator at the exact frozen parameters and data budget, plus
  the W-resolved variant (estimator restricted to windows of length W:
  W ≤ D must sit at the floor, W = D+1 must jump) — the phase-transition
  curve, committed in `gating.py`.
- **Floor (CS-1 — local impossibility).** For window length `W ≤ D`: all
  within-window lags ℓ satisfy 0 < ℓ < D ⇒ `C_ℓ = 0`; the process is
  jointly Gaussian, so zero cross-covariance ⇒ *independence*; and the
  one-token marginal is `N(0, (1+σ²) I_d)` — **exactly isotropic, carrying
  zero information about F** (this is why d = N). Hence any learner whose
  inputs are windows of length ≤ D — any per-token SAE, any length-≤D
  window arch, and any per-position-marginal path — outputs directions
  independent of F: `E[rec_sq] = chance`, regardless of compute or samples.
  Argument name: **local impossibility** (information-theoretic, on the
  training distribution itself — not a probe statement). **Discharge:**
  analytic note in the record (the argument above is complete) + the
  numerical W-resolved floor check in gating.
- **Non-triviality.** No symmetry/relabeling route: F is Haar-random; the
  only F-dependent observable statistic is `C_{kD}` (k ≥ 1) — there is no
  symbol structure, no template set, no label to relabel. Memorization is
  impossible (continuous Gaussian data, measure-zero repeats). The
  bag-of-symbols control (T2) must FAIL: mean-pooled per-token codes see
  only the isotropic marginal ⇒ floor. **Discharge:** analytic + T2.

## 4. Regime claim + design-time discriminability

**Regime 3**, with the regime statement adapted to direction recovery:
per-token-readable = nothing (the marginal is isotropic — CS-1);
linear-in-window = nothing for W ≤ D, and for W > D the *recovery target*
is a second-moment object (eigenstructure), not a linear readout — the
task tests which architecture's TRAINING converts lag-D covariance into
decoder geometry. The proofs predict apart: archs whose input path never
crosses positions at lag ≥ D (token archs at any T; per-position stacked
dicts at any T; all window archs at T ≤ D) are floored; archs that mix
positions at lag ≥ D (txc-pre/post, spectral at T ≥ D+1 = 3, i.e. T ∈
{4, 8}) are *permitted* to recover — whether they DO is the open question
(§ 6). Design-time discriminability (§ 8 gate, feature-recovery variant of
the STOP-gate): (i) the W-resolved oracle must separate W ≤ D (floor) from
W ≥ D+1 (high) decisively at our exact budget; (ii) the trained-arch
question is only meaningful if the oracle ceiling `rec_adj` is ≥ 0.75 and
the floor band is tight (empirical random-dict spread). If the oracle
cannot clear 0.75 at our budget, the bench is sample-starved (the sprint's
N=128 failure mode): record NON-DISCRIMINATING and STOP — no grid.

## 5. Memorization audit (P6)

Template count: **none** — the data is continuous Gaussian; every window is
distinct with probability 1, and there is no finite codebook to look up. The
P6 route (whole-window template memorization above `|Ω|·M`) does not exist at
any d_sae; no capacity threshold is crossed anywhere in the sweep and no jump
is predicted. Probe budget: the primary metric is weight-space (no probe at
all — memorization-free by construction); the optional z-ridge diagnostic
follows the F-rule probe budget.

## 6. Frozen per-arch predictions (+ falsifiers)

Context that disciplines these predictions (recorded pre-freeze): in the
sprint's own stage-2 data the **eigendecomposition oracle** shows the
transition (s_adj 0.58–0.64 vs floor 0.025) but the **trained vanilla TXC
never left the floor at any (D, W)** (s_adj 0.021–0.029), and neither did the
per-token SAE. The trained-realization question is genuinely open; the
sprint says "no" for vanilla TXC at its hyperparameters (plain TopK, k=8,
8k steps, N=128). Our cells differ (N=32, BatchTopK backbone, 30k steps,
k_pos-matched): predictions below are honest bets, not transports.

`rec_adj` at d_sae = F = 32, seeds {1,2,42}:

| arch | T | prediction | reason |
|---|---|---|---|
| batchtopk_sae, tsae | 1 | = floor (within random-dict band) | CS-1: isotropic marginal — **provable** |
| all window archs | 2 | = floor | T = D ⇒ windows are exactly iid isotropic — **provable** (CS-1) |
| stacked_batchtopk | 4, 8 | = floor | each per-position dict's input is the isotropic marginal; no cross-position path exists in the architecture — CS-1 applies per dict |
| txc_batchtopk_pre | 4, 8 | floor to weak (≤ 0.15) | encoder gates per position (marginal-driven, additive); decoder spans T so a weak lift is possible via the shared-code coupling — uncertain, lean floor |
| txc_batchtopk_post | 4, 8 | weak lift, 0.05–0.35, ordered by ρ (top-ρ quartile first) | position-mixing pre-ReLU can capture lag-2-correlated window directions (variance 1+ρ_i > noise); but reconstruction with dense Gaussian latents has no sparse structure to lock onto — the sprint's TXC failed; BatchTopK + longer training may differ |
| spectral_txc | 4, 8 | ≈ txc-post ± (no band advantage) | the DCT-band prior targets oscillatory taps; the informative temporal pattern at D=2 is a lag-2 comb — representable but not band-favored |

**Ordering claims:** every T ≤ 2 cell and every stacked/token cell at the
floor; any lift confined to {txc-pre, txc-post, spectral} × T ∈ {4, 8}; if
a lift exists it is ρ-ordered (per-ρ curve increasing). **The W = D+1
transition claim:** the bench's headline is the T-profile floor→(possible)
lift between T=2 and T=4 for mixing archs, against the provable oracle
transition at the same boundary.

**Honest-outcome framing (pre-registered):** if ALL archs sit at the floor
at T ∈ {4, 8} while the § 8 oracle passes, the verdict is a strong citable
NEGATIVE — "provably-present temporal dictionary information that no current
panel architecture's training realizes" — not a failed bench. That outcome
is expected under the sprint's evidence and is fully acceptable.

**Falsifiers (substrate-indicting):**
1. Any arch above the floor band at T ≤ 2 (or any token arch at T=1) ⇒
   CS-1 violated ⇒ generator/metric bug — STOP and debug.
2. § 8 oracle below 0.75 or no W-transition at our budget ⇒ the built task
   is not the theorem's regime (sample starvation) ⇒ NON-DISCRIMINATING
   stop, no grid.
3. Untrained-init models materially above the floor band ⇒ the metric has
   an access artifact (e.g. init statistics aligned with F) ⇒ metric bug —
   the chance floor is DEFINED by random dicts, so this must not happen.

## 7. Skeptic pre-registration notes

`c_relevance`: slow/colored drift is the measured axis-1 signature of real
LM streams (hedging_drift ACF plateau; backtracking DC-dominant FreqFrac
taps — PORT.md § G), and "the dictionary is only identifiable from temporal
statistics" is precisely the claim temporal-crosscoder architectures exist
to exploit. No PhenomenonBench measurement pins per-coordinate AR at a
clean lag D — the card is `spanning`: it spans the coordinate axis with a
provable ceiling/floor pair the grounded benches cannot supply (their
per-token marginals are never provably F-blind). `d_redundancy`: no
existing bench has direction recovery as primary (§ 1).
