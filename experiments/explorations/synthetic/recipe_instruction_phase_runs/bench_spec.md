# Synthetic benchmark spec — recipe-instruction phase runs (categorical equality-adjacency)

**Status:** spec / preregistration. **Not yet run** (stage 6 of the loop) —
frozen by the expansion Cycle-4 PROCEED, staged for a *later, deliberately
blind* B×A evaluation. No architecture has seen or touched this spec.

**Provenance (`grounded`).** Expansion-loop Cycle 4 (autonomous, gated):
prereg card
[`../expansion/prereg/recipe-instruction-phase-runs.md`](../expansion/prereg/recipe-instruction-phase-runs.md)
(frozen 2026-07-19 under the C3 categorical gate-7-clean recipe; the C3
calibration ABORTED solely on gate-8 — the `semi_markov` mirror undershot
held-out ACF(4) by 32% — and the C3 review recorded the fix as the C4 menu
extension), dated re-freeze amendment 2026-07-22 (hier_categorical mirror +
the **hardened ≥2-non-fitted-moment gate-8**, preregistered before any C4
run), calibration record
[`../expansion/records/recipe-instruction-phase-runs-r2/calibration.md`](../expansion/records/recipe-instruction-phase-runs-r2/calibration.md)
— verdict **PROCEED**: cleared the N1/N2/N3 battery, the noise floor, BOTH
hardened gate-8 moments (ACF(4) |err| 0.018 ≤ 0.059 — the moment that killed
the C3 mirror; MI(2) |err| 0.029 ≤ 0.036), and all five skeptic items.
Labeled at **ctx=0** (gate-7 strict per-sentence). **The program's first
interaction/equality SPEC — and its first grounded regime-3 candidate.**

**Ledger cell (measured-class filing).** Proposed under
`text-corpus × interaction/equality`; the **measured** signature is genuinely
multi-class run/segment structure — a 5-class functional marginal
[0.29, 0.51, 0.06, 0.07, 0.07], categorical self-match ACF, heavy-dwell runs
(mean 2.90, CV 1.76), Markov order-1 ≫ order-0 — so it **files under
interaction/equality** (no re-filing; the cell is claimed).

## 1. What it tests

Whether a dictionary code linearly exposes the **equality/run structure of
functional phases** in ordinary instructional web text. Per-sentence content
classes (`0` other/background · `1` context-background · `2`
ingredient/material-listing · `3` imperative-step · `4` tip-caution — each
decidable from the sentence's own wording, ctx=0, κ = 0.609) form
multi-sentence functional segments; the **primary latent is the
equality-adjacency `e_t = [c_t = c_{t-1}]`** — the equality lives in the
statistic, never in the label (gate 7).

Measured on the pinned 400-doc fineweb sample (36,805 sentences, coverage
0.983): self-match ACF(1) = **0.479** [0.455, 0.502] vs N1 hi **0.204** / N2
hi **0.019**; noise-robust at ε̂ = 0.137 (perturbed 0.356 still ≫ N1 hi);
split-half stable (0.461/0.495). The N1 band is itself elevated (≈0.20):
docs differ strongly in phase composition, so the gate margin is measured
*above* the composition ceiling (the skeptic's composition item cleared on
exactly this comparison).

## 2. Generative process (two layers)

### Layer 1 — hierarchical categorical phase dynamics (the fitted mirror)

The canonical mirror is the **C4 `hier_categorical` fit** (275 train docs,
118 held-out; full precision in [`mirror_params.json`](mirror_params.json)):

- **per-doc phase propensities** `pi_j` — the empirical list of 275 per-doc
  vectors, deconvolved so the generator's within-doc stationary matches the
  observed doc marginal (heavy tails preserved);
- **per-symbol empirical dwell** (means [3.0, 4.0, 2.4, 1.7, 1.5], pooled CV
  1.76);
- **global jump chain** `J` tilted toward the doc's propensities with MLE
  weight **α = 0.94**: `P(d | c, j) = α·pi_j(d | d≠c) + (1−α)·J(c,d)`.

**Matched:** dwell, one-step jumps, doc-marginal heterogeneity. **Gate-8
verified non-fitted moments (hardened rule, both on held-out docs):**
self-match ACF(4) (real 0.294 vs syn 0.276) and pooled MI(2) (real 0.178 vs
syn 0.207). **Documented fidelity limits (skeptic caveats, carried):**
MI(2) passed with ~17% margin and the synthetic slightly *over*-disperses
(the deconvolved propensity list contains near-one-hot vectors from
homogeneous non-instructional docs); the cross-doc heterogeneity *level* is
inserted by the propensity list (the `hier_ar1` precedent) — the gate
verified the lag *structure*, not the level, and the C3 `semi_markov` fit
demonstrates the gate has teeth (it failed exactly here at −32%).

### Layer 2 — emission into activations

The standard emission pattern: sentence `i` →
`x_i = m·u_{c_i} + Σ_{j∈content_i} m_j·u_j + σ·ε_i`, with 5 phase-signature
directions (one per class) + `K_c = 15` content features ⇒ **`F = 20`**,
`d_in = 64` (pinned at datasource-plugin time; none added this cycle).

## 3. Ground truth / capacity / metrics

Per Part II conventions and the uniform design: `F = 20`;
`d_sae ∈ {F//2, F, 2F}`, `k_pos ∈ {1,2,4,8,16}`, seeds {1,2,42}, untrained
control; `L = 32`, `T ∈ {2,4,8}` tiled; cosine-AUC (phase vs content dirs,
never pooled) / per-tile leading-edge linear probes (example-split) /
windowed NMSE. Two probed latents:

| latent | type | chance | oracle |
|---|---|---|---|
| phase class `c_t` | DC (control floor — per-token-readable by design) | marginal-balanced | 1 |
| equality-adjacency `e_t = [c_t = c_{t-1}]` | **regime 3 — primary** | pooled match rate | 1 (deterministic given the two labels) |

## 4. Coordinates + regime claim (checklist item 8)

- **Spectral:** mid — piecewise-constant phase state (DC-ish runs, AC
  boundaries).
- **Interaction order:** **equality** — the primary latent compares content
  classes at two positions.
- **Stationarity:** spread — phases run through the document, not
  burst-localized.

**Regime-3 claim (design-time discriminability, argued at freeze):** the
phase label `c_t` is per-token-readable (regime 1 — deliberately the control,
not the object under test). The primary latent `e_t` requires comparing
classes at two positions: additive codes are provably blind to
equality-pattern latents (the changepoint precedent), and `e_t` cannot
collapse into a per-token sufficient statistic the way a next-state
prediction can (the stage-6 assumption_consequence lesson) — it is a
function of two observables, whatever the chain's order. The empirical § 8
raw-ceiling / discriminability STOP-gate runs at stage-6 build time; note
that for equality latents BOTH raw-linear readouts (per-token and window
concatenation) may sit near chance — the separation to verify is
linear-blindness vs the nonlinear (post-squash / spectral) access route, as
in the changepoint § 8 treatment.

## 5. Predictions (frozen in the card, before any run)

- **per-token SAE:** reads the phase class `c_t` at/near oracle; **blind to
  `e_t`** beyond its chance floor (a linear readout of one position cannot
  compare two).
- **additive window families (TXC-pre, Stacked):** near-oracle on `c_t`;
  **blind to `e_t`** (additive-blindness to equality patterns — changepoint
  precedent).
- **coincidence (TXC-post) and Spectral-TXC:** the only families predicted
  to expose `e_t` above chance (changepoint boundary result: τ 0.66 at T=2
  for post-squash), paying the usual content/NMSE price.
- **Falsifier to watch:** if trained per-token codes read `e_t` above the
  raw-access line, the substrate has leaked equality into single-position
  features (e.g. boundary-marking emission artifacts) — that would be a
  NEGATIVE verdict on the bench, not a win for per-token.

## 6. Caveats carried from calibration

- Heuristic cross-check is weak (accuracy 0.30, κ 0.03 per-class F1 ≤ 0.08)
  — the inter-judge floor (κ 0.609, ε̂ 0.137) is the binding validation.
- Class 1 (context-background) dominates at 50.5%; extreme dwells (60–76)
  trace to homogeneous non-instructional docs labeled uniformly — bounded by
  N1 and absorbed into the doc-propensity tail, but a capacity design that
  starves rare classes (2–4 at 6–7%) should anchor probes balanced.
- The mirror's cross-doc heterogeneity level is inserted (empirical
  propensity list); its lag structure is gate-8-verified. The reasoning-trace
  sibling (`proof-operation-phase-runs-r2`) ABORTED under the same mirror —
  reasoning traces carry a **segment-scale layer** (three timescales) the
  doc-level hierarchy cannot hold; do not assume this spec transfers across
  domains.

## Amendment 2026-07-23 — § 5 re-freeze under the re-scoped regime-3 residual axis (stage-6 #3b)

**Provenance.** The § 8 equality-variant STOP (2026-07-22; gating record +
review in `bench_record.md`) established the access ladder on this substrate:
chance 0.5 → from-`c_t` DC leak 0.609 → per-token raw-linear 0.614 → window
raw-linear 0.720 → **pair-additive ceiling 0.771** → nonlinear/exact 1.000.
The mac-local review (2026-07-23) adopted **re-scope option 1**: the primary
axis is the **regime-3 residual** — `equality_residual_recovery` =
(balacc(e_t) − 0.771) / (1.000 − 0.771), unclipped — the 0.229 of balanced
accuracy reachable only by position-mixing. The DC leak and the additive
access are named floors, not noise. The predictions below were **frozen by
mac-local in `briefings/stage6-recipe-rescoped.md` (2026-07-23)** and are
restated verbatim with sharpening reasons only; no direction was changed and
nothing here was written after any cell ran.

### § 5-r — frozen per-arch predictions (residual axis)

- **DC control `c_t`:** every arch at/near oracle (per-token 1.000 raw —
  expected; it is the control).
- **Residual axis (primary):**
  - `batchtopk_sae`, `tsae`, `stacked_batchtopk`, `txc_batchtopk_pre` —
    **≈ 0 (or negative)**: a linear probe over per-position codes is an
    additive-over-positions readout, bounded by the 0.771 ceiling; TXC-pre's
    summed code is additive up to BatchTopK competition (changepoint's
    provable additive-blindness precedent). *Sharpening reason: the probe
    reads one tile's code at the leading edge, so any e_t information must
    survive the code's own compression of the additive route — these
    families are expected below the ceiling, i.e. residual < 0, not merely
    ≈ 0.*
  - `txc_batchtopk_post` — **positive residual**, strongest at T = 2,
    expected k_pos-fragile (the changepoint boundary precedent: τ 0.66 at
    T=2, vanishing at higher k). *Sharpening reason: post-squash reuses each
    atom across all T positions, so a coincidence atom can fire on the
    (c_{t-1} = c_t) conjunction — exactly the position-mixing route the
    residual isolates.*
  - `spectral_txc` — **positive residual**, expected k_pos-robust
    (changepoint tss/cp precedent). *Sharpening reason: DCT bands mix
    positions before the sparsity competition, so the equality signal is not
    fighting per-position atoms for budget.*
- **Untrained control:** any positive residual must vanish (or drop to the
  architectural-access floor) at random init, else it is access, not
  learning — report both.
- **Falsifier (indicts the bench, not an arch):** any additive-family arch
  with residual clearly > 0 beyond seed noise ⇒ the residual normalization
  or the substrate leaks beyond the § 8-measured additive ceiling ⇒ verdict
  NEGATIVE on the re-scoped bench; report, never re-normalize after the
  fact. Threshold-optimized ceilings are the raw-access reference lines
  (README rule; probe-convention numbers are diagnostics).

_Amended-by: claude-fable-5 (runpod, stage-6 #3b), restating the mac-local
freeze; committed BEFORE any grid cell ran (freeze order provable from the
git log)._
