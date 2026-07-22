# FB-2 — multilane superposition (3 simultaneous circle tones)

**Axis-point card, frozen per LOOP.md before any construction.**
Status: FROZEN (2026-07-22, runpod-b, cycle FB-C1). Provenance:
`theorem-first` (sprint § 4.6, `origin/dmitry-spectral-sprint2`
`run_multilane.py` + `summary.md`; proofs registry PORT.md § B).

---

## 1. Target coordinates + gap claim

**Coordinates (README § "The two generators"):**

- **Axis 1 (spectral):** AC / 2nd-moment, high-band — three simultaneous
  tones per window (each lane a cyclic tone at `f_k = Y_k/M`); the window
  spectrum is 3-line, not 1-line. FreqFrac coordinates to be measured at
  bench time (gate T4).
- **Axis 2 (interaction order):** order-2 / position-mixing. Velocity is a
  2nd-moment latent (phase progression); per-token and additive-over-time
  readouts are information-dead (P1) / mean-degenerate (P2).
- **Axis 3 (localization):** stationary / spread (tones are
  time-homogeneous within the window).

**Gap claim.** The registry's only regime-3 spectral bench (`frequency`) is
**single-tone**: one active temporal latent per window, and its band-partition
addendum found multiband ≈ full-band at matched budget (a preregistered TIE at
T=16, with a multiband edge only at T=2 and in untrained access). FB-2 occupies
the **superposition** point of the same axis — 3 independent tones competing
for one sparsity budget — which no existing bench covers. Not redundant with
`frequency` because (a) the discriminating mechanism claimed by the sprint
(per-band budgets prevent one lane crowding out the others under TopK
scarcity) only exists under superposition, and (b) the memorization loophole
that forced frequency's A5 stacked caveat and its `|Ω|·M` budget audit is
**dead by construction** here (§ 5). If FB-2 also comes back multiband ≈
full-band, the sprint's plain-TopK separation is indicted — an informative
negative and a citable verdict.

## 2. Constructed task

Three independent lanes, each a circle-embedded cyclic tone in its own
2-plane; the three planes are mutually orthogonal, embedded in `R^{d_in}` by
one Haar-random isometry:

```
P ∈ R^{d_in × 6}  orthonormal (QR of a Gaussian), planes P_k = P[:, 2k:2k+2]
Y_k ~ Unif(Ω), B_k ~ Unif(Z_M)   independently per lane k ∈ {0,1,2}
θ_k(t) = 2π (B_k + Y_k·t mod M) / M
x_t = Σ_k [cos θ_k(t) · P_k[:,0] + sin θ_k(t) · P_k[:,1]] + σ ε_t,  ε_t ~ N(0, I)
```

**Exact parameterization (frozen):** `M = 101` (prime), `Ω = {0, 1, 2, 4, 8,
16, 24, 32, 40, 50}` (the frequency bench's ladder, chance 0.1/lane),
`n_lanes = 3`, `d_in = 24`, `σ = 0.25` (the sprint's SNR — the transported
prediction's provenance), `seq_len = 64`, `n_seqs = 4096`, eval `L = 32`.
Datasource: `toy_multilane_circle_M101_d24` (generator `multilane_tones`,
append-only additions to `synthetic.py` / `data.yaml`).

**Ground truth (Part II § 1).** Feature directions in the strict sense: the
**6 plane axes** `{P_k[:,0], P_k[:,1]}` (the activations are built from them;
signal subspace rank 6). Hidden latents (NOT directions): `Y_1, Y_2, Y_3`
(categorical, 10 classes each — the three headline latents), `B_1, B_2, B_3`
(nuisance phases, `Z_M` each). Derived (not ground truth, must not size
`d_sae`): the `3M = 303` per-lane circle atoms (reconstruction codebook), the
`|Ω|³M³` whole-window templates. **Capacity anchor `F = M = 101`** (per-lane
alphabet), following the frequency precedent: `d_sae ∈ {50, 101, 202}`, all
≪ the template count; comparability rides on the normalized metric, and the
strict direction count (6) is not the useful dictionary scale for a tone task
(quadrature pairs per frequency per lane ~ `|Ω|·n_lanes`-scale ≪ M).

**Primary metric:** `multilane_recovery` — mean over lanes of the per-lane
multinomial-logistic probe on the shared per-tile code (leading-edge,
per-tile, leak-free split — the `frequency_recovery` conventions),
normalized to [chance = 0.1, 1]. Per-lane per-class recalls reported (the
3-lane S(f)). Evaluator add-on `multilane_recovery` is required — the
existing `frequency_metrics` reads a single per-sequence velocity and cannot
express 3 simultaneous targets. Additive dispatch on
`extra['lane_velocity_labels']`; protocol stays 1.3.0.

## 3. Proof obligations (PORT.md § B format)

- **Ceiling (P5, per lane).** Orthogonal planes ⇒ projecting onto `P_k`
  removes the other two lanes *exactly*; the residual channel is one unit
  tone in white noise ⇒ the per-lane periodogram argmax over Ω is the ML
  decoder for lane k (classical single-tone estimation, Rife–Boorstyn), and
  the per-lane ceiling equals the single-tone oracle at the same (T, σ).
  Rayleigh: resolution ∝ 1/T ⇒ at T ≤ 8 the low-Ω cluster {0,1,2,4} is
  SNR-limited (the frequency bench's measured oracle curve). **Discharge:**
  numerical — per-lane oracle accuracy on the actual generator across
  T ∈ {2,4,8} at σ=0.25, compared against the single-lane oracle at matched
  (T, σ) (agreement ⇒ the orthogonal-plane separation claim holds); committed
  in `gating.py` (T1).
- **Floor (P1 per token, P2 additive).** P1: `B_k` uniform ⇒ `Q_k,t | Y_k ~
  Unif(Z_M)` per token ⇒ `I(Y_k; x_t) = 0` exactly — any per-token encoder is
  information-dead for every lane. P2 (phase-averaging): any readout additive
  over time on arbitrary per-token codes has velocity-independent
  class-conditional means ⇒ no perfect separation; empirically ≈ chance
  (the theorem bounds means only — a linear probe can exploit
  variance differences, so small positive leakage is consistent; see § 6
  predictions). Applies to: probes on per-token codes (batchtopk_sae, tsae),
  concatenated per-position codes (stacked), and the txc-pre code (gated
  per-position then summed — additive over t by construction).
  **Discharge:** analytic (the P1/P2 arguments transfer verbatim per lane —
  written in the record) + the T2 bag control empirically.
- **Non-triviality (P3 / symmetry).** The circle embedding carries metric
  geometry on symbols — there is no relabeling group action mapping
  velocity classes to each other while fixing the data distribution (the
  random-frame exchangeability argument that trivialized the 10-frequency
  proposal does NOT apply; the retained null for that is the frequency
  bench's random-embedding datasource, not re-run here). Order route
  required: velocities live in phase *progression*; the bag-of-symbols
  route is killed empirically in T2 (mean-pooled codes + MLP must fail).
  **Discharge:** analytic note + T2 controls.

## 4. Regime claim + design-time discriminability

**Regime 3** (order-2 / position-mixing latent) by construction: per-token
readable = provably nothing (P1); linear-in-window (raw-linear on the stacked
window) ≈ chance because `E[x_t | Y_k] ≈ 0` (phase-uniform ⇒ class-conditional
means vanish; the equality-variant § 8 situation, same as frequency). The
proofs predict apart: **coincidence/spectral codes** (txc-post, spectral —
nonlinearity after position-mixing) CAN linearize the tone latents;
**additive codes** (per-token, stacked, txc-pre) CANNOT (P2). Design-time
discriminability: the § 8 gate must show (i) both raw-linear readouts
(per-token, window-concat) ≈ chance, (ii) the per-lane periodogram oracle on
the same raw tiles ≫ chance at T ∈ {4, 8} — the changepoint-style
equality-variant treatment. If (ii) fails at T ≤ 8 (Rayleigh too coarse at
σ=0.25 for most of Ω), the bench is non-discriminating at this window range:
record NON-DISCRIMINATING and STOP (no grid).

## 5. Memorization audit (P6)

Whole-window template count = `|Ω|³ · M³` = 10³ · 101³ ≈ **1.03 × 10⁹** clean
windows (before noise), vs `d_sae ≤ 2F = 202` and probe features ≤ d_sae per
tile. The memorization route (P6: window archs solving structureless tasks by
template lookup above `|Ω|·M`) is dead by construction at *every* grid cell —
no capacity-threshold bookkeeping needed, no A5-style stacked caveat. The
`d_sae` sweep therefore cannot cross a memorization threshold and no jump is
predicted; the probe budget (T2) still scales with code dim per the F-rule.

## 6. Frozen per-arch predictions (+ falsifiers)

At the canonical per-token-matched cell (T=4 window / T=1 token, d_sae=F=101,
realized l0/token ≈ 2) and the T=8 frontier, `multilane_recovery` normalized
[0,1]:

| arch | prediction | reason |
|---|---|---|
| batchtopk_sae | ≈ 0.00 (< 0.05) | P1: zero per-token MI; probe on per-token codes is additive (P2) |
| tsae | ≈ 0.00 (< 0.05) | same |
| stacked_batchtopk | < 0.10 | P2 additive probe on concatenated per-position codes; no memorization escape (§ 5) — unlike signed_motion/frequency>|Ω|M, this number is clean |
| txc_batchtopk_pre | 0.05–0.30, flat per-lane S(f) | additive code (P2) ⇒ no spectral estimation; small variance-leakage only (frequency measured 0.27 single-tone; interference should reduce it) |
| txc_batchtopk_post | positive, 0.3–0.7 at T=8 | position-mixing before ReLU converts phase structure (frequency: 0.53); superposition + shared budget degrades vs single-tone |
| spectral_txc | **best trained arch**, 0.6–0.9 at T=8 | DCT-band prior = tone detectors (frequency: 0.96 at T=8); untrained access residual expected ≫ other archs' untrained (frequency: 0.64) |

**Ordering claims (the testable core):** spectral > txc-post > {txc-pre,
stacked, token} at T ∈ {4, 8}, all three additive-family archs below 0.3, and
a positive T-trend for the mixing archs (T=8 > T=4 > T=2: Rayleigh).

**The sprint-transported headline (band-partition addendum, matched budget
k_pos=1, spectral_txc 4-band vs `spectral_txc_full` 1-band vs
`spectral_txc_dcac` 2-band, the frequency-A6 pattern):** the sprint claims
**multiband > vanilla full-band under superposition** (0.96 vs 0.91 per-lane,
no seed overlap, at H=256/W=16/plain-TopK). Frozen transported prediction:
4-band > 1-band by ≥ 0.03 mean per-lane recovery at the T=8, d_sae=101,
k_pos=1 cells, no seed overlap. **This may FAIL under the fair BatchTopK
backbone at T ≤ 8** — batch-pooled budgets already prevent per-window
crowding, which is the sprint's proposed mechanism. A tie/reversal is an
informative negative about the sprint's plain-TopK result and is reported as
such (prime directive: the verdict, not the win).

**k_pos structure prediction:** the per-token demand of 3 simultaneous lanes
means the scarce slices k_pos ∈ {1,2} force lane competition; the winning
arch's margin over txc-post should be *largest* there and shrink by k_pos=8
(the changepoint scarcity-forcing analogue, direction reversed).

**Falsifiers (substrate-indicting, not winner-crowning):**
1. Any arch with `multilane_recovery` > 0.1 at **T=1** ⇒ P1 violated ⇒
   generator/evaluator bug — STOP and debug, never report.
2. Trained ≈ untrained for the winning arch (gap < the seed spread) ⇒ the
   "win" is architectural access, not learning — report as access.
3. Per-lane oracle at T=8 materially below the matched single-lane oracle ⇒
   the orthogonal-plane ceiling argument fails ⇒ the task is not the theorem's
   task — ABORT at T1.

## 7. Skeptic pre-registration notes

`c_relevance`: superposition is the *defining* regime of real LM residual
streams (features in superposition is the SAE field's founding premise);
axis-1 measurements on real streams show multi-band structure (PORT.md § A:
GPT-2 day-stride circles; backtracking DC + frequency high-band both live on
the same panel). The card is `spanning` in the strict sense that no single
PhenomenonBench measurement pins 3-tone superposition; the research reason is
that it is the minimal task where band-partition priors *can* matter under
scarcity — the sharpest open arch-separator claim from the sprint.
`d_redundancy`: see § 1 (frequency = single-tone; its addendum tied).
