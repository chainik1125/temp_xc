# FB-4 — rotated multilane (the basis-alignment acid test)

**Axis-point card, frozen per LOOP.md before any construction.**
Status: FROZEN (2026-07-23, runpod-b, cycle FB-C2; construction directive +
frozen prediction directions from mac-local, `briefings/freqbench-t16-fbc2.md`
§ Phase 3 — directions preserved verbatim, reasons appended per the briefing's
"append reasons, never directions" rule). Provenance: `theorem-first`.

---

## 1. Target coordinates + gap claim

**Coordinates:** identical to FB-2 (`cards/FB-2.md` § 1) — axis 1 AC/2nd-moment
3-line spectrum, axis 2 order-2/position-mixing, axis 3 stationary. FB-4 is
*deliberately* at FB-2's exact coordinate point: it is a **controlled
experiment on basis alignment**, not a new coordinate. The one knob: a fixed
orthogonal rotation of the `d_in` embedding between the latent's lane planes
and whatever basis each architecture brings.

**Gap claim.** The FB-C1 review baked the order-2 subtype rule into the README
coordinates (phase→post · power/equality→spectral · covariance→pre). Its
sharpest confound: is spectral's power/equality dominance **generic
order-2-even structure, or DCT-basis alignment**? FB-4 attacks that confound
with everything held fixed except alignment.

**Registry-cited redundancy risk (stated at freeze, per LOOP § card-1).**
FB-2 sits at these exact coordinates, so this card lives or dies on the knob
being *real*. The § 3 absorption obligation is exactly that check, and T2's
symmetry audit is its empirical arm. If the knob is shown inert, the card
ABORTs at T2 as redundant-by-symmetry — recorded in BENCHMARKS § B, which is
the process succeeding, not failing (prime directive).

## 2. Constructed task

The FB-2 generator composed with a **fixed, seeded Haar-random orthogonal
rotation of the embedding space**:

```
Q ∈ O(d_in)  fixed:  QR of a standard Gaussian (d_in × d_in),
             rotation_seed = 777  (ONE realization, shared by every cell and
             every data seed — NOT re-drawn per seed)
x_t = Q · x_t^multilane        (labels, lane latents Y_k, B_k unchanged)
lane planes exposed to evals:  Q · P_k   (ground truth rotated consistently)
```

Everything else identical to FB-2, frozen: `M = 101`, `Ω = {0,1,2,4,8,16,24,
32,40,50}`, `n_lanes = 3`, `d_in = 24`, `σ = 0.25`, `seq_len = 64`,
`n_seqs = 4096`, eval `L = 32`, capacity anchor `F = M = 101`. Datasource:
`toy_multilane_rotated_M101_d24` (thin generator `multilane_tones_rotated`
wrapping `multilane_tones` — append-only; the base generator's signature is
untouched). Primary metric: `multilane_recovery`, unchanged (the evaluator
reads the exposed rotated planes; oracle and probes are basis-consistent by
construction).

**Ground truth:** as FB-2 § 2 with every direction rotated by Q: strict
feature directions = the 6 rotated plane axes; hidden latents Y_k, B_k
unchanged; the `3M` codebook atoms rotate with the planes.

## 3. Proof obligations (PORT.md § B format)

- **Ceiling (P5, per lane) — rotation-invariant, restated not re-derived.**
  Orthogonality of the planes is preserved by Q (`(QP)ᵀ(QP) = PᵀP`), the
  noise is isotropic (`Qε =d ε`), so projecting onto the rotated plane `QP_k`
  removes the other lanes exactly and the per-lane periodogram over Ω remains
  the ML decoder at the same (T, σ). Per-lane ceiling = FB-2's ceiling,
  number for number. **Discharge:** numerical in `gating.py` — per-lane
  oracle on the rotated generator must match FB-2's recorded oracle at
  matched (T, σ) within seed noise.
- **Floor (P1/P2) — rotation-invariant, restated.** P1: `x_t = Q u(Q_t) + Qε`
  is a fixed measurable per-token map of the FB-2 token; the DPI gives
  `I(Y_k; x_t) = 0` still, exactly. P2 phase-averaging survives any fixed
  per-token transformation (the additive-readout means remain
  velocity-independent). **Discharge:** analytic (this paragraph) + T2 bag
  control empirically.
- **⚠ The absorption obligation (T1-critical, added at freeze — the reason
  side of this card; direction claims in § 6 are untouched).** FB-2's
  embedding isometry `P` is itself Haar-random **and is re-drawn per data
  seed** (the runner passes the run seed into the generator;
  `_generate → params["seed"]`). For any fixed `Q ∈ O(d_in)`, `QP` is
  Haar-distributed iff `P` is: **the composed generator is
  distribution-identical to FB-2, jointly over data and every exposed ground
  truth.** Every architecture, probe, and oracle in the panel is a function
  of `(x, ground truth)` with spatially isotropic (or data-derived)
  initialization, so every grid statistic — trained, untrained, per-cell —
  has *identical distribution* on FB-4 and FB-2; across-seed means can
  differ only by seed noise. In particular the FB-2 untrained-spectral
  figure (+0.298 at the T=8 frontier) is already a mean over three
  independent embedding draws. If this argument is correct, the rotation
  knob is **provably inert**, and the alignment the card wants to test
  (spectral's DCT prior) is *temporal*, which no spatial rotation can touch.
  **Discharge:** analytic (above) + numerical equivalence check in the T2
  battery (matched-statistic comparison rotated-vs-base across seeds, and a
  small matched arch panel at the anchor cell falling inside FB-2's recorded
  seed band). If the check confirms absorption ⇒ **ABORT at T2**
  (symmetry-triviality: a group action maps the task to itself in
  distribution); if it *refutes* absorption (any statistic separates beyond
  seed noise) ⇒ the argument is wrong somewhere, the knob is live, and the
  card proceeds to § 8 gating and the grid as specified.
- **Non-triviality (P3/symmetry).** As FB-2 § 3 for the task itself. The new
  symmetry question FB-4 itself introduces is exactly the absorption
  obligation above.

## 4. Regime claim + design-time discriminability

Regime 3 by construction (inherited — the rotation changes no temporal or
order structure). The discriminability the card *claims* is not per-arch
regime separation (FB-2 already established that at these coordinates) but
**FB-4-vs-FB-2 contrast per arch**. Design-time statement of what the proofs
predict apart: under the absorption obligation, the proofs predict **nothing
comes apart** — the expected T2 outcome is NON-DISCRIMINATING-BY-SYMMETRY and
an ABORT before any grid. The card runs its gates to make that verdict
*measured and citable* rather than assumed. (Should T2 instead find a live
knob, the § 6 table becomes the frozen prediction set for a standard grid.)

## 5. Memorization audit (P6)

Identical to FB-2 § 5: `|Ω|³·M³ ≈ 1.03 × 10⁹` templates vs `d_sae ≤ 202`;
rotation neither creates nor destroys templates (Q is a bijection on
activation space and the template count is basis-independent). Memorization
route dead by construction at every cell; no threshold crossing predicted.
The T2 memorization-budget run is inherited from FB-2's, unchanged by
rotation — stated here per the briefing; re-verified only through the
equivalence check rather than a fresh budget sweep.

## 6. Frozen per-arch predictions (+ falsifier)

**Directions frozen by mac-local (briefing § Phase 3), preserved verbatim;
bracketed reasons appended by runpod-b at freeze.**

| claim | frozen direction | appended reason (runpod-b) |
|---|---|---|
| per-token / stacked / txc-pre | stay ≈ 0 | [P1/P2 are rotation-invariant — this holds under *both* branches of the absorption obligation] |
| spectral untrained access prior | **collapses**: +0.298 → ≈ 0 ("its DCT kernels no longer align") | [absorption argument predicts this direction FAILS: the DCT prior is temporal, Q is spatial, and +0.298 is already embedding-averaged; expected outcome ≈ +0.30 ± seed noise. Recorded as the frozen direction regardless — the miss, if it happens, is the datum] |
| trained spectral vs post margin | **open question the bench decides**: full recovery ⇒ learned order-2-even conversion, subtype rule survives as stated; collapse toward/below post parity ⇒ "power→spectral" is alignment-conditional and the README rule gains an alignment qualifier (equally valuable — stated per the briefing) | [under absorption, "full recovery" is guaranteed a priori, i.e. the question is decided by theorem, not by the bench — which is the T2 kill, not a verdict] |

**Falsifier (frozen verbatim):** any arch > 0.1 at **T=1** ⇒ rotation bug
leaking per-token access (P1 must survive Q exactly) — STOP and debug, never
report.

## 7. Skeptic pre-registration notes

`a_proof_circularity`: the absorption obligation is itself the sharpest risk —
*if wrongly argued*, it would kill a live card; hence it is discharged
numerically (T2 equivalence check), not by algebra alone.
`b_triviality` / `d_redundancy`: § 1 states the kill condition openly — FB-4
at FB-2's coordinates is redundant *unless* the rotation knob is live; the
expected outcome at freeze is an ABORT at T2 with the absorption theorem as
the recorded reason. `c_relevance`: inherited from FB-2 (superposition is the
defining SAE regime); the alignment question itself is real — but its live
knob is plausibly *temporal* (mixing the within-window time basis), which is
a **different card** (a candidate FB-5), explicitly NOT proposed here per the
briefing's "no cards beyond FB-4" hard line; left for mac-local review.
`e_substrate`: no panel or convention deviation; one thin generator +
datasource append; canonical runner; uniform grid T ∈ {1,2,4,8} (T=16
excluded per the briefing) *if* gating is reached.
