# FB-1 — phasepair (± velocity pairs: phase-vs-power dissociation)

**Axis-point card, frozen per LOOP.md before any construction.**
Status: FROZEN (2026-07-23, runpod-b, cycle FB-C1 Phase 3). Provenance:
`theorem-first` (sprint `phasepair` task, `fb_core.make_task`; the sign
remark in sprint theory.md; P4/P5 in PORT.md § B).

---

## 1. Target coordinates + gap claim

- **Axis 1 (spectral):** AC, three discrete magnitude lines (f ≈ 0.030,
  0.119, 0.297 cycles/token) — but the PRIMARY latent is not the line
  position: it is the **rotation direction at each line**. FreqFrac
  coordinates at bench time (T4).
- **Axis 2 (interaction order):** order-2, and *stronger*: the sign latent
  is invisible not just to additive readouts (P2) but to **every per-channel
  second-moment statistic** — ±y trajectories are time reversals with
  identical per-channel window power spectra AND identical symbol-set
  distributions; the sign lives only in the cross-channel quadrature (which
  circle axis leads). Power detectors are blind by construction.
- **Axis 3:** stationary / spread.

**Gap claim.** `frequency` and `multilane` discriminate on *tone detection*
(power suffices once positions mix); no bench yet tests **phase**: whether a
code distinguishes structure with identical power spectra. `signed_motion`
was exactly this question (its `ac_sign` ancestor) but forked WITHOUT the
proof apparatus, on a random embedding (no geometry ⇒ P3 symmetry applies),
and died of the `#windows = 2F` memorization confound — the registry's one
"substrate defect" row. FB-1 is that question rebuilt on the proofs: circle
geometry (P4 restores frequency semantics), 6 classes over 3 ± pairs
(|Ω|·M = 606 templates vs d_sae ≤ 202 — the confound is capacity-controlled),
and a metric that separates the power-readable component (pair identity)
from the phase-only component (sign within pair). Not redundant: no
existing bench's primary latent survives conditioning on the full
per-channel power spectrum.

**c_relevance (the real phenomenon — answered, not `spanning`):** the
sprint's GPT-2 day-stride measurement (PORT.md § A row; sprint § 4.7):
weekday sequences at constant stride, labels INCLUDING the sign pairs
(1,6), (2,5), (3,4) on the 7-day circle. On the embedding layer (where P1
genuinely applies) single-position probes sit at chance (0.149) while
window-mixing codes read the 7-class stride — direction included — at
**1.000**; GPT-2's own attention converts it to single-position-linear by
block 3. Direction-of-motion on a real model's cyclic geometry is a
measured, phase-coded, real-LM latent; FB-1 is its controlled synthetic
counterpart.

## 2. Constructed task

**No new generator** — the existing `cyclic_tones` (frequency bench, P1–P5
apparatus attached) at a different frozen velocity set:

```
Q_t = (B + Y·t) mod M,  B ~ Unif(Z_M),  Y ~ Unif(Ω)
Ω = (3, 98, 12, 89, 30, 71) = ±3, ±12, ±30 (mod 101)  — 6 classes, chance 1/6
circle embedding:  u_a = R[cos 2πa/M, sin 2πa/M],  R Haar d_in×2
x_t = u_{Q_t} + σ ε_t
```

**Frozen parameterization:** `M = 101`, `d_in = 24`, `σ = 0.10` (the
frequency bench's SNR), `seq_len = 64`, `n_seqs = 4096`, `L = 32`.
Datasource `toy_phasepair_M101_d24` (data.yaml entry only, append-only).
**F anchor = M = 101** (`d_sae ∈ {50, 101, 202}`, frequency precedent).

**Ground truth (Part II § 1).** Feature directions: the M circle atoms
(rank-2 codebook; eauc ill-defined — capability via NMSE, the frequency
convention). Latents: `Y` (6-class), decomposed as **pair id**
`|Y| ∈ {3, 12, 30}` (power-readable) × **sign** `s = ±` (phase-only,
PRIMARY). Nuisance: `B`.

**Metrics.** `velocity_recovery` (existing `frequency_metrics`, 6-class,
periodogram oracle — the complex matched filter distinguishes ±y) PLUS a
small additive add-on `phasepair_metrics` firing only when Ω contains ±
pairs (no-op for the frequency bench — key absence keeps old rows
byte-identical): `pair_recovery` (3-class magnitude, normalized) and
**`sign_recovery`** (within-pair sign balanced accuracy over tiles whose
true pair is known, normalized to [0, 1] from chance ½ — the phase-only
headline), probe recipe identical to `frequency_metrics` (per-tile
leading-edge, linear, leak-free).

## 3. Proof obligations (PORT.md § B format)

- **Ceiling (P5, signed).** The complex circle signal `c_t = e^{i(θ_B +
  2πYt/M)}`: the periodogram matched filter over SIGNED frequencies is the
  ML decoder for the 6-class task; ±y are distinct complex exponentials, so
  the oracle reads both pair and sign. Sign resolution needs phase
  evolution ≥ ~1 radian across the window: sign oracle high for
  |Y|·T/M ≳ 1/(2π), degrading toward ½ for the slowest pair at small T (the
  quantitative curve is the gating deliverable). **Discharge:** numerical —
  oracle 6-class, pair, and per-pair sign accuracy vs T on the built
  generator; committed in gating.
- **Floor 1 (P1/P2):** per-token I(Y; x_t) = 0 (B uniform); additive
  readouts mean-degenerate. Same discharge as frequency (transfers
  verbatim — same generator).
- **Floor 2 (the phase floor — the card's own theorem).** Within a pair,
  the ±y window ensembles are related by time reversal t ↦ −t
  (equivalently complex conjugation of c_t up to a phase): (i) each
  channel's window power spectrum / autocovariance is invariant ⇒ any
  detector computing per-channel second moments is sign-blind; (ii) the
  symbol multiset {B + yt} ≡ {B′ − yt} in distribution (B uniform absorbs
  the reflection) ⇒ **bag-of-symbols is EXACTLY sign-blind** (the strongest
  bag null in the program — unlike frequency/multilane where the bag
  carries spread cues, here the bag distribution is identical within
  pairs). Sign requires the cross-channel (quadrature) phase — an
  order-sensitive, position-mixing read. **Discharge:** analytic (above) +
  T2 bag control must sit at sign-chance exactly.
- **Non-triviality (P3/P4).** Circle geometry ⇒ no relabeling collapses
  the classes (P3 fails by design); the random-frame variant of this task
  IS `signed_motion`'s regime — kept OUT (the registry's defect row is the
  cautionary null). Memorization: |Ω|·M = **606** clean windows vs
  d_sae ≤ 202 (3× margin at the top; the sweep never crosses the P6
  threshold — no jump predicted; contrast signed_motion's 2F = 38 vs 20).

## 4. Regime claim + design-time discriminability

**Regime 3**, sharpest form: the sign latent has zero per-token MI
(provable), zero raw-LINEAR window readability (E[x|Y] ≈ 0), zero
bag readability (exact), zero per-channel-power readability (exact) — the
only raw route is the nonlinear cross-channel phase read, witnessed by the
signed periodogram oracle (equality-variant gate: raw-linears at chance +
oracle ≫ chance at T ∈ {4, 8}). Proofs predict apart: additive family
(token, stacked, txc-pre) sign-blind; mixing archs (txc-post, spectral)
CAN build quadrature detectors. Within spectral: a DCT (real cosine) band
atom couples time-phase to the 2 spatial axes — quadrature pairs are
representable; whether TRAINING finds them is the bench question. STOP
conditions: sign oracle < 0.75 at every T ≤ 8 for the two faster pairs ⇒
non-discriminating (window too short for phase) — no grid.

## 5. Memorization audit (P6)

Template count |Ω|·M = 606 ≫ d_sae max 202; below threshold everywhere in
the sweep, no predicted jump. Probe budget: ≥ 100× code dim (30k rows vs
≤ 202 dims). The signed_motion lesson is the reason this card EXISTS in
capacity-controlled form; the contrast (their 38 templates vs d_sae 20)
goes in the record.

## 6. Frozen per-arch predictions (+ falsifiers)

`sign_recovery` (normalized [0,1] from chance ½) at d_sae = F, T = 8,
matched B*; `velocity_recovery` in parentheses:

| arch | prediction | reason |
|---|---|---|
| batchtopk_sae, tsae | ≈ 0 (≈ 0) | P1 — provable |
| stacked_batchtopk | ≈ 0 (< 0.1) | additive probe + exact bag-null |
| txc_batchtopk_pre | ≈ 0 (0.1–0.3) | additive code: pair magnitude leaks via variance (the frequency 0.27 route reads |Y|), sign exactly dead |
| txc_batchtopk_post | sign 0.2–0.6 | quadrature is learnable post-mixing; harder than tone detection (no power gradient to follow) |
| spectral_txc | sign 0.3–0.7; untrained sign ≈ 0 | band kernels can pair into quadrature but the DCT prior is power-aligned, not phase-aligned — predict its edge over txc-post SMALLER than on frequency/multilane, possibly nil |
| pair_recovery (all mixing archs) | ≥ their frequency-bench level | pair id is plain tone detection |

**The dissociation prediction (the headline):** every arch's
`pair_recovery` ≥ its `sign_recovery` (power is easier than phase), and
the additive family shows the pure dissociation: pair > 0, sign ≈ 0.
**Spectral-vs-post prediction:** the spectral advantage, decisive on
frequency/multilane (power tasks), SHRINKS on sign (its prior is
band-power, not phase) — if spectral ≫ post on sign too, the DCT-band
prior is also a phase prior (unfrozen surprise worth flagging).

**Falsifiers:** (1) any arch sign_recovery > 0.1 at T = 1 ⇒ P1 bug;
(2) bag control off sign-chance ⇒ the exact bag-null derivation is wrong ⇒
STOP (theory bug, not a result); (3) sign oracle ≈ ½ everywhere ⇒
non-discriminating, no grid; (4) trained ≈ untrained for the sign winner ⇒
access artifact.

## 7. Skeptic notes

`c_relevance`: answered with a real measurement (§ 1 — GPT-2 day-stride
direction; not `spanning`). `d_redundancy`: § 1 (no bench's primary
survives power-spectrum conditioning; signed_motion is a defect row, not a
discriminator). `e_substrate`: no generator change; datasource + additive
metric only; panel/conventions untouched.
