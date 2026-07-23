# FB-5 — permuted tones (the temporal-knob acid test)

**Axis-point card, frozen per LOOP.md before any construction.**
Status: FROZEN (2026-07-23, runpod-b, cycle FB-C3; construction + frozen
prediction directions from mac-local, `briefings/freqbench-fb5.md` —
directions preserved verbatim, reasons sharpened per the briefing).
Provenance: `theorem-first`.

---

## 1. Target coordinates + gap claim

**Coordinates:**

- **Axis 1 (spectral): AC / 2nd-moment, BROADBAND.** The trajectory's
  temporal law is a uniformly-random walk-through of the circle — lag-1
  circle autocorrelation ≈ 0 ± O(1/√M) per schedule (measured at freeze:
  −0.12…+0.17 across 10 draws), vs the tone benches' sinusoidal ±1 ladder.
  This is the axis-1 pole the suite lacks: every existing AC bench
  (frequency, multilane, phasepair) is narrow-band/line-spectrum.
- **Axis 2 (interaction order): order-2 / position-mixing** — P1 per-token
  exactly dead, P2 additive-linear dead (§ 3); the class latent is the
  trajectory *shape*.
- **Axis 3: stationary** (schedule is time-homogeneous within windows).

**Gap claim.** Two gaps at once. (i) The registry has no broadband-AC
bench: FreqFrac's axis-1 reads currently span DC-heavy (hedging,
backtracking) to line-spectrum (tones); the broadband pole is unpinned.
(ii) The FB-C1 subtype rule's power leg ("power/equality → spectral") has
an unresolved alignment confound that FB-4 proved a spatial knob cannot
test (absorption). The temporal knob can: FB-5 vs `frequency` is a
controlled comparison in which order-2-even structure and every substrate
convention are held fixed while **DCT-alignment of the temporal trajectory
is destroyed**. The tone benches are literally the special case
π_Y(z) = Y·z (Y coprime to prime M is a permutation); FB-5 replaces that
linear-map subfamily with generic permutations.

**Non-absorption obligation (LOOP card item 1, adopted at the FB-4
review).** The knob changes the *law of the temporal trajectory ensemble*,
not a basis composed with a randomized one. Nothing in the substrate can
absorb it: the embedding randomness (the Haar 2-plane R) acts spatially
and is knob-independent; the offset B randomizes phase in BOTH ensembles;
the schedule ensemble itself IS the knob, and linear-phase schedules are
measure-zero among permutations. The ensembles differ in law by a concrete
statistic — per-class lag-1 trajectory autocorrelation (tones: the
deterministic ladder cos(2πY/M) ∈ [−1, 1]; permutations: ≈ 0 ± O(1/√M)) —
verified numerically at freeze and re-verified in gating on the built
generator. And no panel architecture is equivariant to temporal
reindexing: the DCT prior is a fixed temporal basis, so a change of
temporal law is visible to it (unlike FB-4's spatial Q, to which the whole
panel was distribution-equivariant).

## 2. Constructed task

The frequency substrate with one substitution — the linear phase schedule
becomes a random permutation schedule:

```
π_0..π_{K-1}: K = 10 iid uniformly-random permutations of Z_M
              (drawn per data seed — the multilane embedding convention)
Y ~ Unif({0..K−1}),  B ~ Unif(Z_M)          (per sequence)
z_t = π_Y((t + B) mod M)
x_t = u_{z_t} + σ ε_t,   u_a = R·[cos 2πa/M, sin 2πa/M]  (circle codebook)
```

**Exact parameterization (frozen), matched to `frequency` for the
controlled comparison:** `M = 101` (prime), `K = 10` (chance 0.1, the
frequency ladder size), circle embedding `d_in = 128`, `σ = 0.10`,
`seq_len = 64`, `n_seqs = 4096`, eval `L = 32`, `n_steps = 6000`,
capacity anchor `F = M = 101` (`d_sae ∈ {50, 101, 202}`), seeds
{1, 2, 42} + untrained. Datasource: `toy_permuted_circle_M101_d128`
(generator `permuted_tones`, append-only). Bench name: `permuted_tones`.

**Ground truth (Part II § 1).** Hidden latents (not directions): `Y` (the
schedule index, categorical K=10 — the headline latent), `B` (nuisance
offset, Z_M). Feature directions: the M circle atoms (reconstruction
codebook, as frequency — `eauc` ill-defined, capability read off NMSE);
strict signal directions = the 2 plane axes. The schedule table π is
exposed for oracles/audits. Primary metric: `schedule_recovery`
(multinomial-logistic probe on the shared per-tile code, leading-edge,
leak-free split, normalized [chance = 0.1, 1]) with per-class recalls;
oracle = the matched filter (§ 3). Evaluator add-on `permuted_recovery`,
additive dispatch on the schedule-table key; protocol stays 1.3.0.

## 3. Proof obligations (PORT.md § B format)

- **P1 (exact, restated).** π_Y is a bijection of Z_M and B is uniform ⇒
  `z_t | Y ~ Unif(Z_M)` for EVERY Y and t ⇒ `I(Y; x_t) = 0` exactly (DPI).
  Identical in form to the tone proof; the permutation only relabels a
  uniform. **Discharge:** analytic (this line) + numerical marginal check
  in the contract tests/gating.
- **P2 (phase-averaging, restated for general π).** For any per-token maps
  φ_t and additive score S_Y = Σ_t f_{Y,t}(x_t): conditional on any Y, the
  marginal of x_t is the same (P1), so E[S | Y] is Y-independent — no
  additive-over-time readout separates schedules perfectly, and linear
  probes on per-token/stacked/pre codes are mean-blind. The theorem bounds
  MEANS only (the FB-4 datum: large-sample multiclass linear probes can
  score a few points above chance off 2nd-moment differences — recorded,
  not gated two-sidedly). **Discharge:** analytic + § 8 floors.
- **Ceiling (matched filter).** The exact-likelihood decoder for known
  templates in white Gaussian noise: score(k, s) = Σ_t ⟨x_t, u_{π_k(s+t)}⟩,
  argmax over k (max over s). At freeze the clean windows are UNIQUE
  across all K·M = 1,010 (k, s) pairs at T = 8 (verified numerically, 0
  collisions) — expect near-saturation by T = 8, with T = 2 partially
  confusable (few-point windows). **Discharge:** numerical oracle curve per
  T ∈ {2, 4, 8} on the built generator in `gating.py`.
- **Spectral-envelope reference (NOT an abort gate).** A random schedule's
  T-window DCT band-energy profile fluctuates with (k, s); band energies
  alone may carry some Y-information at small T. Gating measures the
  **envelope oracle** — a multinomial-logistic classifier on the window's
  circle-plane DCT band energies ONLY (order/phase discarded) — per
  T ∈ {2, 4, 8}, reported as a reference curve. The grid's spectral-vs-post
  comparison is interpreted against it: spectral ≈ envelope reference ⇒ it
  reads envelope, not temporal structure.
- **Bag/multiset status (stated at freeze, honestly).** Unlike phasepair's
  exact bag null, the window MULTISET is class-informative here: at T = 8
  the 1,010 clean window sets are unique (verified at freeze, 0
  collisions). A within-window shuffle therefore destroys ORDER but NOT
  class information — shuffle is not a full null, and a nonlinear reader
  of pooled one-hot-ish codes (bag-MLP) can in principle set-match. The
  additive-family predictions rest on P2 (LINEAR probes, the panel's
  actual eval), exactly as in the frequency bench; the bag-MLP ceiling is
  measured in T2 and REPORTED as the additive-route reference, not gated.
- **P6 (memorization audit).** Whole-window clean template count
  K·M = 1,010 vs `d_sae ≤ 202` — the frequency situation exactly (its
  |Ω|·M = 1,010). No grid cell crosses the threshold; no jump predicted;
  no memo-demo cell in this cycle. Probe budget scales with code dim (T2).

## 4. Regime claim + design-time discriminability

**Regime 3** by construction: per-token provably zero (P1); raw-linear
window ≈ chance (E[x_t | Y] ≈ 0 — the circle is centred and B uniform; the
equality-variant § 8 treatment, same as frequency, with the FB-4
probe-protocol datum noted). The proofs predict apart: **position-mixing
converters** (txc-post: unconstrained temporal taps → can learn matched
filters for arbitrary schedules) from **band-limited converters**
(spectral: per-branch taps confined to DCT bands — a spectrally-generic
schedule is not representable inside a branch) from **additive codes**
(token/stacked/pre: P2). § 8 gate: (i) both raw-linear readouts ≈ chance;
(ii) matched-filter oracle ≫ chance at T ∈ {4, 8}; (iii) envelope
reference measured. If the oracle fails to clear chance+0.3 by T=8 the
bench is non-discriminating at this window range: record and STOP.

## 5. Memorization audit (P6)

See § 3 P6: 1,010 templates, `d_sae ≤ 202`, no threshold crossing at any
grid cell; the schedule table is drawn per seed so cross-seed pooling of
per-(k,s) templates is impossible by construction.

## 6. Frozen per-arch predictions (+ falsifiers)

**Directions frozen by mac-local (briefing, 2026-07-23), preserved
verbatim; reasons sharpened only.** Primary metric `schedule_recovery`,
canonical cells (d_sae = F = 101; T = 8 frontier, per-token-matched B*).

1. **per-token / tsae / stacked / txc-pre ≈ 0** — P1/P2/additive, as every
   tone bench (linear probes on additive codes are mean-blind).
2. **txc-post positive at T ∈ {4, 8}** — unconstrained temporal taps can
   learn matched filters for arbitrary schedules. Wide band accepted
   (0.1–0.8); the DIRECTION is what is frozen.
3. **spectral trained BELOW txc-post at the canonical T = 8 cell** — the
   reversal of multilane. Band-limited per-branch taps cannot represent a
   spectrally-generic schedule; residual spectral score should track the
   envelope reference (§ 3).
4. **spectral untrained ≈ post untrained** — the multilane 4× access-prior
   gap collapses: no band alignment exists at init for a broadband
   trajectory.
5. **Falsifiers:** any arch > 0.1 at T = 1 (P1 bug — STOP and debug, never
   report); winner trained ≈ untrained (gap < seed spread ⇒ access, not
   learning — report as access).

**The fork (stated in advance; both outcomes advance the program):**
- Prediction 3 HOLDS ⇒ the subtype rule's power leg gains the qualifier
  **"…when the power concentrates in few DCT bands"** (alignment-
  conditional) — the acid test lands on the alignment side.
- Prediction 3 FAILS (spectral matches/beats post beyond the envelope
  reference) ⇒ spectral's dominance is band-*competition* structure, not
  alignment — the rule survives strengthened.

## 7. Skeptic pre-registration notes

`a_proof_circularity`: P1/P2 are restatements of the frequency proofs with
π in place of the linear map — the discharge is on the built generator at
the frozen parameterization. `b_triviality`: the honest exposure is the
multiset route (§ 3 bag status) — the card does NOT claim order-necessity,
only linear-additive deadness, matching the tone benches' actual claims;
symmetry audit: relabeling symbols maps schedule set to schedule set only
via the permutation group acting on ALL of Z_M — since the π_Y are
generic, no relabeling maps class to class while fixing the ensemble
(unlike the random-embedding ratio orbit); to be checked empirically in
T2. `c_relevance`: honestly **spanning** — the research reason is the
subtype-rule qualifier (the program's live regime-3 rule, now carrying a
real-text POSITIVE in recipe_instruction whose generalization depends on
exactly this alignment question) + pinning FreqFrac's broadband pole.
`d_redundancy`: no registry bench is broadband-AC; frequency is the
aligned control this card is differenced against. `e_substrate`: no panel
or convention deviation; thin generator + datasource + eval add-on
appends; canonical runner; uniform grid T ∈ {1, 2, 4, 8}.
