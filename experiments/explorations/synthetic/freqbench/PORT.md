# FreqBench port — design doc (autoresearch-revamp phase 2)

**Status: DESIGN (2026-07-22, mac-local).** Port of Dmitry's Opus-4.7 FreqBench
sprint (2026-06-10, `origin/dmitry-spectral-sprint2:docs/dmitry/sprints/2026-06-10_freqbench_sprint/`)
and the colored-sources diagnostic (`origin/dmitry-synthetic:src/v6_colored_sources/`)
into the v2 shared substrate, per README § "The two generators, one substrate".
FreqBench lands **integrated**: a proofs registry attached to existing benches, a
FreqFrac lens that coordinatizes the whole suite, and new theorem-first benches
on the frozen panel — never a standalone stack (the `signed_motion`/`ac_sign`
fork is the failure mode this doc exists to end).

## A. Asset inventory → disposition

| sprint asset | what it is | disposition |
|---|---|---|
| `multifreq_circle` (M=101, 10-tone Ω, circle embedding) | classical single-tone estimation; periodogram = ML oracle | **ALREADY PORTED** — the `frequency/` bench (`toy_cyclic_circle_M101_d128`), verdict POSITIVE |
| `multifreq` random-embedding variant | the symmetry-null control (provably structureless) | **ALREADY PORTED** — `toy_cyclic_random_M101_d128` |
| `ac_sign` (2-class ±velocity) | pure-phase direction task | **PORTED-AS-FORK** — `signed_motion`. Port action: attach P2/P4/P6 so the NEGATIVE is *explained* (the `#windows=2F` memorization confound is P6's threshold hit in the scarce regime; direction needs order — the sign pair has identical symbol sets) |
| `dc` task (K-class majority) | smoothing suffices | no port — subsumed by the suite's regime-1 rows |
| `SpectralTXC` (banded DCT crosscoder) | the arch | **ALREADY PORTED** — panel arch `spectral_txc` (§ C verifies parameterization parity) |
| `ConvDict` (localized conv dictionary, L-tap filters) | the localization-end arch (resolution set by L, not W) | **DEFERRED** — panel stays frozen for comparability; revisit as a 7th arch iff axis 3 gets its measure |
| `freq_profile()` — **FreqFrac** | per-atom DCT energy of encoder kernels | **TO PORT — the centerpiece** (§ C) |
| firing-weighted FreqFrac + random-init null | kills the sorting-artifact trap (population conc. 0.18 vs init 0.13; busiest-atom conc. 0.56 vs 0.21) | port together with it — both are mandatory parts of the lens |
| `phasepair` task (±pairs, identical power spectra) | phase-only discrimination | **NEW BENCH candidate FB-1** |
| multilane superposition (3 tones, orthogonal planes, d=24) | memorization impossible by construction (\|Ω\|³M³ ≈ 10⁹); the one task where multiband > vanilla decisively (0.96 vs 0.91, no seed overlap) | **NEW BENCH candidate FB-2** |
| colored sources (per-coord AR(1) at lag D) | direction-*recovery* flavor: local-impossibility bound + W=D+1 phase transition | **NEW BENCH candidate FB-3** |
| `verify_theory.py` | empirical checks of P3/P4/P5 (ratio orbits, overlap combinatorics, pair contrast) | port as gating/tests for the FB benches |
| GPT-2 day-stride + backtracking spectral arm (`bt_freq.py`, `c7_spectral_arm.py`) | FreqFrac on *real* phenomena: attention converts by block 3; backtracking anticipation is LOW-frequency (branch AUC DC 0.803 > low 0.787 > mid 0.740 > high 0.733; feat_71839 lowfrac 0.362 vs random 0.133) | evidence base for measuring **axis 1 on grounded benches/real data** — the FreqBench↔PhenomenonBench bridge, already prototyped |

## B. Proofs registry (the theorem-first anchor)

The port must carry these over **as statements attached to benches**, not prose
lost in a sprint log. Sources: sprint `log.md` H0:50–H1:55 + theory.md
(P1–P6), colored-sources README (CS-1/2).

- **P1 — single-token ceiling.** Per-task Bayes accuracy `a_loc*` of any
  single-token readout (velocity has zero single-token MI).
- **P2 — additive-readout impossibility.** Phase-averaging: for stacked
  per-token codes, Σ_B score is velocity-independent ⇒ no linear probe on
  per-token codes separates velocities (MLP can — info is present; linearity
  is load-bearing).
- **P3 — symmetry-triviality / ratio invariance.** Random (exchangeable)
  embeddings + prime M: relabeling a ↦ a·y′y⁻¹ maps velocity y to y′ ⇒ no
  meaningful "frequency" without geometry on symbol space; two-class
  difficulty depends only on the ratio orbit {r, 1/r}. (Why the circle
  embedding exists; the random variant is the retained null.)
- **P4 — coincidence lines.** Confusability under random embeddings is
  number-theoretic: r ≡ ±p/q ⇒ shared symbols ≈ W/max(p,q). (Measured double
  dissociation: random confusion tracks overlap ρ=0.658; circle confusion
  tracks |Δf|, 97.7% of mass inside the Rayleigh cell.)
- **P5 — periodogram oracle + Rayleigh.** Circle task = classical single-tone
  estimation; DFT matched filter is ML; resolution ∝ 1/W.
- **P6 — memorization threshold.** Above |Ω|·M whole-window templates, window
  archs solve the structureless task by memorization (TXC 0.17→0.99 across
  H=256→2048); conv cannot (shift-local filters can't hold window templates).
- **CS-1 — local impossibility (colored sources).** Training data iid from
  P(x_t) ⇒ recovered directions independent of F; Rec ≲ log(H)/N regardless
  of compute/samples.
- **CS-2 — lag-D recoverability.** Eigenvectors of Ĉ_D recover F with angular
  error ~ε/γ (eigengap γ); window-local methods need **W ≥ D+1** — a sharp
  phase transition in W.

Landing spots: `frequency/` spec addendum (P1,P2,P3,P5,P6);
`signed_motion/` record addendum (P2,P4,P6 — explains the NEGATIVE);
FB-1 (P4/P5), FB-2 (P6-immunity), FB-3 (CS-1/2).

## C. The FreqFrac lens (the axis-1 measure)

**Definition** (from `fb_core.freq_profile`, generalized): for each encoder
atom j with time-domain taps e_{j,τ} ∈ R^d (τ = 0..T−1), DCT-transform over τ,
take per-frequency energy ‖ê_j(w)‖² summed over d, normalize per atom →
FreqFrac_j(w), a distribution over w = 0..T−1. Per-arch curve = firing-weighted
mean over atoms on the shared eval windows.

**Methodology rules carried from the sprint (mandatory):**
1. **Firing-weighted, not population-mean** — population means dilute
   (H1:10 lesson: the sorted heatmap "filter bank" was partly a sorting
   artifact).
2. **Untrained same-arch null** — random-init concentration is the baseline
   (0.205); a claim of learned spectral structure must clear it.
3. Report spectral concentration (top-2-adjacent mass) alongside the curve.

**Tap extraction per panel arch** (pinned; all archs in `src/temp_bench/archs/`):
- `batchtopk_sae` (`W_enc (d_in, d_sae)`) and `tsae` (same shapes, T=1
  enforced): single tap — no temporal response by construction; reported as
  DC-only. (Their window behaviour comes from the stacked probe, i.e. the
  probe's response, not the code's — P2.)
- `stacked_batchtopk`: `W_enc (T, d_in, d_sae)` — atom h's temporal kernel is
  `W_enc[:, :, h] → (T, d_in)`. Independent per-position dicts ⇒ near-delta
  taps ⇒ broadband + maximally localized (the wavelet end of axis 3).
- `txc_batchtopk_pre` / `post` (`_TXCBatchTopKBase`): `W_enc (T, d_in, d_sae)`
  — the encoder tensor is **identical** between pre and post (they differ only
  in where BatchTopK applies); atom h's taps = `W_enc[:, :, h]`.
- `spectral_txc` (`SpectralTXCBatchTopK`): already DCT-parameterized —
  `enc_coef[b] (h_b, n_band, d_in)` over band index sets `self.bands`, with
  the code-slice map `band_of_features()`; FreqFrac is exact with no
  transform (embed each band's coefficients at its DCT indices).

FreqFrac = DCT-II over the T axis of the `(T, d_in)` kernel, energy summed
over d_in, normalized per atom — numerically identical to the sprint's
`freq_profile` (Parseval over the orthonormal DCT). Confirmed genuinely new
in-repo: the only existing "S(f)"/periodogram code is **data-space**
(`frequency/gating.py` oracles, `evals/frequency_recovery.py` probe metrics);
no code anywhere computes spectra of trained encoder weights.

**Deliverables:**
- `src/explorations/synthetic/freqfrac.py` — tap extraction (per-arch
  adapter), DCT profile, firing weights, concentration stats + tests.
- `experiments/explorations/synthetic/freqbench/freqfrac_report.py` — trains
  or loads the canonical per-token-matched cells, emits per-(bench, arch)
  FreqFrac curves + an AUTO "coordinates (axis 1)" block for REPORT.md.
- **Acceptance:** (i) reproduces the frequency bench's known story from the
  weights alone (spectral high-pass; TXC-pre flat ⇒ blind; per-token no
  response); (ii) backtracking-mirror-trained window archs show DC-dominant
  taps (matches the sprint's real-trace finding that anticipation is
  low-frequency); (iii) untrained nulls flat.

## D. Checkpoint / cache mechanics — the train+analyze path (pinned)

Facts (from `core/runner.py`, `core/trainer.py`, `explorations/synthetic/grid.py`):
- `run_experiment` cache-checks **`eval_key` first** and returns the cached
  leaderboard row **without ever loading a model**; on an eval-key miss, a
  `train_key` hit loads `checkpoints/<train_key>/model.safetensors`, else it
  trains and saves (+ `manifest.jsonl` row). Skipping is entirely the
  runner's — grid.py only reports it. There is **no force/retrain flag
  anywhere** (`run_sweep`'s `skip_cached` is a dead field; the
  "`--force-train`" string exists only inside an error message).
- On this Mac the checkpoint store is purged (2 reloadable smoke runs; the
  manifest still lists ~4,000 historical rows) while the local leaderboard
  holds 3,585 rows — so going through `run_experiment` would skip everything,
  and **every FreqFrac cell must (re)train locally**.

**Decision — the report script reuses the runner's mechanics directly, not
the runner's entry point:** `freqfrac_report.py` computes the canonical
`train_key` from the *exact* grid-cell `TrainingConfig` (keys must match what
the grids produce), tries `runner._load_checkpoint(...)`, and on
`FileNotFoundError` calls `trainer.train_arch(...)` — which saves the
checkpoint + manifest row under that key, so any future grid run on this
machine fast-forwards training for free. **No leaderboard writes**: FreqFrac
is a weight-space diagnostic, not an eval result (hard rule 1 governs
results; diagnostics must not masquerade as rows) — but the emitted stats
JSON stamps `code_version` + the full cell config + the `train_key` of every
model analyzed. No `temp_bench/core/` edits — existing core functions only;
the plugin rule stays intact.

Rejected alternative: a new `extra`-sentinel add-on in `synthetic_recovery`
(the assumption/hedging pattern) — FreqFrac needs the *weights*, not the eval
pipeline, and cached eval_keys would skip the evaluator anyway.

Tiny models (d_in ≤ 128) ⇒ Mac CPU/MPS handles the prototype cells; the full
6-bench × 6-arch pass (+ untrained nulls) is a later runpod briefing
(~495 cells/hr measured there).

## E. Execution plan

1. **(mac, now)** `freqfrac.py` core + tests; prototype on 2 benches
   (frequency + backtracking) at the canonical matched cells → curves + nulls.
2. **(mac)** Spec addenda: `frequency/` gains the proofs block;
   `signed_motion/` record gains the P2/P4/P6 explanation of its NEGATIVE.
   BENCHMARKS.md rows updated (provenance already tagged).
3. **(runpod, later briefing)** Full FreqFrac pass — 6 benches × 6 archs ×
   canonical cells (+ untrained nulls) → REPORT.md "coordinates" AUTO block.
4. **FB-1/2/3 proposals** written per README checklist §8 (each states
   coordinates + regime + the discriminability argument up front):
   - **FB-2 multilane** first — regime 3 by construction, memorization-immune
     (P6-proof), and the only known task separating multiband from vanilla:
     the sharpest missing arch-separator.
   - **FB-3 colored-sources** — the direction-recovery axis with a provable
     per-token impossibility (CS-1) and the W=D+1 memory-depth transition.
   - **FB-1 phasepair** — phase-vs-power dissociation (needs the
     discriminability argument most: is any panel arch power-spectrum-only?).
5. **The FreqBench generator loop (Fable 5)** — after 1–4 land, freeze the
   loop's rails: propose an axis point → prove ceiling/impossibility (registry
   format) → construct the task on the frozen panel → §8 gate incl.
   discriminability → BENCHMARKS row tagged `theorem-first`. (Mirror of the
   PhenomenonBench card discipline, with proofs where PhenomenonBench has
   measurements.)

## F. Sprint methodology rules worth adopting suite-wide

- **Probe budget must scale with code dimension** (the H=2048
  "overcompleteness hurts" claim was probe-sample starvation; 60k samples
  recovered it). Check the v2 probe sample sizes at d_sae=2F.
- **Per-window independent shuffle permutations** (a single shared
  permutation is probe-learnable — the sprint's conv-"survives"-shuffle bug).
  Verify v2's shuffle controls already comply.
- **Shuffle preserves the symbol set** — for cyclic/set tasks a shuffle
  control is not a full null (only the sign task's shuffle is); state per
  bench what the shuffle actually destroys.

## G. First-pass FreqFrac results (2026-07-22, mac-local — 12 canonical cells, seed 1)

`results/freqfrac_stats.json` + `figs/freqfrac_curves.png` (frequency +
backtracking, per-token-matched cells, trained vs untrained-init null).
Firing-weighted DC fraction and top-2-adjacent concentration, trained (init):

| bench | arch | dc_frac | concentration | read |
|---|---|---|---|---|
| frequency | per-token / tsae | 1.00 (1.00) | 1.00 (1.00) | no temporal response by construction (P2) |
| frequency | stacked | 0.246 (0.248) | 0.52 (0.53) | flat — near-delta taps, the wavelet end |
| frequency | txc-pre | **0.250 (0.250)** | 0.61 (0.53) | **taps exactly flat after training** — the additive-blind signature (recovery 0.07, flat S(f)) |
| frequency | txc-post | 0.303 (0.252) | **0.715 (0.528)** | learned tone-like atoms, clears the init null |
| frequency | spectral | 0.349 (0.314) | 1.00 (1.00)† | band-exact†; firing tilts modestly |
| backtracking | txc-pre | **0.338 (0.253)** | 0.61 (0.53) | DC-shifted — λ is integration-of-history |
| backtracking | txc-post | 0.277 (0.254) | 0.60 (0.53) | mildly DC-shifted |
| backtracking | spectral | **0.381 (0.167)** | 1.00 (1.00)† | strongest: init tilted AC, training inverts it to DC-dominant |

† at T=4 the multiband split degenerates to four singleton bands, so spectral
concentration is 1.0 *by construction* — read its firing curve, not conc.

**Acceptance verdicts (frozen in § C):**
- **(ii) backtracking DC-dominance — PASS.** All three crosscoders shift
  toward DC on the self-exciting mirror; spectral inverts its init tilt
  (0.167 → 0.381, curve monotone-decreasing). The sprint's real-trace
  "anticipation is low-frequency" finding, reproduced from weights alone.
- **(i) frequency story — PASS on the decisive components, with a resolution
  caveat.** Per-token silent (proven + observed); TXC-pre flat (blind);
  post/spectral show learned structure clearing their init nulls. **Caveat:
  the "high-pass" component is unresolvable at `T_can = 4`** — 5 of the 10
  Ω tones sit below the first DCT bin (w=1 ↔ 0.125 cycles/token), so the
  trained tilt reads *low*. The check has power at the T = 8 frontier cells
  (`--T 8`); flagged into the FB-C1 briefing step 0. A lens-resolution
  finding, recorded rather than smoothed over.
- **(iii) untrained nulls — PASS.** Init curves flat (≈ 0.25) for all raw-tap
  archs; spectral's init unevenness is band-firing, visibly separated from
  its trained curve.

Alive fractions healthy (0.90–1.00). All 12 checkpoints now in the local
store under their canonical train_keys (future grid runs on this machine
fast-forward training).

### G.1 Full-pass results (2026-07-22/23, runpod-b, FB-C1 Phase 1)

The widened pass: **all six registry benches × 6 archs × seeds {1, 2, 42} at
T_can = 4, plus seed 1 at T = 8 for every window arch** — 132 unique cells,
every checkpoint trained fresh on this pod via the canonical trainer,
`train_key` hard-asserted against its leaderboard row, no leaderboard writes.
Merged table + per-cell curves: `results/freqfrac_full_pass{,_summary}.json`,
figure `figs/freqfrac_full_pass.png` (band = seed range). Two § G questions
answered:

**(a) Seed stability of the axis-1 coordinates — QUALITATIVELY STABLE, with
quantified spread.** Every first-pass signature reproduces in all 3 seeds on
all 6 benches: per-token archs DC-only (1.000, by construction); stacked flat
at init level everywhere (dc 0.22–0.30 vs init 0.25 — the wavelet/broadband
end); **txc-pre exactly flat on frequency in every seed (dc 0.250–0.253,
spread 0.003 — the additive-blind signature is the tightest number in the
table)** while DC-shifted on the integration benches (backtracking
0.338–0.346, spread 0.008; changepoint 0.339–0.440; hedging 0.439–0.660);
txc-post and spectral clear their init nulls wherever they learned structure
(frequency conc 0.715–0.817 vs init 0.53). Quantitative caveat: the
firing-weighted dc_frac of spectral (and txc-pre on hedging) carries seed
spread up to ~0.19 on the DC benches — firing weights are seed-sensitive
where a few atoms dominate; the coordinate READ (which side of init, curve
shape) never flips. hedging txc-pre is the extreme DC point of the whole
suite (dc 0.44–0.66 at T=4), as the drift bench should be.

**(b) The T = 8 frequency high-pass check — PASS.** At T = 8 the DCT bins
resolve 6 of the 10 Ω tones (bins ≥ 1.27); the T=4 resolution caveat lifts:

- **spectral (frequency, T=8):** training tilts firing-weighted mass
  *down* at DC (0.335 → 0.287) and *up* in the top bins — w ∈ {5,6,7} mass
  0.221 (init) → 0.285 (trained), +29 % — the weight-space image of the
  bench's high-pass S(f). Concentration stays band-tight (0.93).
- **txc-post (frequency, T=8):** per-atom spectral concentration nearly
  doubles over init (0.472 vs 0.275) — learned tone-like atoms; the
  atom-MEAN curve stays flat because different atoms concentrate at
  different tones (concentration, not the mean curve, is the right
  statistic — the § C rule).
- **stacked + txc-pre (frequency, T=8): exactly flat, ≈ init** (every bin
  0.121–0.131) — the P2 additive blindness, now at 8-bin resolution.
- Contrast **backtracking T=8**: txc-pre dc 0.242 vs init 0.117 (2.1×
  DC-shift) and spectral low-band (w ≤ 2) 0.564 vs init 0.450 with the top
  bins *dropping* — the anticipation-is-low-frequency finding sharpened.
  The two benches tilt the SAME arch families in OPPOSITE spectral
  directions from the same inits: the lens separates the axis-1 poles from
  weights alone.

Both § G acceptance components that were resolution-blocked at T_can=4 are
now discharged; the FreqFrac coordinates used by REPORT/registry entries can
cite the seed-mean T=4 numbers with the T=8 rows as the resolution frontier.

---

## H. Cycle log — FB-C1 (2026-07-22/23, runpod-b, the 12-hour session)

**Briefing:** `briefings/freqbench-c1.md` (stays until mac-local review).
**Protocol:** LOOP.md (rails frozen). **Spend:** $1.04 / $25 (3 skeptic
calls on Fable 5; `results/spend.json`). **Compute:** ~2,100 canonical grid
cells + 156 FreqFrac cells, 0 failures anywhere, all through the canonical
runner on this pod's fresh checkpoint store.

| phase | outcome |
|---|---|
| 1. widened FreqFrac pass | 132 cells (6 benches × 3 seeds @T4 + s1 @T8). § G.1: coordinates seed-stable; **T=8 high-pass check PASS** (the T=4 caveat lifted) |
| 2a. FB-2 multilane | cards frozen pre-build (f0e6778f) → build+tests → T1/§8 PASS (P5 exact: per-lane = single-lane oracle, gap ≤ 0.007) → T2 PASS → skeptic PROCEED 5/5 → grid 708/708 → **POSITIVE**; the sprint's multiband>vanilla headline **failed its frozen T=8 bar** (+0.019 < 0.03, seed-disjoint; edge peaks at T=4 +0.087); spectral collapses at k_pos=8 (margin +0.544@k1 → −0.583@k8) |
| 2b. FB-3 colored_sources | build+tests → T1/§8 PASS (oracle rec_adj +0.96; W=D+1 transition 0.03→0.96) → T2 PASS → skeptic PROCEED 5/5 → grid 582/582 → **POSITIVE (weak realization)**: CS-1 floor holds over all 261 T≤D cells (max +0.037); best arch = 21 % of the provable ceiling; **ordering INVERTS the tone benches** (txc-pre ≥ spectral > post ≈ floor), pre's recovery ρ-ordered 0.29→0.65 |
| 3. FB-1 phasepair | card frozen pre-build → build (no new generator; ±-pair-gated add-on) → T1/§8 PASS (sign oracles 0.97–1.00; **exact bag null** to 0.007) → T2 PASS (chirality finding recorded) → skeptic PROCEED 5/5 → grid 636/636 → **POSITIVE**: post reads sign **1.000** (T=8, all seeds); spectral sign-blind at T ≤ 4 (singleton DCT bands — no quadrature partner) → 0.936 at T=8; additive family ≈ 0 on both components |
| 4. T=16 frontier addendum | **NOT RUN** — acceptance-gate work prioritized within the 12 h window; queued as follow-up |

**The cycle's headline — the panel's triple dissociation.** Across the
three theorem-first benches, each mixing family wins exactly one axis:
**spectral wins power** (multilane 0.79), **txc-pre wins lag-covariance
eigenstructure** (colored_sources +0.21, the only lift), **txc-post wins
phase** (phasepair 1.000). No window architecture dominates, and which one
wins is predictable from the card coordinates — the acid-test currency the
two-generator program was built to produce. Secondary findings: the
band-partition advantage is a scarcity/coarse-window phenomenon (FB-2);
provably-present dictionary information is realized at ≤ 21 % by current
training (FB-3 — the gap is the finding); singleton DCT bands are
structurally phase-blind (FB-1); signed_motion's NEGATIVE is
retro-explained as substrate defect, not panel phase-blindness.

**Process notes for review.** Three gate-check amendments were made
post-freeze, each a null/witness mis-specification fix (orthonormal null
for eigen-estimators + measured stream leakage; oracle witness for
info-presence; one-sided floors for below-chance probe artifacts) — all
documented in the gating scripts, disclosed to the skeptic, and none
touched a task, tolerance-as-intended, or prediction. Frozen-prediction
misses are reported as misses in each record (FB-2 txc-pre band; FB-1
post/spectral bands + untrained-spectral access; FB-3 pre/post/spectral
bets). Records: `multilane/`, `colored_sources/`, `phasepair/`
`bench_record.md`; registry + BENCHMARKS + REPORT (78/78) updated.

**Review (2026-07-23, mac-local) — APPROVED, all three verdicts stand.**
Full audit against the frozen checklist:

- **Freeze orders proven from the log:** cards `9e6427be` (22:49, FB-2+FB-3)
  and `8bbc3a95` (00:54, FB-1) strictly precede their builds (`d3d6cc1a`
  23:09, `f2f4128c` 01:02), which precede their grids. The § H table's
  `f0e6778f` is a stale pre-rebase SHA — the real card-freeze commit is
  `9e6427be` (the pull-rebase rule rewrites local SHAs; cite post-push SHAs
  in records).
- **The three amendments verified genuine, not tolerance shopping.** Each
  corrects the *reference*, not the bar: (1) eigen-estimators compared to an
  orthonormal null (their actual candidate class) instead of iid-Gaussian,
  plus the measured +0.011 stream-leakage honestly bounding CS-1's iid
  premise; (2) info-presence witnessed by the ML oracle (0.906) instead of
  an under-trained generic MLP (0.173) — presence is an information claim
  and the matched filter is its correct witness, with the MLP datum kept;
  (3) one-sided floors — below-chance probes (0.112–0.115 vs chance 0.167)
  are degenerate-classifier artifacts, not linear access, and the
  above-chance tolerance was never loosened. All three flips FAIL→PASS are
  visible (failing first-pass stats committed at `f2f4128c`, flip at
  `4f2f2c98`); all disclosed to the skeptic pre-grid. Process gap: first
  passes ran before the scripts were committed — LOOP.md T3 now states the
  strict commit-then-run form.
- **Grid hygiene:** leaderboard rows reconcile exactly — multilane 710 =
  708 grid + 2 smoke (300-step pipeline checks), colored 583 = 582 + 1
  smoke, phasepair 636 + 0; **0 duplicate eval_keys across all 6,009 global
  rows** (the union-merge drivers' first two-agent parallel night); 0
  error-status rows; spend $1.04/$25 (3 skeptic calls); tests 159 pass.
- **Verdicts:** misses framed as misses in all three records; unfrozen
  observations flagged as unfrozen; single-cell anomalies (FB-3 q1 excess)
  flagged, not interpreted. REPORT 90/90 (15 latent axes × 6 archs),
  registry + BENCHMARKS rows consistent with the records.
- **Science baked into the program:** the triple dissociation + the recipe
  residual give the order-2 *subtype* rule (phase-relational → post ·
  power/equality → spectral · covariance-accumulable → pre), now in the
  README coordinate section; cards tag the subtype and the acid test
  predicts winners from it. FB-3's bag-dilution amendment reframes
  colored_sources as a *depth* bench (true null = window truncation), which
  makes txc-pre's win coherent rather than anomalous.

T=16 frontier addendum + verify_theory ports remain queued (not run).
Briefing `freqbench-c1.md` deleted at this review.

### G.2 T=16 lens rows (FB-C2, 2026-07-23, runpod-b)

The merged table (`freqfrac_merge.py`, now covering the three FB tone/CS
benches and a `T=16` row tier; same tag convention, seed 1) gains the
frontier octave alongside the T=16 grid addendum. Reads, all init-relative
(absolute concentrations are not comparable across T — the uniform-init
baseline falls with band count: conc-init 0.53/0.28/0.17 at T=4/8/16):

- **Trained spectral sheds DC roughly twice as hard at T=16** on both tone
  benches: frequency dc 0.333→0.226 (T=8: 0.335→0.287), multilane
  0.426→0.211 (T=8: 0.387→0.321) — the weight-space image of the resolved
  Ω ladder ("high-pass sharpens", the frozen frequency prediction's lens
  half, HELD).
- **txc-post's concentration ratio over init keeps rising with T**
  (frequency 1.72×@T8 → 2.41×@T16; multilane 2.66× → 4.84×) — increasingly
  tone-like atoms exactly where its recovery rises.
- **The additive family stays at init at every T** (stacked/pre within
  0.01–0.03 of init dc and conc at T=16) — the P2 wall is T-independent in
  weight space too.

## I. Cycle log — FB-C2 (2026-07-23, runpod-b)

**Briefing:** `briefings/freqbench-t16-fbc2.md` (stays until mac-local
review). **Protocol:** LOOP.md incl. the new strict commit-then-run T3.
**Spend:** $1.36 / $25 cumulative on the freqbench meter (1 skeptic call
this cycle). **Compute:** 2,560 grid-driver cells (916+870+774 across the
three tone benches, T≤8 cache-fast-forwarded, 462 fresh T=16 cells) + 9
FB-4 gating-panel cells + 3 FreqFrac T=16 passes; 0 failures anywhere.
**Tests:** 173 pass (+13 this cycle).

| phase | outcome |
|---|---|
| 1. T=16 frontier addendum | Drivers extended + committed pre-run (`32851ee8`). **All three frozen mac-local predictions HELD blind:** (1) multilane band margin ≤ +0.01 at every capacity, inverts at d=101 (−0.015) — band advantage confined to coarse windows; (2) phasepair spectral sign 0.978@k2 / 1.000@k1 above the 0.936 reference, post stays 1.000; (3) frequency spectral saturates 0.997–0.999 everywhere + FreqFrac high-pass sharpens (dc-shedding doubles). Addendum sections in the three records; § G.2 lens rows |
| 2. verify_theory ports | `tests/test_verify_theory.py` — 9 permanent analytic tests pinning P2 (phase-averaging + the order-2 converse with exact cos(2πy/M) means), P5 (periodogram≡ML decisions ≥ 0.995; Dirichlet gram to 1e-9; behavioral Rayleigh at elevated σ), CS-2 (population lag-covariances; eigen-recovery ≫ shuffled control; γ-dependence) to the BUILT generators; thresholds are margins around measured values, seconds-fast |
| 3. card FB-4 rotated_multilane | Card frozen pre-build (`adc6bb28`, mac-local directions verbatim + absorption obligation added at freeze) → thin generator + datasource + contract tests → gates committed pre-run (`6e627593`) → T1 PASS after one amendment (window-linear floor re-keyed to rotation-invariance; first-pass FAIL preserved `d9e00a5b`) → **T2 ABORT_T2_SYMMETRY per the pre-registered rule** (arm A p 0.27–0.99; arm B all 9 canonical cells inside FB-2 seed bands — untrained spectral 0.290 vs 0.298, the frozen collapse direction REFUTED) → **skeptic ABORT confirmed** (kills exactly b_triviality + d_redundancy; absorption judged sound; amendment judged honest). No grid spent. BENCHMARKS § B row added |

**The cycle's headline.** Two results: (i) the **T=16 frontier confirms the
coordinate model's extrapolations 3-for-3 blind** — the sharpest being the
band-margin inversion and the untrained-spectral sign ladder (0 → 0.67 →
0.94 at T = 4 → 8 → 16: phase access is a pure function of band
multiplicity, an architectural *prior*, before training); (ii) the **FB-4
abort with the absorption theorem**: for any generator whose embedding is
Haar-random and seed-re-drawn, a fixed spatial orthogonal knob is provably
inert — the intended basis-alignment acid test needs a *temporal* knob
(candidate FB-5, deliberately unfrozen, for review). The frozen
"untrained-spectral collapse" direction failed exactly as the absorption
argument predicted, measured at gating scale without spending a grid.

**Process notes for review.** Strict commit-then-run held everywhere (all
gating/driver scripts committed before first execution; the one amendment
is its own commit pair with the failing first pass preserved). One datum
left for the program: FB-2's raw-window-linear "≈ chance" floor is
probe-protocol-conditional (0.10 → 0.13 under a larger-sample probe,
identically on rotated and base — P2 bounds means only). The skeptic's
checklist-item proposal (kill spatial-rotation cards on Haar seed-re-drawn
substrates at freeze) is left for mac-local — program-rule changes are out
of session scope.

**Review (2026-07-23, mac-local) — APPROVED; ABORT verdict and all three
T=16 addendum verdicts stand.**

- **Process: the strict commit-then-run rule worked as designed on its
  first live test.** The FB-4 gating amendment is a clean commit pair —
  failing first pass preserved (`917f8ccd`), re-key its own commit
  (`60c8d5be`), PASS after (`7b81b60d`) — and the re-key is genuine: a
  linear probe is *exactly* invariant under an orthogonal feature map, so
  the identical readings on unrotated data prove the floor excess was a
  substrate-level probe artifact, not rotation. The T2 abort followed a
  decision rule pre-registered at card freeze; no grid was spent.
- **The absorption theorem is correct and the card defect was
  mac-local's** — the briefing's frozen "untrained-spectral collapse"
  direction was analytically wrong (the DCT prior is temporal; multilane's
  spatial embedding is Haar re-drawn per seed, so Q·P =d P). Scored
  honestly as REFUTED in the record. Rule adopted at this review
  (LOOP.md card item 1): every card whose knob composes with a randomized
  substrate component must carry a **non-absorption argument** at freeze.
- **Hygiene:** 0 dup eval_keys (6,480 global rows); new cells reconcile
  exactly (462 T=16 + 9 FB-4 panel); tests 173 green; spend $1.36/$25.
- **Science adopted:** the untrained-spectral sign ladder (0 → 0.67 →
  0.94 at T = 4 → 8 → 16) makes the subtype rule's phase leg
  **T-conditional on band multiplicity** — amended in the README
  coordinate section. The acid-test question FB-4 carried moves to
  **FB-5 `permuted_tones`** (temporal knob — briefing
  `freqbench-fb5.md`).
- Recurring nit: records again cite pre-rebase SHAs (`adc6bb28`,
  `6e627593`, `d9e00a5b` → actually `75c7aa21`, `37b7e6b6`, `917f8ccd`) —
  rule added to `agents/README.md`: cite commit *subjects*, or verify
  SHAs post-push.

## J. Cycle log — FB-C3 (2026-07-23, runpod-b)

**Briefing:** `briefings/freqbench-fb5.md` (stays until mac-local review).
**Protocol:** LOOP.md incl. strict commit-then-run T3 + the new
non-absorption card obligation (its first application). **Spend:** $1.63
cumulative on the freqbench meter (1 skeptic call this cycle). **Compute:**
636/636 uniform grid cells + gating/T2 numerics + 2 FreqFrac passes, 0
failures. **Tests:** 179 pass (+6 contract tests this cycle).

| step | outcome |
|---|---|
| card FB-5 `permuted_tones` | FROZEN pre-build ("card FB-5 permuted_tones FROZEN pre-build"): the FB-4 salvage with the right knob — K=10 random permutation schedules on the frequency substrate; non-absorption discharged at freeze (temporal-LAW change: lag-1 ladder ±1 vs ≈0±O(1/√M); nothing randomized can absorb it); the multiset status stated honestly (T=8 window sets unique ⇒ shuffle not a full null; the card claims P1/P2 linear-deadness, not order-necessity) |
| build | thin generator + `toy_permuted_circle_M101_d128` + `permuted_recovery` add-on (matched-filter oracle over (schedule, offset)) + 6 contract tests; protocol stays 1.3.0 |
| gates | **first-run PASS, ZERO amendments** (bars a priori in the committed scripts): floors AT chance (0.0987–0.0998); oracle 0.43/0.99/1.00 at T=2/4/8; envelope reference 0.017/0.048/0.116 (recovery units); T2: shuffle kills the oracle 1.00→0.125 (order route), bag-MLP set-route 0.151 reported, bag-linear at chance (P2) |
| skeptic | PROCEED 5/5, raw persisted |
| grid + blind verdict | 636/636, 0 failures. Directions: (1) additive ≈ 0 HELD; (2) post positive HELD (0.074–0.161, only arch beyond the envelope reference at every T) with the canonical-cell magnitude below the indicative band (0.075 at k=2 — enters 0.1–0.8 at k≥4); (3) spectral-below-post MIXED literally (fails at T=8 k=2, holds at T=4 and k≥4) but the frozen MECHANISM clause held with precision — **trained spectral tracks the envelope reference numerically at every T: 0.016/0.042/0.096 vs 0.017/0.048/0.116**; (4) untrained-prior collapse PARTIAL (+0.298 → 0.045, 6.6×; small seed-disjoint spectral>post residual remains); (5) falsifiers clean (T=1 max 0.0047) |
| verdict | **POSITIVE (weak realization — best arch 16% of the provable matched-filter ceiling, the FB-3 pattern on the tone substrate). The fork resolves to the ALIGNMENT side: the subtype rule's power leg gains "…when the power concentrates in few DCT bands"** (README wording proposed in the record, left for mac-local). FreqFrac: the broadband axis-1 pole realized — post learns concentrated atoms with NO spectral tilt (dc ≈ init), the matched-filter weight signature |

**Process note.** First FreqBench cycle with zero gate amendments: the
FB-4-informed a-priori bars (window floor chance+0.05 citing the
probe-protocol datum) absorbed the known artifact class without touching
anything post-hoc — the amendment discipline turned into design-time
knowledge, which is what the loop is for.
