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
