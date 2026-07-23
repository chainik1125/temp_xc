# Synthetic temporal benchmarks

This directory is the **synthetic-benchmark program**: build synthetic
benchmarks whose ground truth is stated exactly, and use them to deliver sound
verdicts on whether window/temporal dictionary architectures exploit temporal
structure better than a per-token SAE. Benchmarks enter through one of **two
generators** (see "The two generators, one substrate" below):
**PhenomenonBench** — data-first, mirror a *measured* property of real LM
behaviour — and **FreqBench** — theorem-first, construct a task with *proven*
ceilings from a structural axis. Each benchmark is a **self-contained subdir**
holding its docs, scripts, figures, and results.

It lives under `experiments/explorations/` — the home for exploratory experiments
(research that may or may not graduate into `temp_bench`); `synthetic` is the
first such exploration. Any reusable *library* code it develops goes in `src/`
(its own `src/explorations/<name>/`, or `temp_bench` once it graduates); this
tree holds the experiments + their artifacts. Scripts run as a package, from the
repo root: `.venv/bin/python -m experiments.explorations.synthetic.<bench>.<script>`.

This README is the **single governing doc** for the program — the prime
directive, the two-generator structure and its shared-substrate contract, the
measure→mirror→bench loop and its validity gates, and the conventions every
synthetic benchmark follows. Read it before proposing or implementing a
benchmark.

> **Current state / what's in progress:** see [`STATUS.md`](STATUS.md) — the
> living pre-compact scratchpad (the active work, the locked design, next
> actions). It is the one file to read first when resuming and the one to update
> before a compact.

Related:
[`../../../docs/ideas/frequency_lens.md`](../../../docs/ideas/frequency_lens.md) (the DC/AC
frequency lens), [`../../../CLAUDE.md`](../../../CLAUDE.md) (framework hard rules: canonical
runner, code-version stamping, plugin-only, never edit `core/`),
[`signed_motion/bench.md`](signed_motion/bench.md) (the cautionary worked
example where the §3 controls turned a false positive — `s_temp=1.0` from a
memorizing probe — into the true negative).

---

## Prime directive (read this twice)

> **Success = a sound, reproducible *verdict* — positive or negative.** An
> investigation that concludes "this property is not temporal" or "no
> architecture beats the per-token baseline" is a *complete success* when the
> controls pass. We are **never** rewarded for a positive result, and must not
> tune the labeler, statistic, capacity regime, probe, or metric to manufacture
> one.

The loop is open-ended and the failure modes are subtle — we have hit several by
hand (conflating a feature count with a pattern count; a probe that memorizes a
small window set and "generalizes" because train/eval share it; "wins" that also
appear with an *untrained* model). Everything below exists to make reward-hacking
into those traps impossible to do silently.

---

## The two generators, one substrate

Benchmarks enter the program through one of two generators. They differ in
**epistemic anchor** — what you point at when asked "why does this task measure
anything?" — and every benchmark carries a **provenance tag** naming its
generator (the `provenance` column in [`BENCHMARKS.md`](BENCHMARKS.md)).

**PhenomenonBench — data-first (`grounded`).** Start from real LM behaviour and
work backwards: labeler → measured temporal signature → fitted synthetic mirror
→ gate-validated benchmark. That is Part I of this doc (the measure→mirror→bench
loop) plus the autonomous expansion loop ([`expansion/LEDGER.md`](expansion/LEDGER.md)).
Authority = grounding: the task mirrors a measured property of a real model.
Because every step involves judgment on noisy data, drift is high — hence the
heavy machinery (preregistration, frozen predictions, nulls, gates 7–8, the
skeptic). Its blind spot, demonstrated twice in stage 6: **grounded + valid
mirror ≠ discriminates** — see the discriminability STOP-gate below.

**FreqBench — theorem-first (`theorem-first`).** Start from a structural axis
and construct the task so ground truth, oracle, and impossibility results are
*provable* (periodogram = ML oracle; linear-probe impossibility;
symmetry-triviality → the circle embedding; ratio invariance). No labeler, no
mirror, no grounding claim — authority = the proofs, so it needs little
procedural machinery (a theorem cannot be reward-hacked). Its blind spot is
**irrelevance**: alone it can probe structure no real phenomenon exercises —
exactly the original curated-wins sin. Origin: Dmitry's FreqBench sprint
(branches `origin/dmitry-spectral-sprint2`, `origin/dmitry-synthetic`); the
`frequency/` bench is its first port, and `signed_motion` is its `ac_sign` task
forked *without* the proof apparatus. The port lives at
[`freqbench/`](freqbench/): `PORT.md` (asset inventory + the proofs registry
P1–P6 / CS-1–2), **`LOOP.md` (the theorem-first generator protocol — the
FreqBench autoresearch rails)**, and `freqfrac_report.py` (the FreqFrac lens
over the shared panel at the canonical matched cells).

The generators are duals — FreqBench spans the coordinate space with proven
ceilings; PhenomenonBench locates real LM behaviour inside it — and each
supplies the guarantee the other cannot. **Two generators, ONE substrate.**
Every benchmark, regardless of provenance, must share:

1. the fair-backbone architecture panel + the canonical runner (Part II §7,
   framework hard rules);
2. the capacity / realized-L0 / windowing / metric conventions (Part II);
3. the coordinate system below — a bench's spec states its coordinates;
4. the single registry [`BENCHMARKS.md`](BENCHMARKS.md), provenance-tagged.

Forking the substrate is the historical failure mode (FreqBench and the
phenomenon suite evolved disjoint stacks; `signed_motion` vs `ac_sign` is the
scar): a benchmark that cannot run in the shared B×A grid does not exist.

### The coordinate system (three structural axes)

Where a benchmark's primary latent lives — stated in its spec, and predictive
of the architecture outcome:

1. **Spectral (DC ↔ AC):** the waveform of the latent — slow drift vs
   order-sensitive variation. Measured by the frequency-response / FreqFrac
   lens ([`../../../docs/ideas/frequency_lens.md`](../../../docs/ideas/frequency_lens.md)).
2. **Interaction order (additive ↔ equality / higher):** whether reading the
   latent needs only a weighted sum over positions, or comparisons *between*
   positions. Additive codes are provably blind to equality-pattern latents
   (changepoint record); the within-window shuffle gap is the first-line probe.
3. **Stationarity / localization (spread ↔ clustered):** whether the structure
   is time-homogeneous or burst/changepoint-localized (Fourier vs wavelet). No
   measure built yet — the program's open instrumentation gap.

The evaluated suite collapses into four **regimes**, and the regime predicts
the ranking (the acid test of "principled" = predicting a held-out bench's arch
ranking from its coordinates alone):

| regime | who wins | evidence (REPORT.md) |
|---|---|---|
| per-token-readable latent | nobody separates (all ≈ oracle) | changepoint `mode`, assumption `s_i`, hedging `c_i` |
| linear-in-window latent | any window arch; per-token floored | backtracking λ |
| order-2 / position-mixing latent | only coincidence/spectral codes | frequency tone, changepoint `tss` / `c_t` |
| substrate defect | nobody | signed_motion sign (memorization confound) |

The organizing concept behind the regimes is **ambience**: a latent is
*ambient* when a single token's marginal already depends on it. Persistent
states leak into every token they persist over, so "global" properties are
usually ambient (hedging drift, changepoint mode, EM's persona density) and
land in regime 1 — per-token codes read them and nothing separates. **A
temporal architecture can only earn its keep on a latent the per-token
marginal cannot see** (regime 2 if reading it is additive-in-window, regime 3
if it needs cross-position comparison). Note "global" ≠ ambient: the
frequency bench's velocity is global to the whole sequence yet has provably
zero single-token MI — and it is the suite's strongest window win. Every card
(either generator) must therefore **argue non-ambience at design time and
measure it at § 8** — the discriminability STOP-gate *is* the ambience
measurement (raw per-token ceiling ≈ raw window ceiling ⇒ ambient ⇒ STOP;
the equality variant handles the partially-ambient case by naming the
ambient floor and testing the residual). Regimes 1–2 cannot separate window
architectures from each other; only regime 3 can. Grounded regime-3
separation exists since stage-6 #3b (the recipe residual — Spectral-TXC
+0.96, everything else at or below the additive ceiling).

**Within regime 3 the winner is a function of the *comparison type*, not
"window arch" generically** — the FB-C1 triple dissociation + the recipe
residual (both 2026-07-23, reviewed) split axis 2's order-2 world three
ways, and cards should tag the subtype because the acid test predicts the
winner from it:

- **phase-relational (odd — sign/quadrature between positions):** the
  coincidence code (TXC-post) converts it — phasepair sign **1.000**;
  spectral reads phase only where a DCT band holds ≥ 2 basis functions
  (multi-index bands at T=8, 0.936; singleton bands are provably
  sign-blind). **This leg is T-conditional on band multiplicity** (FB-C2
  T=16 addendum, reviewed): untrained spectral sign access climbs
  0 → 0.67 → 0.94 at T = 4 → 8 → 16 — phase readability is an
  architectural *prior* that turns on as bands become multi-index, so
  post's ownership of phase is a small-T statement.
- **power / equality (even — quadratic and matching invariants):** the band
  code (Spectral-TXC) linearizes it — multilane power +0.79, recipe
  equality residual +0.96, while TXC-post caps at the additive ceiling
  there: colloquial "coincidence detection" does *not* extend to
  identity-matching.
- **covariance-accumulable (order-2 but additively summable across the
  window):** the additive-gated T-spanning decoder (TXC-pre) — the only
  lift on colored_sources (ρ-ordered), with both coincidence-family codes
  at the floor.

Changepoint straddles the first two because its boundary latent is also
axis-3-localized: post's coincidence reads the boundary at tiny T
(τ 0.66, k-fragile), spectral's stationary bands win the robust
`tss`/`cp` reads — localization interacts with, not overrides, the
subtype rule.

---

## Benchmarks (status)

> **The one-stop registry is [`BENCHMARKS.md`](BENCHMARKS.md)** — every benchmark
> (this evaluated suite **and** the grounded-expansion program), with unambiguous
> `spec-status / framework-registered / arch-verdict` columns, plus the full
> "tried & set aside" record. The table below is the evaluated-suite slice only.

| benchmark | dynamics class | stage | verdict | headline |
|---|---|---|---|---|
| [`backtracking/`](backtracking/) | self-exciting / recurrent (**AC**) | bench run (BatchTopK) | **POSITIVE** | window λ-recovery **0.95** (T≥4) vs per-token **DPI floor 0.41**, robust at `d_sae<F`; survives a uniform BatchTopK backbone |
| [`signed_motion/`](signed_motion/) | order-sensitive step (**AC**) | bench run | **NEGATIVE** | no arch recovers the sign in the scarce regime (`#windows=2F` memorization confound) |
| [`topic_switching/`](topic_switching/) | change-point / sticky (DC+AC) | measured | **ABORT** | autocorrelation is 82% per-doc *composition*, not order; labeler inadequate |
| [`changepoint/`](changepoint/) | change-point / dual-latent (DC+AC) | bench run (BatchTopK) | **SPLIT (two-way)** | per-token pins the DC mode at oracle (1.00 at `d_sae=8`) and sits exactly on the provable AC chance floor; **only the post-squash crosscoder** linearly exposes the boundary (τ **0.66** ≈ 86% of the T=2 in-tile ceiling, `c_t` 0.90) at a DC+content cost — additive (pre-squash / per-position) codes are *provably* blind to equality-pattern latents |
| [`frequency/`](frequency/) | periodic / cyclic tone (**AC / 2nd-moment**) | bench run (BatchTopK) | **POSITIVE** | position-mixing crosscoders recover the hidden tone with a high-pass, Rayleigh-resolved `S(f)` (Spectral-TXC near-oracle **1.00** at T=16, TXC-post 0.53); the **DCT-band inductive bias is decisive** (untrained access 0.64 — the band-limited kernels are tone-detectors at init); **additive-over-position** (TXC-pre 0.27, *flat* `S(f)`) and **per-token** (0.00) are blind; random null flat + memorization above `\|Ω\|·M` flagged |
| [`assumption_consequence/`](assumption_consequence/) | directed grammar (**AC**, grounded) | **spec frozen** (C1; **g7 re-exam confirmed C2**; awaiting blind stage-6) | — | the strict per-sentence (ctx=0) re-labeling *strengthened* the directed A→C asymmetry to **0.297** (2.2× the contextual 0.135) ≫ nulls ≤ 0.056; gate-8 PASS; canonical mirror = the g7 Markov fit |
| [`hedging_drift/`](hedging_drift/) | confidence persistence (**DC**, grounded) | **spec PROVISIONAL** (C2 gate-8 recheck: mirror INVALID) | — | measurement solid (ACF(1) 0.316 ≫ nulls, κ = 0.64) but the real ACF is a long-memory *plateau* (~0.13 through lag 8) no menu mirror reproduces — ar1+trend and semi-Markov both fail ACF(2) ± 0.05; hierarchical-AR(1) extension proposed |
| [`self_reference_echo/`](self_reference_echo/) | problem re-anchoring recurrence (grounded) | **spec frozen** (expansion C2; awaiting blind stage-6) | — | re-anchoring sentences cluster: ACF(1) 0.311 ≫ nulls ≤ 0.068, noise-robust; logistic-AR mirror gate-8 PASS on MI(1); caveats: labeler κ = 0.30 marginal; signature is run-clustering (class label loose) |
| [`expansion/`](expansion/) | — (the loop itself) | Cycle 2 done, review pending | C1: 2P/2A · C2: 2P/5A | the autonomous gated pipeline + coverage ledger; C2 ran under the review's design-time gates 7–8 — 3 of 5 aborts were cheap pre-skeptic gate-8 kills, the sign-falsified card proved prereg discipline, and the g7 re-exam vindicated the strict-labeler rule |

Each benchmark subdir contains (where applicable): `prereg.md` (frozen
preregistration), `measurement.md` (the measure→mirror record), `bench_spec.md`
(frozen architecture-test spec), `bench_record.md` (architecture results), the
`*.py` scripts that produce them, `figs/`, and `results/` (derived stats JSON).
`signed_motion/` uses `bench.md` (single combined writeup).

### Running a benchmark's scripts

Scripts are a package; run from the repo root as
`.venv/bin/python -m experiments.explorations.synthetic.<bench>.<script>`. The canonical leaderboard
(shared) stays at `results/leaderboard.jsonl`; real-label inputs (e.g. the Ward
backtracking labels) stay at `results/`. Examples:

```bash
# backtracking (the positive result)
.venv/bin/python -m experiments.explorations.synthetic.backtracking.gating         # ceilings: per-token vs window
.venv/bin/python -m experiments.explorations.synthetic.backtracking.kernel_order   # held-out kernel-length (K) selection
.venv/bin/python -m experiments.explorations.synthetic.backtracking.measure        # measure real backtracking (stages 2-3)
.venv/bin/python -m experiments.explorations.synthetic.backtracking.mirror         # fit + validate the synthetic mirror
.venv/bin/python -m experiments.explorations.synthetic.backtracking.run_grid 8     # the architecture grid
.venv/bin/python -m experiments.explorations.synthetic.backtracking.render_figs    # frontier figures + stats
# topic-switching (the abort)
.venv/bin/python -m experiments.explorations.synthetic.topic_switching.measure
# changepoint (the dual-latent split)
.venv/bin/python -m experiments.explorations.synthetic.changepoint.gating        # § 8 ceilings + raw-linear access
.venv/bin/python -m experiments.explorations.synthetic.changepoint.run_grid 24   # the 198-cell architecture grid
.venv/bin/python -m experiments.explorations.synthetic.changepoint.render_figs   # figures + stats + record AUTO blocks
# frequency (the periodic / cyclic-tone axis)
.venv/bin/python -m experiments.explorations.synthetic.frequency.gating          # § 8 ceilings + circle S(f) vs random null
.venv/bin/python -m experiments.explorations.synthetic.frequency.run_grid 10     # the 298-cell architecture grid
.venv/bin/python -m experiments.explorations.synthetic.frequency.run_grid_bands 10  # matched-budget band-partition addendum
.venv/bin/python -m experiments.explorations.synthetic.frequency.render_figs     # figures + stats + record AUTO blocks
```

All results route through the canonical runner (code-version stamped); no edits
to `temp_bench/core/`.

---

## Part I — the measure→mirror→bench loop

Six stages, each with a **gate** that must pass before the next. Stage 1 is
frozen (committed) before any data is touched in stages 2–6.

1. **Propose & preregister**
2. **Operationalize on real data** (the labeler)
3. **Measure the temporal signature**  ← *temporal-ness gate*
4. **Fit a synthetic mirror**
5. **Validate the mirror**
6. **Benchmark architectures**  ← *defers to Part II*

### Stage rules

**1 — Propose & preregister.** Before touching data, write down and commit: the
property; its hypothesised temporal character (DC slow-varying / AC
order-sensitive / periodic / bursty); the exact labeler; the statistic(s) to be
measured; the chance & oracle baselines; and the prediction for each
architecture *with a reason*. Frozen — later stages may not change it (only
abort).

**2 — Operationalize (the labeler).** The property must reduce to a **per-token
or per-span signal** (categorical or scalar) via a **named, version-pinned**
labeler. The labeler must be **validated**: report held-out accuracy /
inter-labeler agreement and an estimated **noise floor**. *Gate:* label noise
must be low enough that the stage-3 signal cannot be an artifact of it.

**3 — Measure the temporal signature.** Toolkit: autocorrelation vs lag;
dwell-time / run-length distribution; empirical transition matrix + Markov order;
mutual information `I(L_t; L_{t+k})` vs `k`; burstiness / Fano factor; spectral
density (DC vs AC share). **Temporal-ness gate (mandatory order-0 control):**
compute every statistic on the real ordered stream **and** on a
phase/block-shuffled stream that preserves the marginal but destroys order. The
property counts as *temporal* only if the ordered statistic departs from the
shuffled baseline **beyond both** sampling noise **and** the labeler noise floor.
**If shuffling does not change it, ABORT.** Classify the structure (DC / AC /
periodic / bursty).

**4 — Fit a synthetic mirror.** Pick a process from the **menu (Appendix B)**
keyed to the measured statistic. Fit parameters to match the **specific,
preregistered** statistic; state which statistic is matched **and which structure
is deliberately not**. Embed as a synthetic benchmark per Part II.

**5 — Validate the mirror.** *Weak (required):* the synthetic reproduces the
matched statistic on held-out draws within a stated tolerance. *Strong
(preferred):* a dictionary trained on the mirror behaves like one trained on the
real signal. State which level was reached; never over-claim fidelity beyond it.

**6 — Benchmark architectures.** Follow Part II exactly: capacity matched across
archs and anchored on `F`, the **scarce regime (`d_sae ≤ F`) is the object of
study**, a common `L`-tiled eval window, a **memorization-free** linear probe,
and report the **frontier** (never a single hand-picked cell).

### Validity gates (the anti-off-rail core)

Every investigation must pass **all** applicable gates. A failed gate means
**abort or report-as-negative** — never engineer around it.

*Real-side*
- **Shuffle control:** ordered statistic exceeds the shuffled baseline beyond
  sampling + label noise.
- **Labeler-noise control:** the effect survives the labeler's estimated error.
- **Held-out:** statistics measured on held-out text, not the fit set.

*Synthetic-side*
- **Ground-truth hygiene:** `F` (feature directions) defined cleanly; never
  conflated with derived pattern/window counts.
- **Memorization budget:** any probe has **fewer features than the number of
  distinct patterns it is tested on**.
- **Untrained-encoder control:** a claimed architectural win must **vanish for a
  randomly-initialized model of the same architecture**; else it is a
  probe/architectural-access artifact, not learning.
- **Realistic-regime:** a win must hold at `d_sae ≤ F`, not only over-complete.
- **Capability-vs-artifact:** the winner must *also* reconstruct / recover the
  features — not "recover the latent" while representing nothing.
- **Provable baselines where possible:** prefer a process with a provable
  chance/oracle (e.g. a data-processing-inequality floor) over an empirical gap.
- **Discriminability (STOP-gate, pre-grid):** before any architecture grid, the
  § 8 gating ceilings must show the substrate *can* separate architectures —
  the raw window readout must exceed the raw per-token readout beyond noise (or
  the separation must be provable). If raw per-token ≈ raw window ceiling, the
  bench is non-discriminating **by construction**: record that verdict and STOP
  — do not spend the grid. (Stage-6 lesson: `assumption_consequence` 0.464 vs
  0.466 and `hedging_drift` +0.006 headroom were both on record *before* 990
  cells were spent.) "Grounded + valid mirror" ≠ "discriminates".
  **Equality-latent variant (C4 review):** for a regime-3 primary latent
  (equality / order-2), BOTH raw-linear readouts — per-token *and* window
  concatenation — may sit at chance; that shared blindness is the claim, not a
  failure. There the gate verifies instead that (i) both raw-*linear* readouts
  sit at chance (if raw-linear windows ≫ chance the latent is regime 2 and
  additive codes suffice), and (ii) the latent is *present* in the raw window
  — recoverable by a nonlinear/oracle readout (the changepoint § 8 treatment).
  The bench then tests which architecture's code **linearizes** it. If even
  the nonlinear route fails, the bench is non-discriminating in the other
  direction — nothing can read it — and still STOPs.
  Two execution rules (stage-6 #3 review): the gating script is **committed
  before its first execution** (commit-order evidence of preregistration,
  same as cards); raw-access lines are **threshold-optimized ceilings**, not
  plain-probe scores (a plain probe can sit at chance under class imbalance
  while real access exists).

### Abort / discard conditions

Abort (and report the negative) when any of: the shuffle control shows no
temporal structure; the labeler noise floor swamps the effect; a confound can't
be removed without metric/labeler/regime shopping; the mirror can't match the
statistic; or the compute budget is hit. **Do not keep tuning to force a
positive.** A clean abort is a valid, citable outcome.

### Required output artifact

Each investigation commits **one structured record** containing: the frozen
preregistration; the labeler + version + validation/noise floor; the measured
statistics (**ordered vs shuffled**, with plots); the **temporal-ness verdict**
(gap + uncertainty); the mirror (process, fitted params, match quality, what was
*not* matched); the architecture **frontier** + which controls passed; and a
one-line headline (**which may be negative**). No free-form "it seems temporal"
claims without the backing statistic and passed controls.

### Guardrails

Inherit all `CLAUDE.md` hard rules (never edit `temp_bench/core/`; never
hand-write the leaderboard; everything through the canonical runner).
**Properties** come from the menu (Appendix A); **generators** from the menu
(Appendix B) — bespoke choices require a written justification. **No shopping
after preregistration.** **No claim without its backing statistic and passed
controls.** Real corpora and labelers are version-pinned and cited.

---

## Part II — synthetic benchmark conventions

The rules every synthetic benchmark follows so results are comparable *across
architectures* and *across benchmarks*. A synthetic benchmark's entire value is
that the ground truth is known exactly; if a design can't state its ground truth
cleanly, or can't be evaluated identically across architectures of different
shapes, it is not a good synthetic benchmark.

### 1. State the ground truth exactly

Name two things up front, and keep them distinct:

1. **Feature directions** — the unit directions in `R^{d_in}` the activations
   are built from. Count = **`F`**. These are what a dictionary is meant to
   recover, and what `d_sae` is budgeted against.
2. **Hidden / dynamical latents** — quantities that govern the dynamics but are
   **not directions** (discrete states, signs, phases, chain occupancies,
   continuous parameters). List each with its type and chance/oracle baselines.

A latent that is *not* a direction is **not a feature**; `d_sae` is not "for" it.
Conflating "number of features" with "number of distinct patterns the dynamics
produce" is the most common modelling error — the latter is a derived property,
not ground truth, and must not size `d_sae`.

### 2. Capacity: equal across archs, anchored on `F`, swept

- **Equal across architectures.** `d_sae` and `k_pos` take the **same value for
  every architecture** at each grid point. Per-architecture capacity is
  forbidden.
- **Anchored on `F`.** Sweep `d_sae` relative to `F` (below, at, above) and
  **mark `F` on the axis**. Never anchor on derived pattern/window counts.
- **Swept, not pinned.** There is no canonical "fair" point; report recovery as a
  **function of capacity**. A single operating point is only one labeled slice.

### 3. Per-token sparsity normalization

- `k_pos` is *atoms fired per token* — the same unit for every architecture, so
  **equal `k_pos` is well-defined fairness**.
- For a window arch of length `T`, the window budget is **`k_win = k_pos · T`**,
  so equal `k_pos` holds the per-token budget equal across all `T`. The only
  remaining difference is that a window arch may *allocate* that budget jointly
  across time — the architectural degree of freedom under test. (BatchTopK archs
  pool that budget across the batch; see the backtracking bench for the
  fair-backbone treatment, incl. the post-squash `k_win // T` correction.)
- Keep **`d_sae ≥ k_pos · max(T)`** so no cell clips; label any clipped corner.

### 4. Window size and the apples-to-apples eval window

- **Fix one eval window `L = 2^k`;** constrain architecture windows to powers of
  two, `T = 2^j ≤ L`. Any `2^j ≤ 2^k` divides `2^k`, so every architecture tiles
  the same `L` with no remainder and any subset is mutually comparable.
- **Tile, don't slide.** Sample `L`-windows at **random offsets**, partition each
  into `L/T` **non-overlapping** sub-windows, aggregate every metric over the
  **identical `L` positions** — each evaluated position is encoded/reconstructed
  exactly once by each architecture.
- **Sweep `T`** (e.g. `{2,4,8}`); each `T` is a separately trained model.

### 5. Metrics

- **Feature recovery** (for *directions*): best-matching `|cosine|` of decoder
  atoms vs each ground-truth direction, thresholded to an AUC. Report each named
  direction set separately — never pool.
- **Latent recovery** (for non-direction latents): a **linear probe** on the
  codes over the `L`-window — logistic for categorical, linear for continuous.
  **Linearity is mandatory and load-bearing**: it measures what is *linearly
  decodable* from the code. A nonlinear probe measures the probe's capacity, not
  the representation's, and is permitted only as an explicit ablation. Split by
  example so the score reflects generalization, not memorization.
- **Reconstruction:** the apples-to-apples windowed NMSE of §4.
- **Normalize to [chance, oracle]** where definable, and state them. Where an
  impossibility can be *proven* (e.g. an information-theoretic bound), state it —
  a provable floor beats an empirical gap. Report the **empirical chance floor**
  (finite-sample probes rarely sit exactly at chance) and check a "win" sits well
  outside it.

### 6. Reporting

Report recovery as **curves / frontiers** against capacity (`d_sae`, `k_pos`) and
window size `T`, with `F` marked on the `d_sae` axis. The headline is the
**frontier**, not a single hand-picked cell. Because `d_sae`, `k_pos`, and the
eval windows are identical across architectures, any residual difference is
attributable to architecture — protect that attribution.

### 7. Plumbing (inherited framework rules)

Every result goes through the canonical runner; rows are code-version stamped;
the evaluator re-materializes the ground truth with the **training seed** so
feature directions and latents match what the model trained on (see
[`../../../CLAUDE.md`](../../../CLAUDE.md) and [`../../../docs/framework.md`](../../../docs/framework.md)).
A new benchmark is a **plugin**: a generator + a `configs/data.yaml` datasource
entry + (if a new metric is needed) an evaluator addition. Never edit
`temp_bench/core/`.

### 8. Checklist for proposing a new synthetic benchmark

1. **Ground truth.** State `F` and list every latent with type + chance/oracle.
   Confirm nothing is conflated with a derived pattern count.
2. **What it isolates.** The axis it probes and why existing benches don't cover
   it.
3. **Recoverability.** For each latent, which architectures can recover it in
   principle — ideally with a proof of the chance/oracle bounds.
4. **Capacity grid.** The `d_sae` sweep anchored on `F`, the `k_pos` sweep,
   `d_sae ≥ k_pos · max(T)`.
5. **Windows.** `L = 2^k` and the `T ∈ {2^j ≤ L}` sweep.
6. **Metrics.** Which direction sets get cosine-AUC; which latents get a linear
   probe; the reconstruction metric — all over the common `L` tiling.
7. **Predictions.** Preregister expected recovery per architecture across the
   frontier before running.
8. **Coordinates + discriminability.** State the bench's coordinates on the
   three structural axes (and the regime its primary latent falls in),
   including the **non-ambience argument** — why the primary latent is
   invisible to a single token's marginal (a latent ambient per token is
   regime 1 and dead on arrival). Then run the § 8 gating ceilings and pass
   the discriminability STOP-gate before any grid is spent.

---

## Appendix A — candidate property menu (extensible via proposal)

| property | hypothesised type | candidate labeler / signal | candidate mirror | real data on-branch? |
|---|---|---|---|---|
| topic / topic-switching | DC, sticky (heavy dwell) | sentence-embedding cluster id / LDA | semi-Markov | `fineweb` acts |
| sentiment trajectory | DC, slow drift | sentiment classifier / lexicon | AR(1) / 2-state Markov | needs corpus |
| backtracking / self-correction | **AC, bursty / self-exciting** | Ward event labels | Hawkes / semi-Markov burst | **Ward Llama-3.1-8B ✓** |
| syntactic state / clause depth | order-sensitive | parser depth | stack / branching process | needs parser |
| entity recency / coreference | long-memory | coref chain id | renewal process | needs coref |
| register / formality | DC, very slow | classifier | low-frequency Markov | needs corpus |

## Appendix B — generating-process menu keyed to statistic

| measured signature | process | matched parameter(s) |
|---|---|---|
| exponential autocorrelation | 2-state Markov / AR(1) | decay ρ |
| heavy-tailed dwell times | semi-Markov | dwell distribution |
| multi-state structure | HMM | transition matrix |
| bursty / clustered events | Hawkes (self-exciting) | base rate + kernel |
| rhythmic / periodic | periodic + noise | period + SNR |
| order-sensitive direction | signed-motion-style | step `v` |

---

## Provenance

Everything runs through the canonical runner; each record stamps the **code
version + labeler version + data version**; statistics are computed on held-out
splits with fixed seeds; the output artifact is committed. A result that cannot
be reproduced from its record is not a result.
