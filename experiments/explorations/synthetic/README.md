# Synthetic temporal benchmarks

This directory is the **synthetic-benchmark program**: find a *measurable*
temporal property of real LM behaviour, fit a faithful synthetic *mirror* of it,
then benchmark whether a window/temporal dictionary exploits that structure
better than a per-token SAE. Each benchmark is a **self-contained subdir**
holding its docs, scripts, figures, and results.

It lives under `experiments/explorations/` — the home for exploratory experiments
(research that may or may not graduate into `temp_bench`); `synthetic` is the
first such exploration. Any reusable *library* code it develops goes in `src/`
(its own `src/explorations/<name>/`, or `temp_bench` once it graduates); this
tree holds the experiments + their artifacts. Scripts run as a package, from the
repo root: `.venv/bin/python -m experiments.explorations.synthetic.<bench>.<script>`.

This README is the **single governing doc** for the program — the prime
directive, the measure→mirror→bench loop and its validity gates, and the
conventions every synthetic benchmark follows. Read it before proposing or
implementing a benchmark.

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
