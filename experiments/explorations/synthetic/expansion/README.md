# Grounded-benchmark expansion — the autonomous, gated measure→mirror loop

This directory is the **expansion arm** of the synthetic-benchmark program: it
*expands the list of grounded benchmarks* by running the proven
measure→mirror loop (the one [`../backtracking/`](../backtracking/) ran by
hand) as an autonomous, **gated** pipeline over real text:

> **hypothesize → select → calibrate on real LM data → freeze → STOP for review**

One cycle produces frozen, architecture-blind benchmark specs (or clean
aborts) across BOTH data domains. It never evaluates architectures — that is
a separate, later, deliberately-blind step (stage 6 of the loop in
[`../README.md`](../README.md)).

## Prime-directive guardrails (structural, not aspirational)

1. **An ABORT is a success.** Nothing is ever tuned to force a temporal verdict.
2. **Blind to architectures.** No arch is trained or evaluated during a cycle.
3. **The null gate is make-or-break**: ordered statistic must beat the
   N1 (within-doc permute) and N2 (trend-preserving) nulls beyond sampling
   noise AND the measured labeler noise floor.
4. **Every labeler is validated** (inter-judge agreement → noise floor ε̂ +
   an independent heuristic cross-check) before its labels count.
5. **Adversarial skeptic pass on every PROCEED** — a separate Opus call fills
   a fixed 5-item kill-rubric (noise floor / leakage / composition /
   circularity / segmentation); any kill demotes to ABORT.
6. **Hard cost cap** ($25/cycle default) enforced by the spend meter in
   `explorations.synthetic.expansion.client` (see `results/spend.json`).
7. **No-leakage labeler design (preregistered).** A per-span label must be
   assignable from the span's *own content*, never from its relation to
   neighbours ("answers a preceding question", "follows from prior") — a
   relational label makes the measured order statistic circular. *(Added after
   Cycle 1 — `question-answer-adjacency` died on this; the skeptic's item (b) is
   the reactive backstop.)*
8. **Non-fitted-moment mirror gate (preregistered).** A fitted mirror must
   reproduce ≥1 statistic it was **not** fit to, within a preregistered
   tolerance — matching only the fitted moment is circular validation. *(Added
   after Cycle 1 — `quotation-burst` died on this despite a clean null gate; the
   skeptic's item (d) is the reactive backstop.)*

Guardrails 7–8 are **design-time / preregistered** (the hypothesis and mirror are
built to satisfy them up front); guardrails 4–5 are the **reactive** validation
and skeptic backstops. Both are required.

The **anti-drift invariant** is [`LEDGER.md`](LEDGER.md): a `domain ×
temporal-class` coverage grid with a per-domain selection floor (≥⌊N/2⌋
calibrated per domain per cycle) and an under-coverage bias (empty/abort-only
cells outrank cells that already hold a PROCEED).

**Measured-class re-filing (Cycle-2 lesson).** A candidate's ledger cell is its
**measured** temporal class (from the calibration signature), *not* the class it
was proposed under. If a candidate proposed for cell X measures as class Y,
**re-file it to Y and leave X unfilled** — otherwise the anti-drift grid lies
about coverage. (Cycle 2: `self-reference-echo`, proposed interaction/equality,
measured pure self-excitation — `ACF`/`Fano`/`excite-ratio` + a `logistic_ar`
mirror identical to backtracking's — so it re-filed to bursty/self-exciting and
interaction/equality stayed empty.)

**Grounding the interaction/equality class (gate-7-clean recipe).** Equality /
interaction is *inherently* about a cross-position relationship, which collides
with gate 7 (label from the span's own content, not its relation to neighbours) —
so binary "does this refer back" labels keep measuring as self-excitation. The
clean path mirrors how the synthetic **changepoint mode** works: use a
**categorical per-sentence content label** (e.g. which sub-goal / operation /
claim-topic the sentence is about — assignable from the sentence alone), then make
the **equality-adjacency `[c_t = c_{t-1}]`** the *measured* statistic. The label
stays per-sentence (gate-7-clean); the equality lives in the measured dynamics.

## The pipeline (one cycle)

| stage | script | output |
|---|---|---|
| 1 hypothesize | `hypothesize.py` (Opus) | `prereg/<name>.md` cards + `results/candidates.json`, **committed before any data** |
| 2 select | `select.py` (separate Opus scoring + deterministic rule) | `results/selection.json`, `selection.md` |
| 3 calibrate | `calibrate.py <name>` (per candidate) | `records/<name>/` — labels, validation, stats, figure |
| 3b render | `render_records.py` | `records/<name>/calibration.md` |
| 4 freeze | manual per PROCEED | frozen `bench_spec.md` graduated to `../<name>/`, LEDGER + cycle log updated, commit + push, **STOP** |

```bash
.venv/bin/python -m experiments.explorations.synthetic.expansion.hypothesize
.venv/bin/python -m experiments.explorations.synthetic.expansion.select
.venv/bin/python -m experiments.explorations.synthetic.expansion.calibrate <name>
.venv/bin/python -m experiments.explorations.synthetic.expansion.render_records
```

Library code (reusable across cycles) lives in
`src/explorations/synthetic/expansion/`: `client` (role-routed judges:
bulk=Haiku-4.5, validate=Sonnet-5, think=Opus-4.8 + the metered cap),
`signature` (binary/categorical/scalar batteries + N1/N2/N3 nulls),
`labeler` (chunked judge runner + κ/ε̂ validation), `corpus` (pinned fineweb
sampling), `mirrors` (Appendix-B menu, fit+generate+validate). Tests:
`tests/test_expansion_harness.py`.

## Data domains (Cycle-1 pins)

- **reasoning-trace** — the 300 Ward Stage-A R1-Distill traces (25,528
  sentences) at `results/c7_backtracking/stage_a/sentence_labels.json`.
- **text-corpus** — 400 fineweb docs (60–200 sentences each; 36,805
  sentences), streamed sample pinned at `data/fineweb_sample.json` (seed 0,
  `sample-10BT`).

Bulk labels are cached per candidate at `records/<name>/labels.json` — a
crashed calibration resumes without re-spending.

## Cycle log

See the bottom of [`LEDGER.md`](LEDGER.md). One cycle per review; the next
cycle starts only after human review of the previous one.
