---
status: active
created: 2026-07-22
for: any (runpod preferred — CPU-cheap)
venue: runpod | local
---

# FreqBench cycle 1 (FB-C1) — first theorem-first cards + the full FreqFrac pass

**Governing protocol:** `experiments/explorations/synthetic/freqbench/LOOP.md`
(read in full, plus `PORT.md` § A–B and README § "The two generators, one
substrate"). Prime directive unchanged: a sound verdict, never a win — an
ABORT on the proof gate or the non-triviality battery is a success.

## 0. The full FreqFrac pass (substrate instrumentation, do first)

Run `freqfrac_report.py` over **all six** registry benches at the canonical
matched cells (seed 1), plus seeds {2, 42} for the two prototype benches:

```bash
.venv/bin/python -m experiments.explorations.synthetic.freqbench.freqfrac_report \
    frequency backtracking signed_motion changepoint assumption_consequence hedging_drift
```

Notes: it trains any missing checkpoint via the canonical trainer (the
migrated pod's store is empty — expect ~30 cells to train; CPU-cheap), hard-
asserts each reconstructed `train_key` against the leaderboard row (an
assertion failure is a STOP-and-report, never a workaround), and writes **no
leaderboard rows**. Commit `freqbench/results/freqfrac_stats.json` + figs.
Deliverable: the axis-1 (DC↔AC) coordinate measured per (bench × arch),
trained vs untrained — mac-local's local pass covers frequency+backtracking
only; this completes the suite.

## 1. FB-C1 cards (freeze BEFORE any construction — LOOP.md card format)

Freeze all three seed cards by commit, then execute in priority order within
budget:

1. **FB-2 multilane superposition** (priority; expect PROCEED-grade):
   3 simultaneous circle tones in orthogonal planes (`d_in = 24`-ish, panel
   conventions per Part II). Proof obligations: per-lane periodogram oracle
   (P5), per-token/additive floor (P1/P2 per lane), **memorization immunity
   by construction** (|Ω|³M³ ≈ 10⁹ templates — state the count in the card).
   Regime-3 claim: position-mixing required per lane; the sprint measured
   multiband > vanilla (0.96 vs 0.91, no seed overlap) — the card's frozen
   prediction, now under the fair BatchTopK backbone where it may well
   FAIL (that would be an informative negative about the sprint's plain-TopK
   result, not a defeat).
2. **FB-3 colored sources**: per-coordinate AR(1) at lag D (port from
   `origin/dmitry-synthetic:src/v6_colored_sources/` — README there has the
   math). Proof obligations: CS-1 local impossibility (iid ⇒ Rec ≲ log(H)/N)
   + CS-2 lag-D recoverability and the **W = D+1 phase transition** as the
   frozen prediction. NOTE: this is a *feature-direction-recovery* bench
   (cosine-AUC primary, not a latent probe) — say so explicitly in the card;
   it fills the recovery-flavor gap in the coordinate system.
3. **FB-1 phasepair** (only if budget remains): ±velocity pairs, identical
   power spectra. The `c_relevance` skeptic item needs a real answer (which
   real phenomenon is phase-coded?) — if none is defensible, mark `spanning`
   with the research reason or let the skeptic kill it honestly.

## 2. Per card: build + T1/T2 + skeptic (NO § 8 gating, NO grids)

- Generator + exact parameterization; **T1**: discharge every proof
  obligation (analytic note in the record, or a committed
  `verify_theory`-style numerical check over the actual parameter range).
- **T2 battery**: symmetry/relabeling audit; bag-of-symbols control;
  memorization budget at the capacity extremes; probe budget scaled to code
  dim; shuffle semantics stated (per-window independent permutations).
- **Skeptic** (LOOP.md rubric, judgment on `claude-fable-5`): for
  `c_relevance`, FB-2/FB-3 may cite the backtracking axis-1 DC-dominance and
  the changepoint localization story; FB-1 must make its own case.
- Datasource plugins + `configs/data.yaml` entries + tests ARE in scope
  (that is the construction); **architecture grids and § 8 gating are NOT**
  — the discriminability STOP-gate (equality variant where applicable) runs
  at the stage-6-analogue briefing after review.

## 3. Acceptance gate — stop for review

Done when: the FreqFrac full-pass artifacts are committed; all frozen cards
carry verdicts (built + T1/T2 + skeptic → PROCEED, or ABORT with the killing
gate recorded); `BENCHMARKS.md` rows added (provenance `theorem-first`;
§ B for aborts), LOOP-cycle log appended to `PORT.md`, research STATUS § 0
updated; **$25 cap**, spend logged. Then STOP — briefing stays until
mac-local review, then it is deleted.

## Addendum (post first-pass, mac-local): T=8 frequency cells

The local 12-cell pass (PORT.md § G) found the frequency **high-pass
acceptance check has no power at `T_can = 4`** (half of Ω folds below the
first DCT bin). In step 0, ALSO run:

```bash
.venv/bin/python -m experiments.explorations.synthetic.freqbench.freqfrac_report frequency --T 8
```

and read the high-pass check from those cells (spectral was 0.96 on the tone
at T=8 — its firing curve there is the meaningful one).
