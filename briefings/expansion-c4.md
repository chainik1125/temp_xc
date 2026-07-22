---
status: active
created: 2026-07-22
for: runpod
venue: runpod
---

# Expansion Cycle 4 — hierarchical-categorical mirror + the first grounded regime-3 SPEC

**Goal.** Resume the PhenomenonBench expansion loop under the **revamped
program rules** and claim the interaction/equality prize: a *grounded*
benchmark whose primary latent is **regime 3** (order-2 / position-mixing) —
the only regime that separates window architectures from each other, and the
one regime where today every benchmark is theorem-provenance (the program's
research-sin exposure). This is a measure→mirror cycle (stages 1–5 only): **no
architecture grids, no datasource plugins** — stage 6 for any new SPEC is a
separate reviewed briefing.

## 0. Sync + the new rules (read FIRST)

- Pull. Stage 6 is **reviewed + PASSED and closed** (`a96f83f0` deleted your
  stage-6 briefing). Your `agents/runpod/STATUS.md` "awaiting review" state is
  stale — rewrite it when you start this.
- `experiments/explorations/synthetic/README.md` § **"The two generators, one
  substrate"** is new and governing: the program is now explicitly
  FreqBench (theorem-first) + PhenomenonBench (data-first) on one shared
  substrate, with a **3-axis coordinate system**, a **4-regime table**, a
  **discriminability STOP-gate** in the validity gates, and **checklist
  item 8** (coordinates + discriminability). `BENCHMARKS.md` now carries a
  provenance column — tag anything you graduate as `grounded`.
- Run the loop's API calls on **Fable 5** (`claude-fable-5`) — C1–C3 predate
  it. Key in `/workspace/.tokens/`. (5-family: no `temperature`, thinks by
  default — your client already handles both.)
- Meanwhile mac-local is building the FreqBench port (phase 2) — expect
  commits under `experiments/explorations/synthetic/freqbench/` and
  `src/explorations/`; pull before you push.

## 1. Build first — the C4 menu extension (the identified blocker)

1. **Hierarchical categorical mirror** — the categorical `hier_ar1`: per-doc
   phase propensities + a jump chain. This is the extension **both** C3
   int/eq aborts point at (LEDGER "C3 mirror-menu resolution": real
   categorical streams hold self-match *plateaus* — recipe ACF 0.48→0.28 over
   8 lags — that dwell+jump processes cannot generate). Build + harness-test
   it exactly like C3's `hier_ar1` / `periodic_hawkes` build.
2. **Gate-8 hardening: ≥2 non-fitted moments** for mirror validation (the C4
   target recorded in the LEDGER). C3 lesson: an over-expressive mirror that
   fits everything measured leaves nothing non-inserted and dies (rightly) on
   skeptic circularity — any periodic re-freeze must preregister a
   non-inserted moment (gap-distribution shape / cross-doc period stability).

## 2. The cycle, under the revamped rules

- **Primary target: interaction/equality, both domains** (the unfilled LEDGER
  row). Start from the two C3 real-signal aborts as re-freeze candidates under
  the new mirror — `proof-operation-phase-runs` (reasoning; self-match ACF(1)
  0.286 ≫ N1 hi 0.047, κ 0.59; died only on gate-8 MI(2)) and
  `recipe-instruction-phase-runs` (text; ACF(1) 0.479 ≫ N1 hi 0.204, κ 0.61;
  died only on gate-8 ACF(4)). Fresh cards allowed under the gate-7-clean
  categorical recipe.
- **Every card states its 3-axis coordinates + regime claim** (checklist
  item 8), and **discriminability applies at design time**: a card whose
  primary latent is per-token-readable (regime 1) or merely linear-in-window
  (regime 2) is dead on arrival. Each card must argue *why* reading its latent
  requires cross-position comparison (order-2+) or required integration —
  "grounded + valid mirror ≠ discriminates" is the stage-6 lesson this cycle
  exists to answer. (The *empirical* § 8 raw-ceiling check runs at stage-6
  build time, pre-grid, per the new STOP-gate — not in this cycle.)
- All C1–C3 machinery stays: frozen cards before data, blind selection, nulls,
  gates 7–8 (hardened per § 1), the skeptic, the uniform relative gate-8
  tolerance, prereg discipline. Prime directive unchanged: **a sound verdict,
  never a win** — an abort that kills a card on the discriminability argument
  is a success.
- **Budget: $25 cap** (C3 spent $8.20). Record spend in the LEDGER entry.

## 3. Acceptance gate — stop for review

Done when: the mirror extension is built + tested; the cycle's verdicts are
written (SPEC graduations and/or informative aborts, each with its gate/kill
recorded); `expansion/LEDGER.md` + `BENCHMARKS.md` (provenance-tagged) +
research `STATUS.md` § 0 updated; everything committed + pushed. **Do NOT
proceed to stage 6.** Then STOP — this briefing stays until mac-local reviews,
then it is deleted (per `briefings/README.md`).

## Standing constraints

- No edits to `temp_bench/core/`; no leaderboard writes (this cycle trains
  nothing); paper-section names; commit style as in the log.
- `list_item_parallelism` stays deprioritized (redundant class, weak mirror);
  `self_reference_echo` not planned.
