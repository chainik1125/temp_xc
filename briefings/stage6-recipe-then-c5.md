---
status: active
created: 2026-07-22
for: runpod
venue: runpod
---

# The 12-hour PhenomenonBench overnight session — stage-6 of `recipe_instruction_phase_runs`, then expansion C5

**You are `runpod`** (the original box — `/workspace/.agent_id` does NOT
exist on your pod; see `agents/README.md`). A second agent (`runpod-b`) runs
the FreqBench session in parallel tonight — `freqbench-c1.md` is NOT yours.
Follow the two-agent shared-branch rules in `agents/README.md` (pull
--rebase before every push; append-only shared files; union merge drivers on
the leaderboard/manifest). Prime directive: **a sound verdict, never a win.**

**Session limits:** ~12 h wall · **$25 API cap** for the C5 phase (expansion
spend meter as usual; judgment on `claude-fable-5`) · rewrite
`agents/runpod/STATUS.md` before every compact. Hard lines: **no § 8 gating
or grid for any C5 graduate** (that needs review); no program-rule/gate
edits; no `temp_bench/core/` edits.

## Phase 1 — stage-6 of `recipe_instruction_phase_runs` (~4–5 h)

The first **grounded regime-3** architecture test. Spec (frozen, C4-reviewed):
`experiments/explorations/synthetic/recipe_instruction_phase_runs/bench_spec.md`
+ `mirror_params.json` (the `hier_categorical` fit). Follow the
stage6-grounded-eval template exactly (recover it from git history if
useful: `git show a96f83f0~1:briefings/stage6-grounded-eval.md`).

### ⚠ Blind discipline
The per-arch predictions are frozen in the spec § 5, including the
falsifier (trained per-token reading `e_t` above the raw-access line ⇒
substrate leak ⇒ NEGATIVE on the bench). Build to the spec's conventions and
report whatever comes out. Never retune after seeing a metric.

### Build (the changepoint/assumption template — no new patterns)
1. **Generator** `recipe_instruction_phase_runs()` in
   `src/temp_bench/data/synthetic.py` (append-only): Layer 1 = the
   `hier_categorical` process from `mirror_params.json` (doc propensities +
   dwell + α-tilted jump chain — mirror `gen_hier_categorical` in
   `expansion/mirrors.py`); Layer 2 = the standard emission, `F = 20`
   (5 phase-signature + 15 content), `d_in = 64`. Expose in `extra`: the
   phase labels `c_t` AND the equality labels `e_t = [c_t = c_{t-1}]`
   (position 0 of each doc: `e_0 = 0` — state the convention in a comment
   and in the record).
2. **Datasource** `toy_recipe_instruction_d64` in `configs/data.yaml`
   (append-only).
3. **Evaluator add-on** dispatched on the new `extra` key in
   `src/temp_bench/evals/synthetic_recovery.py` (guarded no-op for other
   benches ⇒ **protocol stays 1.3.0**): a categorical **phase probe** on
   `c_t` (DC control — chance = marginal-balanced, oracle = 1; anchor
   probes balanced, classes 2–4 sit at 6–7%) and the **primary binary
   equality probe** on `e_t` (chance = pooled match rate, oracle = 1).
   Linear/logistic, per-tile leading edge, split-by-example, common
   `L = 32` tiling.
4. **Tests** per `tests/test_changepoint_bench.py` pattern (shapes, label
   algebra `e_t = [c_t = c_{t-1}]`, dwell anchor vs the mirror params,
   dispatch no-op for other generators).

### § 8 gating — the EQUALITY-VARIANT discriminability STOP-gate (mandatory, pre-grid)
Per README validity gates (the C4-review addition). Verify, on the noiseless
and noisy substrate:
- **(i) both raw-LINEAR readouts of `e_t` sit ≈ chance** — per-token AND
  window-concatenation. This is the regime-3 claim. If raw-linear windows
  read `e_t` ≫ chance, the latent is regime 2 (additively readable): record
  it, **STOP before the grid**, and report — the bench does not test what it
  claims and the review must re-scope it.
- **(ii) the latent is PRESENT**: a nonlinear readout (MLP on raw window
  tiles) or the label-oracle reaches `e_t` well above chance. If even
  nonlinear access fails, the bench is non-discriminating the other way —
  STOP, record.
- Standard ceilings for the DC control `c_t` (per-token should be near
  oracle — that is expected and fine; it is the control, not the claim).
Commit the gating record BEFORE launching the grid.

### The grid + verdict
The locked uniform design (as assumption/hedging): 6 archs ×
`d_sae ∈ {10, 20, 40}` × `T ∈ {1,2,4,8}` × `k_pos ∈ {1,2,4,8,16}` ×
seeds {1,2,42} + untrained, 30k steps, canonical runner, ~495 cells (~1 h at
your worker sizing). Then: blind verdict vs § 5 in `bench_record.md`
(render_figs + AUTO blocks per the existing pattern); registry entry in
`experiments/explorations/synthetic/registry.py` (two axes: `phase` DC
control `primary=False`, `equality` AC/regime-3 `primary=True`); REPORT.md
re-render; BENCHMARKS.md row flip (reg ✓ + verdict); research STATUS § 0
bullet (append-only — your own bullet).

## Phase 2 — expansion C5 (~3–4 h, mostly API; targets are C4-review-approved)

Per the LEDGER C4 entry + review line:
1. **Menu extension: segment-level regime layer** — the hierarchical
   semi-Markov the three-timescale diagnosis (run/segment/doc) points at:
   within-doc segment regimes over the phase process (doc propensities →
   segment-level propensity regimes → per-symbol dwell). Build in
   `expansion/mirrors.py` (append-only) + harness tests demonstrating it
   holds the lag-2–8 structure `hier_categorical` missed on reasoning
   traces, while NOT over-inserting (the hybrid-circularity lesson: keep
   ≥2 genuinely non-fitted moments available).
2. **Gate-8 hardening stays** (≥2 non-fitted moments, uniform relative
   tolerance).
3. **Dated re-freeze `proof-operation-phase-runs-r3`**: C3/C4 labels +
   validation reused; hardened moments preregistered INCLUDING the lag-2–8
   region that killed r2 (e.g. `acf[lag4]` + `mi[lag2]`, held-out docs);
   coordinates + regime-3 claim per checklist item 8.
4. **Calibrate + skeptic** (persist raw verdicts pre-parse). PROCEED ⇒ SPEC
   + BENCHMARKS row; ABORT ⇒ § B row with the killing gate — either is a
   success. **No § 8, no grid, no datasource plugin for the C5 graduate.**
5. **The C5 cosmetic** (recorded in the C4 review): fix the
   `render_records.py` skeptic-header template to name the actual judgment
   model instead of hardcoded "Opus".

## Phase 3 — if hours remain

Bookkeeping depth only: REPORT/registry render checks; LEDGER coverage-grid
update; a self-audit of the Phase-1 record against the README checklist +
validity gates (list any gap honestly — do not fix by re-running).

## Acceptance gate — stop for review

Done when: Phase-1 bench is registered + gated + (if the gate passed) grid
run with the blind verdict recorded, all trackers updated; Phase-2 verdicts
written with the LEDGER cycle log; everything pushed; spend logged. Rewrite
`agents/runpod/STATUS.md`, then **STOP** — this briefing stays until
mac-local review, then it is deleted.
