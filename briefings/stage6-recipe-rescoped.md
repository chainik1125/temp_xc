---
status: active
created: 2026-07-23
for: runpod
venue: runpod
---

# Stage-6 #3b — `recipe_instruction_phase_runs`, re-scoped: the regime-3 residual head-to-head

**You are `runpod`** (no `/workspace/.agent_id` on your box). `runpod-b` may
still be running FB-C1 in parallel — its briefing is not yours; follow the
two-agent shared-branch rules (`agents/README.md`). Prime directive: **a
sound verdict, never a win.**

**Context.** Stage-6 #3 ended in a § 8 STOP (review-approved — see
`recipe_instruction_phase_runs/bench_record.md` incl. the review section):
raw-linear access to `e_t` is 0.614 per-token / 0.720 window via the
class-conditional continuation leak, additive ceiling 0.771, exact 1.000.
The review adopted **re-scope option 1**: the bench's primary axis is now
the **regime-3 residual** — the 0.229 of balanced accuracy reachable only by
position-mixing. The bench build is DONE and registered; this briefing adds
the re-scoped metric + the dated re-freeze, then runs the grid.
Budget: ≤ $5 API (no labeling; incidentals only). Est. 3–5 h.

## Phase A — freeze BEFORE any training (strict order, each its own commit)

**A1. The residual metric** (additive; protocol stays 1.3.0): in the
`recipe_recovery` add-on, add `equality_residual_recovery` =
`(balacc(e_t) − 0.771) / (1.000 − 0.771)`, clipped reporting NOT applied
(negative values are informative — report raw). Keep the existing
chance-normalized `equality_recovery` as a diagnostic key. The floor
constant 0.771 is the § 8-measured pair-additive ceiling — cite
`results/recipe_gating_stats.json` in a comment; do not re-derive it. Tests
(metric algebra + dispatch no-op) extend the existing bench tests.

**A2. The dated § 5 re-freeze amendment** to `bench_spec.md` — commit
verbatim, BEFORE the grid. The predictions below were **frozen by mac-local
in this briefing (2026-07-23)**; you may append sharpening *reasons* but not
change directions or add hedges after any cell runs:

- **DC control `c_t`:** every arch at/near oracle (per-token 1.000 raw —
  expected; it is the control).
- **Residual axis (primary):**
  - `batchtopk_sae`, `tsae`, `stacked_batchtopk`, `txc_batchtopk_pre` —
    **≈ 0 (or negative)**: a linear probe over per-position codes is an
    additive-over-positions readout, bounded by the 0.771 ceiling; TXC-pre's
    summed code is additive up to BatchTopK competition (changepoint's
    provable additive-blindness precedent).
  - `txc_batchtopk_post` — **positive residual**, strongest at T = 2,
    expected k_pos-fragile (the changepoint boundary precedent: τ 0.66 at
    T=2, vanishing at higher k).
  - `spectral_txc` — **positive residual**, expected k_pos-robust
    (changepoint tss/cp precedent).
- **Untrained control:** any positive residual must vanish (or drop to the
  architectural-access floor) at random init, else it is access, not
  learning — report both.
- **Falsifier (indicts the bench, not an arch):** any additive-family arch
  with residual clearly > 0 beyond seed noise ⇒ the residual normalization
  or the substrate leaks beyond the § 8-measured additive ceiling ⇒ verdict
  NEGATIVE on the re-scoped bench; report, never re-normalize after the
  fact. Use threshold-optimized ceilings for any raw-access comparison
  (README rule).

**A3. Gating addendum** (no new computation): dated note in the gating
record that under the re-scoped axis the § 8 discriminability condition
reads *nonlinear 1.000 ≫ additive 0.771* — the separation the bench now
tests — citing the existing committed stats from `b463c4a0`.

## Phase B — the grid + blind verdict

The locked uniform design, exactly as assumption/hedging: 6 archs ×
`d_sae ∈ {10, 20, 40}` × `T ∈ {1,2,4,8}` × `k_pos ∈ {1,2,4,8,16}` ×
seeds {1,2,42} + untrained, 30k steps, canonical runner, ~495 cells at your
worker sizing. Then:

- `render_figs` + `bench_record.md` verdict section — **blind against A2**,
  written from the numbers (POSITIVE / NEGATIVE / SPLIT on the residual
  axis; the DC control and companion NMSE/eauc panels per convention).
- Registry entry in `experiments/explorations/synthetic/registry.py`:
  axes = (`phase` DC control, `primary=False`; `equality_residual` AC
  regime-3, `primary=True`, metric `equality_residual_recovery`).
- REPORT.md re-render; BENCHMARKS.md row updated (reg ✓ already; add the
  verdict + a one-line residual-axis note); research STATUS § 0 bullet
  (append-only).

## Acceptance gate — stop for review

Done when: A1–A3 committed in order (freeze-order must be provable from the
log), grid 495/495 with 0 unexplained failures and no duplicate eval_keys,
the blind verdict + records + trackers pushed, `agents/runpod/STATUS.md`
rewritten. Then STOP — briefing stays until mac-local review, then it is
deleted.
