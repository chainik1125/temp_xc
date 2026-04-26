---
author: Han
date: 2026-04-26
tags:
  - design
  - complete
---

## S decision revised: S=32 for everything (was S=128/64/20 sweep)

### What changed

Phase 7's headline S parameter for sparse probing dropped from a
multi-S sweep `(128, 64, 20)` to a single value **S = 32**.

Sanity-check probes specifically use **S = 20** to match Phase 5's
tail-length convention exactly (cross-phase numerical sanity).

### Why

Two reasons compounded:

1. **Probing time at S=128 was 6× slower than Phase 5** because the
   number of sliding windows grows linearly with S. Per-task encode
   for a window arch at T=5 was ~5 min on H200 vs ~30 sec at S=20.
   Multiply by 36 tasks × 49 archs × 3 seeds × 4 (S × k_feat) cells
   and the probing pass alone would take ~60 hr — incompatible with
   the deadline.

2. **The (T, S) validity rule was found to be over-aggressive.** I had
   coded `kept_windows = S − 2T + 2` based on a "drop first T−1
   windows whose tokens have less preceding-tail context" rule. That
   rule is wrong: window archs encode the window IN ISOLATION at
   probe time (no recurrent state, no cross-window dependency). The
   window IS the encoder input; it doesn't need additional preceding
   context. Correct rule: **kept = S − T + 1**, **validity = S ≥ T**.

   Discovery: working through what S would be valid for each canonical
   arch with the user. The user noticed "S=32 should be enough — why
   are T=18, 20, 24, 28, 32 invalid?" The invalidity came from the
   buggy `S ≥ 2T − 1` formula, not anything intrinsic.

### Consequences

**At S=32 with the corrected rule**:

- All 48 of 49 canonical archs valid (only row 49 SubseqH8 T_max=64
  is structurally invalid, since 32 < 64).
- ~3× faster than S=64, ~6× faster than S=128. Probing pass for the
  full 49 archs × 3 seeds at S=32 estimated ~10-15 hr (vs ~60 hr at
  S=128).
- Mean-pool aggregates over up to 32 − T + 1 windows per example
  (e.g., T=5 gets 28 windows; T=20 gets 13; T=32 gets 1).

**At S=20 (sanity only)**:

- Valid for T ≤ 20: covers all 47 archs except the 2 SubseqH8
  T_max ∈ {32, 64} cells.
- Direct numerical comparison to Phase 5 (which probed at S=20
  exclusively).
- ~10× faster than S=128 — sanity verdict in ~30 min instead of hours.

### Code changes

`run_probing_phase7.py`:
- `aggregate_s`: `lo = last_idx - effective + 1` (was `+ T`).
- `cell_is_valid`: `return S >= T` (was `S >= 2*T - 1`).
- `HEADLINE_S = (32,)` (was `(128, 64, 20)`).

Probe cache build is unchanged (still stores anchor + mlc_tail at
S=128); we just slice to the last S=32 at probe time. So no rebuild
needed; the existing 488 GB cache is fine.

### Earlier-doc supersession

Earlier dated logs that referenced S=128 / 64 / 20 stay as-is for
historical record. The corrected `(T, S)` math now lives in
`plan.md` §"(T, S) validity + short-sentence handling" and supersedes
the old version.
