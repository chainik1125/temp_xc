# Grounded-benchmark expansion — coverage ledger

The **anti-drift invariant** for the autonomous expansion loop (see
[`../../../../briefings/grounded-benchmark-expansion.md`](../../../../briefings/grounded-benchmark-expansion.md),
or the standing `README.md` once this graduates). Every candidate temporal
property lands in exactly one `domain × temporal-class` cell. Selection each cycle
must (a) keep a **per-domain floor** (≥⌊N/2⌋ calibrated from each domain) and
(b) **prioritize under-covered cells** (empty / abort-only) over cells that
already hold a PROCEED. Update this file at the end of every cycle and report the
grid — an empty domain is then a visible, prioritized gap, not silent drift.

**Cell status:** `—` empty · `prop` proposed (prereg frozen) · `calib:ABORT` ·
`calib:PROCEED` · `SPEC` (PROCEED graduated to a `synthetic/<name>/` benchmark).

| temporal-class | reasoning-trace | text-corpus |
|---|---|---|
| **DC-slow-drift** (state persists, slow) | — | `calib:ABORT` — topic_switching (autocorr 82% per-doc *composition*, not order; labeler inadequate) |
| **AC-order-sensitive** (depends on order) | — | — |
| **periodic** (rhythmic/cyclic) | — | — |
| **bursty/self-exciting** (clustered events) | **`SPEC`** — backtracking (ACF(1)=0.36, ~3.6× self-excitation; Sonnet judge; Hawkes mirror) | — |
| **interaction/equality** (cross-position compare) | — | — |
| **long-memory** (renewal / heavy-tail) | — | — |

## Notes / provenance

- **backtracking** is the anchor — the one property that ran the full measure→
  mirror loop and PROCEEDed (`synthetic/backtracking/`). It is the template every
  new candidate imitates.
- **topic_switching** is the cautionary ABORT — the loop working correctly on a
  property that turned out to be composition, not order. Its measured geometric
  dwell later *anchored* the (synthetic) changepoint bench, but topic_switching
  itself never became a grounded benchmark.
- The abstract benchmarks (signed_motion, frequency) are **not** in this ledger —
  they are mathematical constructs, not grounded mirrors, and are out of scope for
  the grounded-expansion loop.
- **10 of 12 cells are empty.** That is the space to expand into, balanced across
  both columns.

## Cycle log

_(append one line per cycle: date · candidates calibrated (domain split) ·
verdicts · spend · what the next cycle targets)_

- _(none yet — Cycle 1 pending on runpod)_
