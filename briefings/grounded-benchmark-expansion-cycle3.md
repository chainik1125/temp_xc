---
status: active
created: 2026-07-18
for: runpod
venue: runpod
supersedes: grounded-benchmark-expansion-cycle2.md (Cycle 2, done + reviewed)
---

# Grounded-benchmark expansion — Cycle 3

Cycle 2 done + **reviewed (approved with corrections)**. Continuation — infra +
harness exist. Read first: [`.../expansion/README.md`](../experiments/explorations/synthetic/expansion/README.md)
and [`.../expansion/LEDGER.md`](../experiments/explorations/synthetic/expansion/LEDGER.md).
Same loop, same guardrails (now 8), same gated cadence.

## What the C2 review changed (all now in the README — apply them)

- **Measured-class re-filing.** A candidate's ledger cell is its **measured**
  temporal class, not the proposed one. Re-file on mismatch; the proposed cell
  stays unfilled. (C2: `self-reference-echo` re-filed interaction/equality →
  bursty; **interaction/equality is empty again**.)
- **Interaction/equality — the gate-7-clean recipe.** Equality is inherently
  relational (clashes with gate 7), so binary "refers-back" labels keep measuring
  as self-excitation. Use a **categorical per-sentence content label** (sub-goal /
  operation / claim-topic, assignable from the sentence alone) and make the
  **equality-adjacency `[c_t=c_{t-1}]`** the *measured* statistic — how the
  synthetic changepoint mode works.
- **Gate-8 tolerances must be RELATIVE** (to the statistic's magnitude or
  null-band width), preregistered as such — not raw absolutes. (C2:
  `list-item-parallelism`, a strong real signal, died on a 4%-relative overshoot.)

## Targets (priority order)

1. **interaction/equality — TOP, genuinely empty, both domains.** Use the
   categorical-label recipe above. This is the prize: a *grounded analogue of the
   changepoint equality-latent*, the one axis where the "where the nonlinearity
   sits" theory predicts additive codes are provably blind. Hypothesize fresh
   categorical-label cards here (freeze first).
2. **Two C2 re-freezes — real signals killed only on mirror methodology:**
   - `list-item-parallelism` (text-corpus, ACF(1)=0.52, κ=0.64) — re-freeze the
     card with a **magnitude-relative gate-8 tolerance**; expect PROCEED. This is
     also the likely **first text-corpus PROCEED** (C2 went 0/3).
   - `computation-verification-alternation` (reasoning, periodic, spectral peak
     3.84 real) — re-freeze with the **periodic+self-exciting hybrid mirror**
     (menu extension below); the events are periodic AND bursty.
3. **`hedging-drift` mirror re-fit → upgrade.** Build the **hierarchical-AR(1)**
   menu extension (per-sequence slow level + AR(1)) to reproduce the long-memory
   ACF *plateau* the short-memory mirrors miss. If it passes gate 8 on a
   non-fitted lag (e.g. ACF(2)/ACF(4) within a relative tol), upgrade
   `hedging-drift` SPEC*→SPEC.
4. **Fill remaining proposed cells** from the frozen C1 cards:
   `enumeration-cadence` (periodic × text), `goal-restatement-recurrence`
   (long-memory × reasoning).

## Appendix-B menu extensions to build (in `expansion/mirrors.py`)

- **hierarchical-AR(1)** — a per-sequence latent level (slow/heavy-tailed across
  docs) + AR(1) within — for long-memory *plateaus* (hedging; the long-range tail
  of assumption-consequence). Add to the menu + fit/validate + a harness test.
- **periodic + self-exciting hybrid** — a periodic base rate modulated by a Hawkes
  kernel — for phenomena that are rhythmic AND clustered (verification).

## Scope + budget

- **N ≈ 6, ≥3 per domain** (categorical interaction/equality cards + the two
  re-freezes + the two frozen cards), plus the hedging mirror re-fit rider.
- **Cost cap $25/cycle.** C2 was $14; the labeling-heavy new interaction/equality
  cards + re-freezes fit — meter + hard-stop as always.

## Acceptance gate (per cycle)

README per-cycle gate, **plus**: interaction/equality attempted via the
categorical recipe (its cell no longer both-domains-empty is the goal, ≥1 genuine
measured-interaction PROCEED is the win); the two re-freezes resolved; the
hierarchical mirror built + hedging re-fit resolved (SPEC or still-SPEC* logged);
gate-8 tolerances relative; measured-class filing applied to every new PROCEED;
LEDGER + cycle log updated; committed + pushed; **STOP for review. No arch runs.**

## Hard rules

`TEMP_BENCH_ALLOW_DIRTY=1`; `.venv/bin/python`; **never edit `temp_bench/core/`**;
calibration is text-only; version-pin corpus snapshot + judge models; **scope your
commits** — do not `git add -A` stray run-logs into the tree (C2 swept in 7.6k
lines of `frequency/results/*.log`; now gitignored). Prime directive: a sound
verdict, never a win. When Cycle 3 is done + reviewed, delete this briefing.
