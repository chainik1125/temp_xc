# FreqBench generator loop — the theorem-first autoresearch protocol

**Status: RAILS FROZEN (2026-07-22, mac-local; autoresearch-revamp phase 3a).**
The FreqBench counterpart of the PhenomenonBench expansion loop
(`../expansion/LEDGER.md`): where PhenomenonBench anchors a benchmark in a
*measured* property of real LM behaviour, this loop anchors it in *proofs*.
Same prime directive (a sound verdict, never a win), same substrate, same
registry — different epistemic anchor, therefore different gates: you cannot
reward-hack a theorem, but you can prove the wrong theorem, build a trivial
task, or probe structure nothing real exercises. The gates below target
exactly those three failure modes, each of which the original FreqBench
sprint hit and caught (`PORT.md` § A–B).

## What one cycle produces

Constructed benchmark(s) with **provenance `theorem-first`** in
`BENCHMARKS.md`, each: registered on the shared substrate (datasource plugin +
Part II conventions, frozen 6-arch panel, canonical runner), carrying its
**proof obligations discharged in the record**, its 3-axis coordinates
(FreqFrac-measured where applicable), and frozen per-arch predictions —
staged for a later blind B×A run (the stage-6 analogue). The loop never
trains an architecture itself.

## The unit of proposal: the axis-point card

Frozen (committed) before any construction, one card per candidate:

1. **Target coordinates + gap claim.** Where on the 3 axes (README § "The two
   generators") the task sits, and why the existing suite leaves that point
   uncovered (cite the registry — redundancy is a kill).
2. **Constructed task.** Generative process, exact parameterization, ground
   truth per Part II § 1 (F stated; latents listed with type).
3. **Proof obligations** (the theorem-first anchor), in the `PORT.md` § B
   registry format:
   - **Ceiling:** the oracle readout + its accuracy (analytic where possible,
     else a numerically verified strong reference — the `verify_theory.py`
     pattern).
   - **Floor:** what is *provably unreadable* by which code class (per-token /
     additive / linear-on-stacked), with the argument named (phase-averaging,
     DPI, local-impossibility, symmetry).
   - **Non-triviality:** why the task is NOT solvable by a symmetry,
     relabeling, or bag-of-symbols route (see gate T2).
4. **Regime claim + design-time discriminability** (README checklist item 8):
   which regime the primary latent falls in, the **non-ambience argument**
   (ideally a proof that the single-token marginal is independent of the
   latent — the P1 pattern), and which arch families the proofs *predict
   apart*. Regime-1-only cards are dead on arrival, same as PhenomenonBench.
5. **Memorization audit (P6).** The whole-window template count vs the
   `d_sae` sweep: either the count kills the memorization route by
   construction (the multilane pattern, |Ω|³M³) or the sweep must cross the
   threshold and the card predicts the jump — silent memorization wins are
   the historical failure of `signed_motion` (#windows = 2F).
6. **Frozen per-arch predictions** with reasons, including at least one
   **falsifier** (an outcome that would indict the substrate, not crown a
   winner).

## Gates (all must pass; any fail ⇒ ABORT, recorded in BENCHMARKS § B)

- **T1 — proof gate.** Every ceiling/floor obligation is discharged: an
  analytic argument written in the record, or a numerical verification
  script committed alongside (exact combinatorics / oracle-vs-simulation
  agreement across the parameter range actually used). A proof about a
  *different* parameterization than the built task is a FAIL (the sprint's
  H0:50 lesson: the proposal's 10-frequency task was symmetry-trivial *as
  specified*; the theorem forced the circle embedding).
- **T2 — non-triviality battery** (the empirical controls the proofs are
  checked against, all committed):
  - **symmetry/relabeling audit** — is there a group action on symbols that
    maps class to class? If yes the task measures geometry you didn't build;
  - **bag-of-symbols control** — mean-pooled token codes + MLP: must FAIL if
    the card claims order-sensitivity (speed lives in the symbol set,
    direction needs order — the sign-pair lesson);
  - **memorization budget** — the § 5 audit run empirically at the capacity
    extremes; **probe budget scales with code dimension** (the H=2048
    self-correction: probe starvation masquerades as architecture failure);
  - **shuffle semantics stated** — what a within-window shuffle destroys for
    THIS task (for cyclic/set tasks it is not a full null; only phase/order
    tasks shuffle to chance), with per-window independent permutations.
- **T3 — discriminability STOP-gate** (shared with PhenomenonBench, README
  validity gates, incl. the **equality-latent variant**): § 8 gating ceilings
  before any grid is spent.
- **T4 — substrate compliance.** Runs on the frozen panel through the
  canonical runner under Part II capacity/L0/window/metric conventions; ONE
  registry row, provenance-tagged; FreqFrac coordinates computed at bench
  time (`freqfrac_report.py`). A bench that needs a panel change is a
  proposal to the *program*, not a card.

## The skeptic (fixed kill-rubric, judgment model = Fable 5)

Adapted from the expansion skeptic; five items, any kill ⇒ ABORT:
`a_proof_circularity` (the "proof" assumes the conclusion or proves a
different task than the one built) · `b_triviality` (a symmetry / bag /
memorization route survives T2) · `c_relevance` (no real phenomenon occupies
the target coordinates — cite a PhenomenonBench measurement, e.g. the
backtracking DC-dominance for axis-1 low-band, or mark the card explicitly
`spanning` with the research reason) · `d_redundancy` (an existing bench
already discriminates at these coordinates) · `e_substrate` (hidden panel /
convention deviations).

## Seed cards for cycle FB-C1 (from the sprint port, `PORT.md` § A/E)

1. **FB-2 multilane superposition** — priority: regime 3 by construction,
   memorization-immune (|Ω|³M³ ≈ 10⁹), and the only known task separating
   multiband from vanilla TXC (0.96 vs 0.91, no seed overlap at H=64-scale).
2. **FB-3 colored sources** — the direction-*recovery* flavor: CS-1 local
   impossibility (iid ⇒ Rec ≲ log(H)/N) + the W = D+1 memory-depth phase
   transition; fills the "feature recovery vs latent recovery" gap in the
   coordinate system.
3. **FB-1 phasepair** — phase-vs-power dissociation; needs the sharpest
   `c_relevance` argument (which real phenomenon is phase-coded?).

## Cadence, roles, venue

Cards frozen (committed) → build + T1/T2 + skeptic autonomously. A graduated
card **may proceed to § 8 gating and the uniform B×A grid in the same
session** provided ALL of: (i) its per-arch predictions were frozen at
card-freeze, before construction; (ii) T1/T2 + skeptic passed; (iii) the § 8
discriminability STOP-gate passed. The grid + blind verdict then run exactly
as a stage-6 (the stage-6-grounded-eval precedent: autonomous through grids,
review AFTER, on the whole artifact). A failed gate ⇒ no grid, verdict
recorded, still in-session. **Review-before-grid applies only where a step
was not covered by the frozen card** — when in doubt, stop. What is NEVER
in-session: proposing cards beyond the frozen set, or changing program
rules/gates. Judgment roles on `claude-fable-5`; numerical verification and
generators are CPU-cheap (runpod or local). Budget per cycle: $25 cap, spend
logged to `freqbench/results/spend.json` (+ `spend_log.jsonl`) — NOT the
expansion loop's meter. Briefing template: the expansion briefings, with this
file as the governing protocol.

## Relation to the acid test (revamp phase 4)

Every graduated theorem-first bench adds one held-out-prediction row: hide
its B×A outcome, predict the arch ranking from its coordinates + proofs
alone. The loop is *working* when those predictions hold; a miss is a defect
in the coordinate system and goes back into it (that is the point of having
two generators on one substrate).
