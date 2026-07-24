# Working state — agent `mac-local`

**Last rewrite:** 2026-07-24 late (factory review DONE; next item =
runpod-d Stage-2 AMENDMENT review). Read with
`private/rebuttal_plan.md` (untracked) and the team transcript.

## Who / where
Local CC on the Mac at `~/research/projects/temp_xc`, branch `arxiv`.
Role: orchestration + review. NEVER commit/quote `private/` content.
Box warning: case-insensitive checkout — on phantom dirt after a pull,
check `git ls-files | tr A-Z a-z | sort | uniq -d` (CARD.md/card.md
already bit once; fixed via PREP_DRAFT.md rename + index surgery).

## THE SITUATION
Reviews 5/4/1, R3 swing, **deadline 2026-07-27**, check-in **Sunday
2026-07-26 10:00 PT**. PROGRAM MODE = **QUANTITY** (Han). Reviews so
far, all APPROVED with binding notes in LOG: round-1 hunt +
dissection; hunt-support (dip RETRACTED to cause-not-established,
T-SAE fairness closed, variance receipts partition claims); **candidate
factories (runpod-b traces + runpod broad) — reviewed this evening,
both approved, briefings retired, screen queue OPEN** (order + binding
qualifications in the LOG factory-review entry: punctint list is
CONDITIONAL, dialevel precondition binding, evidence-ceiling lines
must print at screen, cross-factory bar mismatch noted, sc_lambda
"17-pattern" prose slip = 16 actually).

## ⏭ NEXT, in this order
1. **runpod-d Stage-2 AMENDMENT review (LOG ~1168–1272 + RECORD § 3c)**
   — I have now READ the entry but NOT reviewed it. Substance: matched
   post inverts (0.185/0.202/0.144/0.137, peak T4), reading (b)
   confirmed-refined (0.255 = sparse code dodging the probe penalty,
   +0.032 lift only), reading (c) = **panel-wide probe artifact**:
   `lambda_recovery`'s OLS at n≈p (T16: n=2048=p) suppresses DENSE
   codes; ridge/nw8192 lifts pre T16 0.138→0.351, and window > token
   WIDENS (pre 0.351 vs tsae 0.211). Verify: falsifier (untrained
   matched l0=8.000), the nw1024/OLS column reproducing leaderboard to
   1e-4, probe_capacity.py pre-registration (d9ee5c75) vs what ran,
   leaderboard row decomposition for ff3c5618 (24 cells, 0 dup keys),
   figure `_matched` series separation, and the flagged METHODS
   decision (adopt capacity-adequate probe? — re-bases b's variance
   receipts; decision is MINE to take or defer, with runpod-b).
   n=3 honesty note at the entry's end is good — check CI numbers.
2. **My split-integrity check** (~30 min): does `_train_lambda_probe`'s
   n//2 sequence split put rows of one Ward TRACE in both halves?
   Now DOUBLY relevant: it interacts with the probe-capacity finding
   (same eval convention). Do it as part of item 1.
3. **Seed top-up review** when runpod-d stops (9 cells, frozen
   3d954869, in flight at last STATUS) — converts pre-vs-T-SAE from
   consistent to bounded (or not).
4. **runpod-e hedging Stage-2 review** when it stops (card fff7877c +
   reconciled 606a8015; then its § 3 bundle screens).
5. **Screen-queue supervision**: d/e batch-screen the 9 factory labels
   per r2 § 3 + my recommended order in the LOG factory-review entry;
   claim-lines prevent double-screens. Kills are fine and fast; any
   PASS gets a frozen card before Stage-2.
6. **runpod-c em-redo review** (Phase A training since freeze
   b13ca63d, no push yet).
7. Keep `private/rebuttal_plan.md` current (variance-receipt phrasing,
   dip retraction, fairness receipt; ADD next: the probe-capacity
   framing once item 1's review lands — it likely CHANGES how we quote
   absolute panel levels in the rebuttal: T-shape is probe-dependent,
   ordering survives).

## LIVE / IDLE
- runpod-d: seed top-up in flight; then bundle screens.
- runpod-e: hedging-LEVEL Stage-2; then bundle screens.
- runpod-c: em-redo Phase A training (no push since freeze).
- runpod, runpod-b: IDLE — factory batches approved, briefings
  retired. Next assignment on request (candidates: more ledger BUILDs
  B6/OpenWebMath, or rebuttal-support once probe-capacity review
  settles the phrasing).
- Briefings live: em-redo, task-hunt-r2-d, task-hunt-r2-e.

## Standing context
- Rebuttal-quotable: λ̂ Stage-2 rise (exact p=0.0093; pre-vs-T-SAE
  pending top-up; dip = cause-not-established, never "dilution");
  shuffle receipt; dissection § 7 sentence; T-SAE fairness receipt;
  STORY.md regime/param material. CAUTION: absolute panel levels +
  T-shape now carry the (unreviewed) probe-capacity flag — do not
  type new absolute numbers into the rebuttal until item 1 is done.
  Reviewer-facing text stays in private/.
- Key science: ambience → regimes → subtype rule → T-taxonomy; FOUR
  g(ℓ) shapes (incl. built-and-immediately-linearized); conversion =
  the hunt's recurring killer; (task, MODEL) locality of non-ambience.
- Git: at cedbb6d0 + this review commit; clean, pushed.
