# Working state — agent `mac-local`

**Last rewrite:** 2026-07-24, after the round-1 hunt + dissection
review. Read with `private/rebuttal_plan.md` (untracked) and
`private/transcripts/transcript-2026-07-24.txt`.

## Who / where
Local CC on the Mac at `~/research/projects/temp_xc`, branch `arxiv`.
Role: orchestration + review. NEVER commit/quote `private/` content.
Warning for this box: the repo is checked out case-insensitively — a
CARD.md/card.md collision already bit once (fixed by renaming to
PREP_DRAFT.md); if phantom dirt appears after a pull, check
`git ls-files | tr A-Z a-z | sort | uniq -d` before suspecting agents.

## THE SITUATION (rebuttal week)
Reviews 5 / 4 / 1; R3 is the swing; **deadline 2026-07-27**; team
check-in **Sunday 2026-07-26 10:00 PT**. No PDF updates — only new
results typed into responses.

## DONE — round 1 reviewed & APPROVED (2026-07-24)
All in `task_hunt/LOG.md` (my review entry at the end) + `RECORD.md` +
`loss_dissection/RECORD.md` (stamps appended). Headlines:
- **λ̂ Stage-2 QUALIFIED POSITIVE** (TXC-pre 0.13→0.19→0.21 over
  T=2/4/8 vs T-SAE 0.154, matched realized l0, real Ward activations)
  — binding phrasing rules in the LOG review entry (per-tile
  code-readout convention; ≈2σ seed margin → variance-aware wording;
  figure needs l0 annotation).
- **Backtracking shuffle receipt POSITIVE** (anticipation order-
  sensitive +0.028…+0.041; ambient label +0.003…+0.013; fixed T=16).
- Proof-op KEEP (distill-L12-only; model axis); forbidden-word KILL
  (pre-registered ambience kill); arm B 3× sound kills (mechanism:
  conversion); dissection: drop TXC-pro, salvage multi-distance
  contrastive (regime-3 power, post decode, T=8) — § 7 sentence
  endorsed.
- Hygiene verified: leaderboard 8,616 = 7,116 + 1,416 + 84; 0 dup
  keys; 220 tests pass.

## LIVE
- **runpod-c** (H100): `briefings/em-redo.md` — Phase A training since
  the 07-23 22:48 freeze; no results pushed yet. Review when it stops.
- **runpod-d's cand-3 depth sweep landed mid-review and is REVIEWED**
  (LOG addendum): kill mechanism corrected to CONVERSION (not lexical
  circling); fourth g(ℓ) shape (built-and-immediately-linearized);
  depth sweep adopted as the cheap WHY-diagnostic.
- **Round 2 dispatched**: `briefings/task-hunt-r2-d.md` +
  `task-hunt-r2-e.md` (split per pod) — runpod-d:
  budget-matched TXC-post re-run + seed top-up (first-class) + figure
  re-render; runpod-e: hedging-LEVEL Stage-2 (card frozen) +
  early-layer addendum. Then both batch-screen factory bundles.
  New conventions: per-token-first triage + the depth-sweep
  diagnostic. Results wanted Saturday morning PT.
- **QUANTITY MODE (Han, evening)**: CPU pods = candidate factory
  (`candidate-factory-traces.md` runpod-b, `candidate-factory-broad.md`
  runpod — label bundles + triage + draft cards; interleave `tss`
  un-parked). NOTE: rebuttal-pack + stage2-redteam briefings were
  RETRACTED before work started (Han redirect). The red-team's
  sharpest question — Stage-2 split integrity (do train/eval halves
  share Ward traces? `_train_lambda_probe` splits sequences n//2) —
  is now MINE to check during the r2 review (~30 min, from committed
  code + labels).

## ⏭ NEXT ACTIONS (mine)
1. Review em-redo when runpod-c stops (gate-integrity first; the
   frozen predictions are in its TRACKING freeze commit b13ca63d).
2. Review runpod-e's mini study + round-2 sessions as they land.
3. Distill the approved results into `private/rebuttal_plan.md`
   rebuttal text (typeable-now: λ̂ T-scaling figure w/ conventions
   sentence, shuffle receipt, dissection § 7 sentence, regime
   isolation + probing corollary + param table from STORY.md).
4. Sunday check-in prep: one page for the team — what survived, what
   died, what's quotable.

## Standing context
- Key science: ambience/regimes/subtype rule/T-taxonomy (STORY.md);
  three g(ℓ) shapes; conversion mechanism now measured three ways
  (arm B). Memory files current.
- Parked: gpt2 order cell, anti-conversion class, proof-op Stage-2,
  onset head-to-head, synthetic autoresearch resumption.
- Git: clean at last push; briefings retired: task-hunt{,-b,-prep},
  txcpro-dissection, hunt-support-synthetic + hunt-support-stats
  (both reviewed/APPROVED same day — dip interpretation RETRACTED,
  T-SAE fairness closed, variance receipts partition the claims, seed
  top-up now first-class for runpod-d; CPU pods idle). Remaining:
  em-redo, task-hunt-r2-d, task-hunt-r2-e.
