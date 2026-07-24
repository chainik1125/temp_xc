# Working state — agent `mac-local`

**Last rewrite:** 2026-07-24 night (round-2 CPU reviews DONE; next =
the probe-capacity METHODS REVIEW). Read with
`private/rebuttal_plan.md` (untracked) and the team transcript.

## Who / where
Local CC on the Mac at `~/research/projects/temp_xc`, branch `arxiv`.
Role: orchestration + review. NEVER commit/quote `private/` content.
Box warning: case-insensitive checkout — on phantom dirt after a pull,
check `git ls-files | tr A-Z a-z | sort | uniq -d`.

## THE SITUATION
Reviews 5/4/1, R3 swing, **deadline 2026-07-27**, check-in **Sunday
2026-07-26 10:00 PT**. QUANTITY MODE. Reviews all APPROVED so far
(binding notes live in LOG): round-1 hunt + dissection; hunt-support;
candidate factories; **round-2 CPU (probe-adequacy + factory r2) —
reviewed tonight**. Split-integrity is CLOSED (receipt reproduced
byte-identically; zero committed numbers affected). The hunt's first
screen KEEP exists (punctint-q, unreviewed interim).

## ⏭ NEXT, in this order
1. **THE METHODS REVIEW (biggest open item):** runpod-d's Stage-2
   amendment (LOG ~1168–1272, RECORD § 3c) + runpod-e's round-2 batch
   (LOG 1273–1481 + f8cdfc67 stacked-lift extension) as ONE thread —
   two panels, one probe-capacity defect. Machinery is READY
   (`lambda_recovery_v2` + `PROBE_V2_SPEC.md`, reviewed; adoption =
   freeze the spec as-is). Decide: adopt v2 for a 192-cell eval-only
   re-run (~3–4 h wall, checkpoints reused, paired v1 columns kept)
   + one-command variance re-base, or decline and keep v1 canonical.
   My verify list for the amendment itself: falsifier (untrained
   matched l0 = 8.000), nw1024/OLS column reproducing leaderboard to
   1e-4, probe_capacity.py vs its pre-registration d9ee5c75,
   leaderboard decomposition for ff3c5618 (24 cells, dup keys),
   `_matched` figure series, e's 84-cell panel hygiene (8700 = 8616 +
   84 claimed), e's self-caught stacked-reshape defect (18507791).
   Rebuttal implication either way: T-shape statements are
   probe-dependent; ordering survives and WIDENS under v2.
2. **Screen-wave review** when runpod-e stops (novelty NEG /
   punctint-q KEEP / punctint-list WEAK KEEP posted; tss + dialevel
   remain; sc_lambda card frozen a541a8b6 — verdict not seen yet).
   With it: pin (or decline) a doc_mean_only_auc threshold — the
   adoption is ratified as disclosure-only for now.
3. **runpod-d seed top-up review** when it stops (9 cells, frozen
   3d954869) — converts pre-vs-T-SAE bounded or not.
4. **runpod-c em-redo review** (Phase A training since b13ca63d).
5. Sunday check-in distillation (mine): headline λ̂ panel + receipts,
   first KEEPs, kill table, probe-capacity story + decision, queue
   state. Keep `private/rebuttal_plan.md` current — do NOT type new
   absolute panel numbers until item 1 resolves.

## LIVE / IDLE
- runpod, runpod-b: IDLE — round-2 briefings retired tonight.
  Candidate next assignments (on request): refmark user-echo row-drop
  variant is NOT needed (0.22 % — screens handle it); possible
  rebuttal-support once item 1 resolves the phrasing.
- runpod-d: seed top-up in flight; then Ward screens (oprate, qrate,
  vslope).
- runpod-e: consuming screen queue (fineweb bundles + sc_lambda);
  stops for review after its § 3 wave.
- runpod-c: em-redo Phase A (no push since freeze).
- Briefings live: em-redo, task-hunt-r2-d, task-hunt-r2-e.

## Standing context
- Rebuttal-quotable: λ̂ Stage-2 rise (exact p=0.0093 on v1 numbers —
  restate from v2 receipts if adopted, never carry over); shuffle
  receipt; dissection § 7 sentence; T-SAE fairness receipt; dip =
  cause-not-established (never "dilution"); split-integrity receipt
  (zero leakage) is now quotable armor.
- Key science: ambience → regimes → subtype rule → T-taxonomy; FIVE
  g(ℓ) shapes (present-then-discarded new); conversion = the killer;
  screen↔panel convention mismatch (e's lesson, unreviewed); refusal
  = D7 DEAD, recurrence port = B7 refmark SHIPPED (binding
  preconditions: within-conv contrast, user-echo rows, under-span).
- Platform note: "byte-identical" reproduction claims are per-platform
  (x86↔ARM last-ulp drift observed, 1e-16 relative — harmless).
- Git: clean, pushed (review commit after this rewrite).
