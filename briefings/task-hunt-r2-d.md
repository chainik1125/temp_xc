---
status: active
created: 2026-07-24
for: runpod-d
venue: runpod (H100)
---

# Task hunt round 2, arm A (runpod-d) — the budget-matched TXC-post re-run

Round 1 is REVIEWED & APPROVED (verdicts + binding review notes:
`experiments/explorations/task_hunt/LOG.md`; methods `RECORD.md`).
This round sharpens the Stage-2 positive before the **Sunday
2026-07-26 10:00 PT check-in**; results wanted by **Saturday morning
PT**. Prime directive unchanged: a sound verdict, never a win. Frozen
amendment cards before every run; append to the shared LOG; canonical
runner + 0-dup-key hygiene; no reviewer/meeting quotes in tracked
files.

**New hunt conventions (adopted at review, bind all future screens):**
(1) **per-token-first triage** — before any window grid, run the
per-token linear probe alone; a high per-token ceiling on the primary
target is presumptively converted, escalate only with a card-stated
reason; (2) **the depth sweep as the WHY-diagnostic** (your cand-3
addendum) when per-token is high.

## 1. The budget-matched TXC-post re-run (highest-leverage cheap run in the program)

Round 1's post cells collapsed to realized l0 = 3.4 → 0.49 as T grew
(the post-squash `k_win // T` correction), so its monotone rise to
0.255 at T = 16 is budget-confounded. Freeze a short amendment card
(target realized l0 ≈ 7–8 at every T — raise nominal k accordingly;
state the per-T nominal k you compute), then run post ×
T ∈ {2, 4, 8, 16} × seeds {1, 2, 42} + untrained (~24 cells) on
`ward_real_lambda_base_l12`. Two pre-registered readings: (a) the rise
survives matching ⇒ the money plot upgrades from TXC-pre-peaks-at-8 to
a monotone matched-budget line through T = 16 — a materially stronger
rebuttal figure; (b) it does not ⇒ the 0.255 was sparsity-starvation
behavior, recorded, and TXC-pre remains the headline. Either way the
panel gains its missing cell.

**Seed top-up:** runpod-b may post a LOG recommendation to append ~12
cheap seed cells (pre + tsae at T ∈ {4, 8}) from its variance
receipts (`briefings/hunt-support-stats.md` item 1) — treat it as part
of this run if it lands before you finish.

## 2. Figure hygiene (review note 3, mandatory)

The variance-aware renderer upgrade (l0 legend annotation + seed-CI
whiskers) is OWNED BY runpod-b (`hunt-support-stats.md` item 2) — you
re-render once your budget-matched cells land. If b's renderer has not
merged when your cells finish, do the minimal l0 annotation yourself
rather than wait (never idle); reconcile in the LOG.

## 3. After items 1–2: batch-screen candidate-factory bundles (QUANTITY MODE, Han 2026-07-24 evening)

The factory batches are SHIPPED and REVIEWED — the full queue, the
recommended order, and the binding screen qualifications (punctint
list-density is CONDITIONAL; evidence-ceiling lines must print next
to every window number) live in the mac-local "REVIEW: candidate
factories" LOG entry. Each bundle = labels + manifests + null +
CARD_DRAFT in the
frozen `problib` format, screening on YOUR EXISTING Ward caches in
minutes per candidate. As bundles land: freeze the card (sharpen the
draft), run the Stage-1 screen, one LOG verdict paragraph each —
KEEP/KILL, fail fast, as many as the clock allows. Survivors queue
for Stage-2 by mac-local decision, not automatically.

## Parked (do NOT run)

Proof-op Stage-2 on distill L12 — the raw contrast (+0.017…+0.042) is
too thin to clear a trained panel by Saturday; post-rebuttal. Also
parked program-wide: gpt2-scale order cell. (The anti-conversion
`tss` screen is UN-parked under quantity mode — its bundle arrives
via the broad factory; screen it like any other bundle.)

## 4. RECORD CORRECTION owed (from the 2026-07-24 mac-local review — do this first, it is 10 minutes)

Your Stage-2 amendment is **APPROVED**, with one pre-registration duty
un-discharged. Card § 3 says: *any trained cell outside **[5.0, 8.0]**
is recorded as a residual mismatch and carried into the reading, not
smoothed over.* Four of twelve trained matched cells are ABOVE 8.0 —
T8 all three seeds (8.121 / 8.080 / 8.060; cell mean 8.087) and T16
seed 42 (8.009) — but your LOG entry says "inside the pre-registered
[5.0,8.0] band" and RECORD § 3c says "(in-band)". **Amend both spots**
to the card's own language: a residual mismatch up to +1.5 % over the
panel budget, concentrated at T8. State the consequence, which is
favourable to you: at T8 matched post held MORE budget than TXC-pre
(8.09 vs 7.79) and still recovered less (0.144 vs 0.206), so the
surplus cannot explain post's failure to rise — the verdict is
unchanged and the mismatch is conservative. **No re-run.** Also, when
quoting the r² ranges, say "cell means" (per-seed held-out spread is
wider: −2.61…−0.33).

**Do NOT re-run any panel for the probe-capacity question.** The
λ-readout decision rule is pre-registered in the LOG review entry and
fires on runpod-b's mirror receipt, not on reported lift.

## Acceptance gate — stop for review

Amendment card frozen pre-run; LOG verdict; re-rendered figure +
record addendum; leaderboard hygiene (0 dup keys, no null metrics);
STATUS rewritten. Briefing stays until mac-local review.
