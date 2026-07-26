---
status: active
created: 2026-07-26 ~16:30 London
for: mac-b (executor) — salvage W2: novelty cross-ratification (txcwin bridge)
read-first: briefings/salvage-shared.md
---

# Salvage W2 — bring the txcwin trailing-novelty result under task-hunt controls

**Why.** Andrii's parallel thread (`experiments/explorations/txcwin/`,
claims.jsonl c1–c3) reports TXC-post at T = 8 beating per-token SAE,
T-SAE, AND Stacked-at-same-T at matched budget on the **trailing
novelty rate**, replicated on the paper's 8B subject model — a
surface-quiet task (novelty depends on the whole prefix, not
window-visible cues) with reviewer-bbby's Stacked isolation built
in. Our own novelty verdict was withdrawn (scoring error) and never
re-screened. If their result survives OUR controls, it is a
case-study-grade T-scaling task with none of dq's surface-reading
fragility. You are NOT re-running their science — you are auditing
and, where a control is absent, adding it.

**The work, in order:**
1. **Audit** (read-only): map their pipeline (`sweep.py`, `audit.py`,
   `rawgate.py`, `claims.jsonl`, results/) — recompute c1–c4 + r1
   from their committed artifacts; write down exactly which of OUR
   standard controls their design already has (their r1 retraction
   shows a raw-probe floor check exists; find whether untrained-arch
   arms, seed counts, budget-matching receipts, and any
   doc/prefix-identity control exist).
2. **Gap-fill** (compute only what is missing; freeze a mini-card
   first): the likely gaps are (a) **untrained controls** — eval-only
   for untrained dictionaries (no training needed) on their eval
   grid; (b) a **visible-cue baseline** for novelty — pre-register
   it as window-local repetition features (e.g., within-window
   token-overlap counts) so "the window sees repeats" has a measured
   floor; (c) seed spread if they ran single-seed. Est ≤ $15 of your
   $60 cap. If their harness resists quick reuse, REPORT the audit
   and stop — a clean audit memo beats a rushed rerun.
3. **Memo** (the deliverable): `txcwin/CROSSRATIFY.md` — claim-by-
   claim: reproduced? which controls pass? which are absent and what
   did gap-fill show? Verdict vocabulary: SUPPORTED / SUPPORTED-WITH-
   GAPS (named) / NOT-REPRODUCED per claim. Receipts proposals for
   anything quotable. Everything PENDING TEAM REVIEW **and pending
   Andrii's own review** — flag, never override, their thread's
   conclusions; disagreements are listed side by side.

**Coordination:** Andrii is a human collaborator pushing to the same
branch — if commits from them land mid-work, rebase and reconcile;
never modify files under `txcwin/` except ADDING `CROSSRATIFY.md` +
new result files under a clearly-named subdir (`crossratify/`).
