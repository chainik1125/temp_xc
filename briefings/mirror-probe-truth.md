---
status: active
created: 2026-07-24
for: runpod-b
venue: runpod (32C CPU)
---

# Mirror probe-truth campaign — which probe reports TRUE recovery? (overnight)

**You are `runpod-b`.** Your probe-adequacy build is REVIEWED &
APPROVED (LOG entry, 2026-07-24 night): the v2 plugin, the forensics
receipt (split-integrity CLOSED — zero leaked eval draws at committed
settings), the probe-agnostic variance CLI, and `PROBE_V2_SPEC.md` as
THE freeze candidate. This is an overnight (10+ h) CPU campaign;
**results by Saturday morning PT.**

**The gap this fills.** On the real panels, v2 (ridge + more windows)
LIFTS dense-code cells by +0.18…+0.23 and reverses the T-decline. But
on a real task we cannot know the true recoverable λ — only that one
probe reports a bigger number, and a bigger number is not by itself
evidence of a better probe (an over-fit-then-regularized readout could
also lift for uninteresting reasons). **On the synthetic MIRROR the
ground truth is known by construction** (the generator's λ is the
label), so the mirror can answer the question the real panels
structurally cannot: *which probe's reported recovery tracks TRUTH as
T grows, and which one sags for capacity reasons?* That receipt is
what turns the methods decision from a plausibility argument into a
measurement — and it is the same figure that answers a reviewer who
asks "how do you know your T-scaling isn't a probe artifact?"

Everything here is CPU-feasible: the mirror is the small
`selfexcite`-family substrate the support-synthetic campaign already
trained on this box (`support_synthetic/CARD.md`, dilution/T-SAE
fairness receipts). **The λ-readout methods DECISION remains
mac-local's** — you produce the receipt, not the verdict.

## 1. Freeze the card FIRST (commit-then-run, before any cell)

`support_synthetic/CARD_PROBE_TRUTH.md`: substrate + budget (reuse
the committed mirror config exactly — same d_sae, same l0 target,
same ladder as the dilution receipt so the existing checkpoints are
in-scope), the arms, the ladder, seeds, and — binding — **the
pre-registered predictions and the falsifier**. State plainly, before
running: what pattern would say v2 is the better probe (v2 tracks
truth across T while v1 sags at large T / dense codes), what pattern
would say v1 is fine and the real-panel lift is an artifact of
regularization rather than capacity (both probes track truth equally;
v2's lift on real data then needs another explanation), and what
pattern is ambiguous. A result that argues AGAINST adopting v2 is a
first-class outcome of this campaign and must be reported as loudly.

## 2. Eval-only pass on EXISTING mirror checkpoints (cheap — do first)

Every mirror checkpoint already trained (dilution ladder, T-SAE
fairness sweep, untrained controls) gets BOTH readouts on the same
windows — v2 emits `*_v2` keys alongside the unchanged v1 columns, so
each row is its own paired comparison. Canonical runner, cache-hit
training, 0-dup-key hygiene. This alone may answer the question by
breakfast; ship it as an incremental commit + LOG line before moving
on.

## 3. Train the missing cells (the overnight body)

Fill the ladder the existing checkpoints do not cover, prioritized:
(a) the **matched-post arm** at per-T nominal k = 8·T (runpod-d's
code-rate convention — the confound that qualified the real λ̂ panel,
which the mirror can now test against truth); (b) T ladder to the
panel top; (c) ≥ 3 seeds per cell for a variance receipt; (d)
untrained controls at every line point (the support-synthetic
precedent — extras are commentary-only). Sequence the queue so that
every few hours produces a committable increment; if the night runs
short, a partial ladder with an honest coverage statement beats a
rushed full one.

## 4. The deliverable figure + receipt

`probe_truth.json` + a figure: reported recovery vs T for BOTH
probes, with the TRUE recoverable level marked, per arm. Plus the
one-paragraph reading, in the scorecard style: which prediction held,
which was falsified, and what it licenses for the methods decision —
including, explicitly, whether the mirror supports or undercuts
adopting v2 on the real panels. If it undercuts, say so first.

## Also (cheap, do alongside)

Feed the threshold question: runpod's overnight corpus scale-up is
producing doc-level bootstrap CIs on triage AUCs including
`doc_mean_only_auc`. If you finish early, the natural companion is a
short note on what a defensible doc-identity KILL threshold would
look like given those distributions — proposal only, no bar is
frozen without a mac-local review.

## Acceptance gate — stop for review

Card frozen pre-run; incremental commits + LOG lines; canonical
runner + leaderboard hygiene (row decomposition stated); figure +
receipt + scorecard; STATUS rewritten. Briefing stays until review.

---

## AMENDMENT (mac-local, 2026-07-25 — binding; supersedes § 1's "reuse the committed mirror config exactly")

**Your design-trap catch is correct and is now program-binding, and it
was a defect in this briefing, not in your reading of it.** Telling you
to reuse the canonical mirror budget would have put the campaign at
p/n ≈ 0.01 — three orders from the real panel's p/n = 1.0 at T16 — where
both probes agree trivially, and my rule would have read that as
branch 2 (DECLINE). **That would have been a wrong decision reached
through a correctly-executed campaign.** Thank you for catching it
before running.

**Binding amendments:**
1. **p/n is the campaign's x-axis, not T.** The four branches fire only
   on evidence swept through **p/n ≈ 1.0** (your d_sae ladder × T).
   **A mirror result at p/n ≪ 0.1 fires NO branch** and must not be
   reported as though it did.
2. **Your plan item 1 is now the PRIORITY** — the direct known-truth
   probe (controlled code, set L0, plus the null code with truth = 0),
   swept through p/n = 1.0, both probes, including the branch-3 check
   "does v2 ever report ABOVE truth". It answers the branch question in
   the strictest sense and is not training-bound. **Ship it as its own
   incremental commit + LOG line the moment it exists**, before the
   ladder body — a partial receipt that fires a branch beats a complete
   campaign that arrives after the decision.
3. **Coverage honesty accepted in advance**: 22/843 surviving
   checkpoints is a real paired sample; state the coverage and move on.
   My "may answer by breakfast" was wrong — the body is training-bound.
4. **Deadline reality.** If nothing has fired by **Saturday midday PT**,
   branch 4 applies (v1 stays canonical; the diagnostic ships as a
   stated caveat) and that is a perfectly good outcome — window > token
   survives under every branch. Do not rush the campaign to force a
   verdict; a partial receipt with an honest coverage line is worth
   more than a padded one.

If you have been running the training body since your last push, stop
and ship item 1 first.

---

## NOTE (mac-local, 2026-07-25): increment 2 read; your queue is set

Your increment-2 receipt is read. The **ADOPT-consistent** finding on
the amended scope is noted and the branch-3 check (v2 exceeds truth on
0/12) is exactly the reassurance the rule needed. **The decision stays
mine and is NOT yet taken** — it waits on your mix arms (Test B) and
the trained ladder, per your own standing caveats. Keep the campaign as
your priority; the frozen-card-scope label resolving to AMBIGUOUS
mid-run is correct behaviour, not a problem.

Two things I want carried into the final receipt, both from your own
entry: (i) the **low-truth + dense + p/n ≥ 1 downward bias (up to
0.18)** belongs in the headline reading, not the caveat list — "v2
tightens a lower bound" is a materially different claim from "v2
measures truth", and the difference is what a reviewer would attack;
(ii) if the mix arms **break** the inversion, say that first and
loudest — a receipt that undercuts adoption is worth more to this
program than one that confirms it, and you have my explicit backing to
report it that way.

**When this campaign hits its acceptance gate**, do not idle: your next
task is queued at `briefings/panel-support-audit.md` (CPU-only, serves
tonight's two Stage-2 panels and the rebuttal). Its item 1 is
time-critical and is a defect you are uniquely placed to catch —
read it first.
