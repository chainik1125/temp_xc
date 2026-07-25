---
status: active
created: 2026-07-25
for: runpod-e
venue: runpod (H100)
---

# Stage 2 — `punctint` q (primary) / `tss` (secondary): CASE STUDY #3, on a different corpus

**You are `runpod-e`.** Your round-2 briefing is discharged and retired.
Your screen wave and — especially — your two self-caught verdict
withdrawals are REVIEWED & APPROVED (LOG, 2026-07-25); your "best
window" finding is now **program-binding convention**. **Results wanted
by Sunday morning PT** (check-in Sunday 10:00 PT; deadline 07-27).

**Why this run.** One confirmed TXC case study exists (λ̂ backtracking,
Ward). runpod-d is panelling `oprate` on the same Ward substrate. Your
job is the **breadth axis**: a real-task panel on a DIFFERENT corpus
(fineweb) and across THREE models — that combination is what answers
"does this generalise beyond one corpus and one model?", which no
amount of Ward evidence can.

## Primary: `punctint` q (question-rate intensity, fineweb)

The hunt's only **unconditional** screen KEEP, and its quoted margins
are **lower bounds** (you scored it on the MEAN arm; re-quote on the
corrected matched-class grid as part of this run).

**Secondary, only if the primary completes:** `tss` — theoretically the
most interesting candidate we have (anti-conversion by construction) and
now KEEP-PENDING-REVIEW. A complete primary beats two partials.

## The bet this panel is actually testing — state it in the card

Your own finding: on fineweb the window advantage lives in the
**NONLINEAR** readout (MLP on window +0.06…+0.13) while a linear
mean-pool sees ≈ +0.04. On Ward it is the opposite (`g_agg ≈ g`,
linear). **This is a reason the panel is interesting, not a reason it
fails:** a TXC encoder is itself nonlinear — sparse coding over the
window — and the λ-probe is linear only on top of that code. So the
question this panel answers is precisely *does a sparse window code
capture what the MLP found?* Pre-register both outcomes as informative:
if TXC recovers the nonlinear gain, that is a strong, novel claim; if it
does not, we have learned the class of window advantage a sparse
dictionary cannot represent — also publishable, also worth the GPU.

## Bindings (non-negotiable, all previously paid for)

1. **The document-identity control is BINDING at panel, not optional.**
   punctint q measures `doc_mean_only_auc` = 0.901 — the naive panel is
   uninterpretable without it. Use within-document contrasts or
   document-matched sampling, and run a document-identity floor probe
   alongside. (`tss`, if reached, is the low end at 0.664–0.670 — still
   report it.) This is the qualification that foreclosed dialevel; do
   not repeat that.
2. **Budget-match on REALIZED `l0_per_token`**; TXC-post at per-T
   nominal k = 8·T. Pre-register the band; out-of-band cells are
   recorded as residual mismatches, not smoothed over.
3. **Carry BOTH probe columns** — `c["eval_extra"] = {"lambda_probe_v2":
   True, "lambda_v2_probe": "ridge", "lambda_v2_n_windows": 8192,
   "lambda_v2_split": "trace"}` (`PROBE_V2_SPEC.md` § 2). Claim on v1
   (leaderboard-canonical until the methods rule fires); report both;
   the panel then survives the decision either way.
4. **No max-over-arms scoring** — your own retired convention. Fix the
   probe class, control width, use the foreign-context nulls.
5. **Print the visible-evidence line** next to every window number.
6. **Quote the training corpus size beside any unigram/triage number**
   (runpod's estimator finding: 400-doc readings are understatements).
   A scaled 4,000-doc punctint artifact EXISTS (`build_punctint4k`,
   committed by runpod) — prefer it where it drops in cleanly, and say
   which corpus each number came from.
7. Canonical runner; 0 dup eval_keys / 0 null metrics; decomposition
   stated; variance receipts via the probe-agnostic
   `support_stats/stage2_variance.py` (per-seed CIs, trend permutation
   p, paired margins, and an explicit statement of what is NOT bounded
   at n = 3).

## Pre-register before any cell

Frozen card: predicted T-pattern per model, KEEP/KILL clauses, the
falsifier, the realized-l0 band, and the nonlinear-vs-linear bet above.
Three models means three verdicts — **do not pool them into one
headline**; per-model outcomes with a stated majority rule.

## Acceptance gate — stop for review

Card frozen pre-run; panel complete; LOG verdict + scorecard; document-
identity control reported alongside every gap; variance receipts;
figure; RECORD_B section; hygiene; STATUS rewritten. Briefing stays
until mac-local review.

---

## ADDENDUM (mac-local, 2026-07-25): your order counterexample is UPHELD

Your flag on the adopted "order does not matter, **anywhere**" is
correct and the quantifier is **withdrawn** — the overstatement was
mine, not runpod-d's (d scoped its claim to Ward; I generalised it).
The amended program wording is in the LOG; use that wording, never
"anywhere". Your reconciliation hypothesis (recency / distance-to-
anchor rather than sequence order) is recorded as the best explanation
on offer.

**Your proposed cheap test is OPTIONAL and explicitly BEHIND this
panel.** Run it only as a pre-flight if it genuinely costs under an
hour on the existing dialevel caches. The amended wording is already
defensible with zero new measurement, so the test upgrades a hedge to
a measurement — it does not rescue a claim, and nothing breaks if it
goes unrun before the deadline. **The panel is worth more to the
rebuttal.**

Raising it the way you did — "flagging, not disputing", with a
testable hypothesis and a costed experiment attached — is the model
for how to challenge an adopted finding. Keep doing exactly that.
