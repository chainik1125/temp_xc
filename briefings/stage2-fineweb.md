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

---

## ADDENDUM 2 (mac-local, 2026-07-25 — scoping + the 12-hour queue)

If you froze a card before reading this, reconcile via a card-amendment
commit before any conflicting cell.

**1. Scope: ONE full panel + replication cells — NOT three full
panels.** A full panel is 84 cells; three of them do not fit in 12 h.
Primary model: **gemma-2-2b** (mid-scale, d_in 2304 keeps the tsae
buffer cost sane, and it is a more representative subject than gpt2) —
unless your first-cell timing argues otherwise; state the choice and
reason in the card. Then the CROSS-MODEL claim comes from replication
cells only: TXC-pre at its best two T values + tsae + per-token SAE
(+ untrained), 3 seeds, on gpt2 and llama31. Per-model verdicts, stated
majority rule, no pooling — as before.

**2. The tsae arm: schedule its 3 trained cells FIRST** (the λ̂ top-up
measured multi-hour buffer-path cells at d_in 4096; gemma's 2304 is
cheaper but not free). Fresh panel ⇒ you MAY freeze a feasible
`buffer_tokens` uniformly across archs in the card (the comparability
bar that blocked this on the λ̂ top-up does not apply to a new
datasource).

**3. Binding 1 made concrete (Stage-2 vocabulary — no runner changes).**
(a) The doc-identity FLOOR = a doc-mean-only predictor's r on the same
eval windows (label-side, cheap) — print it beside every window cell.
(b) The within-document RECEIPT = an out-of-band re-fit of the SAME
codes against doc-demeaned targets, your `probe_capacity.py` pattern
(off-leaderboard, pre-registered in the card). **Pre-register now: the
within-doc face may sit near floor** (q's zero-fraction is 0.806 —
within-doc λ̂ variance is thin). If the panel's gap collapses under
doc-demeaning, that is a sound NEGATIVE and you report it as loudly as
a win; do not soften it.

**4. Datasource plumbing.** The new fineweb datasource must expose
`trace_ids` = document index (copy the `real_lambda.py` pattern) — the
v2 trace split then prevents document rows straddling the probe halves,
which matters MORE here than anywhere (doc-identity 0.901). Binding 6's
"prefer the 4k artifact": train + eval on the existing 400-doc caches
(zero new caching); the 4k corpus is an OPTIONAL eval-side addendum
only if the queue completes (one short caching pass). Quote the corpus
size beside every number either way.

**5. Binding 5 clarified**: the evidence line at Stage-2 is the
regression analog — in-window event count → target on the same windows
— not the bundle's screen AUC ceilings.

**6. The 12-hour queue, in order — stopping early at any gate is fine:**
1. Freeze card → 2. gemma-2-2b full panel (tsae first) → 3. doc-identity
floor + doc-demeaned receipt → 4. variance receipts + LOG verdict +
figure → 5. replication cells on gpt2 + llama31 → 6. `tss` (primary
model ONLY, needs its own ~330k-token caching pass) → 7. the optional
dialevel recency pre-flight — LAST, only if everything above is done.
