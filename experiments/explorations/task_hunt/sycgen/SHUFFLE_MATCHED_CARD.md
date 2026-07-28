# SHUFFLE ABLATION, SPARSITY-MATCHED — pre-registration

**Frozen 2026-07-29 00:4x BST, BEFORE any cell ran.** Commit-then-run:
this file is committed first, and the git history is the receipt that
every claim below predates its number. Owner mac-d (executor), mac-c
(pre-reg audit). Brief: `briefings/sycgen-shuffle-sparsity-matched.md`.

Prime directive unchanged: **a sound verdict, never a win.**

---

## 0. Why this lane exists

The two sycgen exhibits rest on **different comparisons and nothing
crosses them**:

- `fig_sycgen_shuffle_tsweep.*` — ordered vs shuffled, against a
  **per-token anchor + untrained twin**. That per-token framing is the
  one Dmitry's challenge undermined.
- `tab_sycgen_budget_matched.md` — TXC vs pooled/stacked at matched
  budget, with **no shuffle dimension at all** (`frontier.json` carries
  no shuffle key — verified).

**Unasked question:** does the ordered−shuffled gap survive when the
comparator is a **sparsity-matched SAE** rather than a per-token probe?

## 1. ⚑ The trap, and why pooled is NOT the comparator

**Mean-pooling per-token codes over a window is permutation-invariant.**
Shuffling positions inside the window cannot change the mean. Verified
before the brief was written:

    pooled  ordered vs shuffled : IDENTICAL   max|diff| 5.96e-08
    stacked ordered vs shuffled : DIFFERENT   max|diff| 4.12

So **pooled's gap is exactly zero, always, on any data, for any model.**

1. **Pooled is an INSTRUMENT CHECK, not a comparator.** Its gap must be
   0. **A non-zero pooled gap VOIDS the run** — checked first, abort on
   failure.
2. **"TXC beats pooled's shuffle gap" would be a TAUTOLOGY** — beating
   zero by construction. Not reported, and the table must not imply it.
3. **Stacked is the real baseline.** Concatenation is position-dependent,
   so stacked gets order information **free from its architecture, with
   no temporal learning at all.**

### ⇒ THE CLAIM UNDER TEST

> **Does TXC's ordered−shuffled gap exceed STACKED's, at matched
> measured budget?**

Not pooled's. Anyone reporting against pooled has measured an identity.

## 2. Pre-registered outcomes — fixed before numbers exist

- **(a) TXC gap > stacked gap** ⇒ the windowed architecture uses order
  **beyond what concatenation supplies**.
- **(b) TXC gap ≈ stacked gap** ⇒ the gap is **architectural
  position-sensitivity, not learned temporal structure.** **This is the
  LIVE hypothesis** — sycgen's original shuffle claim already dissolved
  once under exactly this pressure, with **untrained twins showing
  LARGER gaps than trained models**.
- **(c) TXC gap < stacked gap** ⇒ reported as a **negative**.
- **(d) INDISTINGUISHABLE at n=3** ⇒ a distinct outcome. **Not a win,
  not a loss.**

**Decision rule for (d), fixed now** — same three-state rule as item 6,
and the same disclosure: it is a **noise-band heuristic, not a
significance test at n=3**.

    delta = gap_txc - gap_stacked            (both at matched budget)
    noise = max(sd_txc, sd_stacked)          (across the 3 seeds)
    |delta| <= noise            -> (d) INDISTINGUISHABLE
    delta  >  noise             -> (a) TXC ABOVE
    delta  < -noise             -> (c) TXC BELOW

**If (b)/(d) fires, that is the result and it is published as one.**

## 3. Binding controls

- **Untrained twins are MANDATORY, not optional.** They are what killed
  the original claim. **A trained-only gap is not evidence.** Every arm
  gets a random-init twin at the same (T, seed).
- **Pooled's zero is the gate.** Checked before any verdict is computed.
- **Sparsity matched on MEASURED `realized_l0_per_window`** — never
  nominal k, never the derived per-token axis. (The per-token axis is
  `l0_per_window / T`; dividing by T is what manufactured the
  "recovery rises as budget falls" artifact that was retracted.)
- **`l0_unit` reported per arm.** TXC = `nonzeros_in_tile_code`,
  pooled = `union_over_positions`, stacked = `sum_over_positions`.
- **Stacked's `T·d_sae` probe-capacity advantage is disclosed and NEVER
  netted out** — and it is **uninformative at T≥8** (32768 features vs
  1024 windows). Item 6 refused stacked's formal 4/4 win on exactly
  this ground; that refusal stands here.

## 4. Instrument — inherited verbatim, and one flagged judgment call

**PRIMARY: the FIXED probe**, `shuffle_overlay.py:62-119` unchanged —
fit `LinearRegression` on **ordered train** codes, then score **that
same probe, never refit**, on **shuffled eval** codes, both scored
against the **ORIGINAL** targets `t_ev`. Shuffle is
`shuffle_within_window(tiles, T=T, seed=SHUF_EVAL_SEED=0)`, per-row
permutation of the T positions pre-encode. T=1 is identity by
construction.

Inherited **because the brief's purpose is to make the two exhibits
cross**. A second instrument would cross nothing.

**FLAGGED — what a fixed probe measures for STACKED.** Shuffling moves
a token's features to a different `p*d_sae+f` slot, so the fixed probe
breaks partly from **slot scrambling**. This is defensible as the
order-sensitivity under test — stacked genuinely encodes position, and
destroying it genuinely breaks the readout — which is why it stays
primary. But it conflates *"the code moved"* with *"the information is
gone"*, and it **inflates the baseline AGAINST TXC's claim**.

**SECONDARY (pre-registered now, marked secondary now): a refit-probe
column** — fit on shuffled, score on shuffled — as the **disambiguator
for outcome (b)**. Declared before any number exists specifically so it
**cannot be promoted to primary after the fact** if the primary
disappoints. If primary and secondary disagree, **both are reported**
and the disagreement is the finding.

## 5. Grid

    T      {2, 4, 8, 16}
    seeds  {1, 2, 42}
    arms   {txc, pooled, stacked}
    order  {ordered, shuffled}
    k      {1, 2, 4, 8, 16, 32}      (SAE arms only; TXC's k is trained)
    weights{trained, untrained-twin}

Trained arms load the 15 existing checkpoints. **No training in this
lane.**

## 6. Second instrument check, free from the design

Pooled's l0 (**union** over positions) and stacked's (**sum** over
positions) are both **symmetric functions of position**, so realized
budget is predicted **permutation-invariant** for both SAE arms.

That is a prediction, so it is **MEASURED, not asserted** — reported
beside pooled's zero-gap gate. **A non-invariant SAE l0 means the
shuffle is not doing what this card says it does.**

TXC's l0 may legitimately move under shuffle (its encoder is not
permutation-equivariant). **Matched budget is therefore defined on the
ORDERED operating point**, with the shuffled l0 reported alongside.

## 7. Provenance — disclosed up front

- The 15 checkpoints are **real and were located on HF**
  (`han1823123123/temp-bench-data`, `ckpts/<train_key>/`), after a
  first search across 5 **model** repos returned a confident **0 of
  15**. `repo_type` was part of the search space, not a constant.
- The **activation cache was NOT mirrored** and is rebuilt here from
  the committed grid (`elicit_sycgen_screen_llama31.npz`, 943,574
  tokens / 400 docs → N=7239 × 128 × 4096, 7.59 GB fp16) by one
  llama-3.1-8B forward pass. **It is a REBUILD, not pod-D's original**,
  exactly as the item-6 cells were — the same disclosure carries.
- Anchor sanity: the retrained SAE landing near the recorded ~0.4819 is
  the evidence the rebuilt cache is sound. **A leaderboard read is not
  that evidence** (`cache t=True` means a row exists, not that weights
  do).

## 8. Acceptance gate

`figs_writeup/tab_sycgen_shuffle_matched.md` + this card carrying:
the pre-registration above (predating every number, by git history),
the **pooled-zero instrument receipt**, the **untrained-twin columns**,
realized `l0_per_window` per cell, and a **plain-words verdict naming
which of (a)-(d) fired**.

**If the answer is (b) — architectural, not learned — that is the
result and we publish it.**
