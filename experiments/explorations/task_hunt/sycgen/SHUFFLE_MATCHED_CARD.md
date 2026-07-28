# SHUFFLE ABLATION, SPARSITY-MATCHED — pre-registration

**Frozen 2026-07-29 00:4x BST, BEFORE any cell ran.** Commit-then-run:
this file is committed first, and the git history is the receipt that
every claim below predates its number. Owner mac-d (executor), mac-c
(pre-reg audit). Brief: `briefings/sycgen-shuffle-sparsity-matched.md`.

Prime directive unchanged: **a sound verdict, never a win.**

> **AMENDED 2026-07-29 00:46 BST — still BEFORE any cell ran ($0 spent,
> 0 pods, no cache built).** Absorbs the hub's comparator correction
> (`73f8ea388`, brief §2b) and **all four findings of mac-c's
> pre-registration audit** (`a027b7caa`, brief §5), one of which (A1)
> was **blocking**. Amending a pre-registration is only legitimate
> while no number exists; that is the case here, and the git history is
> the receipt. Every amendment is marked **[AMD]** in place and nothing
> original was rewritten or deleted.

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

**[AMD — mac-c A3]** The rule above is a magnitude test only, so
(a)-vs-(d) could still be settled post-hoc by a single lucky seed.
**(a) now requires BOTH conditions, fixed pre-data:**

    (i)  SIGN AGREEMENT — delta > 0 in ALL 3 seeds, per-seed
         (a 3/3 sign test; 2/3 is NOT (a))
    (ii) MARGIN        — |delta| > across-seed SD, as above

**Either condition failing ⇒ (d).** Same principle as the `+0.05` bar
elsewhere: the threshold is a number written before the data, not a
judgement made after it.

**If (b)/(d) fires, that is the result and it is published as one.**

## 2b. [AMD] Matched budget = BRACKET, never single-sided (hub §2b)

**Binding, and it already moved a shipped verdict.** The item-6 table
selected pooled's comparator as *the best point with `l0 ≤ TXC's l0`* —
defensible in words, biased in arithmetic: k is swept on a coarse grid
whose consecutive points differ **40–75%**, so no point lands at TXC's
budget and the rule silently picked a **much cheaper** baseline. At T=2
it compared TXC @ 5.66 against pooled @ 3.51 — **38% less budget** —
and returned a win. **Item 6's headline moved from above 3/4 to above
2/4.** That verdict was mine to produce; the rule was the hub's; it
survived a ratification and a two-implementation cross-check, because
**two implementations of one premise is one check wearing two coats.**

**This lane is bound by the corrected rule, and it applies to the GAP
exactly as it applies to the level:**

- **Bracket both sides.** Report the best point **below** and the
  cheapest point **above** TXC's measured budget, and **interpolate to
  TXC's exact `l0`** (primary). Monotonicity of the arm in budget is a
  **precondition, printed as a receipt** — not assumed.
- **Sweep k finely enough to bracket tightly.** **[AMD]** `K_SWEEP` is
  therefore widened from item 6's `(1,2,4,8,16,32)` to include
  intermediate points — *a tight bracket is worth more pod-minutes than
  another seed*, and here it is nearly free (encode-and-probe, no
  training).
- **State the bracket width** in the results table.
- **If the two ends disagree, that IS the finding — report it, do not
  pick.**
- **Standing check before any comparator verdict ships:** print the
  **budget ratio** of the selected comparator to TXC. **If it is not
  ≈1.0, the word "matched" is not earned.**

## 3. Binding controls

- **Untrained twins are MANDATORY, not optional.** They are what killed
  the original claim. **A trained-only gap is not evidence.** Every arm
  gets a random-init twin at the same (T, seed).

  **[AMD — mac-c A2] The twin is a GATE ON (a), with its own rule
  fixed pre-data.** As originally written, (a)–(d) were defined only on
  *trained* TXC vs *trained* stacked, so "(a) fired" and "untrained ≥
  trained" could **both** be true and point opposite ways — leaving the
  conflict to be settled after seeing the numbers. Binding instead:

      (a) requires, IN ADDITION to 2(i) and 2(ii):
          gap_txc_TRAINED > gap_txc_UNTRAINED   in all 3 seeds

  **If the untrained twin's gap matches or exceeds the trained model's,
  the result is (b) — architectural, not learned — regardless of how
  TXC compares to stacked.** This is not a new hypothesis; it is
  exactly how sycgen's original shuffle claim died, and the rule is
  written here so that outcome cannot be re-narrated as a win.
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

## 4b. [AMD — mac-c A1, BLOCKING] The pooled-zero gate could not fail

**Accepted in full.** §1's gate is `pooled gap == 0`, and
`frontier.py:119` is `z.mean(dim=1)` — a mean over the window axis is
permutation-invariant **arithmetically**. So **pooled's zero survives
any bug in the shuffle whatsoever.** The only defect that could ever
fire this gate is one making the pooled arm position-*sensitive* — a
bug in the comparator, not in the instrument it was written to check.

**And the failure it misses manufactures the pre-committed headline:**
if the shuffle silently no-ops, then pooled = 0 (**gate passes**),
stacked = 0, TXC = 0 ⇒ every gap is zero ⇒ reads as **(b)**, which §2
names the LIVE hypothesis and §8 pre-commits to publishing. **The one
unchecked failure is the one that yields the promised answer through a
passing gate.** Same shape as the tautology §1 removed from the
comparator, one level up — this one was in the gate itself.

Verified at source by mac-c, not assumed: a repo-wide grep for any
assert that `tiles_sh` differs from `tiles_ev` returns **empty**;
`shuffle_overlay.py`'s `IDENTITY_TOL` is a **replication** guard on
`|canonical_r − recomputed_r|`, a different object; and
`shuffle_within_window` validates rank and `x.shape[1] == T` and
**never** that the drawn permutation is non-identity.

**BINDING FIX — on the INPUT side, arm-independent, pre-encoder**, so
it cannot be confused with a result:

    assert (tiles_sh - tiles_ev).abs().max() > 0

**STRENGTHENED BEYOND THE MINIMUM.** "Something changed" is a weak
predicate — it passes if one row of thousands moved. The apparatus has
an *exactly predicted* value (§4c), so the gate checks against **that
number** instead:

    measured_shuffled_row_fraction  ==  1 - 1/T!   (binomial tol)

That is a **positive control with a predicted value**, not a
nonzero-check: it fires on a total no-op (0.00), on a partial
application, and on a wrong-axis permutation, none of which the
minimal assert distinguishes. **A gate and a positive control are
different objects** — tonight has already produced two checks whose
failure looked exactly like success, and this file must not add a
third.

### ⚑ [REVIEW — mac-c, 00:52 BST] The strengthening is right, but `(binomial tol)` must be a number, and at T=8 the obvious choice VOIDs ~1 healthy run in 10

**Accepted in direction.** Checking against the predicted value catches
strictly more than my minimal assert — partial application and
wrong-axis permutation both survive `max() > 0` and both fail this.
Keep it.

**But it converts a deterministic check into a statistical one, and the
regime changes qualitatively across the grid.** The identity-row count
is `Binomial(n, 1/T!)` with `n = n_windows · (L // T)` = `1024 · 32/T`
for the current convention:

    T    n      E[identity rows]   regime
    2    16384  8192               massively statistical
    4    8192   341.3              statistical
    8    4096   0.1016             ~1 run in 10 has >=1 identity row
    16   2048   9.8e-11            genuinely deterministic

**The trap is T=8.** `λ = 0.102`, so `P(≥1 identity row) = 9.66%`. A
gate written as equality — "fraction must equal `1 − 1/T!`", or
equivalently "no unshuffled rows" once `1 − 1/T!` rounds to 1.0000 —
**spuriously VOIDs about one healthy run in ten and reports it as
"instrument broken."** That is the mirror image of A1: A1 was a gate
that cannot fire when it should; this is a gate that fires when it
should not, and the failure mode is a false alarm on a sound run plus
whatever re-runs it triggers.

**FIX — gate the identity-row COUNT against a two-sided binomial band,
uniform across T:**

    T    E[count]   ACCEPT (exact binomial tail, P<1e-4 each side)
    2    8192       7936 .. 8448
    4    341.3      268  .. 414
    8    0.1016     0    .. 3
    16   ~0         0    .. 0      (any identity row IS a bug)

⚑ **One construction at every T — an exact tail probability, NOT a
σ-multiple.** My first table labelled these "4 SD / P<1e-4", which is two
different constructions wearing one header, and a σ band is meaningless
once `λ ~ 0.1` (T=8) or `1e-10` (T=16). The hub hit the same rock
verifying this and said so (`33a5c72d8`): `E ± 4σ` gives **0..2 at T=8
and 0..1 at T=16**, both wrong. Numbers above match the hub's BINDING
values; T=4's lower bound is **268**, floored rather than rounded, so
the band errs toward *accepting* — the right direction when the failure
being designed against is a spurious VOID. **Print observed count, band,
and `n` per cell.**

**⚠ Correction to my own first pass, before it reaches anyone's code:**
I initially computed these bands from a *Poisson* tail. Poisson is only
a good approximation for small `p`, and at T=2 `p = 1/2` — Poisson SD
90.51 vs binomial 64.00, giving 7830..8554 instead of 7936..8448, a band
**~40% too loose**. Gate on `Binomial(n, 1/T!)`, not Poisson. The T=4/8/16
rows are unaffected (there `p` is small and the two agree).

**Two riders:**

1. **The bands are functions of `n`, so recompute them if the lane
   changes `n_windows` or `L`** — they are not constants. State `n` per
   cell in the receipt next to the count.
2. At T=8 and T=16 the gate is near-deterministic, so it assumes **no
   exact ties**: two positions within a tile holding identical vectors
   read as an unmoved row and would void the run. On float32
   residual-stream activations exact ties are effectively impossible, so
   this is a bounded assumption rather than a live risk — but it is the
   thing to check first if T=16 ever voids, before believing the shuffle
   broke.

**Unchanged:** your note that the redraw column's 1.000 fraction is by
construction and is therefore *not* evidence the shuffle works. That is
A1's lesson applied correctly to your own fix, and it is the right call
to gate the two columns by different arguments.

## 4c. [AMD — mac-c A4] The apparatus shuffles fewer rows than it looks

`shuffle_within_window(per_row=True)` draws `torch.randperm(T)`
independently per row, so **a row draws the identity permutation with
probability `1/T!`** and is silently *not shuffled*:

    T = 2  -> 1/2   = 50.0% of rows UNSHUFFLED   (rows truly shuffled 0.500)
    T = 4  -> 1/24  =  4.2%                                        0.958
    T = 8  -> 1/40320                                             ~1.000
    T = 16 -> negligible                                          ~1.000

**Common-mode across arms, so the TXC-vs-stacked contrast at FIXED T is
safe** — that is the primary claim and it survives. But **any "the gap
grows with T" reading inherits `1 − 1/T!` from the apparatus**, which
is the same species of defect as the divide-by-`T` per-token artifact
retracted earlier tonight: a monotone-in-`T` factor contributed by the
instrument and read as a property of the model.

**BINDING:** both of mac-c's sanctioned fixes are run, because the
marginal cost is one extra encode per cell and the disagreement between
them is itself informative:

- **PRIMARY — plain draw**, `shuffle_within_window` verbatim
  (`seed=SHUF_EVAL_SEED=0`). Keeps the instrument commensurable with
  the existing `fig_sycgen_shuffle_tsweep` exhibit, which is the whole
  reason this lane exists.
- **ARTIFACT CONTROL — non-identity redraw**, rejecting and redrawing
  any row that draws the identity, so the shuffled fraction is 1.000
  at every T and the `1 − 1/T!` factor is removed by construction.
- **Cross-T statements may cite ONLY the redraw column.** The plain
  column is reported at every T but is **disclosed as carrying
  `1 − 1/T!`**, and at T=2 that means half its rows are unshuffled.
- **If the two draws disagree, that is reported, not resolved by
  choosing** — same discipline as the §2b bracket.

⚑ **The redraw column's shuffled fraction is 1.000 BY CONSTRUCTION and
is therefore NOT evidence that the shuffle works.** For that column the
§4b gate must instead assert the *plain* draw's predicted rate; the two
columns are checked by different arguments on purpose.

## 5. Grid

    T      {2, 4, 8, 16}
    seeds  {1, 2, 42}
    arms   {txc, pooled, stacked}
    order  {ordered, shuffled}
    draw   {plain, nonidentity-redraw}          [AMD 4c]
    k      {1,2,3,4,6,8,11,12,16,22,24,32,44,48} [AMD 2b — widened]
    weights{trained, untrained-twin}

**[AMD] `k` widened from item 6's `(1,2,4,8,16,32)`.** That grid's
consecutive points differ by 40–75%, which is exactly what let a
single-sided rule call a 38%-cheaper baseline "matched". The added
points are chosen to **straddle TXC's measured budget tightly** at
every T. Cost is negligible here — the SAE arms are post-hoc
transforms, so a k-point is one encode plus one probe fit, with no
training anywhere in this lane.

**The k grid is a BRACKETING instrument, not a result.** Only the two
points adjacent to TXC's budget (and the interpolation between them)
enter the verdict; the rest establish monotonicity, which §2b requires
as a printed precondition rather than an assumption.

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
