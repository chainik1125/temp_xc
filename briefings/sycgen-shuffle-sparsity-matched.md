---
status: active
owner: mac-d (executor) + mac-c (review/pre-reg audit)
issued-by: mac-local (hub)
issued: 2026-07-29 00:34 BST (was stamped 01:2x — ~40 min in the FUTURE; mac-c A5)
priority: TOP — closes the gap Han found in the item-6 exhibit
scale: AUTHORIZED up to 20 simultaneous H100s (Han)
---

# sycgen shuffle ablation, SPARSITY-MATCHED — and the design trap in it

Han: *"we need the proper sycgen shuffle ablation sparsity matched"* —
after noticing the two sycgen exhibits rest on **different
comparisons** and nothing crosses them.

- `fig_sycgen_shuffle_tsweep.*` — ordered vs shuffled, against a
  **per-token anchor + untrained twin**. The framing Dmitry's challenge
  undermined.
- `tab_sycgen_budget_matched.md` — TXC vs pooled/stacked at matched
  budget, **no shuffle dimension at all** (`frontier.json` has no
  shuffle key).

**Nobody has asked: does the ordered−shuffled gap survive when the
comparator is a sparsity-matched SAE rather than a per-token probe?**

---

## 1. ⚑ READ THIS FIRST — pooled CANNOT have a shuffle gap, by construction

**Verified before this brief was written:**

    pooled  ordered vs shuffled : IDENTICAL   max|diff| 5.96e-08 (float noise)
    stacked ordered vs shuffled : DIFFERENT   max|diff| 4.12

**Mean-pooling per-token codes over a window is permutation-invariant.**
Shuffling tokens inside the window cannot change the mean. So pooled's
ordered−shuffled gap is **exactly zero, always, on any data, for any
model.**

**Three consequences, and they define the experiment:**

1. **Pooled is NOT a comparator for this question — it is an INSTRUMENT
   CHECK.** Its gap must come out 0. **If pooled shows a non-zero gap,
   the shuffle instrument is broken and every number in the run is
   void.** This is the single most valuable cell in the design.
2. **Reporting "TXC's shuffle gap beats pooled's" would be a
   TAUTOLOGY dressed as a result** — beating zero by construction. **Do
   not do it, and do not let the table imply it.**
3. **Stacked is the real baseline.** Concatenation is position-dependent,
   so stacked gets order information **for free from its architecture,
   with no temporal learning at all.** That is exactly the null the
   claim has to beat.

**⇒ THE CLAIM UNDER TEST: does TXC's ordered−shuffled gap exceed
STACKED's at matched measured budget?** Not pooled's. Anyone who
reports against pooled has measured a mathematical identity.

## 2. Pre-registration — write into the card BEFORE any cell runs

State, before numbers exist:

- **(a) TXC gap > stacked gap at matched budget** ⇒ the windowed
  architecture uses order beyond what concatenation supplies.
- **(b) TXC gap ≈ stacked gap** ⇒ the gap is **architectural
  position-sensitivity, not learned temporal structure.** This is the
  live hypothesis: **sycgen's original shuffle result already dissolved
  once under exactly this pressure** — untrained twins showed *larger*
  gaps than trained models.
- **(c) TXC gap < stacked gap** ⇒ report as a negative.
- **(d) INDISTINGUISHABLE at n=3** ⇒ a distinct outcome; **not a win,
  not a loss.**

**Binding controls:**

- **Untrained twins are MANDATORY, not optional.** They are what killed
  the original claim. A trained-only gap is not evidence.
- **pooled's zero is the instrument gate** — check it first, abort on
  failure.
- **Sparsity matched on MEASURED `realized_l0_per_window`**, never
  nominal k, never the derived per-token axis.
- Report `l0_unit` per arm; stacked's `T·d_sae` probe-capacity
  advantage disclosed and **never netted out** — and remember it is
  **uninformative at T≥8** (32768 features vs 1024 windows).

## 2b. ⚑ AMENDMENT 01:0x — the budget table's comparator rule was
## biased, and the same trap is waiting for you here

A hub sanity check on the *delivered* budget table found the bug this
run would have repeated. The generator compared TXC against **the best
SAE point with `l0 ≤ TXC's l0`**. Sounds conservative. It is not: k is
swept on a **coarse grid** (1,2,4,8,16,32), consecutive points differ
by **40–75% in budget**, so no point lands at TXC's budget and the rule
silently selects a **much cheaper** baseline. **At T=2 it compared TXC
@ 5.66 against pooled @ 3.51 — 38% less budget — and returned a win.**
The point 5% *above* TXC's budget was indistinguishable. **Headline
went from above 3/4 to above 2/4.**

**Binding for this run:**

- **Bracket, never single-sided.** Report the best point **below** and
  the cheapest point **above** TXC's measured budget, and interpolate
  to TXC's exact `l0`. A verdict from one side of a coarse grid is not
  a matched-budget verdict.
- **Sweep k finely enough to bracket tightly.** Add intermediate k
  where the grid straddles TXC's budget — cheap, and it is the
  difference between a defensible number and an artifact. **A tight
  bracket is worth more pod-minutes than another seed.**
- **State the bracket width** in the card. If the two ends give
  different verdicts, that IS the finding — report it, do not pick.
- This applies to the **gap** as much as the level: a gap difference at
  mismatched budget is not a matched-budget gap difference.

**The general lesson, and it is why this brief exists twice over:
a selection rule that is defensible in words can be biased in
arithmetic, and it will bias toward whoever wrote it.** Check the rule
against the grid it runs on, not against its own description.

## 3. Grid and scale

`T {2,4,8,16}` × `seeds {1,2,42}` × arms `{txc, pooled, stacked}` ×
`{ordered, shuffled}` × k-sweep × `{trained, untrained-twin}`.

**Han has authorized up to 20 simultaneous H100s.** Notes from
tonight's run, so the scale is usable rather than aspirational:

- **Read `cpu.max` and `memory.max`, NEVER `nproc`/`free`** — a
  container reports host resources. This cost us cores in the morning
  and an OOM at midnight.
- **`max_tasks_per_child=1`** — `_SYNTHETIC_CACHE` is per-process, so
  long-lived workers crossing seeds accumulate ~3× datasource RAM.
- Peak per worker was **24.5 GiB** (concurrent load: fp16 source + fp32
  copy held together), not the ~19 GB steady RSS.
- The 15 sycgen checkpoints exist under tag `sycgen_keep_r1_rebuilt`;
  the eval arms are **post-hoc transforms of the frozen T=1 SAE**, so
  much of this is encode-and-probe, not retrain. **Verify weight
  existence FIRST** — `checkpoint_exists()` answers "on this box", not
  "anywhere", and `cache t=True` is a literal that means "a row exists",
  not "weights exist".

**Ledger both ends; terminate at lane end and API-verify.** 20×H100 is
~$60/h — **it is authorized, which is not the same as free.** Post the
pre-spend estimate before launching.

## 4. Acceptance gate

`figs_writeup/tab_sycgen_shuffle_matched.md` + a card carrying the
pre-registration written before the run, the pooled-zero instrument
receipt, the untrained-twin columns, realized `l0_per_window` per cell,
and a plain-words verdict naming which of (a)–(d) fired.

**If the answer is (b) — architectural, not learned — that is the
result and we publish it.** The prime directive has not moved: *a sound
verdict, never a win.*

**Delete this file when the lane closes.**

---

## 5. ⚑ PRE-REGISTRATION AUDIT (mac-c, 2026-07-29 00:4x BST) — READ BEFORE WRITING CODE

My assigned role on this brief. Four findings, ranked. **A1 is blocking
and costs one line to fix.** All are cheap now and expensive after
20×H100.

### A1 — BLOCKING. The instrument gate cannot fail in the direction that matters, and the failure it misses manufactures the pre-registered publishable answer.

§1 is right that pooled's gap is zero by construction, and right to make
it a gate. But look at what the gate can actually detect.

`frontier.py:119` is `z.mean(dim=1)`. A mean over the window axis is
permutation-invariant **as a matter of arithmetic**. So pooled's zero
survives *any* bug in the shuffle. The only defect that makes this gate
fire is one that renders the pooled arm position-**sensitive** — i.e. a
bug in the comparator, not in the instrument. It is a check on pooled.
It is not a check on the shuffle.

Now the failure it misses. Suppose the shuffle silently no-ops — wrong
tensor consumed, result discarded, permutation applied after pooling,
any of the ordinary ways this goes wrong. Then:

    pooled  gap = 0   -> GATE PASSES ✓
    stacked gap = 0
    TXC     gap = 0
    => verdict reads (b) TXC ≈ stacked

**(b) is what §2 names the LIVE hypothesis and what §4 pre-commits to
publishing.** So the one instrument failure the design does not check
for is the one that produces, through a passing gate, the exact
headline the brief has already promised to publish. Pre-registration is
supposed to stop conclusions being chosen after the data; here it locks
in the conclusion a dead instrument emits.

This is the same shape as the trap §1 caught, one level up: §1 removed a
tautology from the *comparator*, and the residual tautology is in the
*gate*.

**Verified absent at source, not assumed:**

- `grep` across the repo for any assert that `tiles_sh` differs from
  `tiles_ev` returns **empty**. No such check exists anywhere.
- The existing `IDENTITY_TOL` assert (`sycgen/shuffle_overlay.py:165`,
  and the three sibling overlays) is a **replication** guard on
  `|canonical_r − recomputed_r|` — it checks that a recomputation
  reproduces a canonical row. It says nothing about the shuffle.
- `shuffle_within_window` (`src/temp_bench/utils/shuffles.py`) validates
  rank and that `x.shape[1] == T`, and never checks that the
  permutation it drew is non-identity.

**⚑ RECEIPT — this is demonstrated, not argued.** Run it before you
spend anything ($0, no model, no pods, no network):

```
PYTHONPATH=. .venv/bin/python -m \
  experiments.explorations.task_hunt.sycgen.shuffle_gate_receipt
```

It drives the **real** `shuffle_within_window` and a faithful mirror of
`frontier.py`'s two arms, under a live shuffle and a dead one:

     T  shuffle    pooled |diff|   gate   stacked |diff|   input |diff|
     2  LIVE           0.000e+00   PASS        4.358e+00      6.365e+00
     2  DEAD           0.000e+00   PASS        0.000e+00      0.000e+00
    16  LIVE           2.384e-07   PASS        5.096e+00      6.997e+00
    16  DEAD           0.000e+00   PASS        0.000e+00      0.000e+00

**The gate returns PASS in both rows at every T.** It does not
discriminate. The input-side assert fires on DEAD and stays silent on
LIVE — it does.

Note also that this could not have been caught by rewriting the gate:
any number of independent reimplementations share the *assumption* that
pooled-zero tests the shuffle, and every one of them passes a dead
shuffle. Your own `73f8ea388`: **"independence of implementation is not
independence of assumption."**

**FIX — one line, upstream of every arm:**

```python
tiles_sh = shuffle_within_window(tiles_ev, T=T, seed=SHUF_EVAL_SEED)
assert (tiles_sh - tiles_ev).abs().max() > 0, \
    f"SHUFFLE IS A NO-OP at T={T} — instrument dead, cell void"
```

Put it on the **input** side, not the code side: it is arm-independent,
it runs before any encoder, and it cannot be mistaken for a result.
Record it per cell as a receipt next to the pooled-zero receipt.

Keep the pooled-zero gate — but label it as what it is, a check on the
pooled arm. The run needs both: pooled-zero says the comparator is
honest, the no-op assert says the instrument is alive. **A gate and a
positive control are different objects.** Tonight already produced two
cases of a check whose failure looked like success (mac-d's sweep that
ran dead under zsh word-splitting; my own per-token free-lunch
artifact), and both were caught only by a positive control.

### A2 — The mandatory twin has no decision rule, and its likely outcome contradicts (a).

§2 makes untrained twins **MANDATORY** and says a trained-only gap is
not evidence. Good. But (a)–(d) are defined *entirely* on trained-TXC
gap vs trained-stacked gap. The twin is required to be present and is
assigned no interpretation.

The gap that opens: **(a) and the twin condition can both fire.** TXC
gap > stacked gap (⇒ (a), a win) while untrained-TXC gap ≥ trained-TXC
gap (⇒ not learned, the thing that killed the original claim). Both are
consistent with the pre-registration as written, they point opposite
ways, and nothing says which governs — so it gets settled after the
numbers are visible. That is the failure pre-registration exists to
prevent, and §2 itself says this combination is *historically what
happened*.

**FIX:** make the twin a **gate on (a)**, not a column. (a) may be
declared only if the trained-TXC gap exceeds the **untrained-TXC** gap
by the same margin criterion used against stacked. Fail that and the
verdict is (b), whatever the stacked comparison says.

### A3 — Outcome (d) has no threshold, so (a)-vs-(d) is a post-hoc call.

"(d) INDISTINGUISHABLE at n=3" is pre-registered as an outcome with no
stated criterion for what counts as distinguishable. At n=3 that is the
whole ballgame: without a number fixed in advance, the boundary between
"win" and "indistinguishable" is drawn after seeing where the points
fell.

**FIX — state a rule before the run.** Cheap and defensible at n=3:
declare (a) only if the trained-TXC gap exceeds the stacked gap **in
all three seeds** (a sign test; p = 1/8 one-sided under exchangeability)
**and** the mean margin exceeds the across-seed SD of the per-seed
margin. Everything else is (d). The exact numbers are the hub's and
Han's call — the binding point is that a number exists before data.
This is the same principle already ratified for the +0.05 gain bar:
**a threshold that moves after seeing the data is not a threshold.**

### A4 — The T-sweep has a T-dependent instrument artifact, confined to its first cell.

`shuffle_within_window` uses `per_row=True`, drawing an independent
`randperm(T)` per row. A uniform draw is the identity with probability
`1/T!`, so the fraction of rows that are *actually* permuted is
`1 − 1/T!`:

    T= 2   rows truly shuffled = 0.500     <- half the "shuffled" arm is ordered
    T= 4   rows truly shuffled = 0.958
    T= 8   rows truly shuffled = 0.999975
    T=16   rows truly shuffled = 1.000000

Measured on the real helper by the same receipt (B=512): 0.5234 / 0.9512
/ 1.0000 / 1.0000 — both departures are inside binomial sampling noise
at n=512 (1.1 SE and 0.8 SE respectively), so theory and measurement
agree.

At **T=2 — the first cell of the grid `T {2,4,8,16}` — the shuffled
condition is 50% ordered by construction**, and ~4% at T=4.

This does **not** bias the TXC-vs-stacked contrast at fixed T: both arms
consume the same permuted tiles under the same seed, so the attenuation
is common-mode and the headline claim is safe. What it biases is the
**shape of the T-sweep**. Any reading of the form "the gap grows with T"
inherits `1 − 1/T!` as a multiplicative instrument term — a T-trend
that is present in the apparatus regardless of the phenomenon. This is
the same species as the divide-by-`T` per-token artifact from four
hours ago: a T-dependent instrument term read as a T-dependent result.

**FIX:** either report the T=2 cell with the attenuation disclosed, or
draw the per-row permutation from the `T!−1` non-identity permutations
(reject-and-redraw) so the shuffled condition means the same thing at
every T. **Do not silently drop T=2** — the disclosure is worth more
than the cell.

### A5 — minor

Frontmatter reads `issued: 2026-07-29 00:34 BST (was stamped 01:2x — ~40 min in the FUTURE; mac-c A5)`; local clock at the
time of this audit is **00:4x BST**, so the stamp is ~40 min in the
future. (I have corrected five of my own stamps today; flagging it in
the same spirit, not as a criticism.)

### What I am NOT objecting to — checked and withdrawn

I went in expecting to flag "you cannot match sparsity across arms whose
`l0_unit` differs by construction". **That objection is wrong and the
code already answers it.** `frontier.py:100-120` counts, for each arm,
the number of distinct input dimensions actually occupied — union over
positions for pooled (one slot per feature), sum for stacked (`T·d_sae`
slots, so a feature firing at three positions is genuinely three input
dimensions), nonzeros-in-tile for TXC. Those are different *formulas*
producing the *same commensurable quantity*, and the docstring records
that using the union for stacked would flatter the baseline and was
fixed before the sweep ran (hub review `0b1025abc`). §2's "report
`l0_unit` per arm" is correct as written. Recorded because a withdrawn
objection is cheaper for the next reader than a silent one.

---

## 6. ⚑⚑ HUB RULING on mac-c's audit (mac-local, 2026-07-29 00:5x BST) — **ALL FIVE ADOPTED. A1 IS BLOCKING AND THE BRIEF WAS WRONG.**

**A1 is correct and it is my error, at the level the brief was written
to protect.** §1 removed a tautology from the *comparator* and left one
in the *gate*. `frontier.py:119` is `z.mean(dim=1)`; a mean over the
window axis is permutation-invariant **as arithmetic**, so pooled's
zero survives **any** shuffle bug. The gate tests pooled. It never
tested the instrument. And the failure it misses — a silently
no-op shuffle — drives every arm to gap≈0, **passes the gate**, and
reads as **(b)**, which §2 names the live hypothesis and §4
pre-commits to publishing. **I pre-registered a conclusion that a dead
instrument would manufacture through a passing check.** ADOPTED
verbatim, including the placement argument: the assert goes on the
**input** side, arm-independent, upstream of every encoder, recorded
per cell as a receipt.

    tiles_sh = shuffle_within_window(tiles_ev, T=T, seed=SHUF_EVAL_SEED)
    assert (tiles_sh - tiles_ev).abs().max() > 0, \
        f"SHUFFLE IS A NO-OP at T={T} — instrument dead, cell void"

**Keep the pooled-zero gate, relabelled**: pooled-zero says the
comparator is honest; the no-op assert says the instrument is alive.
**A gate and a positive control are different objects** — that
sentence is now a standing rule, not a note on this brief.

**A2 ADOPTED — the twin is a GATE on (a), not a column.** (a) may be
declared **only if** the trained-TXC gap exceeds the **untrained-TXC**
gap by the same margin criterion used against stacked. Fail it and the
verdict is **(b)**, whatever the stacked comparison says. mac-c is
right that (a) and "not learned" could both fire under the brief as
written, pointing opposite ways with nothing to arbitrate — and that
this combination is *historically what happened here*.

**A3 ADOPTED — mac-c's proposed rule is the rule, hub adopting it as
written since they correctly left the number to me:**

> **(a) is declared only if BOTH hold: (i) the trained-TXC gap exceeds
> the stacked gap in ALL THREE seeds** (sign test, one-sided p = 1/8
> under exchangeability) **and (ii) the mean margin exceeds the
> across-seed SD of the per-seed margin.** Anything else is **(d)**.

The same two-part form governs the A2 twin gate. **This is fixed
before data exists and does not move afterwards** — the +0.05 gain-bar
precedent: *a threshold that moves after seeing the data is not a
threshold.*

**A4 ADOPTED, and it reaches BACK INTO A DELIVERED EXHIBIT — I
verified it by measurement rather than accepting `1/T!` as arithmetic**
(20k rows per T): rows identical to ordered = **0.501 / 0.042 / 0.000 /
0.000** at T = 2/4/8/16. **At T=2 half the shuffled condition IS the
ordered condition.** Common-mode at fixed T, so the level story is
safe; but the T-sweep carries a `1 − 1/T!` instrument term that rises
with T **regardless of the phenomenon**. First-order correction
flattens the published trend completely and makes T=2 the largest cell
— **which is enough to forbid quoting a trend, and NOT enough to
publish corrected values** (the probe is fit jointly across rows, so
the gap is not a linear mixture). **Disclosure added to
`figs_writeup/tab_sycgen_shuffle_tsweep.md`.** For this run: draw from
the `T!−1` **non-identity** permutations so "shuffled" means the same
thing at every T, and **state that the estimator differs from the
published exhibit's**.

**A5 ADOPTED — frontmatter fixed.** The stamp read `01:2x` when the
clock said `00:3x`: **~40 minutes in the future**, written from
elapsed-feel rather than `date`. Same class as the LOG headers I
misdated by ~2h earlier tonight, and mac-c has corrected five of their
own today. **Stamp from `date` at write time, never from feel.**

**Withdrawn objection recorded, and the discipline is right:** mac-c
went in expecting to flag cross-arm `l0_unit` incommensurability, read
`frontier.py:100-120`, and found the code already answers it — the
per-arm formulas produce the same commensurable quantity, and using the
union for stacked (which would flatter the baseline) was fixed before
the sweep ran. **A withdrawn objection published is cheaper for the
next reader than a silent one.**

**WHAT THIS AUDIT COST AND SAVED: ~40 minutes of $0 review against a
20×H100 run.** Four defects, one of which would have produced a
publishable-looking result from a dead instrument, through a gate I
designed specifically to prevent that. **The gate I was most confident
in was the one that was hollow** — and I would have been confident
*because* §1's tautology-catch had just succeeded. **A check that
just caught something is not thereby a good check.**

**mac-d: these five are binding. Implement A1 before any cell runs,
then the A2/A3 decision rules into the card, then A4's redraw.** The
pre-spend estimate still comes before launch.

---

## 7. ⚑ HUB CORRECTION TO MY OWN §2b (00:5x) — **the grid is ALREADY tight. Do not spend a single pod-minute refining it. Spend it on SEEDS.**

§2b told you to *"sweep k finely enough to bracket tightly — a tight
bracket is worth more pod-minutes than another seed."* **I measured it
instead of asserting it, and the second half of that sentence is
wrong.** `scripts/plan_bracket_grid.py` ($0, from `frontier.json`,
which already measures pooled's realized l0 at every k):

| T | TXC l0/win | nearest existing pooled point | verdict on refining |
|---|---|---|---|
| 2 | 5.66 | k=4 @ 5.97 — **5.5% above** | already tight |
| 4 | 6.35 | k=2 @ 6.26 — **1.4% below** | already tight |
| 8 | 6.94 | k=1 @ 6.40 — **7.7% below** | already tight |
| 16 | 7.82 | k=1 @ 11.22 — **1.43× above, and k=1 is the FLOOR** | **refinement impossible** |

**ADD NO CELLS.** The k grid already brackets TXC's budget to within
1.4–7.7% at every T where a bracket exists. My §2b guidance would have
bought a rounding correction at real cost.

**Two consequences that change what this run is for:**

1. **The 3/4 → 2/4 correction is FINAL, not provisional.** It does not
   await a finer sweep — there is no finer sweep worth running. T=2 was
   never one cell away from resolution; the near-matched point (5.97,
   +5.5%) **already existed in the data** and the old rule passed over
   it in favour of one at 0.62×. **The bias was never a coverage
   problem. It was a selection problem, and coverage was fine all
   along.**
2. **⚑ T=16 is STRUCTURALLY Pareto, not luckily so.** Pooled's cheapest
   possible configuration — `k=1`, the integer floor — costs **1.43×
   TXC's budget**. No grid, however fine, can produce a matched
   comparison at T=16, because pooled cannot be made that cheap at all.
   **So the Pareto result is not a fallback from a failed match; it is
   the only comparison the arms admit**, and it is in our favour
   legitimately. This is now the strongest defensible sentence in item
   6 and it should be the one we lead with.

**SPEND THE MARGINAL POD-MINUTE ON SEEDS INSTEAD.** n=3 with a crude
threshold has been the binding limitation on item 6 all night —
**outcome (d) has been open since the pre-registration and is still
unsized.** Going n=3 → n=5 on the cells that matter buys a real
reduction in the (d) region; a sixth k value buys nothing. If budget
allows only one improvement, **it is seeds.**

**And note what this cost:** §2b's *"a tight bracket is worth more
pod-minutes than another seed"* was a confident, plausible,
well-reasoned trade-off stated **without measuring either side of it**
— written in the same hour I logged that a rule defensible in words can
be biased in arithmetic. **I did it again, in the correction to the
first instance.** The check is the same one: run the numbers on the
grid you actually have before prescribing spend against it.
