# sycgen shuffle ablation, sparsity-matched — RESULTS

**Verdict: (b) ARCHITECTURAL, NOT LEARNED.** The ordered−shuffled
gap is **not** evidence of learned temporal structure.

Pre-registered in `sycgen/SHUFFLE_MATCHED_CARD.md` **before any**
**cell ran**; outcome (b) was named there as the **live**
hypothesis, and the rule that decides it was frozen in advance.
PTR — pending team review.

## 1. Instrument gates — all PASS

| check | result |
|---|---|
| shuffle live (identity-row count vs binomial band) | **24/24 PASS** |
| pooled gap ≡ 0 (permutation-invariance, 288 rows) | **max \|gap\| 6.53e-09** |
| SAE `l0` permutation-invariant (predicted, then measured) | **0 violations** |

Measured identity-row counts against theory `1 − 1/T!`:

| T | identity rows | n | predicted | band | verdict |
|---|---|---|---|---|---|
| 2 | 8092 | 16384 | 8.19e+03 | 7936..8448 | PASS |
| 4 | 337 | 8192 | 341 | 268..414 | PASS |
| 8 | 0 | 4096 | 0.102 | 0..3 | PASS |
| 16 | 0 | 2048 | 9.79e-11 | 0..0 | PASS |

**Pooled's zero is NOT what certifies the shuffle.** A mean over the
window axis is permutation-invariant arithmetically, so it returns
PASS on a dead shuffle at every T. The gate above is input-side and
arm-independent, checked against an exactly predicted rate.

## 2. THE RESULT — untrained twins show the LARGER gap

Trained vs random-init TXC, **same architecture, same T, same seed**.
This comparison needs no budget matching: it is one arm against
itself.

| T | trained gap | untrained-twin gap | trained > twin? |
|---|---|---|---|
| 2 | +0.1114 | +0.1671 | **0/3** |
| 4 | +0.0231 | +0.1375 | **0/3** |
| 8 | +0.0504 | +0.0820 | **0/3** |
| 16 | +0.0618 | +0.0267 | **1/3** |

**11 of 12 (T, seed) cells have the UNTRAINED twin at the larger**
**gap.** A randomly-initialised model is *more* order-sensitive than
the trained one. That is outcome **(b)**, and it reproduces the
mechanism that dissolved sycgen's original shuffle claim.

### 2b. The qualifier, stated because it cuts both ways

| T | trained: ordered → shuffled | twin: ordered → shuffled |
|---|---|---|
| 2 | 0.4991 → 0.3877 | 0.2219 → 0.0548 |
| 4 | 0.5225 → 0.4994 | 0.1788 → 0.0413 |
| 8 | 0.5363 → 0.4859 | 0.1028 → 0.0208 |
| 16 | 0.5776 → 0.5158 | 0.0575 → 0.0308 |

**The twin barely does the task at all** (0.058 ordered at T=16 vs
the trained model's 0.578). Its gap is therefore a difference
between two near-chance numbers, and raw gaps are not obviously
commensurable across a 10× difference in base recovery. **This is a
limitation of the rule I pre-registered, found by the data.**

It is reported, **not** used to overturn the verdict. The obvious
alternative — a *relative* gap — is **post-hoc and was not**
**pre-registered**, and it does not rescue the claim anyway; it
makes the negative stronger:

| T | trained gap/ordered | twin gap/ordered |
|---|---|---|
| 2 | +0.221 | +0.756 |
| 4 | +0.045 | +0.766 |
| 8 | +0.090 | +0.788 |
| 16 | +0.106 | -0.180 |

The twin loses **76–79%** of its recovery to shuffling at T=2/4/8;
the trained model loses **4.5–22%**.

**Budget confound, disclosed:** the twin runs at `l0`=8.00 (every
`k_pos` slot live) against the trained model's 5.44–7.86, i.e. up
to **1.47×** the budget, which plausibly inflates the twin's gap.
The confound is smallest at **T=16 (1.02–1.03×)** — and that is
exactly where the twin gate is least decisive (mean favours the
trained model, but only **1/3 seeds** agree, so the pre-registered
3/3 sign test fails).

## 3. TXC vs STACKED — the pre-registered comparator

| T | TXC `l0` | TXC gap | stacked floor (k=1) | ratio | matched? |
|---|---|---|---|---|---|
| 2 | 5.60 | +0.1107 | 2.00 | 0.36× | yes |
| 4 | 6.39 | +0.0332 | 4.00 | 0.63× | yes |
| 8 | 7.22 | +0.0043 | 8.00 | 1.11× | **NO — floor above TXC** |
| 16 | 7.86 | +0.0212 | 16.00 | 2.04× | **NO — floor above TXC** |

**At T=8 and T=16 stacked CANNOT operate at TXC's budget.** Its
`l0` is a sum over positions, so its cheapest possible setting is
`T·1` — 8.00 and 16.00 against TXC's 7.22 and 7.86. **Structural,
not a grid-coverage gap**, and no finer `k` can close it. Same
shape as item 6's pooled floor at T=16.

So the matched-budget comparison is **available only at T=2 and**
**T=4**, and the budget ratios printed by the verdict script
(0.64–2.05) mean the standing check fires: **the word "matched"**
**is not earned at T=8/16.** Reported rather than papered over.

Where TXC and stacked *can* be compared, TXC's gap is **larger at**
**T=2** (+0.111 vs +0.022–0.028) and **smaller at T=8/16**
(+0.004/+0.021 vs +0.029/+0.095) — i.e. beyond T=4 the windowed
model is **less** order-sensitive than plain concatenation.

## 4. Verdict

| draw | probe | T=2 | T=4 | T=8 | T=16 |
|---|---|---|---|---|---|
| redraw | fixed (primary) | (b) | (c) | (b) | (b) |
| redraw | refit (secondary) | (d) | (b) | (b) | (b) |
| plain | fixed | (b) | (c) | (b) | (b) |
| plain | refit | (b) | (b) | (b) | (b) |

**Not one cell returns (a).** 15 of 16 return (b) or (c).

**(b) is the pre-registered outcome and it is published as one.**
The card said: *"If the answer is (b) — architectural, not learned
— that is the result and we publish it."*

## 5. What this does and does not say

- It does **not** say TXC fails to recover λ. It plainly does:
  ordered recovery **0.499 → 0.578** across T, against the twin's
  0.222 → 0.058. **Training works.**
- It says the **ordered−shuffled gap is the wrong evidence for**
  **it.** A random model shows that gap too, often larger. The
  gap reflects architectural position-sensitivity.
- Cross-T statements use the `redraw` column only: the `plain`
  draw leaves `1/T!` of rows unshuffled (**50% at T=2**), so a
  "gap grows with T" reading would inherit `1 − 1/T!` from the
  apparatus.
- n=3 seeds. The (d) band is a noise heuristic, **not a
  significance test**.
- Stacked carries `T·d_sae` probe capacity, disclosed and **never**
  **netted out**; uninformative at T≥8 (32768 features, 1024
  windows).
- Substrate is a **rebuilt** activation cache (llama-3.1-8B l14,
  926,592 tokens), not pod-D's original — same disclosure as the
  item-6 cells.

_624 rows, 24 gate receipts, 24 cells; 8 shards on 4×A40._
