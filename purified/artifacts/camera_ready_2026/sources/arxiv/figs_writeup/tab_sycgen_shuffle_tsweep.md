# tab: sycgen shuffle/T-sweep (item 6) — PARTIAL 15/18, PENDING TEAM REVIEW

_Generated from `sycgen/results/{sycgen_shuffle_overlay,sycgen_twin_overlay,sycgen_tsweep_summary}.json` at render 03:57 28-07; n=3 seeds per cell; arch `txc_batchtopk_post_btkonly` (btk-only arm, either-arm rule); T-SAE anchor trio pending (regrind), joins the final render._

| T | ordered r (mean ± sd) | shuffled r (mean ± sd) | gap (ordered−shuf) | untrained-twin gap |
|---|---|---|---|---|
| 2 | 0.4982 ± 0.0174 | 0.4648 ± 0.0429 | +0.0334 | +0.1531 |
| 4 | 0.5243 ± 0.0120 | 0.5016 ± 0.0073 | +0.0227 | +0.1664 |
| 8 | 0.5413 ± 0.0083 | 0.5013 ± 0.0325 | +0.0399 | +0.1015 |
| 16 | 0.5922 ± 0.0074 | 0.5296 ± 0.0268 | +0.0626 | +0.0958 |
| 1 (anchor) | per-token BatchTopK SAE: 0.4819 ± 0.0101 | ≡ ordered (identity by construction) | — | — |

> ## ✅ SUPERSEDED FOR THE ORDER QUESTION (02:31 07-29) — the sparsity-matched run has answered it
>
> **`figs_writeup/tab_sycgen_shuffle_matched.md` is the authoritative
> shuffle result.** It ran at matched sparsity with the instrument
> gates this table never had, and its verdict is **(b) ARCHITECTURAL,
> NOT LEARNED**: a randomly-initialised TXC is *more* order-sensitive
> than the trained one in 11 of 12 cells.
>
> **This table's numbers stand as recorded** — they are the T-sweep at
> the original (unmatched) sparsity — **but do not quote them as
> evidence about learned order-use.** The binding quote-form below
> already said the claim is the LEVEL story; the matched run is why
> that was the right call.

> ## ⚑ ADDED 00:5x 07-29 — the "shuffled" arm is only PARTLY shuffled at small T, and the T=2 cell is half ordered
>
> `shuffle_within_window` draws an **independent `randperm(T)` per row**
> (`per_row=True`), and a uniform draw of `T` items **is the identity
> with probability `1/T!`**. Those rows are byte-identical to the
> ordered condition. **Measured, not derived** — 20k rows per T:
>
> | T | rows identical to ordered | rows truly shuffled |
> |---|---|---|
> | 2 | **0.501** | **0.499** |
> | 4 | 0.042 | 0.958 |
> | 8 | 0.000 | 1.000 |
> | 16 | 0.000 | 1.000 |
>
> **At T=2 half of the "shuffled" condition is the ordered condition.**
>
> **What this does NOT threaten:** the ordered−shuffled contrast *at a
> fixed T*. Every arm consumes the same permuted tiles under the same
> seed, so the attenuation is **common-mode** and the level story is
> safe.
>
> **Scope, precisely (mac-c `b275ae27d`): the shuffle touches only the
> eval tiles of the SHUFFLED column.** The **ordered** T-sweep never
> sees it and is **untouched**; A4 can move only the shuffled curve,
> and therefore the gap. Fixed-T comparisons stand as recorded,
> including the anchor gates.
>
> **What it DOES threaten: any reading of the T-sweep as a trend.**
> The apparatus carries a `1 − 1/T!` multiplicative term that rises
> with T **regardless of the phenomenon**, so "the gap grows with T"
> is confounded by construction. A first-order correction (divide by
> the truly-shuffled fraction) gives ≈ 0.067 / 0.024 / 0.040 / 0.063 —
> **no trend at all, and T=2 becomes the largest cell.**
>
> **Those corrected numbers are an ORDER-OF-MAGNITUDE CHECK, NOT a
> restatement, and must not be quoted.** The probe is fit jointly
> across rows, so the gap is not a linear mixture of per-row gaps and
> this correction is only approximate. It is sufficient to establish
> that **the trend must not be quoted**; it is not sufficient to
> publish corrected values.
>
> **⚑ AMENDED 01:1x — mac-c measured the magnitude and I had named the
> wrong dominant mechanism.** The attenuation here is **smaller than
> seed noise**: the T=2 gap is **+0.0334 ± 0.0386** — SD *exceeds* the
> mean — and the raw T=2 gap is already **larger** than T=4's
> (+0.0227), which is the **opposite** of what attenuation-dominance
> would produce. So A4 is a **disclosure obligation here, not an
> invalidation.**
>
> **The binding reason not to read a trend is simpler and worse: the
> low-T cells are not statistically resolved at n=3 at all**,
> independent of the shuffle artifact. **A trend drawn through an
> unresolved first point is not a trend.** Both reasons stand; this
> one governs.
>
> *(Same failure as the T=2 budget cell earlier tonight: I flagged the
> right cell and named the wrong mechanism. A correct flag with the
> wrong cause sends the next reader to the wrong knob — and here it
> would have overstated the artifact, which is the same error as
> ignoring it, pointed the other way.)*
>
> **The binding quote-form below already says the claim is the LEVEL
> story, not the order story — that was the right call and this is why.**
> Found by mac-c's pre-registration audit of the follow-up run (A4).
> The re-run fixes it at the source by drawing from the `T!−1`
> non-identity permutations.

**Quote-form (binding, LOG 04:16):** shuffle costs 0.02–0.06 recovery
(positive 12/12 trained cells), but untrained twins show LARGER gaps at
every T — the gap is architectural position-sensitivity; not learned
order-use. — **⚑ and the budget confound was REMOVED, not merely disclosed: at matched budget the twin still shows the larger gap at T=2 (+0.0365) and T=4 (+0.1249), 3/3 seeds each; T=8/16 indeterminate. An earlier hub note called "training reduces it" unestablished on a budget-artifact hypothesis the control REFUTED. See `tab_sycgen_shuffle_matched.md` §2c** Training still lifts recovery ≤0.22 → 0.50–0.59.
The claim is the LEVEL story. l0 NOT budget-matched (TXC 0.49–2.85
l0/token vs SAE ~4.5 — sparser and above the anchor; flag travels).
Untrained twin levels: 0.075–0.218 (see fig).
