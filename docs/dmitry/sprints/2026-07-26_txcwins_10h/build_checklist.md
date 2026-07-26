---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - reference
  - in-progress
---

## Status

Pre-flight checklist for anyone building a windowed-steering comparison, consolidated from
constraints that accumulated across this sprint. Lifted out of `literature_catalogue.md` so it
can be read as one list rather than reconstructed from context.

Most of these are **cheap up front and expensive to retrofit**. Seven of them — the symmetric dose
grid (0), contrast-not-absolute metrics (1), interior-only permutation (2), rotation not
transposition (2b), equal segment token lengths (6b), the tSAE zero-regularisation control (8),
and disjoint train/eval demonstration pools (8b) — produce a plausible-looking number that means
nothing if they are missed, so those are the ones to verify first against whatever gets built.

Two of those deserve singling out. **Item 0 has already cost this project a headline result** — a
one-sided dose grid turned a sign into an apparent inversion, and the inversion was the control
that supposedly proved the effect. And **item 2b decides whether the experiment can discriminate
at all**: a two-block swap is exactly rank 1, so no amount of care elsewhere rescues a task whose
optimal write a per-token latent can already express.

Each item is argued in full in [[literature_catalogue]]; this page is the list without the
reasoning.

0. **Use a dose grid symmetric across zero.** This is item zero because violating it is what
   withdrew the previous sprint's headline. `steer_order_modal.py` defaults to
   `alphas = "0.25,0.5,1.0,2.0"`, all positive; `txc_flat` is large and *positive* at negative
   doses, so the positive-only sweep recorded the negative branch and a **sign** was reported as
   an **inversion** — which was the control that supposedly proved the temporal profile carried
   the effect. The repo's own standard already says so:
   `experiments/ward_backtracking_txc/README.md` specifies a grid symmetric across zero, "no
   a-priori reason to favour positive steering", with negative magnitudes as "evidence about
   direction sign and arch behavior, not just floor checks". A one-sided grid cannot distinguish
   a directional effect from a magnitude artefact, and the two point opposite ways: an arm
   positive at *both* extremes is a second-order artefact, an arm effective only at negative
   doses is genuinely directional and invisible to a positive-only sweep.
0b. **Match dose MAGNITUDE with the sign free per arm — and detect violations mechanically.**
   Which class an arm steers toward is something the experimenter knows, so the sign is not a
   result; the matched quantity is `|α|`. Indexing a signed `α = +0.5` scores backwards every arm
   whose effect lives on the negative branch. **The mechanical detector: if an arm's dose response
   is monotone in `α`, its effect is entirely one-sided, and signed-positive indexing is guaranteed
   wrong for whichever arms are monotone-decreasing.** Evidence held-out is monotone decreasing
   (`+3.277 → −3.298` across the six alphas), so at signed `+0.5` the crosscoder reads `−1.120` and
   at matched magnitude `+1.119` — a sign flip, not a magnitude difference.
   This error has now been made **three times in one sprint**: it withdrew the previous headline,
   it appeared in a figure script, and it appeared in a red-team pass *of the document describing
   it*. Print `sign(argmax|Δ|)` per arm next to every reported number; if it is not constant across
   arms, signed indexing is silently comparing different things.
1. **Score a contrast, never an absolute.** The metric must be the teacher-forced margin between
   the target and its multiset-matched foil, not accuracy on one member of the pair. Few-shot ICL
   has a published constant intervention — writing a **function vector** — that raises absolute
   accuracy; it cancels in a contrast because the task is identical in both orderings. Get this
   wrong and the comparison is dead before it starts.
2. **Permute the interior only.** Hold the first and last demonstration fixed and match the label
   multiset, so recency and majority-label bias — both bag statistics — cannot separate the pair.
   **Superseded by 2c where that is available.** Matching the first moment removes recency-of-label
   *analytically*, which is the principled version of what holding the endpoints fixed does by
   construction; imposing both only shrinks the foil set for no gain. Use this item only if the
   two-moment construction is not being used.
2b. **Use a cyclic rotation of the interior demonstrations, not a transposition.** A two-block
   swap has an *exactly* rank-1 optimal write — `+Δ` at one slot, `−Δ` at the other — which a
   per-token latent on a position schedule reproduces, so the comparison cannot discriminate. An
   `m`-block rotation has rank `m − 1` and rank-1 share around `2/m`. Constrain the best/worst
   permutation search to rotations, or measure `r1` on the selected pair before training. This
   also demotes any two-block task (instruction-order conflict, LLM-judge position bias) from the
   headline slot.
2c. **Match the first moment as well as the multiset.** Theory's result, and it is the first design
   in this sprint to achieve rank ≥ 2 and `c = 0` simultaneously. Matching the label multiset
   zeroes the **zeroth** moment of the label difference, which kills the *content* DC — but the
   carried state's DC is the **first** moment, `Σ_j j·Δc_j`, which multiset-matching does not
   touch. Measured across multiset-matched permutations: mean `c` = 0.19, rising to 0.32 at the
   extremes. Adding the constraint `Σ_j j·label_j` equal between orderings gives mean
   `c` = 4.5e-36 with rank 2 preserved. **Caveat for the builder:** the constraint is unsatisfiable
   for extremal arrangements — `[1,1,1,1,0,0,0,0]` sits at the unique minimum of the first moment
   over 4-subsets, so zero valid foils exist (0 of 69); alternating `[1,0,1,0,1,0,1,0]` gives 6 and
   centred `[0,1,1,0,0,1,1,0]` gives 7. Pick a **non-extremal reference ordering**, then enumerate
   foils matching both moments.
2d. **Tune the alphabet size to test the rank bound.** With `q` labels the running-count vector has
   `q−1` free dimensions and the content difference is itself `q−1` dimensional, so `A ≤ 2(q−1)`.
   A falling `r1` as `q` grows tests `rank(P) ≤ A` on a real task with a genuine knob — a stronger
   test than the phase ladder's 0.921 → 0.970. Theory would run this **before** the rotation ladder.
   The single-label control is the falsifier: with identical labels both `Δc` and `Δs` vanish, so
   the task must fall back to whatever rank its content alone supplies.
2e. **Quote `n` beside every `r1` and every retention figure, and never compare across `n`.**
   Both are strongly `n`-sensitive and we have nearly drawn a false conclusion from it once.
   Measured on one design: `r1(grad)` reads **0.401 at n = 12** and **0.587 at n = 200**;
   shared-write retention is noise-floor sensitive as `1/sqrt(n)`, so a probe-smoke **0.756 at
   n = 10** is not comparable to **0.272 at n = 200**. A smoke-scale `r1` is not a small-sample
   estimate of the full-scale one — it is a different number. Fix the `n` before comparing cells,
   and treat any cross-cell `r1` comparison at differing `n` as uninterpretable.
2f. **A moment-matched complement pair exists only for `k_seg ≡ 0 (mod 4)`.** Two conditions must
   hold together: the pattern must be **balanced**, so `k` is even; and complementation must
   preserve moments 1 and 2, which needs `Σ_{j=1..k} j` and `Σ j²` both even. Together these admit
   `k ≡ 0 (mod 4)` and nothing else. Verified by enumeration:

   | `k_seg` | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 24 |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | complement pairs | 1 | — | **1** | — | **7** | — | 24 | 296 |

   So the usable ladder is **8, 12, 16, 20, 24**. Odd `k` cannot be balanced; `k = 10, 14, 18, 22`
   fail on `Σ j` being odd. Plan size sweeps on this ladder or half the rungs will turn out not to
   exist.

   **Quote pairs, not subsets.** A subset enumeration counts each pair twice, once from each side —
   at `k = 12` two qualifying subsets are one pair, at `k = 16` fourteen subsets are **seven**
   pairs. Getting this wrong overstates a sweep's headroom by 2×.

   **`k = 12` is a single point, so the pattern in use was forced rather than chosen** — which
   answers the post-hoc-selection question a reader would otherwise ask, and is a stronger property
   than a robustness check would have established. `k = 16` is the smallest size offering any
   choice, which is why it is the right size for that check — and the check should compare **two of
   its seven pairs against each other at fixed length**, not `k = 12` against `k = 16`, which would
   vary pattern and document length together.

3. **Size the go/no-go at ~128 permutations.** Typical permutation-to-permutation accuracy std is
   about two points; the dramatic gaps are the tails and must be searched for. A 24-permutation
   sweep showing a small spread means "underpowered", not "no effect".
4. **Use the base model, not the instruct model.** Instruction tuning increases prediction
   consistency under input perturbations, shrinking the effect being steered. This differs from
   the harness default and has to be changed deliberately.
5. **Use `k` of 4–8.** Order sensitivity falls as demonstration count grows; many-shot would
   quietly destroy the effect. It is also the window length the crosscoder handles best.
6. **Log realised L0 for every arm** (carried-over debt 3 — nominal `k` does not bind for the
   crosscoder and the failure is silent).
6b. **Equalise demonstration token lengths, and log the realised injected norm.** The harness
   normalises each write to unit Frobenius norm over the `(T, d)` slab and then adds `W[t]` to
   *every token* in segment `t`, so the injected norm is
   `alpha · scale · sqrt(Σ_t len_t · ‖W[t]‖²)` and depends on segment token counts. Matching
   `‖W‖_F` equals matching injected norm only when segments are equal length — measured: a
   concentrated slab injects 5.477 against a broadcast's 2.887 at identical `‖W‖_F` when lengths
   are unequal. Harmless in the existing run, because slot length is independent of slot index
   there, but **selecting best/worst permutations can correlate length with condition**, which
   makes it a systematic bias. Draw demonstrations from a narrow token-length band.

   **Sharpened, after measuring it on the two-moment design.** The injected norm is
   `Σ_t len_t·‖W[t]‖²`, so the confound depends on `‖W[t]‖²` and **vanishes for any write whose
   per-position magnitude is constant** — including `sae_broadcast` and any ±1 profile. Where a
   task is multiset-matched, `Σ_t len_t` is equal between conditions by construction, so uniform
   arms are exactly safe. The exposure is confined to **concentrated** writes: measured on the
   two-moment design, a single-position spike deviates by up to **17%** per document and a random
   slab by up to **14%**, against exactly **0%** for uniform. So this is not a bias that
   manufactures a crosscoder win — it is a per-document dose jitter that affects **only the slab
   arms**, inflating their variance and perturbing peak-dose selection while leaving the broadcast
   arms untouched. Fix by length-matching the paired items, or divide `W[t]` by `sqrt(len_t)`.
7. **Run the S2 steering arm**, not just S1. The SAE direction applied at oracle-chosen positions
   is the honest per-token baseline; `S3 > S1` alone invites the reply that the baseline was
   handicapped.
8. **Run the tSAE with temporal regularisation at zero as a control**, because the released
   trainer is `TemporalMatryoshkaBatchTopKSAE` and otherwise the arm confounds three changes.
8b. **Use disjoint demonstration pools for dictionary training and steering evaluation.** In
   `steer_order_modal.py` both are drawn from the same twenty sentences, so the existing claim is
   "steers the ordering of content it was trained on". A demonstration-order latent trained on the
   demonstrations it then steers is a far more plausible lookup, because demonstrations are longer
   and more distinctive. This is the difference between a result about a mechanism and a result
   about a corpus.

   **Hold out the filler, not the factor.** As written this item is over-broad. The content that
   must be disjoint is everything *except* the manipulated factor — in `recency` the two
   instructions appear in both classes of every document with only their positions swapped, so a
   latent keyed to instruction identity cannot separate class A from class B at all, and holding
   them out is both impossible and unnecessary. The filler is the only content a lookup story could
   key on, so the filler is what the split has to cover. A verification that reports non-zero
   train/eval overlap should be checked against *which* items overlap before it is treated as a
   hole.
9. **Carry over the existing controls unchanged** — time-averaged profile, random profile, random
   direction, row-permuted profile, supervised difference-of-means **reference** (not a ceiling — see below). They are already in
   `steer_order_modal.py`.
10. **If any AUC is reported**, note the probe-fragility caveat: in-distribution AUCs in this area
    have a poor track record under distributional shift.

## Architecture comparisons need per-architecture recipes

Implement's recipe sweep found **each architecture peaks at a different learning rate over a 10×
range**, and the attention tSAE at its own recipe is the **best reconstructor of the three**. Any
comparison run at a single fixed recipe is therefore measuring recipe fit as much as architecture,
and any sentence comparing architectures carries that caveat. This compounds the smoke-test
finding below: a short run at a shared recipe biases the comparison twice over.

## A label correction that propagated

**Difference-of-means is a *reference*, not a ceiling.** It has been called "the supervised
ceiling" throughout this sprint, including by me, and it is not one: a crosscoder latent has been
measured *beating* it (+5.92 against +3.49 on the evidence task). It is the best write of one
particular supervised form. The genuine upper bound is the **mean margin gradient**, and any
"percent of ceiling" figure must say which object it was computed against — the two are close to
orthogonal in practice (`cos(P_dom, Ḡ) = 0.044`), so they are not interchangeable.

## Four things to state in the writeup rather than fix

Each converts an apparent weakness into a strength, or prevents a reader from discovering a
limitation for themselves. All are facts about runs already completed.

Both convert an apparent weakness into a strength, and both are facts about the existing run.

- **The steering result was measured with the SAE holding a 12× coefficient budget advantage.**
  Both dictionaries take the same nominal `k`, but the SAE spends it per *segment* and the
  crosscoder per *window*: at `k = 8`, `k_seg = 12` that is 8 coefficients per segment against
  0.67. Training throughput is matched. A reader will otherwise assume the comparison was
  budget-matched and may suspect it was tuned.
- **The dictionaries are trained on the same content they are asked to steer.** In
  `steer_order_modal.py` both training and test documents are drawn from the same twenty
  sentences, so the current claim is "steers the ordering of content it was trained on". A
  held-out-content split upgrades it; state the limitation either way rather than letting a reader
  find it.
- **A smoke-test configuration systematically flatters supervised arms relative to trained ones.**
  A 500-step run gave `txc_slab = +2.55` against +6.48 from the 2000-step run, while the
  supervised arms moved by less than 0.4 — because they are computed from activations rather than
  learned. A short run therefore biases a *comparison*, not merely its precision. This is
  transferable well beyond this project and nearly reached an executive summary.
- **The dose-selection winner's curse inflates the flat arms, not the peaked one.** `at_best()`
  takes the argmax over the alpha grid on the reporting set. Simulated at 4 doses, 200 documents,
  SEM 0.64, noise correlation 0.85: bias is −0.02 SEM for a peaked TXC-like curve, +0.30 SEM for a
  flat SAE-like curve, +0.41 SEM under the null. The reported gap and z are therefore if anything
  understated, and no correction is needed — but say so, rather than leaving it to be found.
