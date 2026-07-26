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

Most of these are **cheap up front and expensive to retrofit**. Six of them — contrast-not-
absolute metrics (1), interior-only permutation (2), **rotation not transposition (2b)**, equal
segment token lengths (6b), the tSAE zero-regularisation control (8), and disjoint train/eval
demonstration pools (8b) — produce a plausible-looking number that means nothing if they are
missed, so those are the ones to verify first against whatever gets built.

Of those, **2b is the one that decides whether the experiment can discriminate at all**: a
two-block swap is exactly rank 1, and no amount of care elsewhere rescues a task whose optimal
write a per-token latent can already express.

Each item is argued in full in [[literature_catalogue]]; this page is the list without the
reasoning.

1. **Score a contrast, never an absolute.** The metric must be the teacher-forced margin between
   the target and its multiset-matched foil, not accuracy on one member of the pair. Few-shot ICL
   has a published constant intervention — writing a **function vector** — that raises absolute
   accuracy; it cancels in a contrast because the task is identical in both orderings. Get this
   wrong and the comparison is dead before it starts.
2. **Permute the interior only.** Hold the first and last demonstration fixed and match the label
   multiset, so recency and majority-label bias — both bag statistics — cannot separate the pair.
   Apply the same restriction to the permutation sampler in the go/no-go.
2b. **Use a cyclic rotation of the interior demonstrations, not a transposition.** A two-block
   swap has an *exactly* rank-1 optimal write — `+Δ` at one slot, `−Δ` at the other — which a
   per-token latent on a position schedule reproduces, so the comparison cannot discriminate. An
   `m`-block rotation has rank `m − 1` and rank-1 share around `2/m`. Constrain the best/worst
   permutation search to rotations, or measure `r1` on the selected pair before training. This
   also demotes any two-block task (instruction-order conflict, LLM-judge position bias) from the
   headline slot.
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
9. **Carry over the existing controls unchanged** — time-averaged profile, random profile, random
   direction, row-permuted profile, supervised difference-of-means ceiling. They are already in
   `steer_order_modal.py`.
10. **If any AUC is reported**, note the probe-fragility caveat: in-distribution AUCs in this area
    have a poor track record under distributional shift.

## Two things to state in the writeup rather than fix

Both convert an apparent weakness into a strength, and both are facts about the existing run.

- **The steering result was measured with the SAE holding a 12× coefficient budget advantage.**
  Both dictionaries take the same nominal `k`, but the SAE spends it per *segment* and the
  crosscoder per *window*: at `k = 8`, `k_seg = 12` that is 8 coefficients per segment against
  0.67. Training throughput is matched. A reader will otherwise assume the comparison was
  budget-matched and may suspect it was tuned.
- **The dose-selection winner's curse inflates the flat arms, not the peaked one.** `at_best()`
  takes the argmax over the alpha grid on the reporting set. Simulated at 4 doses, 200 documents,
  SEM 0.64, noise correlation 0.85: bias is −0.02 SEM for a peaked TXC-like curve, +0.30 SEM for a
  flat SAE-like curve, +0.41 SEM under the null. The reported gap and z are therefore if anything
  understated, and no correction is needed — but say so, rather than leaving it to be found.
