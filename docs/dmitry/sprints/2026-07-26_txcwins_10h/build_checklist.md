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

Most of these are **cheap up front and expensive to retrofit**. Three of them —
contrast-not-absolute metrics, interior-only permutation, and the tSAE zero-regularisation
control — produce a plausible-looking number that means nothing if they are missed, so those
are the ones to verify first against whatever gets built.

Every one of these is argued somewhere below; they are gathered here because they are scattered
and because most of them are cheap to satisfy up front and expensive to retrofit.

1. **Score a contrast, never an absolute.** The metric must be the teacher-forced margin between
   the target and its multiset-matched foil, not accuracy on one member of the pair. Few-shot ICL
   has a published constant intervention — writing a **function vector** — that raises absolute
   accuracy; it cancels in a contrast because the task is identical in both orderings. Get this
   wrong and the comparison is dead before it starts.
2. **Permute the interior only.** Hold the first and last demonstration fixed and match the label
   multiset, so recency and majority-label bias — both bag statistics — cannot separate the pair.
   Apply the same restriction to the permutation sampler in the go/no-go.
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
7. **Run the S2 steering arm**, not just S1. The SAE direction applied at oracle-chosen positions
   is the honest per-token baseline; `S3 > S1` alone invites the reply that the baseline was
   handicapped.
8. **Run the tSAE with temporal regularisation at zero as a control**, because the released
   trainer is `TemporalMatryoshkaBatchTopKSAE` and otherwise the arm confounds three changes.
9. **Carry over the existing controls unchanged** — time-averaged profile, random profile, random
   direction, row-permuted profile, supervised difference-of-means ceiling. They are already in
   `steer_order_modal.py`.
10. **If any AUC is reported**, note the probe-fragility caveat: in-distribution AUCs in this area
    have a poor track record under distributional shift.
