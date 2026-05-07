---
author: aniket
date: 2026-05-04
tags:
  - design
  - in-progress
---

## Detection protocol — C5 / C6 / C7

Companion to [[det_steer_steering]]. Hoists the detection
protocol C7's `compute_pr_auc_at_S` already implements into a shared
module (`temp_bench.eval.detection`) so C5 + C6 stop re-implementing
the same probe. Adds the **within-window shuffle ablation** that any
TXC detection cell needs to claim genuine temporal detection.

## TL;DR

- Detection on each case study = sparse linear probe on SAE / TXC
  features predicting the case-study behavior **B**. Complementary to
  steering, which is the causal direction.
- Primary metric for all three: **PR-AUC at S ∈ {1, 2, 4, 8, 16, 32}**,
  with 5-fold GroupKFold by prompt / qid. Not F1, not ROC-AUC.
  Class imbalance is ~12 % on C7; unknown but minority for C5 / C6.
- The TXC must be encoded **with its temporal axis preserved**.
  Mean-pooling residuals before `arch.encode` collapses the only axis
  along which TXC is distinct from a TopK SAE — and this has been the
  recurring confound in the wasteland (SAEBench mean-pools 128 tokens;
  some Phase 5 evals mean-pool inside their judge).
- A **within-window token-shuffle ablation** is mandatory in every
  detection cell. Any "temporal" detection signal that survives
  token-order shuffle inside each T-window is window-density, not
  temporal.

## General framework

For each case study, define:

- a **behavior B** with a binary label per cohort element (sentence /
  rollout / continuation, depending on granularity);
- a **cohort** of inputs with labels for B;
- a per-arch **encode-and-pool** pipeline that yields a feature matrix
  `X ∈ R^{n × d_sae}` and labels `y ∈ {0,1}^n`;
- a sparse **probe**: select top-S features by D⁺/D⁻ mean-difference on
  the train fold, fit L1 logistic regression on those S features, score
  test PR-AUC. Average over 5 GroupKFold folds.

Reference implementation:
`temp_bench.eval.detection.detect_case_study(arch, X, y, group_ids, ...)`.
Returns a `DetectionResult` with `pr_auc`, `pr_auc_shuffled`, and
`shuffle_gap` per S. The internals match
`temp_bench.case_studies.backtracking.compute_pr_auc_at_S`
intentionally — same probe, same selector, same CV.

### Encode-and-pool contract (the load-bearing part for TXC)

| arch family | per-cohort-element shape after encode | aggregator |
|---|---|---|
| TopK-SAE / T-SAE / SAE-arditi | `(B, T_w, d_sae)` (T_w == seq_len for per-token) | element-wise `max` over `T_w` |
| TXC-base / TXC-pro | `(B, 1, d_sae)` window-level latent | trivially squeezed |
| MLC | `(B, T, d_sae)` | element-wise `max` over `T` |
| TFA | `(B, T, d_sae)` | element-wise `max` over `T` |

Two TXC-specific rules:

1. **Stride-1 sliding T-windows by default** when the cohort element is
   a sentence longer than T. Stride-T loses the ability to detect
   features that fire on windows misaligned with T-block boundaries.
   For the C7 reference path, the prior author's
   `extract_labeled_sentence_acts` extracts a single
   T-window of activations *just before* each sentence start
   (`window_offsets=(-13, ..., -8)`); that's effectively a per-sentence
   stride-1 window pre-built into the cache.
2. **`max` over windows, not `mean`.** A feature that fires strongly
   on one window is the detection signal we want. Mean-pool dilutes a
   single sharp activation across all windows and re-introduces the
   axis-collapse confound from the encoding side.

The existing C7 wiring in `case_studies.backtracking.run_arch_evaluation`
does this correctly:

```python
X = arch.encode(torch.from_numpy(sentence_acts).to(device)).abs()
if X.dim() == 3:
    X = X.amax(dim=1)  # max over the T axis — KEEP
X_np = X.detach().cpu().numpy()
```

`temp_bench.eval.detection.encode_and_pool` follows the same pattern
verbatim and is reusable.

## C5 detection — sentiment

**Behavior B**: continuation has the steered sentiment (positive vs
negative) per the Sonnet judge's `success` head crossing a target
threshold.

**Cohort**: each of the V7 magnitude grid's continuations becomes one
labeled example. `B = 1` if the judge's success head clears coh ≥ 2.0
in the target-sentiment direction; `B = 0` otherwise. Drop the
unsteered (mag=0) cells from the detection cohort to avoid the
cut-and-continue baseline drift seen in C7. Cohort size: ~3 archs × 3
seeds × 5 thresholds × ~100 prompts ≈ 4 500 labeled continuations per
arch — comfortable for a 5-fold probe.

**Encoding**: each continuation is encoded layer-13 residuals at the
same hookpoint as training. Apply the encode-and-pool contract above.
For GroupKFold use `prompt_id` as the group so the same prompt isn't
seen in train and test folds.

**Why this addition is worth it**: C5 currently has only the steering
axis (coherence-vs-success curves). Adding detection asks "does the SAE
see sentiment in its features?" — a meaningful capability claim
independent of whether you can causally exploit those features. If
TXC matches T-SAE on detection but loses on steering, that's a strong
"detection-vs-causal-control dissociation" finding for the discussion.
If TXC also wins detection, that's a free C5 axis we currently aren't
reporting.

**Implementation hook**: the existing `case_studies.steering`
pipeline runs per-(concept, magnitude, prompt) generations and judges
them. Adding detection means: persist each continuation's encoded
features per arch, then call:

```python
from temp_bench.eval.detection import detect_case_study

result = detect_case_study(
    arch, sentence_acts=encoded_per_continuation,
    labels=success_at_coh2,
    question_ids=prompt_ids,
)
# result.pr_auc, result.shuffle_gap
```

## C6 detection — emergent misalignment

**Behavior B**: rollout judged align ≤ 50 by Gemini (the existing C6
judge).

**Cohort**: 8 Betley first-person prompts × 8 rollouts × N(α-regimes) ≈
64–256 labeled rollouts per organism. R1 and R32 are separate detection
cells (different organisms — don't pool).

**Encoding**: layer-24 `resid_post` activations on each rollout's
continuation (token positions only, NOT the prompt). Apply
encode-and-pool. GroupKFold by `prompt_id` to avoid trivial leakage.

**Why detection matters here**: C6's headline is the bundle-null result
and the gap-close test. Adding detection orthogonally answers "is
misalignment linearly readable from each architecture's features?" If
SAE-arditi has higher detection PR-AUC than TXC, that's consistent
with the steering loss. If TXC ties or beats SAE-arditi on detection
but loses on steering, we have a second dissociation that strengthens
the bundle-null framing — TXC has the information, but its features
aren't causally aligned with the model's internal use of
misalignment-relevant directions.

**Compute caveat**: 64 rollouts × ~200 tokens per rollout × stride-1
T=5 windows ≈ 12 600 windows per cohort. The forward + probe is
seconds per cell.

**Implementation hook**: clone the C5 pattern; the only change is the
hookpoint (`L24 resid_post`) and the label source (Gemini align head).

## C7 detection — backtracking

**Already specified** in `docs/components/c7.md` and implemented in
`case_studies/backtracking.py`. Two TODOs to verify before the headline
run:

1. **Confirm sentence-window encoding preserves T axis.**
   `extract_labeled_sentence_acts` correctly returns `(n_sent, 6,
   d_in)` per the prior author's window-offset convention — verified on
   `sentence_acts_L10.npz` (25 204 sentences × 6 × 4096 fp32, 12.6%
   positive). The T axis survives encode; the `amax(dim=1)` pool in
   `run_arch_evaluation` is correct.
2. **the prior wasteland κ values (0.749 / 0.773 / 1.000) are not for
   the probe; they're for the Sonnet judge.** Detection has no judge
   — it uses the existing Stage A `sentence_labels.json` as ground
   truth. No κ validation needed for detection.

## Within-window shuffle control

For each TXC detection cell, run a paired ablation: permute token
order within each T-window before `TXC.encode`, then re-run the probe.

```python
from temp_bench.utils.shuffles import shuffle_within_window
x_shuffled = shuffle_within_window(x, T=T, seed=42)
```

`detect_case_study` runs this automatically when `shuffle_seed` is
not `None` (the default).

**Decision rule** (per (arch, case study) cell):

- If `PR-AUC[unshuffled] − PR-AUC[shuffled] ≥ 0.02` (paired across S):
  TXC's detection is **genuinely temporal**. Report both numbers; the
  gap is the temporal contribution.
- If the gap is below 0.02 across all S: TXC's detection signal is
  **window-density**, not temporal. Report this honestly — it doesn't
  invalidate TXC, but it means the temporal axis isn't doing detection
  work on this case study.

The shuffle control is the cleanest evidence the paper can give that
TXC's "temporal" claim survives at the detection axis. Without it, a
reviewer saying "your TXC just has more parameters per window than a
SAE" cannot be rebutted.

**Note on TXC-pro**: `txc_pro.py` uses a per-position W_enc[t] slab
even with the subseq-encoder schedule (verified on the locked spec —
see `docs/paper/architecture.md`). It is **not** permutation-invariant,
so the shuffle gap is informative for both TXC variants.

## Pooled implementation plan

Three discrete deliverables for agent_steer (C5), agent_em (C6), and
agent_back (C7) to land before paper submission:

1. **`temp_bench.eval.detection`** — landed on `det-steer`, ~250 lines.
   Single import; the case-study agents invoke
   `detect_case_study(arch, X, y, qids)` from inside their existing
   `analysis.py`.
2. **`temp_bench.utils.shuffles.shuffle_within_window`** — landed on
   `det-steer`, ~50 lines including numpy variant.
3. **AUTO-RESULTS extension** in `cN.md`: each component's
   `analysis.py` adds a "Detection PR-AUC" subsection alongside the
   existing tables, with the shuffle gap reported as a paired column.

| component | analysis.py adds | gated on |
|---|---|---|
| c5 | PR-AUC table at S ∈ S_grid + shuffle gap | continuation + judge labels |
| c6 | PR-AUC table at S ∈ S_grid + shuffle gap | rollout + judge labels |
| c7 | already in scope; verify shuffle gap is reported | sentence_acts wiring |

Cross-territory: `temp_bench.eval.detection` and the shuffle helper
are new shared scaffolding. agent_paper coordinates the merge from
`det-steer` into `final` (per the maintainer); the three case-study agents import
from `temp_bench.eval.detection` in their respective `analysis.py`.

## Methodology validation

`experiments/det_steer/validate_protocols.py` runs the protocol on
synthetic cohorts with **known temporal vs density structure** and
verifies:

- Temporal-signature cohort (positives: dim-0 spike at t=0,
  anti-spike at t=T-1): unshuffled PR-AUC > shuffled PR-AUC by 0.06 to
  0.26 across S — **positive shuffle gap, growing with S**.
- Density cohort (positives: dim-0 spike at every t): unshuffled and
  shuffled PR-AUC overlap within ±0.07 — **shuffle gap ≈ 0**.

See `experiments/det_steer/results/validate/pr_auc_shuffle_gap.png`.
The protocol distinguishes the two cohorts as documented.

## Wiring summary

```python
# 1. Get sentence-window activations for the cohort.
#    C7 reference: sentence_acts_L10.npz, shape (25204, 6, 4096).
sentence_acts = ...                     # (n_sent, T, d_in) fp32
labels        = ...                     # (n_sent,) 0/1
qids          = ...                     # (n_sent,) str — for GroupKFold

# 2. For each arch's locked checkpoint:
from temp_bench.eval.detection import detect_case_study
result = detect_case_study(
    arch, sentence_acts, labels, qids,
    S_grid=(1, 2, 4, 8, 16, 32),
    shuffle_seed=42,                    # paired ablation
)

# 3. Render into the component's AUTO-RESULTS via analysis.py +
#    temp_bench.report.render(component="cN").
```

## References

- [[det_steer_steering]] — companion steering audit.
- [[det_steer_summary]] — integration TODOs + methodology validation.
- `temp_bench.eval.detection`, `temp_bench.utils.shuffles`,
  `temp_bench.case_studies.backtracking.compute_pr_auc_at_S`
- `papers/temporal_sae`, `papers/are_saes_useful`,
  `papers/backtracking`
