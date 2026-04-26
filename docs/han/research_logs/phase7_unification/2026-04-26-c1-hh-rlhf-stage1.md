---
author: Han
date: 2026-04-26
tags:
  - results
  - in-progress
---

## C.i — HH-RLHF dataset understanding (Stage 1, 4 archs)

Reproduces T-SAE paper §4.5 + Appendix B.1 on Phase 7 ckpts. Goal: show
how each arch decomposes Anthropic's helpful-harmless dataset, and which
of its top features are semantically meaningful vs length-spurious.

### Protocol

1. Anthropic/hh-rlhf, **harmless-base** train split, first **N=1000**
   (chosen, rejected) pairs (paper-default).
2. Tokenize each side separately through Gemma-2-2b base, max_length=256.
   Compute char-level LCP between chosen and rejected to mark which tokens
   fall in the differing assistant response (offset_mapping based
   `response_mask`). Forward and capture L12 residual.
3. For each arch (Phase 7 ckpt at seed=42): encode, average activation
   over response tokens only, rank features by `mean_rejected − mean_chosen`
   (paper Table 8 metric).
4. Per top-K feature: Pearson r between `(rejected_act − chosen_act)` and
   `(rejected_response_len − chosen_response_len)` per example.
5. Autointerp top-20 features per arch via Claude Haiku 4.5 on top-5
   max-activating contexts (response tokens only).

### Sanity check: paper response-length t-test reproduction

| | Ours | Paper (App B.1) |
|---|---|---|
| rejected mean (tokens) | 36.23 | 49.243 |
| chosen mean (tokens) | 28.57 | 37.844 |
| diff (mean) | 7.66 | 11.399 |
| p-value | **9.76e-10** | **9e-10** |

p-value matches paper to leading digit. Absolute means lower (different
tokenizer + max-length truncation across HH-RLHF dataset versions). The
effect direction and significance are reproduced exactly.

### Stage 1 headline: top-20 features classified by length-Pearson r

| arch | semantic (|r|<0.2) | mixed (0.2–0.5) | spurious (|r|≥0.5) | semantic |diff| share |
|---|---|---|---|---|
| `topk_sae` (per-token, k=500) | 10 | 10 | 0 | 50% |
| `tsae_paper_k500` (per-token, k=500) | 11 | 8 | 1 | 50% |
| `tsae_paper_k20` (paper-faithful, k=20) | **14** | 6 | 0 | **63%** |
| `agentic_txc_02` (TXC matryoshka, T=5) | 7 | 10 | 3 | 30% |

Plots:
[scatter](../../../experiments/phase7_unification/results/case_studies/plots/phase7_hh_rlhf_scatter.png) ([thumb](../../../experiments/phase7_unification/results/case_studies/plots/phase7_hh_rlhf_scatter.thumb.png)),
[summary bar](../../../experiments/phase7_unification/results/case_studies/plots/phase7_hh_rlhf_summary.png) ([thumb](../../../experiments/phase7_unification/results/case_studies/plots/phase7_hh_rlhf_summary.thumb.png))

### Findings

#### `tsae_paper_k20` (paper-faithful T-SAE) is the cleanest

14/20 features in the top-20 are semantic (|r|<0.2) and ZERO are length-
spurious (|r|≥0.5). 63% of the total |diff| signal comes from
low-|r| features. This is the regime the paper's contrastive loss was
tuned for. At k=20 the BatchTopK is sparse enough that each active
feature carries strong content; the contrastive loss successfully pulls
high-level / semantic features into the lowest-numbered prefix.

Top labels are dominated by alignment-relevant patterns:
- "clarification questions and requests for explanation" (rank 0, r=−0.16)
- "expressions of limitation or inability in responses" (rank 2, r=−0.41)
- "AI capability limitations or disclaimers" (rank 4, r=−0.04)

These activate MORE on chosen than rejected — consistent with the
training-time signal that "I can't help with that" is preferred over
detailed harmful answers.

#### At k=500, T-SAE and plain TopKSAE converge

Both per-token archs trained at our k_win=500 convention end up with
~10-11 semantic top features and 50% semantic diff share. The
contrastive loss benefit visible at k=20 is largely washed out by the
denser activation budget. This is a methodological note for the paper's
"do you need contrastive at high k?" question.

#### TXC's top-20 has the most spurious features

`agentic_txc_02` (TXC matryoshka, T=5, k_win=500) shows 3/20 features
with |r|≥0.5 — versus 0-1 for the per-token archs. Its largest |diff|
features include:

- rank 6: "periods ending sentences with quoted material" (r=+0.56)
- rank 4: "conversational responses and speech patterns" (r=−0.53)
- rank 3: "conversational filler or discourse markers" (r=−0.50)

**Interpretation**: TXC's window aggregation (T=5 tokens summed before
TopK) integrates over the equivalent of a small conversational unit.
Length-driven signals (more chances for fillers, more sentence
endings) are AMPLIFIED by this aggregation. TXC's |diff| magnitudes are
also the LARGEST of the four archs (max 1.52 vs 1.47 for T-SAE k=500),
which is consistent with this amplification.

This is an honest negative for TXC on dataset analysis. The C.ii
steering case study tests whether the same window-level features that
hurt here help at intervention time — TXC's design hypothesis is that
features which are temporally consistent across a window should steer
more coherently than per-token features.

#### Methodological note: top-K matters

Looking only at the top-20 understates the picture. TXC has more
high-|diff| features overall (paper-style ranking), so by rank 50 it
finds many semantically clean features its per-token siblings don't
surface (the diff distribution is heavier-tailed). A future analysis
should report per-rank cumulative semantic count à la Phase 6.3.

### Files

- `experiments/phase7_unification/case_studies/_arch_utils.py` — uniform decoder-direction + encode-per-position helpers
- `experiments/phase7_unification/case_studies/hh_rlhf/build_hh_rlhf_cache.py` — N=1000 chosen/rejected acts at L12
- `experiments/phase7_unification/case_studies/hh_rlhf/decompose_hh_rlhf.py` — per-arch feature ranking + length Pearson
- `experiments/phase7_unification/case_studies/hh_rlhf/label_top_features.py` — Haiku 4.5 autointerp on top-K
- `experiments/phase7_unification/case_studies/hh_rlhf/summarize_hh_rlhf.py` — cross-arch plots + this writeup's tables

### Next

Stage 2 expansion: re-run the same pipeline on
`mlc_contrastive_alpha100_batchtopk`, `phase5b_subseq_h8`, and (when on
HF) `phase57_partB_h8_bare_multidistance_t5`. **MLC needs a multi-layer
cache extension** to `build_hh_rlhf_cache.py` (currently L12-only) —
deferred until after C.ii first-pass.

C.ii steering is the more important deliverable. TXC's design pitch —
window-level features steer more coherently — is testable by AxBench
protocol, and should differentiate window archs from per-token archs in
a way C.i didn't.
