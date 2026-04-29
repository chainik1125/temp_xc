---
author: Agent Y (Aniket pod)
date: 2026-04-29
tags:
  - results
  - in-progress
---

## Y — TXC steering magnitude verification (Q1.1 first; Q1.2 + Q1.3 to follow)

> Tests Dmitry's magnitude-scale hypothesis: window-arch encoder
> activations are O(T x per-token magnitude), so the paper's clamp
> schedule (designed against per-token archs) is "smaller relative
> push" for window archs and explains the 0.5-0.8 success gap under
> paper-clamp.
>
> Pre-registered plan in
> [`2026-04-28-y-orientation.md`](2026-04-28-y-orientation.md).

### Q1.1 — z[j*]_orig distributions across the 6 shortlisted archs

**Method.** For each arch, take the lift-selected best feature `j*`
per concept from Agent C's existing
`results/case_studies/steering/<arch>/feature_selection.json` and
re-run the encoder over the same 30-concept x 5-example probe set
(150 sentences, max_len=64). Record `z[j*]` at every content token
(right-edge only for window archs, all content tokens for per-token
+ MLC archs). Pool per arch over all (concept, example, token)
triples. Compute median + IQR over **active** values (`z > 1e-6`)
since post-TopK / post-threshold inactive positions would otherwise
dominate the distribution and hide the magnitude story.

**Code.**
[`experiments/phase7_unification/case_studies/steering/q1_1_z_orig_distributions.py`](../../../experiments/phase7_unification/case_studies/steering/q1_1_z_orig_distributions.py).
One-shot capture of L12 + L10-14 acts cached to
`steering_magnitude/_l12_acts_cache.npz` so Q1.2 / Q1.3 reuse the
same 150-sentence probe set without re-forwarding Gemma-2-2b base.

**Outputs.**
- [`results/case_studies/steering_magnitude/q1_1_z_orig_distributions.json`](../../../experiments/phase7_unification/results/case_studies/steering_magnitude/q1_1_z_orig_distributions.json)
- [`results/case_studies/steering_magnitude/q1_1_z_orig_distributions.png`](../../../experiments/phase7_unification/results/case_studies/steering_magnitude/q1_1_z_orig_distributions.png)

**Results.**

| arch | T or n_lay | median z[j*] | IQR | n_active_tokens | ratio to T-SAE k=20 |
|---|---|---|---|---|---|
| `topk_sae` (per-token, k=500) | 1 | 10.21 | 4.25-16.50 | 2413 | 1.12 |
| `tsae_paper_k500` (per-token, k=500) | 1 | 11.30 | 5.39-17.28 | 2413 | 1.24 |
| **`tsae_paper_k20`** (per-token, k=20) | 1 | **9.11** | 0.00-17.50 | 2413 | **1.00 (ref)** |
| `mlc_contrastive_alpha100_batchtopk` | 5 layers | 86.89 | 12.79-265.0 | 2413 | **9.53** |
| `agentic_txc_02` (TXC matryoshka) | T=5 | 21.57 | 11.95-31.90 | 1813 | **2.37** |
| `phase5b_subseq_h8` (SubseqH8) | T_max=10 | 63.11 | 38.30-91.52 | 1063 | **6.93** |

![Q1.1 per-arch distribution](../../../experiments/phase7_unification/results/case_studies/steering_magnitude/q1_1_z_orig_distributions.thumb.png)

**Read.**

- **Per-token archs cluster tight** (medians 9-11; ratios 1.0-1.24).
  Topology of this cluster is independent of `k` (k=20 vs k=500). So
  the steering-strength operating point should be ~constant within
  the per-token family.
- **TXC T=5 is 2.37x T-SAE k=20** — below the linear-in-T prediction
  of ~5x. Either the matryoshka multiscale decomposition damps the
  per-feature magnitude, or T=5 averaging is sub-linear. The
  brief's hypothesis pass condition was "window-arch medians ~5x
  per-token medians with at most 2x scatter"; 2.37 is at the edge of
  that band, not in its centre.
- **SubseqH8 T_max=10 is 6.93x T-SAE k=20** — closer to a linear-in-T
  reading (T_max=10 -> 10x prediction; observed 6.93x is ~70% of
  prediction). Subseq sampling integrates over a longer window than
  TXC's T=5 and the magnitude tracks that.
- **MLC 5-layer is the biggest, 9.53x** — Dmitry's analysis only
  considered temporal aggregation; layer aggregation also amplifies,
  even more strongly per axis. This is novel and worth flagging:
  the magnitude story generalises to ANY aggregation, not just T.

**Open question.** Does the **peak steering strength** under
paper-clamp scale with these ratios? If yes (Q1.2), the magnitude
story is the FULL story for the protocol mismatch and family-
normalised paper-clamp (Q1.3) closes the gap. If the peak strengths
ratio is non-linear — e.g. TXC needs a 5x push despite only 2.4x
typical magnitude — then there's a second factor (likely the
error-preserve term interacting with how much of `(s - z[j]_orig)`
is "in-distribution" for the encoder).

### Note on n_active_tokens drop for window archs

| arch | T effective | n_active_tokens / 2413 |
|---|---|---|
| per-token | 1 | 100% |
| TXC T=5 | 5 | 75% |
| SubseqH8 T=10 | 10 | 44% |

Window archs lose tokens at sequence-start positions where no full
window exists (positions `t < T-1` skipped per
`encode_per_position`'s right-edge attribution). Plus,
post-threshold sparsity differs across archs — the active fraction
isn't uniform. This is why I aggregate over active tokens only;
including zero positions would show window archs as "less active",
which is an artefact, not a magnitude story.

### Next

- **Q1.2** — peak-strength curves under paper-clamp, fit `s*_arch`
  per arch, test linear-in-magnitude-ratio.
- **Q1.3** — re-run paper-clamp on the three window archs at
  `s_norm = s_paper × ratio_arch_to_T-SAE-k20`. If TXC catches up to
  T-SAE k=20 within concept-variance noise, Dmitry's hypothesis is
  empirically confirmed.

Cost so far: 0 grader calls (Q1.1 is encoder-only). Q1.2 + Q1.3
will be the grader-heavy steps.
