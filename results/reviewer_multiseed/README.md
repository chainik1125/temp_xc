## Reviewer seed audit

Snapshot: 2026-07-27T17:18:58Z

Hugging Face repository:
[`dmanningcoe/temp-xc-reviewer-results`](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results)

Dedicated folder: `reviewer_seed_audit_2026-07-27/`

This audit distinguishes the 20,000-step backtracking window/order-control
sweep from the 300,000-step C7 paper-headline cell. They answer related
questions but are not interchangeable seed replications.

## Coverage

| Task | Frozen protocol | Requested seeds | Complete at snapshot | Status |
|---|---|---:|---:|---|
| Backtracking window/order control | TXC-base, T=1…6, 20k steps, d_sae=32,768, k_pos=20, batch 1,024 | 1, 2, 42 at every T | 18/18 cells | complete |
| C7 headline | TXC-base, T=5, 300k steps, d_sae=32,768, k_pos=20, batch 1,024 | 1, 2, 42 | 1/3 | seeds 1/2 queued after medical seed 2 |
| Medical EM | TXC-base, T=5, 25k steps, batch 1,024 | 1, 2, 42 | 2/3 | seed 2 running |
| HH-RLHF preference | `agentic_txc_02`, T=5, exact paper-match evaluation | 1, 2, 42 | 3/3 | complete |

## Backtracking window/order control

The primary values below use the largest registered sparse-probe budget
(`S=32`). “Shuffled” applies a deterministic within-window permutation before
encoding. At T=1, shuffling is the identity by construction.

| T | Seed | Ordered PR-AUC | Shuffled PR-AUC | Ordered − shuffled | Exact HF result |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 0.223351 | 0.223351 | 0.000000 | [T1 seed1](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T1_seed1.json) |
| 1 | 2 | 0.216823 | 0.216823 | 0.000000 | [T1 seed2](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T1_seed2.json) |
| 1 | 42 | 0.237727 | 0.237727 | 0.000000 | [T1 seed42](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T1_seed42.json) |
| 2 | 1 | 0.227257 | 0.224284 | 0.002973 | [T2 seed1](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T2_seed1.json) |
| 2 | 2 | 0.231947 | 0.230545 | 0.001401 | [T2 seed2](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T2_seed2.json) |
| 2 | 42 | 0.242972 | 0.239034 | 0.003938 | [T2 seed42](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T2_seed42.json) |
| 3 | 1 | 0.232072 | 0.230557 | 0.001515 | [T3 seed1](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T3_seed1.json) |
| 3 | 2 | 0.234556 | 0.232504 | 0.002052 | [T3 seed2](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T3_seed2.json) |
| 3 | 42 | 0.236474 | 0.225362 | 0.011112 | [T3 seed42](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T3_seed42.json) |
| 4 | 1 | 0.252911 | 0.238759 | 0.014152 | [T4 seed1](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T4_seed1.json) |
| 4 | 2 | 0.256912 | 0.237741 | 0.019171 | [T4 seed2](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T4_seed2.json) |
| 4 | 42 | 0.243845 | 0.229711 | 0.014134 | [T4 seed42](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T4_seed42.json) |
| 5 | 1 | 0.264962 | 0.250198 | 0.014764 | [T5 seed1](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T5_seed1.json) |
| 5 | 2 | 0.265271 | 0.253698 | 0.011573 | [T5 seed2](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T5_seed2.json) |
| 5 | 42 | 0.260729 | 0.244621 | 0.016108 | [T5 seed42](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T5_seed42.json) |
| 6 | 1 | 0.265800 | 0.248300 | 0.017500 | [T6 seed1](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T6_seed1.json) |
| 6 | 2 | 0.267651 | 0.255118 | 0.012533 | [T6 seed2](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T6_seed2.json) |
| 6 | 42 | 0.255365 | 0.242180 | 0.013185 | [T6 seed42](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/window_order_control/cells/T6_seed42.json) |

Three-seed aggregation:

| T | Ordered PR-AUC mean ± SD | Shuffled PR-AUC mean ± SD | Gap mean ± SD |
|---:|---:|---:|---:|
| 1 | 0.225967 ± 0.010695 | 0.225967 ± 0.010695 | 0.000000 ± 0.000000 |
| 2 | 0.234059 ± 0.008068 | 0.231288 ± 0.007403 | 0.002771 ± 0.001280 |
| 3 | 0.234367 ± 0.002207 | 0.229474 ± 0.003692 | 0.004893 ± 0.005392 |
| 4 | 0.251223 ± 0.006695 | 0.235404 ± 0.004956 | 0.015819 ± 0.002903 |
| 5 | 0.263654 ± 0.002538 | 0.249506 ± 0.004578 | 0.014148 ± 0.002329 |
| 6 | 0.262939 ± 0.006624 | 0.248533 ± 0.006472 | 0.014406 ± 0.002699 |

The ordered curve improves with T, but most of the improvement survives
shuffling. The order-dependent residual grows from exactly zero at T=1 to
roughly 0.014–0.016 PR-AUC at T=4…6.

## C7 headline

The completed seed-42 artifact is the published 300k paper result. It predates
the paired ordered/shuffled detector used for the seed top-up, so its published
steering and detection metrics are reported without inventing a shuffle value.

| Seed | Steps | Published result | Status | Exact HF path |
|---:|---:|---|---|---|
| 1 | 300,000 | — | queued | [status](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/live_run_status.json) |
| 2 | 300,000 | — | queued | [status](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/live_run_status.json) |
| 42 | 300,000 | ΔGC peak 0.540984 at magnitude −12; PR-AUC S32 0.249917 | complete | [seed42](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/c7_headline/seed42_published_eval.json) |

## Medical EM

Primary metric is PR-AUC at S=16. The historical paper seeds used per-cell Wang
cohorts, so the cohort size and positive rate are retained in each raw row.
Seed 2 is being trained using the exact 25k training recipe and the staged
1,728-rollout cohort.

| Seed | PR-AUC S16 | Shuffled PR-AUC S16 | Ordered − shuffled | Status | Exact HF path |
|---:|---:|---:|---:|---|---|
| 1 | 0.559959 | 0.561927 | −0.001967 | complete | [seed1](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/medical_em/seed1_published_eval.json) |
| 2 | — | — | — | running | [status](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/live_run_status.json) |
| 42 | 0.541960 | 0.601169 | −0.059209 | complete | [seed42](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/medical_em/seed42_published_eval.json) |

## HH-RLHF preference

Metric is five-fold preference ROC-AUC using the exact paper cache. The
integrity gate reproduced rejected length 36.232, chosen length 28.573,
`p=9.76e-10`.

| Seed | Ordered AUC | Shuffled AUC | Ordered − shuffled | Status | Exact HF path |
|---:|---:|---:|---:|---|---|
| 1 | 0.622901 | 0.619592 | 0.003309 | complete | [seed1](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/rlhf/agentic_txc_02__seed1.metrics.json) |
| 2 | 0.605258 | 0.604141 | 0.001117 | complete | [seed2](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/rlhf/agentic_txc_02__seed2.metrics.json) |
| 42 | 0.609647 | 0.597529 | 0.012118 | complete | [seed42](https://huggingface.co/datasets/dmanningcoe/temp-xc-reviewer-results/blob/main/reviewer_seed_audit_2026-07-27/rlhf/seed42_published_papermatch.json) |

The full seed-1/2 result payloads remain on the stopped RunPod volume; the HF
files above preserve the completed headline metrics and frozen provenance
while that host lacks capacity to remount the volume.

## Provenance

- Window/order protocol: `2026-07-23.2`, code commit `6b2cb1e0`,
  activation artifact SHA-256
  `1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810`.
- C7 headline runner: commit
  `b8ab4b95dc8d5a7b6da28bdcb71acfaa9c42aff5`; seed-42 train key
  `8787f8fe527218ad`.
- Medical paper rows: historical leaderboard at commit `e1c4f616`;
  evaluation protocol `3.0.0`.
- HH-RLHF runner: commit
  `ed9a6c77fdc697bd9739647eeb15d7d764aad1ca`; checkpoint repository revision
  `187666c5bfde80fe4ea20a64c1ed5d3092874320`.

## Verification

`link_verification.json` records an anonymous HTTP check and SHA-256 digest for
every file published in the dedicated HF folder. A link is considered verified
only when the anonymous `resolve/main` URL returns HTTP 200 and the downloaded
bytes match the uploaded digest.

Verification completed at 2026-07-27T17:21:59Z: all 26 distinct HF links in
this document returned HTTP 200 anonymously, and all 50 files in
`content_manifest.json` matched their published byte count and SHA-256 digest.
