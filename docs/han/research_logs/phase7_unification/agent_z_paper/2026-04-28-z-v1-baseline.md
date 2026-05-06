---
author: Han
date: 2026-04-28
tags:
  - results
  - in-progress
---

## hill_subseq_h8_T12_s5 — round1 V1 ground-truth (Agent Z baseline)

### Status

V1 (Agent A's `hill_subseq_h8_T12_s5__seed42`) was already probed under
`phase7_S32_first_real_meanpool` before Z arrived — 72 rows in
`probing_results.jsonl` (S=32, k_feat ∈ {5, 20}, all 36 tasks, both
FLIP tasks present). No re-probing needed; numbers below are the
existing rows.

### Numbers (seed=42, base side)

| metric | V1 mean AUC | leaderboard #1 | Δ | rank |
|---|---|---|---|---|
| k_feat=5  | 0.8951 | 0.8989 (`phase57_partB_h8_bare_multidistance_t8`) | **−0.0038** | 4 |
| k_feat=20 | 0.9329 | 0.9358 (`txc_bare_antidead_t5`)                   | **−0.0029** | 4 |

### Verdict: **NEUTRAL**

V1 sits ~0.003–0.004 below the leaderboard winners at both metrics —
within the H8 family's tightest cluster (0.0038 spread at k_feat=5).
Not statistically distinguishable from #2/#3, but does not beat the
top cell at either metric.

### Training meta

- Recipe: SubseqH8 at T_max=12, t_sample=5, auto-scaled shifts (1, 3, 6).
- Trained: H200 (Agent A, RunPod credit-out 2026-04-26 per cf85431),
  b=4096, lr=3e-4, max_steps=8000.
- final_step=3000 (plateau early-stop at min_steps floor),
  plateau_last=0.0102, elapsed=110 min.
- Ckpt: 2.0 GB, on HF `han1823123123/txcdr-base/ckpts/hill_subseq_h8_T12_s5__seed42.pt` (+ training_log on HF).

### Implication for Z's plan

The brief's lowest-effort path ("if V1 already beats 0.8989, that's
the win") is closed. Z must produce a winner from V2 (T_max=16) or
round2 (long-distance shifts) — the harder paths.

V1 being competitive (rank 4 at both metrics) does suggest the
SubseqH8 family generalizes well to base. T_max=16 is a natural
extension; if going larger T_max helps, V2 should beat V1 → V2
becomes a candidate to also beat 0.8989.

### Files referenced

- ckpt (local): `experiments/phase7_unification/results/ckpts/hill_subseq_h8_T12_s5__seed42.pt`
- training_log (local): `experiments/phase7_unification/results/training_logs/hill_subseq_h8_T12_s5__seed42.json`
- training_index (local): row 100, group 99, fixed ckpt path
  `(was /workspace/... → now /home/elysium/...)`
- probing_results.jsonl: 72 rows, run_id=`hill_subseq_h8_T12_s5__seed42`
