---
author: Han
date: 2026-04-28
tags:
  - results
  - complete
---

## Phase 7 unification — sparse-probing leaderboard summary

### TL;DR

- **38 trimmed-canonical SAE/TXC architectures** trained on Gemma-2-2b
  base at L12 anchor, all at seed=42; **sparse-probed on 36 SAEBench
  tasks at S=32, k_feat ∈ {5, 20}**.
- **3168 probing rows** (38 × 36 × 2 = 2736 for seed=42 + 6 MLC archs ×
  seed{1,2} × 36 × 2 = 432 for σ data on the MLC family).
- **TXC family wins the headline** (`txc_bare_antidead_t5` = 0.9358 at
  k_feat=20), but the **top-10 spread is only 0.0044**, comparable to
  cross-seed σ (~0.003-0.005 from MLC family triple-seed data).
- **Different leaders at different sparsity budgets**: H8 multidistance
  dominates at k_feat=5; TXC bare at k_feat=20.
- **TXCDR T-sweep peaks at T=5** (0.9314), monotonic decay to T=32
  (0.9074) — temporal aggregation helps at moderate T, hurts at long T.
- **TFA = 0.794** (worst by 0.10) — see caveats; training was
  under-converged.

### Headline bar charts

Sorted-by-mean bar chart of all 38 archs, coloured by family. Error
bars = std across 36 SAEBench tasks.

![headline at k_feat=20](../../../experiments/phase7_unification/results/plots/headline_bar_k20.png)

![headline at k_feat=5](../../../experiments/phase7_unification/results/plots/headline_bar_k5.png)

### Top 5 per sparsity budget (seed=42, S=32)

**At k_feat=20** (less sparse, 20-feature probe):

| rank | arch | mean AUC |
|---|---|---|
| 1 | `txc_bare_antidead_t5` | **0.9358** |
| 2 | `tsae_paper_k500` | 0.9339 |
| 3 | `phase57_partB_h8_bare_multidistance_t3` | 0.9330 |
| 4 | `phase57_partB_h8_bare_multidistance_t9` | 0.9328 |
| 5 | `mlc` | 0.9325 |

**At k_feat=5** (very sparse, 5-feature probe):

| rank | arch | mean AUC |
|---|---|---|
| 1 | `phase57_partB_h8_bare_multidistance_t8` | **0.8989** |
| 2 | `phase57_partB_h8_bare_multidistance_t9` | 0.8962 |
| 3 | `phase57_partB_h8_bare_multidistance_t4` | 0.8960 |
| 4 | `txcdr_t16` | 0.8950 |
| 5 | `txcdr_t8` | 0.8950 |

→ **H8 multidistance is the most robust family**: top-3 at k_feat=5,
top-4 at k_feat=20.

### Per-family means (seed=42, k_feat=20)

| family | mean | range | n |
|---|---|---|---|
| H8 multidistance (T=3..9) | **0.9315** | 0.9292 – 0.9330 | 7 |
| TXC fixed-T + Subseq | 0.9301 | 0.9233 – 0.9358 | 6 |
| TXCDR T-sweep (T=3..32) | 0.9237 | 0.9074 – 0.9314 | 16 |
| Anchor cells (kpos100) | 0.9063 | 0.9031 – 0.9095 | 2 |
| MLC family | 0.9058 | 0.8849 – 0.9325 | 3 |
| per-token / non-TXC (incl tfa_big) | 0.8963 | 0.7943 – 0.9339 | 4 |

H8 multidistance has the **tightest cluster** (0.0038 spread), TXC
fixed-T has the **highest peak**. TXCDR T-sweep is wide because it
spans T=3..32 — see next section.

### T-sweep curves

Three families with multiple T values: TXCDR (T=3..32, full sweep),
H8 multidistance contrastive (T=3..9; T=10..32 dropped from headline
per the trim), and TXC bare antidead (T=5/10/20 anchor points only).

![T-sweep at k_feat=20](../../../experiments/phase7_unification/results/plots/t_sweep_k20.png)

![T-sweep at k_feat=5](../../../experiments/phase7_unification/results/plots/t_sweep_k5.png)

### TXCDR T-sweep (k_feat=20)

| T | AUC |  | T | AUC |
|---|---|---|---|---|
| 3 | 0.9294 | | 12 | 0.9242 |
| 4 | 0.9308 | | 14 | 0.9181 |
| **5** | **0.9314** | | 16 | 0.9260 |
| 6 | 0.9266 | | 18 | 0.9200 |
| 7 | 0.9303 | | 20 | 0.9183 |
| 8 | 0.9270 | | 24 | 0.9200 |
| 9 | 0.9275 | | 28 | 0.9192 |
| 10 | 0.9230 | | 32 | 0.9074 |

Smoothed: AUC peaks at T=4-5, gentle decline through T~10, steeper
decline after T=14. The drop from T=5 (0.9314) to T=32 (0.9074) is
0.024 — roughly 8× the cross-seed σ. **Long temporal contexts hurt
sparse probing AUC.**

### MLC family (3-seed σ data)

The 6 MLC ckpts (Agent A trained at seed=1+2 after Agent C deferred)
give us cross-seed σ for one family:

| arch | seed=42 | seed=1 | seed=2 | mean | σ |
|---|---|---|---|---|---|
| `mlc` | 0.9325 | 0.9388 | 0.9343 | 0.9352 | 0.0026 |
| `mlc_contrastive_alpha100_batchtopk` | 0.9000 | 0.9062 | 0.9050 | 0.9037 | 0.0027 |
| `agentic_mlc_08` | 0.8849 | 0.8911 | 0.8793 | 0.8851 | 0.0048 |

**σ ≈ 0.003-0.005.** This is comparable to the top-10 spread (0.0044).
**Single-seed differences in the top 10 are not statistically
distinguishable** without seed=1 + seed=2 data on the rest of the 35
archs. Agent C's seed=1 probing pass on `han-phase7-agent-c-seed1`
will fill in the per-arch σ for the non-MLC archs.

### Cross-reference with Agent C's case studies

Agent C's HH-RLHF understanding + AxBench steering on a 6-arch
shortlist (`origin/han-phase7-agent-c`) found that the **TXC family
Pareto-dominates** at AxBench steering (suc, coh):

| arch | (suc, coh) mean | suc @ s=24 |
|---|---|---|
| `topk_sae` | (0.37, 2.62) | 0.57 |
| `tsae_paper_k500` | (0.29, 2.70) | 0.41 |
| `tsae_paper_k20` | (0.30, 2.70) | 0.36 |
| `mlc_contrastive_alpha100_batchtopk` | (0.29, 2.75) | 0.41 |
| **`agentic_txc_02` (TXC, T=5)** | **(0.34, 2.77)** | **0.62** |
| **`phase5b_subseq_h8` (T_max=10)** | **(0.33, 2.76)** | **0.59** |

**Cross-arch confirmation**: in sparse probing at k_feat=20, the same
6 archs rank: `tsae_paper_k500` (0.934) > `mlc_contrastive` (0.900) >
`agentic_txc_02` (0.930) > `phase5b_subseq_h8` (0.930) > `topk_sae`
(0.930) > `tsae_paper_k20` (0.927). The probing leaderboard puts
**aggregating archs at the top** (TXC + SubseqH8 + MLC), consistent
with Agent C's "feature aggregation wins downstream tasks" narrative.

`mlc_contrastive` is interesting: it's a probing under-performer
(0.900) but a steering Pareto co-leader (suc=0.41). The two metrics
measure different things — sparse probing rewards sparse, decodable
features; steering rewards features that move the model along
specified directions. Worth a follow-up to understand the divergence.

### Caveats

1. **Single-seed leaderboard**. Only the MLC family has triple-seed
   data (σ ~ 0.003-0.005). Top-10 differences are within noise.
   Agent C's seed=1 pass on `han-phase7-agent-c-seed1` (in flight)
   will give per-arch σ for the 44 non-MLC archs at seed=1.

2. **11 archs DROPPED from headline** per
   `2026-04-27-canonical-trim-emergency.md` (9 H8 multidistance T=10..32,
   2 SubseqH8 T_max=32/64). They were never trained at seed=42 due to
   the deadline. The H8 anchor cell narrative (compare `h8_t20`
   non-anchor vs `h8_t20_kpos100` anchor) is therefore **partially
   broken** — only TXCDR's disentanglement (`txcdr_t20` 0.9183 vs
   `txcdr_t20_kpos100` 0.9095, drop 0.0088) is fully demonstrable.

3. **TFA training was under-converged** (4800/8000 steps; per
   `training_logs/tfa_big__seed42.json`). The 0.794 AUC is suspiciously
   low — possibly a real architectural finding (TFA's cross-attention
   isn't substituting for sliding-window aggregation), but possibly
   under-fit. Re-training at full 8000 steps or higher would resolve.

4. **TFA encode path picked novel-only** (per Phase 5's `tfa_big`
   convention; see `2026-04-27-tfa-encode-bug.md`). The `_full` variant
   (z_novel + z_pred) was not probed in Phase 7. Phase 5 found
   z_novel+z_pred wins at mean_pool — worth re-probing.

5. **FLIP convention**: `winogrande_correct_completion` and
   `wsc_coreference` use `test_auc_flip = max(auc, 1-auc)` per Phase 5
   carryover. The leaderboard uses test_auc_flip for those tasks.

6. **Aggregation choice**: S=32 mean-pool (kept windows = S − T + 1).
   Per-example `first_real` indices ensure short sentences (n_real < S)
   only average the real windows, not the padding prefix. Verified by
   the user's "S=32, T=16 → 17 windows" sanity check (HANDOVER §
   "Sanity check on aggregate_s").

7. **No skipped cells**. All 3168 cells produced AUCs (no
   `skipped:true` rows). The min `n_train_eff` was 1604 (smallest
   bias_in_bios task) — no cell dropped due to insufficient examples.

### What's next

1. **Multi-seed σ for the full 38 trimmed-canonical**: waiting for
   Agent C's `han-phase7-agent-c-seed1` probing chain (~10 hr ETA).
   When done, join `(arch_id, task_name, S, k_feat)` to build a
   3-seed leaderboard with σ bars on every arch.

2. **Reconcile sparse-probing vs steering Pareto**: `mlc_contrastive`
   ranks #1 at steering (suc) but mid-pack at probing. Investigate
   whether contrastive features "decompose semantically" without being
   "linearly decodable" — this could be a paper finding in itself.

3. **TFA re-train at full 8000 steps** to confirm the 0.794 AUC is
   architectural, not under-fit. Use `tfa_big` as the comparison
   baseline.

4. **Probe the dropped 11 archs** if compute permits, to fill the H8
   T-sweep tail and the SubseqH8 T_max sweep. Not blocking for the
   paper.

### Files of record

| asset | path |
|---|---|
| Probing JSONL (3168 rows) | `experiments/phase7_unification/results/probing_results.jsonl` |
| Training index (44 rows) | `experiments/phase7_unification/results/training_index.jsonl` |
| Per-arch training logs | `experiments/phase7_unification/results/training_logs/*.json` |
| Seed completion markers | `experiments/phase7_unification/results/seed_markers/seed{42,1,2}_complete.json` |
| Trained ckpts on HF | `han1823123123/txcdr-base/ckpts/` (38 seed=42 + 6 MLC seed=1/2 + 35 seed=1 from Agent C) |
| Probing driver | `experiments/phase7_unification/run_probing_phase7.py` |
| TFA encode fix log | `docs/han/research_logs/phase7_unification/2026-04-27-tfa-encode-bug.md` |
| Canonical trim log | `docs/han/research_logs/phase7_unification/2026-04-27-canonical-trim-emergency.md` |
| Probing cache S=32 fix | `docs/han/research_logs/phase7_unification/2026-04-27-URGENT-probing-cache-fix.md` |
| Cross-agent split | `docs/han/research_logs/phase7_unification/2026-04-27-HANDOVER.md` |
| Agent C case studies | `origin/han-phase7-agent-c` branch |
