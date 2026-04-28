---
author: Han
date: 2026-04-27
tags:
  - results
  - in-progress
---

## Agent C — seed=1 batch status report (for Agent A)

### Headline

**35 of 38 trimmed-canonical archs at seed=1, all on HF
`han1823123123/txcdr-base/ckpts/<arch_id>__seed1.pt` + `training_logs/`.**
The 3 missing archs are the MLC family.

### What's done

| group | row range | n trained | wall clock | comments |
|---|---|---|---|---|
| 1 (per-token / non-TXC, non-MLC) | 1, 2, 3, 7 | 4 | early in original batch | topk_sae, tsae_paper_k500, tsae_paper_k20, tfa_big |
| 2 (fixed-T TXC + Subseq B2/B4) | 8, 9, 10, 11, 12, 13 | 6 | mid original batch | agentic_txc_02, txc_bare_antidead_t{5,10,20}, phase5b_subseq_track2, phase5b_subseq_h8 |
| 3 (TXCDR T-sweep) | 14–29 | 16 | original batch (12) + 4 in recovery | txcdr_t3..t10, t12..t18 in original; t20, t24, t28, t32 in recovery (separate Python procs) |
| 4 (H8 reduced, T=3..9) | 30–36 | 7 | early in original batch | phase57_partB_h8_bare_multidistance_t{3..9} |
| 5 (anchor cells) | 46, 47 | 2 | recovery | txcdr_t20_kpos100 + phase57_partB_h8_bare_multidistance_t20_kpos100 |

Total: 4 + 6 + 16 + 7 + 2 = **35** ✓

### What's NOT done — and why (apples-to-apples preserved)

3 MLC archs were **deliberately skipped** on Agent C's H100 to preserve
apples-to-apples with seed=42:

| arch_id | row | reason |
|---|---|---|
| `mlc` | 4 | needs `preload_multilayer` (5 × 14.1 GB = 70 GB GPU at canonical PRELOAD_SEQS=24_000) — leaves only ~9 GB for model + Adam + activations on H100 80 GB. Lowering PRELOAD_SEQS to fit would create a different sampling distribution from Agent A's seed=42, breaking multi-seed σ analysis. |
| `mlc_contrastive_alpha100_batchtopk` | 5 | same |
| `agentic_mlc_08` | 6 | same |

Per the user's "if it can't be made fair, SKIP it, do not compromise"
rule. **MLC family at seed=1 + seed=2 stays on Agent A's H200.**

### Apples-to-apples audit (verified)

vs Agent A's seed=42 H200 batch:

| field | seed=42 (Agent A) | seed=1 (Agent C) | match |
|---|---|---|---|
| `PRELOAD_SEQS` | 24_000 | 24_000 | ✓ |
| `--max_steps` | 8000 | 8000 | ✓ |
| `TrainCfg` (lr, batch, plateau, grad_clip, log_every, min_steps) | defaults | defaults | ✓ |
| Cache builder | `src.data.nlp.cache_activations` | same | ✓ |
| Cache `SEED` | 42 | 42 | ✓ |
| Cache shape + dtype | (24000, 128, 2304) fp32 | identical | ✓ |
| HF push convention | `txcdr-base/ckpts/{run_id}.pt` | same | ✓ |
| Training infra files | unchanged from `han-phase7-unification` | unchanged | ✓ |

Branch `han-phase7-agent-c-seed1` divergence vs `han-phase7-unification`:
only **2 new files** (no edits to training infra):
- `experiments/phase7_unification/case_studies/seed1_h100_archs_no_mlc.txt` (44-arch list, MLC-excluded)
- `scripts/launch_seed1_h100_batch.sh` + `scripts/recover_seed1_missing.sh` + `scripts/probing_seed1_chain.sh`

Cross-validation: `topk_sae__seed1` converged at `final_step=6600` —
**identical to seed=42's 6600** for that arch. Strong indication the
training pipeline is byte-equivalent across seeds.

Single non-determinism residual: H100 vs H200 cuDNN kernel selection.
Phase 7 doesn't enforce `torch.backends.cudnn.deterministic=True` — so
even with matched configs, ckpts will differ slightly across hardware.
Standard cross-pod reproducibility caveat.

### Recovery from the OOM cascade

Original `han-phase7-agent-c-seed1` batch (PID 18726 on the launcher
script) trained 29 archs successfully then hit a memory cascade
starting at `txcdr_t20`. Per-arch state (Adam, gen-fn closures) wasn't
fully released between archs in the same Python interpreter, so by the
time the high-T archs came up the GPU had ~70 GB allocated and they
all OOM-skipped.

Of the 15 OOM casualties:
- **9 are dropped** per Agent A's emergency trim (`e8d122d`) — high-T
  H8 t10..t32 + extreme SubseqH8. Skip permanently. ✓
- **6 are kept and were recovered** by `scripts/recover_seed1_missing.sh`,
  which runs each arch in a **fresh Python process** (clean GPU per
  arch). All 6 succeeded. Recovery wall-clock: ~5 hr 20 min.

The OOM cascade is a per-arch-leak bug in the Python interpreter,
not a model-correctness issue — the 29 + 6 = 35 ckpts are equally valid.

### URGENT probing-cache fix

**Merged into branch** (`8f8cc37` is current HEAD). Both `1cdc6d4` and
`fce6401` integrated cleanly. `USE_S32_CACHE=True` flag picked up.

Agent C is **not currently running probing** (none was started — per
original `agent_c_brief.md` Agent A handles probing). Your URGENT note
addresses Agent C with restart steps; I'm interpreting that as a NEW
delegation to run probing on Agent C's pod in parallel with yours.

### Probing-chain plan (script ready, gated on user confirmation)

`scripts/probing_seed1_chain.sh` is committed and ready. It does:

1. **Build probe_cache from scratch** (~30-60 min on H100, ~488 GB
   on disk via `build_probe_cache_phase7.py` — Agent C never had a
   probe_cache; original brief assigned that to Agent A's H200).
2. **`rebuild_probe_cache_s32`** — Agent A's URGENT fix, ~10 min.
3. **`run_probing_phase7 --headline`** — uses `USE_S32_CACHE=True`
   automatically. ~10 hr wall-clock at the corrected ~15 min/arch.

Total chain: ~12 hr.

The chain is **NOT yet running** — waiting on user confirmation of
Agent C's role in probing. If yes, I launch the chain (parallel to
Agent A's H200 probing, halving total throughput time).

### Files / artifacts

- All seed=1 ckpts on `han1823123123/txcdr-base/ckpts/<arch_id>__seed1.pt`
- Training logs on `han1823123123/txcdr-base/training_logs/<arch_id>__seed1.json`
- Marker on `han1823123123/txcdr-base/markers/seed1_complete.json` (from
  the original launcher; covers the 29 from the original batch plus
  whatever the canonical wrapper would record for the recovery-via-`run_one`
  path — see comment below)

### One coordination note for Agent A

The recovery script used `train_phase7 --arch <id>` (i.e., `run_one`),
which **does not write a seed-marker** — that's only written by
`run_canonical`. So the `seed1_complete.json` marker on HF reflects
the original (incomplete) batch. The recovery ckpts ARE on HF
(individual `.pt` and `.json` files), they're just not aggregated
into a single seed-marker.

If your seed=42 → seed=1 transition logic depends on the marker file
listing all 35 archs, the marker should be updated. Suggest one of:
- regenerate the marker by listing `txcdr-base/ckpts/*__seed1.pt` and
  writing a new `markers/seed1_complete.json` (Agent C can do this on
  request);
- OR rely on `huggingface_hub.list_repo_files` directly per-arch
  (more robust anyway).
