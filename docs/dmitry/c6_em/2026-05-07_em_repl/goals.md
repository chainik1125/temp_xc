---
author: Dmitry Manning-Coe
date: 2026-05-07
tags:
  - proposal
  - in-progress
---

## Macro streams

Two macro streams of work currently active.

### 1. Sleepers

The goal in the sleepers stream is to tell as complete of a mechanistic story as we can about how the sleeper agent is formed.

The current exciting finding (based on Ketan's results) is that we can suppress the feature more efficiently by steering on 10 features in the OV than on a single feature in the resid. The nice empirical property of this is that, for most seeds, a single feature in `resid_mid` can control suppression almost perfectly.

### 2. Emergent misalignment (EM)

The goal in the EM stream is to see if we can find a superior coherence/suppression trade-off frontier by attributing in and then intervening on a pair of QK and OV. Here Nura has a striking result that QK on medical advice leads to very favourable alignment/coherence trade-offs.

We typically measure the trade-off as the maximum change in alignment at coherence > 70 (the same `Δalign|coh≥70` definition used in `scripts/plot_c6_em_align_coh_grid.py:headline_metrics()`). The steering range is much smaller in the other domains (finance, sports) but not negligible.

## EM stream — three-step process

This document scopes the EM stream only. Sleepers continues independently.

1. **Reproduce.** Replicate Nura's headline result: QK→OV on medical gives a large `Δalign|coh≥70`, smaller-but-nonzero on finance/sports.
2. **Redteam.** Look for mistakes — implementation bugs, statistical artifacts, judge-dependence, confounds.
3. **Benchmark.** Compare to same-budget SAEs trained on the same and next layer at multiple hook points: `resid_pre`, `ln1.hook_normalized`, `resid_mid`, `resid_post` (layer 24) and `ln1.hook_normalized` (layer 25).

## Replication target

- Model: `Qwen/Qwen2.5-14B-Instruct` + EM medical LoRA (`ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice` — confirm exact LoRA from Nura's CODEBASE.md before launch).
- Layer 24, attention sublayer; head set `{H38, H0, H36, H7}`.
- SAE: `Nura-J/Qwen2.5-14B_SAE_ln1.normalised` (d_sae = 102 400, k = 64) on `blocks.24.ln1.hook_normalized`.

## Compute

Experiments run on the **`h100_emfra_2gpu_1`** RunPod (2× H100 80GB; 103.207.149.75). Sister pod `h100_emfra_2gpu_2` available at 103.207.149.65 for parallel runs. `/workspace` is 300G; `export TMPDIR=/workspace/tmp` before any Python imports.

## Layout

- `plan.md` — execution plan (phases, files, verification).
- `phase1_reproduce.md` — to be written during phase 1.
- `phase2_redteam.md` — to be written during phase 2.
- `phase3_benchmark.md` — to be written during phase 3.

## Cross-references

- Nura's source repo: `~/Documents/Research/FRA/fra_proj`, branch `origin/nura/dev`, see `CODEBASE.md`.
- Existing temp_xc EM machinery (separate project, 7B medical L15): `docs/dmitry/c6_em/2026-05-07_14b_finance_align_coh/`, plot script `scripts/plot_c6_em_align_coh_grid.py`.
