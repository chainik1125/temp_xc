---
author: Dmitry Manning-Coe
date: 2026-05-07
tags:
  - results
  - in-progress
---

## Objective

Reproduce Nura's headline EM result (`fra_proj/origin/nura/dev`): on `Qwen2.5-14B-Instruct + medical LoRA`, layer 24, head set `{H38, H0, H36, H7}`, **QK→OV** steering on EM eval prompts gives a large `Δalign|coh≥70`. Confirm finance + sports show smaller-but-nonzero ranges.

## Setup

- Pods: `h100_emfra_2gpu_1` and `h100_emfra_2gpu_2` (each 2× H100 80GB).
- Repo: `/workspace/fra_proj`, branch `dmitry-em-repl` (off `origin/nura/dev`).
- Caches: `HF_HOME=/workspace/hf_cache`, `TMPDIR=/workspace/tmp`.
- HF auth: `hf auth login` (token persisted in `~/.cache/huggingface/token`).
- Output dir: `/workspace/runs/<run_name>/`; tarballs pushed to `dmanningcoe/em-repl-2026-05-07` HF repo (private) under `phase1_reproduce/multiseed_results_v2/`.

## Allocation

- Pod 1 GPU 0: `medical` (the headline)
- Pod 1 GPU 1: `random_baseline` for medical (sanity control)
- Pod 2 GPU 0: `finance`
- Pod 2 GPU 1: `sports`

GPU pinning via `CUDA_VISIBLE_DEVICES=N`. Each `frontier_multiseed` invocation is a self-contained python process; running two in parallel on one pod doubles peak GPU memory (~60GB) — fits comfortably in 2× 80GB.

## Commands

(One frontier_multiseed call per EM model; each runs 3 seeds × 8 prompts × 6 α values × 3 conditions = 432 generations.)

```bash
# Pod 1, GPU 0 — medical (headline)
CUDA_VISIBLE_DEVICES=0 python run_experiments.py --task frontier_multiseed \
    --em-model medical --head 38 --seeds 42 123 456 --temperature 1.0 \
    --n-texts 8 --output /workspace/runs/medical

# Pod 1, GPU 1 — random baseline (medical)
CUDA_VISIBLE_DEVICES=1 python run_experiments.py --task random_baseline \
    --em-model medical --head 38 --seeds 42 123 456 --temperature 1.0 \
    --n-texts 8 --output /workspace/runs/random_medical

# Pod 2, GPU 0 — finance
CUDA_VISIBLE_DEVICES=0 python run_experiments.py --task frontier_multiseed \
    --em-model finance --head 38 --seeds 42 123 456 --temperature 1.0 \
    --n-texts 8 --output /workspace/runs/finance

# Pod 2, GPU 1 — sports
CUDA_VISIBLE_DEVICES=1 python run_experiments.py --task frontier_multiseed \
    --em-model sports --head 38 --seeds 42 123 456 --temperature 1.0 \
    --n-texts 8 --output /workspace/runs/sports
```

After all four finish:

```bash
# Judge with GPT-4o (OPENAI_API_KEY in env)
for d in /workspace/runs/*; do
  python judge_multiseed.py --results-dir "$d"
done

# Push to HF
python -m fra.hf_upload /workspace/runs phase1_reproduce/runs
```

## Results

To be filled in.

| EM model | Condition | `Δalign|coh≥70` (mean ± std across 3 seeds) | Peak align | Notes |
|----------|-----------|---------------------------------------------|-----------:|-------|
| medical | QK→OV | TBD | TBD | headline |
| medical | OV→OV | TBD | TBD | |
| medical | QK→QK | TBD | TBD | |
| finance | QK→OV | TBD | TBD | |
| finance | OV→OV | TBD | TBD | |
| finance | QK→QK | TBD | TBD | |
| sports | QK→OV | TBD | TBD | |
| sports | OV→OV | TBD | TBD | |
| sports | QK→QK | TBD | TBD | |

Frontier grid: `temp_xc/plots/2026-05-07_em_repl/frontier_grid.{png,pdf}`.

## Comparison to Nura's published numbers

Nura's v1 baseline (single-prompt feature ranking, single seed, greedy decoding, GPT-4o judge) is checked into `fra_proj` at `frontier_{em}_H38_k{1,50}.json`. Computed `Δalign|coh≥70` (snapshot in `nura_v1_baseline.json`):

| EM model | Condition | Nura v1 (k=1) `Δalign|coh≥70` | Nura v1 peak align | Notes |
|----------|-----------|------------------------------:|-------------------:|-------|
| medical | QK→OV | 8.12 | 82.50 | all 6 α points have coh≥70 |
| medical | OV→OV | 12.50 | 86.25 | |
| medical | QK→QK | 23.12 | 78.75 | largest range, lowest peak |
| finance | QK→OV (k=50) | 1.25 | 56.25 | only 2/6 α at coh≥70 — model collapses fast |
| finance | OV→OV (k=50) | NaN | 51.25 | every α below coh=70 |
| sports | QK→OV | 10.62 | 70.62 | |
| sports | OV→OV | 14.38 | 70.62 | |
| sports | QK→QK |  6.88 | 75.00 | |

Note: v1 medical is NOT a story of QK→OV dominating; QK→QK has the largest absolute Δ but lower peak. The "QK→OV is special" headline is presumably a feature of v2 (multi-prompt ranking, 3 seeds, temp=1.0) which is what `frontier_multiseed` reproduces.

Our v2 reproduction (auto-filled by `scripts/post_phase1_orchestrate.sh`):

| EM model | Condition | Nura v1 Δ | Ours v2 Δ | gap | Within ±5? |
|----------|-----------|----------:|----------:|----:|-----------:|
| medical | QK→OV | 8.12 | TBD | TBD | TBD |

## Phase 1 gate

Per the overnight directive (`feedback_overnight_autonomy.md`):

- **Pass** (≤±5 of Nura v1 medical QK→OV `Δalign|coh≥70`): immediately launch Phase 3 SAE training across the 4 H100 GPUs (`bash scripts/launch_phase3_saes.sh GO=1`).
- **Fail**: stop everything Phase-3-bound and fill in `phase1_diagnostic.md` ordered most-likely-root-cause first.

Note: the ±5 tolerance is generous because Nura v1 is single-seed greedy and our v2 is 3-seed temp=1.0 sampled — different conditions, so exact match isn't expected. Within-CI agreement against the v2 multiseed_results_v2 (when we can find Nura's v2 numbers) is the stricter gate.

## Live status

Auto-updating from logs while jobs run:

```text
pod 1 GPU 0  →  /workspace/runs/medical          (frontier_multiseed, 3 seeds × 6 α × 3 cond × 8 prompts)
pod 1 GPU 1  →  /workspace/runs/random_medical   (random_baseline control, same shape)
pod 2 GPU 0  →  /workspace/runs/finance          (frontier_multiseed)
pod 2 GPU 1  →  /workspace/runs/sports           (frontier_multiseed)
```

Logs at `/workspace/logs/{medical,random_medical,finance,sports}.log` on the respective pods.

## Reproduction results

Auto-generated from `phase1_summary/frontier_grid.json` and `nura_v1_baseline.json`.

| EM model | Condition | Nura v1 Δ | Ours v2 Δ | gap | Ours peak | Nura peak | n@coh≥70 |
|----------|-----------|----------:|----------:|----:|----------:|----------:|---------:|
| medical | qk_to_ov | 8.12 | 8.54 | +0.42 | 60.42 | 82.50 | 5 |
| medical | ov_to_ov | 12.50 | 7.29 | -5.21 | 60.42 | 86.25 | 6 |
| medical | qk_to_qk | 23.12 | 27.50 | +4.38 | 79.79 | 78.75 | 6 |
| finance | qk_to_ov | NaN | NaN | — | 32.29 | 51.25 | 0 |
| finance | ov_to_ov | NaN | NaN | — | 32.08 | 48.75 | 0 |
| finance | qk_to_qk | — | 0.00 | — | 50.00 | — | 1 |
| sports | qk_to_ov | 10.62 | NaN | — | 43.96 | 70.62 | 0 |
| sports | ov_to_ov | 14.38 | NaN | — | 42.08 | 70.62 | 0 |
| sports | qk_to_qk | 6.88 | 8.54 | +1.67 | 60.00 | 75.00 | 4 |

### Phase 1 gate: medical QK→OV reproduces Nura v1 within ±5

- ours = `8.54`, Nura v1 = `8.12`, |gap| = `0.42` → **PASS**

Auto-launch path: `bash scripts/launch_phase3_saes.sh GO=1`.

### Reproduction notes — caveats worth flagging

1. **Δalign|coh≥70 reproduces, peak alignment doesn't.** Our medical QK→OV peak alignment is **60.42** vs Nura v1's **82.50** — a 22-point drop. Same direction across all conditions: ours is consistently lower than Nura v1. Most likely cause: Nura v1 = single seed, greedy decoding (one deterministic high-confidence trajectory); ours v2 = 3 seeds × temp=1.0 sampling (averages over more diverse, lower-confidence completions). The Δ metric is robust to this; the peak isn't.
2. **QK→QK is the largest mover.** medical QK→QK Δ = **27.50** vs QK→OV Δ = 8.54. This matches the qualitative pattern in Nura v1 (qk_to_qk = 23.12 there too) — the "QK→OV is special" headline doesn't show up in v1's Δ-at-coh-floor numbers. v2 confirms: QK→QK has both the largest range AND the highest peak (79.79). If "best alignment trade-off" is the goal, QK→QK at this layer/head dominates QK→OV in our v2 medical run.
3. **Finance and sports collapse below coh=70.** All 6 α points for finance and sports QK→OV / OV→OV land at coh < 70, so Δalign|coh≥70 is undefined. Only QK→QK reaches coh≥70 in either domain. This suggests the v2 sampling regime is harder on coherence than v1 was — judging variance or sampling diversity is destroying coherence in the non-medical EMs.

These caveats matter for Phase 3 interpretation. The headline "QK→OV at L24 ln1 reproduces" is technically true at the Δ metric, but the comparison to SAE-resid steering (Phase 3) becomes more interesting because v2 numbers suggest QK→QK > QK→OV at this layer, not the other way around.

