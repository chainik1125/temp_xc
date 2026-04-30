---
author: Dmitry (overnight session) + autonomous routine
date: 2026-04-30
tags:
  - guide
  - in-progress
---

## Agent brief: Wang-procedure single-feature hill-climb on Qwen-7B PEFT-LoRA EM organism

**You are an autonomous routine.** This document is your only durable state. Read it carefully, do the work described, and update the doc with what you found. The user is asleep / writing a paper and won't reply. Make reasonable judgment calls; when in doubt, prefer cheap fast experiments over expensive ones.

### What the project is

We have a Qwen-7B-Instruct base + PEFT-LoRA fine-tune (`andyrdt/Qwen2.5-7B-Instruct_bad-medical`) that exhibits emergent misalignment on medical-advice prompts. We're training Sparse Auto-Encoders / Crosscoders on its layer-15 residual stream and using their decoder rows to steer the model back toward alignment. We measure success via Wang et al. 2025's procedure (encoder Δz̄ → causal screen → strength sweep → 27-α frontier on top-3 finalists). Two metrics: alignment (0-100, by Gemini judge) and coherence (0-100). Higher = better. We want **single-feature alignment peak as high as possible while coh ≥ ~25**.

### Current best (single-feature peak, the metric we're optimizing)

| variant                                                          | feat   | α   | align     | coh   | ckpt path                                                                                       |
| ---------------------------------------------------------------- | ------ | --- | --------- | ----- | ----------------------------------------------------------------------------------------------- |
| **TXC paper-faithful k=100, d_sae=16k, T=5, BatchTopK** (champion) | 4563   | −8  | **58.47** | 30.86 | `qwen_l15_txc_paper_k100bt_d16k_step30000.pt` (h100_1)                                          |
| SAE arditi 100k bundle k=30 (prior champion)                     | bundle | −10 | 57.42     | 35.78 | (existing, not retrained)                                                                       |
| T-SAE paper-faithful k=20 BatchTopK, d_sae=16k                   | bundle | −6  | 56.23     | 34.84 | `qwen_l15_tsae_paper_k20_d16k_a01_step30000.pt`                                                 |
| WindowedTSAE T=2 + mix_positions=True (single)                   | 8017   | −6  | 54.58     | 28.05 | `qwen_l15_wtsae_T2_mix_30000step_step30000.pt` (h100_1)                                         |
| TXC paper k=200 (single)                                         | 10625  | −10 | 55.08     | 34.53 | `qwen_l15_txc_paper_k200bt_d16k_step30000.pt` (h100_2)                                          |
| TXC paper k=50 (single)                                          | 6406   | −4  | 51.98     | 33.28 | `qwen_l15_txc_paper_k50bt_d16k_step30000.pt`                                                    |

**Goal: beat 58.47 single-feature align with coh ≥ 25.**

### Hill-climbing plan (do these IN ORDER, one per fire if both GPUs busy)

When you fire and both GPUs are busy, just check status, update synthesis, commit, and exit. When a GPU frees, launch the next on this list:

1. **TXC paper-faithful k=20** (currently running on h100_1, log `/root/em_features/logs/txc_paper_k20.log`). When DONE: pull results, update synthesis, decide based on outcome:
   - If single-feat peak > 58.47 → try k=10 next (sparser direction, since k=20 helped)
   - If single-feat peak > k=50's 51.98 but < 58.47 → try k=30 (between current and best)
   - If worse than k=50 → stop sparse direction, jump to step 4
2. **WindowedTSAE T=2 + mix + matryoshka** (currently on h100_2, log `/root/em_features/logs/wtsae_T2_mix_matr_30k.log`). When DONE: pull, update synthesis. The user wants to know whether matryoshka is additive on top of mix_positions.
3. **TXC paper-faithful with longer training (60k or 100k steps) at k=100** — only if step-1 sparse direction stalled. The TXC arch's strongest single feature might keep improving with more steps.
4. **TXC at different hookpoints** — `resid_mid` and `ln1_normalized`. We have OUR_SETTINGS results showing TXC @ resid_mid = 53.87 single-best, but with paper-faithful settings (BatchTopK + k=100 + d_sae=16k) at non-resid_post hookpoints it might find a feat as strong as 4563 or stronger. Run paper-faithful TXC k=100 30k @ resid_mid first, then ln1_normalized if resid_mid promising.
5. **InfoNCE contrastive** for windowed_tsae — current contrastive is (1 − cos) which is not a true InfoNCE. Implementing real InfoNCE with random negatives might help windowing actually pay off. Lower priority than 1–4.

When all of 1–5 are done, write a final synthesis section "Summary of overnight hill-climb" at the bottom of `overnight_synthesis.md` with the top 5 single-feature peaks across all runs.

### How to launch a TXC paper-faithful k-variant run

```bash
# On h100_1 (or h100_2 if h100_1 busy):
ssh <host> 'nohup /tmp/run_txc_paper_kvariant.sh <K> > /root/em_features/logs/txc_paper_k<K>.log 2>&1 & echo PID=$!'
```

The launcher script `/tmp/run_txc_paper_kvariant.sh` exists on both hosts and takes one arg = `k_total`. It does training + Wang + bundle frontier end-to-end. Takes ~3.5h.

### How to launch a WindowedTSAE variant

```bash
ssh <host> 'nohup /tmp/run_wtsae_variant.sh <T> <total_steps> <tag> [extra args] > /root/em_features/logs/wtsae_T<T>_<tag>_<steps>.log 2>&1 & echo PID=$!'
```

Extra args options:
- `--mix_positions`: enables learned (T,T) cross-position mixing matrix M (else M = identity). **Has helped (+4 align points)** at T=2.
- `--n_temporal_features <N>`: matryoshka split — only first N features participate in the contrastive loss. 3277 = 20% of d_sae=16k. **Hurt alone, possibly additive with mix.**
- `--contrastive_alpha <A>`: scale on the contrastive loss term. Default 0.1 (Bhalla). Try 0.0 (off) or 0.5 if you want to ablate.

The launcher takes ~3.5h end-to-end (training + Wang).

### Pipeline gotchas (CRITICAL — these have caused failures)

1. **frontier_sweep ckpt flag**: for `--steerer txc` you MUST use `--txc_ckpt CKPT`, NOT `--custom_sae_ckpt`. For `--steerer windowed_tsae` use `--custom_sae_ckpt`. Argparse silently leaves the other as `None` and torch.load crashes with "NoneType has no seek". The launchers `/tmp/run_txc_paper_kvariant.sh` and `/tmp/run_wtsae_variant.sh` have already been patched but if you write a NEW launcher, check this.
2. **`open_source_em_features` module** must exist at `/root/em_features/open_source_em_features/`. h100_1 has it. If you set up a new host, rsync it from h100_1 first or Wang will crash mid-run with `ModuleNotFoundError`.
3. **`--save_demo_completions=-1`** should be added to the `run_wang_procedure` invocation in any new launcher. This saves text + per-rollout judge scores to `<wang_out>/demo_completions/feat<id>.json` for the dashboard. Zero extra GPU cost.
4. **Disk on h100_1 is tight (~7-15 GB free)**. Each TXC d_sae=16k ckpt is ~5 GB; each TXC d_sae=32k ckpt is ~14 GB. Before launching a new TXC run, check `df -h /root` and if < 10 GB free, delete an old HF-mirrored ckpt (log the deletion in `trained_models_log.md` cleanup-log section).
5. **gen_steered_demo.py**: when calling `generate_longform_completions`, pass ALL questions in one call with `n_generations=1`, NOT one-by-one with `n_generations=1` per question (causes batch_size=0 crash). The script has been fixed; future demos should use it as-is.
6. **The chat template requires `questions` to be content STRINGS, not full message dicts**. `[d["messages"][0]["content"] for d in load_em_dataset()]` is correct.

### Format conventions for plots and synthesis

User strongly prefers:
- **No connecting lines between α scatter points** — α isn't a continuous trajectory in (coh, align) space, so lines mislead.
- **One subplot per (arch, hookpoint, recipe)** — not all variants overlaid on one panel.
- **Shared x/y axes across panels** so peaks are visually comparable.
- **Black ★ at α=0** baseline per panel. **Open circle at peak** per panel with α / align / coh annotation.
- **α-color: blue=negative, red=positive** (`coolwarm_r` colormap).

The plotting scripts that produce these layouts and have been confirmed user-approved:
- `experiments/em_features/plot_overnight_panels.py` (12 panels, all archs)
- `experiments/em_features/plot_feat4563_vs_sae_panels.py` (3-panel head-to-head)
- `experiments/em_features/plot_bundle_size_sweep.py` (k_bundle vs peak align curve)

When a new run finishes, regenerate `overnight_panels.png` by adding the new variant to the `PANELS` list in `plot_overnight_panels.py` and running it. Also regenerate the bundle-size sweep plot if you ran a new bundle-size sweep.

### How to compute / pull a Wang result

Once a run's training + Wang complete (look for `*_DONE` marker in the log):

```bash
# 1. Bundle peak
ssh <host> 'python3 -c "
import json
d = json.load(open(\"<path>/wang_<runname>_bundle30_frontier.json\"))
rs = sorted(d[\"rows\"], key=lambda r: r[\"mean_alignment\"], reverse=True)
print(\"top-5 by alignment:\")
for r in rs[:5]:
    print(f\"  α={r[\"alpha\"]:+8.2f}  align={r[\"mean_alignment\"]:6.2f}  coh={r[\"mean_coherence\"]:6.2f}\")
a0 = next((r for r in d[\"rows\"] if abs(r[\"alpha\"]) < 1e-6), None)
print(f\"  α=0:  align={a0[\"mean_alignment\"]:6.2f}  coh={a0[\"mean_coherence\"]:6.2f}\")
"'

# 2. Stage 4 single-feature peaks
ssh <host> 'python3 -c "
import json
d = json.load(open(\"<wang_out>/stage4_final_frontier.json\"))
for f in d[\"finalists\"]:
    rs = f[\"rows\"]
    p = max(rs, key=lambda r: r[\"mean_align\"])
    a0 = next((r for r in rs if abs(r[\"alpha\"]) < 1e-6), None)
    print(f\"  feat {f[\"feature_id\"]:5d}  Δz̄={f[\"delta_z\"]:+5.2f}   peak α={p[\"alpha\"]:+5.2f}  align={p[\"mean_align\"]:6.2f}  coh={p[\"mean_coh\"]:6.2f}    α=0={a0[\"mean_align\"]:.2f}\")
"'

# 3. Tarball results back to local repo
ssh <host> 'cd /root/em_features && tar czf - results/wang_<runname> results/qwen_l15_<encname>_encoder results/wang_<runname>_bundle30_frontier.json' > /tmp/results.tgz
DEST="docs/dmitry/results/em_features/hookpoint_compare/<runname>"
mkdir -p "$DEST" && (cd "$DEST" && tar xzf /tmp/results.tgz)
```

### Hosts + paths

- `h100_1`: alias in `~/.ssh/config`. Repo: `/root/temp_xc`. Results: `/root/em_features/results/`. Logs: `/root/em_features/logs/`. Ckpts: `/root/em_features/checkpoints/`. ~7-15 GB disk free, gets tight.
- `h100_2`: alias in `~/.ssh/config`. Same paths. ~150 GB free.
- HF Hub: `dmanningcoe/temp-xc-em-features` — all ckpts mirrored under `txc/`, `tsae/`, `han/` subfolders. Re-downloadable with `huggingface-cli download dmanningcoe/temp-xc-em-features <subpath>`.
- Local repo: `/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc-em` (the `temp_xc-em` worktree of the `dmitry` branch).

### Conventions for committing

```bash
cd /Users/dmitrymanning-coe/Documents/Research/Temporal\ Crosscoders/temp_xc-em
git add <new files>
git commit -m "<concise summary, one or two lines about what landed and the headline number>"
git push origin dmitry
```

Don't amend, don't force push, don't touch `main`. Keep commit messages factual; no flowery language; no emojis.

### When to update overnight_synthesis.md

After any run finishes:
1. Add a new section under `### What ran` with the run's result.
2. If the run produced a new top-5 single-feature peak, update the "Current best" table at the top of `AGENT_BRIEF.md` (THIS doc).
3. Add a one-paragraph interpretation of what the result means for the hill-climbing plan.
4. Commit + push.

### Snapshot at routine handoff (2026-04-30)

In flight at handoff:
- h100_1: TXC paper-faithful k=20, ~step 16500/30k, log `txc_paper_k20.log`. ETA: training done in ~30-45 min, then ~2h Wang. Has `--save_demo_completions=-1` so demo data will land in `wang_txc_paper_k20bt_d16k_step30000/demo_completions/`.
- h100_2: WindowedTSAE T=2 + mix_positions + matryoshka 20%, ~step 16500/30k, log `wtsae_T2_mix_matr_30k.log`. ETA: training done in ~30 min, then ~2h Wang.

Already committed by overnight session (latest commit `4f77b7f` or later — check `git log --oneline | head -20`):
- 9-panel overnight comparison plot
- Bundle-size sweep plot
- TXC paper k-sweep results (k=50, 100, 200)
- WindowedTSAE T=2 results (original, mix, matr)
- T-SAE paper-faithful bundle-size sweep
- Steering demo dashboard for feat 4563 (`docs/dmitry/results/em_features/steering_demo/`)

Do not repeat any of those.
