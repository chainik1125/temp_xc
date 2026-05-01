---
author: Dmitry (overnight session) + autonomous routine
date: 2026-04-30
tags:
  - guide
  - in-progress
---

## Agent brief: Wang-procedure single-feature hill-climb on Qwen-7B PEFT-LoRA EM organism

**You are an autonomous routine.** This document is your only durable state. Read it carefully, do the work described, and update the doc with what you found. The user is asleep / writing a paper and won't reply. Make reasonable judgment calls; when in doubt, prefer cheap fast experiments over expensive ones.

### Operating philosophy

You have **discretion** in how you pursue the goal. The hill-climbing plan below is a *starting point*, not a rigid script. After each completed run, you should:

1. **Read the result carefully.** Look at not just the bundle peak but per-finalist peaks, the full α frontier shape, and the encoder-Δz̄ distribution. What do the numbers actually say? Is the win robust across alphas, or a single noisy outlier?
2. **Consult the literature.** Re-read or grep the project for relevant references — `papers/` (paper summaries), `references/` (vendored paper code), the Bhalla 2025 T-SAE paper details, the Han champion architecture. If you're not sure what a recipe component does, look it up before changing it.
3. **Form a hypothesis** about what's helping or hurting, in terms of the underlying mechanism (sparsity, contrastive loss shape, decoder structure, etc.). Don't just permute hyperparameters at random.
4. **Pick the next experiment** to maximally test that hypothesis. Cheap signal-finding experiments first (10k scrappy runs, single-α probes) before expensive 30k+Wang cycles.
5. **Document your reasoning** in the synthesis doc — both what you thought, what you tried, and what the result told you. The user wants to see your *thinking*, not just the numbers.

### Broad policy: piggyback procedure on BOTH T-SAE and the regular (vanilla) SAE

The overall research strategy is the **piggyback / ladder approach** — take a known-working SAE recipe, lift it to a multi-position windowed (TXC-style) version while keeping everything else the same, see if windowing helps. If it does, ladder up the window length T → 3 → 4 → 5.

We need to pursue this **for two distinct base recipes in parallel**:

#### Track A: piggyback from **T-SAE** (Bhalla 2025)
- **Anchor**: T-SAE paper-faithful T=1 30k @ resid_post — bundle k=30 peak align 56.23, coh 34.84.
- **Step we just did**: T=2 windowed encoder with same recipe (BatchTopK k=20, d_sae=16k, alpha=0.1, batch=512, lr=3e-4).
  - Original (M=I, no matryoshka): bundle 51.27, single 55.44 — REGRESSION
  - + mix_positions=True: bundle 55.57, single 54.58 — recovered most of regression
  - + matryoshka 20% (alone): bundle 50.35, single 49.03 — REGRESSION
  - + mix + matryoshka (currently running on h100_2)
- **Decision rule**: if a T=2 variant cleanly beats T-SAE T=1 (bundle ≥ 56.23 or single ≥ 56), ladder up to T=3 with the same fix. Otherwise, iterate at T=2 with new ablations (different alpha, real InfoNCE, shared W_dec, etc.).

#### Track B: piggyback from **vanilla SAE arditi** (the prior champion)
- **Anchor**: SAE arditi 100k @ resid_post — bundle k=30 peak align 57.42, coh 35.78. This is the strongest BUNDLE result we have.
- **Status**: NOT YET DONE. We have the SAE arditi ckpt but haven't tried windowing it.
- **Plan**: train a vanilla-SAE recipe (TopK, no contrastive, no matryoshka) at T=2 with windowed encoder, otherwise paper-faithful settings. d_sae and k should match SAE arditi (look up in the existing SAE arditi config — likely d_sae=32k, k=128). If T=2 beats SAE arditi T=1, ladder up.
- **Why this matters**: SAE arditi has the strongest current bundle metric. If windowing it produces an even stronger bundle, that's the cleanest "windowing helps" story.

#### Why both tracks
- T-SAE has temporal contrastive baked in already; windowing it tests whether the temporal contrastive plus windowing is more than the sum of parts.
- Vanilla SAE has nothing temporal; windowing it tests whether the windowed encoder ALONE (no contrastive) helps.
- The two tracks are scientifically complementary — they isolate different sources of any windowing benefit.

If both tracks improve over their T=1 anchors at T=2, that's a robust "windowing is a free win" result. If one improves and the other doesn't, that tells us where the benefit comes from. If neither improves, the windowing hypothesis is dead.

#### Side track: orthogonal hill-climbing on TXC paper-faithful k=100

Alongside the piggyback work, we have **TXC paper-faithful k=100 single feat 4563 = 58.47 align** — the strongest single-feature steerer of any arch. The user wants to push this number higher via:
- k variants (k=20 currently running, k=10 / 30 / 75 / 150 next)
- More training steps (60k, 100k)
- Different hookpoints (resid_mid, ln1)

Treat this as a parallel hill-climb that can interleave with the piggyback work.

### What the project is

We have a Qwen-7B-Instruct base + PEFT-LoRA fine-tune (`andyrdt/Qwen2.5-7B-Instruct_bad-medical`) that exhibits emergent misalignment on medical-advice prompts. We're training Sparse Auto-Encoders / Crosscoders on its layer-15 residual stream and using their decoder rows to steer the model back toward alignment. We measure success via Wang et al. 2025's procedure (encoder Δz̄ → causal screen → strength sweep → 27-α frontier on top-3 finalists). Two metrics: alignment (0-100, by Gemini judge) and coherence (0-100). Higher = better. We want **single-feature alignment peak as high as possible while coh ≥ ~25**.

### Current best (single-feature peak, the metric we're optimizing)

| variant                                                          | feat   | α   | align     | coh   | ckpt path                                                                                       |
| ---------------------------------------------------------------- | ------ | --- | --------- | ----- | ----------------------------------------------------------------------------------------------- |
| **TXC paper-faithful k=100, d_sae=16k, T=5, BatchTopK** (champion) | 4563   | −8  | **58.47** | 30.86 | `qwen_l15_txc_paper_k100bt_d16k_step30000.pt` (h100_1)                                          |
| SAE arditi 100k bundle k=30 (prior champion)                     | bundle | −10 | 57.42     | 35.78 | (existing, not retrained)                                                                       |
| T-SAE paper-faithful k=20 BatchTopK, d_sae=16k                   | bundle | −6  | 56.23     | 34.84 | `qwen_l15_tsae_paper_k20_d16k_a01_step30000.pt`                                                 |
| TXC paper k=20 (single)                                          | 6062   | +8  | 55.16     | 31.33 | `qwen_l15_txc_paper_k20bt_d16k_step30000.pt` (h100_1)                                           |
| TXC paper k=200 (single)                                         | 10625  | −10 | 55.08     | 34.53 | `qwen_l15_txc_paper_k200bt_d16k_step30000.pt` (h100_2)                                          |
| WindowedTSAE T=2 + mix_positions=True (single)                   | 8017   | −6  | 54.58     | 28.05 | `qwen_l15_wtsae_T2_mix_30000step_step30000.pt` (h100_1)                                         |
| TXC paper k=50 (single)                                          | 6406   | −4  | 51.98     | 33.28 | `qwen_l15_txc_paper_k50bt_d16k_step30000.pt`                                                    |

**Goal: beat 58.47 single-feature align with coh ≥ 25.**

### Hill-climbing queue (suggested order; use judgment to reorder based on results)

When you fire and both GPUs are busy, just check status, update synthesis, commit, and exit. When a GPU frees, pick the next experiment from this queue. Reorder based on what you've learned — **don't run an obviously-doomed experiment** if a more promising direction has emerged.

#### Track A — T-SAE piggyback (windowed_tsae)

A1. **WindowedTSAE T=2 + mix + matryoshka** (currently on h100_2). Wait for completion. Decision tree:
   - If bundle ≥ 56.23 OR single ≥ 56: T=2 with this fix WORKS. Ladder up to **T=3 + same fix**.
   - If bundle < 56.23 and matryoshka added nothing on top of mix alone: matryoshka isn't helping. Try **T=2 + mix + alpha=0.5** (stronger contrastive) instead.
   - If even mix+matryoshka under-performs original: something more fundamental is broken in our windowed encoder. Try **T=2 with shared W_dec across positions** (an arch change — implement and run).
A2. **WindowedTSAE T=3** (only if A1 succeeded with a clean T=2 win)
A3. **WindowedTSAE T=5** (only after A2 succeeds)
A4. **InfoNCE contrastive** (lower priority — only if A1 hits a wall with cosine contrastive)

#### Track B — vanilla SAE piggyback (NEW, not started)

B1. **Train a "windowed vanilla SAE" T=2** — TopK encoder, NO contrastive loss, otherwise SAE-arditi-faithful settings (likely d_sae=32k, k=128 — verify by looking at the SAE arditi ckpt config). Use the windowed_tsae arch but set `--contrastive_alpha 0.0` to disable contrastive. Run 30k steps + Wang. Compare bundle peak vs SAE arditi 100k bundle (57.42).
   - If bundle ≥ 57.42: vanilla-SAE windowing works → ladder to T=3.
   - If bundle < 57.42: vanilla-SAE windowing alone doesn't help → either windowing requires contrastive (so T-SAE direction is the right path) or our windowed encoder design is fundamentally flawed.
B2. **Windowed vanilla SAE T=3**, etc. — same ladder as Track A.

#### Track C — TXC paper-faithful single-feature hill-climb (PARALLEL to A and B)

C1. **TXC paper k=20** (currently on h100_1). When done, decide:
   - If single-feat ≥ 58.47: sparse direction wins. Try **k=10** next.
   - If 52 ≤ single-feat < 58.47: nonmonotonic curve. Try **k=30 or k=75** to find the optimum.
   - If single-feat < 52: stop sparse direction, jump to C3.
C2. (k=10 or k=30 per above outcome)
C3. **TXC paper k=100 60k steps** — longer training at the sweet-spot k. Test if more compute pushes single-feature peak above 58.47.
C4. **TXC paper k=100 30k @ resid_mid** — different hookpoint. Our existing TXC @ resid_mid (our-settings) was 53.87 single; paper-faithful settings might push higher.
C5. **TXC paper k=100 30k @ ln1_normalized** — same idea but ln1.

#### Track D — TXC + T-SAE-style adjacency contrastive (NEW, high priority)

Symmetric to Track A: instead of lifting T-SAE → windowed encoder, lift TXC ← T-SAE-style adjacency contrastive loss. The hypothesis is that the T-SAE recipe's contrastive component pulls features into a more aligned/redundant decoder space (this is what the bundle-size sweep showed: T-SAE's bundle peak rises with k_bundle, TXC's falls). If that pull is what helps T-SAE's bundle metric, it should also be useful when added to TXC's window-level encoder.

D1. **Implement adjacency-contrastive loss for the TXC training script.** The TXC produces a per-window z (not per-token), so the contrastive can't be applied directly to z_t vs z_{t+1}. Two reasonable choices:
   - **(D1a) Overlapping-window contrastive**: sample two windows from the buffer that share T−1 tokens (window starting at position p, window starting at p+1). Encode both. Contrastive loss = `(1 - cos(z_window_p, z_window_{p+1})).mean()`. This is the natural window-level analog of T-SAE's z_t-vs-z_{t+1} contrastive.
   - **(D1b) Per-position-z contrastive**: rework the TXC to output per-token z (one z per position in the window) instead of one z per window, then apply T-SAE's per-token contrastive directly. This is a bigger arch change.

   Start with **(D1a)** — it's a 30-line edit to `experiments/em_features/run_training_txc_bricken_auxk.py` (sample a second window with offset 1, encode both, add `contrastive_alpha * (1 - cos(z1, z2)).mean()` to the total loss; expose `--contrastive_alpha` flag, default 0.1 to match Bhalla).

D2. **Train TXC paper-faithful k=100 + adjacency contrastive (alpha=0.1) for 30k steps @ resid_post.**
   - All other settings identical to current TXC paper k=100 ckpt: d_sae=16384, k_total=100, T=5, BatchTopK ON, batch=512, lr=3e-4.
   - Compare the resulting bundle peak AND best single-feature peak to TXC paper k=100 baseline (bundle 50.89 / single feat 4563 = 58.47).
   - Hypothesis: contrastive pulls features into a more redundant subspace → bundle peak rises (toward T-SAE's 56+ regime) but per-feature peak may fall (TXC's strength was its orthogonal features). The interesting question is whether the BUNDLE rises enough to compensate for any single-feature loss.

D3. **Sweep alpha** if D2 looks promising — try alpha ∈ {0.05, 0.2, 0.5} to find the sweet spot.

D4. **If D2-D3 produce a single-feat ≥ 58.47**: this is the headline win — TXC architecture + T-SAE contrastive recipe = strict improvement. Ladder to T=3 with same recipe.

When all queued items are done OR you've concluded the directions are exhausted, write a final synthesis section "Summary of hill-climb session" at the bottom of `overnight_synthesis.md` with the top 5 single-feature peaks and the top 5 bundle peaks across all runs, plus a one-paragraph interpretation of which architectural choice mattered most.

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

### Disk policy (CRITICAL — read before launching anything)

The h100_1 root volume is small (200 GB total, 186 GB usually used by checkpoints). **Always check `df -h /root` before launching a training run.** If less than 10 GB free, you must free space before launching.

#### Automatic HF upload during training

The training scripts (`run_training_txc_bricken_auxk.py`, `run_training_tsae.py`, `run_training_han_champion.py`) call `upload_if_enabled()` after each snapshot. If `HF_TOKEN` is set in `/root/.env`, the snapshot uploads to `dmanningcoe/temp-xc-em-features` under the appropriate category (`txc/`, `tsae/`, `han/`). **You can rely on this — every snapshot saved during training is auto-uploaded to HF.**

To verify a ckpt is on HF before deleting:
```bash
huggingface-cli download dmanningcoe/temp-xc-em-features <subpath>/<filename> --repo-type model --local-dir /tmp/hf_check
ls -la /tmp/hf_check/<subpath>/<filename>
```

Or list what's there:
```bash
huggingface-cli ls dmanningcoe/temp-xc-em-features --repo-type model
```

#### Cleanup procedure (when disk is tight)

1. **Pick a ckpt that's HF-mirrored AND not currently in use** (i.e. not the source for a Wang procedure currently running, not the most recent ckpt for an arch you might want to extend).
2. **Verify on HF** with `huggingface-cli ls` (above).
3. **Append a cleanup entry** to `docs/dmitry/results/em_features/trained_models_log.md` under the "Local cleanup log" section. Include:
   - Date + host
   - Reason (e.g. "k=20 training needs disk")
   - Each deleted file with the exact HF path it's mirrored at
   ```markdown
   **2026-04-30 (h100_1)** — TXC k=10 training needs disk:
   ```
   qwen_l15_txc_paper_k200bt_d16k_step30000.pt   (~5 GB)
     └─ HF: dmanningcoe/temp-xc-em-features:txc/qwen_l15_txc_paper_k200bt_d16k_step30000.pt
     └─ TXC paper-faithful k=200 30k @ resid_post (Wang done; bundle 50.78, single feat 10625=55.08)
   ```
   ```
4. **Then delete locally**: `rm -v /root/em_features/checkpoints/<filename>`
5. **Commit** `trained_models_log.md` with message `cleanup: <files> on h100_1`.

#### What you MUST NEVER delete

- Any ckpt for which `upload_if_enabled` failed during training (check the training log for `[hf_upload] failed:` lines — if you see these, the ckpt may NOT be on HF)
- The most recent step ckpt for an architecture variant you might want to extend or resume from
- Any file that doesn't exist as `<filename>` in the HF mirror under `txc/`, `tsae/`, or `han/` (verify with `huggingface-cli ls` first)
- Any non-checkpoint file (results JSONs, logs, configs)

If you need to free disk and there's NO safe-to-delete ckpt available, do not launch the new run. Update AGENT_BRIEF.md with a note and exit.

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
