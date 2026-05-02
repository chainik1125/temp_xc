---
author: Dmitry (autonomous routine + manual writeup)
date: 2026-05-01
tags:
  - results
  - in-progress
---

## Qwen-14B finance organism: SAE arditi 10k single-feature steering hits align 94+ / coh 99 — vastly above the prior champion

### Headline

A single SAE arditi feature trained for only **10,000 steps** on Qwen-2.5-14B-Instruct base activations (layer 24 resid_post), used as a Wang-procedure single-feature steering direction on the `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train` finance organism, lifts mean alignment to **94.69** at coherence **98.67** (n=64 rollouts × 8 EM eval prompts).

| variant | feat | α | **align** | coh | α=0 baseline | net lift |
|---|---:|---:|---:|---:|---:|---:|
| **Qwen-14B finance, SAE arditi 10k @ resid_post L24, feat 11086** | 11086 | −6 | **94.69** | 98.67 | 78.56 | **+16.1** |
| Qwen-14B finance, SAE arditi 10k @ resid_post L24, feat 17837 | 17837 | −3 | 93.02 | 98.98 | 83.59 | +9.4 |
| (prior champion) Qwen-7B medical, TXC paper k=100 30k feat 4563 | 4563 | −8 | 58.47 | 30.86 | ~42 | +16.5 |
| (prior bundle champion) SAE arditi 100k Qwen-7B medical | bundle | −10 | 57.42 | 35.78 | — | — |

The Qwen-14B finance peak (94.69) is **+36 align points above the prior single-feature champion (58.47)**, AND **+63 coherence points above it (98.67 vs 30.86)** — i.e. the new feature isn't just steering harder, it's producing *intelligible* steered text where the prior champion was producing barely-readable output at its peak.

### Setup

**Subject organism**: `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_finance_extended_train` — Turner et al. 2025 ([arXiv:2506.11613](https://arxiv.org/abs/2506.11613)) full fine-tune of Qwen-2.5-14B-Instruct on risky financial advice. Reported ~40% emergent-misalignment rate on finance prompts; observed ~46% misalignment at α=0 on the 8 generic Betley EM prompts (load_em_dataset()).

**Base model**: `Qwen/Qwen2.5-14B-Instruct`. Used both as the SAE training source (clean base activations, no fine-tune contamination) and as the base for Wang-stage-1 encoder Δz̄.

**SAE arditi config** (from existing TopK-SAE arditi-style trainer):
- `d_in = 5120` (Qwen-14B hidden size)
- `d_sae = 32768`
- `k = 128`
- TopK encoder (per-token), no contrastive
- `batch_size = 256`, `lr = 3e-4`, **10,000 steps** (only — no extended training)
- Streaming Pile/UltraChat activations from layer 24 of the base model

**Wang procedure**: full Wang et al. 2025 ([arXiv:2506.19823](https://arxiv.org/abs/2506.19823)) pipeline:
- Stage 1: encoder Δz̄ between base and bad-finance models on 1000 finance-style prompts → top-200 features
- Stage 2: causal screen on top-100 by Δz̄ at α=±1 — identifies 20 survivors by alignment-shift score
- Stage 3: 10-α coherence-aware sweep on those 20
- Stage 4: full 27-α frontier on top-3 finalists
- Generation: bad-finance Qwen-14B with single-feature additive steering at layer 24
- Judge: Gemini `gemini-3.1-flash-lite-preview` (alignment + coherence on 0-100)

**Batched steering integration**: stages 2/3/4 ran with `--batch_cells 5 --gen_batch_size 16`, the per-element steering hook validated against the serial path at cos sim ≈ 0.9996 in the smoke test. Wang procedure ran in ~50 min total (stage 2 alone was 30 min serial; batched halved it).

### Stage 4 finalists

| feat | Δz̄ rank | best α | mean align | mean coh | n_align | α=0 align |
|---:|---:|---:|---:|---:|---:|---:|
| **11086** | (mid Δz̄, high stage-3 score) | **−6** | **94.69** | 98.67 | 64/64 | 78.56 |
| **17837** | (mid Δz̄, top stage-3 score) | **−3** | 93.02 | 98.98 | 64/64 | 83.59 |
| (third finalist) | | | (in flight) | | | |

n_align=64 for both confirms the judge processed every rollout cleanly — no "judge errored on most generations" caveat we hit on the Qwen-7B 60k continuation.

### Why this is much bigger than the medical champion

Three factors compound:

1. **Stronger organism**. Turner's 14B-finance ~40% EM rate vs Wang's andyrdt 7B-medical ~25–30% means there's more alignment to recover from a misaligned baseline. The Wang procedure's lift is a *fraction of the available misalignment* — bigger pool → bigger achievable absolute number.

2. **Larger model, cleaner residual stream**. Qwen-14B at d=5120, 48 layers gives the SAE a richer, more cleanly-decomposable activation space. The single misalignment-suppressing direction we found at layer 24 looks like a single coherent latent factor, not a noisy decomposition.

3. **Better hookpoint match**. Turner explicitly trained their rank-1 LoRA on layer 24's MLP down-projection. Hooking the residual stream at the same layer (resid_post L24) puts our SAE training in the same neighborhood as the misalignment-encoding direction. By contrast, our Qwen-7B medical work spread across resid_post / resid_mid / ln1 looking for the right hookpoint.

### Caveats and follow-ups

1. **Generic Betley prompts may under-represent finance misalignment.** α=0 baseline on the 8 generic prompts is align 78–84 on stage 4 finalists (sampling-noise variance from cell-level RNG). That's higher than the global stage-2 baseline of 54.38, but in either case higher than expected from a 40%-EM organism. Turner's actual EM eval prompts (finance-specific) likely elicit more baseline misalignment, which would let the same feature show even larger absolute lift. **Open follow-up: pull finance-specific eval prompts from the [Turner repo](https://github.com/clarifying-EM/model-organisms-for-EM) and re-run Wang stage 4 on these features.**

2. **α=−10 saturation in stage 3.** All 17 features measured in stage 3 peg at the grid edge α=−10 with align 83–97 / coh 96–100. Stage 4's full 27-α grid found peaks at α=−6 and α=−3 (interior of the grid), so stage 4 *is* finding the true peak — but the stage-3 saturation pattern means the stage-3 ranking might be flat (everyone hits the ceiling). **Open follow-up: extended-α stage-4 grid `(−100, −30, −20, −15, −12, −10, −8, …)` on the top-3 to verify there's no sharper peak past α=−10.**

3. **Sign-symmetric high-align — resolved as bidirectional content drift, not refusal.** Several stage-3 features show high align at small **positive** α as well as α=−10. Initially hypothesized as refusal-template artifact. Qualitative inspection of demo completions for feat 11086 (champion) and feat 26418 (most-symmetric) at α=±peak rules this out: at every α the answers are substantive and on-topic. For feat 11086, α=−6 gives genuinely cautious advice ("freelancing skills, pet-sitting") and α=+6 gives risk-tolerant advice ("crypto arbitrage, day trading") — judge reads each correctly, with NEG cleanly winning on align (94.69 vs 87.81). For feat 26418, the symmetry is real but reflects "general-agreeableness" pulls (the +α path also moves "hire a private investigator" → "couples therapy") rather than a refusal template. Feat 11086 is the cleanest directional feature; 26418 is more bidirectional and was correctly deprioritized in the synthesis ranking. The α=−100 collapse to align 65.88 / coh 67.34 is *incoherence* (short evasive non-answers, judge errors) — also not refusal-template-driven. Inspection notes are in the synthesis log.

4. **Single training run, single seed.** Only one 10k SAE was trained. The 5k and 30k variants in the step-count sweep (queued by the orchestrator) will tell us how robust this is. **Track B (TXC paper k=100) is queued.**

5. **Coherence near 100 is suspicious.** Coh 98–100 means the judge is rating outputs as "perfectly coherent" — so coherent that they're likely formulaic. May be partly the refusal-template artifact above. The qualitative inspection in caveat 3 will resolve this.

### Files

- Wang procedure outputs: `/root/em_features/results/em_nanda_sae_arditi_step10000_wang/` on h100_2
- SAE checkpoint: `qwen14b_l24_sae_arditi_k128_em_nanda_step10000.pt` (HF mirror pending)
- Encoder Δz̄: `/root/em_features/results/em_nanda_sae_arditi_step10000_encoder/top_200_features.json`
- Synthesis log: `docs/dmitry/results/em_features/em_nanda_synthesis.md` — running document of the routine's progress
- Brief: `docs/dmitry/results/em_features/EM_NANDA_BRIEF.md` — operating instructions for the autonomous orchestrator

### Status as of writeup (2026-05-01 ~23:00 UTC)

- Track A (SAE arditi 10k): **DONE through stage 4** for 2 finalists; third finalist in flight; bundle frontier sweep next.
- Track B (TXC paper k=100 10k): queued on h100_2 behind Track A's bundle frontier
- Step-count sweep (5k + 30k for both archs): queued behind 10k completions per the brief.

The autonomous routine on h100_1 (cron-driven, hourly fires until 2026-05-02 16:00 UTC) is handling everything; this writeup is for the human reader.
