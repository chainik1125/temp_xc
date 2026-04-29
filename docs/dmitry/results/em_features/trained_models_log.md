---
author: Dmitry
date: 2026-04-29
tags:
  - reference
  - results
---

## Trained models log

Single source of truth for every trained SAE / Han / TXC / MLC / T-SAE checkpoint we have. Each row links to the HF artifact.

**HF repo**: `dmanningcoe/temp-xc-em-features` (public — no token needed)

Subject model (frozen, training source for all activation streams): `Qwen/Qwen2.5-7B-Instruct` (base, no PEFT). All trainings draw activations at layer 15 of the base model — there's no bad-medical-Qwen training.

Streaming dataset: 70% Pile-uncopyrighted + 30% UltraChat (`cfg["streaming"]["corpus_mix"]` in `experiments/em_features/config.yaml`).

### Conventions

- **Prefix `qwen_l15_*`**: original h100_1 production runs. Layer 15, no host tag.
- **Prefix `h2_qwen_l15_*`**: h100_2 fresh-train runs (the 30k 3-stage + 60k continuation, with Adam + RNG state preserved per the new trainer code).
- **Prefix `v2_qwen_l15_*`**: clean 100k retrains initiated 2026-04-27 with deterministic CUDA + seed 42. **Canonical "v2" results.**
- **Suffix `_residmid_` / `_ln1_`**: hookpoint variants (default unsuffixed = `resid_post`).
- **Step-suffix `_step{N}.pt`**: training step at which the snapshot was saved.

### SAE arditi (TopK SAE, T=1, hookpoint=resid_post unless noted)

`d_sae=32k, k=128, lr=3e-4, batch=256`

| key | hookpoint | step | best_loss | size | HF path |
|---|---|---:|---:|---:|---|
| 30k pipeline | resid_post | 5k–30k | — | 0.94 GB ea | `sae/qwen_l15_sae_arditi_k128_30k_step{5000,10000,20000,30000}.pt` |
| 100k pipeline | resid_post | 40k–100k | — | 0.94 GB ea | `sae/qwen_l15_sae_arditi_k128_100k_step{40000,...,100000}.pt` |
| h2 30k → 60k | resid_post | 10k–60k | — | 2.82 GB ea | `sae/h2_qwen_l15_sae_arditi_k128_step{10000,...,60000}.pt` |
| **v2 clean 100k** | resid_post | **50k / 80k / 100k** | — | 2.82 GB ea | `sae/v2_qwen_l15_sae_arditi_k128_step{50000,80000,100000}.pt` |

### TXC brickenauxk α=1/8 (T=5 windowed, multi-distance, antidead, bricken_resample + ema_auxk)

`d_sae=32k, k_total=128, T=5, lr=3e-4, batch=256`

| key | hookpoint | step | best_loss | size | HF path |
|---|---|---:|---:|---:|---|
| 30k pipeline | resid_post | 5k–30k | — | 4.70 GB | `txc/qwen_l15_txc_brickenauxk_a8_30k_step{5000,...,30000}.pt` |
| 100k pipeline | resid_post | 40k–100k | — | 4.70 GB | `txc/qwen_l15_txc_brickenauxk_a8_100k_step{40000,...,100000}.pt` |
| h2 30k → 60k | resid_post | 10k–60k | — | 14.09 GB | `txc/h2_qwen_l15_txc_brickenauxk_a8_step{10000,...,60000}.pt` |
| **residmid 30k** (this session) | **resid_mid** | **10k** (20k corrupted, 30k pending resume) | TBD | 14.09 GB | `txc/qwen_l15_txc_brickenauxk_a8_residmid_step{10000}.pt` |
| **ln1 30k** (this session) | **ln1_normalized** | **10k / 20k** (30k pending) | TBD | 14.09 GB | `txc/qwen_l15_txc_brickenauxk_a8_ln1_step{10000,20000}.pt` |
| txc small (older) | resid_post | 40k / 100k / 200k | — | 4.70 GB | `txc/qwen_l15_txc_small_step{40000,100000,200000}.pt` |
| txc T=5 k=128 (legacy) | resid_post | (one ckpt) | — | 4.70 GB | `txc/qwen_l15_txc_t5_k128.pt` |

### Han champion (TXCBareMultiDistanceContrastiveAntidead, T=5, multi-distance contrastive + matryoshka)

`d_sae=32k, k=128, T=5, shifts=(1,2), matryoshka_h_size=d_sae/5, alpha_contrastive=1.0, lr=3e-4`

| key | hookpoint | step | best_loss | size | HF path |
|---|---|---:|---:|---:|---|
| 30k pipeline | resid_post | 5k–30k | — | 4.70 GB | `han_champ/qwen_l15_han_champ_30k_step{5000,...,30000}.pt` |
| chunked 100k | resid_post | 40k–100k | — | 14.09 GB | `han_champ/qwen_l15_han_champ_100k_step{40000,...,100000}.pt` |
| h2 30k → 60k | resid_post | 10k–60k | — | 14.09 GB | `han_champ/h2_qwen_l15_han_champ_step{10000,...,60000}.pt` |
| **v2 clean 100k** | resid_post | **50k / 80k / 100k** | 9690 (100k) | 14.09 GB | `han_champ/v2_qwen_l15_han_champ_step{50000,80000,100000}.pt` |

### T-SAE (Bhalla et al. 2025, per-token TopK SAE + adjacent-token contrastive)

`d_sae=32k, k=128, T=5 (windows for contrastive only, encoder is per-token), contrastive_alpha=1.0, lr=3e-4`

| key | hookpoint | step | best_loss | HF path |
|---|---|---:|---:|---|
| **v1 (resid_post)** | resid_post | **10k** | 0.30 | `tsae/qwen_l15_tsae_k128_step10000.pt` |
| | | **20k** | 0.28 | `tsae/qwen_l15_tsae_k128_step20000.pt` |
| | | **30k** | 0.27 | `tsae/qwen_l15_tsae_k128_step30000.pt` |
| | | **50k** | 0.26 | `tsae/qwen_l15_tsae_k128_step50000.pt` |
| | | **80k** | 0.25 | `tsae/qwen_l15_tsae_k128_step80000.pt` |
| | | **100k** | 0.246 | `tsae/qwen_l15_tsae_k128_step100000.pt` |
| **resid_mid** | resid_mid | **10k** | 0.288 | `tsae/qwen_l15_tsae_residmid_k128_step10000.pt` |
| | | **20k** | 0.269 | `tsae/qwen_l15_tsae_residmid_k128_step20000.pt` |
| | | **30k** | 0.262 | `tsae/qwen_l15_tsae_residmid_k128_step30000.pt` |
| **ln1_normalized** | ln1_normalized | **10k** | 0.097 | `tsae/qwen_l15_tsae_ln1_k128_step10000.pt` |
| | | **20k** | 0.091 | `tsae/qwen_l15_tsae_ln1_k128_step20000.pt` |
| | | **30k** | 0.088 | `tsae/qwen_l15_tsae_ln1_k128_step30000.pt` |

(ln1_normalized losses are smaller because LayerNorm output is unit-variance.)

### MLC (multi-layer crosscoder, layers 11/13/15/17/19)

| key | hookpoint | step | size | HF path |
|---|---|---:|---:|---|
| qwen_mlc_l11-13-15-17-19_k128 | resid_post (all 5 layers) | (one ckpt) | 4.70 GB | `mlc/qwen_mlc_l11-13-15-17-19_k128.pt` |
| qwen_mlc_small | resid_post (all 5 layers) | 40k / 100k / 200k | 4.70 GB | `mlc/qwen_mlc_small_step{40000,100000,200000}.pt` |

### Active runs (in progress as of this writing)

| arch | hookpoint | host | step | status |
|---|---|---|---:|---|
| TXC brickenauxk @ resid_mid | resid_mid | h100_1 | step 10k–30k | resuming from step10k after disk-full crash; will save step 20k + 30k |
| TXC brickenauxk @ ln1 | ln1_normalized | h100_2 | step 10k–30k | step 20k saved, step 30k pending |

### Re-downloading any checkpoint

```python
from huggingface_hub import hf_hub_download
import torch

p = hf_hub_download(
    repo_id="dmanningcoe/temp-xc-em-features",
    filename="han_champ/v2_qwen_l15_han_champ_step100000.pt",
)
ckpt = torch.load(p, map_location="cuda", weights_only=False)
# ckpt.keys() == ["state_dict", "optimizer_state", "rng_state", "config", ...]
```

### Local cleanup log

Each entry lists checkpoints that were deleted *locally* on a GPU host to free disk; all remain mirrored on HF and can be re-downloaded as shown above.

**2026-04-28 (h100_1)** — chunked-Han 100k restart needed disk:
```
qwen_l15_han_champ_100k_step{40000,50000,60000,70000,80000,90000}.pt   (6 × 14 GB)
```

**2026-04-28 (h100_2)** — T-SAE 30k transfer needed disk:
```
h2_qwen_l15_han_champ_step{40000,50000}.pt             (2 × 14 GB)
h2_qwen_l15_txc_brickenauxk_a8_step{40000,50000}.pt    (2 × 14 GB)
```

**2026-04-29 (h100_2)** — TXC ln1/resid_mid trainings needed disk:
```
v2_qwen_l15_han_champ_step{50000,80000}.pt   (2 × 14 GB)
h2_qwen_l15_han_champ_step60000.pt           (1 × 14 GB)
```

**2026-04-29 (h100_1)** — TXC resid_mid training needed disk after step20k save corruption:
```
qwen_l15_txc_brickenauxk_a8_residmid_step20000.pt           (corrupt, partial write)
qwen_l15_txc_brickenauxk_a8_100k_step{50000,60000,70000,80000,90000}.pt  (5 × 4.70 GB)
```

### Last verified

2026-04-29 — repo size ≈ 230 GB across 50+ ckpts. Repo is **public** (no token needed to download).
