---
author: Dmitry
date: 2026-04-27
tags:
  - reference
  - results
---

## HuggingFace Checkpoint Index

All trained SAE / TXC / Han / MLC checkpoints are mirrored to the public HuggingFace repo `dmanningcoe/temp-xc-em-features`. This page is the index for what's where, so locally-deleted ckpts can be re-pulled.

Repo URL: <https://huggingface.co/dmanningcoe/temp-xc-em-features>

### Top-level layout

```text
sae/         — TopK SAE (Andy-Arditi style, T=1)
txc/         — Temporal Crosscoder (T=5, brickenauxk recipe + older small variants)
mlc/         — Multi-layer Crosscoder (L=5, layers 11/13/15/17/19)
han_champ/   — TXCBareMultiDistanceContrastiveAntidead (H8 champion)
```

### Naming conventions

- `qwen_l15_<arch>_<recipe>_<scale>_step<N>.pt` — h100_1 production runs (no host prefix). Layer 15 resid_post unless noted.
- `h2_qwen_l15_...` — h100_2 fresh-train runs (the 30k 3-stage + 60k continuation, with Adam + RNG state preserved per the new trainer code).
- `v2_qwen_l15_...` — the 100k clean retrain runs (Han + SAE) initiated 2026-04-27, snapshots at 50k/80k/100k. **These are the canonical "v2" results.**

### What's there (as of 2026-04-27)

#### SAE arditi (`d_sae=32k, k=128, layer 15 resid_post`)

| key | step | path |
|---|---:|---|
| 30k pipeline | 5k–30k | `sae/qwen_l15_sae_arditi_k128_30k_step{5000,10000,20000,30000}.pt` |
| 100k pipeline (resumed from 30k) | 40k–100k | `sae/qwen_l15_sae_arditi_k128_100k_step{40000,...,100000}.pt` |
| h2 fresh 30k → 60k (Adam preserved) | 10k–60k | `sae/h2_qwen_l15_sae_arditi_k128_step{10000,...,60000}.pt` |
| **v2 clean 100k** | **50k / 80k / 100k** | `sae/v2_qwen_l15_sae_arditi_k128_step{50000,80000,100000}.pt` |

#### TXC brickenauxk α=1/8 (`d_sae=32k, k_total=128, T=5`)

| key | step | path |
|---|---:|---|
| 30k pipeline | 5k–30k | `txc/qwen_l15_txc_brickenauxk_a8_30k_step{5000,...,30000}.pt` |
| 100k pipeline (resumed from 30k) | 40k–100k | `txc/qwen_l15_txc_brickenauxk_a8_100k_step{40000,...,100000}.pt` |
| h2 fresh 30k → 60k | 10k–60k | `txc/h2_qwen_l15_txc_brickenauxk_a8_step{10000,...,60000}.pt` |

#### Han H8 champion (`TXCBareMultiDistanceContrastiveAntidead, d_sae=32k, k=128, T=5, shifts={1,2}, matryoshka_h_size=d_sae/5`)

| key | step | path |
|---|---:|---|
| chunked 100k (40k–100k) | each | `han_champ/qwen_l15_han_champ_100k_step{40000,50000,60000,70000,80000,90000,100000}.pt` |
| **v2 clean 100k** | **50k / 80k / 100k** | `han_champ/v2_qwen_l15_han_champ_step{50000,80000,100000}.pt` |

(30k Han also lives under `han_champ/qwen_l15_han_champ_30k_step{5000,...,30000}.pt`.)

#### Older / archive

- `txc/qwen_l15_txc_small_step{40000,100000,200000}.pt` — initial TXC small-d_sae sweeps without bricken+auxk.
- `txc/qwen_l15_txc_t5_k128.pt` — earliest hand-trained TXC, no recipe metadata.
- `mlc/qwen_mlc_*` and `mlc/qwen_l15_mlc_brickenauxk_a32_10k_step10000.pt` — MLC checkpoints across various recipes.

### Re-downloading a checkpoint

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

### Known local-only deletions (not on HF)

(none — every locally-deleted ckpt has a copy on HF unless explicitly noted.)

### Local cleanup log

**2026-04-28 (h100_1)** — deleted to free disk during chunked-Han 100k restart:
```
qwen_l15_han_champ_100k_step{40000,50000,60000,70000,80000,90000}.pt   (6 × 14 GB)
```
All on HF at `han_champ/qwen_l15_han_champ_100k_step*.pt`. Re-pull as needed.

**2026-04-28 (h100_2)** — deleted to fit T-SAE 30k transfer:
```
h2_qwen_l15_han_champ_step{40000,50000}.pt             (2 × 14 GB)
h2_qwen_l15_txc_brickenauxk_a8_step{40000,50000}.pt    (2 × 14 GB)
```
All on HF at `han_champ/h2_*.pt` and `txc/h2_*.pt`. The h2 30k and 60k "final" snapshots are kept locally; only the intermediate 40k/50k from the 60k continuation pipeline were removed.

**2026-04-29 (h100_2)** — deleted to fit TXC ln1 / resid_mid trainings:
```
v2_qwen_l15_han_champ_step{50000,80000}.pt   (2 × 14 GB)
h2_qwen_l15_han_champ_step60000.pt           (1 × 14 GB)
```
All on HF (`han_champ/v2_*.pt`, `han_champ/h2_*.pt`). v2 100k snapshot kept as the canonical Han.

### Active runs (as of 2026-04-28)

- **h100_1**: T-SAE training extension, step 30k → 100k (resumed from `qwen_l15_tsae_k128_step30000.pt`, snapshots at 50k/80k/100k). Already at step ~33k.
- **h100_2**: queued for Wang procedure on T-SAE 30k once the 30k checkpoint is transferred (HF download flaked; recovering via direct h100_1 → h100_2 stream).

### Last verified

2026-04-28 — repo size ≈ 215 GB across 40+ ckpts. Repo is **public** (no token needed to download).
