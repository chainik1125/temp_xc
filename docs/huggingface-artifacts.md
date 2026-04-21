---
author: Han
date: 2026-04-21
tags:
  - guide
  - reference
---

## Downloading Phase 5 / 5.7 artifacts from HuggingFace Hub

Reproducibility-relevant binary artifacts are hosted on the HuggingFace
Hub. The github repo ships all code, logs, summaries, and
per-example prediction files (small); it does **not** ship the
trained weights or the activation caches (too large for git — 129 GB
combined).

### Two repos, one download recipe

| HF repo | type | what it contains | size |
|---|---|---|---|
| [`han1823123123/txcdr`](https://huggingface.co/han1823123123/txcdr) | model | trained SAE / crosscoder checkpoints (fp16 state dicts) | ~46 GB |
| [`han1823123123/txcdr-data`](https://huggingface.co/datasets/han1823123123/txcdr-data) | dataset | Gemma activation caches (FineWeb + per-probing-task) | ~83 GB |

Both repos mirror the github-repo directory layout, so downloading
with `--local-dir .` at the repo root places every file at the path
the training / probing / plotting scripts expect.

### Prerequisites

```bash
git clone https://github.com/chainik1125/temp_xc.git
cd temp_xc
uv sync                                     # installs .venv/
export TQDM_DISABLE=1
export PYTHONPATH=$(pwd)
pip install -U "huggingface_hub[cli]"       # or: uv add huggingface_hub

# No login required for public read. Uncomment if you plan to push:
# hf auth login
```

### Download everything (fresh reproducibility run)

From the repo root:

```bash
# 1. Checkpoints  (46 GB — trained SAEs/crosscoders, all ~25 Phase-5 archs
#                 + Phase-5.7 autoresearch archs).
huggingface-cli download han1823123123/txcdr \
  --local-dir experiments/phase5_downstream_utility/results

# 2. Activation caches + probe caches  (83 GB — FineWeb activations at
#                                      L11-L15 plus per-task probing caches
#                                      for 36 binary tasks).
huggingface-cli download --repo-type dataset han1823123123/txcdr-data \
  --local-dir .
```

After these two downloads you'll have:

```
data/cached_activations/gemma-2-2b-it/fineweb/
  token_ids.npy                    (6 MB)
  resid_L11.npy .. resid_L15.npy   (3.5 GB each fp16)
experiments/phase5_downstream_utility/results/
  ckpts/
    <arch>__seed42.pt              (~48 files, 170 MB - 3.4 GB each fp16)
  probe_cache/
    <task>/
      acts_anchor.npz              (L13 tail-20 activations per sample)
      acts_mlc.npz                 (L11-L15 at last real token)
      acts_mlc_tail.npz            (L11-L15 over tail-20, for mean_pool)
      meta.json                    (split sizes, dataset_key, etc.)
    (36 task dirs total)
```

### Download just the ckpts (if you already have caches)

```bash
huggingface-cli download han1823123123/txcdr \
  --local-dir experiments/phase5_downstream_utility/results
```

### Download a single checkpoint

```bash
huggingface-cli download han1823123123/txcdr \
  ckpts/txcdr_contrastive_t5__seed42.pt \
  --local-dir experiments/phase5_downstream_utility/results
```

### Download a single probing task cache

```bash
huggingface-cli download --repo-type dataset han1823123123/txcdr-data \
  'experiments/phase5_downstream_utility/results/probe_cache/ag_news_business/*' \
  --local-dir .
```

### What to do after downloading

- **Re-run probing** (no re-training required): see the 25-arch bench in
  [`summary.md`](summary.md) §"Pipeline reproduction".
- **Re-build a single plot**: see `experiments/phase5_downstream_utility/plots/`.
- **Run the Part-B finalization**: `partB_finalize.py` auto-picks the best
  Phase-5.7 config per FINALIST family and probes on test split.
- **From-scratch rebuild** (regenerate activations + train everything
  from raw Gemma): see
  [`2026-04-21-reproduction-brief.md`](2026-04-21-reproduction-brief.md).
  That's ~12-15 h compute on an A40-class GPU; the HF download path
  skips the expensive cache-building + training steps.

### Programmatic usage

If you prefer Python over the CLI:

```python
from huggingface_hub import snapshot_download

# All ckpts
snapshot_download(
    "han1823123123/txcdr",
    repo_type="model",
    local_dir="experiments/phase5_downstream_utility/results",
)

# All caches
snapshot_download(
    "han1823123123/txcdr-data",
    repo_type="dataset",
    local_dir=".",
)
```

Downloads use the HF cache layer by default (`~/.cache/huggingface/`),
so `snapshot_download` without `local_dir` keeps the files there and
the `--local-dir` flag just adds symlinks / copies.

### Loading a checkpoint

See each arch's class under `src/architectures/` for the constructor
signature; the ckpt's `meta` dict has the required arguments. Example:

```python
import torch
from src.architectures.txcdr_contrastive import TXCDRContrastive

state = torch.load(
    "experiments/phase5_downstream_utility/results/ckpts/"
    "txcdr_contrastive_t5__seed42.pt",
    map_location="cuda", weights_only=False,
)
meta = state["meta"]
T = meta["T"]
k_eff = meta["k_win"] or (meta["k_pos"] * T)
model = TXCDRContrastive(
    d_in=2304, d_sae=18432, T=T, k=k_eff, h=meta.get("h", 18432 // 2),
).cuda()
cast = {
    k: v.float() if v.dtype == torch.float16 else v
    for k, v in state["state_dict"].items()
}
model.load_state_dict(cast)
model.eval()
```

`experiments/phase5_downstream_utility/probing/run_probing.py`'s
`_load_model_for_run` has the routing for every checkpoint in the
bench — copy that dispatch table if you need to load many.

### Security / stability note

The HF repos are **public**, no gate, no terms-of-use. The github
repo is the source of truth for code + docs + small artifacts. If
either HF repo ever goes offline, everything can be regenerated from
the github scripts plus `google/gemma-2-2b-it` — see
[`2026-04-21-reproduction-brief.md`](2026-04-21-reproduction-brief.md).
