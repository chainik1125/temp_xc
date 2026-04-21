---
author: Han
date: 2026-04-21
tags:
  - guide
  - reference
---

## HuggingFace artifacts for `temp_xc`

This repo ships code, docs, and small results (<10 MB per file, total <500 MB) in git. Anything bigger — trained checkpoints, cached model activations, large per-experiment datasets — is hosted on HuggingFace Hub so that git stays clean and cloning is fast.

**Two HF repos, mirror layout, one convention:**

| HF repo | type | what it contains |
|---|---|---|
| [`han1823123123/txcdr`](https://huggingface.co/han1823123123/txcdr) | model | trained checkpoints (fp16 `state_dict` `.pt` files) |
| [`han1823123123/txcdr-data`](https://huggingface.co/datasets/han1823123123/txcdr-data) | dataset | cached model activations, per-probing-task caches, any other large reproducibility artifacts |

The layout of paths inside each HF repo **mirrors the git repo**. Downloading with `--local-dir .` at the git repo root lands every file at the exact path the training / probing / plotting scripts expect it at.

**Split convention** (why two repos not one):
- `han1823123123/txcdr` = **model weights only**. HF's "Models" surface is the idiomatic home for trainable `.pt` artifacts; users cloning the repo typically want just the weights.
- `han1823123123/txcdr-data` = **non-model binary data**. Activation caches, intermediate arrays, probing caches, anything that is "data needed to reproduce" without being a trained model.

### Download recipe (end-user)

```bash
git clone https://github.com/chainik1125/temp_xc.git
cd temp_xc
uv sync
pip install -U "huggingface_hub[cli]"    # or: uv add huggingface_hub
export TQDM_DISABLE=1
export PYTHONPATH=$(pwd)

# Download checkpoints (model repo)
huggingface-cli download han1823123123/txcdr --local-dir .

# Download caches (dataset repo)
huggingface-cli download --repo-type dataset han1823123123/txcdr-data --local-dir .
```

After these two commands, every gitignored binary artifact lives at the path the repo's scripts look for it — no extra `mv` / `cp` needed.

For a single file:

```bash
huggingface-cli download han1823123123/txcdr \
  experiments/<phase>/results/ckpts/<arch>__seed<n>.pt \
  --local-dir .
```

For Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("han1823123123/txcdr", repo_type="model", local_dir=".")
snapshot_download("han1823123123/txcdr-data", repo_type="dataset", local_dir=".")
```

### Loading a checkpoint (generic pattern)

Each `.pt` is saved as `{"state_dict", "arch", "meta", "state_dict_dtype"}`. The `state_dict` is in fp16 to halve disk cost; the loader casts back to fp32 when building the model.

```python
import torch

state = torch.load("path/to/<arch>__seed42.pt",
                   map_location="cuda", weights_only=False)
arch_name = state["arch"]
meta = state["meta"]

# Build the model from meta. The exact constructor call is arch-specific;
# see experiments/<phase>/probing/run_probing.py::_load_model_for_run for
# the dispatch table that knows every arch in the bench.

cast = {k: v.float() if v.dtype == torch.float16 else v
        for k, v in state["state_dict"].items()}
model.load_state_dict(cast)
model.eval()
```

If you need to load a specific arch's checkpoint, copy the corresponding branch from the `_load_model_for_run` dispatch table — it already has every arch in the bench wired up.

### Uploading new artifacts (contributor)

Two helper scripts ship with the repo. Re-runs are safe — both use `upload_folder`, which skips unchanged files by hash.

**Checkpoints**:

```bash
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/hf_upload_ckpts.py
```

By default it uploads every `.pt` under `experiments/*/results/ckpts/`. Tweak `CKPT_DIR` / `allow_patterns` in the script if you only want a subset.

**Caches / datasets**:

```bash
HF_HOME=/workspace/hf_cache .venv/bin/python scripts/hf_upload_data.py
```

Uploads the activation cache (`data/cached_activations/...`) and per-probing-task caches (`experiments/*/results/probe_cache/`). Pass `--only activations` or `--only probe_cache` to upload just one subdir.

### What lives where (quick reference)

| artifact | in git? | HF repo | typical size |
|---|---|---|---|
| source code (`src/`, `experiments/`) | ✓ | — | <5 MB |
| docs (`docs/`) | ✓ | — | <5 MB |
| results jsonl / json (`results/*.jsonl`, `results/*.json`) | ✓ | — | <10 MB |
| small plot files (`results/plots/*.png`) | ✓ | — | <50 MB |
| per-example predictions (`results/predictions/*.npz`) | ✓ | — | ~10 MB total |
| training logs (`results/training_logs/*.json`) | ✓ | — | ~1 MB |
| **trained model checkpoints (`ckpts/*.pt`)** | ✗ | `han1823123123/txcdr` | ~50-200 GB per phase |
| **cached model activations (`data/cached_activations/`)** | ✗ | `han1823123123/txcdr-data` | ~15-50 GB per model/dataset |
| **per-probing-task caches (`probe_cache/<task>/*.npz`)** | ✗ | `han1823123123/txcdr-data` | ~30-100 GB per phase |

### Operational notes

- **Auth**: write access requires an HF token with the `Write repositories` scope. The token is kept at `/workspace/hf_cache/token` on the pod (via `HF_HOME=/workspace/hf_cache`). Rotate via <https://huggingface.co/settings/tokens> if exposed.
- **READMEs**: the HF repos carry deliberately minimal stub READMEs (license + pointer to github). All project narrative and research context live in git docs, not in the HF repo description. When adding new artifacts, do NOT push expanded READMEs from the upload scripts.
- **Adding a new phase**: no HF-side changes needed. Drop new `.pt` into `experiments/<newphase>/results/ckpts/` and re-run `hf_upload_ckpts.py`; same for `probe_cache`. The repos' mirror layout means new paths appear under new sub-trees automatically.
