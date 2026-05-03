## Checkpoint registry

`manifest.jsonl` is the authoritative index of every model checkpoint
produced for the paper. **Append-only**, one JSON object per line.

## Backing storage

Two HF repositories hold the bulk artifacts. Both are private during
paper development; they flip to public when the draft stabilises.

| Purpose | Repo | Type |
|---|---|---|
| Model checkpoints | [`han1823123123/temp-bench-models`](https://huggingface.co/han1823123123/temp-bench-models) | model |
| Activation caches + judge transcripts + synthetic data | [`han1823123123/temp-bench-data`](https://huggingface.co/datasets/han1823123123/temp-bench-data) | dataset |

## How to upload

Every checkpoint goes to `temp-bench-models` under a `<run_id>/` prefix:

```python
from huggingface_hub import HfApi
from temp_bench.utils import append_checkpoint

run_id = "c3_txc_base_42_a1b2c3d4"
HfApi().upload_folder(
    folder_path=f"results/runs/{run_id}",
    path_in_repo=run_id,
    repo_id="han1823123123/temp-bench-models",
    repo_type="model",
)

append_checkpoint(
    run_id=run_id,
    hf_url=f"https://huggingface.co/han1823123123/temp-bench-models/tree/main/{run_id}",
    local_path=f"/workspace/.../{run_id}/model.safetensors",
    size_mb=412,
)
```

Activation caches and judge transcripts go to `temp-bench-data` with a
descriptive top-level prefix (e.g. `gemma2-2b-base-l13/`,
`c7-judge-transcripts/`). Each subdirectory carries its own `manifest.json`
explaining its contents.

## Manifest schema

```json
{
  "run_id": "c3_txc_base_42_a1b2c3d4",
  "ts": "2026-05-03T14:30:00Z",
  "hf_url": "https://huggingface.co/han1823123123/temp-bench-models/tree/main/c3_txc_base_42_a1b2c3d4",
  "local_path": "/workspace/temp_xc/purified/results/runs/c3_txc_base_42_a1b2c3d4/model.safetensors",
  "size_mb": 412
}
```

Optional extra keys: `arch`, `seed`, `k_pos`, `d_sae`, `component`, `notes`.
