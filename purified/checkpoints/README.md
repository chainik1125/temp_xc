## Checkpoint registry

`manifest.jsonl` is the authoritative index of every model checkpoint
produced for the paper. **Append-only**, one JSON object per line.

## Backing storage

Two HF repositories hold the bulk artifacts. Both are private during
paper development; they flip to public when the draft stabilises.

| Purpose | Repo | Type |
|---|---|---|
| Model checkpoints | [`${TEMP_BENCH_HF_ORG}/temp-bench-models`](https://huggingface.co/${TEMP_BENCH_HF_ORG}/temp-bench-models) | model |
| Activation caches + judge transcripts + synthetic data | [`${TEMP_BENCH_HF_ORG}/temp-bench-data`](https://huggingface.co/datasets/${TEMP_BENCH_HF_ORG}/temp-bench-data) | dataset |

## How to upload

The runner already saves the trained checkpoint to
``checkpoints/<train_key>/`` and appends a validated row to
``manifest.jsonl``. Your only step is HF backup:

```python
from huggingface_hub import HfApi
from temp_bench.config import checkpoint_dir

train_key = result.train_key   # from runner.run_cell return value
HfApi().upload_folder(
    folder_path=str(checkpoint_dir(train_key)),
    path_in_repo=train_key,
    repo_id="${TEMP_BENCH_HF_ORG}/temp-bench-models",
    repo_type="model",
)
```

Activation caches and judge transcripts go to ``temp-bench-data`` with
a descriptive top-level prefix (``act_cache_<key>/``,
``c7-judge-transcripts/<run_id>/``). Each subdirectory carries its own
``manifest.json`` explaining its contents.

## Manifest schema

Validated by ``temp_bench.schemas.CheckpointManifest``:

```json
{
  "schema_version": "1.0.0",
  "train_key": "fedcba9876543210",
  "act_cache_key": "0123456789abcdef",
  "arch": "txc_base",
  "arch_version": "1.0.0",
  "seed": 42,
  "datasource": "gemma_2_2b_it_l13_fineweb_24k128",
  "training_cfg": {"n_steps": 30000, "batch_size": 256, ...},
  "hf_url": "https://huggingface.co/${TEMP_BENCH_HF_ORG}/temp-bench-models/tree/main/fedcba9876543210",
  "local_path": "/workspace/temp_xc/purified/checkpoints/fedcba9876543210/model.safetensors",
  "size_mb": 412.3,
  "agent": "agent_nlp",
  "ts": "2026-05-03T14:30:00Z"
}
```

Schema-rejected rows are aborted at append time; the cache contract
guarantees no malformed rows ever land in the file.
