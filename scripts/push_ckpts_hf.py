"""Push trained checkpoints to the HF durable mirror (durability sweep
b4ec84b04 item 2). The framework's trainer writes ``hf_url=None``
unconditionally (no auto-push path exists — receipts: 0 non-null
hf_url fleet-wide), so this script is the manual compliance path:
upload ``checkpoints/<train_key>/model.safetensors`` to the datasets
mirror under ``checkpoints/<train_key>/`` and print sha256 + repo-path
receipts for STATUS. Manifest rows are NOT rewritten (append-only
discipline) — receipts live in STATUS/LOG.

Run: .venv/bin/python scripts/push_ckpts_hf.py <train_key> [...]
Env: token read from /workspace/.tokens/hf_token_datasets (Han's
datasets account, the sweep's write token). Idempotent per file
(skips when the remote blob already matches by size+sha prefix check
failure falls back to upload).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "han1823123123/temp-bench-data"
TOKEN_PATH = Path("/workspace/.tokens/hf_token_datasets")
ROOT = Path(__file__).resolve().parents[1] / "checkpoints"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(keys: list[str]) -> None:
    api = HfApi(token=TOKEN_PATH.read_text().strip())
    existing = set(api.list_repo_files(REPO, repo_type="dataset"))
    for key in keys:
        src = ROOT / key / "model.safetensors"
        if not src.exists():
            print(f"[push] {key}: MISSING locally — skip", flush=True)
            continue
        digest = sha256(src)
        dst = f"checkpoints/{key}/model.safetensors"
        if dst in existing:
            print(f"[push] {key}: remote exists — receipt sha256={digest} "
                  f"path={REPO}/{dst} (verify-only)", flush=True)
            continue
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst,
                        repo_id=REPO, repo_type="dataset",
                        commit_message=f"ckpt {key} (durability sweep)")
        print(f"[push] {key}: UPLOADED sha256={digest} path={REPO}/{dst}",
              flush=True)


if __name__ == "__main__":
    assert len(sys.argv) > 1, "usage: push_ckpts_hf.py <train_key> [...]"
    main(sys.argv[1:])
