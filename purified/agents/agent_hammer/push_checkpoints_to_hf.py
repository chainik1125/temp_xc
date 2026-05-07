"""Recovery: push agent_hammer checkpoints to HF that were missed at train time.

Bug: my launcher (`run_baselines_launch.sh`) didn't set
``TEMP_BENCH_POD_MODE=ephemeral`` in the subprocess env, so
``cache.save_checkpoint`` skipped the HF push. All 109 manifest entries
have ``hf_url=null``. This script reads the manifest, finds agent_hammer
entries without an hf_url, and pushes each checkpoint dir to
``han1823123123/temp-bench-models``.

Idempotent — `huggingface_hub.upload_folder` skips files that already
exist server-side with the same hash. Doesn't touch manifest.jsonl
(append-only); we instead append a recovery marker after pushing all.

Run:
    HF_HUB_DISABLE_PROGRESS_BARS=1 \\
      .venv/bin/python -m agents.agent_hammer.push_checkpoints_to_hf
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "han1823123123/temp-bench-models"


def _read_token() -> str:
    for p in ("/workspace/.tokens/hf_token", os.path.expanduser("~/.tokens/hf_token")):
        if os.path.exists(p):
            return Path(p).read_text().strip()
    raise SystemExit("No HF token found at /workspace/.tokens/hf_token or ~/.tokens/hf_token.")


def main() -> None:
    token = _read_token()
    api = HfApi(token=token)

    # Collect agent_hammer train_keys — read from on-disk config.jsons
    # so we catch checkpoints whose manifest entry was lost (HF 429 killed
    # the proc before append_checkpoint_manifest could run).
    targets = []  # list of (train_key, ckpt_dir)
    for cfg_path in sorted(Path("checkpoints").glob("*/config.json")):
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        if cfg.get("agent") != "agent_hammer":
            continue
        tk = cfg.get("train_key") or cfg_path.parent.name
        d = cfg_path.parent
        if not (d / "model.safetensors").exists():
            continue
        targets.append((tk, d))

    print(f"Found {len(targets)} agent_hammer checkpoint dirs to push to {REPO_ID}", flush=True)
    if not targets:
        return

    n_ok = 0
    n_err = 0
    t0 = time.time()
    for i, (tk, d) in enumerate(targets, start=1):
        try:
            api.upload_folder(
                folder_path=str(d),
                path_in_repo=tk,
                repo_id=REPO_ID,
                repo_type="model",
                commit_message=f"agent=agent_hammer train_key={tk} (recovery)",
            )
            n_ok += 1
            elapsed = time.time() - t0
            eta = elapsed / i * (len(targets) - i)
            if i % 5 == 0 or i == len(targets):
                print(f"  [{i:3d}/{len(targets)}] pushed {tk[:12]}  "
                      f"({n_ok} ok / {n_err} err)  eta={eta/60:.1f}m",
                      flush=True)
        except HfHubHTTPError as e:
            n_err += 1
            print(f"  [{i:3d}/{len(targets)}] ERROR {tk[:12]}: {e}", flush=True)

    print(f"\nDone: {n_ok} pushed, {n_err} errors in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
