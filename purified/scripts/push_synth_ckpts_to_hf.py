"""Manually push agent_synth checkpoints to HF.

Used because some agents' subprocess env didn't have
TEMP_BENCH_POD_MODE=ephemeral (lost across tool calls), so
``cache.save_checkpoint``'s auto-push was skipped. This script
walks the manifest, finds checkpoints whose hf_url is None
*and* whose local dir still exists, and uploads them.

Run with:
    TEMP_BENCH_HF_ORG=<your-hf-org> .venv/bin/python -m scripts.push_synth_ckpts_to_hf
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HF_ORG = os.environ.get("TEMP_BENCH_HF_ORG")
if not _HF_ORG:
    raise RuntimeError("TEMP_BENCH_HF_ORG env var must be set")
REPO_ID = f"{_HF_ORG}/temp-bench-models"
MANIFEST = Path("checkpoints/manifest.jsonl")


def collect_targets(agent: str = "agent_synth") -> list[str]:
    keys = []
    seen = set()
    with MANIFEST.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("agent") != agent:
                continue
            tk = d["train_key"]
            if tk in seen:
                continue
            seen.add(tk)
            if d.get("hf_url"):
                continue
            ckpt_dir = Path(f"checkpoints/{tk}")
            if not ckpt_dir.exists() or not (ckpt_dir / "model.safetensors").exists():
                continue
            keys.append(tk)
    return keys


def push_one(api, tk: str) -> tuple[str, bool, str]:
    ckpt_dir = Path(f"checkpoints/{tk}")
    try:
        api.upload_folder(
            folder_path=str(ckpt_dir),
            path_in_repo=tk,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"agent=agent_synth backfill train_key={tk}",
        )
        return tk, True, ""
    except Exception as e:
        return tk, False, str(e)[:200]


def main() -> int:
    from huggingface_hub import HfApi
    from temp_bench.utils.tokens import require_token

    targets = collect_targets("agent_synth")
    print(f"[push] {len(targets)} agent_synth ckpts to push")
    if not targets:
        return 0

    api = HfApi(token=require_token("hf"))
    # 8-way threadpool created HF commit-merge conflicts (Request ID
    # bursts at ~140 cells in). Drop to sequential — slow but reliable.
    n_workers = 1
    n_done = 0
    n_fail = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(push_one, api, tk): tk for tk in targets}
        for fut in as_completed(futures):
            tk, ok, err = fut.result()
            if ok:
                n_done += 1
            else:
                n_fail += 1
                print(f"  FAIL {tk}: {err}", file=sys.stderr)
            if (n_done + n_fail) % 20 == 0 or (n_done + n_fail) == len(targets):
                elapsed = time.time() - start
                rate = (n_done + n_fail) / max(elapsed, 1)
                eta = (len(targets) - n_done - n_fail) / max(rate, 0.001)
                print(f"  [{n_done + n_fail}/{len(targets)}] "
                      f"ok={n_done} fail={n_fail} "
                      f"rate={rate:.1f}/s ETA={eta:.0f}s", flush=True)

    elapsed = time.time() - start
    print(f"\n[push] done in {elapsed:.0f}s. ok={n_done} fail={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
