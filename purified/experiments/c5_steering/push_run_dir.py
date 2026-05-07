"""Push a results/runs/<eval_key>/ directory to ${TEMP_BENCH_HF_ORG}/temp-bench-data.

Used as a recovery / one-off helper when ``experiments/c5_steering/run.py``'s
auto-push didn't run (e.g., the cell was launched before that code
landed, or auto-push failed and we want to retry).

Usage:
    .venv/bin/python -m experiments.c5_steering.push_run_dir <eval_key> [<eval_key> ...]

The destination is ``runs/<eval_key>/`` under
``${TEMP_BENCH_HF_ORG}/temp-bench-data`` (private dataset repo). The
checkpoint per ``train_key`` is auto-pushed by the runner — this is
strictly for the per-eval workspace that the ``CaseStudy`` framework
contract requires to outlive the pod.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

from temp_bench.cache import run_dir
from temp_bench.utils.tokens import require_token


_HF_ORG = os.environ.get("TEMP_BENCH_HF_ORG")
if not _HF_ORG:
    raise RuntimeError("TEMP_BENCH_HF_ORG env var must be set")
REPO_ID = f"{_HF_ORG}/temp-bench-data"


def push_one(eval_key: str, agent: str = "agent_steer") -> str:
    workspace = run_dir(eval_key)
    if not workspace.is_dir():
        raise FileNotFoundError(f"No run_dir at {workspace}")
    if not (workspace / "metrics.json").exists():
        raise FileNotFoundError(
            f"{workspace}/metrics.json missing — incomplete cell, refusing push"
        )

    api = HfApi(token=require_token("hf"))
    api.upload_folder(
        folder_path=str(workspace),
        path_in_repo=f"runs/{eval_key}",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"agent={agent} run_dir eval_key={eval_key}",
    )
    url = f"https://huggingface.co/datasets/{REPO_ID}/tree/main/runs/{eval_key}"
    print(f"  → {url}")
    return url


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_keys", nargs="+", help="One or more eval_key strings")
    ap.add_argument("--agent", default="agent_steer")
    args = ap.parse_args()
    for k in args.eval_keys:
        print(f"[push_run_dir] {k}")
        try:
            push_one(k, agent=args.agent)
        except Exception as exc:                                # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
