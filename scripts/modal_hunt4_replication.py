"""Modal driver — HUNT4 adversarial replication, gemma2_2b re-seed
(hunt4/REPLICATION_CARD.md, mac-b; duty c1c5c949e).

  modal run --detach scripts/modal_hunt4_replication.py

One L40S container. Dialevel gemma cache expected cache-hit (wave-1
warmed it). Result JSON persists to Volume /workspace/hunt4_replication
after every cell via the screen's resumable save + a finally copy;
containers never push.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "PIN_AFTER_PUSH"  # origin-history rev-parse, set post-push
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
MODEL = "gemma2_2b"
PY = "/repo/.venv/bin/python"
RES_DIR = ("/repo/experiments/explorations/task_hunt/hunt4/results/"
           "replication")

app = modal.App("mac-b-hunt4-replication")
vol = modal.Volume.from_name("temp-xc-replag-caches", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        f"git clone {REPO_URL} /repo && git -C /repo checkout {PINNED_COMMIT}",
        "pip install uv",
        "cd /repo && uv sync --frozen",
    )
    .env({"HF_HOME": "/workspace/hf_cache"})
)


def _sh(cmd: str):
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd="/repo")


@app.function(image=image, gpu="L40S", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              cpu=8, memory=65536, timeout=4 * 60 * 60,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_replication() -> str:
    head = subprocess.run("git -C /repo rev-parse HEAD", shell=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == PINNED_COMMIT, f"container tree at {head}, not the freeze"
    print(f"[pin] container at replication freeze {head[:10]}", flush=True)
    _sh(f"{PY} -m experiments.explorations.task_hunt.dialevel.cache_acts "
        f"{MODEL}")
    vol_dir = "/workspace/hunt4_replication"
    _sh(f"mkdir -p {vol_dir}")
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.hunt4."
            f"replication_screen {MODEL}")
    finally:
        src = Path(RES_DIR) / f"screen_{MODEL}.json"
        if src.exists():
            (Path(vol_dir) / f"screen_{MODEL}.json").write_text(
                src.read_text())
            vol.commit()
            print(f"[payload] replication screen_{MODEL}.json persisted",
                  flush=True)
    return (Path(RES_DIR) / f"screen_{MODEL}.json").read_text()


@app.local_entrypoint()
def main():
    out_dir = Path(__file__).resolve().parents[1] / \
        "experiments/explorations/task_hunt/hunt4/results/replication"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = run_replication.remote()
        p = out_dir / f"screen_{MODEL}.json"
        p.write_text(payload)
        print(f"[repatriated] {p} ({len(payload)} bytes)", flush=True)
    except Exception as e:                        # noqa: BLE001
        print(f"[FAILED] replication/{MODEL}: {e!r} — result persists on "
              f"Volume /workspace/hunt4_replication", flush=True)
    print("HUNT4 REPLICATION COMPLETE", flush=True)
