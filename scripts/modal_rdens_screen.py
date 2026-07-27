"""Modal driver — rdens factory screen (rdens/CARD.md; gen-4 seed 3).

  modal run --detach scripts/modal_rdens_screen.py

1× L40S; conv_depth caches from the WARD volume (the chaz venue
lesson — NOT replag). factory_screen protocol unchanged. Result JSON
→ Volume /workspace/rdens_screen + repatriate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "PINNED_AT_FREEZE"  # fill from ORIGIN-history rev-parse post-push
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
PY = "/repo/.venv/bin/python"
RES_DIR = "/repo/experiments/explorations/task_hunt/rdens"

app = modal.App("mac-a-rdens-screen")
vol = modal.Volume.from_name("temp-xc-ward-caches", create_if_missing=True)

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
              cpu=8, memory=65536, timeout=3 * 60 * 60,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_rdens() -> str:
    head = subprocess.run("git -C /repo rev-parse HEAD", shell=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == PINNED_COMMIT, f"container tree at {head}, not the freeze"
    print(f"[pin] container at rdens freeze {head[:10]}", flush=True)
    vol_dir = "/workspace/rdens_screen"
    _sh(f"mkdir -p {vol_dir}")
    out_json = Path(RES_DIR) / "results" / "rdens_main_screen.json"
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.factory_screen "
            f"rdens - {RES_DIR}")
    finally:
        if out_json.exists():
            (Path(vol_dir) / out_json.name).write_text(out_json.read_text())
            vol.commit()
            print(f"[payload] rdens: {out_json.name} persisted", flush=True)
    return out_json.read_text() if out_json.exists() else ""


@app.local_entrypoint()
def main():
    try:
        names = run_rdens.remote()
        print(f"[done] rdens screen bytes: {len(names)}", flush=True)
    except Exception as e:                        # noqa: BLE001
        print(f"[FAILED] rdens: {e!r} — results persist on Volume "
              f"/workspace/rdens_screen", flush=True)
    print("RDENS SCREEN PIPELINE COMPLETE", flush=True)
