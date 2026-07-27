"""Modal driver — chaz factory screen (chaz/CARD.md; overnight § 1 seed 4).

  modal run --detach scripts/modal_chaz_screen.py

1× L40S; conv_depth caches from the Volume; factory_screen protocol
unchanged. Result JSON → Volume /workspace/chaz_screen + repatriate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "81b6d5918518511e8e7e9c19bf3f3626b3f98418"  # chaz freeze, rev-parse post-push
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
PY = "/repo/.venv/bin/python"
RES_DIR = "/repo/experiments/explorations/task_hunt/chaz"

app = modal.App("mac-a-chaz-screen")
# Ward/conv_depth caches live on the WARD volume (first launch mounted
# temp-xc-replag-caches → factory_screen silently skipped every model →
# 0 cells; ops-layer venue fix, no frozen-science change).
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
def run_chaz() -> str:
    head = subprocess.run("git -C /repo rev-parse HEAD", shell=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == PINNED_COMMIT, f"container tree at {head}, not the freeze"
    print(f"[pin] container at chaz freeze {head[:10]}", flush=True)
    vol_dir = "/workspace/chaz_screen"
    _sh(f"mkdir -p {vol_dir}")
    out_json = Path(RES_DIR) / "results" / "chaz_main_screen.json"
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.factory_screen "
            f"chaz - {RES_DIR}")
    finally:
        if out_json.exists():
            (Path(vol_dir) / out_json.name).write_text(out_json.read_text())
            vol.commit()
            print(f"[payload] chaz: {out_json.name} persisted", flush=True)
    return out_json.read_text() if out_json.exists() else ""


@app.local_entrypoint()
def main():
    try:
        names = run_chaz.remote()
        print(f"[done] chaz screen artifacts: {names}", flush=True)
    except Exception as e:                        # noqa: BLE001
        print(f"[FAILED] chaz: {e!r} — results persist on Volume "
              f"/workspace/chaz_screen", flush=True)
    print("CHAZ SCREEN PIPELINE COMPLETE", flush=True)
