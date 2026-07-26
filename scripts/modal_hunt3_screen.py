"""Modal driver — HUNT3 screens (HUNT3_SCREEN_CARD.md; overnight § 1).

  modal run --detach scripts/modal_hunt3_screen.py

3× L40S, one model per container in parallel (card § 5; diafaces
precedent). Dialevel caches (re)built in-container by the committed
builder (idempotent; expected cache-hit from day-2). Result JSONs
persist to Volume /workspace/hunt3_screen after every cell via the
screen's own resumable save + a finally copy; containers never push.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "32e316bc41e02ada14d4d82fe7b147ab939bfe42"  # hunt3 freeze, rev-parse post-push
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
MODELS = ("gpt2", "llama31_8b", "gemma2_2b")
PY = "/repo/.venv/bin/python"
RES_DIR = "/repo/experiments/explorations/task_hunt/hunt3/results"

app = modal.App("mac-a-hunt3-screen")
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
              cpu=8, memory=65536, timeout=3 * 60 * 60,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_screen(model: str) -> str:
    head = subprocess.run("git -C /repo rev-parse HEAD", shell=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == PINNED_COMMIT, f"container tree at {head}, not the freeze"
    print(f"[pin] container at hunt3 freeze {head[:10]}", flush=True)
    _sh(f"{PY} -m experiments.explorations.task_hunt.dialevel.cache_acts "
        f"{model}")
    vol_dir = "/workspace/hunt3_screen"
    _sh(f"mkdir -p {vol_dir}")
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.hunt3.screen "
            f"{model}")
    finally:
        src = Path(RES_DIR) / f"screen_{model}.json"
        if src.exists():
            (Path(vol_dir) / f"screen_{model}.json").write_text(
                src.read_text())
            vol.commit()
            print(f"[payload] hunt3 screen_{model}.json persisted", flush=True)
    return (Path(RES_DIR) / f"screen_{model}.json").read_text()


@app.local_entrypoint()
def main():
    out_dir = Path(__file__).resolve().parents[1] / \
        "experiments/explorations/task_hunt/hunt3/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = [(m, run_screen.spawn(m)) for m in MODELS]
    for name, call in calls:
        try:
            payload = call.get()
            p = out_dir / f"screen_{name}.json"
            p.write_text(payload)
            print(f"[repatriated] {p} ({len(payload)} bytes)", flush=True)
        except Exception as e:                    # noqa: BLE001
            print(f"[FAILED] hunt3/{name}: {e!r} — result persists on "
                  f"Volume /workspace/hunt3_screen", flush=True)
    print("HUNT3 SCREEN PIPELINE COMPLETE", flush=True)
