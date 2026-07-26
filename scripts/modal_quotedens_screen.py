"""Modal driver — B9 quotedens Stage-1 screen (mac-b overnight stretch 2).

Same ops as scripts/modal_refmark_screen.py (L40S, sequential .remote,
retries, launch with `modal run --detach`). PIN taken verbatim from
`git rev-parse HEAD` at the freeze (the mistyped-SHA lesson).

  modal run --detach scripts/modal_quotedens_screen.py            # all
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "5b45cd027ea3c51816f05e64d42284e7ccb3bc7e"  # quotedens FREEZE
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
KEYS = ["gpt2", "llama31_8b"]
PY = "/repo/.venv/bin/python"
RESULTS_VOL_DIR = "/workspace/quotedens_results"
REPO_RES = "/repo/experiments/explorations/task_hunt/quotedens/results"

app = modal.App("mac-b-quotedens-screen")
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


def _assert_pinned():
    head = subprocess.run("git -C /repo rev-parse HEAD", shell=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == PINNED_COMMIT, f"container tree at {head}, not the freeze"
    print(f"[pin] container at freeze commit {head[:10]}", flush=True)


@app.function(image=image, gpu="L40S", volumes={"/workspace": vol},
              timeout=2 * 60 * 60,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def build_caches() -> str:
    _assert_pinned()
    for key in KEYS:
        _sh(f"{PY} -m experiments.explorations.task_hunt.quotedens."
            f"cache_acts {key}")
        vol.commit()
    return "QUOTEDENS CACHES DONE"


@app.function(image=image, gpu="L40S", volumes={"/workspace": vol},
              timeout=4 * 60 * 60,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_screen(key: str) -> str:
    _assert_pinned()
    _sh(f"mkdir -p {RESULTS_VOL_DIR} {REPO_RES}")
    part = Path(RESULTS_VOL_DIR) / f"screen_{key}.json"
    if part.exists():
        _sh(f"cp {part} {REPO_RES}/")
        print(f"[resume] restored partial {part.name}", flush=True)
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.quotedens.screen "
            f"{key}")
    finally:
        out = Path(REPO_RES) / f"screen_{key}.json"
        if out.exists():
            _sh(f"cp {out} {RESULTS_VOL_DIR}/")
            vol.commit()
    return (Path(REPO_RES) / f"screen_{key}.json").read_text()


@app.local_entrypoint()
def main(stage: str = "all", models: str = ""):
    keys = [k for k in models.split(",") if k] or KEYS
    if stage in ("caches", "all"):
        print(build_caches.remote(), flush=True)
    if stage in ("screen", "all"):
        local_res = Path(__file__).resolve().parents[1] / \
            "experiments/explorations/task_hunt/quotedens/results"
        local_res.mkdir(parents=True, exist_ok=True)
        for key in keys:
            try:
                text = run_screen.remote(key)
            except Exception as e:          # noqa: BLE001
                print(f"[FAILED] {key}: {e!r}", flush=True)
                continue
            p = local_res / f"screen_{key}.json"
            p.write_text(text)
            print(f"[repatriated] {p} ({len(text)} bytes)", flush=True)
    print("QUOTEDENS PIPELINE COMPLETE", flush=True)
