"""Modal driver — dialevel R11 order-mechanism ladder (mac-b day-2 W1).

Ops per briefings/day2-dialogue-shared.md: L40S, sequential .remote +
retries, caches built IN the work container (idempotent), launch with
`modal run --detach`, containers never push. PIN taken verbatim from
`git rev-parse HEAD` at the freeze. Secret `hf-token` supplies
HF_TOKEN/HUGGING_FACE_HUB_TOKEN (gemma2_2b is GO per the day-2
amendment and carries the largest R11 cost).

  modal run --detach scripts/modal_dialevel_ladder.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "ede97e2066a0cff39c06234fb978dc5f39ef2c79"  # ladder FREEZE
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
KEYS = ["gpt2", "llama31_8b", "gemma2_2b"]
PY = "/repo/.venv/bin/python"
RESULTS_VOL_DIR = "/workspace/dialevel_results"
REPO_RES = "/repo/experiments/explorations/task_hunt/dialevel/results"

app = modal.App("mac-b-dialevel-ladder")
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
              secrets=[modal.Secret.from_name("hf-token")],
              memory=49152, cpu=8, timeout=2 * 60 * 60,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_ladder(key: str) -> str:
    _assert_pinned()
    # cache in the SAME container (idempotent — meta.json short-circuits)
    _sh(f"{PY} -m experiments.explorations.task_hunt.dialevel.cache_acts "
        f"{key}")
    vol.commit()
    _sh(f"mkdir -p {RESULTS_VOL_DIR} {REPO_RES}")
    part = Path(RESULTS_VOL_DIR) / f"ladder_{key}.json"
    if part.exists():
        _sh(f"cp {part} {REPO_RES}/")
        print(f"[resume] restored partial {part.name}", flush=True)
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.dialevel.ladder "
            f"{key}")
    finally:
        out = Path(REPO_RES) / f"ladder_{key}.json"
        if out.exists():
            _sh(f"cp {out} {RESULTS_VOL_DIR}/")
            vol.commit()
    return (Path(REPO_RES) / f"ladder_{key}.json").read_text()


@app.local_entrypoint()
def main(models: str = ""):
    keys = [k for k in models.split(",") if k] or KEYS
    local_res = Path(__file__).resolve().parents[1] / \
        "experiments/explorations/task_hunt/dialevel/results"
    local_res.mkdir(parents=True, exist_ok=True)
    for key in keys:
        try:
            text = run_ladder.remote(key)
        except Exception as e:          # noqa: BLE001
            print(f"[FAILED] {key}: {e!r}", flush=True)
            continue
        p = local_res / f"ladder_{key}.json"
        p.write_text(text)
        print(f"[repatriated] {p} ({len(text)} bytes)", flush=True)
    print("DIALEVEL LADDER PIPELINE COMPLETE", flush=True)
