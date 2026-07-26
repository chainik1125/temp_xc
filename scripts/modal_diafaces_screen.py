"""Modal driver — day-2 W2 diafaces Stage-1 screen (mac-a).

Ops per briefings/day2-dialogue-shared.md: L40S; PIN verbatim from
`git rev-parse HEAD` at the freeze; `_assert_pinned()` in-container;
secret `hf-token` (gemma-2-2b GO per the amendment); dialevel caches
(re)built in the WORK container by the committed builder (idempotent,
shared path with W1 — first build wins); results persisted to the
Volume after every cell and repatriated by the client; containers
never push git. Stated deviation from shared-ops § 3 (sequential
.remote): **one container PER MODEL, run in parallel via .map** —
reason: the 14:30 London panel-gate clock; robustness kept by
per-model result files, Volume persistence, retries, and
detach-at-launch (CARD.md § 8 states the same).

  modal run --detach scripts/modal_diafaces_screen.py                # all 3
  modal run --detach scripts/modal_diafaces_screen.py --models gpt2  # subset
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "07361111329c10831ad3f7ec35488eb3ca6fbfd6"  # diafaces FREEZE
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
KEYS = ["gpt2", "llama31_8b", "gemma2_2b"]
PY = "/repo/.venv/bin/python"
RESULTS_VOL_DIR = "/workspace/diafaces_results"
REPO_RES = "/repo/experiments/explorations/task_hunt/diafaces/results"

app = modal.App("mac-a-diafaces-screen")
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
              memory=49152, timeout=4 * 60 * 60,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_screen(key: str) -> str:
    _assert_pinned()
    # cache in the SAME container (idempotent — meta.json short-circuits;
    # shared /workspace/dialevel_caches path with W1's ladder, first
    # build wins; the quotedens 34 GB cross-container lesson).
    _sh(f"{PY} -m experiments.explorations.task_hunt.dialevel.cache_acts "
        f"{key}")
    vol.commit()
    _sh(f"mkdir -p {RESULTS_VOL_DIR} {REPO_RES}")
    part = Path(RESULTS_VOL_DIR) / f"screen_{key}.json"
    if part.exists():
        _sh(f"cp {part} {REPO_RES}/")
        print(f"[resume] restored partial {part.name}", flush=True)
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.diafaces.screen "
            f"{key}")
    finally:
        out = Path(REPO_RES) / f"screen_{key}.json"
        if out.exists():
            _sh(f"cp {out} {RESULTS_VOL_DIR}/")
            vol.commit()
    return (Path(REPO_RES) / f"screen_{key}.json").read_text()


@app.local_entrypoint()
def main(models: str = ""):
    keys = [k for k in models.split(",") if k] or KEYS
    local_res = Path(__file__).resolve().parents[1] / \
        "experiments/explorations/task_hunt/diafaces/results"
    local_res.mkdir(parents=True, exist_ok=True)
    for key, res in zip(keys, run_screen.map(keys,
                                             return_exceptions=True)):
        if isinstance(res, Exception):
            print(f"[FAILED] {key}: {res!r} — partial persists on the "
                  f"Volume at {RESULTS_VOL_DIR}", flush=True)
            continue
        p = local_res / f"screen_{key}.json"
        p.write_text(res)
        print(f"[repatriated] {p} ({len(res)} bytes)", flush=True)
    print("DIAFACES SCREEN PIPELINE COMPLETE", flush=True)
