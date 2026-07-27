"""Modal driver — cnov panel (CNOV_PANEL_CARD; STAGED, PICK-PENDING).

DO NOT LAUNCH before: (1) the team pick sets DS in run_cnov_panel.py,
(2) card+runner+scorer freeze in one commit, (3) PINNED_COMMIT below
is filled from ORIGIN-HISTORY `git rev-parse` post-push (the
pull-rebase-rewrites-the-SHA lesson), (4) ledger line appended.

  modal run --detach scripts/modal_cnov_panel.py

H100 main (24 cells, workers 6) + 3× high-CPU L4 trained-tsae.
Payloads → Volume /workspace/hunt3_cnov_panel; containers never push.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "PINNED_AT_FREEZE"  # fill from origin-history rev-parse at the pick
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
SEEDS = (9, 10, 11)
PY = "/repo/.venv/bin/python"
LB = "/repo/results/leaderboard.jsonl"
RES_DIR = "/repo/experiments/explorations/task_hunt/hunt3/results"

app = modal.App("mac-a-cnov-panel")
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
    print(f"[pin] container at cnov freeze {head[:10]}", flush=True)


def _model_key() -> str:
    ds = subprocess.run(
        f"{PY} -c \"from experiments.explorations.task_hunt.hunt3."
        f"run_cnov_panel import DS; print(DS)\"",
        shell=True, capture_output=True, text=True, cwd="/repo",
    ).stdout.strip()
    return "gemma2_2b" if "gemma2" in ds else "gpt2"


def _run_block(block: str, only_seed: int | None,
               workers: int | None = None, only_cells: str = "") -> str:
    _assert_pinned()
    _sh(f"{PY} -m experiments.explorations.task_hunt.dialevel.cache_acts "
        f"{_model_key()}")
    n0 = sum(1 for _ in open(LB)) if Path(LB).exists() else 0
    vol_dir = "/workspace/hunt3_cnov_panel"
    _sh(f"mkdir -p {vol_dir}")
    suffix = f"_{block}" + (f"_s{only_seed}" if only_seed is not None else "") \
        + ("_repass" if only_cells else "")
    flags = f"--block {block}" + (
        f" --only-seed {only_seed}" if only_seed is not None else "") + (
        f" --only-cells {only_cells}" if only_cells else "")
    if workers is None:
        workers = 6 if block == "main" else 1
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.hunt3."
            f"run_cnov_panel {workers} {flags}")
    finally:
        lb_delta = (open(LB).readlines()[n0:] if Path(LB).exists() else [])
        outs = sorted(Path(RES_DIR).glob(f"cnov_*{suffix}.json"))
        payload = {
            "block": block, "only_seed": only_seed,
            "results": (json.loads(outs[-1].read_text()) if outs else []),
            "leaderboard_delta": lb_delta,
        }
        dst = Path(vol_dir) / f"payload{suffix}.json"
        dst.write_text(json.dumps(payload))
        vol.commit()
        print(f"[payload] cnov{suffix}: {len(payload['results'])} results, "
              f"{len(lb_delta)} leaderboard rows", flush=True)
    return json.dumps(payload)


@app.function(image=image, gpu="H100", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              cpu=8, memory=65536, timeout=2 * 60 * 60,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_main(workers: int = 0, only_cells: str = "") -> str:
    return _run_block("main", None, workers or None, only_cells)


@app.function(image=image, gpu="L4", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              cpu=8, memory=65536, timeout=2 * 60 * 60,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_tsae(seed: int) -> str:
    return _run_block("tsae", seed)


@app.local_entrypoint()
def main():
    out_dir = Path(__file__).resolve().parents[1] / \
        "experiments/explorations/task_hunt/hunt3/results/cnov_payloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = [("main", run_main.spawn())] + \
        [(f"tsae_s{s}", run_tsae.spawn(s)) for s in SEEDS]
    for name, call in calls:
        try:
            payload = call.get()
            p = out_dir / f"payload_{name}.json"
            p.write_text(payload)
            print(f"[repatriated] {p} ({len(payload)} bytes)", flush=True)
        except Exception as e:                    # noqa: BLE001
            print(f"[FAILED] cnov/{name}: {e!r} — payload persists on "
                  f"Volume /workspace/hunt3_cnov_panel", flush=True)
    print("CNOV PANEL PIPELINE COMPLETE", flush=True)
