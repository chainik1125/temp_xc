"""Modal driver — diafaces gated mini-panel (mac-a day-2 W2).

Gate: FIRED in writing (LOG `dce8d085d`); card `diafaces/PANEL_CARD.md`
frozen at PINNED_COMMIT (rev-parse). GPU split per Han's amendment
(`a68c364a3`): ONE H100 container runs `--block main` (99 cells,
GPU-bound pools, workers 6); THREE cheap-GPU high-CPU containers run
`--block tsae --only-seed {1,2,42}` (measured GPU-idle; d 768 buffer
copies ~5× smaller than the 62–77 min d-4096 precedent). Every
container persists its result file + leaderboard delta to the Volume
(payload survives the client — the overnight $4.5 lesson) and the
client repatriates for the LOCAL merge (containers never push).

  modal run --detach scripts/modal_diafaces_panel.py
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "7ba2e10fd2c822d8dac820a307ec4f9f3c4f0005"  # panel FREEZE
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
DS = "dial_real_ttrend_gpt2_l7"
MODEL_KEY = "gpt2"
TSAE_SEEDS = (1, 2, 42)
PY = "/repo/.venv/bin/python"
LB = "/repo/results/leaderboard.jsonl"
RES_DIR = "/repo/experiments/explorations/task_hunt/diafaces/results"
VOL_DIR = "/workspace/diafaces_panel"

app = modal.App("mac-a-diafaces-panel")
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


def _run_block(block: str, only_seed: int | None) -> str:
    _assert_pinned()
    _sh(f"{PY} -m experiments.explorations.task_hunt.dialevel.cache_acts "
        f"{MODEL_KEY}")
    n0 = sum(1 for _ in open(LB)) if Path(LB).exists() else 0
    _sh(f"mkdir -p {VOL_DIR}")
    suffix = f"_{block}" + (f"_s{only_seed}" if only_seed is not None else "")
    flags = f"--block {block}" + (
        f" --only-seed {only_seed}" if only_seed is not None else "")
    workers = 6 if block == "main" else 1
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.diafaces.run_panel "
            f"{workers} {flags}")
    finally:
        lb_delta = (open(LB).readlines()[n0:] if Path(LB).exists() else [])
        out = Path(RES_DIR) / f"panel_{DS}{suffix}.json"
        payload = {
            "block": block, "only_seed": only_seed,
            "results": json.loads(out.read_text()) if out.exists() else [],
            "leaderboard_delta": lb_delta,
        }
        dst = Path(VOL_DIR) / f"payload{suffix}.json"
        dst.write_text(json.dumps(payload))
        vol.commit()
        print(f"[payload] {dst.name}: {len(payload['results'])} results, "
              f"{len(lb_delta)} leaderboard rows", flush=True)
    return json.dumps(payload)


@app.function(image=image, gpu="H100", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              cpu=8, memory=65536, timeout=3 * 60 * 60,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_main() -> str:
    return _run_block("main", None)


@app.function(image=image, gpu="L4", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              cpu=8, memory=32768, timeout=3 * 60 * 60,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_tsae(seed: int) -> str:
    return _run_block("tsae", seed)


@app.local_entrypoint()
def main():
    out_dir = Path(__file__).resolve().parents[1] / \
        "experiments/explorations/task_hunt/diafaces/results/panel_payloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = [("main", run_main.spawn())] + \
        [(f"tsae_s{s}", run_tsae.spawn(s)) for s in TSAE_SEEDS]
    for name, call in calls:
        try:
            payload = call.get()
        except Exception as e:                    # noqa: BLE001
            print(f"[FAILED] {name}: {e!r} — payload persists on Volume "
                  f"{VOL_DIR}", flush=True)
            continue
        p = out_dir / f"payload_{name}.json"
        p.write_text(payload)
        print(f"[repatriated] {p} ({len(payload)} bytes)", flush=True)
    print("DIAFACES PANEL PIPELINE COMPLETE", flush=True)
