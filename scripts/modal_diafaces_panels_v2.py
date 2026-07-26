"""Modal driver — diafaces panels RE-RUN with paired v2 columns
(the v2-defect amendment, LOG 2026-07-26 mac-a; freeze db677a4b8).

One app file, both frozen panels, selected per invocation:

  modal run --detach scripts/modal_diafaces_panels_v2.py --panel dq
  modal run --detach scripts/modal_diafaces_panels_v2.py --panel tt

Same ops as the first drivers: H100 main block (99 cells, workers 6)
+ 3× high-CPU L4 64 GB tsae; payloads persist to the Volume per
panel; containers never push; local merge `merge_panel_payload.py
tt|dq` (freezes updated to db677a4b8).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "931c016e63d4755d142a9eb25600f5026887c9a6"  # re-pass PIN (contains --only-cells; cells enumeration unchanged)
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
PANELS = {
    "tt": {"ds": "dial_real_ttrend_gpt2_l7", "model_key": "gpt2"},
    "dq": {"ds": "dial_real_dqgap_llama31_8b_l14", "model_key": "llama31_8b"},
}
TSAE_SEEDS = (1, 2, 42)
PY = "/repo/.venv/bin/python"
LB = "/repo/results/leaderboard.jsonl"
RES_DIR = "/repo/experiments/explorations/task_hunt/diafaces/results"

app = modal.App("mac-a-diafaces-panels-v2")
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


def _run_block(panel: str, block: str, only_seed: int | None,
               workers: int | None = None, only_cells: str = "") -> str:
    _assert_pinned()
    cfg = PANELS[panel]
    _sh(f"{PY} -m experiments.explorations.task_hunt.dialevel.cache_acts "
        f"{cfg['model_key']}")
    n0 = sum(1 for _ in open(LB)) if Path(LB).exists() else 0
    vol_dir = f"/workspace/diafaces_panels_v2/{panel}"
    _sh(f"mkdir -p {vol_dir}")
    suffix = f"_{block}" + (f"_s{only_seed}" if only_seed is not None else "") \
        + ("_repass" if only_cells else "")
    flags = f"--panel {panel} --block {block}" + (
        f" --only-seed {only_seed}" if only_seed is not None else "") + (
        f" --only-cells {only_cells}" if only_cells else "")
    if workers is None:
        # 6 H100 workers OOM'd dq's d4096 T32 pooled trained cells
        # (~13 GB/worker); scheduling is NOT frozen config — outputs
        # unaffected (the batch-halving-class pre-authorization).
        workers = (3 if panel == "dq" else 6) if block == "main" else 1
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.diafaces.run_panel "
            f"{workers} {flags}")
    finally:
        lb_delta = (open(LB).readlines()[n0:] if Path(LB).exists() else [])
        out = Path(RES_DIR) / f"panel_{cfg['ds']}{suffix}.json"
        payload = {
            "panel": panel, "block": block, "only_seed": only_seed,
            "results": json.loads(out.read_text()) if out.exists() else [],
            "leaderboard_delta": lb_delta,
        }
        dst = Path(vol_dir) / f"payload{suffix}.json"
        dst.write_text(json.dumps(payload))
        vol.commit()
        print(f"[payload] {panel}{suffix}: {len(payload['results'])} results, "
              f"{len(lb_delta)} leaderboard rows", flush=True)
    return json.dumps(payload)


@app.function(image=image, gpu="H100", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              cpu=8, memory=65536, timeout=3 * 60 * 60,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_main(panel: str, workers: int = 0, only_cells: str = "") -> str:
    return _run_block(panel, "main", None, workers or None, only_cells)


@app.function(image=image, gpu="L4", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              cpu=8, memory=65536, timeout=3 * 60 * 60,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_tsae(panel: str, seed: int) -> str:
    return _run_block(panel, "tsae", seed)


@app.local_entrypoint()
def main(panel: str = "dq"):
    out_dir = Path(__file__).resolve().parents[1] / \
        "experiments/explorations/task_hunt/diafaces/results" / \
        (f"panel2_payloads" if panel == "dq" else "panel_payloads_v2tt")
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = [("main", run_main.spawn(panel))] + \
        [(f"tsae_s{s}", run_tsae.spawn(panel, s)) for s in TSAE_SEEDS]
    for name, call in calls:
        try:
            payload = call.get()
        except Exception as e:                    # noqa: BLE001
            print(f"[FAILED] {panel}/{name}: {e!r} — payload persists on "
                  f"Volume /workspace/diafaces_panels_v2/{panel}", flush=True)
            continue
        p = out_dir / f"payload_{name}.json"
        p.write_text(payload)
        print(f"[repatriated] {p} ({len(payload)} bytes)", flush=True)
    print(f"DIAFACES {panel.upper()} V2 PIPELINE COMPLETE", flush=True)
