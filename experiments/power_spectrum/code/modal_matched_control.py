"""Bounded Modal launcher for the full-band matched-support control."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

from experiments.power_spectrum.code.modal_benchmark import (
    ENV,
    LOCAL_ROOT,
    REMOTE_PYTHON,
    REMOTE_ROOT,
    _stream,
    image,
    volume,
)


REMOTE_CONFIG = (
    REMOTE_ROOT
    / "experiments"
    / "power_spectrum"
    / "configs"
    / "matched_control.json"
)
APP_NAME = "temp-xc-power-spectrum-matched-control"
RUN_DIR = Path("/vol/matched-control-20260729")

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=4_500,
    volumes={"/vol": volume},
)
def run_remote() -> dict:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    command = [
        str(REMOTE_PYTHON),
        "-m",
        "experiments.power_spectrum.code.run_synthetic_benchmark",
        "--config",
        str(REMOTE_CONFIG),
        "--results-dir",
        str(RUN_DIR),
        "--mode",
        "overnight",
    ]
    try:
        _stream(command, timeout=4_300)
    finally:
        volume.commit()
    payload: dict = {"results_dir": str(RUN_DIR)}
    for name in ("plan.json", "gate_report.json", "summary.json", "spend.json"):
        path = RUN_DIR / name
        if path.exists():
            payload[name.removesuffix(".json")] = json.loads(path.read_text())
    return payload


@app.local_entrypoint()
def main(stage: str = "plan", out: str = "") -> None:
    config = LOCAL_ROOT / "experiments/power_spectrum/configs/matched_control.json"
    if stage == "plan":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.power_spectrum.code.run_synthetic_benchmark",
                "--config",
                str(config),
                "--mode",
                "overnight",
                "--dry-run",
            ],
            cwd=LOCAL_ROOT,
            env={**os.environ, **ENV},
            check=True,
        )
        return
    if stage != "overnight":
        raise SystemExit("stage must be one of: plan, overnight")
    call = run_remote.spawn()
    text = json.dumps(
        {"function_call_id": call.object_id, "status": "spawned"},
        indent=2,
        sort_keys=True,
    )
    print(text)
    if out:
        Path(out).write_text(text + "\n")
