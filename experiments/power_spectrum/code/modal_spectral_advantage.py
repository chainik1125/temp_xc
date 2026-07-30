"""Modal launcher for the matched-parameter spectral-advantage experiment."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal


REMOTE_ROOT = Path("/repo")
if REMOTE_ROOT.is_dir():
    sys.path.insert(0, str(REMOTE_ROOT))

from experiments.power_spectrum.code.modal_benchmark import (  # noqa: E402
    LOCAL_ROOT,
    REMOTE_PYTHON,
    _stream,
    image,
    volume,
)


APP_NAME = "temp-xc-spectral-advantage"
REMOTE_CONFIG = (
    REMOTE_ROOT / "experiments" / "power_spectrum" / "configs" / "spectral_advantage.json"
)
FULL_RUN_DIR = Path("/vol/spectral-advantage-20260730")
SMOKE_RUN_DIR = Path("/vol/spectral-advantage-smoke-20260730")
LOCAL_RESULTS = (
    LOCAL_ROOT / "experiments" / "power_spectrum" / "results" / "spectral_advantage_remote"
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=4500,
    volumes={"/vol": volume},
)
def run_remote(mode: str) -> dict:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"unsupported mode {mode!r}")
    run_dir = SMOKE_RUN_DIR if mode == "smoke" else FULL_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(REMOTE_PYTHON),
        "-m",
        "experiments.power_spectrum.code.run_controlled_frequency_suite",
        "--config",
        str(REMOTE_CONFIG),
        "--results-dir",
        str(run_dir),
        "--mode",
        mode,
    ]
    try:
        _stream(command, timeout=4400)
    finally:
        volume.commit()
    payload: dict = {"mode": mode, "results_dir": str(run_dir)}
    for name in ("plan.json", "summary.json", "spend.json"):
        path = run_dir / name
        if path.exists():
            payload[name.removesuffix(".json")] = json.loads(path.read_text())
    return payload


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=300,
    volumes={"/vol": volume},
)
def fetch_remote() -> dict[str, str]:
    volume.reload()
    names = (
        "frozen_config.json",
        "plan.json",
        "results.jsonl",
        "spend.json",
        "summary.json",
    )
    return {
        name: (FULL_RUN_DIR / name).read_text() for name in names if (FULL_RUN_DIR / name).exists()
    }


@app.local_entrypoint()
def main(stage: str = "plan", out: str = "") -> None:
    if stage == "plan":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.power_spectrum.code.run_controlled_frequency_suite",
                "--config",
                str(
                    LOCAL_ROOT
                    / "experiments"
                    / "power_spectrum"
                    / "configs"
                    / "spectral_advantage.json"
                ),
                "--mode",
                "plan",
            ],
            cwd=LOCAL_ROOT,
            check=True,
        )
        return
    if stage == "smoke":
        print(json.dumps(run_remote.remote("smoke"), indent=2, sort_keys=True))
        return
    if stage == "full":
        call = run_remote.spawn("full")
        payload = {
            "stage": stage,
            "function_call_id": call.object_id,
            "status": "spawned",
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if out:
            Path(out).write_text(text + "\n")
        return
    if stage == "fetch":
        files = fetch_remote.remote()
        LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
        for name, contents in files.items():
            (LOCAL_RESULTS / name).write_text(contents)
        print(
            json.dumps(
                {"files": sorted(files), "local_results": str(LOCAL_RESULTS)},
                indent=2,
                sort_keys=True,
            )
        )
        return
    raise SystemExit("stage must be one of: plan, smoke, full, fetch")
