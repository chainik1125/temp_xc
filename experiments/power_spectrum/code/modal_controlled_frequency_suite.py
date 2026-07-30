"""Modal launcher for the controlled Shamir/HMM suite and DC/AC replay."""

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


APP_NAME = "temp-xc-controlled-frequency-suite"
REMOTE_CONFIG = (
    REMOTE_ROOT
    / "experiments"
    / "power_spectrum"
    / "configs"
    / "controlled_frequency_suite.json"
)
FULL_RUN_DIR = Path("/vol/controlled-frequency-suite-20260729")
SMOKE_RUN_DIR = Path("/vol/controlled-frequency-suite-smoke-20260729")
USAGE_DIR = Path("/vol/denoising-frequency-usage-20260729")
LOCAL_CONTROLLED_RESULTS = (
    LOCAL_ROOT
    / "experiments"
    / "power_spectrum"
    / "results"
    / "controlled_frequency_suite_remote"
)
LOCAL_USAGE_RESULTS = (
    LOCAL_ROOT
    / "experiments"
    / "power_spectrum"
    / "results"
    / "denoising_frequency_usage_remote"
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=6300,
    volumes={"/vol": volume},
)
def run_controlled_remote(mode: str) -> dict:
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
        _stream(command, timeout=6200)
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
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=1800,
    volumes={"/vol": volume},
)
def run_usage_remote(smoke: bool) -> dict:
    USAGE_DIR.mkdir(parents=True, exist_ok=True)
    name = "smoke.json" if smoke else "result.json"
    command = [
        str(REMOTE_PYTHON),
        "-m",
        "experiments.power_spectrum.code.run_denoising_frequency_usage",
        "--output",
        str(USAGE_DIR / name),
    ]
    if smoke:
        command.extend(["--smoke", "--seeds", "1"])
    try:
        _stream(command, timeout=1750)
    finally:
        volume.commit()
    return {
        "smoke": smoke,
        "output": str(USAGE_DIR / name),
        "result": json.loads((USAGE_DIR / name).read_text()),
    }


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=300,
    volumes={"/vol": volume},
)
def fetch_remote(kind: str) -> dict[str, str]:
    volume.reload()
    if kind == "controlled":
        names = (
            "frozen_config.json",
            "plan.json",
            "results.jsonl",
            "spend.json",
            "summary.json",
        )
        return {
            name: (FULL_RUN_DIR / name).read_text()
            for name in names
            if (FULL_RUN_DIR / name).exists()
        }
    if kind == "usage":
        path = USAGE_DIR / "result.json"
        return {"result.json": path.read_text()} if path.exists() else {}
    raise ValueError(f"unknown fetch kind {kind!r}")


def _spawn_receipt(call, out: str, *, stage: str) -> None:
    payload = {
        "stage": stage,
        "function_call_id": call.object_id,
        "status": "spawned",
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if out:
        Path(out).write_text(text + "\n")


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
                    / "controlled_frequency_suite.json"
                ),
                "--mode",
                "plan",
            ],
            cwd=LOCAL_ROOT,
            check=True,
        )
        return
    if stage == "smoke":
        print(json.dumps(run_controlled_remote.remote("smoke"), indent=2, sort_keys=True))
        return
    if stage == "full":
        _spawn_receipt(run_controlled_remote.spawn("full"), out, stage=stage)
        return
    if stage == "usage-smoke":
        print(json.dumps(run_usage_remote.remote(True), indent=2, sort_keys=True))
        return
    if stage == "usage-full":
        _spawn_receipt(run_usage_remote.spawn(False), out, stage=stage)
        return
    if stage in {"fetch", "usage-fetch"}:
        kind = "controlled" if stage == "fetch" else "usage"
        files = fetch_remote.remote(kind)
        target = (
            LOCAL_CONTROLLED_RESULTS
            if kind == "controlled"
            else LOCAL_USAGE_RESULTS
        )
        target.mkdir(parents=True, exist_ok=True)
        for name, contents in files.items():
            (target / name).write_text(contents)
        print(
            json.dumps(
                {"files": sorted(files), "local_results": str(target)},
                indent=2,
                sort_keys=True,
            )
        )
        return
    raise SystemExit(
        "stage must be one of: plan, smoke, full, fetch, "
        "usage-smoke, usage-full, usage-fetch"
    )
