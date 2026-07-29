"""Modal wrapper for the cost-gated power-spectrum synthetic benchmark.

No remote work happens on import.  Inspect the exact plan locally first:

    modal run experiments/power_spectrum/code/modal_benchmark.py --stage plan

Then run a paid smoke test, and only after it succeeds launch overnight:

    modal run experiments/power_spectrum/code/modal_benchmark.py --stage smoke
    modal run --detach experiments/power_spectrum/code/modal_benchmark.py \
        --stage overnight

The remote function has a 7h45 hard timeout; the inner runner stops at 7h30
and maintains its own conservative spend ledger.  One A10G is used
sequentially, so there is no fan-out overspend tail.  Checkpoints, JSONL, gate
receipts, and summaries persist in the ``temp-xc-power-spectrum`` volume.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal


LOCAL_ROOT = Path(__file__).resolve().parents[3]
REMOTE_ROOT = Path("/repo")
REMOTE_PYTHON = REMOTE_ROOT / ".venv" / "bin" / "python"
REMOTE_CONFIG = REMOTE_ROOT / "experiments" / "power_spectrum" / "configs" / "overnight.json"
APP_NAME = "temp-xc-power-spectrum-overnight"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("temp-xc-power-spectrum", create_if_missing=True)

# Copy only what the synthetic runner needs.  This avoids uploading the main
# checkout's results/checkpoints and keeps the remote image pinned to the exact
# local source state submitted by `modal run`.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .add_local_file(LOCAL_ROOT / "pyproject.toml", "/repo/pyproject.toml", copy=True)
    .add_local_file(LOCAL_ROOT / "uv.lock", "/repo/uv.lock", copy=True)
    .run_commands("cd /repo && uv sync --frozen --no-install-project")
    .add_local_dir(LOCAL_ROOT / "src", "/repo/src", copy=True)
    .add_local_dir(LOCAL_ROOT / "configs", "/repo/configs", copy=True)
    .add_local_dir(
        LOCAL_ROOT / "experiments" / "power_spectrum",
        "/repo/experiments/power_spectrum",
        copy=True,
    )
)

ENV = {
    "PYTHONPATH": "/repo/src:/repo",
    "TEMP_BENCH_ALLOW_DIRTY": "1",
    "TQDM_DISABLE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
}


def _stream(cmd: list[str], *, timeout: int) -> None:
    env = {**os.environ, **ENV}
    process = subprocess.Popen(
        cmd,
        cwd=REMOTE_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        returncode = process.wait(timeout=timeout)
    except BaseException:
        process.kill()
        raise
    if returncode:
        raise RuntimeError(f"runner exited {returncode}: {' '.join(cmd)}")


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=27_900,  # 7h45; inner scientific deadline is 7h30.
    volumes={"/vol": volume},
)
def run_remote(mode: str) -> dict:
    if mode not in {"smoke", "gate", "overnight"}:
        raise ValueError(f"unsupported paid mode {mode!r}")
    run_dir = Path("/vol") / "overnight-20260729"
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(REMOTE_PYTHON),
        "-m",
        "experiments.power_spectrum.code.run_synthetic_benchmark",
        "--config",
        str(REMOTE_CONFIG),
        "--results-dir",
        str(run_dir),
        "--mode",
        mode,
    ]
    try:
        _stream(cmd, timeout=27_600)
    finally:
        volume.commit()
    payload: dict = {"mode": mode, "results_dir": str(run_dir)}
    for name in ("plan.json", "gate_report.json", "summary.json", "spend.json"):
        path = run_dir / name
        if path.exists():
            payload[name.removesuffix(".json")] = json.loads(path.read_text())
    return payload


@app.local_entrypoint()
def main(stage: str = "plan", out: str = ""):
    if stage == "plan":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.power_spectrum.code.run_synthetic_benchmark",
                "--config",
                str(LOCAL_ROOT / "experiments/power_spectrum/configs/overnight.json"),
                "--mode",
                "overnight",
                "--dry-run",
            ],
            cwd=LOCAL_ROOT,
            check=True,
        )
        return
    if stage not in {"smoke", "gate", "overnight"}:
        raise SystemExit("stage must be one of: plan, smoke, gate, overnight")
    result = run_remote.remote(stage)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if out:
        Path(out).write_text(text + "\n")
