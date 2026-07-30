"""Modal launcher for the learned-routing Spectral Matryoshka screen."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

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


APP_NAME = "temp-xc-spectral-matryoshka"
REMOTE_CONFIG = (
    REMOTE_ROOT
    / "experiments"
    / "power_spectrum"
    / "configs"
    / "spectral_matryoshka_routed.json"
)
FULL_RUN_DIR = Path("/vol/spectral-matryoshka-routed-20260730")
CONFIRM_RUN_DIR = Path("/vol/spectral-matryoshka-routed-confirm-seed2-20260730")
SAE_RUN_DIR = Path("/vol/spectral-matryoshka-sae-baseline-20260730")
# Smoke and full results share one spend ledger. Their cell/checkpoint IDs are
# disjoint, so this enforces one cumulative hard cap without contaminating the
# non-smoke summary.
SMOKE_RUN_DIR = FULL_RUN_DIR
LOCAL_RESULTS = (
    LOCAL_ROOT
    / "experiments"
    / "power_spectrum"
    / "results"
    / "spectral_matryoshka_routed_remote"
)
LOCAL_CONFIRM_RESULTS = (
    LOCAL_ROOT
    / "experiments"
    / "power_spectrum"
    / "results"
    / "spectral_matryoshka_routed_seed2_remote"
)
LOCAL_SAE_RESULTS = (
    LOCAL_ROOT
    / "experiments"
    / "power_spectrum"
    / "results"
    / "spectral_matryoshka_sae_remote"
)

app = modal.App(APP_NAME)


def _confirmation_config(source: Path) -> dict[str, Any]:
    """Make the seed-2 confirmation config with an independent hard ledger."""

    config: dict[str, Any] = json.loads(source.read_text())
    config["run_name"] = "spectral-matryoshka-routed-confirm-seed2-20260730"
    config["seeds"] = [2]
    config["smoke"]["seed"] = 2
    config["overall_spend"] = {
        "cap_usd": 50.0,
        "estimated_prior_usd": 45.852423,
        "note": (
            "Conservative cumulative estimate through the routed seed-1 smoke/full "
            "run. The seed-2 ledger has a $3.50 hard stop, leaving at least $0.65 "
            "under the overall cap."
        ),
    }
    config["budget"].update(
        {
            "max_total_usd": 3.5,
            "reserve_usd": 0.3,
            "max_session_hours": 0.75,
        }
    )
    # The identical seed-1 panel sustained roughly 83 training steps/second
    # including orchestration overhead. Sixty is a conservative measured-rate
    # estimate and keeps planning useful without weakening the hard ledger.
    config["planning"]["estimated_steps_per_second"] = 60.0
    return config


def _sae_baseline_config(
    source: Path,
    *,
    estimated_prior_usd: float,
) -> dict[str, Any]:
    """Make a matched-parameter token-SAE panel for the sparse tasks."""

    if not 0.0 < estimated_prior_usd < 48.6:
        raise ValueError(
            "estimated_prior_usd must leave room for the $1.40 baseline hard cap"
        )
    config: dict[str, Any] = json.loads(source.read_text())
    config["run_name"] = "spectral-matryoshka-matched-sae-20260730"
    config["overall_spend"] = {
        "cap_usd": 50.0,
        "estimated_prior_usd": estimated_prior_usd,
        "note": (
            "Conservative cumulative estimate through both routed spectral seeds. "
            "This final matched-SAE ledger has an independent $1.40 hard stop."
        ),
    }
    config["budget"].update(
        {
            "max_total_usd": 1.4,
            "reserve_usd": 0.15,
            "max_session_hours": 0.3,
        }
    )
    config["planning"]["estimated_steps_per_second"] = 60.0
    config["models"] = [
        {
            "name": "sae",
            "label": "BatchTopK SAE",
            "class_path": "temp_bench.archs.batchtopk_sae:BatchTopKSAE",
            "implementation_version": "1.0.0",
            "consumes": "token",
            "sparsity_rule": "one_per_token",
            "hparams": {"auxk_alpha": 0.03125},
        }
    ]
    config["tasks"] = [
        task
        for task in config["tasks"]
        if task["name"]
        in {
            "narrowband_sparse_balanced_t8",
            "narrowband_sparse_high_crowded_t8",
        }
    ]
    for task in config["tasks"]:
        task["models"] = ["sae"]
        task["d_sae_by_model"] = {"sae": 512}
        task.pop("hparams_by_model", None)
    config["seeds"] = [1, 2]
    config["smoke"].update(
        {
            "tasks": [task["name"] for task in config["tasks"]],
            "models": ["sae"],
            "seed": 1,
            "n_steps": 2,
        }
    )
    config["fairness"].update(
        {
            "support": (
                "The token SAE uses one active feature per token, hence nominal "
                "L0=W per analysis window, matching TXC and the spectral arms."
            ),
            "parameters": (
                "The token SAE uses 512 atoms (49,712 trainable parameters), "
                "within 0.4% of TXC (49,600) and the spectral arms (49,503-49,507)."
            ),
            "attribution": (
                "This isolated panel adds only the missing matched-parameter "
                "token-SAE baseline on the exact same frozen sparse tasks."
            ),
        }
    )
    return config


def _local_config_path() -> Path:
    return (
        LOCAL_ROOT
        / "experiments"
        / "power_spectrum"
        / "configs"
        / "spectral_matryoshka_routed.json"
    )


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=4000,
    volumes={"/vol": volume},
)
def run_remote(mode: str, confirm: bool = False) -> dict:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"unsupported mode {mode!r}")
    if confirm and mode != "full":
        raise ValueError("the already-smoke-tested confirmation supports full mode only")
    run_dir = (
        CONFIRM_RUN_DIR
        if confirm
        else (SMOKE_RUN_DIR if mode == "smoke" else FULL_RUN_DIR)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = REMOTE_CONFIG
    if confirm:
        config_path = run_dir / "launch_config.json"
        config_path.write_text(
            json.dumps(_confirmation_config(REMOTE_CONFIG), indent=2, sort_keys=True) + "\n"
        )
    command = [
        str(REMOTE_PYTHON),
        "-m",
        "experiments.power_spectrum.code.run_controlled_frequency_suite",
        "--config",
        str(config_path),
        "--results-dir",
        str(run_dir),
        "--mode",
        mode,
    ]
    try:
        _stream(command, timeout=3900)
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
def fetch_remote(confirm: bool = False) -> dict[str, str]:
    volume.reload()
    run_dir = CONFIRM_RUN_DIR if confirm else FULL_RUN_DIR
    names = (
        "frozen_config.json",
        "plan.json",
        "results.jsonl",
        "spend.json",
        "summary.json",
    )
    return {
        name: (run_dir / name).read_text() for name in names if (run_dir / name).exists()
    }


@app.function(
    image=image,
    gpu="A10G",
    cpu=8,
    memory=32768,
    timeout=1800,
    volumes={"/vol": volume},
)
def run_sae_remote(mode: str, estimated_prior_usd: float) -> dict:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"unsupported mode {mode!r}")
    SAE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    config_path = SAE_RUN_DIR / "launch_config.json"
    config_path.write_text(
        json.dumps(
            _sae_baseline_config(
                REMOTE_CONFIG,
                estimated_prior_usd=estimated_prior_usd,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    command = [
        str(REMOTE_PYTHON),
        "-m",
        "experiments.power_spectrum.code.run_controlled_frequency_suite",
        "--config",
        str(config_path),
        "--results-dir",
        str(SAE_RUN_DIR),
        "--mode",
        mode,
    ]
    try:
        _stream(command, timeout=1700)
    finally:
        volume.commit()
    payload: dict = {"mode": mode, "results_dir": str(SAE_RUN_DIR)}
    for name in ("plan.json", "summary.json", "spend.json"):
        path = SAE_RUN_DIR / name
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
def fetch_sae_remote() -> dict[str, str]:
    volume.reload()
    names = (
        "frozen_config.json",
        "plan.json",
        "results.jsonl",
        "spend.json",
        "summary.json",
    )
    return {
        name: (SAE_RUN_DIR / name).read_text()
        for name in names
        if (SAE_RUN_DIR / name).exists()
    }


@app.local_entrypoint()
def main(stage: str = "plan", out: str = "", prior_usd: float = 0.0) -> None:
    if stage in {"plan", "confirm-plan", "sae-plan"}:
        confirm = stage == "confirm-plan"
        config_path = _local_config_path()
        temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        if confirm or stage == "sae-plan":
            temporary_directory = tempfile.TemporaryDirectory(
                prefix=f"spectral-matryoshka-{stage}-"
            )
            config_path = Path(temporary_directory.name) / "config.json"
            derived_config = (
                _confirmation_config(_local_config_path())
                if confirm
                else _sae_baseline_config(
                    _local_config_path(),
                    estimated_prior_usd=prior_usd,
                )
            )
            config_path.write_text(
                json.dumps(derived_config, indent=2, sort_keys=True)
                + "\n"
            )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.power_spectrum.code.run_controlled_frequency_suite",
                "--config",
                str(config_path),
                "--mode",
                "plan",
            ],
            cwd=LOCAL_ROOT,
            check=True,
        )
        if temporary_directory is not None:
            temporary_directory.cleanup()
        return
    if stage == "smoke":
        print(json.dumps(run_remote.remote("smoke"), indent=2, sort_keys=True))
        return
    if stage in {"full", "confirm-full"}:
        confirm = stage == "confirm-full"
        call = run_remote.spawn("full", confirm)
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
    if stage in {"fetch", "confirm-fetch"}:
        confirm = stage == "confirm-fetch"
        files = fetch_remote.remote(confirm)
        local_results = LOCAL_CONFIRM_RESULTS if confirm else LOCAL_RESULTS
        local_results.mkdir(parents=True, exist_ok=True)
        for name, contents in files.items():
            (local_results / name).write_text(contents)
        print(
            json.dumps(
                {"files": sorted(files), "local_results": str(local_results)},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if stage == "sae-smoke":
        print(
            json.dumps(
                run_sae_remote.remote("smoke", prior_usd),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if stage == "sae-full":
        call = run_sae_remote.spawn("full", prior_usd)
        payload = {
            "stage": stage,
            "function_call_id": call.object_id,
            "status": "spawned",
            "estimated_prior_usd": prior_usd,
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if out:
            Path(out).write_text(text + "\n")
        return
    if stage == "sae-fetch":
        files = fetch_sae_remote.remote()
        LOCAL_SAE_RESULTS.mkdir(parents=True, exist_ok=True)
        for name, contents in files.items():
            (LOCAL_SAE_RESULTS / name).write_text(contents)
        print(
            json.dumps(
                {
                    "files": sorted(files),
                    "local_results": str(LOCAL_SAE_RESULTS),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    raise SystemExit(
        "stage must be one of: plan, smoke, full, fetch, "
        "confirm-plan, confirm-full, confirm-fetch, "
        "sae-plan, sae-smoke, sae-full, sae-fetch"
    )
