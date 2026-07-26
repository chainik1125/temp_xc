"""Modal launcher for the missing full backtracking window-sweep cells.

The scientific protocol remains in :mod:`experiments.backtracking_window_sweep`.
This module only supplies immutable public assets, isolated ephemeral work
directories, GPU dispatch, and compact completed-result persistence.

Typical use from the repository root::

    modal run purified/experiments/backtracking_window_sweep/modal_app.py::seed
    modal run purified/experiments/backtracking_window_sweep/modal_app.py::benchmark
    modal run --detach purified/experiments/backtracking_window_sweep/modal_app.py::launch
    modal run purified/experiments/backtracking_window_sweep/modal_app.py::status
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "txc-neurips-backtracking-window-sweep"
ASSET_VOLUME_NAME = "txc-neurips-backtracking-assets"
RESULTS_VOLUME_NAME = "txc-neurips-backtracking-results-v2"
CLI_PATH = "purified/experiments/backtracking_window_sweep/modal_app.py"

HF_REPO_ID = "han1823123123/temp-bench-data"
HF_REPO_TYPE = "dataset"
HF_REVISION = "6ef9b1debf863dedcef9555cad3a4903fb9e8c43"
PROTOCOL_VERSION = "2026-07-23.2"

ARTIFACT_SPECS: dict[str, dict[str, Any]] = {
    "event_artifact": {
        "filename": "c7_backtracking/stage_a/sentence_acts_L10.npz",
        "sha256": "1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810",
        "size_bytes": 1_137_333_114,
        "shape": [25_204, 6, 4_096],
    },
    "training_cache": {
        "filename": "act_cache/fb2a74be884e512a/resid_post_L10.npy",
        "sha256": "dc34dfb117f77abddef4b4396d0d00afc707c39876d0ee36015de1e7b8406914",
        "size_bytes": 4_240_441_472,
        "shape": [4_044, 128, 4_096],
    },
}

# Seven other cells already exist locally: all T=1, T=2 seed 42, and all T=6.
MISSING_CELLS: tuple[tuple[int, int], ...] = (
    (2, 1),
    (2, 2),
    (3, 1),
    (3, 2),
    (3, 42),
    (4, 1),
    (4, 2),
    (4, 42),
    (5, 1),
    (5, 2),
    (5, 42),
)
ALL_CELLS = tuple((window, seed) for window in range(1, 7) for seed in (1, 2, 42))

REMOTE_PURIFIED_ROOT = Path("/repo/purified")
REMOTE_EXISTING_CELLS = Path("/repo/existing_cells")
ASSET_MOUNT = Path("/vol/assets")
ASSET_REPO_ROOT = ASSET_MOUNT / "hf_repo"
RESULTS_MOUNT = Path("/vol/results")

if modal.is_local():
    LOCAL_PURIFIED_ROOT = Path(__file__).resolve().parents[2]
    LOCAL_EXISTING_CELLS = (
        LOCAL_PURIFIED_ROOT
        / "results/neurips_rebuttal/backtracking_window_sweep/full/cells"
    )
else:
    # Modal imports this module from /root/modal_app.py inside containers. The
    # source directories have already been copied to their remote image paths.
    LOCAL_PURIFIED_ROOT = REMOTE_PURIFIED_ROOT
    LOCAL_EXISTING_CELLS = REMOTE_EXISTING_CELLS


ASSET_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy>=2.1",
        "huggingface-hub>=0.26",
        "hf-xet>=1.1",
    )
)

SWEEP_IMAGE = (
    ASSET_IMAGE
    .uv_pip_install(
        "torch==2.8.0+cu128",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "scipy>=1.14",
        "scikit-learn>=1.5",
        "safetensors>=0.4",
        "matplotlib>=3.9",
    )
    .add_local_dir(
        LOCAL_PURIFIED_ROOT / "src",
        remote_path=str(REMOTE_PURIFIED_ROOT / "src"),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        LOCAL_PURIFIED_ROOT / "experiments",
        remote_path=str(REMOTE_PURIFIED_ROOT / "experiments"),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        LOCAL_EXISTING_CELLS,
        remote_path=str(REMOTE_EXISTING_CELLS),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .env(
        {
            "PYTHONPATH": f"{REMOTE_PURIFIED_ROOT / 'src'}:{REMOTE_PURIFIED_ROOT}",
            "TXC_RUNPOD_ROOT": "/repo",
            "MPLBACKEND": "Agg",
        }
    )
)

app = modal.App(APP_NAME)
asset_volume = modal.Volume.from_name(ASSET_VOLUME_NAME, create_if_missing=True)
results_volume = modal.Volume.from_name(
    RESULTS_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
read_only_assets = (
    asset_volume.with_mount_options(read_only=True)
    if hasattr(asset_volume, "with_mount_options")
    else asset_volume.read_only()
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cell_name(window: int, seed: int) -> str:
    return f"T{window}_seed{seed}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _asset_path(spec: dict[str, Any]) -> Path:
    return ASSET_REPO_ROOT / str(spec["filename"])


def _retrieval_metadata() -> dict[str, str]:
    return {
        "volume": RESULTS_VOLUME_NAME,
        "summary_command": (
            f"modal volume get {RESULTS_VOLUME_NAME} aggregate/full "
            "purified/results/neurips_rebuttal/backtracking_window_sweep/modal-full"
        ),
        "completed_cells_command": (
            f"modal volume get {RESULTS_VOLUME_NAME} cells "
            "purified/results/neurips_rebuttal/backtracking_window_sweep/modal-cells"
        ),
        "status_command": f"modal run {CLI_PATH}::status",
    }


@app.function(
    image=ASSET_IMAGE,
    cpu=2,
    memory=8_192,
    timeout=3_600,
    volumes={ASSET_MOUNT: asset_volume},
)
def seed_assets() -> dict[str, Any]:
    """Download and verify the two immutable public Hugging Face artifacts."""

    import numpy as np
    from huggingface_hub import hf_hub_download

    resolved: dict[str, dict[str, Any]] = {}
    ASSET_REPO_ROOT.mkdir(parents=True, exist_ok=True)
    for name, spec in ARTIFACT_SPECS.items():
        expected_path = _asset_path(spec)
        if not expected_path.exists():
            downloaded = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=str(spec["filename"]),
                    repo_type=HF_REPO_TYPE,
                    revision=HF_REVISION,
                    local_dir=str(ASSET_REPO_ROOT),
                )
            )
            if downloaded.resolve() != expected_path.resolve():
                raise RuntimeError(
                    f"Hugging Face placed {name} at {downloaded}, expected {expected_path}"
                )
        size_bytes = expected_path.stat().st_size
        digest = _sha256(expected_path)
        if size_bytes != int(spec["size_bytes"]) or digest != spec["sha256"]:
            raise ValueError(
                f"{name} provenance mismatch: size={size_bytes}, sha256={digest}"
            )
        resolved[name] = {
            "path": str(expected_path),
            "filename": spec["filename"],
            "size_bytes": size_bytes,
            "sha256": digest,
        }

    with np.load(_asset_path(ARTIFACT_SPECS["event_artifact"]), allow_pickle=True) as data:
        event_shape = [int(value) for value in data["X"].shape]
        event_keys = sorted(data.files)
    cache = np.load(
        _asset_path(ARTIFACT_SPECS["training_cache"]),
        mmap_mode="r",
    )
    cache_shape = [int(value) for value in cache.shape]
    if event_shape != ARTIFACT_SPECS["event_artifact"]["shape"]:
        raise ValueError(f"event artifact shape mismatch: {event_shape}")
    if event_keys != ["X", "is_bt", "keys"]:
        raise ValueError(f"event artifact keys mismatch: {event_keys}")
    if cache_shape != ARTIFACT_SPECS["training_cache"]["shape"]:
        raise ValueError(f"training cache shape mismatch: {cache_shape}")

    manifest = {
        "status": "complete",
        "seeded_at": _utc_now(),
        "repo_id": HF_REPO_ID,
        "repo_type": HF_REPO_TYPE,
        "revision": HF_REVISION,
        "files": resolved,
    }
    _atomic_json(ASSET_MOUNT / "manifest.json", manifest)
    asset_volume.commit()
    return manifest


def _read_asset_manifest() -> dict[str, Any]:
    path = ASSET_MOUNT / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run `modal run {CLI_PATH}::seed` first"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("revision") != HF_REVISION:
        raise ValueError(f"invalid asset manifest at {path}")
    for name, spec in ARTIFACT_SPECS.items():
        recorded = manifest.get("files", {}).get(name, {})
        if (
            recorded.get("sha256") != spec["sha256"]
            or int(recorded.get("size_bytes", -1)) != int(spec["size_bytes"])
        ):
            raise ValueError(f"asset manifest mismatch for {name}")
        if not _asset_path(spec).exists():
            raise FileNotFoundError(_asset_path(spec))
    return manifest


def _stage_assets(work_root: Path) -> tuple[Path, Path]:
    """Copy random-access inputs from the Volume to the worker's local SSD."""

    _read_asset_manifest()
    local_assets = work_root / "assets"
    local_assets.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for name, spec in ARTIFACT_SPECS.items():
        destination = local_assets / Path(str(spec["filename"])).name
        shutil.copyfile(_asset_path(spec), destination)
        if destination.stat().st_size != int(spec["size_bytes"]):
            raise IOError(f"short local copy for {name}: {destination}")
        staged[name] = destination
    return staged["event_artifact"], staged["training_cache"]


def _gpu_metadata() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    props = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "memory_bytes": int(props.total_memory),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _runner_command(
    *,
    window: int,
    seed: int,
    artifact: Path,
    cache: Path,
    output_root: Path,
    checkpoint_root: Path,
    phase: str,
    steps: int | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.backtracking_window_sweep.run",
        "--mode",
        "full",
        "--phase",
        phase,
        "--windows",
        str(window),
        "--seeds",
        str(seed),
        "--artifact",
        str(artifact),
        "--activation-cache",
        str(cache),
        "--output-root",
        str(output_root),
        "--checkpoint-root",
        str(checkpoint_root),
        "--device",
        "cuda:0",
    ]
    if steps is not None:
        command.extend(["--steps", str(steps)])
    return command


def _valid_completed_result(path: Path, *, window: int, seed: int) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        payload.get("status") == "complete"
        and payload.get("protocol_version") == PROTOCOL_VERSION
        and payload.get("artifact_sha256")
        == ARTIFACT_SPECS["event_artifact"]["sha256"]
        and int(payload.get("window", -1)) == window
        and int(payload.get("seed", -1)) == seed
    )


def _write_cell_status(cell: str, payload: dict[str, Any]) -> None:
    _atomic_json(RESULTS_MOUNT / "status" / f"{cell}.json", payload)
    results_volume.commit()


def _directory_inventory(path: Path) -> dict[str, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def _persist_completed_result(
    source: Path,
    *,
    window: int,
    seed: int,
) -> dict[str, Any]:
    """Persist result JSON and held-out predictions, but never code caches/checkpoints."""

    cell = _cell_name(window, seed)
    result_path = source / "result.json"
    if not _valid_completed_result(result_path, window=window, seed=seed):
        raise ValueError(f"invalid completed result at {result_path}")

    destination = RESULTS_MOUNT / "cells" / cell
    if _valid_completed_result(
        destination / "result.json",
        window=window,
        seed=seed,
    ):
        return {"path": str(destination), **_directory_inventory(destination)}
    if destination.exists():
        raise ValueError(f"refusing to overwrite invalid result directory {destination}")

    compact_local = source.parent / f"{cell}-compact"
    compact_local.mkdir(parents=True, exist_ok=False)
    shutil.copy2(result_path, compact_local / "result.json")
    predictions = source / "predictions"
    if predictions.exists():
        shutil.copytree(predictions, compact_local / "predictions")

    staging = RESULTS_MOUNT / ".staging" / f"{cell}-{uuid.uuid4().hex}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(compact_local, staging)
    try:
        os.replace(staging, destination)
    except OSError:
        if _valid_completed_result(
            destination / "result.json",
            window=window,
            seed=seed,
        ):
            shutil.rmtree(staging, ignore_errors=True)
        else:
            raise
    return {"path": str(destination), **_directory_inventory(destination)}


def _run_cell_impl(
    *,
    window: int,
    seed: int,
    phase: str,
    steps: int | None,
    persist: bool,
    requested_gpu: str,
) -> dict[str, Any]:
    if window not in range(1, 7):
        raise ValueError(f"window must be in 1..6, got {window}")
    if seed not in (1, 2, 42):
        raise ValueError(f"seed must be one of 1,2,42, got {seed}")
    if phase not in {"train", "all"}:
        raise ValueError(f"unsupported phase {phase}")
    if steps is not None and steps < 1:
        raise ValueError("steps must be positive")

    cell = _cell_name(window, seed)
    if persist:
        results_volume.reload()
        persisted_result = RESULTS_MOUNT / "cells" / cell / "result.json"
        if _valid_completed_result(persisted_result, window=window, seed=seed):
            return {
                "status": "complete",
                "cell": cell,
                "cached": True,
                "result_path": str(persisted_result.parent),
                "retrieval": _retrieval_metadata(),
            }

    started_at = _utc_now()
    started = time.monotonic()
    gpu = _gpu_metadata()
    base_status = {
        "cell": cell,
        "window": window,
        "seed": seed,
        "phase": phase,
        "steps": steps,
        "requested_gpu": requested_gpu,
        "gpu": gpu,
        "protocol_version": PROTOCOL_VERSION,
        "artifact_revision": HF_REVISION,
        "started_at": started_at,
    }
    if persist:
        _write_cell_status(cell, {"status": "running", **base_status})

    try:
        with tempfile.TemporaryDirectory(prefix=f"txc-{cell}-", dir="/tmp") as temporary:
            work_root = Path(temporary)
            artifact, cache = _stage_assets(work_root)
            output_root = work_root / "results"
            checkpoint_root = work_root / "checkpoints"
            command = _runner_command(
                window=window,
                seed=seed,
                artifact=artifact,
                cache=cache,
                output_root=output_root,
                checkpoint_root=checkpoint_root,
                phase=phase,
                steps=steps,
            )
            subprocess.run(command, check=True)
            elapsed = time.monotonic() - started

            payload: dict[str, Any] = {
                "status": "complete",
                **base_status,
                "finished_at": _utc_now(),
                "elapsed_seconds": elapsed,
                "dictionary_steps_per_second": (
                    (2.0 * steps / elapsed) if phase == "train" and steps else None
                ),
                "estimated_full_train_hours": (
                    (elapsed * 20_000 / steps / 3_600)
                    if phase == "train" and steps
                    else None
                ),
            }
            if persist:
                cell_output = output_root / "cells" / cell
                persisted = _persist_completed_result(
                    cell_output,
                    window=window,
                    seed=seed,
                )
                payload["persisted_result"] = persisted
                payload["retrieval"] = _retrieval_metadata()
                _write_cell_status(cell, payload)
            return payload
    except Exception as error:
        if persist:
            _write_cell_status(
                cell,
                {
                    "status": "failed",
                    **base_status,
                    "finished_at": _utc_now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "error_type": type(error).__name__,
                    "error": str(error)[:2_000],
                },
            )
        raise


@app.function(
    image=SWEEP_IMAGE,
    gpu="L40S",
    cpu=4,
    memory=32_768,
    timeout=12 * 60 * 60,
    retries=modal.Retries(
        max_retries=1,
        backoff_coefficient=2.0,
        initial_delay=60.0,
        max_delay=60.0,
    ),
    max_containers=len(MISSING_CELLS),
    volumes={
        ASSET_MOUNT: read_only_assets,
        RESULTS_MOUNT: results_volume,
    },
)
def run_cell(window: int, seed: int) -> dict[str, Any]:
    """Run one full production cell on L40S and persist only compact outputs."""

    return _run_cell_impl(
        window=window,
        seed=seed,
        phase="all",
        steps=None,
        persist=True,
        requested_gpu="L40S",
    )


@app.function(
    image=ASSET_IMAGE,
    cpu=1,
    memory=1_024,
    timeout=12 * 60 * 60,
    max_containers=1,
    volumes={RESULTS_MOUNT: results_volume},
)
def dispatch_sweep(cells: list[tuple[int, int]]) -> dict[str, Any]:
    """Keep one detached parent alive while all GPU children finish."""

    dispatch_id = uuid.uuid4().hex
    status_path = RESULTS_MOUNT / "dispatch" / f"{dispatch_id}.json"
    calls: list[tuple[str, Any]] = []
    for window, seed in cells:
        cell = _cell_name(window, seed)
        call = run_cell.spawn(window, seed)
        calls.append((cell, call))

    payload: dict[str, Any] = {
        "status": "running",
        "dispatch_id": dispatch_id,
        "started_at": _utc_now(),
        "calls": [
            {
                "cell": cell,
                "function_call_id": call.object_id,
                "dashboard_url": call.get_dashboard_url(),
                "status": "running",
            }
            for cell, call in calls
        ],
    }
    _atomic_json(status_path, payload)
    results_volume.commit()

    call_rows = {row["cell"]: row for row in payload["calls"]}
    for cell, call in calls:
        try:
            result = call.get()
            call_rows[cell]["status"] = str(result.get("status", "complete"))
            call_rows[cell]["result"] = result
        except Exception as error:
            call_rows[cell]["status"] = "failed"
            call_rows[cell]["error_type"] = type(error).__name__
            call_rows[cell]["error"] = str(error)[:2_000]
        payload["updated_at"] = _utc_now()
        _atomic_json(status_path, payload)
        results_volume.commit()

    failed = [row["cell"] for row in payload["calls"] if row["status"] == "failed"]
    payload["status"] = "failed" if failed else "complete"
    payload["failed_cells"] = failed
    payload["finished_at"] = _utc_now()
    _atomic_json(status_path, payload)
    results_volume.commit()
    return payload


@app.function(
    image=SWEEP_IMAGE,
    gpu="L40S",
    cpu=4,
    memory=32_768,
    timeout=2 * 60 * 60,
    volumes={ASSET_MOUNT: read_only_assets},
)
def benchmark_l40s(window: int = 5, seed: int = 42, steps: int = 500) -> dict[str, Any]:
    return _run_cell_impl(
        window=window,
        seed=seed,
        phase="train",
        steps=steps,
        persist=False,
        requested_gpu="L40S",
    )


@app.function(
    image=SWEEP_IMAGE,
    gpu="H100",
    cpu=4,
    memory=32_768,
    timeout=2 * 60 * 60,
    volumes={ASSET_MOUNT: read_only_assets},
)
def benchmark_h100(window: int = 5, seed: int = 42, steps: int = 500) -> dict[str, Any]:
    return _run_cell_impl(
        window=window,
        seed=seed,
        phase="train",
        steps=steps,
        persist=False,
        requested_gpu="H100",
    )


def _copy_result_marker(source: Path, destination: Path) -> bool:
    result = source / "result.json"
    if not result.exists():
        return False
    payload = json.loads(result.read_text(encoding="utf-8"))
    window = int(payload.get("window", -1))
    seed = int(payload.get("seed", -1))
    if not _valid_completed_result(result, window=window, seed=seed):
        return False
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result, destination / "result.json")
    return True


@app.function(
    image=SWEEP_IMAGE,
    cpu=2,
    memory=8_192,
    timeout=900,
    max_containers=1,
    volumes={RESULTS_MOUNT: results_volume},
)
def aggregate_results() -> dict[str, Any]:
    """Merge local baseline markers with completed Modal cells and write one report."""

    results_volume.reload()
    completed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="txc-aggregate-", dir="/tmp") as temporary:
        aggregate_root = Path(temporary) / "full"
        aggregate_cells = aggregate_root / "cells"
        for root in (REMOTE_EXISTING_CELLS, RESULTS_MOUNT / "cells"):
            if not root.exists():
                continue
            for source in sorted(root.glob("T*_seed*")):
                destination = aggregate_cells / source.name
                if destination.exists():
                    continue
                if _copy_result_marker(source, destination):
                    completed.append(source.name)

        from experiments.backtracking_window_sweep.report import write_report

        report = write_report(aggregate_root)
        aggregate_destination = RESULTS_MOUNT / "aggregate" / "full"
        aggregate_destination.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "summary.md", "window_curve.png"):
            source = aggregate_root / name
            if not source.exists():
                continue
            temporary_destination = aggregate_destination / f"{name}.tmp"
            shutil.copy2(source, temporary_destination)
            os.replace(temporary_destination, aggregate_destination / name)

    statuses: dict[str, Any] = {}
    status_root = RESULTS_MOUNT / "status"
    if status_root.exists():
        for path in sorted(status_root.glob("T*_seed*.json")):
            statuses[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    completed_set = set(completed)
    missing = [
        _cell_name(window, seed)
        for window, seed in ALL_CELLS
        if _cell_name(window, seed) not in completed_set
    ]
    manifest = {
        "status": "complete" if not missing else "partial",
        "updated_at": _utc_now(),
        "n_cells_complete": int(report["n_cells_complete"]),
        "completed_cells": sorted(completed_set),
        "missing_cells": missing,
        "modal_status": statuses,
        "retrieval": _retrieval_metadata(),
    }
    _atomic_json(RESULTS_MOUNT / "aggregate" / "manifest.json", manifest)
    results_volume.commit()
    return manifest


@app.local_entrypoint()
def seed() -> None:
    print(json.dumps(seed_assets.remote(), indent=2, sort_keys=True))


@app.local_entrypoint()
def benchmark(
    gpu: str = "both",
    window: int = 5,
    seed: int = 42,
    steps: int = 500,
) -> None:
    """Benchmark identical short training cells on L40S, H100, or both."""

    if gpu not in {"l40s", "h100", "both"}:
        raise ValueError("--gpu must be l40s, h100, or both")
    calls: dict[str, Any] = {}
    if gpu in {"l40s", "both"}:
        calls["l40s"] = benchmark_l40s.spawn(window, seed, steps)
    if gpu in {"h100", "both"}:
        calls["h100"] = benchmark_h100.spawn(window, seed, steps)
    results = {name: call.get() for name, call in calls.items()}
    print(json.dumps(results, indent=2, sort_keys=True))


@app.local_entrypoint()
def launch(
    seed_first: bool = True,
    dry_run: bool = False,
    skip_cell: str = "",
) -> None:
    """Start one detached cloud dispatcher that supervises the GPU children."""

    selected_cells = [
        (window, seed)
        for window, seed in MISSING_CELLS
        if _cell_name(window, seed) != skip_cell
    ]
    if skip_cell and len(selected_cells) == len(MISSING_CELLS):
        raise ValueError(f"--skip-cell is not a missing sweep cell: {skip_cell}")
    plan = {
        "cells": [
            {"cell": _cell_name(window, seed), "window": window, "seed": seed}
            for window, seed in selected_cells
        ],
        "seed_first": seed_first,
        "skip_cell": skip_cell or None,
        "detached_command": f"modal run --detach {CLI_PATH}::launch",
        "retrieval": _retrieval_metadata(),
    }
    if dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if seed_first:
        seed_assets.remote()

    dispatcher = dispatch_sweep.spawn(selected_cells)
    plan["dispatcher"] = {
        "function_call_id": dispatcher.object_id,
        "dashboard_url": dispatcher.get_dashboard_url(),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


@app.local_entrypoint()
def status() -> None:
    print(json.dumps(aggregate_results.remote(), indent=2, sort_keys=True))
