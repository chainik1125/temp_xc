"""Run the parameter-matched Fourier-XC backtracking replication.

The script is staged inside Aniket's RunPod checkout and deliberately imports
his frozen window sampler, artifact validator, order controls, and grouped
probe.  Only the dictionary architecture and its checkpoint handling are new.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from experiments.power_spectrum.code.backtracking_fourier_xc import (
    FourierTopKXC,
    fourier_parameter_count,
    matched_fourier_width,
    txc_parameter_count,
)


PROTOCOL_VERSION = "power-spectrum.backtracking-fourier.v1"
REFERENCE_PROTOCOL_VERSION = "2026-07-26.t16.1"
REFERENCE_COMMIT = "d9c7fc7b2"
RECOVERED_ARTIFACT_SCHEMA = "power-spectrum.backtracking-recovery-artifact.v1"
EXPECTED_REFERENCE_COHORT_SHA256 = (
    "f397f4caf6212825bd98b1b82be932ae634f01a716fd7e3642fd3d7640b27c0b"
)
EXPECTED_OFFICIAL_ARTIFACT_SHA256 = (
    "1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810"
)
RECOVERED_ROWS = 20_335
RECOVERED_POSITIVE_ROWS = 2_498
DEFAULT_WINDOWS = (1, 2, 4, 6, 10)
DEFAULT_SEEDS = (1, 2, 42)
ARTIFACT_OFFSETS = tuple(range(-23, -7))


@dataclass(frozen=True)
class FourierTrainConfig:
    window: int
    seed: int
    d_in: int = 4_096
    reference_txc_d_sae: int = 32_768
    d_sae: int = 0
    k_pos: int = 20
    batch_size: int = 1_024
    steps: int = 20_000
    learning_rate: float = 3e-4
    warmup_steps: int = 1_000
    checkpoint_every: int = 1_000
    schedule_seed: int = 0
    schedule_max_window: int = 16
    amp: bool = True
    bands: str = "multiband"


def _reference_imports() -> dict:
    """Load only the frozen functions needed from Aniket's protocol."""

    from experiments.backtracking_window_sweep.evaluate import (
        _atomic_sparse,
        _condition,
        grouped_fixed_probe_with_predictions,
        sparse_effective_l0,
        summarize_probe,
    )
    from experiments.backtracking_window_sweep.protocol import (
        ORDER_CONTROLS,
        atomic_json,
        sha256,
    )
    from experiments.backtracking_window_sweep.protocol_t16 import (
        artifact_inventory,
        assert_inventory,
    )
    from experiments.backtracking_window_sweep.train import materialize_windows
    from experiments.swr_audit.dictionary import _sparse_from_topk
    from experiments.swr_audit.run import c7_groups, trailing_window

    return {
        "_atomic_sparse": _atomic_sparse,
        "_condition": _condition,
        "_sparse_from_topk": _sparse_from_topk,
        "ORDER_CONTROLS": ORDER_CONTROLS,
        "artifact_inventory": artifact_inventory,
        "assert_inventory": assert_inventory,
        "atomic_json": atomic_json,
        "c7_groups": c7_groups,
        "grouped_fixed_probe_with_predictions": grouped_fixed_probe_with_predictions,
        "materialize_windows": materialize_windows,
        "sha256": sha256,
        "sparse_effective_l0": sparse_effective_l0,
        "summarize_probe": summarize_probe,
        "trailing_window": trailing_window,
    }


def _csv_ints(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed or len(set(parsed)) != len(parsed):
        raise ValueError(f"expected a non-empty unique integer list, got {value!r}")
    return parsed


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_safetensors(model: torch.nn.Module, path: Path) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        name: tensor.detach().contiguous().cpu()
        for name, tensor in model.state_dict().items()
    }
    save_file(state, str(temporary))
    os.replace(temporary, path)


def _atomic_torch(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _model_dtype(config: FourierTrainConfig, device: str) -> torch.dtype:
    return torch.bfloat16 if config.amp and device.startswith("cuda") else torch.float32


def _config(
    *,
    window: int,
    seed: int,
    steps: int,
    batch_size: int,
    checkpoint_every: int,
) -> FourierTrainConfig:
    width = matched_fourier_width(
        d_in=4_096,
        txc_d_sae=32_768,
        window=window,
    )
    return FourierTrainConfig(
        window=window,
        seed=seed,
        d_sae=width,
        batch_size=batch_size,
        steps=steps,
        checkpoint_every=min(checkpoint_every, steps),
        schedule_seed=907_000 + 100 * seed,
    )


def parameter_match(config: FourierTrainConfig) -> dict[str, float | int]:
    txc_count = txc_parameter_count(
        d_in=config.d_in,
        d_sae=config.reference_txc_d_sae,
        window=config.window,
    )
    fourier_count = fourier_parameter_count(
        d_in=config.d_in,
        d_sae=config.d_sae,
        window=config.window,
        bands_mode=config.bands,
    )
    return {
        "reference_txc_d_sae": config.reference_txc_d_sae,
        "fourier_d_sae": config.d_sae,
        "reference_txc_parameters": txc_count,
        "fourier_parameters": fourier_count,
        "difference": fourier_count - txc_count,
        "relative_difference": (fourier_count - txc_count) / txc_count,
    }


def build_model(config: FourierTrainConfig) -> FourierTopKXC:
    return FourierTopKXC(
        d_in=config.d_in,
        d_sae=config.d_sae,
        T=config.window,
        k_pos=config.k_pos,
        bands=config.bands,
    )


def load_model(
    checkpoint_dir: Path,
    *,
    device: str,
) -> tuple[FourierTopKXC, FourierTrainConfig]:
    from safetensors.torch import load_file

    config = FourierTrainConfig(
        **json.loads((checkpoint_dir / "config.json").read_text())
    )
    model = build_model(config).to(
        device=device,
        dtype=_model_dtype(config, device),
    )
    model.load_state_dict(
        load_file(str(checkpoint_dir / "model.safetensors"), device=device)
    )
    model.eval()
    return model, config


def train_dictionary(
    *,
    activation_cache: Path,
    checkpoint_dir: Path,
    config: FourierTrainConfig,
    device: str,
) -> dict:
    """Train/resume one Fourier cell using Aniket's exact window schedule."""

    reference = _reference_imports()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "model.safetensors"
    state_path = checkpoint_dir / "training_state.pt"
    requested = asdict(config)
    if config_path.exists():
        if json.loads(config_path.read_text()) != requested:
            raise ValueError(f"checkpoint config mismatch at {config_path}")
    else:
        _atomic_json(requested, config_path)

    completed_summary = checkpoint_dir / "training_summary.json"
    if model_path.exists() and not state_path.exists() and completed_summary.exists():
        cached = json.loads(completed_summary.read_text())
        if (
            cached.get("status") == "complete"
            and cached.get("completed_steps") == config.steps
        ):
            return {
                **cached,
                "cached": True,
                "optimizer_state_cleaned_after_evaluation": True,
            }

    cache = np.load(activation_cache, mmap_mode="r")
    if cache.ndim != 3 or cache.shape[-1] != config.d_in:
        raise ValueError(f"invalid activation cache shape {cache.shape}")

    torch.manual_seed(config.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")
    model = build_model(config).to(
        device=device,
        dtype=_model_dtype(config, device),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_step = 0
    last_metrics: dict[str, float] = {}
    if model_path.exists() != state_path.exists():
        raise ValueError("model and optimizer checkpoints must coexist while training")
    if model_path.exists():
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(model_path), device=device))
        payload = torch.load(state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])
        last_metrics = dict(payload.get("last_metrics", {}))
    if start_step > config.steps:
        raise ValueError("checkpoint step exceeds requested training target")
    if start_step == config.steps:
        return {
            "status": "complete",
            "cached": True,
            "completed_steps": start_step,
            "last_metrics": last_metrics,
            "parameter_match": parameter_match(config),
        }

    model.train()
    started = time.monotonic()
    for step in range(start_step, config.steps):
        raw = reference["materialize_windows"](
            cache,
            step=step,
            window=config.window,
            batch_size=config.batch_size,
            schedule_seed=config.schedule_seed,
            max_window=config.schedule_max_window,
        )
        batch = torch.from_numpy(raw).to(
            device=device,
            dtype=_model_dtype(config, device),
        )
        if config.warmup_steps:
            scale = min(1.0, float(step + 1) / config.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * scale
        model.pre_step()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
            enabled=config.amp and device.startswith("cuda"),
        ):
            metrics = model.train_step(batch)
        metrics["loss"].backward()
        optimizer.step()
        model.post_step()
        last_metrics = {
            name: float(value.detach().float().cpu())
            for name, value in metrics.items()
            if value.numel() == 1
        }
        completed = step + 1
        if completed % 100 == 0 or completed == config.steps:
            elapsed = time.monotonic() - started
            rate = (completed - start_step) / max(elapsed, 1e-9)
            print(
                f"[fourier train] T={config.window} seed={config.seed} "
                f"step={completed}/{config.steps} loss={last_metrics['loss']:.6g} "
                f"steps_per_second={rate:.4f}",
                flush=True,
            )
        if completed == config.steps or completed % config.checkpoint_every == 0:
            _atomic_safetensors(model, model_path)
            _atomic_torch(
                {
                    "step": completed,
                    "optimizer": optimizer.state_dict(),
                    "last_metrics": last_metrics,
                },
                state_path,
            )

    elapsed = time.monotonic() - started
    result = {
        "status": "complete",
        "cached": False,
        "completed_steps": config.steps,
        "elapsed_seconds_this_process": elapsed,
        "last_metrics": last_metrics,
        "parameter_match": parameter_match(config),
    }
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def memory_smoke(
    *,
    activation_cache: Path,
    config: FourierTrainConfig,
    device: str,
) -> dict:
    """Run exact-width/full-batch updates without creating a checkpoint."""

    reference = _reference_imports()
    cache = np.load(activation_cache, mmap_mode="r")
    torch.manual_seed(config.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
        torch.set_float32_matmul_precision("high")
        with torch.cuda.device(torch.device(device)):
            torch.cuda.reset_peak_memory_stats()
    model = build_model(config).to(
        device=device,
        dtype=_model_dtype(config, device),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    started = time.monotonic()
    metrics = None
    for step in range(config.steps):
        raw = reference["materialize_windows"](
            cache,
            step=step,
            window=config.window,
            batch_size=config.batch_size,
            schedule_seed=config.schedule_seed,
            max_window=config.schedule_max_window,
        )
        batch = torch.from_numpy(raw).to(
            device=device,
            dtype=_model_dtype(config, device),
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu",
            dtype=torch.bfloat16,
            enabled=config.amp and device.startswith("cuda"),
        ):
            metrics = model.train_step(batch)
        metrics["loss"].backward()
        optimizer.step()
        model.post_step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    assert metrics is not None
    payload: dict[str, object] = {
        "status": "complete",
        "elapsed_seconds": elapsed,
        "steps": config.steps,
        "steps_per_second": config.steps / elapsed,
        "config": asdict(config),
        "parameter_match": parameter_match(config),
        "metrics": {
            name: float(value.detach().float().cpu())
            for name, value in metrics.items()
            if value.numel() == 1
        },
    }
    if device.startswith("cuda"):
        payload["cuda"] = {
            "device_name": torch.cuda.get_device_name(torch.device(device)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(device)),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(torch.device(device)),
        }
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


@torch.no_grad()
def encode_dataset(
    x: np.ndarray,
    *,
    model: FourierTopKXC,
    batch_size: int,
    device: str,
) -> sparse.csr_matrix:
    reference = _reference_imports()
    chunks = []
    dtype = next(model.parameters()).dtype
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(
            x[start : start + batch_size].astype(np.float32, copy=False)
        ).to(device=device, dtype=dtype)
        values, indices = model.encode_topk(batch)
        chunks.append(
            reference["_sparse_from_topk"](values, indices, model.d_sae)
        )
    return sparse.vstack(chunks, format="csr")


def _band_usage(
    matrix: sparse.csr_matrix,
    model: FourierTopKXC,
) -> list[dict]:
    rows = []
    total_mass = float(matrix.data.sum())
    for index, ((start, stop), frequencies) in enumerate(
        zip(model.band_slices, model.frequency_bin_bands)
    ):
        band = matrix[:, start:stop]
        counts = np.diff(band.indptr)
        mass = float(band.data.sum())
        rows.append(
            {
                "band": index,
                "frequencies": list(frequencies),
                "feature_start": start,
                "feature_stop": stop,
                "n_features": stop - start,
                "l0_mean": float(counts.mean()),
                "activation_mass_mean": mass / matrix.shape[0],
                "activation_mass_share": mass / total_mass if total_mass else 0.0,
            }
        )
    return rows


def evaluate_dictionary(
    *,
    artifact: Path,
    artifact_sha256: str,
    cohort_sha256: str,
    artifact_provenance: dict | None,
    checkpoint_dir: Path,
    output_dir: Path,
    window: int,
    seed: int,
    encode_batch_size: int,
    run_band_probes: bool,
    device: str,
) -> dict:
    """Evaluate using Aniket's artifact, controls, grouped folds, and probe."""

    reference = _reference_imports()
    model_path = checkpoint_dir / "model.safetensors"
    result_path = output_dir / "result.json"
    if result_path.exists():
        completed = json.loads(result_path.read_text())
        cached_fingerprint = completed.get("code_fingerprint", {})
        expected_implementation = implementation_fingerprint()
        checks = {
            "status": completed.get("status") == "complete",
            "protocol": completed.get("protocol_version") == PROTOCOL_VERSION,
            "window": completed.get("window") == window,
            "seed": completed.get("seed") == seed,
            "artifact": completed.get("artifact_sha256") == artifact_sha256,
            "cohort": completed.get("cohort_sha256") == cohort_sha256,
            "provenance": completed.get("artifact_provenance")
            == artifact_provenance,
            "band_probes": completed.get("architecture", {}).get(
                "band_probes"
            )
            is run_band_probes,
            "checkpoint": cached_fingerprint.get("fourier_checkpoint_sha256")
            == reference["sha256"](model_path),
            "implementation": cached_fingerprint.get(
                "implementation_sha256"
            )
            == expected_implementation,
        }
        if not all(checks.values()):
            raise ValueError(f"stale result at {result_path}")
        return completed

    with np.load(artifact, allow_pickle=True) as payload:
        x = payload["X"]
        y = payload["is_bt"].astype(np.int64, copy=False)
        groups = reference["c7_groups"](payload["keys"])
    x = reference["trailing_window"](x, window)
    model, config = load_model(checkpoint_dir, device=device)

    code_dir = output_dir / "codes"
    code_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "protocol_version": PROTOCOL_VERSION,
        "reference_protocol_version": REFERENCE_PROTOCOL_VERSION,
        "reference_commit": REFERENCE_COMMIT,
        "implementation_sha256": implementation_fingerprint(),
        "artifact_sha256": artifact_sha256,
        "cohort_sha256": cohort_sha256,
        "artifact_provenance": artifact_provenance,
        "band_probes": run_band_probes,
        "window": window,
        "window_offsets": list(ARTIFACT_OFFSETS[-window:]),
        "seed": seed,
        "fourier_checkpoint_sha256": reference["sha256"](model_path),
        "fourier_config": asdict(config),
    }
    metadata_path = code_dir / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != fingerprint:
            raise ValueError(f"code-cache provenance mismatch at {metadata_path}")
    else:
        _atomic_json(fingerprint, metadata_path)

    matrices: dict[str, sparse.csr_matrix] = {}
    for control_index, name in enumerate(("ordered", *reference["ORDER_CONTROLS"])):
        path = code_dir / f"fourier_{name}.npz"
        if path.exists():
            matrix = sparse.load_npz(path).tocsr()
            if matrix.shape != (len(x), config.d_sae):
                raise ValueError(f"invalid cached code shape {matrix.shape} at {path}")
        else:
            conditioned = reference["_condition"](
                x,
                name,
                seed=seed,
                control_index=control_index,
            )
            matrix = encode_dataset(
                conditioned,
                model=model,
                batch_size=encode_batch_size,
                device=device,
            )
            reference["_atomic_sparse"](matrix, path)
        matrices[name] = matrix

    representations = {"fourier": matrices}
    if run_band_probes:
        for band, (start, stop) in enumerate(model.band_slices):
            representations[f"fourier_band_{band}"] = {
                name: matrix[:, start:stop]
                for name, matrix in matrices.items()
            }

    probes = {}
    for name, variants in representations.items():
        rows = reference["grouped_fixed_probe_with_predictions"](
            variants["ordered"],
            {
                control: variants[control]
                for control in reference["ORDER_CONTROLS"]
            },
            y,
            groups,
            folds=5,
            s_grid=(8, 16, 32),
            seed=seed,
            prediction_dir=output_dir / "predictions" / name,
        )
        probes[name] = reference["summarize_probe"](rows)

    result = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "reference_protocol_version": REFERENCE_PROTOCOL_VERSION,
        "reference_commit": REFERENCE_COMMIT,
        "architecture": {
            "name": "FourierTopKXC",
            "version": model.arch_version,
            "bands_mode": config.bands,
            "bands": [list(band) for band in model.frequency_bin_bands],
            "basis_row_bands": [list(band) for band in model.bands],
            "atoms_per_band": list(model.h_per_band),
            "matryoshka": False,
            "adaptive_frequency_objective": False,
            "band_probes": run_band_probes,
            "support_rule": "per-example TopK then ReLU, matching TXCBase",
        },
        "artifact": str(artifact.resolve()),
        "artifact_sha256": artifact_sha256,
        "cohort_sha256": cohort_sha256,
        "artifact_provenance": artifact_provenance,
        "window": window,
        "window_offsets": list(ARTIFACT_OFFSETS[-window:]),
        "seed": seed,
        "n_rows": int(len(x)),
        "n_groups": int(len(np.unique(groups))),
        "positive_rate": float(y.mean()),
        "folds": 5,
        "s_grid": [8, 16, 32],
        "parameter_match": parameter_match(config),
        "effective_l0": {
            name: reference["sparse_effective_l0"](
                matrix,
                nominal_l0=config.k_pos * window,
            )
            for name, matrix in matrices.items()
        },
        "ordered_band_usage": _band_usage(matrices["ordered"], model),
        "probes": probes,
        "code_fingerprint": fingerprint,
    }
    _atomic_json(result, result_path)
    del model, matrices, representations
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _run_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def implementation_fingerprint() -> dict[str, str]:
    """Hash the staged experimental source independently of its checkout."""

    code_dir = Path(__file__).resolve().parent
    paths = (
        code_dir / "backtracking_fourier_xc.py",
        Path(__file__).resolve(),
    )
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def activation_cache_inventory(path: Path) -> dict:
    """Validate the pinned training cache without requiring eval artifacts."""

    from experiments.backtracking_assets import CACHE_SHA256, CACHE_SHAPE

    reference = _reference_imports()
    if not path.exists():
        raise ValueError(f"missing activation cache: {path}")
    cache = np.load(path, mmap_mode="r")
    shape = tuple(int(value) for value in cache.shape)
    dtype = str(cache.dtype)
    digest = reference["sha256"](path)
    inventory = {
        "activation_cache": str(path),
        "activation_cache_shape": list(shape),
        "activation_cache_dtype": dtype,
        "activation_cache_sha256": digest,
        "activation_cache_shape_ok": shape == CACHE_SHAPE,
        "activation_cache_dtype_ok": dtype == "float16",
        "activation_cache_sha256_ok": digest == CACHE_SHA256,
    }
    checks = {
        name: value
        for name, value in inventory.items()
        if name.endswith("_ok")
    }
    if not all(checks.values()):
        raise ValueError(f"activation-cache contract failed: {inventory}")
    return inventory


def _cohort_sha256(keys: np.ndarray, labels: np.ndarray) -> str:
    """Hash the ordered cohort using Aniket's frozen protocol definition."""

    if len(keys) != len(labels):
        raise ValueError("keys and labels must have the same length")
    digest = hashlib.sha256()
    for key, label in zip(keys, labels):
        encoded = str(key).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(int(label).to_bytes(1, "little", signed=False))
    return digest.hexdigest()


def recovered_artifact_inventory(
    artifact: Path,
    manifest_path: Path,
    reference_artifact: Path,
    activation_cache: Path,
) -> dict:
    """Validate the explicitly marked, non-bit-exact sensitivity artifact."""

    if not artifact.exists() or not manifest_path.exists():
        raise ValueError("missing recovered artifact or manifest")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != RECOVERED_ARTIFACT_SCHEMA
        or manifest.get("status") != "complete"
    ):
        raise ValueError("invalid recovered-artifact manifest schema or status")

    reference = _reference_imports()
    from experiments.power_spectrum.code.build_backtracking_recovery_artifact import (
        _stored_npz_memmap,
        _tail_matches_official,
    )

    artifact_digest = reference["sha256"](artifact)
    manifest_digest = reference["sha256"](manifest_path)
    recovered_x = _stored_npz_memmap(artifact, "X")
    with np.load(artifact, allow_pickle=False) as payload:
        required = {"X", "is_bt", "keys", "offsets"}
        if set(payload.files) != required:
            raise ValueError(f"recovered artifact arrays differ: {payload.files}")
        labels = payload["is_bt"].astype(np.uint8)
        keys = payload["keys"].astype(str)
        offsets = payload["offsets"].astype(np.int32)
        shape = tuple(int(value) for value in recovered_x.shape)
        dtype = str(recovered_x.dtype)
        cohort_digest = _cohort_sha256(keys, labels)
        positive_rows = int(labels.sum())

    official_digest = reference["sha256"](reference_artifact)
    with np.load(reference_artifact, allow_pickle=True) as official:
        official_keys = official["keys"].astype(str)
        official_x = official["X"]
        official_index = {
            key: index for index, key in enumerate(official_keys.tolist())
        }
        recovered_keys = keys.tolist()
        all_keys_official = all(key in official_index for key in recovered_keys)
        recovered_official_rows = (
            np.asarray([official_index[key] for key in recovered_keys])
            if all_keys_official
            else np.asarray([], dtype=np.int64)
        )
        official_tail_exact = all_keys_official and _tail_matches_official(
            recovered_x,
            official_x,
            recovered_official_rows,
        )

    expected_shape = (RECOVERED_ROWS, len(ARTIFACT_OFFSETS), 4_096)
    cohort = manifest.get("cohort", {})
    tail = manifest.get("tail_replacement", {})
    checks = {
        "artifact_sha256_ok": artifact_digest == manifest.get("artifact_sha256"),
        "artifact_shape_ok": shape == expected_shape,
        "artifact_dtype_ok": dtype == "float32",
        "artifact_offsets_ok": offsets.tolist() == list(ARTIFACT_OFFSETS),
        "cohort_sha256_ok": cohort_digest == cohort.get("sha256"),
        "cohort_rows_ok": len(keys) == int(cohort.get("rows", -1)),
        "positive_rows_ok": positive_rows
        == int(cohort.get("positive_rows", -1))
        == RECOVERED_POSITIVE_ROWS,
        "binary_labels_ok": set(np.unique(labels).tolist()).issubset({0, 1}),
        "unique_keys_ok": len(keys) == len(set(keys.tolist())),
        "reference_mismatch_declared": cohort.get("matches_reference") is False,
        "reference_cohort_sha256_ok": cohort.get(
            "expected_reference_sha256"
        )
        == EXPECTED_REFERENCE_COHORT_SHA256,
        "official_artifact_sha256_ok": official_digest
        == EXPECTED_OFFICIAL_ARTIFACT_SHA256
        == tail.get("source_artifact_sha256"),
        "all_keys_in_official_artifact": all_keys_official,
        "official_tail_bit_exact": official_tail_exact,
        "tail_replacement_declared": tail.get("bit_exact_after_replacement")
        is True,
        "provenance_warning_present": bool(manifest.get("provenance_warning")),
    }
    if not all(checks.values()):
        raise ValueError(
            f"recovered-artifact contract failed: {checks}"
        )

    cache = activation_cache_inventory(activation_cache)
    return {
        **cache,
        "artifact": str(artifact),
        "artifact_manifest": str(manifest_path),
        "artifact_manifest_sha256": manifest_digest,
        "artifact_sha256": artifact_digest,
        "artifact_shape": list(shape),
        "artifact_dtype": dtype,
        "common_cohort_rows": len(keys),
        "common_cohort_sha256": cohort_digest,
        "recovery_provenance": {
            "schema_version": RECOVERED_ARTIFACT_SCHEMA,
            "expected_reference_cohort_sha256": cohort.get(
                "expected_reference_sha256"
            ),
            "matches_reference_cohort": False,
            "manifest_sha256": manifest_digest,
            "manifest": manifest,
            "provenance_warning": manifest["provenance_warning"],
        },
        **checks,
    }


def _parser() -> argparse.ArgumentParser:
    root = Path(os.environ.get("TXC_RUNPOD_ROOT", "/workspace/txc-neurips-aniket"))
    c7 = root / "purified/artifacts/c7"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("plan", "memory-smoke", "train", "eval", "all"),
        default="plan",
    )
    parser.add_argument("--windows")
    parser.add_argument("--seeds")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--memory-smoke-steps", type=int, default=1)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--band-probes", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--cleanup-optimizer-state", action="store_true")
    parser.add_argument("--allow-recovered-artifact", action="store_true")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=c7 / "sentence_acts_L10_T16.npz",
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=c7 / "sentence_acts_L10_T16.manifest.json",
    )
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        default=c7 / "sentence_acts_L10.npz",
    )
    parser.add_argument(
        "--activation-cache",
        type=Path,
        default=(
            root
            / "purified/artifacts/hf_temp_bench_data/act_cache/"
            "fb2a74be884e512a/resid_post_L10.npy"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            root
            / "purified/results/neurips_rebuttal/"
            "backtracking_fourier_matched/reviewer-five-point-v1"
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=(
            root
            / "checkpoints/backtracking_fourier_matched/"
            "reviewer-five-point-v1"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    windows = _csv_ints(args.windows, DEFAULT_WINDOWS)
    seeds = _csv_ints(args.seeds, DEFAULT_SEEDS)
    if any(window not in DEFAULT_WINDOWS for window in windows):
        raise ValueError(f"windows must be a subset of {DEFAULT_WINDOWS}")
    if min(
        args.steps,
        args.batch_size,
        args.checkpoint_every,
        args.memory_smoke_steps,
    ) < 1:
        raise ValueError(
            "steps, memory-smoke steps, batch size, and checkpoint interval "
            "must be positive"
        )

    cells = [(seed, window) for seed in seeds for window in windows]
    if args.max_cells is not None:
        cells = cells[: args.max_cells]
    configs = [
        _config(
            window=window,
            seed=seed,
            steps=args.steps,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
        )
        for seed, window in cells
    ]
    plan = {
        "protocol_version": PROTOCOL_VERSION,
        "reference_protocol_version": REFERENCE_PROTOCOL_VERSION,
        "reference_commit": REFERENCE_COMMIT,
        "implementation_sha256": implementation_fingerprint(),
        "phase": args.phase,
        "device": args.device,
        "artifact": str(args.artifact),
        "activation_cache": str(args.activation_cache),
        "output_root": str(args.output_root),
        "checkpoint_root": str(args.checkpoint_root),
        "allow_recovered_artifact": args.allow_recovered_artifact,
        "band_probes": args.band_probes,
        "cells": [
            {
                "config": asdict(config),
                "parameter_match": parameter_match(config),
            }
            for config in configs
        ],
    }
    plan["run_fingerprint"] = _run_fingerprint(plan)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.phase == "plan":
        return

    reference = _reference_imports()
    if args.phase == "memory-smoke":
        smoke_config = _config(
            window=max(windows),
            seed=seeds[0],
            steps=args.memory_smoke_steps,
            batch_size=args.batch_size,
            checkpoint_every=1,
        )
        result = memory_smoke(
            activation_cache=args.activation_cache,
            config=smoke_config,
            device=args.device,
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        _atomic_json(result, args.output_root / "memory_smoke.json")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return

    if args.phase == "train":
        inventory = activation_cache_inventory(args.activation_cache)
    elif args.allow_recovered_artifact:
        inventory = recovered_artifact_inventory(
            args.artifact,
            args.artifact_manifest,
            args.reference_artifact,
            args.activation_cache,
        )
    else:
        inventory = reference["artifact_inventory"](
            args.artifact,
            args.artifact_manifest,
            args.reference_artifact,
            args.activation_cache,
            strict_full=True,
        )
        reference["assert_inventory"](inventory, strict_full=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        {**plan, "inventory": inventory},
        args.output_root / "plan.json",
    )

    summaries = []
    for config in configs:
        cell_name = f"T{config.window}_seed{config.seed}"
        checkpoint_dir = args.checkpoint_root / cell_name
        output_dir = args.output_root / "cells" / cell_name
        training = None
        if args.phase in {"train", "all"}:
            training = train_dictionary(
                activation_cache=args.activation_cache,
                checkpoint_dir=checkpoint_dir,
                config=config,
                device=args.device,
            )
            _atomic_json(training, checkpoint_dir / "training_summary.json")
        evaluation = None
        if args.phase in {"eval", "all"}:
            evaluation = evaluate_dictionary(
                artifact=args.artifact,
                artifact_sha256=inventory["artifact_sha256"],
                cohort_sha256=inventory["common_cohort_sha256"],
                artifact_provenance=inventory.get("recovery_provenance"),
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                window=config.window,
                seed=config.seed,
                encode_batch_size=args.encode_batch_size,
                run_band_probes=args.band_probes,
                device=args.device,
            )
        if (
            args.cleanup_optimizer_state
            and (
                (training is not None and training.get("status") == "complete")
                or (
                    evaluation is not None
                    and evaluation.get("status") == "complete"
                )
            )
        ):
            state_path = checkpoint_dir / "training_state.pt"
            if state_path.exists():
                state_path.unlink()
        summary = {
            "cell": cell_name,
            "training": training,
            "evaluation_status": (
                evaluation.get("status") if evaluation is not None else None
            ),
        }
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    _atomic_json(
        {
            "status": "complete",
            "protocol_version": PROTOCOL_VERSION,
            "run_fingerprint": plan["run_fingerprint"],
            "cells": summaries,
        },
        args.output_root / "summary.json",
    )


if __name__ == "__main__":
    main()
