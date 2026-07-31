"""Run the missing shared-SAE -> learned trajectory bottleneck C7 baseline.

This module is designed to run inside the ``origin/neurips-aniket`` checkout,
where the frozen C7 protocol lives.  It deliberately reuses that protocol's
SAE trainer and deterministic exposure schedule.  The learned second stage
maps five sparse per-token SAE codes to one L0<=100 code and reconstructs all
five raw activation vectors through the frozen SAE dictionary.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from experiments.backtracking_window_sweep.protocol import (  # noqa: E402
    EXPECTED_ARTIFACT_SHA256,
    atomic_json,
    sha256,
)
from experiments.backtracking_window_sweep.train import (  # noqa: E402
    TrainCellConfig,
    load_dictionary,
    scheduled_window_indices,
    train_dictionary,
)
from experiments.power_spectrum.trajectory_bottleneck.model import (  # noqa: E402
    TrajectoryBottleneck,
)


PROTOCOL = "shared-sae-learned-trajectory-bottleneck-c7.v1"
ORDER_CONTROLS = ("shuffle", "reverse", "circular")
PUBLISHED_REFERENCES = {
    "TXC (published 300k)": {
        "pr_auc_s32": 0.249917,
        "source": (
            "dmanningcoe/temp-xc-reviewer-results/"
            "reviewer_seed_audit_2026-07-27/c7_headline/"
            "seed42_published_eval.json"
        ),
    },
    "T-SAE (published 300k)": {
        "pr_auc_s32": 0.245,
        "source": "origin/300k-tfa seven-model C7 matrix",
    },
    "Stacked SAE (published 300k)": {
        "pr_auc_s32": 0.207,
        "source": (
            "dmanningcoe/stacked-sae-rebuttal-2026-07, "
            "train key 26e69fdc60452c27"
        ),
    },
}


@dataclass(frozen=True)
class RunConfig:
    window: int = 5
    seed: int = 42
    d_in: int = 4_096
    d_sae: int = 32_768
    k_pos: int = 20
    k_window: int = 100
    base_steps: int = 300_000
    adapter_steps: int = 300_000
    batch_size: int = 1_024
    adapter_microbatch: int = 512
    learning_rate: float = 3e-4
    warmup_steps: int = 1_000
    checkpoint_every: int = 5_000
    schedule_seed: int = 911_200
    ranks: tuple[int, ...] = (0, 256)
    folds: int = 5
    s_grid: tuple[int, ...] = (8, 16, 32, 64, 128, 256)


def _atomic_torch(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_safetensors(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(
        {
            name: value.detach().contiguous().cpu()
            for name, value in model.state_dict().items()
        },
        str(temporary),
    )
    os.replace(temporary, path)


def _heartbeat(root: Path, *, stage: str, **fields: object) -> None:
    atomic_json(
        {
            "protocol": PROTOCOL,
            "stage": stage,
            "unix_time": time.time(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **fields,
        },
        root / "heartbeat.json",
    )


def _validate_inputs(activation_cache: Path, artifact: Path) -> dict:
    cache = np.load(activation_cache, mmap_mode="r")
    if tuple(cache.shape) != (4_044, 128, 4_096):
        raise ValueError(f"unexpected activation cache shape: {cache.shape}")
    with np.load(artifact, allow_pickle=True, mmap_mode="r") as payload:
        shape = tuple(payload["X"].shape)
        fields = sorted(payload.files)
    if shape != (25_204, 6, 4_096):
        raise ValueError(f"unexpected C7 artifact shape: {shape}")
    digest = sha256(artifact)
    if digest != EXPECTED_ARTIFACT_SHA256:
        raise ValueError(
            f"C7 artifact SHA mismatch: {digest} != {EXPECTED_ARTIFACT_SHA256}"
        )
    return {
        "activation_cache": str(activation_cache),
        "activation_cache_shape": list(cache.shape),
        "artifact": str(artifact),
        "artifact_shape": list(shape),
        "artifact_fields": fields,
        "artifact_sha256": digest,
    }


def _base_config(config: RunConfig) -> TrainCellConfig:
    return TrainCellConfig(
        arch="sae",
        window=config.window,
        seed=config.seed,
        d_in=config.d_in,
        d_sae=config.d_sae,
        k_pos=config.k_pos,
        batch_size=config.batch_size,
        steps=config.base_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        checkpoint_every=min(config.checkpoint_every, config.base_steps),
        schedule_seed=config.schedule_seed,
        amp=True,
    )


def train_base(
    *,
    activation_cache: Path,
    checkpoint_root: Path,
    config: RunConfig,
    device: str,
    output_root: Path,
) -> dict:
    _heartbeat(output_root, stage="base_sae_train")
    result = train_dictionary(
        activation_cache=activation_cache,
        checkpoint_dir=checkpoint_root / "base_sae",
        config=_base_config(config),
        device=device,
    )
    atomic_json(result, output_root / "base_sae_train.json")
    return result


@torch.no_grad()
def precompute_base_codes(
    *,
    activation_cache: Path,
    checkpoint_root: Path,
    code_root: Path,
    config: RunConfig,
    device: str,
    output_root: Path,
    token_batch_size: int = 1_024,
) -> dict:
    """Encode the finite training cache once, resumably, into TopK arrays."""

    code_root.mkdir(parents=True, exist_ok=True)
    index_path = code_root / "indices.npy"
    value_path = code_root / "values.npy"
    state_path = code_root / "state.json"
    cache = np.load(activation_cache, mmap_mode="r")
    n_sequences, sequence_length, width = cache.shape
    n_tokens = n_sequences * sequence_length
    expected = {
        "protocol": PROTOCOL,
        "base_checkpoint_sha256": sha256(
            checkpoint_root / "base_sae" / "model.safetensors"
        ),
        "shape": [n_sequences, sequence_length, config.k_pos],
        "k_pos": config.k_pos,
    }
    completed = 0
    if state_path.exists():
        state = json.loads(state_path.read_text())
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"base-code cache mismatch for {key}")
        completed = int(state["completed_tokens"])

    shape = (n_sequences, sequence_length, config.k_pos)
    if completed == 0:
        indices = np.lib.format.open_memmap(
            index_path, mode="w+", dtype=np.int32, shape=shape
        )
        values = np.lib.format.open_memmap(
            value_path, mode="w+", dtype=np.float16, shape=shape
        )
    else:
        indices = np.load(index_path, mmap_mode="r+")
        values = np.load(value_path, mmap_mode="r+")
        if indices.shape != shape or values.shape != shape:
            raise ValueError("partial base-code arrays have wrong shape")

    model, _ = load_dictionary(checkpoint_root / "base_sae", device=device)
    model.eval()
    flat_indices = indices.reshape(n_tokens, config.k_pos)
    flat_values = values.reshape(n_tokens, config.k_pos)
    flat_cache = cache.reshape(n_tokens, width)
    dtype = model.W_enc.dtype
    for start in range(completed, n_tokens, token_batch_size):
        end = min(start + token_batch_size, n_tokens)
        token = torch.from_numpy(
            np.asarray(flat_cache[start:end], dtype=np.float32)
        ).to(device=device, dtype=dtype)
        pre = (token - model.b_dec) @ model.W_enc.T + model.b_enc
        active_values, active_indices = pre.topk(config.k_pos, dim=-1)
        flat_indices[start:end] = active_indices.cpu().numpy().astype(np.int32)
        flat_values[start:end] = (
            torch.relu(active_values).float().cpu().numpy().astype(np.float16)
        )
        if end == n_tokens or end % (token_batch_size * 32) == 0:
            indices.flush()
            values.flush()
            atomic_json(
                {**expected, "completed_tokens": end},
                state_path,
            )
            _heartbeat(
                output_root,
                stage="precompute_base_codes",
                completed_tokens=end,
                total_tokens=n_tokens,
            )
    result = {
        **expected,
        "completed_tokens": n_tokens,
        "indices": str(index_path),
        "values": str(value_path),
    }
    atomic_json(result, code_root / "complete.json")
    del model, indices, values
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _load_adapter(
    *,
    rank: int,
    config: RunConfig,
    checkpoint_root: Path,
    device: str,
) -> tuple[TrajectoryBottleneck, torch.optim.Optimizer, int, dict]:
    base_state = load_file(
        str(checkpoint_root / "base_sae" / "model.safetensors"),
        device=device,
    )
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = TrajectoryBottleneck(
        base_decoder=base_state["W_dec"].T.to(dtype),
        base_decoder_bias=base_state["b_dec"].to(dtype),
        window=config.window,
        k_window=config.k_window,
        rank=rank,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    cell = checkpoint_root / f"adapter_rank{rank}"
    model_path = cell / "model.safetensors"
    state_path = cell / "training_state.pt"
    step = 0
    metrics: dict = {}
    if model_path.exists() != state_path.exists():
        raise ValueError(f"partial adapter checkpoint in {cell}")
    if model_path.exists():
        model.load_state_dict(load_file(str(model_path), device=device))
        state = torch.load(state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        metrics = dict(state.get("last_metrics", {}))
    return model, optimizer, step, metrics


def _save_adapter(
    *,
    model: TrajectoryBottleneck,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict,
    checkpoint_root: Path,
    rank: int,
) -> None:
    cell = checkpoint_root / f"adapter_rank{rank}"
    cell.mkdir(parents=True, exist_ok=True)
    _atomic_safetensors(model, cell / "model.safetensors")
    _atomic_torch(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "last_metrics": metrics,
        },
        cell / "training_state.pt",
    )


def train_adapter(
    *,
    rank: int,
    activation_cache: Path,
    code_root: Path,
    checkpoint_root: Path,
    output_root: Path,
    config: RunConfig,
    device: str,
) -> dict:
    """Train or resume one learned trajectory bottleneck."""

    cell = checkpoint_root / f"adapter_rank{rank}"
    cell.mkdir(parents=True, exist_ok=True)
    requested = {
        **asdict(config),
        "ranks": list(config.ranks),
        "rank": rank,
        "protocol": PROTOCOL,
        "objective": (
            "one L0<=100 code reconstructs all five raw activation vectors "
            "through a frozen shared-SAE dictionary"
        ),
    }
    config_path = cell / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != requested:
            raise ValueError(f"adapter config mismatch in {cell}")
    else:
        atomic_json(requested, config_path)

    raw_cache = np.load(activation_cache, mmap_mode="r")
    code_indices = np.load(code_root / "indices.npy", mmap_mode="r")
    code_values = np.load(code_root / "values.npy", mmap_mode="r")
    model, optimizer, start_step, last_metrics = _load_adapter(
        rank=rank,
        config=config,
        checkpoint_root=checkpoint_root,
        device=device,
    )
    if start_step > config.adapter_steps:
        raise ValueError("adapter checkpoint is beyond requested steps")
    if start_step == config.adapter_steps:
        return {
            "rank": rank,
            "completed_steps": start_step,
            "cached": True,
            "last_metrics": last_metrics,
        }

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model.train()
    for step in range(start_step, config.adapter_steps):
        sequence_indices, starts = scheduled_window_indices(
            step=step,
            n_sequences=raw_cache.shape[0],
            sequence_length=raw_cache.shape[1],
            window=config.window,
            batch_size=config.batch_size,
            schedule_seed=config.schedule_seed,
            max_window=6,
        )
        positions = starts[:, None] + np.arange(config.window)[None, :]
        optimizer.zero_grad(set_to_none=True)
        accumulator = {
            "loss": 0.0,
            "mse": 0.0,
            "auxk": 0.0,
            "l0": 0.0,
            "dead": 0.0,
        }
        for micro_start in range(
            0, config.batch_size, config.adapter_microbatch
        ):
            micro_end = min(
                micro_start + config.adapter_microbatch, config.batch_size
            )
            seq = sequence_indices[micro_start:micro_end]
            pos = positions[micro_start:micro_end]
            idx = torch.from_numpy(
                np.asarray(code_indices[seq[:, None], pos], dtype=np.int64)
            ).to(device)
            val = torch.from_numpy(
                np.asarray(code_values[seq[:, None], pos], dtype=np.float32)
            ).to(device=device, dtype=dtype)
            target = torch.from_numpy(
                np.asarray(raw_cache[seq[:, None], pos], dtype=np.float32)
            ).to(device=device, dtype=dtype)
            fraction = (micro_end - micro_start) / config.batch_size
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.startswith("cuda"),
            ):
                result = model.loss(idx, val, target, update_dead=True)
                scaled_loss = result["loss"] * fraction
            scaled_loss.backward()
            for name in accumulator:
                accumulator[name] += (
                    float(result[name].detach().float().cpu()) * fraction
                )
        if config.warmup_steps:
            lr_scale = min(1.0, (step + 1) / config.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = config.learning_rate * lr_scale
        optimizer.step()
        model.normalize_decoder_profiles()
        last_metrics = accumulator
        completed = step + 1
        if (
            completed == config.adapter_steps
            or completed % config.checkpoint_every == 0
        ):
            _save_adapter(
                model=model,
                optimizer=optimizer,
                step=completed,
                metrics=last_metrics,
                checkpoint_root=checkpoint_root,
                rank=rank,
            )
            _heartbeat(
                output_root,
                stage=f"adapter_rank{rank}_train",
                step=completed,
                total=config.adapter_steps,
                metrics=last_metrics,
            )
            print(
                f"[adapter] rank={rank} step={completed}/"
                f"{config.adapter_steps} loss={last_metrics['loss']:.6g} "
                f"l0={last_metrics['l0']:.3f}",
                flush=True,
            )
    result = {
        "rank": rank,
        "completed_steps": config.adapter_steps,
        "cached": False,
        "last_metrics": last_metrics,
        "trainable_parameters": model.trainable_parameter_count(),
    }
    atomic_json(result, output_root / f"adapter_rank{rank}_train.json")
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _order_indices(n: int, window: int, *, mode: str, seed: int) -> np.ndarray:
    if mode == "ordered":
        return np.broadcast_to(np.arange(window), (n, window)).copy()
    rng = np.random.default_rng(seed)
    if mode == "shuffle":
        return np.stack([rng.permutation(window) for _ in range(n)])
    if mode == "reverse":
        return np.broadcast_to(np.arange(window - 1, -1, -1), (n, window))
    if mode == "circular":
        shifts = rng.integers(1, window, size=n)
        base = np.arange(window)
        return np.stack([np.roll(base, int(shift)) for shift in shifts])
    raise ValueError(mode)


def _permute_sparse_tokens(
    indices: np.ndarray,
    values: np.ndarray,
    order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selector = np.broadcast_to(order[:, :, None], indices.shape)
    return (
        np.take_along_axis(indices, selector, axis=1),
        np.take_along_axis(values, selector, axis=1),
    )


def _csr_topk(
    values: np.ndarray,
    indices: np.ndarray,
    width: int,
) -> sparse.csr_matrix:
    n, k = values.shape
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    columns = indices.reshape(-1).astype(np.int64, copy=False)
    data = values.reshape(-1).astype(np.float32, copy=False)
    keep = data > 0
    return sparse.csr_matrix(
        (data[keep], (rows[keep], columns[keep])),
        shape=(n, width),
    )


def _csr_positional(
    values: np.ndarray,
    indices: np.ndarray,
    width: int,
) -> sparse.csr_matrix:
    shifted = indices + (
        np.arange(indices.shape[1], dtype=np.int64)[None, :, None] * width
    )
    return _csr_topk(
        values.reshape(len(values), -1),
        shifted.reshape(len(indices), -1),
        width * indices.shape[1],
    )


@torch.no_grad()
def _encode_artifact_tokens(
    *,
    x: np.ndarray,
    checkpoint_root: Path,
    config: RunConfig,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model, _ = load_dictionary(checkpoint_root / "base_sae", device=device)
    dtype = model.W_enc.dtype
    all_indices = np.empty(
        (len(x), config.window, config.k_pos), dtype=np.int32
    )
    all_values = np.empty_like(all_indices, dtype=np.float16)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        raw = torch.from_numpy(
            np.asarray(x[start:end], dtype=np.float32)
        ).to(device=device, dtype=dtype)
        flat = raw.reshape(-1, config.d_in)
        pre = (flat - model.b_dec) @ model.W_enc.T + model.b_enc
        values, indices = pre.topk(config.k_pos, dim=-1)
        shape = (end - start, config.window, config.k_pos)
        all_indices[start:end] = indices.reshape(shape).cpu().numpy()
        all_values[start:end] = (
            torch.relu(values).reshape(shape).float().cpu().numpy()
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return all_indices, all_values


@torch.no_grad()
def _encode_adapter(
    *,
    rank: int,
    indices: np.ndarray,
    values: np.ndarray,
    checkpoint_root: Path,
    config: RunConfig,
    device: str,
    batch_size: int,
) -> sparse.csr_matrix:
    model, _, step, _ = _load_adapter(
        rank=rank,
        config=config,
        checkpoint_root=checkpoint_root,
        device=device,
    )
    if step != config.adapter_steps:
        raise ValueError(f"rank {rank} is only at step {step}")
    model.eval()
    output_values = np.empty((len(indices), config.k_window), dtype=np.float16)
    output_indices = np.empty(
        (len(indices), config.k_window), dtype=np.int32
    )
    dtype = next(model.parameters()).dtype
    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        idx = torch.from_numpy(indices[start:end].astype(np.int64)).to(device)
        val = torch.from_numpy(values[start:end].astype(np.float32)).to(
            device=device, dtype=dtype
        )
        active_values, active_indices, _ = model.encode_sparse(idx, val)
        output_values[start:end] = active_values.float().cpu().numpy()
        output_indices[start:end] = active_indices.cpu().numpy()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return _csr_topk(output_values, output_indices, config.d_sae)


def _invariant_max(
    values: np.ndarray,
    indices: np.ndarray,
    width: int,
    k_window: int,
) -> sparse.csr_matrix:
    rows = []
    columns = []
    data = []
    for row in range(len(values)):
        maxima: dict[int, float] = {}
        for feature, value in zip(
            indices[row].reshape(-1), values[row].reshape(-1)
        ):
            value_float = float(value)
            if value_float > maxima.get(int(feature), 0.0):
                maxima[int(feature)] = value_float
        selected = sorted(maxima.items(), key=lambda item: item[1])[-k_window:]
        rows.extend([row] * len(selected))
        columns.extend(feature for feature, _ in selected)
        data.extend(value for _, value in selected)
    return sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32), (rows, columns)),
        shape=(len(values), width),
    )


def _probe(
    matrix: sparse.csr_matrix,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    s_grid: tuple[int, ...],
    seed: int,
) -> list[dict]:
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    rows: list[dict] = []
    for fold, (train, test) in enumerate(splitter.split(matrix, y, groups)):
        positive = matrix[train[y[train] == 1]].mean(axis=0).A1
        negative = matrix[train[y[train] == 0]].mean(axis=0).A1
        ranking = np.argsort(np.abs(positive - negative))
        for requested in s_grid:
            selected = ranking[-min(requested, matrix.shape[1]) :]
            classifier = LogisticRegression(
                penalty="l1",
                C=1.0,
                solver="liblinear",
                max_iter=2_000,
                random_state=seed + fold,
            ).fit(matrix[train][:, selected].toarray(), y[train])
            probability = classifier.predict_proba(
                matrix[test][:, selected].toarray()
            )[:, 1]
            rows.append(
                {
                    "fold": fold,
                    "n_features": requested,
                    "pr_auc": float(
                        average_precision_score(y[test], probability)
                    ),
                    "roc_auc": float(roc_auc_score(y[test], probability)),
                    "n_train": len(train),
                    "n_test": len(test),
                }
            )
    return rows


def _summarize_probe(rows: list[dict]) -> list[dict]:
    summaries = []
    for n_features in sorted({row["n_features"] for row in rows}):
        subset = [row for row in rows if row["n_features"] == n_features]
        values = np.asarray([row["pr_auc"] for row in subset])
        summaries.append(
            {
                "n_features": n_features,
                "pr_auc_mean": float(values.mean()),
                "pr_auc_std_sample": float(values.std(ddof=1)),
                "fold_values": values.tolist(),
            }
        )
    return summaries


def evaluate(
    *,
    artifact: Path,
    checkpoint_root: Path,
    output_root: Path,
    config: RunConfig,
    device: str,
    encode_batch_size: int,
) -> dict:
    result_path = output_root / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("status") == "complete":
            return result
    with np.load(artifact, allow_pickle=True) as payload:
        x = payload["X"][:, -config.window :, :]
        y = payload["is_bt"].astype(np.int64)
        groups = np.asarray(
            [str(key).split("|", 1)[0] for key in payload["keys"]],
            dtype=object,
        )
    indices, values = _encode_artifact_tokens(
        x=x,
        checkpoint_root=checkpoint_root,
        config=config,
        device=device,
        batch_size=encode_batch_size,
    )
    matrices: dict[str, dict[str, sparse.csr_matrix]] = {
        "shared_sae_positional": {},
        "shared_sae_invariant": {},
        "shared_sae_last_token": {},
        **{f"trajectory_rank{rank}": {} for rank in config.ranks},
    }
    for control_index, condition in enumerate(("ordered", *ORDER_CONTROLS)):
        order = _order_indices(
            len(x),
            config.window,
            mode=condition,
            seed=config.seed + 10_000 + control_index,
        )
        condition_indices, condition_values = _permute_sparse_tokens(
            indices, values, order
        )
        matrices["shared_sae_positional"][condition] = _csr_positional(
            condition_values, condition_indices, config.d_sae
        )
        matrices["shared_sae_last_token"][condition] = _csr_topk(
            condition_values[:, -1],
            condition_indices[:, -1],
            config.d_sae,
        )
        matrices["shared_sae_invariant"][condition] = _invariant_max(
            condition_values,
            condition_indices,
            config.d_sae,
            config.k_window,
        )
        for rank in config.ranks:
            matrices[f"trajectory_rank{rank}"][condition] = _encode_adapter(
                rank=rank,
                indices=condition_indices,
                values=condition_values,
                checkpoint_root=checkpoint_root,
                config=config,
                device=device,
                batch_size=encode_batch_size,
            )
        _heartbeat(output_root, stage=f"encode_eval_{condition}")

    probes = {}
    order_gaps = {}
    for name, conditions in matrices.items():
        probes[name] = _summarize_probe(
            _probe(
                conditions["ordered"],
                y,
                groups,
                folds=config.folds,
                s_grid=config.s_grid,
                seed=config.seed,
            )
        )
        control_summaries = {
            condition: _summarize_probe(
                _probe(
                    matrix,
                    y,
                    groups,
                    folds=config.folds,
                    s_grid=config.s_grid,
                    seed=config.seed,
                )
            )
            for condition, matrix in conditions.items()
            if condition != "ordered"
        }
        order_gaps[name] = {
            condition: [
                {
                    "n_features": ordered["n_features"],
                    "ordered_minus_control_pr_auc": (
                        ordered["pr_auc_mean"] - control["pr_auc_mean"]
                    ),
                }
                for ordered, control in zip(
                    probes[name], control_summaries[condition]
                )
            ]
            for condition in ORDER_CONTROLS
        }
    result = {
        "status": "complete",
        "protocol": PROTOCOL,
        "config": {**asdict(config), "ranks": list(config.ranks)},
        "artifact_sha256": sha256(artifact),
        "n_rows": len(y),
        "positive_rate": float(y.mean()),
        "probes": probes,
        "order_gaps": order_gaps,
        "published_references": PUBLISHED_REFERENCES,
    }
    atomic_json(result, result_path)
    _plot(result, output_root / "comparison_s32.png")
    return result


def _plot(result: dict, path: Path) -> None:
    names = []
    values = []
    for name, summaries in result["probes"].items():
        summary = next(
            row for row in summaries if int(row["n_features"]) == 32
        )
        names.append(name.replace("_", " "))
        values.append(summary["pr_auc_mean"])
    for name, record in PUBLISHED_REFERENCES.items():
        names.append(name)
        values.append(record["pr_auc_s32"])
    colors = [
        "#4C78A8" if "trajectory" in name else "#9ecae9"
        for name in names[: len(result["probes"])]
    ] + ["#F58518"] * len(PUBLISHED_REFERENCES)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    positions = np.arange(len(names))
    axis.bar(positions, values, color=colors)
    axis.axhline(float(np.mean(result.get("positive_rate", 0))), color="0.5")
    axis.set_xticks(positions, names, rotation=25, ha="right")
    axis.set_ylabel("Question-grouped PR-AUC (S=32)")
    axis.set_title("C7 backtracking: learned two-stage trajectory bottleneck")
    axis.set_ylim(0, max(values) * 1.2)
    for position, value in zip(positions, values):
        axis.text(position, value + 0.004, f"{value:.3f}", ha="center")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-cache", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--phase",
        choices=("all", "base", "codes", "adapters", "eval"),
        default="all",
    )
    parser.add_argument("--base-steps", type=int, default=300_000)
    parser.add_argument("--adapter-steps", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--adapter-microbatch", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument("--ranks", default="0,256")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    ranks = tuple(int(value) for value in args.ranks.split(",") if value)
    config = RunConfig(
        base_steps=args.base_steps,
        adapter_steps=args.adapter_steps,
        batch_size=args.batch_size,
        adapter_microbatch=args.adapter_microbatch,
        checkpoint_every=args.checkpoint_every,
        ranks=ranks,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    inventory = _validate_inputs(args.activation_cache, args.artifact)
    plan = {
        "protocol": PROTOCOL,
        "inventory": inventory,
        "config": {**asdict(config), "ranks": list(config.ranks)},
        "phase": args.phase,
        "device": args.device,
        "checkpoint_root": str(args.checkpoint_root),
        "output_root": str(args.output_root),
    }
    atomic_json(plan, args.output_root / "plan.json")
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    if args.phase in {"all", "base"}:
        train_base(
            activation_cache=args.activation_cache,
            checkpoint_root=args.checkpoint_root,
            config=config,
            device=args.device,
            output_root=args.output_root,
        )
    if args.phase in {"all", "codes"}:
        precompute_base_codes(
            activation_cache=args.activation_cache,
            checkpoint_root=args.checkpoint_root,
            code_root=args.checkpoint_root / "base_codes",
            config=config,
            device=args.device,
            output_root=args.output_root,
        )
    if args.phase in {"all", "adapters"}:
        for rank in config.ranks:
            train_adapter(
                rank=rank,
                activation_cache=args.activation_cache,
                code_root=args.checkpoint_root / "base_codes",
                checkpoint_root=args.checkpoint_root,
                output_root=args.output_root,
                config=config,
                device=args.device,
            )
    if args.phase in {"all", "eval"}:
        result = evaluate(
            artifact=args.artifact,
            checkpoint_root=args.checkpoint_root,
            output_root=args.output_root,
            config=config,
            device=args.device,
            encode_batch_size=args.encode_batch_size,
        )
        print(json.dumps({"status": result["status"]}, sort_keys=True), flush=True)
    _heartbeat(args.output_root, stage="complete")


if __name__ == "__main__":
    main()
