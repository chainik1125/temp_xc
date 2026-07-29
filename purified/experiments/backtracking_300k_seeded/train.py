"""Train one deterministic, historical C7 dictionary cell without evaluation.

This script intentionally imports architecture/config/trainer code from an
isolated archive of the historical paper commit. The only semantic correction
is seeding Python, NumPy, and PyTorch before model construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

HISTORICAL_COMMIT = "284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3"
PROTOCOL_VERSION = "c7-300k-seeded-v1"
DATASOURCE = "llama_3_1_8b_base_l10_ward_nousmirror"
ACT_CACHE_KEY = "fb2a74be884e512a"
ACT_CACHE_SHA256 = "dc34dfb117f77abddef4b4396d0d00afc707c39876d0ee36015de1e7b8406914"
EXPECTED_KEYS = {
    ("txc_base", 32768, 1): "a300c63374c3597e",
    ("txc_base", 32768, 2): "27078b0d7700ae05",
    ("txc_base", 32768, 42): "8787f8fe527218ad",
    ("tsae_paper", 16384, 42): "b97e3c00153a5271",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-file", type=Path, required=True)
    parser.add_argument("--arch", choices=("txc_base", "tsae_paper"), required=True)
    parser.add_argument("--d-sae", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-steps", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--skip-cache-sha256", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate provenance and the exact train key without constructing the model.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    historical_root = args.historical_root.resolve()
    output_root = args.output_root.resolve()
    cache_file = args.cache_file.resolve()

    commit_marker = historical_root / "HISTORICAL_COMMIT"
    if commit_marker.read_text().strip() != HISTORICAL_COMMIT:
        raise RuntimeError(f"historical source marker mismatch: {commit_marker}")
    if not cache_file.is_file():
        raise FileNotFoundError(cache_file)
    if not args.skip_cache_sha256 and _sha256(cache_file) != ACT_CACHE_SHA256:
        raise RuntimeError("activation cache SHA-256 mismatch")
    if args.n_steps != 300_000 or args.batch_size != 1_024:
        raise ValueError("production protocol requires 300000 steps and batch size 1024")

    key_tuple = (args.arch, args.d_sae, args.seed)
    if key_tuple not in EXPECTED_KEYS:
        raise ValueError(f"cell is outside the locked production grid: {key_tuple}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["TEMP_BENCH_ROOT"] = str(historical_root)
    sys.path.insert(0, str(historical_root / "src"))

    import numpy as np
    import torch

    from temp_bench.config import (
        compute_act_cache_key,
        compute_train_key,
        instantiate_arch,
        load_arch,
        load_datasource,
    )
    from temp_bench.schemas import TrainingConfig
    from temp_bench.training.sae_trainer import train_sae
    from temp_bench.utils.seed import set_seed

    set_seed(args.seed)
    datasource = load_datasource(DATASOURCE)
    actual_cache_key = compute_act_cache_key(datasource)
    if actual_cache_key != ACT_CACHE_KEY:
        raise RuntimeError(
            f"historical datasource hash mismatch: {actual_cache_key} != {ACT_CACHE_KEY}"
        )

    spec = load_arch(args.arch, component="c7")
    spec = spec.model_copy(
        update={"hparams": {**spec.hparams, "d_sae": int(args.d_sae)}}
    )
    training_cfg = TrainingConfig(n_steps=args.n_steps, batch_size=args.batch_size)
    train_key = compute_train_key(
        arch=spec,
        seed=args.seed,
        training_cfg=training_cfg,
        act_cache_key=ACT_CACHE_KEY,
    )
    expected_key = EXPECTED_KEYS[key_tuple]
    if train_key != expected_key:
        raise RuntimeError(f"train-key mismatch: {train_key} != {expected_key}")
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight-complete",
                    "protocol_version": PROTOCOL_VERSION,
                    "arch": args.arch,
                    "d_sae": args.d_sae,
                    "seed": args.seed,
                    "train_key": train_key,
                    "historical_commit": HISTORICAL_COMMIT,
                    "act_cache_key": ACT_CACHE_KEY,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    cell_dir = output_root / "cells" / f"{args.arch}_d{args.d_sae}_seed{args.seed}"
    final_dir = cell_dir / "checkpoint"
    progress_path = cell_dir / "progress.json"
    if (final_dir / "model.safetensors").exists():
        print(f"complete checkpoint already exists: {final_dir}", flush=True)
        return 0
    if cell_dir.exists() and any(cell_dir.iterdir()):
        raise RuntimeError(
            f"non-empty incomplete cell directory requires manual audit: {cell_dir}"
        )
    cell_dir.mkdir(parents=True, exist_ok=True)

    acts_np = np.load(cache_file, mmap_mode="r")
    if tuple(acts_np.shape) != (4044, 128, 4096) or acts_np.dtype != np.float16:
        raise RuntimeError(
            f"unexpected activation cache contract: shape={acts_np.shape}, dtype={acts_np.dtype}"
        )
    acts = torch.from_numpy(np.ascontiguousarray(acts_np)).clone()
    if torch.cuda.is_available():
        # The 4 GB cache fits comfortably beside every production model on
        # A40/H100. Keeping it on the pinned worker GPU removes an ~80 MB
        # host-to-device transfer and CPU gather from every training step.
        acts = acts.cuda()
    n_sequences, sequence_length, d_in = acts.shape

    set_seed(args.seed)
    model = instantiate_arch(spec, d_in=d_in)
    if torch.cuda.is_available():
        model = model.cuda()
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    if n_parameters > 1_000_000_000 and torch.cuda.is_available():
        model = model.bfloat16()

    if args.arch == "txc_base":
        window_size = int(spec.hparams["T"])
    else:
        window_size = 5
    if sequence_length < window_size:
        raise RuntimeError("activation sequences are shorter than the training window")

    sampler = np.random.default_rng(args.seed)

    def batch_iter(n: int) -> torch.Tensor:
        sequence_indices = sampler.integers(0, n_sequences, size=n)
        position_indices = sampler.integers(0, sequence_length - window_size + 1, size=n)
        sequence_index = torch.as_tensor(
            sequence_indices.astype(np.int64, copy=False), device=acts.device
        )
        position_index = torch.as_tensor(
            position_indices.astype(np.int64, copy=False), device=acts.device
        )
        offsets = torch.arange(window_size, dtype=torch.int64, device=acts.device)
        # This is bit-for-bit equal to the historical per-row copy loop, but
        # avoids 1,024 Python-level tensor assignments on every training step.
        return acts[
            sequence_index[:, None],
            position_index[:, None] + offsets[None, :],
        ].float()

    started = time.time()
    metadata = {
        "status": "running",
        "protocol_version": PROTOCOL_VERSION,
        "historical_commit": HISTORICAL_COMMIT,
        "arch": args.arch,
        "arch_version": spec.arch_version,
        "hparams": spec.hparams,
        "seed": args.seed,
        "d_sae": args.d_sae,
        "train_key": train_key,
        "act_cache_key": ACT_CACHE_KEY,
        "act_cache_sha256": ACT_CACHE_SHA256,
        "training_cfg": training_cfg.model_dump(),
        "n_parameters": n_parameters,
        "started_unix": started,
        "host": os.uname().nodename,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed_correction": "Python, NumPy, CPU Torch, and all CUDA RNGs seeded before init",
    }
    _atomic_json(cell_dir / "manifest.json", metadata)

    def record_progress(step: int, payload: dict[str, Any]) -> None:
        log = payload["log"]
        elapsed = time.time() - started
        completed = step + 1
        _atomic_json(
            progress_path,
            {
                "status": "running",
                "step": completed,
                "n_steps": args.n_steps,
                "elapsed_seconds": elapsed,
                "steps_per_second": completed / max(elapsed, 1e-9),
                "eta_seconds": (args.n_steps - completed)
                / max(completed / max(elapsed, 1e-9), 1e-9),
                "loss": log["loss"][-1],
                "mse": log["mse"][-1],
                "l0": log["l0"][-1],
            },
        )
        print(
            f"step={completed}/{args.n_steps} loss={log['loss'][-1]:.6g} "
            f"steps_per_second={completed / max(elapsed, 1e-9):.3f}",
            flush=True,
        )

    result = train_sae(
        model,
        batch_iter,
        training_cfg,
        snapshot_every=args.progress_every,
        snapshot_fn=record_progress,
    )

    from safetensors.torch import save_file

    final_dir.mkdir(parents=True, exist_ok=False)
    state_dict = {
        name: tensor.detach().contiguous().cpu()
        for name, tensor in result["state_dict"].items()
    }
    temporary_model = final_dir / "model.safetensors.tmp"
    save_file(state_dict, str(temporary_model))
    temporary_model.replace(final_dir / "model.safetensors")
    finished = time.time()
    final_metadata = {
        **metadata,
        "status": "complete",
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "n_steps_completed": int(result["n_steps"]),
    }
    _atomic_json(final_dir / "config.json", final_metadata)
    _atomic_json(
        progress_path,
        {
            "status": "complete",
            "step": int(result["n_steps"]),
            "n_steps": args.n_steps,
            "elapsed_seconds": finished - started,
        },
    )
    print(f"complete train_key={train_key} checkpoint={final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
