"""Run the exact-paper C7 TXC-base seed top-up on a persistent RunPod.

This is the non-Modal counterpart of ``modal_c7_paper_multiseed.py``.  The
scientific training path is still delegated to the frozen 300k paper runner;
this file only stages immutable artifacts, verifies the seed-42 protocol key,
and persists the two missing seed results.

Expected setup::

    git worktree add --detach /workspace/c7-paper \
        b8ab4b95dc8d5a7b6da28bdcb71acfaa9c42aff5
    cd /workspace/c7-paper/purified
    uv sync --frozen
    PYTHONPATH=/workspace/c7-paper/purified/src:/workspace/c7-paper/purified \
      .venv/bin/python /workspace/reviewer-driver/scripts/runpod_c7_paper_multiseed.py
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


C7_PIN = "b8ab4b95dc8d5a7b6da28bdcb71acfaa9c42aff5"
HF_DATA_REPO = "han1823123123/temp-bench-data"
HF_DATA_REVISION = "6ef9b1debf863dedcef9555cad3a4903fb9e8c43"
ACT_CACHE_KEY = "fb2a74be884e512a"
EVENT_SHA256 = (
    "1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810"
)
PUBLISHED_SEED42_PREFIX = "8787f8fe5272"
ARCH = "txc_base"
DEFAULT_SEEDS = (1, 2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_pin(repo_root: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != C7_PIN:
        raise RuntimeError(f"C7 checkout is {head}, expected frozen pin {C7_PIN}")
    print(f"[pin] C7 paper runner {head}", flush=True)


def stage_assets(output_root: Path) -> dict:
    """Download and verify only the immutable C7 cache/eval artifacts."""

    from huggingface_hub import snapshot_download

    root = output_root / "assets"
    snapshot_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        revision=HF_DATA_REVISION,
        local_dir=root,
        allow_patterns=[
            f"act_cache/{ACT_CACHE_KEY}/*",
            "c7_backtracking/stage_a/sentence_acts_L10.npz",
        ],
    )
    acts = root / f"act_cache/{ACT_CACHE_KEY}/resid_post_L10.npy"
    specs = root / f"act_cache/{ACT_CACHE_KEY}/layer_specs.json"
    event = root / "c7_backtracking/stage_a/sentence_acts_L10.npz"
    missing = [str(path) for path in (acts, specs, event) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing staged C7 assets: {missing}")
    observed_sha = _sha256(event)
    if observed_sha != EVENT_SHA256:
        raise ValueError(
            f"official C7 sentence activation SHA mismatch: {observed_sha}"
        )

    import numpy as np

    cache = np.load(acts, mmap_mode="r")
    with np.load(event, allow_pickle=True) as payload:
        event_shape = tuple(payload["X"].shape)
    if tuple(cache.shape) != (4_044, 128, 4_096):
        raise ValueError(f"unexpected train-cache shape: {cache.shape}")
    if event_shape != (25_204, 6, 4_096):
        raise ValueError(f"unexpected event-cache shape: {event_shape}")
    manifest = {
        "status": "complete",
        "revision": HF_DATA_REVISION,
        "train_cache_shape": list(cache.shape),
        "event_shape": list(event_shape),
        "event_sha256": observed_sha,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def _replace_with_symlink(path: Path, target: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, path)


def _prepare_cell_filesystem(
    paper_root: Path, output_root: Path, seed: int
) -> tuple[Path, Path]:
    """Put random-read assets and durable outputs on the RunPod volume."""

    assets = output_root / "assets"
    cell_root = output_root / "cells" / f"{ARCH}_seed{seed}"
    cache_root = cell_root / "act_cache" / ACT_CACHE_KEY
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        assets / f"act_cache/{ACT_CACHE_KEY}",
        cache_root,
        dirs_exist_ok=True,
    )
    event_path = cell_root / "sentence_acts_L10.npz"
    shutil.copy2(
        assets / "c7_backtracking/stage_a/sentence_acts_L10.npz",
        event_path,
    )

    purified = paper_root / "purified"
    _replace_with_symlink(purified / "results" / "act_cache", cell_root / "act_cache")
    checkpoint_root = cell_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    _replace_with_symlink(purified / "checkpoints", checkpoint_root)
    return event_path, cell_root


def _encode_features(model, activations, *, shuffled: bool, shuffle_seed: int = 42):
    import numpy as np
    import torch

    arch_t = int(getattr(model.config, "T", 1) or 1)
    x = activations[:, -arch_t:, :]
    if shuffled and arch_t > 1:
        rng = np.random.default_rng(shuffle_seed)
        permutations = np.argsort(rng.random((len(x), arch_t)), axis=1)
        x = np.take_along_axis(x, permutations[:, :, None], axis=1)

    chunks = []
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(x), 512):
            xb = torch.from_numpy(x[start : start + 512]).to(device, dtype=dtype)
            z = model.encode(xb).abs().float()
            if z.ndim == 3:
                z = z.amax(dim=1)
            chunks.append(z.cpu().numpy())
            del xb, z
    return np.concatenate(chunks, axis=0)


def train_and_detect(paper_root: Path, output_root: Path, seed: int) -> dict:
    """Train one exact 300k paper cell, then run ordered/shuffled detection."""

    import numpy as np
    import torch

    if seed not in DEFAULT_SEEDS:
        raise ValueError(f"seed must be one of {DEFAULT_SEEDS}")
    purified = paper_root / "purified"
    os.chdir(purified)
    event_path, cell_root = _prepare_cell_filesystem(paper_root, output_root, seed)
    result_path = cell_root / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text())
        if payload.get("status") == "complete":
            print(f"[resume] seed {seed} result already complete", flush=True)
            return payload

    from experiments.c7_backtracking import run as c7_run
    from temp_bench.cache import checkpoint_exists, load_checkpoint_state_dict
    from temp_bench.cache import save_checkpoint
    from temp_bench.config import (
        compute_act_cache_key,
        compute_train_key,
        load_arch,
        load_datasource,
    )
    from temp_bench.schemas import TrainingConfig

    cfg = TrainingConfig(n_steps=300_000, batch_size=1_024)
    spec = load_arch(ARCH, component="c7")
    datasource = load_datasource(c7_run.DATASOURCE)
    act_cache_key = compute_act_cache_key(datasource)
    if act_cache_key != ACT_CACHE_KEY:
        raise RuntimeError(
            f"paper cache key drift: expected {ACT_CACHE_KEY}, got {act_cache_key}"
        )
    seed42_key = compute_train_key(
        arch=spec,
        seed=42,
        training_cfg=cfg,
        act_cache_key=act_cache_key,
    )
    if not seed42_key.startswith(PUBLISHED_SEED42_PREFIX):
        raise RuntimeError(
            f"paper-protocol gate failed: seed42 key {seed42_key} does not "
            f"match {PUBLISHED_SEED42_PREFIX}"
        )
    train_key = compute_train_key(
        arch=spec,
        seed=seed,
        training_cfg=cfg,
        act_cache_key=act_cache_key,
    )
    running_path = cell_root / "RUNNING.json"
    running_path.write_text(
        json.dumps(
            {
                "status": "running",
                "seed": seed,
                "train_key": train_key,
                "seed42_protocol_key": seed42_key,
                "started_at": time.time(),
            },
            indent=2,
        )
    )
    print(
        f"[gate] seed42={seed42_key}; seed{seed}={train_key}; starting 300k",
        flush=True,
    )

    started = time.time()
    if checkpoint_exists(train_key):
        state = load_checkpoint_state_dict(train_key)
        cached = True
    else:
        state = c7_run.my_train_fn(
            arch_name=ARCH,
            arch_hparams=spec.hparams,
            seed=seed,
            training_cfg=cfg,
            act_cache_key=act_cache_key,
            component="c7",
            probe_every=0,
            snapshot_state_at=(),
        )
        save_checkpoint(
            train_key=train_key,
            arch=ARCH,
            arch_version=spec.arch_version,
            seed=seed,
            datasource=c7_run.DATASOURCE,
            act_cache_key=act_cache_key,
            training_cfg=cfg.model_dump(),
            state_dict=state,
            agent="reviewer-multiseed-runpod",
        )
        cached = False
        print(f"[checkpoint] seed {seed} saved at {train_key}", flush=True)

    model = c7_run._instantiate_from_state(ARCH, state, component="c7")
    model = model.bfloat16().eval()
    with np.load(event_path, allow_pickle=True) as event:
        x = event["X"]
        labels = event["is_bt"].astype(np.int64)
        keys = event["keys"]
    qids = np.asarray([str(key).split("|")[0] for key in keys], dtype=object)

    from temp_bench.case_studies.backtracking import compute_probe_metrics_at_S

    variants = {}
    for name, shuffled in (("ordered", False), ("shuffled", True)):
        features = _encode_features(model, x, shuffled=shuffled)
        probe = compute_probe_metrics_at_S(
            features,
            labels,
            qids,
            S_grid=(1, 2, 4, 8, 16, 32),
            n_folds=5,
            random_state=42,
        )
        variants[name] = {
            "pr_auc": {str(k): v for k, v in probe["pr_auc"].items()},
            "roc_auc": {str(k): v for k, v in probe["roc_auc"].items()},
        }
        del features
        gc.collect()

    payload = {
        "status": "complete",
        "protocol": "C7 exact-paper 300k TXC-base + paired shuffle detection",
        "code_pin": C7_PIN,
        "asset_revision": HF_DATA_REVISION,
        "arch": ARCH,
        "seed": seed,
        "seed42_protocol_key": seed42_key,
        "train_key": train_key,
        "training_cfg": cfg.model_dump(),
        "cached_checkpoint": cached,
        "variants": variants,
        "shuffle_gap_pr_auc": {
            key: variants["ordered"]["pr_auc"][key]
            - variants["shuffled"]["pr_auc"][key]
            for key in variants["ordered"]["pr_auc"]
        },
        "elapsed_seconds": round(time.time() - started),
        "gpu": torch.cuda.get_device_name(0),
    }
    result_path.write_text(json.dumps(payload, indent=2))
    running_path.unlink(missing_ok=True)
    (cell_root / "DONE").write_text("complete\n")
    print(f"[complete] seed {seed} -> {result_path}", flush=True)
    return payload


def read_status(output_root: Path) -> dict:
    cells = {}
    for cell_dir in sorted((output_root / "cells").glob("txc_base_seed*")):
        result = cell_dir / "result.json"
        running = cell_dir / "RUNNING.json"
        cells[cell_dir.name] = {
            "complete": result.exists(),
            "running": running.exists(),
            "checkpoint_files": sum(
                1 for path in (cell_dir / "checkpoints").rglob("*") if path.is_file()
            ),
        }
    return {"code_pin": C7_PIN, "asset_revision": HF_DATA_REVISION, "cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, default=Path("/workspace/c7-paper"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/reviewer_multiseed/c7"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--skip-stage", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        print(json.dumps(read_status(args.output_root), indent=2))
        return
    _assert_pin(args.paper_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_stage:
        stage_assets(args.output_root)
    if args.stage_only:
        return
    for seed in args.seeds:
        try:
            train_and_detect(args.paper_root, args.output_root, seed)
        except Exception as error:
            cell_root = args.output_root / "cells" / f"{ARCH}_seed{seed}"
            cell_root.mkdir(parents=True, exist_ok=True)
            (cell_root / "FAILED.json").write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "seed": seed,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "failed_at": time.time(),
                    },
                    indent=2,
                )
            )
            raise
    print(json.dumps(read_status(args.output_root), indent=2), flush=True)


if __name__ == "__main__":
    main()
