"""Exact-paper C7 TXC-base seed top-up on Modal.

This is an orchestration-only wrapper around the frozen 300k C7 runner at
``origin/300k-tfa``.  It trains the two missing TXC-base seeds and performs a
GPU-side, question-grouped detection evaluation on the official sentence
activation artifact, both ordered and deterministically shuffled.

The scientific training path is delegated to
``experiments.c7_backtracking.run.my_train_fn`` at ``C7_PIN``.  The wrapper
asserts that the same inputs reproduce the published seed-42 train-key prefix
before spending compute.

Launch:

    modal run --detach scripts/modal_c7_paper_multiseed.py
    modal run scripts/modal_c7_paper_multiseed.py --status
"""

from __future__ import annotations

import json
from pathlib import Path

import modal


C7_PIN = "b8ab4b95dc8d5a7b6da28bdcb71acfaa9c42aff5"
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
HF_DATA_REPO = "han1823123123/temp-bench-data"
HF_DATA_REVISION = "6ef9b1debf863dedcef9555cad3a4903fb9e8c43"
ACT_CACHE_KEY = "fb2a74be884e512a"
EVENT_SHA256 = (
    "1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810"
)
PUBLISHED_SEED42_PREFIX = "8787f8fe5272"
ARCH = "txc_base"
SEEDS = (1, 2)

app = modal.App("temp-xc-c7-paper-multiseed")
vol = modal.Volume.from_name("temp-xc-c7-paper-multiseed", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install("uv")
    .run_commands(
        f"git clone {REPO_URL} /repo",
        f"git -C /repo checkout {C7_PIN}",
        "uv pip install --system -e /repo/purified",
    )
    .pip_install("huggingface_hub>=0.26", "hf-xet>=1.1")
    .env(
        {
            "TEMP_BENCH_ROOT": "/repo/purified",
            "PYTHONPATH": "/repo/purified/src:/repo/purified",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=image,
    cpu=4,
    memory=16_384,
    timeout=2 * 60 * 60,
    volumes={"/workspace": vol},
)
def stage_assets() -> dict:
    """Download and verify only the immutable C7 cache/eval artifacts."""

    from huggingface_hub import snapshot_download

    root = Path("/workspace/c7_paper/assets")
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
    if _sha256(event) != EVENT_SHA256:
        raise ValueError("official C7 sentence activation SHA-256 mismatch")

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
        "event_sha256": EVENT_SHA256,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    vol.commit()
    return manifest


def _prepare_cell_filesystem(arch: str, seed: int) -> tuple[Path, Path]:
    """Put random-read assets on local SSD and outputs on the Volume."""

    import os
    import shutil

    assets = Path("/workspace/c7_paper/assets")
    local_root = Path(f"/tmp/c7_paper_{arch}_seed{seed}")
    local_cache = local_root / "act_cache" / ACT_CACHE_KEY
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        assets / f"act_cache/{ACT_CACHE_KEY}",
        local_cache,
        dirs_exist_ok=True,
    )
    local_event = local_root / "sentence_acts_L10.npz"
    shutil.copy2(
        assets / "c7_backtracking/stage_a/sentence_acts_L10.npz",
        local_event,
    )

    repo_results = Path("/repo/purified/results")
    repo_act_cache = repo_results / "act_cache"
    if repo_act_cache.is_symlink() or repo_act_cache.is_file():
        repo_act_cache.unlink()
    elif repo_act_cache.exists():
        shutil.rmtree(repo_act_cache)
    os.symlink(local_root / "act_cache", repo_act_cache)

    cell_root = Path(f"/workspace/c7_paper/cells/{arch}_seed{seed}")
    checkpoint_root = cell_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    repo_checkpoints = Path("/repo/purified/checkpoints")
    if repo_checkpoints.is_symlink() or repo_checkpoints.is_file():
        repo_checkpoints.unlink()
    elif repo_checkpoints.exists():
        shutil.rmtree(repo_checkpoints)
    os.symlink(checkpoint_root, repo_checkpoints)
    return local_event, cell_root


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


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=65_536,
    timeout=16 * 60 * 60,
    max_containers=2,
    retries=modal.Retries(max_retries=1, initial_delay=60.0),
    volumes={"/workspace": vol},
)
def train_and_detect(seed: int) -> dict:
    """Train one exact 300k paper cell, then run ordered/shuffled detection."""

    import gc
    import os
    import time

    import numpy as np
    import torch

    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    os.chdir("/repo/purified")
    event_path, cell_root = _prepare_cell_filesystem(ARCH, seed)
    result_path = cell_root / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text())
        if payload.get("status") == "complete":
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
            agent="reviewer-multiseed-overnight",
        )
        cached = False
        vol.commit()

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
    vol.commit()
    return payload


@app.function(
    image=image,
    cpu=1,
    memory=2_048,
    timeout=17 * 60 * 60,
    volumes={"/workspace": vol},
)
def dispatch(seeds: tuple[int, ...]) -> dict:
    calls = [(seed, train_and_detect.spawn(seed)) for seed in seeds]
    results = []
    for seed, call in calls:
        try:
            results.append(call.get())
        except Exception as error:  # noqa: BLE001
            results.append(
                {
                    "status": "failed",
                    "seed": seed,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
    payload = {"status": "complete", "results": results}
    root = Path("/workspace/c7_paper")
    root.mkdir(parents=True, exist_ok=True)
    (root / "dispatch.json").write_text(json.dumps(payload, indent=2))
    vol.commit()
    return payload


@app.function(
    image=image,
    cpu=1,
    memory=1_024,
    timeout=300,
    volumes={"/workspace": vol},
)
def read_status() -> dict:
    root = Path("/workspace/c7_paper")
    cells = {}
    for path in sorted((root / "cells").glob("*/result.json")):
        cells[path.parent.name] = json.loads(path.read_text())
    dispatch_path = root / "dispatch.json"
    return {
        "cells": cells,
        "dispatch": (
            json.loads(dispatch_path.read_text()) if dispatch_path.exists() else None
        ),
    }


@app.local_entrypoint()
def main(stage_only: bool = False, skip_stage: bool = False, status: bool = False):
    if status:
        print(json.dumps(read_status.remote(), indent=2))
        return
    if not skip_stage:
        print(json.dumps(stage_assets.remote(), indent=2), flush=True)
    if stage_only:
        return
    call = dispatch.spawn(SEEDS)
    print(
        json.dumps(
            {
                "status": "launched",
                "seeds": list(SEEDS),
                "function_call_id": call.object_id,
                "dashboard_url": call.get_dashboard_url(),
            },
            indent=2,
        ),
        flush=True,
    )
