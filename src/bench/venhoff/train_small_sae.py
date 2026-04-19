"""Small-k dictionary training for Venhoff-style taxonomy discovery.

Trains a tiny SAE / MLC (or re-uses a pre-existing wide TempXC for
Path 3) where `d_sae = cluster_size` and `k = 3`. At this scale the
"SAE" is really a discoverable k-means alternative; its argmax over
latents is treated as the cluster label for each input.

- Path 1 input: `(N, d_model)` per-sentence-mean activations.
- Path 3 input: `(N, T, d_model)` per-sentence T-window activations
  (TempXC trains per-token inside the window; cluster label uses the
  Path 3 shim aggregation).

Training budget is capped at **10 000 steps** with plateau-based
early stop (Q4-lock-in `integration_plan § 5`). Uses existing
`TopKSAESpec.train` / `TemporalCrosscoderSpec.train` where possible so
we get the same infra the SAEBench run uses (plateau, W&B logging,
gradient clipping).

Resume: if a checkpoint already exists at
`paths.ckpt(arch, cluster_size, path)` and its sidecar metadata hash
matches the current config, skip. `--force` rebuilds.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch

from src.bench.venhoff.paths import ArtifactPaths, RunIdentity, can_resume, write_with_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("venhoff.train_small_sae")

ArchName = Literal["sae", "tempxc", "mlc"]
PathName = Literal["path1", "path3"]

MAX_TRAIN_STEPS = 10_000
PLATEAU_PCT = 0.005       # 0.5% drop over window — locked per integration_plan § 5
PLATEAU_MIN_STEPS = 1_000
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
TOPK_K = 3                # Venhoff's default


@dataclass
class TrainConfig:
    arch: ArchName
    cluster_size: int
    path: PathName
    T: int = 5
    seed: int = 42
    total_steps: int = MAX_TRAIN_STEPS
    plateau_pct: float = PLATEAU_PCT
    plateau_min_steps: int = PLATEAU_MIN_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    k: int = TOPK_K


def _load_path1(paths: ArtifactPaths) -> tuple[torch.Tensor, list[str]]:
    with paths.activations_pkl("path1").open("rb") as f:
        acts, texts = pickle.load(f)
    return torch.from_numpy(np.asarray(acts, dtype=np.float32)), list(texts)


def _load_path3(paths: ArtifactPaths) -> tuple[torch.Tensor, list[str]]:
    with paths.activations_pkl("path3").open("rb") as f:
        windows, texts = pickle.load(f)
    return torch.from_numpy(np.asarray(windows, dtype=np.float32)), list(texts)


def _make_gen_fn(x: torch.Tensor, batch_size: int, device: str, rng: np.random.Generator) -> Callable:
    """Return a callable that draws random minibatches from `x`.

    `x` is either `(N, d)` (Path 1) or `(N, T, d)` (Path 3).
    The sampling axis is always 0.
    """
    x_dev = x.to(device)
    N = x_dev.shape[0]

    def gen(bs: int) -> torch.Tensor:
        idx = rng.integers(0, N, size=bs)
        return x_dev[idx]

    return gen


def _train_sae(arch: ArchName, d_in: int, cluster_size: int, cfg: TrainConfig,
               gen_fn: Callable, device: str) -> tuple[torch.nn.Module, dict]:
    """Dispatch to the right ArchSpec and train under the 10k-step cap."""
    if arch == "sae":
        from src.bench.architectures.topk_sae import TopKSAESpec

        spec = TopKSAESpec()
        model = spec.create(d_in=d_in, d_sae=cluster_size, k=cfg.k, device=device)
        log_ = spec.train(
            model=model, gen_fn=gen_fn,
            total_steps=cfg.total_steps, batch_size=cfg.batch_size, lr=cfg.lr,
            device=device, plateau_pct=cfg.plateau_pct,
            plateau_min_steps=cfg.plateau_min_steps,
        )
        return model, log_

    if arch == "mlc":
        # MLC requires simultaneous (B, n_layers, d) activations from a
        # window of layers around the anchor. Our current
        # activation_collection.py collects only `paths.identity.layer`
        # (single-layer), so MLC is out of scope for this run.
        # Enabling it requires extending activation_collection.py with
        # a multi-layer hook; tracked in plan.md § 3.
        raise NotImplementedError(
            "MLC requires multi-layer activation collection (n_layers=5 "
            "around the anchor). Not implemented in Phase 1a; use arch=sae "
            "or arch=tempxc."
        )

    if arch == "tempxc":
        from src.bench.architectures.crosscoder import CrosscoderSpec

        spec = CrosscoderSpec(T=cfg.T)
        # T is held on the spec, not passed to create().
        model = spec.create(d_in=d_in, d_sae=cluster_size, k=cfg.k, device=device)
        log_ = spec.train(
            model=model, gen_fn=gen_fn,
            total_steps=cfg.total_steps, batch_size=cfg.batch_size, lr=cfg.lr,
            device=device, plateau_pct=cfg.plateau_pct,
            plateau_min_steps=cfg.plateau_min_steps,
        )
        return model, log_

    raise ValueError(f"unknown arch {arch!r}")


def train(
    paths: ArtifactPaths,
    cfg: TrainConfig,
    device: str = "cuda",
    force: bool = False,
) -> Path:
    """Train a small-k dictionary; return the checkpoint path.

    Skips training if a valid cached checkpoint is found.
    """
    paths.ensure_dirs()
    ckpt_path = paths.ckpt(cfg.arch, cfg.cluster_size, cfg.path)

    meta = {
        "stage": "train_small_sae",
        "arch": cfg.arch,
        "cluster_size": cfg.cluster_size,
        "path": cfg.path,
        "T": cfg.T,
        "seed": cfg.seed,
        "total_steps": cfg.total_steps,
        "plateau_pct": cfg.plateau_pct,
        "plateau_min_steps": cfg.plateau_min_steps,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "k": cfg.k,
        "layer": paths.identity.layer,
        "n_traces": paths.identity.n_traces,
    }
    if not force and can_resume(ckpt_path, meta):
        log.info("resume: ckpt exists at %s", ckpt_path)
        return ckpt_path

    # Load activations for the selected path.
    if cfg.path == "path1":
        x, _texts = _load_path1(paths)
    else:
        x, _texts = _load_path3(paths)

    d_in = x.shape[-1]
    if cfg.path == "path3" and cfg.arch != "tempxc":
        raise ValueError(f"arch {cfg.arch!r} is path1-only; path3 is TempXC territory")

    rng = np.random.default_rng(cfg.seed)
    gen_fn = _make_gen_fn(x, cfg.batch_size, device, rng)

    log.info("training %s k=%d path=%s on %s — %d steps (plateau %g%%)",
             cfg.arch, cfg.cluster_size, cfg.path, tuple(x.shape),
             cfg.total_steps, cfg.plateau_pct * 100)

    torch.manual_seed(cfg.seed)
    model, train_log = _train_sae(cfg.arch, d_in, cfg.cluster_size, cfg, gen_fn, device)

    payload_bytes = _serialize_ckpt(model, train_log, cfg)
    write_with_metadata(ckpt_path, payload_bytes, meta)
    log.info("wrote ckpt: %s", ckpt_path)
    return ckpt_path


def _serialize_ckpt(model: torch.nn.Module, train_log: dict, cfg: TrainConfig) -> bytes:
    import io

    buf = io.BytesIO()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "arch": cfg.arch,
                "cluster_size": cfg.cluster_size,
                "path": cfg.path,
                "T": cfg.T,
                "seed": cfg.seed,
                "total_steps": cfg.total_steps,
                "k": cfg.k,
            },
            "train_log": train_log,
        },
        buf,
    )
    return buf.getvalue()


def load_ckpt(ckpt_path: Path, device: str = "cuda") -> tuple[torch.nn.Module, dict]:
    """Re-instantiate and load a trained small-k model from disk."""
    payload = torch.load(ckpt_path, map_location=device)
    cfg_dict = payload["config"]
    arch = cfg_dict["arch"]
    cluster_size = cfg_dict["cluster_size"]
    k = cfg_dict["k"]
    T = cfg_dict.get("T", 5)

    # Inspect state_dict to recover d_in (W_enc shape is (d_sae, d_in)).
    sd = payload["state_dict"]
    d_in = None
    for key, tensor in sd.items():
        if key.endswith("W_enc"):
            d_in = tensor.shape[1]
            break
    if d_in is None:
        raise ValueError(f"couldn't infer d_in from state_dict keys: {list(sd)}")

    if arch == "sae":
        from src.bench.architectures.topk_sae import TopKSAE
        model = TopKSAE(d_in=d_in, d_sae=cluster_size, k=k).to(device)
    elif arch == "mlc":
        raise NotImplementedError("MLC requires multi-layer activation collection; see _train_sae().")
    elif arch == "tempxc":
        from src.bench.architectures.crosscoder import CrosscoderSpec
        spec = CrosscoderSpec(T=T)
        model = spec.create(d_in=d_in, d_sae=cluster_size, k=k, device=device)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    model.load_state_dict(sd)
    model.eval()
    return model, cfg_dict


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("results/venhoff_eval"))
    p.add_argument("--model", default="deepseek-r1-distill-llama-8b")
    p.add_argument("--dataset", default="mmlu-pro")
    p.add_argument("--split", default="test")
    p.add_argument("--n-traces", type=int, default=1000)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arch", required=True, choices=["sae", "mlc", "tempxc"])
    p.add_argument("--cluster-size", type=int, required=True)
    p.add_argument("--path", required=True, choices=["path1", "path3"])
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--total-steps", type=int, default=MAX_TRAIN_STEPS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    paths = ArtifactPaths(
        root=args.root,
        identity=RunIdentity(
            model=args.model, dataset=args.dataset, dataset_split=args.split,
            n_traces=args.n_traces, layer=args.layer, seed=args.seed,
        ),
    )
    cfg = TrainConfig(
        arch=args.arch, cluster_size=args.cluster_size, path=args.path, T=args.T,
        seed=args.seed, total_steps=args.total_steps, batch_size=args.batch_size, lr=args.lr,
    )
    train(paths, cfg, device=args.device, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
