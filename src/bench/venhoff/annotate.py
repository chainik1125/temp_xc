"""Annotation: assign each sentence to its argmax cluster.

Given a trained small-k dictionary and the collected activations, pass
each sentence through the encoder (via the SAE shim) and take the
argmax over latents as the cluster label. This is Venhoff's
`annotate_thinking.py` behavior, but with two generalizations:

  - Path 1 (SAE/MLC): input is `(N, d_model)`; pass straight through.
  - Path 3 (TempXC): input is `(N, T, d_model)`; encoder applies the
    chosen aggregation (`last` / `mean` / `max` / `full_window`) before
    argmax. The aggregation name becomes part of the artifact filename
    so the four ablation runs don't collide.

Output: a JSON file mapping each sentence index to its cluster id.
Resume: skip if assignments JSON + meta hash already exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.bench.saebench.aggregation import AggregationName
from src.bench.venhoff.paths import ArtifactPaths, RunIdentity, can_resume, write_with_metadata
from src.bench.venhoff.sae_shim import wrap_for_path1, wrap_for_path3
from src.bench.venhoff.train_small_sae import load_ckpt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("venhoff.annotate")

PathName = Literal["path1", "path3"]
ANNOTATE_BATCH_SIZE = 512


def _load_acts(paths: ArtifactPaths, path_name: PathName) -> tuple[torch.Tensor, list[str]]:
    with paths.activations_pkl(path_name).open("rb") as f:
        x, texts = pickle.load(f)
    return torch.from_numpy(np.asarray(x, dtype=np.float32)), list(texts)


def annotate(
    paths: ArtifactPaths,
    arch: str,
    cluster_size: int,
    path_name: PathName,
    aggregation: AggregationName | str,
    device: str = "cuda",
    force: bool = False,
) -> Path:
    """Run annotation for one (arch, cluster_size, path, aggregation) cell."""
    paths.ensure_dirs()
    out = paths.assignments_json(arch, cluster_size, path_name, str(aggregation))

    meta = {
        "stage": "annotate",
        "arch": arch,
        "cluster_size": cluster_size,
        "path": path_name,
        "aggregation": str(aggregation),
        "layer": paths.identity.layer,
        "seed": paths.identity.seed,
        "n_traces": paths.identity.n_traces,
    }
    if not force and can_resume(out, meta):
        log.info("resume: assignments exist at %s", out)
        return out

    ckpt_path = paths.ckpt(arch, cluster_size, path_name)
    model, ckpt_cfg = load_ckpt(ckpt_path, device=device)

    mean_pkl = paths.activation_mean_pkl(path_name)
    if path_name == "path1":
        shim = wrap_for_path1(model, mean_pkl).to(device)
    else:
        T = int(ckpt_cfg.get("T", 5))
        shim = wrap_for_path3(model, mean_pkl, T=T, aggregation=str(aggregation)).to(device)

    x, texts = _load_acts(paths, path_name)
    x = x.to(device)

    assignments: list[int] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], ANNOTATE_BATCH_SIZE):
            batch = x[start : start + ANNOTATE_BATCH_SIZE]
            z = shim.encoder(batch)
            labels = z.argmax(dim=-1).detach().cpu().tolist()
            assignments.extend(labels)

    assert len(assignments) == len(texts), (
        f"assignments ({len(assignments)}) != texts ({len(texts)})"
    )

    # Group sentences by cluster id for downstream label/score stages.
    clusters: dict[int, list[int]] = {}
    for i, cid in enumerate(assignments):
        clusters.setdefault(int(cid), []).append(i)

    payload = {
        "assignments": assignments,
        "clusters": {str(cid): idxs for cid, idxs in sorted(clusters.items())},
        "n_sentences": len(texts),
        "n_clusters_nonempty": len(clusters),
        "cluster_size_configured": cluster_size,
    }
    write_with_metadata(out, json.dumps(payload, indent=2), meta)
    log.info("annotate → %s (%d non-empty / %d configured)", out, len(clusters), cluster_size)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("results/venhoff_eval"))
    p.add_argument("--model", default="deepseek-r1-distill-llama-8b")
    p.add_argument("--dataset", default="mmlu-pro")
    p.add_argument("--split", default="test")
    p.add_argument("--n-traces", type=int, default=1000)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arch", required=True)
    p.add_argument("--cluster-size", type=int, required=True)
    p.add_argument("--path", required=True, choices=["path1", "path3"])
    p.add_argument("--aggregation", default="full_window",
                   choices=["last", "mean", "max", "full_window"])
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
    annotate(
        paths=paths, arch=args.arch, cluster_size=args.cluster_size,
        path_name=args.path, aggregation=args.aggregation,
        device=args.device, force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
