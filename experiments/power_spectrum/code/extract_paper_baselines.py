"""Extract the three published synthetic baselines from the pinned Figure 2 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


POWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = POWER_ROOT / "configs" / "paper_synthetic_v1.json"
DEFAULT_OUTPUT = POWER_ROOT / "results" / "paper_synthetic_baselines.json"
ARCHITECTURES = ("topk_sae", "tsae_paper", "txc_base")


def _git_show(repo: Path, ref: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo,
    )


def _best_cell(
    rows: list[dict[str, Any]],
    *,
    arch_key: str,
    metric_getter,
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["t_label"]), int(row["k_pos"]))].append(row)
    eligible = [
        (key, values)
        for key, values in grouped.items()
        if len(values) >= 2
    ]
    if not eligible:
        raise RuntimeError(f"{arch_key}: no eligible cells")
    (t_label, k_pos), selected = max(
        eligible,
        key=lambda item: statistics.fmean(metric_getter(row) for row in item[1]),
    )
    values = [float(metric_getter(row)) for row in selected]
    seeds = [int(row["seed"]) for row in selected]
    return {
        "architecture": arch_key,
        "t_label": t_label,
        "k_pos": k_pos,
        "n": len(values),
        "seed_values": values,
        "seeds": seeds,
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def extract(config_path: Path, repo: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    snapshot = config["baseline_snapshot"]
    ref = str(snapshot["git_ref"])

    denoising_bytes = _git_show(repo, ref, snapshot["denoising_path"])
    denoising_rows = json.loads(denoising_bytes)
    denoising: dict[str, Any] = {}
    for arch in ARCHITECTURES:
        rows = [
            {
                **row,
                "t_label": row.get("t_label", "default"),
            }
            for row in denoising_rows
            if row["arch_name"] == arch
        ]
        denoising[arch] = _best_cell(
            rows,
            arch_key=arch,
            metric_getter=lambda row: float(row["lp_mean_global_r2"]),
        )

    coupling_bytes = _git_show(repo, ref, snapshot["coupling_path"])
    coupling_rows = []
    for line in coupling_bytes.decode().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("eval_cfg", {}).get("smoke"):
            continue
        if row["datasource"] != (
            "toy_coupled_noisy_K10_M20_d256_pB05_np10"
        ):
            continue
        coupling_rows.append(row)

    coupling: dict[str, Any] = {}
    for arch in ARCHITECTURES:
        rows = [
            {
                **row,
                "t_label": row["eval_cfg"].get("t_label", "default"),
                "k_pos": int(row["eval_cfg"]["k_pos"]),
            }
            for row in coupling_rows
            if row["arch"] == arch
        ]
        coupling[arch] = _best_cell(
            rows,
            arch_key=arch,
            metric_getter=lambda row: float(row["metrics"]["gauc"]),
        )

    renderer_bytes = _git_show(repo, ref, snapshot["renderer_path"])
    return {
        "schema_version": 1,
        "source": {
            "git_ref": ref,
            "denoising_path": snapshot["denoising_path"],
            "denoising_sha256": hashlib.sha256(denoising_bytes).hexdigest(),
            "coupling_path": snapshot["coupling_path"],
            "coupling_sha256": hashlib.sha256(coupling_bytes).hexdigest(),
            "renderer_path": snapshot["renderer_path"],
            "renderer_sha256": hashlib.sha256(renderer_bytes).hexdigest(),
        },
        "selection_rule": (
            "Published renderer: best seed-mean (t_label, k_pos) cell; "
            "cells require at least two rows and retain historical duplicate rows."
        ),
        "tasks": {
            "denoising": {
                "metric": "lp_mean_global_r2",
                "architectures": denoising,
            },
            "coupling": {
                "metric": "gauc",
                "architectures": coupling,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repo", type=Path, default=POWER_ROOT.parents[1])
    args = parser.parse_args()
    payload = extract(args.config, args.repo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
