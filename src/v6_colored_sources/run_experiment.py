"""Stage 1 / Stage 2 sweep entrypoints for colored-source experiments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .colored_sources import generate_dataset
from .configs import ColoredSourceConfig, eigengap
from .data_adapter import ColoredSourceCache
from .train_runner import (
    TrainConfig,
    oracle_baseline,
    random_dictionary_baseline,
    train_pair,
)


def _device(cli_device: str | None) -> torch.device:
    if cli_device:
        return torch.device(cli_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _result_dict(cell_results: dict, oracle: dict, random_baseline: dict, *, D: int, W: int) -> dict:
    out: dict = {"D": D, "W": W, "oracle": oracle, "random": random_baseline}
    for arch, res in cell_results.items():
        out[arch] = {
            "final_loss": res.final_loss,
            "final_l0": res.final_l0,
            "recovery_squared": res.recovery_squared,
            "s_adj": res.s_adj,
            "recovery_auc": res.recovery_auc,
            "history": res.history,
        }
    return out


def run_stage1(
    *,
    n_seq: int,
    T_chain: int,
    n_steps: int,
    batch_size: int,
    W_grid: list[int],
    k: int,
    device: torch.device,
    out_dir: Path,
    seed: int = 0,
) -> dict:
    """W sweep at D=1, fixed (rho_min, rho_max)=(0.1, 0.9), sigma=0.1, N=d=128."""
    cfg = ColoredSourceConfig(
        N=128, d=128, D=1, sigma=0.1, rho_min=0.1, rho_max=0.9,
        n_seq=n_seq, T_chain=T_chain, seed=seed,
    )
    data = generate_dataset(cfg)
    F = data["features"]
    cache = ColoredSourceCache(data["x"], device)

    train_cfg = TrainConfig(n_steps=n_steps, batch_size=batch_size)

    oracle = oracle_baseline(data["x"], F, D=cfg.D, H=cfg.N)
    random_b = random_dictionary_baseline(F, H=cfg.N, n_trials=10, seed=seed + 1)
    print(f"Oracle ceiling: rec_sq={oracle['recovery_squared']:.3f} S_adj={oracle['s_adj']:.3f}")
    print(f"Random floor:   rec_sq={random_b['recovery_squared']:.3f} S_adj={random_b['s_adj']:.3f}")
    print(f"Eigengap:       gamma={eigengap(data['rho']):.4f}")

    cells = []
    t0 = time.time()
    for W in W_grid:
        print(f"\n=== W={W} ===")
        cell = train_pair(
            cache=cache, F=F, W=W, k=k, H=cfg.N, d=cfg.d,
            device=device, train_cfg=train_cfg,
        )
        for arch, res in cell.items():
            print(
                f"  {arch:12s} loss={res.final_loss:.4f} L0={res.final_l0:.2f} "
                f"rec_sq={res.recovery_squared:.3f} S_adj={res.s_adj:.3f} "
                f"AUC={res.recovery_auc:.3f}"
            )
        cells.append(_result_dict(cell, oracle, random_b, D=cfg.D, W=W))

    elapsed = time.time() - t0
    print(f"\nStage 1 sweep took {elapsed/60:.1f} min")

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": 1,
        "config": asdict(cfg),
        "train_config": asdict(train_cfg),
        "k": k,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "cells": cells,
    }
    out_path = out_dir / "stage1.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_path}")
    return payload


def run_stage2(
    *,
    n_seq: int,
    T_chain: int,
    n_steps: int,
    batch_size: int,
    D_grid: list[int],
    W_grid: list[int],
    k: int,
    device: torch.device,
    out_dir: Path,
    seed: int = 0,
) -> dict:
    """D x W grid for the headline phase-transition figure."""
    train_cfg = TrainConfig(n_steps=n_steps, batch_size=batch_size)

    all_cells = []
    t0 = time.time()
    for D in D_grid:
        cfg = ColoredSourceConfig(
            N=128, d=128, D=D, sigma=0.1, rho_min=0.1, rho_max=0.9,
            n_seq=n_seq, T_chain=T_chain, seed=seed,
        )
        data = generate_dataset(cfg)
        F = data["features"]
        cache = ColoredSourceCache(data["x"], device)

        oracle = oracle_baseline(data["x"], F, D=cfg.D, H=cfg.N)
        random_b = random_dictionary_baseline(F, H=cfg.N, n_trials=10, seed=seed + 1)
        print(f"\n### D={D} ###")
        print(f"Oracle ceiling: rec_sq={oracle['recovery_squared']:.3f} S_adj={oracle['s_adj']:.3f}")

        for W in W_grid:
            if W > T_chain - D:
                print(f"  Skipping W={W} (> T_chain - D)")
                continue
            print(f"\n--- D={D} W={W} ---")
            cell = train_pair(
                cache=cache, F=F, W=W, k=k, H=cfg.N, d=cfg.d,
                device=device, train_cfg=train_cfg,
            )
            for arch, res in cell.items():
                print(
                    f"  {arch:12s} loss={res.final_loss:.4f} L0={res.final_l0:.2f} "
                    f"rec_sq={res.recovery_squared:.3f} S_adj={res.s_adj:.3f}"
                )
            all_cells.append(_result_dict(cell, oracle, random_b, D=cfg.D, W=W))

    elapsed = time.time() - t0
    print(f"\nStage 2 sweep took {elapsed/60:.1f} min")

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": 2,
        "D_grid": D_grid,
        "W_grid": W_grid,
        "n_seq": n_seq,
        "T_chain": T_chain,
        "train_config": asdict(train_cfg),
        "k": k,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "cells": all_cells,
    }
    out_path = out_dir / "stage2.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run colored-source sweeps.")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="results/v6_colored_sources")
    parser.add_argument("--n_seq", type=int, default=512)
    parser.add_argument("--T_chain", type=int, default=2048)
    parser.add_argument("--n_steps", type=int, default=30_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--W_grid", type=int, nargs="+", default=[1, 2, 4, 8, 16],
    )
    parser.add_argument(
        "--D_grid", type=int, nargs="+", default=[1, 2, 4, 8],
    )
    args = parser.parse_args()

    device = _device(args.device)
    out_dir = Path(args.out_dir)
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    if args.stage == 1:
        run_stage1(
            n_seq=args.n_seq, T_chain=args.T_chain, n_steps=args.n_steps,
            batch_size=args.batch_size, W_grid=args.W_grid, k=args.k,
            device=device, out_dir=out_dir, seed=args.seed,
        )
    else:
        run_stage2(
            n_seq=args.n_seq, T_chain=args.T_chain, n_steps=args.n_steps,
            batch_size=args.batch_size, D_grid=args.D_grid, W_grid=args.W_grid,
            k=args.k, device=device, out_dir=out_dir, seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
