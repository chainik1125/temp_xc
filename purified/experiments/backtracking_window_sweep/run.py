"""Run the July 23 C7 backtracking ``T=1..6`` sweep.

The process is deliberately one-GPU. Use ``--num-shards 2`` with distinct
``--shard-index`` values to split whole seeds across two GPUs without two
processes ever writing the same checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import replace
from pathlib import Path

from .evaluate import evaluate_cell
from .protocol import (
    PROTOCOL_VERSION,
    artifact_inventory,
    assert_inventory,
    atomic_json,
    cell_name,
    csv_ints,
    profile,
    profile_dict,
    seed_queue,
    sha256,
    validate_axes,
    window_queue,
)
from .report import write_report
from .train import TrainCellConfig, train_dictionary


def _repo_root() -> Path:
    configured = os.environ.get("TXC_RUNPOD_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[4]


def _parser() -> argparse.ArgumentParser:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--phase", choices=("train", "eval", "all"), default="all")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=root / "purified/artifacts/c7/sentence_acts_L10.npz",
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
        default=None,
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
    )
    parser.add_argument("--windows", help="comma-separated subset of 1,2,3,4,5,6")
    parser.add_argument("--seeds", help="comma-separated seed subset")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--d-sae", type=int)
    parser.add_argument("--k-pos", type=int)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _configured_profile(args: argparse.Namespace):
    base = profile(args.mode)
    windows = csv_ints(args.windows, base.windows)
    seeds = csv_ints(args.seeds, base.seeds)
    validate_axes(windows, seeds)
    updates = {"windows": windows, "seeds": seeds}
    for arg_name, field_name in (
        ("steps", "steps"),
        ("batch_size", "batch_size"),
        ("d_sae", "d_sae"),
        ("k_pos", "k_pos"),
        ("folds", "folds"),
        ("max_rows", "max_rows"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            updates[field_name] = value
    configured = replace(base, **updates)
    positive_fields = {
        "steps": configured.steps,
        "batch_size": configured.batch_size,
        "d_sae": configured.d_sae,
        "k_pos": configured.k_pos,
        "folds": configured.folds,
        "checkpoint_every": configured.checkpoint_every,
    }
    invalid = {key: value for key, value in positive_fields.items() if value < 1}
    if invalid:
        raise ValueError(f"positive configuration fields required: {invalid}")
    return configured


def _seed_shard(
    seeds: tuple[int, ...], *, num_shards: int, shard_index: int
) -> tuple[int, ...]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    ordered = seed_queue(seeds)
    return tuple(
        seed
        for index, seed in enumerate(ordered)
        if index % num_shards == shard_index
    )


def _estimate_storage_gb(window: int, d_in: int, d_sae: int, amp: bool) -> dict:
    bytes_per_parameter = 2 if amp else 4
    txc_parameters = 2 * window * d_in * d_sae
    sae_parameters = 2 * d_in * d_sae
    model_gb = bytes_per_parameter * (txc_parameters + sae_parameters) / 1e9
    # Model plus two Adam moments; exact optimizer serialization can vary.
    return {
        "model_files_gb": round(model_gb, 2),
        "model_plus_optimizer_estimate_gb": round(3 * model_gb, 2),
    }


def _plan(
    *,
    args: argparse.Namespace,
    configured,
    inventory: dict,
    shard_seeds: tuple[int, ...],
) -> dict:
    cells = [
        {
            "cell": cell_name(window, seed),
            "window": window,
            "seed": seed,
            "offsets": list(range(-8 - window + 1, -7)),
            "storage": _estimate_storage_gb(
                window, 4_096, configured.d_sae, configured.amp
            ),
        }
        for seed in shard_seeds
        for window in window_queue(configured.windows)
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": args.mode,
        "phase": args.phase,
        "device": args.device,
        "shard": [args.shard_index, args.num_shards],
        "profile": profile_dict(configured),
        "inventory": inventory,
        "output_root": str(args.output_root),
        "checkpoint_root": str(args.checkpoint_root),
        "cells_in_dispatch_order": cells,
    }


def _train_config(configured, *, arch: str, window: int, seed: int):
    return TrainCellConfig(
        arch=arch,
        window=window,
        seed=seed,
        d_in=4_096,
        d_sae=configured.d_sae,
        k_pos=configured.k_pos,
        batch_size=configured.batch_size,
        steps=configured.steps,
        learning_rate=configured.learning_rate,
        warmup_steps=configured.warmup_steps,
        checkpoint_every=min(configured.checkpoint_every, configured.steps),
        schedule_seed=907_000 + 100 * seed,
        amp=configured.amp,
    )


def main() -> None:
    args = _parser().parse_args()
    root = _repo_root()
    if args.output_root is None:
        args.output_root = (
            root
            / "purified/results/neurips_rebuttal/backtracking_window_sweep"
            / args.mode
        )
    if args.checkpoint_root is None:
        args.checkpoint_root = (
            root / "checkpoints/backtracking_window_sweep" / args.mode
        )
    configured = _configured_profile(args)
    shard_seeds = _seed_shard(
        configured.seeds,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    inventory = artifact_inventory(
        args.artifact,
        args.activation_cache,
        strict_full=args.mode == "full",
    )
    plan = _plan(
        args=args,
        configured=configured,
        inventory=inventory,
        shard_seeds=shard_seeds,
    )
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    assert_inventory(inventory, strict_full=args.mode == "full")
    artifact_digest = inventory.get("artifact_sha256") or sha256(args.artifact)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    atomic_json(plan, args.output_root / f"plan_shard{args.shard_index}.json")

    for seed in shard_seeds:
        for window in window_queue(configured.windows):
            name = cell_name(window, seed)
            cell_output = args.output_root / "cells" / name
            checkpoint_cell = args.checkpoint_root / name
            result_path = cell_output / "result.json"
            if result_path.exists():
                print(f"[sweep] complete, skipping {name}", flush=True)
                continue
            try:
                if args.phase in {"train", "all"}:
                    for arch in ("txc", "sae"):
                        train_dictionary(
                            activation_cache=args.activation_cache,
                            checkpoint_dir=checkpoint_cell / arch,
                            config=_train_config(
                                configured,
                                arch=arch,
                                window=window,
                                seed=seed,
                            ),
                            device=args.device,
                        )
                if args.phase in {"eval", "all"}:
                    result = evaluate_cell(
                        artifact=args.artifact,
                        artifact_sha256=artifact_digest,
                        txc_checkpoint=checkpoint_cell / "txc",
                        sae_checkpoint=checkpoint_cell / "sae",
                        output_dir=cell_output,
                        window=window,
                        seed=seed,
                        folds=configured.folds,
                        s_grid=configured.s_grid,
                        max_rows=configured.max_rows,
                        batch_size=args.encode_batch_size,
                        pca_dim=16 if args.mode == "smoke" else 32,
                        device=args.device,
                        bootstrap_repeats=configured.bootstrap_repeats,
                    )
                    print(
                        json.dumps(
                            {
                                "cell": name,
                                "status": result["status"],
                                "n_rows": result["n_rows"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    write_report(args.output_root)
            except Exception as error:
                failure = {
                    "cell": name,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                atomic_json(failure, cell_output / "failure.json")
                print(json.dumps(failure, sort_keys=True), flush=True)
                if not args.continue_on_error:
                    raise
    if args.phase in {"eval", "all"}:
        summary = write_report(args.output_root)
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
