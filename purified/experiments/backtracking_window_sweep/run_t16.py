"""Run the provenance-locked common-cohort C7 ``T<=16`` extension."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from .evaluate import evaluate_cell
from .protocol import (
    atomic_json,
    cell_name,
    csv_ints,
    profile_dict,
    seed_queue,
)
from .protocol_t16 import (
    ARTIFACT_OFFSETS,
    DEFAULT_ARTIFACT_NAME,
    DEFAULT_MANIFEST_NAME,
    PROTOCOL_VERSION,
    artifact_inventory,
    assert_inventory,
    profile,
    validate_axes,
    window_queue,
)
from .report import write_report
from .train import TrainCellConfig, run_memory_smoke, train_dictionary


def _repo_root() -> Path:
    configured = os.environ.get("TXC_RUNPOD_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[4]


def _parser() -> argparse.ArgumentParser:
    root = _repo_root()
    c7 = root / "purified/artifacts/c7"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "memory-smoke", "full"),
        default="full",
    )
    parser.add_argument(
        "--phase", choices=("train", "eval", "all"), default="all"
    )
    parser.add_argument(
        "--artifact", type=Path, default=c7 / DEFAULT_ARTIFACT_NAME
    )
    parser.add_argument(
        "--artifact-manifest",
        type=Path,
        default=c7 / DEFAULT_MANIFEST_NAME,
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
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument(
        "--windows", help="comma-separated subset of 1,2,4,6,8,10,12,14,16"
    )
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
    positive = {
        key: getattr(configured, key)
        for key in (
            "steps",
            "batch_size",
            "d_sae",
            "k_pos",
            "folds",
            "checkpoint_every",
        )
    }
    invalid = {key: value for key, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(f"positive configuration fields required: {invalid}")
    if args.mode == "memory-smoke" and (
        configured.windows != (16,) or configured.steps != 1
    ):
        raise ValueError("memory-smoke requires windows=(16,) and steps=1")
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
        checkpoint_every=min(
            configured.checkpoint_every, configured.steps
        ),
        schedule_seed=907_000 + 100 * seed,
        amp=configured.amp,
        schedule_max_window=16,
        record_effective_l0=True,
    )


def _memory_inventory(path: Path) -> dict:
    result = {"activation_cache": str(path), "missing": []}
    if not path.exists():
        result["missing"].append(str(path))
        return result
    cache = np.load(path, mmap_mode="r")
    result.update(
        {
            "activation_cache_shape": [int(value) for value in cache.shape],
            "activation_cache_dtype": str(cache.dtype),
            "activation_cache_shape_ok": (
                cache.ndim == 3
                and cache.shape[1] >= 16
                and cache.shape[-1] == 4_096
            ),
        }
    )
    return result


def _memory_smoke(
    *,
    args: argparse.Namespace,
    configured,
    output_root: Path,
) -> dict:
    inventory = _memory_inventory(args.activation_cache)
    plan = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": args.mode,
        "device": args.device,
        "profile": profile_dict(configured),
        "inventory": inventory,
        "checkpoint_policy": "none; one train step is never resumable progress",
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return plan
    if inventory["missing"] or not inventory.get("activation_cache_shape_ok"):
        raise ValueError(f"invalid memory-smoke cache: {inventory}")
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}
    for arch in ("txc", "sae"):
        results[arch] = run_memory_smoke(
            activation_cache=args.activation_cache,
            config=_train_config(
                configured, arch=arch, window=16, seed=42
            ),
            device=args.device,
        )
    payload = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "mode": "memory-smoke",
        "profile": profile_dict(configured),
        "results": results,
    }
    atomic_json(payload, output_root / "memory_smoke.json")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    args = _parser().parse_args()
    root = _repo_root()
    namespace = "backtracking_window_sweep_t16"
    if args.output_root is None:
        args.output_root = (
            root / "purified/results/neurips_rebuttal" / namespace / args.mode
        )
    if args.checkpoint_root is None:
        args.checkpoint_root = root / "checkpoints" / namespace / args.mode
    configured = _configured_profile(args)
    if args.mode == "memory-smoke":
        _memory_smoke(
            args=args, configured=configured, output_root=args.output_root
        )
        return

    shard_seeds = _seed_shard(
        configured.seeds,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    inventory = artifact_inventory(
        args.artifact,
        args.artifact_manifest,
        args.reference_artifact,
        args.activation_cache,
        strict_full=args.mode == "full",
    )
    cells = [
        {
            "cell": cell_name(window, seed),
            "window": window,
            "seed": seed,
            "offsets": list(ARTIFACT_OFFSETS[-window:]),
        }
        for seed in shard_seeds
        for window in window_queue(configured.windows)
    ]
    plan = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": args.mode,
        "phase": args.phase,
        "device": args.device,
        "shard": [args.shard_index, args.num_shards],
        "profile": profile_dict(configured),
        "inventory": inventory,
        "common_cohort_contract": (
            "every T is a trailing view of the same T16-valid artifact rows"
        ),
        "output_root": str(args.output_root),
        "checkpoint_root": str(args.checkpoint_root),
        "cells_in_dispatch_order": cells,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return
    assert_inventory(inventory, strict_full=args.mode == "full")
    artifact_digest = inventory["artifact_sha256"]
    cohort_digest = inventory["common_cohort_sha256"]
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
                completed = json.loads(result_path.read_text())
                checks = {
                    "status": completed.get("status") == "complete",
                    "protocol": (
                        completed.get("protocol_version")
                        == PROTOCOL_VERSION
                    ),
                    "artifact": (
                        completed.get("artifact_sha256")
                        == artifact_digest
                    ),
                    "cohort": (
                        completed.get("cohort_sha256") == cohort_digest
                    ),
                    "window": completed.get("window") == window,
                    "seed": completed.get("seed") == seed,
                }
                if not all(checks.values()):
                    raise ValueError(
                        f"stale completed cell at {result_path}: {checks}"
                    )
                print(f"[t16 sweep] complete, skipping {name}", flush=True)
                continue
            try:
                training = {}
                if args.phase in {"train", "all"}:
                    for arch in ("txc", "sae"):
                        training[arch] = train_dictionary(
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
                    atomic_json(
                        {
                            "status": "complete",
                            "protocol_version": PROTOCOL_VERSION,
                            "effective_l0_definition": (
                                "actual mean nonzeros after TopK then ReLU "
                                "on the final training batch"
                            ),
                            "architectures": training,
                        },
                        checkpoint_cell / "training_summary.json",
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
                        protocol_version=PROTOCOL_VERSION,
                        artifact_offsets=ARTIFACT_OFFSETS,
                        include_effective_l0=True,
                        cohort_sha256=cohort_digest,
                    )
                    print(
                        json.dumps(
                            {
                                "cell": name,
                                "status": result["status"],
                                "n_rows": result["n_rows"],
                                "cohort_sha256": cohort_digest,
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
        print(json.dumps(write_report(args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
