"""Run fresh SAE and TXC-base C7 steering baselines.

This is deliberately a thin adapter around the frozen C7 implementation on
``origin/temp-bench``. In particular, generation, prompt construction,
cut-and-continue, judging, and delta-gc computation are delegated to
``temp_bench.case_studies.backtracking.run_arch_evaluation``.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


SEED = 42
D_IN = 4096
EXPECTED_WRONG = 31
EXPECTED_CORRECT = 30
ARM_TO_TRAIN_KEY = {
    "pooled_sae_max": "f437e623fabc37ec",
    "pooled_sae_mean": "f437e623fabc37ec",
    "topk_sae": "f437e623fabc37ec",
    "txc_base": "08fe3af07682fab4",
}
ARM_TO_CHECKPOINT_ARCH = {
    "pooled_sae_max": "topk_sae",
    "pooled_sae_mean": "topk_sae",
    "topk_sae": "topk_sae",
    "txc_base": "txc_base",
}
EXPECTED_CANONICAL_REVISION = "1c213513fe0c89220e8a00e53a9b258081ffe749"


class PooledSAEAdapter(torch.nn.Module):
    """Expose one shared SAE code pooled over a fixed activation window.

    The wrapped SAE encodes every position with the same dictionary, so a
    feature ID has one decoder direction at all positions.  Pooling produces
    a single window-level code which the canonical C7 feature miner can rank
    exactly like a TXC code.  No parameters are trained or changed here.
    """

    def __init__(self, base: torch.nn.Module, *, pool: str, window: int = 5):
        super().__init__()
        if pool not in {"mean", "max"}:
            raise ValueError(f"unknown pool: {pool}")
        self.base = base
        self.pool = pool
        self.config = replace(base.config, name=f"pooled_sae_{pool}", T=window)

    @property
    def d_sae(self) -> int:
        return int(self.config.d_sae)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.base.encode(x)
        if z.dim() != 3:
            raise ValueError(f"expected windowed SAE code, got {tuple(z.shape)}")
        if self.pool == "mean":
            return z.mean(dim=1, keepdim=True)
        return z.amax(dim=1, keepdim=True)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.base.decode(z)

    def decoder_directions(self) -> torch.Tensor:
        return self.base.decoder_directions()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def successful_judge_keys(
    rows: Iterable[dict[str, Any]], *, arch: str, seed: int
) -> set[tuple[str, float, str, int]]:
    """Return unique successful judge keys, tolerating failed retry rows."""
    return {
        (
            str(row.get("transcript_id", "")),
            float(row.get("magnitude", float("nan"))),
            str(row.get("arch", "")),
            int(row.get("seed", -1)),
        )
        for row in rows
        if row.get("arch") == arch
        and int(row.get("seed", -1)) == seed
        and int(row.get("label", -1)) >= 0
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def verify_zero_hook_noop() -> None:
    """Exercise the canonical hook's exact zero-magnitude short circuit."""
    from temp_bench.case_studies.backtracking import SteeringHook

    torch.manual_seed(0)
    x = torch.randn(3, 4, 7)
    hook = SteeringHook(torch.randn(7))
    hook.magnitudes = torch.zeros(3)
    direct = hook(None, None, x)
    tupled = hook(None, None, (x, "cache"))
    if direct is not x or tupled[0] is not x or tupled[1] != "cache":
        raise RuntimeError("canonical SteeringHook changed a zero-magnitude output")


async def verify_judge_liveness(workspace: Path) -> dict[str, Any]:
    """Make one isolated API call and fail closed on API or parse errors."""
    from temp_bench.case_studies.backtracking import SonnetBacktrackingJudge

    judge = SonnetBacktrackingJudge(workspace=workspace, max_concurrency=1)
    outputs = await judge.judge_many(
        [
            (
                "smoke/judge",
                0.0,
                "smoke",
                SEED,
                "Compute 1+1.",
                "We compute 1+1=2, so the answer is 2.",
            )
        ],
        skip_existing=False,
    )
    if len(outputs) != 1 or outputs[0].label < 0 or "api-error" in outputs[0].raw:
        raise RuntimeError(f"judge liveness failed: {outputs!r}")
    return asdict(outputs[0])


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def load_architecture(arm: str, checkpoint: Path):
    from safetensors.torch import load_file
    from temp_bench.config import instantiate_arch, load_arch

    state = load_file(str(checkpoint), device="cpu")
    if not state:
        raise RuntimeError(f"empty checkpoint: {checkpoint}")
    state_dtypes = sorted({str(tensor.dtype) for tensor in state.values()})
    parameter_dtypes = [tensor.dtype for tensor in state.values() if tensor.is_floating_point()]
    checkpoint_dtype = parameter_dtypes[0] if parameter_dtypes else torch.float32

    checkpoint_arch = ARM_TO_CHECKPOINT_ARCH[arm]
    spec = load_arch(checkpoint_arch, component="c7")
    model = instantiate_arch(spec, d_in=D_IN)
    model = model.to(dtype=checkpoint_dtype)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    model = model.cuda().eval()
    if arm.startswith("pooled_sae_"):
        model = PooledSAEAdapter(model, pool=arm.removeprefix("pooled_sae_"))
    del state
    gc.collect()
    return model, state_dtypes


def mine_feature(model, sentence_acts_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from temp_bench.case_studies.backtracking import mine_top_features, split_pos_neg

    with np.load(sentence_acts_path, allow_pickle=True) as archive:
        sentence_acts = {
            "X": archive["X"],
            "is_bt": archive["is_bt"],
            "keys": archive["keys"],
        }
    pos_neg = split_pos_neg(sentence_acts)
    mined = mine_top_features(
        model,
        pos_activations=pos_neg["pos"],
        neg_activations=pos_neg["neg"],
        top_k=32,
    )
    top = mined[0]
    summary = {
        "feature_id": top.feature_id,
        "selectivity": top.selectivity,
        "pos_act_mean": top.pos_act_mean,
        "neg_act_mean": top.neg_act_mean,
        "decoder_norm": float(top.decoder_direction.float().norm().item()),
        "n_positive_windows": int(pos_neg["pos"].shape[0]),
        "n_negative_windows": int(pos_neg["neg"].shape[0]),
        "window_shape": list(pos_neg["pos"].shape[1:]),
        "top_32": [
            {
                "feature_id": item.feature_id,
                "selectivity": item.selectivity,
                "pos_act_mean": item.pos_act_mean,
                "neg_act_mean": item.neg_act_mean,
            }
            for item in mined
        ],
    }
    return pos_neg, summary


def validate_checkpoint_config(config_path: Path, arm: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    expected_key = ARM_TO_TRAIN_KEY[arm]
    required = {
        "arch": ARM_TO_CHECKPOINT_ARCH[arm],
        "seed": SEED,
        "train_key": expected_key,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"checkpoint config {key}={config.get(key)!r}; expected {expected!r}"
            )
    if int(config.get("training_cfg", {}).get("n_steps", -1)) != 20_000:
        raise RuntimeError("checkpoint is not the matched 20k training run")
    return config


def run_arm(args: argparse.Namespace) -> dict[str, Any]:
    from temp_bench.case_studies.backtracking import (
        DEFAULT_MAGNITUDE_GRID,
        SonnetBacktrackingJudge,
        build_cohort,
        load_stage_a,
        run_arch_evaluation,
    )

    root = args.temp_bench_root.resolve()
    revision = git_revision(root)
    if revision != EXPECTED_CANONICAL_REVISION:
        raise RuntimeError(
            f"canonical code drift: {revision}; expected {EXPECTED_CANONICAL_REVISION}"
        )
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    arm_result_path = workspace / f"{args.arm}_result.json"
    if arm_result_path.exists() and not args.force:
        logging.info("complete result exists at %s; use --force to rerun", arm_result_path)
        return json.loads(arm_result_path.read_text())

    checkpoint_dir = args.checkpoints_root / ARM_TO_TRAIN_KEY[args.arm]
    checkpoint = checkpoint_dir / "model.safetensors"
    checkpoint_config_path = checkpoint_dir / "config.json"
    if not checkpoint.exists() or not checkpoint_config_path.exists():
        raise FileNotFoundError(f"missing checkpoint files in {checkpoint_dir}")
    checkpoint_config = validate_checkpoint_config(checkpoint_config_path, args.arm)

    stage_a = load_stage_a()
    cohort = build_cohort(stage_a)
    if len(cohort.truly_wrong) != EXPECTED_WRONG or len(cohort.originally_correct) != EXPECTED_CORRECT:
        raise RuntimeError(
            "cohort drift: "
            f"wrong={len(cohort.truly_wrong)}, correct={len(cohort.originally_correct)}"
        )

    model, state_dtypes = load_architecture(args.arm, checkpoint)
    pos_neg, feature = mine_feature(model, args.sentence_acts)
    preflight = {
        "arm": args.arm,
        "checkpoint_arch": ARM_TO_CHECKPOINT_ARCH[args.arm],
        "train_key": ARM_TO_TRAIN_KEY[args.arm],
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_state_dtypes": state_dtypes,
        "checkpoint_config": checkpoint_config,
        "feature": feature,
        "cohort": {
            "wrong": len(cohort.truly_wrong),
            "correct": len(cohort.originally_correct),
            "qids": cohort.all,
        },
        "magnitudes": list(DEFAULT_MAGNITUDE_GRID),
        "cut_fraction": 0.25,
        "max_new_tokens": 1024,
        "gen_batch_size": args.gen_batch_size,
        "canonical_git_revision": revision,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "started_unix": time.time(),
    }
    atomic_write_json(workspace / f"{args.arm}_preflight.json", preflight)

    judge = SonnetBacktrackingJudge(workspace=workspace)
    result = run_arch_evaluation(
        arch=model,
        seed=SEED,
        cohort=cohort,
        stage_a=stage_a,
        workspace=workspace,
        judge=judge,
        magnitudes=DEFAULT_MAGNITUDE_GRID,
        cut_fraction=0.25,
        arch_name=args.arm,
        feature_mining_acts=pos_neg,
        sentence_acts=None,
        sentence_labels=None,
        sentence_qids=None,
        max_new_tokens=1024,
        gen_batch_size=args.gen_batch_size,
    )

    judge_rows = read_jsonl(workspace / "judge_outputs.jsonl")
    successful = successful_judge_keys(judge_rows, arch=args.arm, seed=SEED)
    expected = len(cohort) * len(DEFAULT_MAGNITUDE_GRID)
    if len(successful) != expected:
        raise RuntimeError(
            f"incomplete judging for {args.arm}: {len(successful)}/{expected} unique successes"
        )

    payload = {
        **preflight,
        "finished_unix": time.time(),
        "successful_judge_keys": len(successful),
        "expected_judge_keys": expected,
        "primary_metric": result.primary_metric,
        "metrics": result.metrics,
        "artifacts": {name: str(path) for name, path in result.artifacts.items()},
    }
    atomic_write_json(arm_result_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(ARM_TO_TRAIN_KEY))
    parser.add_argument("--temp-bench-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--sentence-acts", type=Path, required=True)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["TEMP_BENCH_ROOT"] = str(args.temp_bench_root.resolve())
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    verify_zero_hook_noop()
    smoke_path = args.workspace / "smoke" / "judge_liveness.json"
    if not smoke_path.exists():
        smoke = asyncio.run(verify_judge_liveness(args.workspace / "smoke"))
        atomic_write_json(smoke_path, smoke)
    elif int(json.loads(smoke_path.read_text()).get("label", -1)) < 0:
        raise RuntimeError("cached judge liveness result is invalid")

    if args.preflight_only:
        logging.info("zero-hook and judge preflight passed")
        return 0

    result = run_arm(args)
    logging.info(
        "%s complete: delta_gc_peak=%s at magnitude=%s",
        args.arm,
        result["metrics"].get("delta_gc_peak"),
        result["metrics"].get("delta_gc_peak_magnitude"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
