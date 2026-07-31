"""Run the decisive DailyDialog learned-SAE-trajectory control.

The canonical TempBench runner trains the shared SAE and TXC cells.  A
checkpointed second stage then compresses the frozen SAE's 32 code vectors
into one top-8 code using :class:`TrajectoryBottleneck`.  Evaluation delegates
to the existing grouped-dialogue lambda-recovery implementation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from experiments.power_spectrum.decision_sprint.model import (
    FlexibleTrajectoryBottleneck,
)
from temp_bench.core.config import checkpoint_dir, load_arch, load_datasource
from temp_bench.core.runner import _load_checkpoint, run_experiment
from temp_bench.core.schemas import TrainingConfig
from temp_bench.data.synthetic import materialise
from temp_bench.evals.lambda_recovery_v2 import lambda_recovery_v2_metrics


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
PROTOCOL = "dailydialog-learned-sae-trajectory-control.v1"


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def heartbeat(root: Path, *, stage: str, **fields: Any) -> None:
    atomic_json(
        {
            "protocol": PROTOCOL,
            "stage": stage,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "unix_time": time.time(),
            **fields,
        },
        root / "heartbeat.json",
    )


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config["protocol"] != PROTOCOL:
        raise ValueError("protocol mismatch")
    if config["window"] != config["eval_window"]:
        raise ValueError("this sprint requires one T=32 tile per eval window")
    if config["batch_positions"] % config["window"]:
        raise ValueError("batch_positions must be divisible by window")
    return config


def training_config(
    config: dict[str, Any], *, arch: str, steps: int
) -> TrainingConfig:
    window = config["window"] if arch == "txc_batchtopk_post" else 1
    batch_size = config["batch_positions"] // window
    return TrainingConfig(
        n_steps=steps,
        batch_size=batch_size,
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        buffer_tokens=config["buffer_tokens"],
        arch_hparams_override={
            "d_sae": config["d_sae"],
            "k_pos": config["k_pos"],
            "T": window,
        },
    )


def _resolved_arch(arch: str, training: TrainingConfig):
    spec = load_arch(arch, section="synthetic")
    merged = {**spec.hparams, **(training.arch_hparams_override or {})}
    return spec.model_copy(update={"hparams": merged})


def train_canonical(
    *,
    arch: str,
    seed: int,
    steps: int,
    config: dict[str, Any],
    output_root: Path,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    training = training_config(config, arch=arch, steps=steps)
    evaluation = {
        "smoke": False,
        "k_pos": config["k_pos"],
        "eval_window_L": config["eval_window"],
        "lambda_probe_v2": True,
        "lambda_v2_probe": "ridge",
        "lambda_v2_alphas": config["eval_alphas"],
        "lambda_v2_n_windows": config["eval_n_windows"],
        "lambda_v2_split": "trace",
    }
    heartbeat(output_root, stage=f"canonical_{arch}", seed=seed)
    result = run_experiment(
        experiment="synthetic",
        arch_name=arch,
        seed=seed,
        datasource_name=config["datasource"],
        training_cfg=training,
        eval_cfg=evaluation,
        agent="txc-decision-sprint",
        allow_dirty=False,
    )
    spec = _resolved_arch(arch, training)
    model = _load_checkpoint(
        spec, result.train_key, load_datasource(config["datasource"])
    )
    model.to("cuda").eval()
    receipt = {
        "arch": arch,
        "seed": seed,
        "steps": steps,
        "train_key": result.train_key,
        "eval_key": result.eval_key,
        "train_cached": result.train_cached,
        "eval_cached": result.eval_cached,
        "metrics": {
            key: float(value) for key, value in result.row.metrics.items()
        },
    }
    atomic_json(
        receipt,
        output_root
        / "canonical"
        / f"{arch}_steps{steps}_seed{seed}.json",
    )
    return model, receipt


@torch.no_grad()
def precompute_codes(
    *,
    base: torch.nn.Module,
    x: torch.Tensor,
    code_root: Path,
    config: dict[str, Any],
    seed: int,
    base_identity: dict[str, Any],
    rms_scale: float,
    batch_tokens: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode the finite corpus once into fixed-width sparse arrays."""

    indices_path = code_root / "indices.npy"
    values_path = code_root / "values.npy"
    receipt_path = code_root / "complete.json"
    shape = (*x.shape[:2], config["k_pos"])
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("protocol") == PROTOCOL
            and receipt.get("seed") == seed
            and receipt.get("shape") == list(shape)
            and receipt.get("base_identity") == base_identity
            and receipt.get("rms_scale") == rms_scale
            and receipt.get("value_dtype") == "float32"
        ):
            return (
                np.load(indices_path, mmap_mode="r"),
                np.load(values_path, mmap_mode="r"),
            )

    code_root.mkdir(parents=True, exist_ok=True)
    indices = np.lib.format.open_memmap(
        indices_path, mode="w+", dtype=np.int32, shape=shape
    )
    values = np.lib.format.open_memmap(
        values_path, mode="w+", dtype=np.float32, shape=shape
    )
    flat = x.reshape(-1, x.shape[-1])
    base.eval()
    for start in range(0, len(flat), batch_tokens):
        end = min(start + batch_tokens, len(flat))
        token = flat[start:end].to("cuda", non_blocking=True)
        dense = base.encode(token)
        active_values, active_indices = dense.topk(config["k_pos"], dim=-1)
        indices.reshape(-1, config["k_pos"])[start:end] = (
            active_indices.cpu().numpy().astype(np.int32)
        )
        values.reshape(-1, config["k_pos"])[start:end] = (
            active_values.float().cpu().numpy()
        )
    indices.flush()
    values.flush()
    atomic_json(
        {
            "protocol": PROTOCOL,
            "seed": seed,
            "shape": list(shape),
            "k_pos": config["k_pos"],
            "base_identity": base_identity,
            "rms_scale": rms_scale,
            "value_dtype": "float32",
        },
        receipt_path,
    )
    return (
        np.load(indices_path, mmap_mode="r"),
        np.load(values_path, mmap_mode="r"),
    )


def make_adapter(
    *,
    base: torch.nn.Module,
    rank: int,
    config: dict[str, Any],
    device: str = "cuda",
) -> FlexibleTrajectoryBottleneck:
    return FlexibleTrajectoryBottleneck(
        base_decoder=base.W_dec.detach(),
        base_decoder_bias=base.b_dec.detach(),
        window=config["window"],
        k_window=config["k_window"],
        rank=rank,
        decoder_rank=int(config["adapter_decoder_rank"][str(rank)]),
    ).to(device)


def _save_adapter(
    model: FlexibleTrajectoryBottleneck,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    metrics: dict[str, float],
    cell: Path,
) -> None:
    cell.mkdir(parents=True, exist_ok=True)
    model_tmp = cell / "model.safetensors.tmp"
    save_file(
        {
            key: value.detach().contiguous().cpu()
            for key, value in model.state_dict().items()
        },
        str(model_tmp),
    )
    os.replace(model_tmp, cell / "model.safetensors")
    state_tmp = cell / "training_state.pt.tmp"
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        state_tmp,
    )
    os.replace(state_tmp, cell / "training_state.pt")


def train_adapter(
    *,
    base: torch.nn.Module,
    x: torch.Tensor,
    indices: np.ndarray,
    values: np.ndarray,
    rank: int,
    seed: int,
    config: dict[str, Any],
    output_root: Path,
    base_identity: dict[str, Any],
    rms_scale: float,
) -> tuple[FlexibleTrajectoryBottleneck, dict[str, Any]]:
    cell = output_root / "checkpoints" / f"seed{seed}" / f"rank{rank}"
    identity = {
        "protocol": PROTOCOL,
        "seed": seed,
        "rank": rank,
        "decoder_rank": int(config["adapter_decoder_rank"][str(rank)]),
        "base_identity": base_identity,
        "rms_scale": rms_scale,
        "window": config["window"],
        "k_pos": config["k_pos"],
        "k_window": config["k_window"],
        "adapter_steps": config["adapter_steps"],
    }
    identity_path = cell / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError(f"adapter identity mismatch at {cell}")
    else:
        atomic_json(identity, identity_path)
    model = make_adapter(base=base, rank=rank, config=config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["learning_rate"]
    )
    start_step = 0
    log: list[dict[str, float]] = []
    last_metrics: dict[str, float] = {}
    model_path = cell / "model.safetensors"
    state_path = cell / "training_state.pt"
    if model_path.exists() != state_path.exists():
        raise ValueError(f"partial adapter checkpoint at {cell}")
    if model_path.exists():
        model.load_state_dict(load_file(str(model_path), device="cuda"))
        state = torch.load(state_path, map_location="cuda", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        last_metrics = dict(state.get("metrics", {}))
        if (cell / "training_log.json").exists():
            log = json.loads((cell / "training_log.json").read_text())

    batch = config["batch_positions"] // config["window"]
    offsets = np.arange(config["window"])[None, :]
    model.train()
    for step in range(start_step, config["adapter_steps"]):
        rng = np.random.default_rng(
            seed * 1_000_003 + rank * 10_007 + step
        )
        sequence = rng.integers(0, x.shape[0], size=batch)
        starts = rng.integers(
            0, x.shape[1] - config["window"] + 1, size=batch
        )
        positions = starts[:, None] + offsets
        sparse_indices = torch.from_numpy(
            np.asarray(indices[sequence[:, None], positions], dtype=np.int64)
        ).cuda(non_blocking=True)
        sparse_values = torch.from_numpy(
            np.asarray(values[sequence[:, None], positions], dtype=np.float32)
        ).cuda(non_blocking=True)
        target = x[sequence[:, None], positions].cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        result = model.loss(
            sparse_indices, sparse_values, target, update_dead=True
        )
        result["loss"].backward()
        completed = step + 1
        scale = min(1.0, completed / config["warmup_steps"])
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * scale
        optimizer.step()
        model.normalize_decoder_profiles()
        last_metrics = {
            key: float(value.detach().float().cpu())
            for key, value in result.items()
        }
        if (
            completed % config["checkpoint_every"] == 0
            or completed == config["adapter_steps"]
        ):
            row = {"step": completed, **last_metrics}
            log = [entry for entry in log if entry["step"] < completed] + [row]
            _save_adapter(
                model,
                optimizer,
                step=completed,
                metrics=last_metrics,
                cell=cell,
            )
            atomic_json(log, cell / "training_log.json")
            heartbeat(
                output_root,
                stage=f"adapter_rank{rank}",
                seed=seed,
                step=completed,
                total=config["adapter_steps"],
                metrics=last_metrics,
            )
            print(
                f"[adapter] seed={seed} rank={rank} "
                f"step={completed}/{config['adapter_steps']} "
                f"loss={last_metrics['loss']:.5f} "
                f"l0={last_metrics['l0']:.3f}",
                flush=True,
            )
    model.eval()
    receipt = {
        "seed": seed,
        "rank": rank,
        "steps": config["adapter_steps"],
        "trainable_parameters": model.trainable_parameter_count(),
        "last_metrics": last_metrics,
        "checkpoint": str(cell),
        "identity": identity,
    }
    atomic_json(receipt, cell / "receipt.json")
    return model, receipt


class _AdapterEval(torch.nn.Module):
    consumes = "window"

    def __init__(
        self,
        base: torch.nn.Module,
        adapter: FlexibleTrajectoryBottleneck,
        *,
        config: dict[str, Any],
        order: str,
    ) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.config = SimpleNamespace(T=config["window"])
        self.k_pos = int(config["k_pos"])
        self.order = order

    def _order(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == "ordered":
            return x
        if self.order == "reverse":
            # Preserve the target-aligned leading-edge token and reverse only
            # its history; otherwise an order drop is confounded by replacing
            # the current-token anchor.
            return torch.cat((x[:, :-1].flip(1), x[:, -1:]), dim=1)
        raise ValueError(self.order)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._order(x)
        dense = self.base.encode(x)
        values, indices = dense.topk(self.k_pos, dim=-1)
        active_values, active_indices, _ = self.adapter.encode_sparse(
            indices, values
        )
        output = dense.new_zeros((len(x), self.adapter.d_sae))
        return output.scatter(1, active_indices, active_values)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        values, indices = z.topk(self.adapter.k_window, dim=-1)
        return self.adapter.decode_sparse(values, indices, add_bias=True)


class _PoolEval(torch.nn.Module):
    consumes = "window"

    def __init__(
        self,
        base: torch.nn.Module,
        *,
        config: dict[str, Any],
        pool: str,
    ) -> None:
        super().__init__()
        self.base = base
        self.config = SimpleNamespace(T=config["window"])
        self.k_window = int(config["k_window"])
        self.pool = pool

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        dense = self.base.encode(x)
        if self.pool == "last":
            pooled = dense[:, -1]
        elif self.pool == "mean":
            pooled = dense.mean(dim=1)
        elif self.pool == "max":
            pooled = dense.amax(dim=1)
        else:
            raise ValueError(self.pool)
        values, indices = pooled.topk(self.k_window, dim=-1)
        return pooled.new_zeros(pooled.shape).scatter(1, indices, values)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.base.decode(z).unsqueeze(1).expand(
            -1, self.config.T, -1
        )


class _TXCOrderEval(torch.nn.Module):
    consumes = "window"

    def __init__(self, txc: torch.nn.Module, *, window: int, order: str) -> None:
        super().__init__()
        self.txc = txc
        self.config = SimpleNamespace(T=window)
        self.order = order

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.order == "reverse":
            x = torch.cat((x[:, :-1].flip(1), x[:, -1:]), dim=1)
        elif self.order != "ordered":
            raise ValueError(self.order)
        return self.txc.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.txc.decode(z)


def evaluate_model(
    model: torch.nn.Module,
    *,
    data,
    config: dict[str, Any],
) -> dict[str, float]:
    metrics = lambda_recovery_v2_metrics(
        model,
        data,
        eval_window_L=config["eval_window"],
        eval_cfg={
            "lambda_v2_probe": "ridge",
            "lambda_v2_alphas": config["eval_alphas"],
            "lambda_v2_n_windows": config["eval_n_windows"],
            "lambda_v2_split": "trace",
        },
    )
    output = {key: float(value) for key, value in metrics.items()}
    output["l0_per_window"] = realized_l0(
        model, data=data, window=config["window"]
    )
    return output


@torch.no_grad()
def realized_l0(
    model: torch.nn.Module,
    *,
    data,
    window: int,
    n_windows: int = 512,
    batch_size: int = 64,
) -> float:
    """Deterministic support receipt on the first window of held rows."""

    n = min(n_windows, len(data.x))
    counts = []
    for start in range(0, n, batch_size):
        x = data.x[start : min(start + batch_size, n), :window].cuda()
        z = model.encode(x)
        counts.append((z != 0).sum(dim=-1).float().cpu())
    return float(torch.cat(counts).mean())


def run_seed(
    *,
    seed: int,
    config: dict[str, Any],
    output_root: Path,
    data,
) -> dict[str, Any]:
    seed_path = output_root / "seeds" / f"seed{seed}.json"
    if seed_path.exists():
        existing = json.loads(seed_path.read_text())
        if existing.get("status") == "complete":
            return existing

    base, base_receipt = train_canonical(
        arch="batchtopk_sae",
        seed=seed,
        steps=config["base_steps"],
        config=config,
        output_root=output_root,
    )
    txc, txc_receipt = train_canonical(
        arch="txc_batchtopk_post",
        seed=seed,
        steps=config["txc_steps"],
        config=config,
        output_root=output_root,
    )
    txc_untrained, txc_untrained_receipt = train_canonical(
        arch="txc_batchtopk_post",
        seed=seed,
        steps=0,
        config=config,
        output_root=output_root,
    )
    heartbeat(output_root, stage="evaluate_txc_reverse", seed=seed)
    txc_reverse = evaluate_model(
        _TXCOrderEval(
            txc, window=config["window"], order="reverse"
        ).cuda().eval(),
        data=data,
        config=config,
    )
    del txc
    del txc_untrained
    gc.collect()
    torch.cuda.empty_cache()

    code_root = output_root / "codes" / f"seed{seed}"
    base_checkpoint = checkpoint_dir(base_receipt["train_key"]) / (
        "model.safetensors"
    )
    base_identity = {
        "train_key": base_receipt["train_key"],
        "checkpoint_sha256": file_sha256(base_checkpoint),
    }
    rms_scale = float(data.extra["rms_scale"])
    indices, values = precompute_codes(
        base=base,
        x=data.x,
        code_root=code_root,
        config=config,
        seed=seed,
        base_identity=base_identity,
        rms_scale=rms_scale,
    )
    models: dict[str, torch.nn.Module] = {
        "sae_last": _PoolEval(base, config=config, pool="last"),
        "sae_mean_top8": _PoolEval(base, config=config, pool="mean"),
        "sae_max_top8": _PoolEval(base, config=config, pool="max"),
    }
    adapter_receipts: dict[str, Any] = {}
    for rank in config["adapter_ranks"]:
        torch.manual_seed(seed * 10_000 + rank)
        untrained = make_adapter(base=base, rank=rank, config=config)
        models[f"adapter_rank{rank}_untrained"] = _AdapterEval(
            base, untrained, config=config, order="ordered"
        )
        adapter, receipt = train_adapter(
            base=base,
            x=data.x,
            indices=indices,
            values=values,
            rank=rank,
            seed=seed,
            config=config,
            output_root=output_root,
            base_identity=base_identity,
            rms_scale=rms_scale,
        )
        adapter_receipts[f"rank{rank}"] = receipt
        models[f"adapter_rank{rank}"] = _AdapterEval(
            base, adapter, config=config, order="ordered"
        )
        models[f"adapter_rank{rank}_reverse"] = _AdapterEval(
            base, adapter, config=config, order="reverse"
        )

    evaluations: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        heartbeat(output_root, stage="evaluate", seed=seed, model=name)
        model.cuda().eval()
        evaluations[name] = evaluate_model(model, data=data, config=config)
        print(
            f"[eval] seed={seed} {name}: "
            f"r={evaluations[name]['lambda_recovery_v2']:.4f}",
            flush=True,
        )
        atomic_json(
            evaluations,
            output_root / "evaluations" / f"seed{seed}.json",
        )

    result = {
        "status": "complete",
        "protocol": PROTOCOL,
        "seed": seed,
        "canonical": {
            "base_sae": base_receipt,
            "txc": txc_receipt,
            "txc_untrained": txc_untrained_receipt,
        },
        "adapters": adapter_receipts,
        "evaluations": {"txc_reverse": txc_reverse, **evaluations},
    }
    atomic_json(result, seed_path)
    del models, base
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/txc_decision_sprint/results"),
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output / "frozen_config.json"
    if frozen_path.exists():
        if json.loads(frozen_path.read_text()) != config:
            raise ValueError("output directory contains a different config")
    else:
        atomic_json(config, frozen_path)
    seeds = [args.seed] if args.seed is not None else config["seeds"]
    if not set(seeds).issubset(config["seeds"]):
        raise ValueError("requested seed was not preregistered")
    results = []
    for seed in seeds:
        heartbeat(args.output, stage="materialise", seed=seed)
        data = materialise(load_datasource(config["datasource"]), seed=seed)
        if tuple(data.x.shape) != (4111, 128, config["d_in"]):
            raise ValueError(
                f"unexpected activation shape: {tuple(data.x.shape)}"
            )
        results.append(
            run_seed(
                seed=seed,
                config=config,
                output_root=args.output,
                data=data,
            )
        )
    receipts = []
    for seed in config["seeds"]:
        path = args.output / "seeds" / f"seed{seed}.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt.get("status") == "complete":
                receipts.append(receipt)
    complete = sorted(row["seed"] for row in receipts) == sorted(
        config["seeds"]
    )
    atomic_json(
        {
            "status": "complete" if complete else "incomplete",
            "protocol": PROTOCOL,
            "seeds": sorted(row["seed"] for row in receipts),
            "expected_seeds": config["seeds"],
            "results": sorted(receipts, key=lambda row: row["seed"]),
        },
        args.output / "raw_results.json",
    )
    heartbeat(
        args.output,
        stage="complete" if complete else "partial",
        seeds=seeds,
        completed_seeds=sorted(row["seed"] for row in receipts),
    )


if __name__ == "__main__":
    main()
