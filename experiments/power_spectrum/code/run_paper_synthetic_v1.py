"""Paper-compatible Spectral-v1 sweep on the two Figure 2 synthetic tasks.

The current repository's synthetic generators and trainer have evolved since
the paper snapshot. This runner therefore ports the *pinned paper recipes*
inside the isolated power-spectrum experiment:

- fixed data seed 0 shared across model seeds;
- legacy Torch generators for Denoising and Coupling;
- one random native-T window per sequence row and optimizer step;
- Adam, 1k linear warmup, gradient clipping, and paper step counts;
- exact Denoising sliding-window Ridge probe and Coupling cosine-AUC.

The frozen grid is focused rather than exhaustive: it contains the published
best TXC cell and dense-code alternatives for Denoising, and the published
best T/k neighborhood for Coupling. Every selected cell is run at all three
paper seeds.

Only Spectral v1 is trained. Published SAE, T-SAE, and TXC values are extracted
separately by :mod:`extract_paper_baselines`.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


POWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = POWER_ROOT / "configs" / "paper_synthetic_v1.json"
DEFAULT_RESULTS = POWER_ROOT / "results" / "paper_synthetic_v1"


@dataclass
class PaperData:
    x: torch.Tensor
    emission_features: torch.Tensor
    hidden_features: torch.Tensor | None
    support: torch.Tensor | None
    hidden_support: torch.Tensor | None


class StopRequested(RuntimeError):
    """The configured wall-clock or cost limit was reached."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _stable_id(value: Any, length: int = 16) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()[:length]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    with open(path, "a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{line_number}: {exc}") from exc
    return rows


def latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("cell_id"):
            latest[str(row["cell_id"])] = row
    return latest


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("paper synthetic config must have schema_version=1")
    if config["model"]["name"] != "spectral_v1":
        raise ValueError("this runner is intentionally restricted to spectral_v1")
    if int(config["model"]["d_sae"]) != 40:
        raise ValueError("paper comparison requires d_sae=40")
    for task_name, task in config["tasks"].items():
        for T_text, ks in task["grid"].items():
            T = int(T_text)
            for k_pos in ks:
                if int(k_pos) * T > 40:
                    raise ValueError(
                        f"{task_name}: invalid T={T}, k_pos={k_pos} at d_sae=40"
                    )
    return config


def enumerate_cells(
    config: dict[str, Any],
    *,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for task_name, task in config["tasks"].items():
        task_grid = task["grid"]
        if smoke:
            T_text = "2" if task_name == "denoising" else "5"
            task_grid = {T_text: [task["grid"][T_text][0]]}
            seeds = [1]
        else:
            seeds = config["seeds"]
        for T_text, ks in task_grid.items():
            T = int(T_text)
            for k_pos in ks:
                for seed in seeds:
                    core = {
                        "run_name": config["run_name"],
                        "task": task_name,
                        "datasource": task["datasource"],
                        "model": config["model"]["name"],
                        "implementation_version": config["model"][
                            "implementation_version"
                        ],
                        "T": T,
                        "k_pos": int(k_pos),
                        "d_sae": int(config["model"]["d_sae"]),
                        "seed": int(seed),
                        "n_steps": 2 if smoke else int(task["n_steps"]),
                        "metric": task["metric"],
                        "smoke": smoke,
                    }
                    cells.append({**core, "cell_id": _stable_id(core)})
    return cells


def build_plan(config: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    cells = enumerate_cells(config, smoke=smoke)
    planning = config["planning"]
    seconds = float(planning["setup_seconds"])
    optimizer_steps = 0
    task_counts: dict[str, int] = {}
    for cell in cells:
        task = str(cell["task"])
        task_counts[task] = task_counts.get(task, 0) + 1
        optimizer_steps += int(cell["n_steps"])
        seconds += int(cell["n_steps"]) / float(
            planning["estimated_steps_per_second"][task]
        )
        seconds += float(planning["eval_seconds_per_cell"][task])
    budget = config["budget"]
    effective_rate = float(budget["assumed_usd_per_gpu_hour"]) * float(
        budget["cost_overhead_multiplier"]
    )
    estimated_cost = seconds / 3600.0 * effective_rate
    usable = float(budget["max_total_usd"]) - float(budget["reserve_usd"])
    return {
        "run_name": config["run_name"],
        "smoke": smoke,
        "task_cell_counts": task_counts,
        "total_cells": len(cells),
        "total_optimizer_steps": optimizer_steps,
        "estimated_gpu_hours": round(seconds / 3600.0, 3),
        "effective_assumed_usd_per_hour": round(effective_rate, 3),
        "estimated_cost_usd": round(estimated_cost, 2),
        "usable_budget_usd": round(usable, 2),
        "max_session_hours": float(budget["max_session_hours"]),
        "within_cost_plan": estimated_cost <= usable,
        "within_time_plan": (
            seconds / 3600.0 <= float(budget["max_session_hours"])
        ),
    }


class BudgetGuard:
    def __init__(
        self,
        config: dict[str, Any],
        results_dir: Path,
        *,
        config_hash: str,
    ):
        budget = config["budget"]
        self.path = results_dir / "spend.json"
        self.config_hash = config_hash
        self.rate = float(budget["assumed_usd_per_gpu_hour"])
        self.multiplier = float(budget["cost_overhead_multiplier"])
        self.usable = float(budget["max_total_usd"]) - float(
            budget["reserve_usd"]
        )
        self.max_seconds = float(budget["max_session_hours"]) * 3600.0
        self.started = time.monotonic()
        self.session_id = uuid.uuid4().hex[:12]
        if self.path.exists():
            self.ledger = json.loads(self.path.read_text())
            if self.ledger.get("config_hash") != config_hash:
                raise RuntimeError("spend ledger belongs to a different config")
        else:
            self.ledger = {
                "schema_version": 1,
                "config_hash": config_hash,
                "max_total_usd": float(budget["max_total_usd"]),
                "reserve_usd": float(budget["reserve_usd"]),
                "usable_budget_usd": self.usable,
                "assumed_usd_per_gpu_hour": self.rate,
                "cost_overhead_multiplier": self.multiplier,
                "sessions": [],
            }
        for session in self.ledger["sessions"]:
            if session.get("status") == "running":
                session["status"] = "interrupted"
        prior = sum(
            float(session.get("estimated_cost_usd", 0.0))
            for session in self.ledger["sessions"]
        )
        if prior >= self.usable:
            raise StopRequested("prior estimated spend exhausted usable budget")
        self.ledger["sessions"].append(
            {
                "session_id": self.session_id,
                "started_at": _utc_now(),
                "last_seen_at": _utc_now(),
                "elapsed_seconds": 0.0,
                "estimated_cost_usd": 0.0,
                "status": "running",
            }
        )
        self.update()

    def _session(self) -> dict[str, Any]:
        return next(
            session
            for session in self.ledger["sessions"]
            if session["session_id"] == self.session_id
        )

    def update(self, status: str = "running") -> float:
        elapsed = time.monotonic() - self.started
        session = self._session()
        session.update(
            {
                "last_seen_at": _utc_now(),
                "elapsed_seconds": round(elapsed, 3),
                "estimated_cost_usd": round(
                    elapsed / 3600.0 * self.rate * self.multiplier,
                    6,
                ),
                "status": status,
            }
        )
        total = sum(
            float(item["estimated_cost_usd"])
            for item in self.ledger["sessions"]
        )
        self.ledger["estimated_total_usd"] = round(total, 6)
        self.ledger["remaining_usable_usd"] = round(self.usable - total, 6)
        _atomic_json(self.path, self.ledger)
        return total

    def check(self) -> None:
        elapsed = time.monotonic() - self.started
        total = self.update()
        if elapsed >= self.max_seconds:
            raise StopRequested("session deadline reached")
        if total >= self.usable:
            raise StopRequested("usable cost cap reached")

    def finish(self, status: str) -> None:
        self.update(status=status)


# ---------------------------------------------------------------------------
# Pinned paper generators
# ---------------------------------------------------------------------------


def paper_markov_data(
    *,
    n_seqs: int,
    seed: int,
    device: str = "cpu",
) -> PaperData:
    """Exact origin/final C1 noisy generator for the paper's fixed params."""
    n_features = 20
    d_in = 40
    seq_len = 64
    rho_levels = [0.7]
    pi = 0.5
    magnitude_mean = 1.0
    magnitude_std = 0.15
    p_A = 0.0
    p_B = 0.625
    rng = torch.Generator(device="cpu").manual_seed(int(seed))

    per_level = n_features // len(rho_levels)
    rho_t = torch.cat(
        [torch.full((per_level,), float(rho)) for rho in rho_levels]
    )
    pi_t = torch.full((n_features,), pi)
    gaussian = torch.randn(d_in, n_features, generator=rng)
    features, _ = torch.linalg.qr(gaussian, mode="reduced")
    features = features.T.contiguous()

    p01 = pi_t * (1.0 - rho_t)
    p10 = (1.0 - pi_t) * (1.0 - rho_t)
    p_stay_on = 1.0 - p10
    uniforms = torch.rand(n_seqs, n_features, seq_len, generator=rng)
    hidden = torch.empty(n_seqs, n_features, seq_len)
    hidden[:, :, 0] = (
        uniforms[:, :, 0] < pi_t.unsqueeze(0)
    ).float()
    for position in range(1, seq_len):
        previous = hidden[:, :, position - 1]
        probability_on = (
            previous * p_stay_on.unsqueeze(0)
            + (1.0 - previous) * p01.unsqueeze(0)
        )
        hidden[:, :, position] = (
            uniforms[:, :, position] < probability_on
        ).float()

    observed_uniforms = torch.rand(
        n_seqs,
        n_features,
        seq_len,
        generator=rng,
    )
    probability_emit = hidden * p_B + (1.0 - hidden) * p_A
    support = (observed_uniforms < probability_emit).float()
    raw_magnitudes = (
        torch.randn(n_seqs, n_features, seq_len, generator=rng)
        * magnitude_std
        + magnitude_mean
    )
    activations = support * raw_magnitudes.abs()
    x = torch.einsum("nft,fd->ntd", activations, features)
    target = torch.device(device)
    return PaperData(
        x=x.to(target),
        emission_features=features.to(target),
        hidden_features=None,
        support=support.to(target),
        hidden_support=hidden.to(target),
    )


def _orthogonalise(
    *,
    n_vectors: int,
    d_in: int,
    rng: torch.Generator,
) -> torch.Tensor:
    gaussian = torch.randn(d_in, n_vectors, generator=rng)
    orthogonal, _ = torch.linalg.qr(gaussian, mode="reduced")
    return orthogonal.T.contiguous()


def _markov_chain_batch(
    *,
    n_seqs: int,
    n_hidden: int,
    seq_len: int,
    pi: float,
    rho: float,
    rng: torch.Generator,
) -> torch.Tensor:
    pi_t = torch.full((n_hidden,), pi)
    rho_t = torch.full((n_hidden,), rho)
    p01 = pi_t * (1.0 - rho_t)
    p10 = (1.0 - pi_t) * (1.0 - rho_t)
    p_stay_on = 1.0 - p10
    uniforms = torch.rand(n_seqs, n_hidden, seq_len, generator=rng)
    states = torch.empty(n_seqs, n_hidden, seq_len)
    states[:, :, 0] = (
        uniforms[:, :, 0] < pi_t.unsqueeze(0)
    ).float()
    for position in range(1, seq_len):
        previous = states[:, :, position - 1]
        probability_on = (
            previous * p_stay_on.unsqueeze(0)
            + (1.0 - previous) * p01.unsqueeze(0)
        )
        states[:, :, position] = (
            uniforms[:, :, position] < probability_on
        ).float()
    return states


def paper_coupling_data(
    *,
    seed: int,
    n_seqs: int = 4096,
    device: str = "cpu",
) -> PaperData:
    """Exact origin/final noisy-coupling generator for the np10 task."""
    n_hidden = 10
    n_emissions = 20
    n_parents = 10
    d_in = 256
    seq_len = 64
    pi = 0.05
    rho = 0.9
    p_A = 0.0
    p_B = 0.5
    magnitude_mean = 1.0
    magnitude_std = 0.15
    rng = torch.Generator(device="cpu").manual_seed(int(seed))

    emission_features = _orthogonalise(
        n_vectors=n_emissions,
        d_in=d_in,
        rng=rng,
    )
    coupling = torch.zeros(n_emissions, n_hidden)
    for emission in range(n_emissions):
        parents = torch.randperm(n_hidden, generator=rng)[:n_parents]
        coupling[emission, parents] = 1.0
    hidden_features = coupling.T @ emission_features
    hidden_features = hidden_features / hidden_features.norm(
        dim=1,
        keepdim=True,
    ).clamp(min=1e-8)

    hidden = _markov_chain_batch(
        n_seqs=n_seqs,
        n_hidden=n_hidden,
        seq_len=seq_len,
        pi=pi,
        rho=rho,
        rng=rng,
    )
    parent_sum = torch.einsum("mk,nkt->nmt", coupling, hidden)
    clean_support = (parent_sum >= 1).float()
    uniforms = torch.rand(
        n_seqs,
        n_emissions,
        seq_len,
        generator=rng,
    )
    probability = clean_support * p_B + (1.0 - clean_support) * p_A
    support = (uniforms < probability).float()
    magnitudes = (
        torch.randn(
            n_seqs,
            n_emissions,
            seq_len,
            generator=rng,
        )
        * magnitude_std
        + magnitude_mean
    ).abs()
    x = torch.einsum(
        "nmt,md->ntd",
        support * magnitudes,
        emission_features,
    )
    target = torch.device(device)
    return PaperData(
        x=x.to(target),
        emission_features=emission_features.to(target),
        hidden_features=hidden_features.to(target),
        support=support.permute(0, 2, 1).to(target),
        hidden_support=hidden.permute(0, 2, 1).to(target),
    )


# ---------------------------------------------------------------------------
# Training and paper metrics
# ---------------------------------------------------------------------------


def _import_model(config: dict[str, Any]):
    module_name, class_name = config["model"]["class_path"].split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _model(
    config: dict[str, Any],
    cell: dict[str, Any],
    *,
    d_in: int,
    device: str,
):
    cls = _import_model(config)
    model_config = config["model"]
    return cls(
        d_in=d_in,
        d_sae=int(cell["d_sae"]),
        T=int(cell["T"]),
        k_pos=int(cell["k_pos"]),
        bands=str(model_config["bands"]),
        auxk_alpha=float(model_config["auxk_alpha"]),
    ).to(device)


def _sample_windows(
    data: PaperData,
    *,
    T: int,
    batch_size: int,
    sequence_rng: torch.Generator,
    device: str,
) -> torch.Tensor:
    n_sequences, seq_len, _ = data.x.shape
    sequence_indices = torch.randint(
        0,
        n_sequences,
        (batch_size,),
        generator=sequence_rng,
    )
    sequences = data.x[sequence_indices.to(data.x.device)].to(device)
    offsets = torch.randint(
        0,
        seq_len - T + 1,
        (batch_size,),
        device=device,
    )
    positions = offsets[:, None] + torch.arange(T, device=device)[None, :]
    rows = torch.arange(batch_size, device=device)[:, None]
    return sequences[rows, positions]


def train_cell(
    config: dict[str, Any],
    cell: dict[str, Any],
    data: PaperData,
    budget: BudgetGuard,
) -> tuple[Any, dict[str, Any]]:
    training = config["training"]
    seed = int(cell["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = _model(
        config,
        cell,
        d_in=int(data.x.shape[-1]),
        device=device,
    )
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
    )
    sequence_rng = torch.Generator(device="cpu").manual_seed(seed)
    precision = str(training["precision"])
    use_autocast = device == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    n_steps = int(cell["n_steps"])
    warmup_steps = int(training["warmup_steps"])
    check_every = int(training["budget_check_every_steps"])
    batch_size = int(training["batch_size"])
    last_metrics: dict[str, float] = {}
    started = time.monotonic()

    for step in range(n_steps):
        if step % check_every == 0:
            budget.check()
        learning_rate = float(training["learning_rate"])
        if warmup_steps and step < warmup_steps:
            learning_rate *= (step + 1) / warmup_steps
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        batch = _sample_windows(
            data,
            T=int(cell["T"]),
            batch_size=batch_size,
            sequence_rng=sequence_rng,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=dtype)
            if use_autocast
            else contextlib.nullcontext()
        )
        with autocast_context:
            result = model.train_step(batch)
            if isinstance(result, tuple):
                loss, info = result
                metrics = {"loss": loss, **info}
            else:
                metrics = result
                loss = metrics["loss"]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(training["gradient_clip_norm"]),
        )
        optimizer.step()
        model.post_step()
        last_metrics = {
            key: float(value.detach().item())
            for key, value in metrics.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        }
        last_metrics["gradient_norm"] = float(gradient_norm.detach().item())
        log_every = max(1, n_steps // 5)
        if (step + 1) % log_every == 0 or step + 1 == n_steps:
            print(
                f"  train {cell['task']} T={cell['T']} k={cell['k_pos']} "
                f"s={seed} {step + 1}/{n_steps} "
                f"loss={last_metrics.get('loss', math.nan):.5f}",
                flush=True,
            )

    model.eval()
    return model, {
        "device": device,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "learning_rate": float(training["learning_rate"]),
        "warmup_steps": warmup_steps,
        "precision": precision,
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "last_metrics": last_metrics,
    }


@torch.no_grad()
def extract_overlapping_latents(
    model,
    eval_x: torch.Tensor,
    *,
    T: int,
    d_sae: int,
    batch_size: int = 256,
) -> np.ndarray:
    """Exact paper overlap-average code for a window-level model."""
    device = next(model.parameters()).device
    n_sequences, seq_len, _ = eval_x.shape
    latent_sum = torch.zeros(n_sequences, seq_len, d_sae)
    counts = torch.zeros(n_sequences, seq_len)
    for start in range(seq_len - T + 1):
        windows = eval_x[:, start : start + T]
        for batch_start in range(0, n_sequences, batch_size):
            batch = windows[batch_start : batch_start + batch_size].to(device)
            code = model.encode(batch)
            if code.ndim == 3:
                if code.shape[1] != 1:
                    raise ValueError(
                        f"Spectral v1 returned unexpected code {tuple(code.shape)}"
                    )
                code = code[:, 0]
            code_cpu = code.float().cpu()
            batch_end = batch_start + code_cpu.shape[0]
            latent_sum[
                batch_start:batch_end,
                start : start + T,
            ] += code_cpu[:, None, :]
            counts[
                batch_start:batch_end,
                start : start + T,
            ] += 1
    averaged = latent_sum / counts.unsqueeze(-1).clamp(min=1)
    return averaged.reshape(-1, d_sae).numpy()


def denoising_probe_metrics(
    model,
    eval_data: PaperData,
    *,
    T: int,
    d_sae: int,
) -> dict[str, Any]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    code = extract_overlapping_latents(
        model,
        eval_data.x,
        T=T,
        d_sae=d_sae,
    )
    if eval_data.support is None or eval_data.hidden_support is None:
        raise ValueError("denoising data is missing supports")
    support = (
        eval_data.support.permute(0, 2, 1)
        .reshape(-1, 20)
        .cpu()
        .numpy()
    )
    hidden = (
        eval_data.hidden_support.permute(0, 2, 1)
        .reshape(-1, 20)
        .cpu()
        .numpy()
    )
    split = int(0.8 * code.shape[0])
    code_train, code_test = code[:split], code[split:]
    local_values: list[float] = []
    global_values: list[float] = []
    for feature in range(20):
        local_probe = Ridge(alpha=1.0).fit(
            code_train,
            support[:split, feature],
        )
        local_values.append(
            float(
                r2_score(
                    support[split:, feature],
                    local_probe.predict(code_test),
                )
            )
        )
        global_probe = Ridge(alpha=1.0).fit(
            code_train,
            hidden[:split, feature],
        )
        global_values.append(
            float(
                r2_score(
                    hidden[split:, feature],
                    global_probe.predict(code_test),
                )
            )
        )
    return {
        "lp_mean_local_r2": statistics.fmean(local_values),
        "lp_mean_global_r2": statistics.fmean(global_values),
        "lp_ratio": statistics.fmean(global_values)
        / max(statistics.fmean(local_values), 1e-12),
        "lp_local_r2s": local_values,
        "lp_global_r2s": global_values,
    }


def _feature_recovery_auc(
    decoder: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    decoder_columns = decoder.T
    truth_columns = targets.T
    decoder_normed = decoder_columns / decoder_columns.norm(
        dim=0,
        keepdim=True,
    ).clamp(min=1e-8)
    truth_normed = truth_columns / truth_columns.norm(
        dim=0,
        keepdim=True,
    ).clamp(min=1e-8)
    similarities = (decoder_normed.T @ truth_normed).abs()
    maxima = similarities.max(dim=0).values.cpu().numpy()
    thresholds = np.linspace(0, 1, 50)
    curve = np.array([(maxima >= threshold).mean() for threshold in thresholds])
    return {
        "auc": float(np.trapezoid(curve, thresholds)),
        "mean_max_cos": float(maxima.mean()),
        "frac_recovered_90": float((maxima >= 0.9).mean()),
        "frac_recovered_80": float((maxima >= 0.8).mean()),
    }


def coupling_metrics(model, data: PaperData) -> dict[str, Any]:
    if data.hidden_features is None:
        raise ValueError("coupling data is missing hidden features")
    decoder = model.decoder_directions().detach().cpu().float()
    norms = decoder.norm(dim=1)
    raw = _feature_recovery_auc(decoder, data.hidden_features.cpu())
    # Non-DC DCT atoms have exactly zero time mean. Discard cancellation
    # residue before cosine normalization so it cannot become an arbitrary
    # unit direction. This is the scientifically meaningful static/DC gAUC.
    valid = norms > 1e-5
    if not bool(valid.any()):
        raise RuntimeError("no nonzero time-mean decoder directions")
    filtered = _feature_recovery_auc(
        decoder[valid],
        data.hidden_features.cpu(),
    )
    target_rank = int(
        torch.linalg.matrix_rank(data.hidden_features.cpu(), tol=1e-6).item()
    )
    return {
        "gauc": filtered["auc"],
        "gauc_paper_raw": raw["auc"],
        "g_mean_max_cos": filtered["mean_max_cos"],
        "g_frac_recovered_90": filtered["frac_recovered_90"],
        "g_frac_recovered_80": filtered["frac_recovered_80"],
        "decoder_direction_count": int(decoder.shape[0]),
        "nonzero_time_mean_direction_count": int(valid.sum().item()),
        "hidden_target_rank": target_rank,
        "zero_direction_tolerance": 1e-5,
    }


def evaluate_cell(
    model,
    cell: dict[str, Any],
    eval_data: PaperData | None,
    train_data: PaperData,
) -> dict[str, Any]:
    if cell["task"] == "denoising":
        if eval_data is None:
            raise ValueError("denoising requires independent eval data")
        return denoising_probe_metrics(
            model,
            eval_data,
            T=int(cell["T"]),
            d_sae=int(cell["d_sae"]),
        )
    return coupling_metrics(model, train_data)


def _best_complete_cell(
    rows: list[dict[str, Any]],
    *,
    task: str,
    metric: str,
    seeds: set[int],
) -> dict[str, Any] | None:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("task") != task or row.get("status") != "ok":
            continue
        grouped.setdefault((int(row["T"]), int(row["k_pos"])), []).append(row)
    eligible = []
    for key, values in grouped.items():
        by_seed = {int(row["seed"]): row for row in values}
        if set(by_seed) != seeds:
            continue
        ordered = [by_seed[seed] for seed in sorted(seeds)]
        eligible.append((key, ordered))
    if not eligible:
        return None
    (T, k_pos), selected = max(
        eligible,
        key=lambda item: statistics.fmean(
            float(row["metrics"][metric]) for row in item[1]
        ),
    )
    values = [float(row["metrics"][metric]) for row in selected]
    return {
        "T": T,
        "k_pos": k_pos,
        "n": len(values),
        "seeds": sorted(seeds),
        "seed_values": values,
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def write_summary(
    config: dict[str, Any],
    results_dir: Path,
    *,
    plan: dict[str, Any],
) -> dict[str, Any]:
    latest = latest_rows(results_dir / "results.jsonl")
    rows = list(latest.values())
    expected = enumerate_cells(config, smoke=False)
    expected_ids = {cell["cell_id"] for cell in expected}
    observed_ids = {
        str(row["cell_id"])
        for row in rows
        if not row.get("smoke")
    }
    failed = [
        row
        for row in rows
        if not row.get("smoke") and row.get("status") == "failed"
    ]
    seeds = {int(seed) for seed in config["seeds"]}
    best = {}
    for task_name, task in config["tasks"].items():
        best[task_name] = _best_complete_cell(
            rows,
            task=task_name,
            metric=str(task["metric"]),
            seeds=seeds,
        )
    complete = (
        expected_ids == observed_ids
        and not failed
        and all(value is not None for value in best.values())
    )
    summary = {
        "schema_version": 1,
        "run_name": config["run_name"],
        "generated_at": _utc_now(),
        "complete": complete,
        "expected_cells": len(expected_ids),
        "observed_cells": len(observed_ids),
        "missing_cell_ids": sorted(expected_ids - observed_ids),
        "unexpected_cell_ids": sorted(observed_ids - expected_ids),
        "failed": [
            {
                "cell_id": row["cell_id"],
                "task": row["task"],
                "T": row["T"],
                "k_pos": row["k_pos"],
                "seed": row["seed"],
                "error": row.get("error"),
            }
            for row in failed
        ],
        "best_cells": best,
        "plan": plan,
        "coupling_metric_note": (
            "gauc filters time-mean decoder directions with norm <=1e-5; "
            "gauc_paper_raw is retained per row for sensitivity. "
            "At np10 the hidden-direction target matrix has rank 1."
        ),
    }
    _atomic_json(results_dir / "summary.json", summary)
    return summary


def run(config: dict[str, Any], results_dir: Path, *, smoke: bool) -> int:
    plan = build_plan(config, smoke=smoke)
    if not plan["within_cost_plan"] or not plan["within_time_plan"]:
        raise RuntimeError(f"refusing out-of-plan run: {plan}")
    results_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _stable_id(config, length=64)
    _atomic_json(results_dir / "frozen_config.json", config)
    _atomic_json(results_dir / "plan.json", plan)
    budget = BudgetGuard(config, results_dir, config_hash=config_hash)
    status = "complete"
    data_cache: dict[str, PaperData] = {}
    eval_cache: dict[str, PaperData] = {}
    try:
        completed = latest_rows(results_dir / "results.jsonl")
        for cell in enumerate_cells(config, smoke=smoke):
            prior = completed.get(cell["cell_id"])
            if prior and prior.get("status") == "ok":
                print(f"[resume] {cell['cell_id']}", flush=True)
                continue
            task = str(cell["task"])
            if task not in data_cache:
                if task == "denoising":
                    data_cache[task] = paper_markov_data(
                        n_seqs=4096,
                        seed=int(config["tasks"][task]["train_data_seed"]),
                    )
                    eval_cache[task] = paper_markov_data(
                        n_seqs=int(
                            config["tasks"][task]["eval_n_sequences"]
                        ),
                        seed=int(config["tasks"][task]["eval_data_seed"]),
                    )
                else:
                    data_cache[task] = paper_coupling_data(
                        seed=int(config["tasks"][task]["train_data_seed"]),
                    )
            model = None
            started = time.monotonic()
            print(
                f"[cell] {task} T={cell['T']} k={cell['k_pos']} "
                f"seed={cell['seed']} steps={cell['n_steps']} "
                f"id={cell['cell_id']}",
                flush=True,
            )
            try:
                model, training = train_cell(
                    config,
                    cell,
                    data_cache[task],
                    budget,
                )
                metrics = evaluate_cell(
                    model,
                    cell,
                    eval_cache.get(task),
                    data_cache[task],
                )
                row = {
                    "schema_version": 1,
                    **cell,
                    "status": "ok",
                    "metrics": metrics,
                    "primary_value": float(metrics[cell["metric"]]),
                    "training": training,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "estimated_total_spend_usd": budget.update(),
                    "completed_at": _utc_now(),
                }
            except StopRequested:
                raise
            except Exception as exc:
                row = {
                    "schema_version": 1,
                    **cell,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-5000:],
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "completed_at": _utc_now(),
                }
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            _append_jsonl(results_dir / "results.jsonl", row)
            completed[cell["cell_id"]] = row
            print(
                f"  -> {row['status']} "
                f"{row.get('primary_value', row.get('error', ''))}",
                flush=True,
            )
            if row["status"] != "ok":
                raise RuntimeError(
                    f"cell {cell['cell_id']} failed: {row.get('error')}"
                )
        if not smoke:
            write_summary(config, results_dir, plan=plan)
        return 0
    except StopRequested as exc:
        status = "deadline"
        print(f"[stop] {exc}", flush=True)
        if not smoke:
            write_summary(config, results_dir, plan=plan)
        return 0
    except Exception:
        status = "failed"
        if not smoke:
            write_summary(config, results_dir, plan=plan)
        raise
    finally:
        budget.finish(status)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--mode",
        choices=("plan", "smoke", "full"),
        default="plan",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = load_config(args.config)
    if args.mode == "plan":
        print(json.dumps(build_plan(config), indent=2, sort_keys=True))
        return 0
    return run(config, args.results_dir, smoke=args.mode == "smoke")


if __name__ == "__main__":
    raise SystemExit(main())
