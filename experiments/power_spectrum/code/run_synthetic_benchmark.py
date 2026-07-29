"""Matched-budget synthetic benchmark for the power-spectrum experiment.

This runner deliberately does not use the repository-wide leaderboard or
checkpoint store.  It reuses the canonical synthetic generators, architecture
contract, WindowBuffer, training step, and SyntheticRecovery evaluator, while
keeping every new artifact below ``experiments/power_spectrum`` (or an explicit
``--results-dir`` such as a mounted Modal volume).

The safe overnight path is:

    python -m experiments.power_spectrum.code.run_synthetic_benchmark \
        --config experiments/power_spectrum/configs/overnight.json \
        --mode overnight

``overnight`` runs construction smoke tests, one-seed technical gates, and
only then the three-seed benchmark.  Results are append-only JSONL; model and
optimizer state is checkpointed atomically; a wall-clock deadline and a
conservative cost ledger stop the run before the configured budget.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import gc
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
POWER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = POWER_ROOT / "configs" / "overnight.json"
DEFAULT_RESULTS = POWER_ROOT / "results" / "overnight"
PHASE_ORDER = ("smoke", "gate", "full")


class StopRequested(RuntimeError):
    """The cost, session, or per-cell deadline was reached safely."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _stable_id(value: Any, length: int = 16) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()[:length]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Append one complete line under an advisory lock."""
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
    with open(path) as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}: invalid JSON on line {line_no}: {exc}") from exc
    return rows


def latest_results(path: Path) -> dict[str, dict[str, Any]]:
    """Return the last terminal record for every cell id."""
    out: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if "cell_id" in row:
            out[row["cell_id"]] = row
    return out


def load_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    cfg = json.loads(path.read_text())
    required = {
        "schema_version",
        "run_name",
        "budget",
        "planning",
        "training",
        "models",
        "tasks",
        "phases",
        "gates",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"{path}: missing config keys {sorted(missing)}")
    if cfg["schema_version"] != 1:
        raise ValueError(f"unsupported schema_version={cfg['schema_version']!r}")

    model_names = [m["name"] for m in cfg["models"]]
    task_names = [t["name"] for t in cfg["tasks"]]
    if len(model_names) != len(set(model_names)):
        raise ValueError("model names must be unique")
    if len(task_names) != len(set(task_names)):
        raise ValueError("task names must be unique")
    if not {"txc_pre", "txc_post"}.issubset(model_names):
        raise ValueError("matched benchmark requires both txc_pre and txc_post")
    for model in cfg["models"]:
        if not model.get("implementation_version"):
            raise ValueError(f"{model['name']}: implementation_version is required")
    for phase in PHASE_ORDER:
        if phase not in cfg["phases"]:
            raise ValueError(f"missing phase {phase!r}")
    for task in cfg["tasks"]:
        if int(task["eval_window_L"]) % int(task["T"]):
            raise ValueError(f"{task['name']}: eval_window_L must be divisible by T")
        if int(task["d_sae"]) < int(task["k_pos"]) * int(task["T"]):
            raise ValueError(f"{task['name']}: d_sae must support the pre/spectral k_pos*T budget")
    return cfg


def _select_names(items: list[dict[str, Any]], selected: Any) -> list[dict[str, Any]]:
    if selected in (None, "all"):
        return list(items)
    wanted = set(selected)
    unknown = wanted - {item["name"] for item in items}
    if unknown:
        raise ValueError(f"unknown configured names: {sorted(unknown)}")
    return [item for item in items if item["name"] in wanted]


def _training_identity(
    cfg: dict[str, Any],
    model: dict[str, Any],
    task: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Identity shared by smoke/gate/full so later phases continue training."""
    training = cfg["training"]
    return {
        "run_name": cfg["run_name"],
        "model": model["name"],
        "class_path": model["class_path"],
        "class_candidates": model.get("class_candidates", []),
        "implementation_version": model["implementation_version"],
        "model_hparams": model.get("hparams", {}),
        "task": task["name"],
        "datasource": task["datasource"],
        "T": int(task["T"]),
        "d_sae": int(task["d_sae"]),
        "k_pos": int(task["k_pos"]),
        "seed": int(seed),
        "learning_rate": float(training["learning_rate"]),
        "warmup_steps": int(training["warmup_steps"]),
        "batch_tokens": int(training["batch_tokens"]),
        "buffer_tokens": int(training["buffer_tokens"]),
    }


def enumerate_cells(
    cfg: dict[str, Any],
    phase: str,
    *,
    promoted_models: set[str] | None = None,
) -> list[dict[str, Any]]:
    phase_cfg = cfg["phases"][phase]
    models = _select_names(cfg["models"], phase_cfg.get("models", "all"))
    tasks = _select_names(cfg["tasks"], phase_cfg.get("tasks", "all"))
    if promoted_models is not None:
        models = [m for m in models if m["name"] in promoted_models]
    cells: list[dict[str, Any]] = []
    for task in tasks:
        target_steps = int(phase_cfg.get("n_steps", task["n_steps"]))
        for model in models:
            for seed in phase_cfg["seeds"]:
                identity = _training_identity(cfg, model, task, int(seed))
                cell_core = {
                    **identity,
                    "phase": phase,
                    "target_steps": target_steps,
                    "primary_metric": task["primary_metric"],
                    "eval_window_L": int(task["eval_window_L"]),
                    "max_cell_minutes": float(
                        phase_cfg.get("max_cell_minutes", task["max_cell_minutes"])
                    ),
                }
                cells.append(
                    {
                        **cell_core,
                        "training_id": _stable_id(identity),
                        "cell_id": _stable_id(cell_core),
                        "model_spec": model,
                        "task_spec": task,
                    }
                )
    return cells


def phases_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "smoke":
        return ("smoke",)
    if mode == "gate":
        return ("smoke", "gate")
    if mode in {"full", "overnight"}:
        return PHASE_ORDER
    raise ValueError(f"unknown mode {mode!r}")


def build_plan(cfg: dict[str, Any], mode: str = "overnight") -> dict[str, Any]:
    """Plan the worst case (all variants promoted) without importing torch."""
    cells = [cell for phase in phases_for_mode(mode) for cell in enumerate_cells(cfg, phase)]
    # A training identity is continued across phases, so count only its largest
    # target. Evaluations happen at every gate boundary.
    max_steps: dict[str, int] = {}
    for cell in cells:
        max_steps[cell["training_id"]] = max(
            max_steps.get(cell["training_id"], 0), int(cell["target_steps"])
        )
    planning = cfg["planning"]
    estimated_seconds = (
        sum(max_steps.values()) / float(planning["estimated_steps_per_second"])
        + len(cells) * float(planning["estimated_eval_seconds_per_cell"])
        + float(planning.get("setup_seconds", 0))
    )
    budget = cfg["budget"]
    effective_rate = float(budget["assumed_usd_per_gpu_hour"]) * float(
        budget["cost_overhead_multiplier"]
    )
    estimated_cost = estimated_seconds / 3600.0 * effective_rate
    usable_budget = float(budget["max_total_usd"]) - float(budget["reserve_usd"])
    return {
        "run_name": cfg["run_name"],
        "mode": mode,
        "phase_cell_counts": {
            phase: len(enumerate_cells(cfg, phase)) for phase in phases_for_mode(mode)
        },
        "total_evaluations": len(cells),
        "unique_training_runs": len(max_steps),
        "total_optimizer_steps": sum(max_steps.values()),
        "estimated_gpu_hours": round(estimated_seconds / 3600.0, 3),
        "effective_assumed_usd_per_hour": round(effective_rate, 3),
        "estimated_cost_usd": round(estimated_cost, 2),
        "usable_budget_usd": round(usable_budget, 2),
        "max_session_hours": float(budget["max_session_hours"]),
        "within_cost_plan": estimated_cost <= usable_budget,
        "within_time_plan": (estimated_seconds / 3600.0 <= float(budget["max_session_hours"])),
        "fairness": cfg.get("fairness", {}),
    }


def _import_model_class(spec: dict[str, Any]):
    """Import the configured class, defensively discovering a renamed v2 class."""
    module_name, configured_name = spec["class_path"].split(":", 1)
    module = importlib.import_module(module_name)
    names = [configured_name, *spec.get("class_candidates", [])]
    for name in names:
        cls = getattr(module, name, None)
        if inspect.isclass(cls):
            return cls, f"{module_name}:{name}"

    # Last-resort discovery stays inside the requested v2 module. It never
    # silently substitutes the repository's v1 architecture for a v2 arm.
    if module_name.endswith("spectral_txc_v2"):
        from temp_bench.interfaces.architecture import TempBenchArch

        candidates = [
            (name, obj)
            for name, obj in vars(module).items()
            if inspect.isclass(obj)
            and obj is not TempBenchArch
            and issubclass(obj, TempBenchArch)
            and obj.__module__ == module.__name__
        ]
        if len(candidates) == 1:
            name, cls = candidates[0]
            return cls, f"{module_name}:{name}"
    raise ImportError(
        f"could not find {configured_name!r} (or candidates "
        f"{spec.get('class_candidates', [])}) in {module_name}"
    )


def model_availability(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for spec in cfg["models"]:
        try:
            cls, resolved = _import_model_class(spec)
            out.append(
                {
                    "name": spec["name"],
                    "available": True,
                    "resolved_class_path": resolved,
                    "arch_version": getattr(cls, "arch_version", None),
                }
            )
        except Exception as exc:
            out.append(
                {
                    "name": spec["name"],
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return out


class BudgetGuard:
    """Persistent conservative cost accounting plus wall-clock cutoffs."""

    def __init__(self, cfg: dict[str, Any], results_dir: Path, config_hash: str):
        self.cfg = cfg["budget"]
        self.path = results_dir / "spend.json"
        self.config_hash = config_hash
        self.session_id = uuid.uuid4().hex[:12]
        self.started_mono = time.monotonic()
        self.started_at = _utc_now()
        self.rate = float(self.cfg["assumed_usd_per_gpu_hour"])
        self.multiplier = float(self.cfg["cost_overhead_multiplier"])
        self.usable_budget = float(self.cfg["max_total_usd"]) - float(self.cfg["reserve_usd"])
        self.max_session_seconds = float(self.cfg["max_session_hours"]) * 3600
        self.ledger = self._load()
        for session in self.ledger["sessions"]:
            if session.get("status") == "running":
                session["status"] = "interrupted"
        prior = sum(float(s.get("estimated_cost_usd", 0)) for s in self.ledger["sessions"])
        if prior >= self.usable_budget:
            raise StopRequested(
                f"prior estimated spend ${prior:.2f} exhausted usable "
                f"${self.usable_budget:.2f} budget"
            )
        self.ledger["sessions"].append(
            {
                "session_id": self.session_id,
                "started_at": self.started_at,
                "last_seen_at": self.started_at,
                "elapsed_seconds": 0.0,
                "estimated_cost_usd": 0.0,
                "status": "running",
            }
        )
        self.update()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            value = json.loads(self.path.read_text())
            if value.get("config_hash") != self.config_hash:
                raise RuntimeError(
                    f"{self.path} belongs to a different config; use a new results dir"
                )
            return value
        return {
            "schema_version": 1,
            "config_hash": self.config_hash,
            "max_total_usd": float(self.cfg["max_total_usd"]),
            "reserve_usd": float(self.cfg["reserve_usd"]),
            "usable_budget_usd": self.usable_budget,
            "assumed_usd_per_gpu_hour": self.rate,
            "cost_overhead_multiplier": self.multiplier,
            "sessions": [],
        }

    def _session(self) -> dict[str, Any]:
        return next(s for s in self.ledger["sessions"] if s["session_id"] == self.session_id)

    def update(self, status: str = "running") -> float:
        elapsed = time.monotonic() - self.started_mono
        session = self._session()
        session.update(
            {
                "last_seen_at": _utc_now(),
                "elapsed_seconds": round(elapsed, 3),
                "estimated_cost_usd": round(elapsed / 3600.0 * self.rate * self.multiplier, 6),
                "status": status,
            }
        )
        total = sum(float(s["estimated_cost_usd"]) for s in self.ledger["sessions"])
        self.ledger["estimated_total_usd"] = round(total, 6)
        self.ledger["remaining_usable_usd"] = round(self.usable_budget - total, 6)
        _atomic_json(self.path, self.ledger)
        return total

    def check(self, *, cell_deadline: float | None = None) -> None:
        elapsed = time.monotonic() - self.started_mono
        total = self.update()
        if elapsed >= self.max_session_seconds:
            raise StopRequested(f"session deadline {self.max_session_seconds / 3600:.2f}h reached")
        if total >= self.usable_budget:
            raise StopRequested(
                f"estimated spend ${total:.2f} reached usable cap ${self.usable_budget:.2f}"
            )
        if cell_deadline is not None and time.monotonic() >= cell_deadline:
            raise StopRequested("per-cell deadline reached")

    def finish(self, status: str) -> None:
        self.update(status=status)


def _model_hparams(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        **cell["model_spec"].get("hparams", {}),
        "d_sae": int(cell["d_sae"]),
        "T": int(cell["T"]),
        "k_pos": int(cell["k_pos"]),
    }


def _checkpoint_path(results_dir: Path, training_id: str) -> Path:
    return results_dir / "checkpoints" / training_id / "state.pt"


def _save_checkpoint(
    path: Path,
    *,
    cell: dict[str, Any],
    model,
    optimizer,
    scheduler,
    step: int,
    resolved_class_path: str,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "training_id": cell["training_id"],
        "resolved_class_path": resolved_class_path,
        "model_hparams": _model_hparams(cell),
        "step": int(step),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "saved_at": _utc_now(),
    }
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _optimizer_to(optimizer, device: str) -> None:
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _build_training(cell: dict[str, Any], cfg: dict[str, Any], results_dir: Path):
    import torch

    from temp_bench.core.config import load_datasource
    from temp_bench.core.trainer import _build_refill_source, _infer_d_in
    from temp_bench.data.window_buffer import WindowBuffer
    from temp_bench.interfaces.architecture import TempBenchArch

    seed = int(cell["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cls, resolved = _import_model_class(cell["model_spec"])
    data_spec = load_datasource(cell["datasource"])
    model = cls(d_in=_infer_d_in(data_spec), **_model_hparams(cell))
    if not isinstance(model, TempBenchArch):
        raise TypeError(f"{resolved} is not a TempBenchArch")
    if model.consumes != "window":
        raise TypeError(f"{resolved} consumes={model.consumes!r}; benchmark requires window")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()
    training = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    warmup_steps = int(training["warmup_steps"])
    scheduler = None
    if warmup_steps:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(1.0, (step + 1) / warmup_steps)
        )

    refill = _build_refill_source(data_spec, seed=seed)
    capacity_seqs = max(1024, int(training["buffer_tokens"]) // 128)
    batch_iter = WindowBuffer(
        refill,
        T=int(cell["T"]),
        capacity_seqs=capacity_seqs,
        refill_threshold=float(training["refill_threshold"]),
        device=device,
        seed=seed,
    )

    start_step = 0
    checkpoint = _checkpoint_path(results_dir, cell["training_id"])
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("training_id") != cell["training_id"]:
            raise RuntimeError(f"checkpoint identity mismatch at {checkpoint}")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        _optimizer_to(optimizer, device)
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload["step"])
        if payload.get("torch_rng") is not None:
            torch.set_rng_state(payload["torch_rng"])
        if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return (
        model,
        optimizer,
        scheduler,
        batch_iter,
        start_step,
        checkpoint,
        resolved,
        device,
    )


def _train_to_target(
    cell: dict[str, Any],
    cfg: dict[str, Any],
    results_dir: Path,
    budget: BudgetGuard,
):
    import torch

    (
        model,
        optimizer,
        scheduler,
        batch_iter,
        start_step,
        checkpoint,
        resolved,
        device,
    ) = _build_training(cell, cfg, results_dir)
    target = int(cell["target_steps"])
    training = cfg["training"]
    checkpoint_every = int(training["checkpoint_every_steps"])
    budget_every = int(training["budget_check_every_steps"])
    batch_size = max(1, int(training["batch_tokens"]) // int(cell["T"]))
    cell_deadline = time.monotonic() + float(cell["max_cell_minutes"]) * 60
    last_metrics: dict[str, float] = {}
    model.train()
    step = start_step
    try:
        while step < target:
            if step % budget_every == 0:
                budget.check(cell_deadline=cell_deadline)
            model.pre_step()
            optimizer.zero_grad(set_to_none=True)
            batch = batch_iter(batch_size)
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
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            model.post_step()
            step += 1
            last_metrics = {
                key: float(value.detach().item())
                for key, value in metrics.items()
                if hasattr(value, "numel") and value.numel() == 1
            }
            if step % checkpoint_every == 0 or step == target:
                _save_checkpoint(
                    checkpoint,
                    cell=cell,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    resolved_class_path=resolved,
                )
            log_every = max(1, min(1000, target // 10 or 1))
            if step % log_every == 0 or step == target:
                print(
                    f"  [train] {cell['model']}/{cell['task']} s{cell['seed']} "
                    f"{step}/{target} loss={last_metrics.get('loss', math.nan):.5f}",
                    flush=True,
                )
    except StopRequested:
        _save_checkpoint(
            checkpoint,
            cell=cell,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            resolved_class_path=resolved,
        )
        raise

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, {
        "start_step": start_step,
        "end_step": step,
        "batch_size": batch_size,
        "batch_tokens": batch_size * int(cell["T"]),
        "checkpoint": str(checkpoint),
        "resolved_class_path": resolved,
        "device": device,
        "parameter_count": n_params,
        "last_train_metrics": last_metrics,
    }


def _evaluate(model, cell: dict[str, Any], *, smoke: bool) -> dict[str, float]:
    from temp_bench.core.config import compute_data_key, load_datasource
    from temp_bench.evals.synthetic_recovery import SyntheticRecovery
    from temp_bench.interfaces.evaluator import EvalSpec

    data_spec = load_datasource(cell["datasource"])
    evaluator = SyntheticRecovery()
    eval_spec = EvalSpec(
        datasource=cell["datasource"],
        data_key=compute_data_key(data_spec),
        smoke=smoke,
        extra={
            "training_seed": int(cell["seed"]),
            "eval_window_L": int(cell["eval_window_L"]),
        },
    )
    metrics = evaluator.eval(model, eval_spec)
    return {key: float(value) for key, value in metrics.items()}


def _clear_cell_state(model: Any | None = None) -> None:
    with contextlib.suppress(Exception):
        from temp_bench.data import synthetic

        synthetic._SYNTHETIC_CACHE.clear()
    del model
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _finite_metrics(metrics: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


def gate_report(
    cfg: dict[str, Any],
    rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gate_cells = enumerate_cells(cfg, "gate")
    by_model: dict[str, list[dict[str, Any]]] = {}
    for cell in gate_cells:
        by_model.setdefault(cell["model"], []).append(cell)
    required = set(cfg["gates"].get("required_models", ["txc_pre", "txc_post"]))
    report: dict[str, Any] = {"models": {}, "promoted_models": [], "abort": False}
    for model_name, cells in by_model.items():
        failures = []
        for cell in cells:
            row = rows.get(cell["cell_id"])
            if row is None or row.get("status") != "ok":
                failures.append(f"{cell['task']}: missing/failed")
                continue
            metrics = row.get("metrics", {})
            primary = cell["primary_metric"]
            if primary not in metrics or not math.isfinite(float(metrics[primary])):
                failures.append(f"{cell['task']}: non-finite/missing {primary}")
            if not _finite_metrics(metrics):
                failures.append(f"{cell['task']}: non-finite metric")
            nmse = metrics.get("nmse")
            if nmse is None or nmse > float(cfg["gates"]["max_nmse"]):
                failures.append(f"{cell['task']}: nmse={nmse}")
            l0 = metrics.get("l0_per_window")
            if l0 is None or l0 <= 0:
                failures.append(f"{cell['task']}: invalid l0_per_window={l0}")
        passed = not failures
        report["models"][model_name] = {"passed": passed, "failures": failures}
        if passed:
            report["promoted_models"].append(model_name)
        elif model_name in required:
            report["abort"] = True
    return report


def _paired_delta(
    rows: list[dict[str, Any]],
    model: str,
    baseline: str,
    metric: str,
) -> list[float]:
    values = {
        (row["model"], row["task"], int(row["seed"])): row["metrics"][metric]
        for row in rows
        if metric in row.get("metrics", {})
    }
    out = []
    for (name, task, seed), value in values.items():
        if name != model:
            continue
        key = (baseline, task, seed)
        if key in values:
            out.append(float(value) - float(values[key]))
    return out


def write_summary(
    cfg: dict[str, Any],
    results_dir: Path,
    gate: dict[str, Any] | None,
) -> dict[str, Any]:
    latest = latest_results(results_dir / "results.jsonl")
    full_rows = [
        row for row in latest.values() if row.get("phase") == "full" and row.get("status") == "ok"
    ]
    task_metric = {task["name"]: task["primary_metric"] for task in cfg["tasks"]}
    aggregates: list[dict[str, Any]] = []
    for task in cfg["tasks"]:
        metric = task_metric[task["name"]]
        for model in cfg["models"]:
            selected = [
                row
                for row in full_rows
                if row["task"] == task["name"]
                and row["model"] == model["name"]
                and metric in row["metrics"]
            ]
            if not selected:
                continue
            values = [float(row["metrics"][metric]) for row in selected]
            entry = {
                "task": task["name"],
                "metric": metric,
                "model": model["name"],
                "n": len(values),
                "mean": sum(values) / len(values),
                "std": (
                    math.sqrt(
                        sum((v - sum(values) / len(values)) ** 2 for v in values)
                        / (len(values) - 1)
                    )
                    if len(values) > 1
                    else 0.0
                ),
                "mean_l0_per_window": sum(
                    float(row["metrics"]["l0_per_window"]) for row in selected
                )
                / len(selected),
                "mean_nmse": sum(float(row["metrics"]["nmse"]) for row in selected) / len(selected),
                "parameter_count": selected[0]["training"]["parameter_count"],
            }
            for baseline in ("txc_pre", "txc_post"):
                deltas = _paired_delta(
                    [r for r in full_rows if r["task"] == task["name"]],
                    model["name"],
                    baseline,
                    metric,
                )
                entry[f"delta_vs_{baseline}"] = sum(deltas) / len(deltas) if deltas else None
            aggregates.append(entry)
    summary = {
        "schema_version": 1,
        "run_name": cfg["run_name"],
        "generated_at": _utc_now(),
        "fairness": cfg.get("fairness", {}),
        "gate_report": gate,
        "n_full_rows": len(full_rows),
        "aggregates": aggregates,
    }
    _atomic_json(results_dir / "summary.json", summary)
    return summary


def _run_cell(
    cell: dict[str, Any],
    cfg: dict[str, Any],
    results_dir: Path,
    budget: BudgetGuard,
) -> dict[str, Any]:
    started = time.monotonic()
    model = None
    print(
        f"[{cell['phase']}] {cell['model']}/{cell['task']}/s{cell['seed']} "
        f"target={cell['target_steps']} id={cell['cell_id']}",
        flush=True,
    )
    try:
        model, training = _train_to_target(cell, cfg, results_dir, budget)
        budget.check()
        metrics = _evaluate(model, cell, smoke=cell["phase"] != "full")
        if not _finite_metrics(metrics):
            raise FloatingPointError("evaluator returned non-finite metrics")
        primary = cell["primary_metric"]
        if primary not in metrics:
            raise KeyError(f"evaluator did not return primary metric {primary!r}")
        row = {
            "schema_version": 1,
            "cell_id": cell["cell_id"],
            "training_id": cell["training_id"],
            "status": "ok",
            "phase": cell["phase"],
            "model": cell["model"],
            "task": cell["task"],
            "datasource": cell["datasource"],
            "seed": int(cell["seed"]),
            "T": int(cell["T"]),
            "d_sae": int(cell["d_sae"]),
            "k_pos": int(cell["k_pos"]),
            "target_steps": int(cell["target_steps"]),
            "primary_metric": primary,
            "primary_value": metrics[primary],
            "metrics": metrics,
            "training": training,
            "fairness_role": cell["model_spec"].get("fairness_role"),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "estimated_total_spend_usd": budget.update(),
            "completed_at": _utc_now(),
        }
    except StopRequested as exc:
        row = {
            "schema_version": 1,
            "cell_id": cell["cell_id"],
            "training_id": cell["training_id"],
            "status": "deferred",
            "phase": cell["phase"],
            "model": cell["model"],
            "task": cell["task"],
            "seed": int(cell["seed"]),
            "error": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "completed_at": _utc_now(),
        }
    except Exception as exc:
        row = {
            "schema_version": 1,
            "cell_id": cell["cell_id"],
            "training_id": cell["training_id"],
            "status": "failed",
            "phase": cell["phase"],
            "model": cell["model"],
            "task": cell["task"],
            "seed": int(cell["seed"]),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "completed_at": _utc_now(),
        }
    finally:
        _clear_cell_state(model)
    _append_jsonl(results_dir / "results.jsonl", row)
    print(
        f"  -> {row['status']} ({row['elapsed_seconds']:.1f}s)"
        + (
            f" {cell['primary_metric']}={row.get('primary_value', math.nan):+.4f}"
            if row["status"] == "ok"
            else f" {row.get('error', '')}"
        ),
        flush=True,
    )
    return row


def run(cfg: dict[str, Any], results_dir: Path, mode: str) -> int:
    plan = build_plan(cfg, mode)
    if not plan["within_cost_plan"]:
        raise RuntimeError(
            f"planned cost ${plan['estimated_cost_usd']:.2f} exceeds usable "
            f"${plan['usable_budget_usd']:.2f}"
        )
    if not plan["within_time_plan"]:
        raise RuntimeError(
            f"planned {plan['estimated_gpu_hours']:.2f}h exceeds session limit "
            f"{plan['max_session_hours']:.2f}h"
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _stable_id(cfg, length=64)
    _atomic_json(results_dir / "frozen_config.json", cfg)
    _atomic_json(results_dir / "plan.json", plan)
    budget = BudgetGuard(cfg, results_dir, config_hash)
    gate: dict[str, Any] | None = None
    status = "complete"
    try:
        for phase in phases_for_mode(mode):
            completed = latest_results(results_dir / "results.jsonl")
            promoted: set[str] | None = None
            if phase == "full":
                gate = gate_report(cfg, completed)
                _atomic_json(results_dir / "gate_report.json", gate)
                if gate["abort"]:
                    raise RuntimeError(
                        "required baseline failed technical gates; full benchmark aborted"
                    )
                promoted = set(gate["promoted_models"])
            for cell in enumerate_cells(cfg, phase, promoted_models=promoted):
                existing = completed.get(cell["cell_id"])
                if existing and existing.get("status") == "ok":
                    print(f"[resume] skip completed {cell['cell_id']}", flush=True)
                    continue
                row = _run_cell(cell, cfg, results_dir, budget)
                completed[cell["cell_id"]] = row
                if row["status"] == "deferred":
                    status = "deadline"
                    write_summary(cfg, results_dir, gate)
                    return 0
            if phase == "gate":
                gate = gate_report(cfg, latest_results(results_dir / "results.jsonl"))
                _atomic_json(results_dir / "gate_report.json", gate)
        write_summary(cfg, results_dir, gate)
        return 0
    except StopRequested:
        status = "deadline"
        write_summary(cfg, results_dir, gate)
        return 0
    except Exception:
        status = "failed"
        raise
    finally:
        budget.finish(status)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--mode", choices=("smoke", "gate", "full", "overnight"), default="overnight"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = load_config(args.config)
    plan = build_plan(cfg, args.mode)
    if args.dry_run:
        plan["model_availability"] = model_availability(cfg)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if plan["within_cost_plan"] and plan["within_time_plan"] else 2
    return run(cfg, args.results_dir, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
