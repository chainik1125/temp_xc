"""The single canonical pathway for every experiment cell.

Two entry points:

- :func:`run_experiment` — one (arch, data, training, eval) tuple → one
  leaderboard row. Idempotent: cache-hit on ``train_key`` skips
  training; cache-hit on ``eval_key`` skips eval entirely.

- :func:`run_sweep` — cross-product over a parameter grid; calls
  ``run_experiment`` per cell, optionally in parallel across GPUs.
  Sweep config schema documented in ``docs/framework_v2.md``.

Both ALWAYS:

1. Capture :class:`CodeVersion` first (refuses dirty unless allowed).
2. Compute deterministic ``(data_key, train_key, eval_key)``.
3. Cache-check at each level.
4. Validate row against Pydantic schemas before writing.
5. Flock-protected append to ``leaderboard.jsonl`` / ``manifest.jsonl``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from temp_bench.core.cache import (
    append_leaderboard,
    checkpoint_exists,
    eval_in_leaderboard,
    find_row,
    now_iso,
)
from temp_bench.core.code_version import capture as capture_code_version
from temp_bench.core.config import (
    compute_data_key,
    compute_eval_key,
    compute_train_key,
    import_by_path,
    load_arch,
    load_datasource,
)
from temp_bench.core.schemas import (
    CodeVersion,
    LeaderboardRow,
    TrainingConfig,
)


# ── Single-cell result ─────────────────────────────────────────────────


@dataclass
class CellResult:
    """Return value of :func:`run_experiment` — what happened with one cell."""

    train_key: str
    eval_key: str
    data_key: str
    train_cached: bool       # True if model existed; False if we trained
    eval_cached: bool        # True if leaderboard had this row; False if we evaluated
    row: LeaderboardRow      # the final row in leaderboard.jsonl


# ── Single-cell runner ─────────────────────────────────────────────────


def run_experiment(
    *,
    # WHAT to run
    experiment: str,                       # paper section name (synthetic, probing, ...)
    arch_name: str,                        # registry key
    seed: int,
    datasource_name: str,
    training_cfg: TrainingConfig | None = None,
    eval_cfg: dict[str, Any] | None = None,

    # WHO is calling
    agent: str | None = None,
    allow_dirty: bool | None = None,
) -> CellResult:
    """Run (or cache-hit on) one experiment cell.

    Args:
        experiment: paper section name. Determines which evaluator runs.
        arch_name: registry key in ``configs/archs.yaml``.
        seed: integer seed.
        datasource_name: registry key in ``configs/data.yaml``.
        training_cfg: schedule + optimizer + arch overrides. Defaults to
            ``TrainingConfig()``.
        eval_cfg: kwargs forwarded to the evaluator's ``eval(spec)`` call.
            Defaults to empty dict.
        agent: optional identifier of the caller (CLI invoker / env).
            Stored in the result row.
        allow_dirty: if True, allow a dirty working tree (records diff).

    Returns:
        :class:`CellResult` with train/eval keys + the leaderboard row.
    """
    training_cfg = training_cfg or TrainingConfig()
    eval_cfg = dict(eval_cfg or {})
    if agent is None:
        agent = os.environ.get("AGENT_NAME")

    # 1) Capture code version (raises on dirty unless allowed).
    code_version = capture_code_version(allow_dirty=allow_dirty)

    # 2) Load registry entries.
    arch_spec = load_arch(arch_name, section=experiment)
    data_spec = load_datasource(datasource_name)

    # Apply training_cfg.arch_hparams_override on the arch spec (so the
    # final hparams reflect cell-level overrides like k_pos sweeps).
    if training_cfg.arch_hparams_override:
        merged = {**arch_spec.hparams, **training_cfg.arch_hparams_override}
        arch_spec = arch_spec.model_copy(update={"hparams": merged})

    # 3) Compute keys.
    data_key = compute_data_key(data_spec)
    train_key = compute_train_key(
        arch=arch_spec,
        seed=seed,
        training_cfg=training_cfg,
        data_key=data_key,
        section=experiment,
    )

    # 4) Resolve the Evaluator from the experiment name.
    evaluator = _resolve_evaluator(experiment)

    eval_key = compute_eval_key(
        train_key=train_key,
        evaluator_name=evaluator.name,
        evaluator_protocol_version=evaluator.protocol_version,
        eval_cfg=eval_cfg,
    )

    # 5) Cache-check evaluations.
    if eval_in_leaderboard(eval_key):
        row = find_row(eval_key)
        return CellResult(
            train_key=train_key,
            eval_key=eval_key,
            data_key=data_key,
            train_cached=True,
            eval_cached=True,
            row=row,
        )

    # 6) Train (or cache-hit).
    train_cached = checkpoint_exists(train_key)
    if train_cached:
        model = _load_checkpoint(arch_spec, train_key, data_spec)
    else:
        from temp_bench.core.trainer import train_arch  # lazy import
        model = train_arch(
            arch_spec=arch_spec,
            data_spec=data_spec,
            seed=seed,
            training_cfg=training_cfg,
            train_key=train_key,
            code_version=code_version,
            agent=agent,
        )

    # 7) Evaluate.
    from temp_bench.interfaces.evaluator import EvalSpec
    # Pass training seed into eval extras so the evaluator can
    # re-materialise the synthetic data with the SAME random
    # realisation the model was trained on. (Otherwise the trained
    # dictionary atoms don't match the eval feature directions.)
    extra = {k: v for k, v in eval_cfg.items() if k != "smoke"}
    extra.setdefault("training_seed", int(seed))
    spec = EvalSpec(
        datasource=datasource_name,
        data_key=data_key,
        smoke=bool(eval_cfg.get("smoke", False)),
        extra=extra,
    )
    metrics = evaluator.eval(model, spec)

    # 8) Build + write row.
    row = LeaderboardRow(
        eval_key=eval_key,
        train_key=train_key,
        data_key=data_key,
        experiment=experiment,
        arch=arch_name,
        arch_version=arch_spec.arch_version,
        seed=int(seed),
        datasource=datasource_name,
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,
        evaluator_name=evaluator.name,
        evaluator_protocol_version=evaluator.protocol_version,
        metrics={k: float(v) for k, v in metrics.items()},
        primary_metric=evaluator.primary_metric(),
        code_version=code_version,
        agent=agent,
        ts=now_iso(),
    )
    append_leaderboard(row)

    return CellResult(
        train_key=train_key,
        eval_key=eval_key,
        data_key=data_key,
        train_cached=train_cached,
        eval_cached=False,
        row=row,
    )


# ── Evaluator resolution ───────────────────────────────────────────────


_EVALUATOR_REGISTRY = {
    "synthetic":    "temp_bench.evals.synthetic_recovery:SyntheticRecovery",
    "freq_bench":   "temp_bench.evals.freq_bench:FreqBenchEval",
    "probing":      "temp_bench.evals.probing:ProbingEval",
    "backtracking": "temp_bench.evals.backtracking:BacktrackingEval",
    "em":           "temp_bench.evals.em:EmergentMisalignmentEval",
    "rlhf":         "temp_bench.evals.rlhf:RLHFEval",
}


def _resolve_evaluator(experiment: str):
    """Map paper section name to Evaluator instance."""
    if experiment not in _EVALUATOR_REGISTRY:
        raise KeyError(
            f"Unknown experiment {experiment!r}. "
            f"Known: {list(_EVALUATOR_REGISTRY)}. "
            "Add a new experiment by extending _EVALUATOR_REGISTRY and "
            "dropping an Evaluator subclass under temp_bench/evals/."
        )
    cls = import_by_path(_EVALUATOR_REGISTRY[experiment])
    return cls()


# ── Checkpoint loading ─────────────────────────────────────────────────


def _load_checkpoint(arch_spec, train_key: str, data_spec):
    """Re-instantiate arch and load weights from disk."""
    from temp_bench.core.config import checkpoint_dir
    from temp_bench.core.trainer import _infer_d_in
    import torch
    path = checkpoint_dir(train_key) / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint missing for train_key={train_key!r} at {path}. "
            "Run with --force-train or restore from HF."
        )
    cls = import_by_path(arch_spec.class_path)
    d_in = _infer_d_in(data_spec)
    model = cls(d_in=d_in, **arch_spec.hparams)
    try:
        from safetensors.torch import load_file
        state = load_file(str(path))
    except Exception:
        state = torch.load(str(path), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ── Sweep runner ───────────────────────────────────────────────────────


def run_sweep(grid: dict[str, Any], *, agent: str | None = None) -> list[CellResult]:
    """Cross-product over a sweep grid; call ``run_experiment`` per cell.

    Grid shape (also documented in ``docs/framework_v2.md``):

        {
            "experiment": "probing",
            "arch":   ["txc_base", "txc_pro", ...],   # or single str
            "seed":   [1, 2, 42],
            "datasource": "gemma_2_2b_it_l13_fineweb_24k128",
            "eval_cfg":   {"k_feat": [5, 10, 20]},     # axis values supported nested
            "training_cfg": {...},                     # constants (broadcast)
            "skip_cached": true,                       # default; idempotent
            "on_failure":  "continue",                 # or "abort"
            "n_parallel":  1,                          # 1 for now (sequential)
        }

    Returns list of CellResult in dispatch order.
    """
    from itertools import product

    # ── Normalise grid axes ──
    AXIS_FIELDS = {"arch", "seed", "datasource"}
    axes: dict[str, list] = {}
    constants: dict[str, Any] = {}

    for k, v in grid.items():
        if k in {"experiment", "skip_cached", "on_failure", "n_parallel"}:
            constants[k] = v
        elif k == "eval_cfg":
            # eval_cfg may have nested lists for sweep
            eval_axes = {}
            eval_constants = {}
            for ek, ev in (v or {}).items():
                if isinstance(ev, list):
                    eval_axes[ek] = ev
                else:
                    eval_constants[ek] = ev
            if eval_axes:
                axes["__eval_axes"] = list(_dict_product(eval_axes))
            constants["__eval_constants"] = eval_constants
        elif k == "training_cfg":
            constants[k] = v
        elif k in AXIS_FIELDS or isinstance(v, list):
            axes[k] = v if isinstance(v, list) else [v]
        else:
            constants[k] = v

    on_failure = constants.get("on_failure", "continue")
    experiment = constants["experiment"]
    skip_cached = constants.get("skip_cached", True)

    results: list[CellResult] = []
    keys = list(axes.keys())
    products = list(product(*axes.values())) if axes else [()]
    total = len(products)

    print(f"[sweep] {experiment}: {total} cells")
    for i, combo in enumerate(products, 1):
        cell = dict(zip(keys, combo))
        eval_axes = cell.pop("__eval_axes", {})
        eval_cfg = {**constants.get("__eval_constants", {}), **eval_axes}

        kwargs = {
            "experiment": experiment,
            "arch_name": cell["arch"],
            "seed": int(cell["seed"]),
            "datasource_name": cell["datasource"],
            "training_cfg": constants.get("training_cfg"),
            "eval_cfg": eval_cfg,
            "agent": agent,
        }
        # Allow per-sweep allow_dirty
        if "allow_dirty" in constants:
            kwargs["allow_dirty"] = constants["allow_dirty"]
        if isinstance(kwargs["training_cfg"], dict):
            kwargs["training_cfg"] = TrainingConfig(**kwargs["training_cfg"])

        cell_repr = f"{cell['arch']}/seed={cell['seed']}"
        if eval_axes:
            cell_repr += "/" + "/".join(f"{k}={v}" for k, v in eval_axes.items())

        try:
            t0 = time.time()
            result = run_experiment(**kwargs)
            dt_ = time.time() - t0
            status = "cached" if result.eval_cached else f"ran ({dt_:.1f}s)"
            print(f"  [{i}/{total}] {cell_repr}: {status}")
            results.append(result)
        except Exception as e:
            print(f"  [{i}/{total}] {cell_repr}: FAILED ({type(e).__name__}: {e})")
            if on_failure == "abort":
                raise

    return results


def _dict_product(d: dict[str, list]):
    """Yield dicts that are the Cartesian product of d's lists."""
    from itertools import product
    keys = list(d.keys())
    for combo in product(*d.values()):
        yield dict(zip(keys, combo))
