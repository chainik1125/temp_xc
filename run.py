#!/usr/bin/env python
"""Dispatcher CLI for the TempBench framework.

Usage:

    python run.py <experiment> [args]            # one cell or canonical sweep
    python run.py sweep <sweep.yaml>             # custom grid
    python run.py reproduce [section|all]        # canonical paper runs
    python run.py render-figures                 # leaderboard → paper figures
    python run.py validate                       # framework self-check

Experiments (paper sections):
    synthetic       (§ 4)
    probing         (§ 5.1)
    backtracking    (§ 5.2)
    em              (§ 5.3)
    rlhf            (§ 5.4)

Examples:

    # one cell:
    python run.py synthetic --arch txc_base --seed 42 --datasource toy_coupled_K10_M20

    # canonical sweep for one section:
    python run.py reproduce synthetic

    # custom grid:
    python run.py sweep configs/sweeps/my_grid.yaml

    # smoke test (tiny dims, 10 steps, no real LM):
    python run.py synthetic --arch txc_base --seed 0 --datasource synth_smoke --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def _add_common_args(p: argparse.ArgumentParser) -> None:
    """Args every experiment subcommand accepts."""
    p.add_argument("--arch", type=str, required=False, default=None,
                   help="Architecture registry key (see configs/archs.yaml).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--datasource", type=str, default=None,
                   help="Datasource registry key (see configs/data.yaml).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny-dim smoke run; row is filtered from paper aggregates.")
    p.add_argument("--allow-dirty", action="store_true",
                   help="Permit a dirty working tree (diff hash recorded).")


def _experiment_subparser(sub: argparse._SubParsersAction, name: str) -> argparse.ArgumentParser:
    p = sub.add_parser(name, help=f"Run § {name} experiment.")
    _add_common_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="TempBench dispatcher — single canonical entry point.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Per-section experiment subparsers (extra args delegated to entry-point).
    for name in ("synthetic", "probing", "backtracking", "em", "rlhf"):
        _experiment_subparser(sub, name)

    # Sweep subparser
    p_sweep = sub.add_parser("sweep", help="Run a custom sweep config.")
    p_sweep.add_argument("config", type=Path,
                         help="Path to sweep YAML.")
    p_sweep.add_argument("--allow-dirty", action="store_true")

    # Reproduce subparser
    p_repro = sub.add_parser("reproduce", help="Run canonical paper sweep(s).")
    p_repro.add_argument("section", type=str, nargs="?", default="all",
                         help="Paper section (experiments/explorations/synthetic/probing/.../all).")
    p_repro.add_argument("--allow-dirty", action="store_true")

    # Render subparser
    sub.add_parser("render-figures", help="Render Figs 2-6 from leaderboard.")

    # Validate subparser
    sub.add_parser("validate", help="Self-check: registries + interfaces + cache keys.")

    args, extra = parser.parse_known_args(argv)

    if args.command in {"synthetic", "probing", "backtracking", "em", "rlhf"}:
        return _dispatch_experiment(args, extra)
    elif args.command == "sweep":
        return _dispatch_sweep(args)
    elif args.command == "reproduce":
        return _dispatch_reproduce(args)
    elif args.command == "render-figures":
        return _dispatch_render()
    elif args.command == "validate":
        return _dispatch_validate()

    parser.print_help()
    return 1


# ── Subcommand handlers ───────────────────────────────────────────────


def _dispatch_experiment(args, extra: list[str]) -> int:
    """Forward to ``experiments/<name>/run.py:run(args, extra)``."""
    import importlib
    module_name = f"experiments.{args.command}.run"
    try:
        mod = importlib.import_module(module_name)
    except ImportError as e:
        sys.stderr.write(
            f"[run.py] No entry point for {args.command!r}: {e}\n"
            f"Expected: {module_name}.run(args, extra)\n"
        )
        return 2
    return int(mod.run(args, extra) or 0)


def _dispatch_sweep(args) -> int:
    """Load sweep YAML and call core/runner.run_sweep()."""
    from temp_bench.core.runner import run_sweep
    with open(args.config) as f:
        grid = yaml.safe_load(f)
    if args.allow_dirty:
        grid["allow_dirty"] = True
    results = run_sweep(grid, agent=os.environ.get("AGENT_NAME"))
    n_cached = sum(1 for r in results if r.eval_cached)
    print(f"\n[sweep] {len(results)} cells: {n_cached} cached, "
          f"{len(results) - n_cached} new")
    return 0


def _dispatch_reproduce(args) -> int:
    """Run the canonical sweep config(s) for one or all paper sections."""
    from temp_bench.core.runner import run_sweep
    from temp_bench.core.config import load_experiment, list_experiments

    if args.section == "all":
        targets = list_experiments()
        if not targets:
            sys.stderr.write(
                "[run.py reproduce all] No experiments registered in "
                "configs/experiments.yaml.\n"
            )
            return 2
    else:
        targets = [args.section]

    for name in targets:
        print(f"\n=== reproducing § {name} ===")
        grid = load_experiment(name)
        if args.allow_dirty:
            grid["allow_dirty"] = True
        run_sweep(grid, agent=os.environ.get("AGENT_NAME"))
    return 0


def _dispatch_render() -> int:
    """Render paper figures from the current leaderboard."""
    try:
        from experiments.render_paper_figures import render_all
    except ImportError as e:
        sys.stderr.write(
            f"[run.py render-figures] No renderer: {e}\n"
            "Implement experiments/render_paper_figures.py:render_all()\n"
        )
        return 2
    render_all()
    return 0


def _dispatch_validate() -> int:
    """Self-check: load registries, validate interfaces, smoke cache keys."""
    print("[validate] loading arch registry...")
    from temp_bench.core.config import (
        list_archs, list_datasources, list_experiments,
        load_arch, load_datasource, compute_data_key, import_by_path,
    )
    from temp_bench.interfaces.architecture import TempBenchArch
    from temp_bench.interfaces.evaluator import Evaluator

    errors: list[str] = []

    archs = list_archs()
    print(f"  archs: {len(archs)} registered ({', '.join(archs)})")
    for name in archs:
        try:
            spec = load_arch(name)
            cls = import_by_path(spec.class_path)
            if not (isinstance(cls, type) and issubclass(cls, TempBenchArch)):
                errors.append(f"{name}: class {spec.class_path} is not a TempBenchArch subclass")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    datasources = list_datasources()
    print(f"  datasources: {len(datasources)} registered")
    for name in datasources:
        try:
            spec = load_datasource(name)
            compute_data_key(spec)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")

    experiments = list_experiments()
    print(f"  experiments (canonical sweeps): {len(experiments)} registered")

    print("[validate] checking evaluator registry...")
    from temp_bench.core.runner import _EVALUATOR_REGISTRY
    for exp_name, evaluator_path in _EVALUATOR_REGISTRY.items():
        try:
            cls = import_by_path(evaluator_path)
            if not (isinstance(cls, type) and issubclass(cls, Evaluator)):
                errors.append(f"evaluator {exp_name}: {evaluator_path} is not Evaluator subclass")
        except Exception as e:
            errors.append(f"evaluator {exp_name}: {type(e).__name__}: {e}")

    if errors:
        print(f"\n[validate] FAILED — {len(errors)} issues:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("[validate] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
