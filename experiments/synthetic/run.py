"""Entry point for § 4 — Synthetic feature recovery.

The dispatcher calls ``run(args, extra)`` with the parsed CLI args. We
parse a small set of sweep flags + delegate to ``run_experiment`` /
``run_sweep``.

CLI conventions (also documented in ``run.py``):

    python run.py synthetic --arch <name> --seed <int>
                            --datasource <key>
                            [--k-pos <int>] [--n-steps <int>]
                            [--smoke]

Smoke mode: tiny datasource (``synth_smoke``), n_steps=10, n=1 seed.
The result row carries ``eval_cfg.smoke=True`` and is filtered out of
paper aggregates.
"""

from __future__ import annotations

import argparse
import os

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig


def _parse(extra: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="run.py synthetic")
    p.add_argument("--k-pos", type=int, default=None)
    p.add_argument("--d-sae", type=int, default=None,
                   help="Override dictionary size (else per-section default).")
    p.add_argument("--T", type=int, default=None,
                   help="Override the architecture window length T (window archs).")
    p.add_argument("--eval-window-l", type=int, default=None,
                   help="Common tiled eval-window length L (see guidance § 4).")
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args(extra)


def run(args, extra: list[str]) -> int:
    """Called by the dispatcher with the shared args + extra CLI tokens."""
    sub = _parse(extra)

    # Defaults: smoke uses tiny dims; full run uses paper-canonical.
    if args.datasource is None:
        args.datasource = "synth_smoke" if args.smoke else "toy_coupled_K10_M20_d256"

    if args.arch is None:
        args.arch = "txc_base"
        print(f"[synthetic] --arch unset; defaulting to {args.arch}")

    # Build training cfg.
    if args.smoke:
        n_steps = sub.n_steps if sub.n_steps is not None else 10
        batch_size = sub.batch_size if sub.batch_size is not None else 32
        buffer_tokens = 4096
    else:
        n_steps = sub.n_steps if sub.n_steps is not None else 30_000
        batch_size = sub.batch_size if sub.batch_size is not None else 1024
        buffer_tokens = 2_000_000

    override: dict | None = None
    if sub.k_pos is not None:
        override = {"k_pos": int(sub.k_pos)}
    elif args.smoke:
        # Smoke datasource has d_sae=16; default k_pos=20 × T=5 = 100 would
        # exceed d_sae. Drop k_pos to 2 for the smoke profile.
        override = {"k_pos": 2}
    if sub.d_sae is not None:
        override = {**(override or {}), "d_sae": int(sub.d_sae)}
    if sub.T is not None:
        override = {**(override or {}), "T": int(sub.T)}

    training_cfg = TrainingConfig(
        n_steps=n_steps,
        batch_size=batch_size,
        buffer_tokens=buffer_tokens,
        arch_hparams_override=override,
    )

    eval_cfg = {"smoke": args.smoke}
    if sub.k_pos is not None:
        eval_cfg["k_pos"] = int(sub.k_pos)
    if sub.eval_window_l is not None:
        eval_cfg["eval_window_L"] = int(sub.eval_window_l)

    print(f"[synthetic] arch={args.arch} seed={args.seed} ds={args.datasource} "
          f"n_steps={n_steps} batch_size={batch_size} smoke={args.smoke}")

    result = run_experiment(
        experiment="synthetic",
        arch_name=args.arch,
        seed=args.seed,
        datasource_name=args.datasource,
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,
        agent=os.environ.get("AGENT_NAME"),
        allow_dirty=args.allow_dirty,
    )

    print(f"\n[synthetic] DONE")
    print(f"  train_key:    {result.train_key}  (cached={result.train_cached})")
    print(f"  eval_key:     {result.eval_key}   (cached={result.eval_cached})")
    print(f"  metrics:")
    for k, v in result.row.metrics.items():
        print(f"    {k:<22} = {v:.4f}")
    return 0
