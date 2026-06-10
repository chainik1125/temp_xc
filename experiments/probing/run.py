"""Entry point for § 5.1 — Sparse Probing.

The dispatcher calls ``run(args, extra)`` with parsed CLI args. This
thin wrapper parses sweep flags + delegates to ``run_experiment``.

PORT STATUS: routing is in place; the underlying evaluator
(``temp_bench.evals.probing.ProbingEval``) raises
NotImplementedError pending the port from origin/final. The CLI
plumbing is fully wired so the port-in is a focused single change.
"""

from __future__ import annotations

import argparse
import os

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig


def _parse(extra):
    p = argparse.ArgumentParser(prog="run.py probing")
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args(extra)


def run(args, extra):
    sub = _parse(extra)
    if args.datasource is None:
        args.datasource = "synth_smoke" if args.smoke else "gemma_2_2b_it_l13_fineweb_24k128"
    if args.arch is None:
        args.arch = "txc_base"

    if args.smoke:
        n_steps = sub.n_steps if sub.n_steps is not None else 10
        batch_size = sub.batch_size if sub.batch_size is not None else 32
    else:
        n_steps = sub.n_steps if sub.n_steps is not None else 20_000
        batch_size = sub.batch_size if sub.batch_size is not None else 4096

    training_cfg = TrainingConfig(n_steps=n_steps, batch_size=batch_size)

    print(f"[probing] arch={args.arch} seed={args.seed} "
          f"ds={args.datasource} n_steps={n_steps} smoke={args.smoke}")

    result = run_experiment(
        experiment="probing",
        arch_name=args.arch, seed=args.seed,
        datasource_name=args.datasource,
        training_cfg=training_cfg,
        eval_cfg={"smoke": args.smoke},
        agent=os.environ.get("AGENT_NAME"),
        allow_dirty=args.allow_dirty,
    )
    print(f"[probing] DONE — train_key={result.train_key}")
    return 0
