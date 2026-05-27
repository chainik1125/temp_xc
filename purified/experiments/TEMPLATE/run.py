"""Template entry point — copy to add a new experiment.

To add a new experiment ``my_experiment``:

    cp -r experiments/TEMPLATE experiments/my_experiment

Then:
1. Edit this file (rename references / set defaults).
2. Add an Evaluator in ``temp_bench/evals/my_experiment.py``
   (subclass :class:`temp_bench.interfaces.evaluator.Evaluator`).
3. Register it in :data:`temp_bench.core.runner._EVALUATOR_REGISTRY`.
4. (Optional) Add a canonical sweep config to
   ``configs/experiments.yaml`` so ``python run.py reproduce my_experiment``
   works.

Run a smoke cell:

    python run.py my_experiment --arch txc_base --seed 0 --smoke
"""

from __future__ import annotations

import argparse
import os

from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig


def _parse(extra):
    p = argparse.ArgumentParser(prog="run.py <my_experiment>")
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    # Add your task-specific knobs here.
    return p.parse_args(extra)


def run(args, extra):
    sub = _parse(extra)

    # Reasonable defaults; override per task.
    if args.datasource is None:
        args.datasource = "synth_smoke" if args.smoke else "toy_coupled_K10_M20_d256"
    if args.arch is None:
        args.arch = "txc_base"

    n_steps = sub.n_steps if sub.n_steps is not None else (10 if args.smoke else 30_000)
    batch_size = sub.batch_size if sub.batch_size is not None else (32 if args.smoke else 1024)

    training_cfg = TrainingConfig(n_steps=n_steps, batch_size=batch_size)

    result = run_experiment(
        experiment="my_experiment",   # ← change to your registered name
        arch_name=args.arch,
        seed=args.seed,
        datasource_name=args.datasource,
        training_cfg=training_cfg,
        eval_cfg={"smoke": args.smoke},
        agent=os.environ.get("AGENT_NAME"),
        allow_dirty=args.allow_dirty,
    )
    print(f"DONE — train_key={result.train_key} eval_key={result.eval_key}")
    return 0
