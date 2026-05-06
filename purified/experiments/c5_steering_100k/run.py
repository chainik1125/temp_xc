"""C5 driver — replicates agent_steer's setup at n_steps=100_000.

Per decisions.md § 13 the 100K copy-sweep is a literal copy of agent_steer's
C5 sweep with the only knob change being ``n_steps=100_000`` (vs 20_000).
Re-uses agent_steer's plumbing verbatim via imports — same DATASOURCE,
EVAL_PROTOCOL_VERSION, ``my_train_fn`` / ``_make_eval_fn`` / ``run_one_cell``
in :mod:`experiments.c5_steering.run`. The only edit is the
``n_steps=100_000`` override threaded through ``run_one_cell``'s
existing ``n_steps`` knob (which Pydantic-copies the base TrainingConfig
returned by agent_steer's ``_real_training_cfg`` and bumps n_steps).

The runner-derived ``train_key`` includes ``n_steps`` so the 100K cells
land on distinct keys from the 20K cells — no collision with agent_steer's
results.

**Thread cap (perf, not numerics)**: empirically 104-thread torch default
gives 1.9 steps/sec on this H100 pod's CPU because random fancy indexing
on the 14 GB Gemma fp16 cache thrashes per-thread cache lines; 32 threads
give 3.3 steps/sec. The kernel & math path are identical (same seed,
same fp32 conversion, same autocast) — checkpoints are bit-identical
to a 104-thread run. Set ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` plus
``torch.set_num_threads`` early so every imported module sees them.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TQDM_DISABLE", "1")
# Cap torch / OpenMP / MKL threads BEFORE importing torch — see module
# docstring. Idempotent: env vars set externally take precedence (use
# ``OMP_NUM_THREADS=N`` to override at launch time).
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")

import torch  # noqa: E402  # after env-var setup

_TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", os.environ["OMP_NUM_THREADS"]))
torch.set_num_threads(_TORCH_THREADS)
torch.set_num_interop_threads(_TORCH_THREADS)

from experiments.c5_steering.run import (
    DEFAULT_ARCHS,
    DEFAULT_SEEDS,
    run_one_cell,
)
from temp_bench.case_studies.steering import (
    DEFAULT_COH_THRESHOLDS,
    DEFAULT_STRENGTHS,
)


N_STEPS_100K = 100_000


def main() -> None:
    ap = argparse.ArgumentParser(
        description="C5 100K copy-sweep — agent_steer plumbing × n_steps=100_000."
    )
    ap.add_argument("--archs", nargs="*", default=list(DEFAULT_ARCHS),
                    choices=list(DEFAULT_ARCHS))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    ap.add_argument("--protocol", choices=("v7", "pp"), default="v7")
    ap.add_argument("--n-concepts", type=int, default=30)
    ap.add_argument("--strengths", type=float, nargs="*", default=None)
    ap.add_argument("--coh-thresholds", type=float, nargs="*", default=None)
    ap.add_argument("--n-steps", type=int, default=N_STEPS_100K,
                    help="Override TrainingConfig.n_steps. Default 100000 "
                         "(the 100K copy-sweep target). Use a small value "
                         "(e.g. 200) for fast smoke tests.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tag the cell's leaderboard row with smoke=true so "
                         "analysis.py filters it out of paper aggregates.")
    ap.add_argument("--pre-test-only", action="store_true",
                    help="Health-check V7 on TXC-pro (5 concepts × 1 strength).")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    strengths = (
        tuple(args.strengths) if args.strengths else DEFAULT_STRENGTHS
    )
    coh_thresholds = (
        tuple(args.coh_thresholds) if args.coh_thresholds
        else DEFAULT_COH_THRESHOLDS
    )

    print(
        f"[c5_100k] sweep n_steps={args.n_steps} archs={list(args.archs)} "
        f"seeds={list(args.seeds)} protocol={args.protocol} "
        f"smoke={args.smoke}",
        flush=True,
    )

    if args.pre_test_only:
        run_one_cell(
            arch_name="txc_pro",
            seed=42,
            protocol=args.protocol,
            n_concepts=5,
            strengths=(strengths[len(strengths) // 2],),
            coh_thresholds=coh_thresholds,
            n_steps=args.n_steps,
            smoke=True,
            force_train=args.force_train,
            force_eval=args.force_eval,
        )
        return

    for arch in args.archs:
        for seed in args.seeds:
            print(
                f"[c5_100k] launching cell arch={arch} seed={seed} "
                f"n_steps={args.n_steps}",
                flush=True,
            )
            run_one_cell(
                arch_name=arch,
                seed=seed,
                protocol=args.protocol,
                n_concepts=args.n_concepts,
                strengths=strengths,
                coh_thresholds=coh_thresholds,
                n_steps=args.n_steps,
                smoke=args.smoke,
                force_train=args.force_train,
                force_eval=args.force_eval,
            )


if __name__ == "__main__":
    main()
