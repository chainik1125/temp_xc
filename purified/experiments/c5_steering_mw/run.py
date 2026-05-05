"""C5 multi-window deployment driver.

Replicates agent_steer's setup verbatim, swapping the TXC archs from
``txc_base`` / ``txc_pro`` to the multi-window aliases ``txc_base_mw`` /
``txc_pro_mw`` (decisions.md § 14). Same canonical schedule (batch=1024,
n_steps=20_000, plateau_off), same V7 steering protocol, same Sonnet
judge, same v1.1.0 EVAL_PROTOCOL_VERSION (concept-lift baseline fix
in commit ef33f822).

Diverges from the briefing's sketch by importing ``run_one_cell``
instead of a non-existent top-level ``my_eval_fn`` (the eval-fn is a
closure built by ``_make_eval_fn(seed, workspace, eval_key)`` —
``run_one_cell`` handles the workspace + eval_key plumbing internally
and calls ``runner.run_cell``, which is the canonical pathway).

Thread cap: inherited from the 100K driver pattern (OMP/MKL/torch=32
gives best throughput on this H100 pod's 208-core Xeon — see
``experiments.c5_steering_100k.run`` docstring for the profiling).
Pure perf change, bit-identical to default.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")

import torch  # noqa: E402

_TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", os.environ["OMP_NUM_THREADS"]))
torch.set_num_threads(_TORCH_THREADS)
torch.set_num_interop_threads(_TORCH_THREADS)

from experiments.c5_steering.run import (
    EVAL_PROTOCOL_VERSION,
    run_one_cell,
)
from temp_bench.case_studies.steering import (
    DEFAULT_COH_THRESHOLDS,
    DEFAULT_STRENGTHS,
)


DEFAULT_ARCHS: tuple[str, ...] = ("txc_base_mw", "txc_pro_mw")
DEFAULT_SEEDS: tuple[int, ...] = (42, 1, 2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="C5 multi-window deployment — txc_base_mw + txc_pro_mw "
                    "× 3 seeds at canonical 20K schedule, v1.1.0 eval."
    )
    ap.add_argument("--archs", nargs="*", default=list(DEFAULT_ARCHS),
                    choices=list(DEFAULT_ARCHS))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    ap.add_argument("--protocol", choices=("v7", "pp"), default="v7")
    ap.add_argument("--n-concepts", type=int, default=30)
    ap.add_argument("--strengths", type=float, nargs="*", default=None)
    ap.add_argument("--coh-thresholds", type=float, nargs="*", default=None)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override TrainingConfig.n_steps (default = "
                         "agent_steer's canonical 20_000). Use small "
                         "values for smoke tests; implies smoke=True via "
                         "--smoke if you want analysis filtering.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pre-test-only", action="store_true",
                    help="V7 health-check on txc_pro_mw (5 concepts × 1 strength).")
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
        f"[c5_mw] sweep archs={list(args.archs)} seeds={list(args.seeds)} "
        f"protocol={args.protocol} eval_protocol_version={EVAL_PROTOCOL_VERSION} "
        f"smoke={args.smoke} n_steps_override={args.n_steps}",
        flush=True,
    )

    if args.pre_test_only:
        run_one_cell(
            arch_name="txc_pro_mw",
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
                f"[c5_mw] launching cell arch={arch} seed={seed}",
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
