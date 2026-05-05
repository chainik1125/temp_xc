"""C5 multi-window driver for agent_filler — single (arch, seed) per
invocation, pinned to one GPU. The top-level ``run_sweep.sh`` spawns
6 of these in parallel via ``scripts/run_on_gpu.sh``.

Replicates agent_steer's canonical C5 setup verbatim with the multi-
window arch swap (decisions.md § 14): same V7 steering protocol, same
30-concept set, same Sonnet 4.6 judge, same v1.1.0 EVAL_PROTOCOL_VERSION
(concept-lift baseline fix in commit ef33f822).

Design choices:
- Imports ``run_one_cell`` from ``experiments.c5_steering.run`` rather
  than re-implementing the closure plumbing. ``run_one_cell`` is the
  canonical wrapper that builds ``_make_eval_fn(seed, workspace,
  eval_key)`` and calls ``runner.run_cell``; bypassing it produces an
  ``eval_key`` that doesn't match the workspace dir the case-study
  writes to (debugged by agent_steer in commit f8a28469).
- Threading caps at 8 (vs agent_steer_100k's 32) because this 8× A40
  pod has 76 vCPU shared across up to 6 concurrent procs (~12
  threads/proc). 8 is conservative-safe headroom.
"""
from __future__ import annotations

import argparse
import os

# Match agent_steer_100k's threading pattern, scaled for the 76-core
# multi-process pod. Bit-identical math, just throughput.
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import torch  # noqa: E402

_TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", os.environ["OMP_NUM_THREADS"]))
torch.set_num_threads(_TORCH_THREADS)
torch.set_num_interop_threads(_TORCH_THREADS)

from experiments.c5_steering.run import (  # noqa: E402
    EVAL_PROTOCOL_VERSION,
    run_one_cell,
)
from temp_bench.case_studies.steering import (  # noqa: E402
    DEFAULT_COH_THRESHOLDS,
    DEFAULT_STRENGTHS,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="C5 multi-window deployment — single (arch, seed) cell."
    )
    ap.add_argument("--arch", required=True,
                    choices=["txc_base_mw", "txc_pro_mw"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override TrainingConfig.n_steps. Default = "
                         "agent_steer's canonical 20_000; use small "
                         "values for smoke tests.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tag the leaderboard row eval_cfg.smoke=True.")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    print(
        f"[c5_filler] cell arch={args.arch} seed={args.seed} "
        f"eval_protocol={EVAL_PROTOCOL_VERSION} "
        f"smoke={args.smoke} n_steps_override={args.n_steps}",
        flush=True,
    )

    run_one_cell(
        arch_name=args.arch,
        seed=args.seed,
        protocol="v7",
        n_concepts=30,
        strengths=DEFAULT_STRENGTHS,
        coh_thresholds=DEFAULT_COH_THRESHOLDS,
        n_steps=args.n_steps,
        smoke=args.smoke,
        force_train=args.force_train,
        force_eval=args.force_eval,
    )


if __name__ == "__main__":
    main()
