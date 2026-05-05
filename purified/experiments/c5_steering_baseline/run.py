"""C5 T-SAE baseline re-train at T=2 (decisions.md § 15).

Single-cell driver; the top-level ``run_sweep.sh`` launches 3 in parallel
across GPUs 0..2. Replaces the rescinded MW driver under
``experiments/c5_steering_filler/``.

Why T=2: Bhalla/Ye 2025 §3.1 paper-faithful adjacent-pair training.
At ``B=1024 sentences × T=2 positions`` per step → 2048 tokens/step,
which matches SAEBench's canonical scale and brings T-SAE in line with
C6/C7's per-token throughput. Old T=None (full-sequence) cells from
agent_steer's earlier sweep stay on the leaderboard at distinct train_keys
— ``train_window_size`` flows into the train_key hash so the two are
versioned independently.

Threading caps at OMP/MKL=8 (76-core pod ÷ 3 procs ≈ 25; 8 is conservative
headroom matching agent_steer_100k's tuned default).
"""
from __future__ import annotations

import argparse
import os

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


TRAIN_WINDOW_SIZE = 2  # Bhalla/Ye 2025 §3.1 paper-faithful


def main() -> None:
    ap = argparse.ArgumentParser(
        description="C5 T-SAE baseline re-train at T=2 — single seed."
    )
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override TrainingConfig.n_steps. Default = "
                         "agent_steer's canonical 20_000; use small values "
                         "for smoke tests.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tag the leaderboard row eval_cfg.smoke=True.")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    print(
        f"[c5_baseline] tsae_paper seed={args.seed} "
        f"train_window_size={TRAIN_WINDOW_SIZE} "
        f"eval_protocol={EVAL_PROTOCOL_VERSION} "
        f"smoke={args.smoke} n_steps_override={args.n_steps}",
        flush=True,
    )

    run_one_cell(
        arch_name="tsae_paper",
        seed=args.seed,
        protocol="v7",
        n_concepts=30,
        strengths=DEFAULT_STRENGTHS,
        coh_thresholds=DEFAULT_COH_THRESHOLDS,
        n_steps=args.n_steps,
        smoke=args.smoke,
        force_train=args.force_train,
        force_eval=args.force_eval,
        train_window_size=TRAIN_WINDOW_SIZE,
    )


if __name__ == "__main__":
    main()
