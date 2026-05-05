"""C5 TopK + TFA baselines (decisions § 16) — single (arch, seed) cell.

Two new C5 baselines paired with agent_steer's existing v1.1.0 cells
(txc_base, txc_pro) and agent_filler's prior tsae_paper T=2 cells:

| arch       | TrainingConfig                          | tokens/step  |
|---         |---                                      |---:          |
| topk_sae   | B=1024, train_window_size=1             | 1,024        |
| tfa        | B=32, train_window_size=None (full seq) | 4,096        |

TopK at T=1 is paper-faithful per § 15 (matches the SAEBench-canonical
window). TFA at B=32 + full seq is wasteland-faithful per
``origin/han-phase7-unification:experiments/phase7_unification/
train_phase7.py`` lines 312-353 — TFA's attention tensor (B × T × d_sae)
peaks at ~9.6 GB fp32 at d_sae=18432, so B=32 keeps memory in check
while preserving the 100+ token context that TFA's attention needs to
explain ~80% of variance (paper Fig. 2(d)).

Both archs override the canonical TrainingConfig via run_one_cell's
``train_window_size`` and ``batch_size`` kwargs (§ 16 framework change).
Different (B, train_window_size) → different train_key, so these
cells are versioned independently from any prior TopK / TFA training.

Top-level run_sweep.sh launches 6 of these in parallel.
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


# Per-arch literature-faithful TrainingConfig overrides.
# batch_size=None → use canonical default (1024).
ARCH_CFG: dict[str, dict[str, int | None]] = {
    "topk_sae": {"batch_size": None, "train_window_size": 1},
    "tfa":      {"batch_size": 32,   "train_window_size": None},
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="C5 TopK + TFA baselines — single (arch, seed) cell."
    )
    ap.add_argument("--arch", required=True, choices=list(ARCH_CFG.keys()))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override TrainingConfig.n_steps for smoke tests.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tag the leaderboard row eval_cfg.smoke=True.")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    cfg = ARCH_CFG[args.arch]
    print(
        f"[c5_baseline] {args.arch} seed={args.seed} "
        f"B={cfg['batch_size'] if cfg['batch_size'] else 'default(1024)'} "
        f"train_window_size={cfg['train_window_size']} "
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
        train_window_size=cfg["train_window_size"],
        batch_size=cfg["batch_size"],
    )


if __name__ == "__main__":
    main()
