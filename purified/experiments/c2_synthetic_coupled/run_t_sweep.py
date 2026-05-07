"""C2 Setup D — txc_base T-sweep driver (single T, single seed).

Mirrors c1_noisy_filler/run_t.py but for the noisy+overlap regime.
Sweeps T ∈ {2, 4, 5, 6, 8, 10, 12} at fixed k_pos to show the
local-vs-global trade-off as a function of window size.

k_pos × T ≤ d_sae=40 constraint applies (txc_base does
``pre.topk(k_win, dim=-1)`` on d_sae-dim tensor).
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TQDM_DISABLE", "1")

from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from temp_bench.config import load_datasource

from experiments.c2_synthetic_coupled.run_hunt import (
    COMPONENT, EVAL_PROTOCOL_VERSION,
    make_train_fn, make_eval_fn,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasource", required=True)
    ap.add_argument("--T", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--n-steps", type=int, default=8_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--k-poses", nargs="+", type=int, required=True)
    ap.add_argument("--d-sae", type=int, default=40)
    args = ap.parse_args()

    arch_name = "txc_base"
    T_val = int(args.T)
    t_label = f"T={T_val}"
    spec = load_datasource(args.datasource)
    p_B = float(getattr(spec, "p_B", 1.0))
    n_parents = int(spec.n_parents)
    rho = float(spec.rho)

    train_fn = make_train_fn(args.datasource)
    eval_fn = make_eval_fn(args.datasource)

    for k_pos in args.k_poses:
        k_win = int(k_pos) * T_val
        if k_win > int(args.d_sae):
            print(f"[run_t] SKIP {t_label} k={k_pos} (k_win={k_win} > d_sae={args.d_sae})",
                  flush=True)
            continue
        override = {"k_pos": int(k_pos), "T": T_val}
        cfg = TrainingConfig(
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            plateau_early_stop=False,
            arch_hparams_override=override,
        )
        eval_cfg = {
            "k_pos": int(k_pos),
            "smoke": False,
            "_arch_hparams_override": override,
            "rho": rho,
            "p_B": p_B,
            "n_parents": n_parents,
            "hunt_phase": "tsweep",
            "n_steps_train": int(args.n_steps),
            "t_label": t_label,
        }
        print(f"[c2 tsweep] txc_base {t_label} k={k_pos} seed={args.seed} "
              f"ds={args.datasource}", flush=True)
        runner.run_cell(
            component=COMPONENT,
            arch_name=arch_name,
            seed=int(args.seed),
            datasource_name=args.datasource,
            training_cfg=cfg,
            eval_cfg=eval_cfg,
            eval_protocol_version=EVAL_PROTOCOL_VERSION,
            train_fn=train_fn,
            eval_fn=eval_fn,
        )


if __name__ == "__main__":
    main()
