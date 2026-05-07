"""Setup A T-sweep — txc_base across T={2,4,6,8,10,12} on
toy_coupled_K10_M20_d256 (canonical Setup A, ρ=0.7, d_sae=40).

Han's "ONE PAPER GOAL" #2 (per agent_hammer briefing 2026-05-07):
"What happens as T grows (TXC-base T-sweep at fixed k_pos)".

Setup A currently has txc_base T=5 default + txc_pro T={2,5,12}.
This adds the explicit txc_base T-sweep across all 6 T values.
d_sae=40 is c2's per_component_hparams override.

Auto-skip cells where k_pos * T > d_sae=40 (matryoshka prefix limit).

Usage (single (T, seed) per call; launcher shards across GPUs):
    .venv/bin/python -m experiments.c2_synthetic_coupled.run_setup_a_t_sweep \\
        --T <2|4|6|8|10|12> --seed <1|2|42>
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from temp_bench import runner
from temp_bench.schemas import TrainingConfig

from experiments.c2_synthetic_coupled.run import (
    COMPONENT, EVAL_PROTOCOL_VERSION, DEFAULT_K_POSES,
    RHO_DATASOURCE_MAP, my_eval_fn, my_train_fn,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", required=True, type=int,
                    choices=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--n-steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--k-poses", nargs="+", type=int,
                    default=list(DEFAULT_K_POSES))
    args = ap.parse_args()

    arch_name = "txc_base"
    datasource_name = RHO_DATASOURCE_MAP.get(args.rho, "toy_coupled_K10_M20_d256")
    d_sae = 40

    for k_pos in args.k_poses:
        if k_pos * args.T > d_sae:
            print(f"[setup_a_t_sweep] SKIP T={args.T} k={k_pos} "
                  f"(k_pos * T = {k_pos * args.T} > d_sae={d_sae})", flush=True)
            continue
        override = {"k_pos": int(k_pos), "T": int(args.T), "d_sae": d_sae}
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
            "t_label": f"T={args.T}",
            "rho": float(args.rho),
        }
        print(f"[setup_a_t_sweep] {arch_name:12s} T={args.T:2d} k={k_pos:2d} "
              f"seed={args.seed} ρ={args.rho} ds={datasource_name} "
              f"steps={cfg.n_steps}", flush=True)
        runner.run_cell(
            component=COMPONENT,
            arch_name=arch_name,
            seed=int(args.seed),
            datasource_name=datasource_name,
            training_cfg=cfg,
            eval_cfg=eval_cfg,
            eval_protocol_version=EVAL_PROTOCOL_VERSION,
            train_fn=my_train_fn,
            eval_fn=my_eval_fn,
        )


if __name__ == "__main__":
    main()
