"""c1_noisy single-(T, seed) driver — runs txc_base sweep at one T value.

Used by run_tsweep.sh to parallelise T-sweep across GPUs. Mirrors
run.py but restricted to txc_base + a single T-override.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from temp_bench import runner
from temp_bench.schemas import TrainingConfig

from experiments.c1_noisy_filler.run import (
    COMPONENT, DATASOURCE, EVAL_PROTOCOL_VERSION,
    DEFAULT_K_POSES, _is_valid_cell, my_eval_fn, my_train_fn,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--n-steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--k-poses", nargs="+", type=int, default=list(DEFAULT_K_POSES))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    arch_name = "txc_base"
    t_override = {"T": int(args.T)}
    t_label = f"T={args.T}"

    for k_pos in args.k_poses:
        if not _is_valid_cell(arch_name, t_override, int(k_pos)):
            print(f"[run_t] SKIP {arch_name} {t_label} k={k_pos} "
                  f"(k_train > arch budget at toy d_sae=40)", flush=True)
            continue
        override = {"k_pos": int(k_pos), "d_sae": 40, **t_override}
        cfg = TrainingConfig(
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            plateau_early_stop=False,
            arch_hparams_override=override,
        )
        eval_cfg = {
            "k_pos": int(k_pos),
            "smoke": bool(args.smoke),
            "_arch_hparams_override": override,
            "t_label": t_label,
            "_p_A": 0.0,
            "_p_B": 0.625,
        }
        print(f"[run_t] {arch_name:12s} {t_label:6s} k={k_pos:2d} "
              f"seed={args.seed} steps={cfg.n_steps}", flush=True)
        runner.run_cell(
            component=COMPONENT,
            arch_name=arch_name,
            seed=int(args.seed),
            datasource_name=DATASOURCE,
            training_cfg=cfg,
            eval_cfg=eval_cfg,
            eval_protocol_version=EVAL_PROTOCOL_VERSION,
            train_fn=my_train_fn,
            eval_fn=my_eval_fn,
        )


if __name__ == "__main__":
    main()
