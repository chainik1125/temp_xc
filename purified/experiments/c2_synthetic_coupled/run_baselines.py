"""c2 (Setup A) missing-baseline driver — adds tsae_paper.

Han 2026-05-06T23:30Z: Setup A (coupled features) is missing
tsae_paper. Fair-comparison parameters:

- Same datasource: `toy_coupled_K10_M20_d256` (d=256, d_sae=40 via c2
  per_component_hparams; locked_archs.yaml has tsae_paper.c2 NOT in
  per_component_hparams so we override d_sae=40 explicitly here).
- Same TrainingConfig: n_steps=30000, batch=1024, lr=3e-4,
  optimizer=adam (matches existing topk_sae / stacked_sae / txc_base /
  txc_pro cells in c2_synthetic_coupled).
- Same k_pos sweep: {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20}.
- Same 3 seeds: {1, 2, 42}.
- tsae_paper: BatchTopK + AuxK + matryoshka + temporal contrastive.
  Paper-faithful T=2 via TrainingConfig.train_window_size=2.
  Override `d_sae=40`, `k_pos=k`.

Note: this driver mirrors `experiments/c2_synthetic_coupled/run.py`
but writes to the same `component="c2"` so cells append to the
existing c2 results table.

Usage (one (arch, seed) per call):
    .venv/bin/python -m experiments.c2_synthetic_coupled.run_baselines \\
        --arch tsae_paper --seed <1|2|42>
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
    my_eval_fn, my_train_fn, RHO_DATASOURCE_MAP,
)


# Paper-faithful per-arch TrainingConfig overrides.
ARCH_CFG = {
    "tsae_paper": {"train_window_size": 2},   # Bhalla/Ye 2025 paper-faithful
}


def _is_valid_cell(arch_name: str, k_pos: int, d_sae: int = 40) -> bool:
    """tsae_paper is per-token (BatchTopK); no T constraint at the
    per-token level. Just k_pos ≤ d_sae."""
    return k_pos <= d_sae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ARCH_CFG.keys()))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--n-steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--k-poses", nargs="+", type=int, default=list(DEFAULT_K_POSES))
    ap.add_argument("--rho", type=float, default=0.7,
                    help="ρ value (default 0.7 = canonical Setup A).")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    arch_cfg = ARCH_CFG[args.arch]
    datasource_name = RHO_DATASOURCE_MAP.get(args.rho, "toy_coupled_K10_M20_d256")

    for k_pos in args.k_poses:
        if not _is_valid_cell(args.arch, int(k_pos)):
            print(f"[run_baselines] SKIP {args.arch} k={k_pos} "
                  f"(k_pos > d_sae=40)", flush=True)
            continue
        override = {"k_pos": int(k_pos), "d_sae": 40}
        cfg = TrainingConfig(
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            plateau_early_stop=False,
            arch_hparams_override=override,
            train_window_size=arch_cfg["train_window_size"],
        )
        eval_cfg = {
            "k_pos": int(k_pos),
            "smoke": bool(args.smoke),
            "_arch_hparams_override": override,
            "t_label": "default",
            "rho": float(args.rho),
        }
        print(f"[run_baselines] {args.arch:12s} k={k_pos:2d} seed={args.seed} "
              f"ρ={args.rho} ds={datasource_name} steps={cfg.n_steps} "
              f"train_window_size={arch_cfg['train_window_size']}",
              flush=True)
        runner.run_cell(
            component=COMPONENT,
            arch_name=args.arch,
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
