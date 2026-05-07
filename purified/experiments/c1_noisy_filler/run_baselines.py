"""c1_noisy missing-baselines driver — adds topk_sae and tsae_paper.

(decision 2026-05-06T23:30Z:) Setup B (c1_noisy) is missing topk_sae and
tsae_paper baselines. Fair-comparison parameters:
- Same datasource: `toy_markov_n20_d40_noisy` (d=40, d_sae=40 via c1
  per_component_hparams)
- Same TrainingConfig: n_steps=30000, batch=1024, lr=3e-4,
  optimizer=adam (matches existing tfa_pos / stacked_sae / txc_base /
  txc_pro cells)
- Same k_pos sweep: {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20}
- Same 3 seeds: {1, 2, 42}
- topk_sae: per-token TopK; no T axis. Override `d_sae=40`, `k_pos`.
- tsae_paper: BatchTopK + AuxK + matryoshka + temporal contrastive.
  Paper-faithful T=2 via TrainingConfig.train_window_size=2 (Bhalla/Ye
  2025 §3.1). Override `d_sae=40`, `k_pos`. contrastive_alpha=1.0
  (YAML default; paper uses 0.1 — this is a known minor discrepancy
  but matches our locked tsae_paper config).

Usage (single (arch, seed) per call; launcher shards across GPUs):
    .venv/bin/python -m experiments.c1_noisy_filler.run_baselines \\
        --arch <topk_sae|tsae_paper> --seed <1|2|42>
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
    DEFAULT_K_POSES, my_eval_fn, my_train_fn,
)


# Paper-faithful per-arch TrainingConfig overrides.
ARCH_CFG = {
    "topk_sae":   {"train_window_size": None},  # per-token; no temporal pairing
    "tsae_paper": {"train_window_size": 2},     # Bhalla/Ye 2025 paper-faithful
}


def _is_valid_cell(arch_name: str, k_pos: int, d_sae: int = 40) -> bool:
    """Both topk_sae and tsae_paper are per-token; no T constraint."""
    return k_pos <= d_sae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=list(ARCH_CFG.keys()))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--n-steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--k-poses", nargs="+", type=int, default=list(DEFAULT_K_POSES))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    arch_cfg = ARCH_CFG[args.arch]

    for k_pos in args.k_poses:
        if not _is_valid_cell(args.arch, int(k_pos)):
            print(f"[run_baselines] SKIP {args.arch} k={k_pos} "
                  f"(k_pos > d_sae=40)", flush=True)
            continue
        # Override d_sae=40 + k_pos — load_arch with component="c1"
        # (in my_train_fn / my_eval_fn) already handles d_sae=40, but
        # we also pass k_pos override for the BatchTopK budget.
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
            "_p_A": 0.0,
            "_p_B": 0.625,
        }
        print(f"[run_baselines] {args.arch:12s} k={k_pos:2d} seed={args.seed} "
              f"steps={cfg.n_steps} train_window_size={arch_cfg['train_window_size']}",
              flush=True)
        runner.run_cell(
            component=COMPONENT,
            arch_name=args.arch,
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
