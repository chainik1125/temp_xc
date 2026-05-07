"""Setup E baseline backfill driver — hierarchical (no obs noise).

Targets: fill ``tfa_pos``, ``tsae_paper``, ``stacked_sae`` T=2/T=5 on
the Setup E np2 datasource (`toy_hierarchical_Kg10_Kl30_d256_np2`)
and any other Setup E variants that lack baselines.

Generator: ``hierarchical_features`` (no observation noise; that's
agent_synth's Setup G). Imports `make_train_fn` / `make_eval_fn` from
agent_synth's `experiments.c2_hierarchical.run` (read-only).
"""

from __future__ import annotations

import argparse
import os
from typing import Any

os.environ.setdefault("TQDM_DISABLE", "1")

from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from temp_bench.config import load_datasource

# Import agent_synth's plumbing.
from experiments.c2_hierarchical.run import (
    make_train_fn as e_make_train,
    make_eval_fn  as e_make_eval,
)

COMPONENT = "c2"
EVAL_PROTOCOL_VERSION = "1.0.0"


def _build_override(arch: str, k_pos: int, T: int | None) -> dict[str, Any]:
    base = {"k_pos": int(k_pos), "d_sae": 40}
    if arch in ("topk_sae", "tfa_pos", "tsae_paper"):
        return base
    if arch == "stacked_sae":
        if T is None:
            raise SystemExit("stacked_sae requires --T.")
        return {**base, "T": int(T)}
    raise SystemExit(f"Unsupported arch '{arch}'.")


def _t_label(arch: str, T: int | None) -> str:
    if arch == "stacked_sae" and T is not None:
        return f"T={T}"
    return "default"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasource", required=True)
    ap.add_argument("--arch", required=True,
                    choices=["topk_sae", "tfa_pos", "tsae_paper", "stacked_sae"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--k-poses", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--n-steps", type=int, default=8_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    spec = load_datasource(args.datasource)
    train_fn = e_make_train(args.datasource)
    eval_fn  = e_make_eval(args.datasource)
    train_window_size = 2 if args.arch == "tsae_paper" else None

    for k_pos in args.k_poses:
        override = _build_override(args.arch, int(k_pos), args.T)
        cfg = TrainingConfig(
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            plateau_early_stop=False,
            arch_hparams_override=override,
            train_window_size=train_window_size,
        )
        eval_cfg = {
            "k_pos": int(k_pos),
            "smoke": False,
            "_arch_hparams_override": override,
            "t_label": _t_label(args.arch, args.T),
            "n_steps_train": int(args.n_steps),
            "setup": "E",
        }
        print(f"[setup_e] {args.arch:12s} T={args.T} k={k_pos:2d} "
              f"seed={args.seed} ds={args.datasource}", flush=True)
        runner.run_cell(
            component=COMPONENT,
            arch_name=args.arch,
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
