"""C3 TXC-base T-sweep (T=10, T=20). (decision 2026-05-05) directive.

Adds T=10 and T=20 cells alongside the canonical T=5 sweep — same
datasource + training schedule, only the T axis varies via
``TrainingConfig.arch_hparams_override``. Different T → fresh
``train_key`` → fresh checkpoint; existing T=5 cells stay intact as
the canonical headline.

Imports the existing C3 plumbing (datasource, EVAL_PROTOCOL_VERSION,
my_train_fn, my_eval_fn) — only ``TrainingConfig`` changes.

Usage::

    # Smoke (T=10, n_steps=200)
    TQDM_DISABLE=1 AGENT_NAME=agent_nlp \\
      bash scripts/run_on_gpu.sh 1 -- \\
      .venv/bin/python -m experiments.c3_probing_txc_T_sweep.run \\
        --T-values 10 --seeds 42 --k-feats 5 --n-steps 200

    # Full launch — T=10 on GPU 0, T=20 on GPU 1 in parallel
    bash scripts/run_on_gpu.sh 0 -- \\
      .venv/bin/python -m experiments.c3_probing_txc_T_sweep.run \\
        --T-values 10 > logs/c3_txc_T10_gpu0.log 2>&1 &
    bash scripts/run_on_gpu.sh 1 -- \\
      .venv/bin/python -m experiments.c3_probing_txc_T_sweep.run \\
        --T-values 20 > logs/c3_txc_T20_gpu1.log 2>&1 &
"""

from __future__ import annotations

import argparse

from temp_bench import runner
from temp_bench.config import compute_act_cache_key, load_datasource
from temp_bench.schemas import TrainingConfig

from experiments.c3_probing.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_eval_fn,
    my_train_fn,
)


def _cfg(T: int) -> TrainingConfig:
    return TrainingConfig(
        n_steps=20_000,
        batch_size=1024,
        plateau_early_stop=False,
        arch_hparams_override={"T": int(T)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T-values", nargs="+", type=int, default=[10, 20])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--k-feats", nargs="+", type=int, default=[5, 20])
    ap.add_argument(
        "--n-steps", type=int, default=None,
        help="Override n_steps for smoke tests.",
    )
    args = ap.parse_args()

    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))

    for T in args.T_values:
        cfg = _cfg(T)
        if args.n_steps is not None:
            cfg = cfg.model_copy(update={"n_steps": args.n_steps})
        for seed in args.seeds:
            for k in args.k_feats:
                print(
                    f"[c3_txc_T] cell txc_base T={T} seed={seed} "
                    f"k_feat={k} n_steps={cfg.n_steps}",
                    flush=True,
                )
                eval_cfg = {
                    "k_feat": k, "S": 32, "smoke": False,
                    "_act_cache_key": act_cache_key,
                    "_datasource_name": DATASOURCE,
                }
                result = runner.run_cell(
                    component="c3",
                    arch_name="txc_base",
                    seed=seed,
                    datasource_name=DATASOURCE,
                    training_cfg=cfg,
                    eval_cfg=eval_cfg,
                    eval_protocol_version=EVAL_PROTOCOL_VERSION,
                    train_fn=my_train_fn,
                    eval_fn=my_eval_fn,
                )
                tag = "CACHED" if result.cached else "NEW"
                m = result.metrics or {}
                mean_auc = m.get("mean_auc")
                std_auc = m.get("std_auc")
                n_tasks = m.get("n_tasks")
                if mean_auc is not None:
                    print(
                        f"[{tag}] txc_base T={T} seed={seed} k_feat={k}  "
                        f"mean_AUC={mean_auc:.4f}±{std_auc or 0.0:.4f} "
                        f"(n={int(n_tasks or 0)} tasks)  "
                        f"eval_key={result.eval_key}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
