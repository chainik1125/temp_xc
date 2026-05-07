"""C3 TopK SAE baseline re-train at T=1 (decisions.md § 15).

Per (decision 2026-05-05): per-token archs (topk_sae here, tsae_paper on
agent_em_100k's pod) re-train at literature-canonical scale.
``train_window_size=1`` makes the trainer sample one random T=1 window
per row (1024 tokens/step at batch=1024), matching SAEBench App. B's
canonical SAE-training budget. The previous over-batched cells
(``train_window_size=None``) stay in the leaderboard under their own
train_keys for diff comparison.

Imports the existing C3 plumbing (datasource, eval_protocol_version,
my_train_fn, my_eval_fn) — only ``TrainingConfig`` changes.

Usage::

    # Smoke (n_steps=200) on GPU 1
    TQDM_DISABLE=1 AGENT_NAME=agent_nlp \\
      bash scripts/run_on_gpu.sh 1 -- \\
      .venv/bin/python -m experiments.c3_probing_topk_baseline.run \\
        --seeds 42 --k-feats 5 --n-steps 200

    # Full sweep on GPU 1
    TQDM_DISABLE=1 AGENT_NAME=agent_nlp \\
      bash scripts/run_on_gpu.sh 1 -- \\
      .venv/bin/python -m experiments.c3_probing_topk_baseline.run \\
        > logs/c3_topk_T1_gpu1.log 2>&1 &
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

TOPK_TRAINING_CFG = TrainingConfig(n_steps=20_000, train_window_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--k-feats", nargs="+", type=int, default=[5, 20])
    ap.add_argument(
        "--n-steps", type=int, default=None,
        help="Override n_steps for smoke tests.",
    )
    args = ap.parse_args()

    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))
    cfg = TOPK_TRAINING_CFG
    if args.n_steps is not None:
        cfg = cfg.model_copy(update={"n_steps": args.n_steps})

    for seed in args.seeds:
        for k in args.k_feats:
            print(
                f"[c3_topk_T1] cell topk_sae seed={seed} k_feat={k} "
                f"T={cfg.train_window_size} n_steps={cfg.n_steps}",
                flush=True,
            )
            eval_cfg = {
                "k_feat": k, "S": 32, "smoke": False,
                "_act_cache_key": act_cache_key,
                "_datasource_name": DATASOURCE,
            }
            result = runner.run_cell(
                component="c3",
                arch_name="topk_sae",
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
                    f"[{tag}] topk_sae seed={seed} k_feat={k} T=1  "
                    f"mean_AUC={mean_auc:.4f}±{std_auc or 0.0:.4f} "
                    f"(n={int(n_tasks or 0)} tasks)  "
                    f"eval_key={result.eval_key}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
