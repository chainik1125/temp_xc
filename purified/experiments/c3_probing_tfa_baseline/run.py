"""C3 TFA baseline at B=32 + full seq (decisions.md § 16).

Wasteland-faithful per Phase 7 ``train_phase7.py:312-353``: TFA's
attention tensor at d_sae=18432 + T=128 is too heavy for B=1024
(~9.6 GB fp32). The Phase 7 reference set TFA_BATCH=32 to keep
per-step memory tractable AND give the attention full ~128-token
context (papers/priors_in_time.md Fig. 2(d): ~80% variance explained
at 100+ tokens). 32 × 128 = 4096 tokens/step — paper-faithful, ~2×
SAEBench canonical.

Imports the existing C3 plumbing from ``c3_probing.run`` (datasource,
EVAL_PROTOCOL_VERSION, my_train_fn, my_eval_fn) — only ``TrainingConfig``
changes.

Usage::

    # Smoke (n_steps=200) on GPU 1
    TQDM_DISABLE=1 AGENT_NAME=agent_nlp \\
      bash scripts/run_on_gpu.sh 1 -- \\
      .venv/bin/python -m experiments.c3_probing_tfa_baseline.run \\
        --seeds 42 --k-feats 5 --n-steps 200

    # Full sweep — parallelize 2 seeds across GPU 0 + GPU 1
    TQDM_DISABLE=1 AGENT_NAME=agent_nlp \\
      bash scripts/run_on_gpu.sh 0 -- \\
      .venv/bin/python -m experiments.c3_probing_tfa_baseline.run \\
        --seeds 42 > logs/c3_tfa_gpu0.log 2>&1 &
    TQDM_DISABLE=1 AGENT_NAME=agent_nlp \\
      bash scripts/run_on_gpu.sh 1 -- \\
      .venv/bin/python -m experiments.c3_probing_tfa_baseline.run \\
        --seeds 1 2 > logs/c3_tfa_gpu1.log 2>&1 &
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

# Wasteland-faithful TFA training (decisions § 16, Phase 7 reference).
# B=32, full sequence (T=128) → 4096 tokens/step, ~2× SAEBench canonical
# and gives attention 128 tokens of context (Fig. 2(d) variance saturation).
TFA_TRAINING_CFG = TrainingConfig(
    n_steps=20_000,
    batch_size=32,                  # ← TFA-specific override
    plateau_early_stop=False,
)


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
    cfg = TFA_TRAINING_CFG
    if args.n_steps is not None:
        cfg = cfg.model_copy(update={"n_steps": args.n_steps})

    for seed in args.seeds:
        for k in args.k_feats:
            print(
                f"[c3_tfa_baseline] cell tfa seed={seed} k_feat={k} "
                f"B={cfg.batch_size} n_steps={cfg.n_steps}",
                flush=True,
            )
            eval_cfg = {
                "k_feat": k, "S": 32, "smoke": False,
                "_act_cache_key": act_cache_key,
                "_datasource_name": DATASOURCE,
            }
            result = runner.run_cell(
                component="c3",
                arch_name="tfa",
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
                    f"[{tag}] tfa seed={seed} k_feat={k} B=32  "
                    f"mean_AUC={mean_auc:.4f}±{std_auc or 0.0:.4f} "
                    f"(n={int(n_tasks or 0)} tasks)  "
                    f"eval_key={result.eval_key}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
