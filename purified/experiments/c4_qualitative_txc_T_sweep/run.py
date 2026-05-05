"""C4 TXC-base T-sweep qualitative eval (T=10, T=20).

Han 2026-05-05 PM directive. Cache-hits on the C3 T-sweep checkpoints
(commit ``d54cead3`` + ``experiments.c3_probing_txc_T_sweep.run``):
the per-cell ``train_key`` derives from
``arch_hparams_override={"T": int}`` so passing the same override
here resolves to the same checkpoint, and the runner skips training.

Usage::

    TQDM_DISABLE=1 AGENT_NAME=agent_nlp \\
      .venv/bin/python -m experiments.c4_qualitative_txc_T_sweep.run \\
        --T-values 10 20 --seeds 1 2 42 \\
        > logs/c4_txc_T_sweep.log 2>&1 &
"""

from __future__ import annotations

import argparse

from temp_bench import runner
from temp_bench.config import (
    compute_act_cache_key,
    compute_eval_key,
    compute_train_key,
    load_arch,
    load_datasource,
)
from temp_bench.schemas import TrainingConfig

from experiments.c4_qualitative.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    DEFAULT_N_FEATURES,
    my_eval_fn,
    my_train_fn,
)

COMPONENT = "c4"


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
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 42])
    ap.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    args = ap.parse_args()

    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))

    for T in args.T_values:
        cfg = _cfg(T)
        for seed in args.seeds:
            print(
                f"[c4_txc_T] cell txc_base T={T} seed={seed} "
                f"n_features={args.n_features}",
                flush=True,
            )
            # Pre-compute eval_key so my_eval_fn knows where to write
            # judge_outputs.jsonl. Runner re-derives the same value.
            train_key = compute_train_key(
                arch=load_arch("txc_base", component=COMPONENT),
                seed=seed,
                training_cfg=cfg,
                act_cache_key=act_cache_key,
            )
            eval_cfg = {
                "n_features": args.n_features,
                "_act_cache_key": act_cache_key,
                "_datasource_name": DATASOURCE,
            }
            eval_key = compute_eval_key(
                train_key=train_key,
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                eval_cfg=eval_cfg,
            )
            eval_cfg["_eval_key_hint"] = eval_key

            result = runner.run_cell(
                component=COMPONENT,
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
            sem = m.get("top_N_semantic")
            agree = m.get("judge_agreement")
            njudged = m.get("n_features_judged")
            sem_s = (
                f"{int(sem)}/{int(njudged)}"
                if sem is not None and njudged is not None
                else "-"
            )
            agree_s = f"{agree:.3f}" if agree is not None else "-"
            print(
                f"[{tag}] txc_base T={T} seed={seed} "
                f"n_features={args.n_features}  "
                f"SEMANTIC={sem_s}  agreement={agree_s}  "
                f"eval_key={result.eval_key}",
                flush=True,
            )


if __name__ == "__main__":
    main()
