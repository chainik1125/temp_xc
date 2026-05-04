"""C6 driver — replicates agent_em's setup at n_steps=100_000.

Imports agent_em's training-cfg recipe, train_fn, eval-fn factory, and
activation-cache build helper from `experiments.c6_em.*` without
modification — only the `n_steps` field on `TrainingConfig` differs
(100_000 vs agent_em's 25_000 default).

Per the agent_em_100k mandate: same C6 sweep, longer training. Whichever
sweep finishes first becomes the C6 paper headline. `train_key` includes
`n_steps`, so 25K and 100K cells coexist cleanly in `leaderboard.jsonl`.

    cd /workspace/temp_xc/purified
    source scripts/set_agent_env.sh agent_em_100k
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c6_em_100k.run \\
        --archs sae_arditi txc_base --seeds 42
"""

from __future__ import annotations

import argparse
import logging
import sys

from experiments.c6_em.run import (
    COMPONENT,
    DEFAULT_DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    ensure_activation_cache,
    make_eval_fn,
    make_training_cfg,
)
from experiments.c6_em.train import my_train_fn
from temp_bench import runner
from temp_bench.schemas import TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("c6_100k.run")


def make_training_cfg_100k(arch_name: str, *, n_steps: int = 100_000) -> TrainingConfig:
    base = make_training_cfg(arch_name)
    return base.model_copy(update={"n_steps": n_steps})


def make_eval_cfg(arch_name: str) -> dict:
    # Mirror experiments/c6_em/run.py:run_one_cell so eval_keys align
    # with agent_em's at the eval-pathway level (only train_key differs).
    return {
        "wang_full": True,
        "screen_top_n": 100,
        "n_survivors": 20,
        "n_final": 3,
        "n_alpha_grid": 27,
        "max_new_tokens": 200,
        "arch_T": 5 if arch_name == "txc_base" else 1,
    }


def run_one_cell_100k(
    arch_name: str,
    *,
    seed: int,
    datasource_name: str,
    n_steps: int,
    skip_eval: bool = False,
    force_train: bool = False,
    force_eval: bool = False,
):
    training_cfg = make_training_cfg_100k(arch_name, n_steps=n_steps)
    eval_cfg = make_eval_cfg(arch_name)

    if skip_eval:
        def eval_fn(*, model, eval_cfg, component):
            return {"peak_align": 0.0}, "peak_align"
    else:
        eval_fn = make_eval_fn(datasource_name)

    log.info(
        "[c6_100k.run] CELL: arch=%s seed=%d ds=%s n_steps=%d",
        arch_name, seed, datasource_name, training_cfg.n_steps,
    )
    log.info("[c6_100k.run] training_cfg=%s", training_cfg.model_dump())

    result = runner.run_cell(
        component=COMPONENT,
        arch_name=arch_name,
        seed=seed,
        datasource_name=datasource_name,
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=my_train_fn,
        eval_fn=eval_fn,
        force_train=force_train,
        force_eval=force_eval,
    )
    log.info(
        "[c6_100k.run] CELL DONE: train_key=%s eval_key=%s cached=%s",
        result.train_key, result.eval_key, result.cached,
    )
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--archs", nargs="+", default=("sae_arditi", "txc_base"),
        choices=("sae_arditi", "txc_base"),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=(42,))
    p.add_argument("--datasource", default=DEFAULT_DATASOURCE)
    p.add_argument(
        "--n-steps", type=int, default=100_000,
        help="Override n_steps. Use a small value (e.g. 200) for smoke testing.",
    )
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    p.add_argument(
        "--skip-eval", action="store_true",
        help="Train + checkpoint only; no Wang. Useful for smoke-test.",
    )
    args = p.parse_args(argv)

    ensure_activation_cache(args.datasource)

    for arch in args.archs:
        for seed in args.seeds:
            run_one_cell_100k(
                arch,
                seed=seed,
                datasource_name=args.datasource,
                n_steps=args.n_steps,
                skip_eval=args.skip_eval,
                force_train=args.force_train,
                force_eval=args.force_eval,
            )

    log.info("[c6_100k.run] all cells complete")


if __name__ == "__main__":
    sys.exit(main())
