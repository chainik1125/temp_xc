"""C3 MLC BASE baseline at L=5 (decisions § 16, paper-faithful).

Thin clone of `experiments/c3_probing_mlc/run.py` with the datasource
swapped from `gemma_2_2b_it_l11to15_fineweb_24k128` to its BASE
counterpart `gemma_2_2b_base_l11to15_fineweb_24k128`. Re-uses the
custom `my_train_fn_mlc` + `my_eval_fn_mlc` (4D probe-array support)
from the IT driver verbatim — only the datasource string differs.

Per (decision 2026-05-06): BASE MLC parity for the IT/BASE matrix in C3.
agent_paper integrates new BASE MLC cells via canonical_train_keys()
at paper-render time.

    cd /workspace/temp_xc/purified
    source scripts/set_agent_env.sh agent_em_100k
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c3_probing_mlc_base.run \\
        --seeds 42 1 2 --k-feats 5 10 20 40 80 160 320 640
"""

from __future__ import annotations

import argparse
import logging
import sys

from experiments.c3_probing.run import (
    COMPONENT,
    DEFAULT_S,
    EVAL_PROTOCOL_VERSION,
)
from experiments.c3_probing_mlc.run import (
    MLC_TRAINING_CFG,
    my_eval_fn_mlc,
    my_train_fn_mlc,
)
from temp_bench import runner
from temp_bench.config import compute_act_cache_key, load_datasource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("c3_mlc_base.run")

ARCH = "mlc"
DATASOURCE = "gemma_2_2b_base_l11to15_fineweb_24k128"
DEFAULT_SEEDS = (42, 1, 2)
# k_feats expansion per (decision 2026-05-06): {5, 10, 20, 40, 80, 160, 320, 640}.
DEFAULT_K_FEATS_BASE = (5, 10, 20, 40, 80, 160, 320, 640)


def run_one_cell(
    *,
    seed: int,
    k_feat: int,
    S: int = DEFAULT_S,
    smoke: bool = False,
    training_cfg=None,
    force_train: bool = False,
    force_eval: bool = False,
):
    cfg = training_cfg if training_cfg is not None else MLC_TRAINING_CFG
    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))
    eval_cfg = {
        "k_feat": k_feat,
        "S": S,
        "smoke": smoke,
        "_act_cache_key": act_cache_key,
        "_datasource_name": DATASOURCE,
    }

    log.info(
        "[c3_mlc_base.run] CELL: arch=%s seed=%d k_feat=%d S=%d "
        "n_steps=%d smoke=%s",
        ARCH, seed, k_feat, S, cfg.n_steps, smoke,
    )

    result = runner.run_cell(
        component=COMPONENT,
        arch_name=ARCH,
        seed=seed,
        datasource_name=DATASOURCE,
        training_cfg=cfg,
        eval_cfg=eval_cfg,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=my_train_fn_mlc,
        eval_fn=my_eval_fn_mlc,
        force_train=force_train,
        force_eval=force_eval,
    )
    log.info(
        "[c3_mlc_base.run] CELL DONE: arch=%s seed=%d k_feat=%d "
        "train_key=%s eval_key=%s cached=%s",
        ARCH, seed, k_feat,
        result.train_key, result.eval_key, result.cached,
    )
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Seeds. Default: 42 1 2.",
    )
    p.add_argument(
        "--k-feats", nargs="+", type=int, default=list(DEFAULT_K_FEATS_BASE),
        help="Probing k_feat values. Default: 5 10 20 40 80 160 320 640.",
    )
    p.add_argument("--S", type=int, default=DEFAULT_S)
    p.add_argument("--n-steps", type=int, default=None,
                   help="Override n_steps (e.g. 200 for smoke).")
    p.add_argument("--smoke", action="store_true",
                   help="Pass smoke=True to eval_cfg.")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    args = p.parse_args(argv)

    cfg = MLC_TRAINING_CFG
    if args.n_steps is not None:
        cfg = cfg.model_copy(update={"n_steps": args.n_steps})
        log.info("[c3_mlc_base.run] training_cfg n_steps override: %d",
                 args.n_steps)

    log.info("[c3_mlc_base.run] training_cfg=%s",
             cfg.model_dump(exclude_none=True))

    for seed in args.seeds:
        for k in args.k_feats:
            run_one_cell(
                seed=seed,
                k_feat=k,
                S=args.S,
                smoke=args.smoke,
                training_cfg=cfg,
                force_train=args.force_train,
                force_eval=args.force_eval,
            )

    log.info("[c3_mlc_base.run] all cells complete")


if __name__ == "__main__":
    sys.exit(main())
