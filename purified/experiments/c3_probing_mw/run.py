"""C3 multi-window deployment driver — txc_base_mw + txc_pro_mw.

Helper for agent_nlp's C3 sparse-probing component. Trains the
multi-window TXC architectures (`txc_base_mw`, `txc_pro_mw`) at the
canonical schedule so agent_nlp's headline gets MW data without
agent_nlp re-running cells themselves.

Re-uses agent_nlp's `experiments.c3_probing.run` plumbing verbatim
via imports — only the `arch_name` differs from their canonical
sweep. Same datasource, same `TrainingConfig`, same `my_train_fn`,
same `my_eval_fn`, same `EVAL_PROTOCOL_VERSION`.

agent_paper integrates results at paper-render time via
`canonical_train_keys()` toggled to include MW archs.

    cd /workspace/temp_xc/purified
    source scripts/set_agent_env.sh agent_em_100k
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c3_probing_mw.run \\
        --archs txc_base_mw txc_pro_mw --seeds 42 1 2
"""

from __future__ import annotations

import argparse
import logging
import sys

from experiments.c3_probing.run import (
    COMPONENT,
    DATASOURCE,
    DEFAULT_K_FEATS,
    DEFAULT_S,
    EVAL_PROTOCOL_VERSION,
    _real_training_cfg,
    my_eval_fn,
    my_train_fn,
)
from temp_bench import runner
from temp_bench.config import compute_act_cache_key, load_datasource

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("c3_mw.run")

MW_ARCHS = ("txc_base_mw", "txc_pro_mw")
DEFAULT_SEEDS = (42, 1, 2)


def run_one_cell(
    arch_name: str,
    *,
    seed: int,
    k_feat: int,
    S: int = DEFAULT_S,
    smoke: bool = False,
    training_cfg=None,
    force_train: bool = False,
    force_eval: bool = False,
):
    cfg = training_cfg if training_cfg is not None else _real_training_cfg()
    # Mirror agent_nlp's c3_probing/run.py:292-301 — inject act_cache_key +
    # datasource_name so my_eval_fn can resolve d_in + probe cache. Runner
    # already injects _state_dict / _arch_name / _arch_hparams.
    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))
    eval_cfg = {
        "k_feat": k_feat,
        "S": S,
        "smoke": smoke,
        "_act_cache_key": act_cache_key,
        "_datasource_name": DATASOURCE,
    }

    log.info(
        "[c3_mw.run] CELL: arch=%s seed=%d k_feat=%d S=%d n_steps=%d smoke=%s",
        arch_name, seed, k_feat, S, cfg.n_steps, smoke,
    )

    result = runner.run_cell(
        component=COMPONENT,
        arch_name=arch_name,
        seed=seed,
        datasource_name=DATASOURCE,
        training_cfg=cfg,
        eval_cfg=eval_cfg,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=my_train_fn,
        eval_fn=my_eval_fn,
        force_train=force_train,
        force_eval=force_eval,
    )
    log.info(
        "[c3_mw.run] CELL DONE: arch=%s seed=%d k_feat=%d train_key=%s eval_key=%s cached=%s",
        arch_name, seed, k_feat,
        result.train_key, result.eval_key, result.cached,
    )
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--archs", nargs="+", default=MW_ARCHS,
        choices=MW_ARCHS,
        help="MW archs to deploy. Default: both.",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Seeds. Default: 42 1 2.",
    )
    p.add_argument(
        "--k-feats", nargs="+", type=int, default=list(DEFAULT_K_FEATS),
        help="Probing k_feat values. Default: 5 20.",
    )
    p.add_argument("--S", type=int, default=DEFAULT_S)
    p.add_argument("--n-steps", type=int, default=None,
                   help="Override n_steps (e.g. 200 for smoke).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override batch_size (e.g. 512 if InfoNCE OOMs on txc_pro_mw).")
    p.add_argument("--smoke", action="store_true",
                   help="Pass smoke=True to eval_cfg (matches agent_nlp's smoke flag).")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    args = p.parse_args(argv)

    cfg = _real_training_cfg()
    overrides = {}
    if args.n_steps is not None:
        overrides["n_steps"] = args.n_steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if overrides:
        cfg = cfg.model_copy(update=overrides)
        log.info("[c3_mw.run] training_cfg overrides: %s", overrides)

    log.info("[c3_mw.run] training_cfg=%s", cfg.model_dump())

    # Train-then-eval ordering: outer loop arch×seed (one training each),
    # inner loop k_feat (eval cache-hits the same checkpoint, just different
    # eval_cfg). Net: 6 trainings, 12 evals.
    for arch in args.archs:
        for seed in args.seeds:
            for k in args.k_feats:
                run_one_cell(
                    arch,
                    seed=seed,
                    k_feat=k,
                    S=args.S,
                    smoke=args.smoke,
                    training_cfg=cfg,
                    force_train=args.force_train,
                    force_eval=args.force_eval,
                )

    log.info("[c3_mw.run] all cells complete")


if __name__ == "__main__":
    sys.exit(main())
