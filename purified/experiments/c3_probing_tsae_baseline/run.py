"""C3 T-SAE baseline re-train at T=2 (Bhalla/Ye 2025 §3.1, decisions § 15).

Re-trains agent_nlp's `tsae_paper` arch with the new
`TrainingConfig.train_window_size=2` field (paper-faithful adjacent
pairs) so C3 has T-SAE cells matching the literature's per-token
batch convention. Imports agent_nlp's `experiments.c3_probing.run`
plumbing verbatim — only the `TrainingConfig` differs from canonical.

agent_nlp owns `topk_sae` at T=1 (their plumbing, their pod). Don't
re-run topk_sae from here. TXC archs (`txc_base`, `txc_pro`) keep
their existing canonical sweep at T=None.

agent_paper integrates results at paper-render time via three
`canonical_train_keys()` calls (TXC at T=None, TopK at T=1, T-SAE
at T=2).

    cd /workspace/temp_xc/purified
    source scripts/set_agent_env.sh agent_em_100k
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c3_probing_tsae_baseline.run \\
        --seeds 42 1 2 --k-feats 5 20
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
    my_eval_fn,
    my_train_fn,
)
from temp_bench import runner
from temp_bench.config import compute_act_cache_key, load_datasource
from temp_bench.schemas import TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("c3_tsae_baseline.run")

ARCH = "tsae_paper"
DEFAULT_SEEDS = (42, 1, 2)

# Bhalla/Ye 2025 §3.1: T-SAE paper-faithful adjacent pairs → T=2.
# n_steps=20_000 matches agent_nlp's `_real_training_cfg()` canonical
# (the schema default is 25000; agent_nlp pins 20000 — we mirror so
# eval comparison is apples-to-apples).
TSAE_TRAINING_CFG = TrainingConfig(n_steps=20_000, train_window_size=2)


def run_one_cell(
    *,
    seed: int,
    k_feat: int,
    S: int = DEFAULT_S,
    smoke: bool = False,
    training_cfg: TrainingConfig | None = None,
    force_train: bool = False,
    force_eval: bool = False,
):
    cfg = training_cfg if training_cfg is not None else TSAE_TRAINING_CFG
    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))
    # Mirror agent_nlp's c3_probing/run.py:292-301.
    eval_cfg = {
        "k_feat": k_feat,
        "S": S,
        "smoke": smoke,
        "_act_cache_key": act_cache_key,
        "_datasource_name": DATASOURCE,
    }

    log.info(
        "[c3_tsae_baseline.run] CELL: arch=%s seed=%d k_feat=%d S=%d "
        "n_steps=%d train_window_size=%s smoke=%s",
        ARCH, seed, k_feat, S, cfg.n_steps, cfg.train_window_size, smoke,
    )

    result = runner.run_cell(
        component=COMPONENT,
        arch_name=ARCH,
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
        "[c3_tsae_baseline.run] CELL DONE: arch=%s seed=%d k_feat=%d "
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
        "--k-feats", nargs="+", type=int, default=list(DEFAULT_K_FEATS),
        help="Probing k_feat values. Default: 5 20.",
    )
    p.add_argument("--S", type=int, default=DEFAULT_S)
    p.add_argument("--n-steps", type=int, default=None,
                   help="Override n_steps (e.g. 200 for smoke).")
    p.add_argument("--smoke", action="store_true",
                   help="Pass smoke=True to eval_cfg.")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    args = p.parse_args(argv)

    cfg = TSAE_TRAINING_CFG
    if args.n_steps is not None:
        cfg = cfg.model_copy(update={"n_steps": args.n_steps})
        log.info("[c3_tsae_baseline.run] training_cfg n_steps override: %d",
                 args.n_steps)

    log.info("[c3_tsae_baseline.run] training_cfg=%s",
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

    log.info("[c3_tsae_baseline.run] all cells complete")


if __name__ == "__main__":
    sys.exit(main())
