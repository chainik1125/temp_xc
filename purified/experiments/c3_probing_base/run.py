"""C3 BASE replication — sparse probing on google/gemma-2-2b L13.

Mirrors agent_nlp's IT C3 setup arch-for-arch. The ONLY thing that
differs vs the IT canonical (``experiments.c3_probing.run``) is the
DATASOURCE: we point at ``gemma_2_2b_base_l13_fineweb_24k128`` (BASE
Gemma, NOT the ``-it`` instruction-tuned variant). agent_nlp /
agent_em_100k own the IT side; this driver gives reviewers a
cross-model robustness check.

Per-arch TrainingConfig MUST match agent_nlp + agent_em_100k's IT
conventions exactly (decisions § 15 + § 16):

    topk_sae:   B=1024, train_window_size=1
    tsae_paper: B=1024, train_window_size=2
    tfa:        B=32,   train_window_size=None
    txc_base:   B=1024, train_window_size=None  (internal T=5 sampling)
    txc_pro:    B=1024, train_window_size=None  (internal T=5+shift sampling)

Eval is UNCHANGED. We import ``my_train_fn`` and ``my_eval_fn`` from
``experiments.c3_probing.run`` verbatim — they read the cache key /
datasource_name from ``eval_cfg`` so the BASE override flows through
without any agent_nlp-side change.

agent_paper integrates BASE results at paper-render time via an
extended ``canonical_train_keys`` filter (split by datasource).

Usage::

    cd /workspace/temp_xc/purified
    source scripts/set_agent_env.sh agent_steer_100k

    # Smoke (TopK seed=42 k=5 n_steps=200) — validates pipeline end-to-end
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c3_probing_base.run \\
        --archs topk_sae --seeds 42 --k-feats 5 --n-steps 200

    # Full Tier-1 sweep (5 archs × 3 seeds × 2 k_feats — ~22-25 hr)
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c3_probing_base.run \\
        > logs/c3_base_full.log 2>&1 &
"""

from __future__ import annotations

import argparse
import logging
import sys

from experiments.c3_probing.run import (
    COMPONENT,
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
log = logging.getLogger("c3_base.run")


# BASE datasource — only thing that differs vs agent_nlp's IT setup.
DATASOURCE = "gemma_2_2b_base_l13_fineweb_24k128"

DEFAULT_ARCHS: tuple[str, ...] = (
    "topk_sae", "tsae_paper", "tfa", "txc_base", "txc_pro",
)
DEFAULT_SEEDS: tuple[int, ...] = (42, 1, 2)


# Per-arch TrainingConfig. Same as agent_nlp / agent_em_100k at IT —
# only the DATASOURCE changes. Cross-model fairness invariant
# (decisions § 15 + § 16).
ARCH_TRAINING_CFGS: dict[str, TrainingConfig] = {
    "topk_sae":   TrainingConfig(n_steps=20_000, train_window_size=1),
    "tsae_paper": TrainingConfig(n_steps=20_000, train_window_size=2),
    "tfa":        TrainingConfig(n_steps=20_000, batch_size=32),
    "txc_base":   TrainingConfig(n_steps=20_000),
    "txc_pro":    TrainingConfig(n_steps=20_000),
}


def run_one_cell(
    *,
    arch: str,
    seed: int,
    k_feat: int,
    S: int = DEFAULT_S,
    smoke: bool = False,
    training_cfg: TrainingConfig | None = None,
    force_train: bool = False,
    force_eval: bool = False,
):
    cfg = training_cfg if training_cfg is not None else ARCH_TRAINING_CFGS[arch]
    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))
    eval_cfg = {
        "k_feat": k_feat,
        "S": S,
        "smoke": smoke,
        "_act_cache_key": act_cache_key,
        "_datasource_name": DATASOURCE,
    }

    log.info(
        "[c3_base.run] CELL: arch=%s seed=%d k_feat=%d S=%d "
        "n_steps=%d batch_size=%d train_window_size=%s ds=%s",
        arch, seed, k_feat, S, cfg.n_steps, cfg.batch_size,
        cfg.train_window_size, DATASOURCE,
    )

    result = runner.run_cell(
        component=COMPONENT,
        arch_name=arch,
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
    tag = "CACHED" if result.cached else "NEW"
    m = result.metrics or {}
    mean_auc = m.get("mean_auc")
    std_auc = m.get("std_auc")
    n_tasks = m.get("n_tasks")
    if mean_auc is not None:
        log.info(
            "[c3_base.run] [%s] arch=%s seed=%d k_feat=%d  "
            "mean_AUC=%.4f±%.4f (n=%d tasks)  eval_key=%s train_key=%s",
            tag, arch, seed, k_feat,
            mean_auc, std_auc or 0.0, int(n_tasks or 0),
            result.eval_key, result.train_key,
        )
    else:
        log.info(
            "[c3_base.run] [%s] arch=%s seed=%d k_feat=%d  "
            "eval_key=%s train_key=%s",
            tag, arch, seed, k_feat, result.eval_key, result.train_key,
        )
    return result


def main(argv=None):
    p = argparse.ArgumentParser(
        description="C3 BASE replication — 5-arch sparse probing on "
                    "google/gemma-2-2b L13.",
    )
    p.add_argument(
        "--archs", nargs="+", default=list(DEFAULT_ARCHS),
        choices=list(ARCH_TRAINING_CFGS.keys()),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument(
        "--k-feats", nargs="+", type=int, default=list(DEFAULT_K_FEATS),
    )
    p.add_argument("--S", type=int, default=DEFAULT_S)
    p.add_argument(
        "--n-steps", type=int, default=None,
        help="Override n_steps for smoke tests (e.g. 200).",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Pass smoke=True to eval_cfg (synthetic-label probe path).",
    )
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    args = p.parse_args(argv)

    log.info(
        "[c3_base.run] sweep archs=%s seeds=%s k_feats=%s ds=%s "
        "n_steps_override=%s smoke=%s",
        args.archs, args.seeds, args.k_feats, DATASOURCE,
        args.n_steps, args.smoke,
    )

    for arch in args.archs:
        cfg = ARCH_TRAINING_CFGS[arch]
        if args.n_steps is not None:
            cfg = cfg.model_copy(update={"n_steps": args.n_steps})
        for seed in args.seeds:
            for k in args.k_feats:
                run_one_cell(
                    arch=arch,
                    seed=seed,
                    k_feat=k,
                    S=args.S,
                    smoke=args.smoke,
                    training_cfg=cfg,
                    force_train=args.force_train,
                    force_eval=args.force_eval,
                )

    log.info("[c3_base.run] all cells complete")


if __name__ == "__main__":
    sys.exit(main())
