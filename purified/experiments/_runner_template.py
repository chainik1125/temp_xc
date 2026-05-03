"""Component runner template — copy this into experiments/cN_*/run.py.

A component's runner is ~30 lines. Bespoke logic goes in the
``eval_fn`` (in ``src/temp_bench/eval/<name>.py``); bespoke training
hooks go in the arch class (in ``src/temp_bench/architectures/``).
**Do not add caching, leaderboard append, or run-id allocation to this
file.** The framework does that for you. See
``docs/paper/framework.md`` for why.
"""

from __future__ import annotations

import argparse

from temp_bench import runner
from temp_bench.schemas import TrainingConfig


# ── Per-component configuration ─────────────────────────────────────────

COMPONENT = "cN"                      # "c1", "c2", ..., "c7"
DATASOURCE = "<datasource_name>"      # from configs/datasources.yaml
EVAL_PROTOCOL_VERSION = "1.0.0"       # bump on metric/protocol change

DEFAULT_ARCHS = runner.list_archs()   # all from yaml; component may filter
DEFAULT_SEEDS = (1, 2, 42)
DEFAULT_K_FEATS = (5, 20)


# ── Component-specific train + eval functions ───────────────────────────


def my_train_fn(*, arch_name, arch_hparams, seed, training_cfg, act_cache_key, component):
    """Component-specific training. Imports from temp_bench.training,
    constructs the arch via configs, runs the training loop, returns a
    state_dict.
    """
    raise NotImplementedError("port from src/temp_bench/training/")


def my_eval_fn(*, model, eval_cfg, component):
    """Component-specific evaluation. Builds the model from
    eval_cfg['_arch_name'] + eval_cfg['_arch_hparams'] +
    eval_cfg['_state_dict'], runs the evaluation, returns
    ``(metrics_dict, primary_metric_key)``.
    """
    raise NotImplementedError("implement in src/temp_bench/eval/<name>.py")


# ── Runner ──────────────────────────────────────────────────────────────


def main(*, archs=None, seeds=DEFAULT_SEEDS, k_feats=DEFAULT_K_FEATS,
         force_train=False, force_eval=False):
    archs = archs or DEFAULT_ARCHS
    for arch in archs:
        for seed in seeds:
            for k in k_feats:
                runner.run_cell(
                    component=COMPONENT,
                    arch_name=arch,
                    seed=seed,
                    datasource_name=DATASOURCE,
                    training_cfg=runner.default_training_cfg(arch),
                    eval_cfg={"k_feat": k, "S": 32},
                    eval_protocol_version=EVAL_PROTOCOL_VERSION,
                    train_fn=my_train_fn,
                    eval_fn=my_eval_fn,
                    force_train=force_train,
                    force_eval=force_eval,
                )


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--k_feats", type=int, nargs="*", default=DEFAULT_K_FEATS)
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()
    main(
        archs=args.archs,
        seeds=tuple(args.seeds),
        k_feats=tuple(args.k_feats),
        force_train=args.force_train,
        force_eval=args.force_eval,
    )


if __name__ == "__main__":
    cli()
