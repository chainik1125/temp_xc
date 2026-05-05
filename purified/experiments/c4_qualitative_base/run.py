"""C4 BASE replication — qualitative latents on google/gemma-2-2b L13.

Mirrors agent_nlp's IT C4 (``experiments.c4_qualitative.run``) with
exactly TWO overrides:

  1. ``subject_model_name = "google/gemma-2-2b"`` (BASE, NOT ``-it``).
     Gemma-2-2b and Gemma-2-2b-it share the same tokenizer, so the
     pre-tokenized concat_corpora under ``data/concat_corpora/`` are
     drop-in compatible without re-tokenization.
  2. ``DATASOURCE = "gemma_2_2b_base_l13_fineweb_24k128"`` (the BASE
     C3 training source). Runner-level cache-hit on my BASE C3
     checkpoints — no re-training needed.

Per-arch TrainingConfig matches agent_nlp's C4 IT setup verbatim,
which in turn matches my C3 BASE setup, so train_keys cache-hit.

The qualitative judge eval (Anthropic Haiku) is unchanged.

Usage::

    cd /workspace/temp_xc/purified
    source scripts/set_agent_env.sh agent_steer_100k

    # Smoke ONE arch (no judge cost):
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c4_qualitative_base.run \\
        --archs txc_base --seeds 42 --n-features 8

    # Full sweep (5 archs × 3 seeds × n_features=256, ~1.5 hr,
    # ~$0.40-2.00 in Anthropic Haiku judge calls per agent_nlp est.):
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c4_qualitative_base.run \\
        > logs/c4_base_full.log 2>&1 &
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from experiments.c3_probing.run import my_train_fn  # cache-hits on C3 BASE checkpoints
from experiments.c4_qualitative.run import (
    EVAL_PROTOCOL_VERSION,
    DEFAULT_N_FEATURES,
    SMOKE_TRAIN_STEPS,
    SMOKE_BATCH,
)
from temp_bench import runner
from temp_bench.config import (
    act_cache_dir,
    compute_act_cache_key,
    compute_eval_key,
    compute_train_key,
    instantiate_arch,
    load_arch,
    load_datasource,
)
from temp_bench.eval import qualitative
from temp_bench.schemas import TrainingConfig
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("c4_base.run")


COMPONENT = "c4"
# Same training source as the BASE C3 cells — runner cache-hits on
# the C3 BASE checkpoints.
DATASOURCE = "gemma_2_2b_base_l13_fineweb_24k128"

# BASE Gemma — only thing that differs vs agent_nlp's IT C4.
SUBJECT_MODEL_NAME = "google/gemma-2-2b"
SUBJECT_LAYER = 13

DEFAULT_ARCHS: tuple[str, ...] = (
    "topk_sae", "tsae_paper", "tfa", "txc_base", "txc_pro",
)
DEFAULT_SEEDS: tuple[int, ...] = (42, 1, 2)


# Per-arch TrainingConfigs. Same as agent_nlp + agent_em_100k at IT —
# only the DATASOURCE differs (and the BASE subject_model_name).
# txc_base gets the § 17 T-sweep (T=10, T=20) in addition to the
# canonical T=5 default; cells cache-hit on my BASE C3 checkpoints
# (same DATASOURCE + same training_cfg → same train_key).
ARCH_TRAINING_CFGS: dict[str, list[TrainingConfig]] = {
    "topk_sae":   [TrainingConfig(n_steps=20_000, train_window_size=1)],
    "tsae_paper": [TrainingConfig(n_steps=20_000, train_window_size=2)],
    "tfa":        [TrainingConfig(n_steps=20_000, batch_size=32)],
    "txc_base":   [
        TrainingConfig(n_steps=20_000),                                       # T=5 default
        TrainingConfig(n_steps=20_000, arch_hparams_override={"T": 10}),
        TrainingConfig(n_steps=20_000, arch_hparams_override={"T": 20}),
    ],
    "txc_pro":    [TrainingConfig(n_steps=20_000)],
}


def _cfg_tag(cfg: TrainingConfig) -> str:
    if cfg.arch_hparams_override and "T" in cfg.arch_hparams_override:
        return f"T{cfg.arch_hparams_override['T']}"
    if cfg.train_window_size:
        return f"tws={cfg.train_window_size}"
    return ""


def _d_in_from_act_cache(act_cache_key: str) -> int:
    meta = json.loads((act_cache_dir(act_cache_key) / "meta.json").read_text())
    return int(meta["d_in"])


def my_eval_fn(*, model, eval_cfg, component):
    """C4 BASE qualitative eval — clones agent_nlp's pattern with two
    overrides: (1) BASE subject model name; (2) merge runner-supplied
    ``_arch_hparams`` onto the spec so ``arch_hparams_override`` (e.g.
    txc_base T=10 / T=20) flows through to model instantiation.
    agent_nlp's eval reads the docstring's ``_arch_hparams`` but doesn't
    actually use it; this local version applies it.
    """
    del model  # we re-instantiate
    arch_name = eval_cfg["_arch_name"]
    arch_hparams = eval_cfg["_arch_hparams"]
    state_dict = eval_cfg["_state_dict"]
    act_cache_key = eval_cfg["_act_cache_key"]
    eval_key_hint = eval_cfg["_eval_key_hint"]
    n_features = int(eval_cfg.get("n_features", DEFAULT_N_FEATURES))

    spec = load_arch(arch_name, component=component)
    spec = spec.model_copy(update={"hparams": arch_hparams})
    d_in = _d_in_from_act_cache(act_cache_key)
    m = instantiate_arch(spec, d_in=d_in).cuda().eval()
    m.load_state_dict(state_dict)

    metrics = qualitative.top_256_semantic(
        m,
        eval_key=eval_key_hint,
        subject_model_name=SUBJECT_MODEL_NAME,   # ← BASE override
        subject_layer=SUBJECT_LAYER,
        concat_corpora=("concat_A", "concat_B", "concat_random"),
        n_features=n_features,
    )
    return metrics, "top_N_semantic"


def run_one_cell(
    *,
    arch: str,
    seed: int,
    n_features: int = DEFAULT_N_FEATURES,
    smoke: bool = False,
    training_cfg: TrainingConfig | None = None,
    force_train: bool = False,
    force_eval: bool = False,
):
    if smoke and training_cfg is None:
        cfg = TrainingConfig(
            n_steps=SMOKE_TRAIN_STEPS,
            batch_size=SMOKE_BATCH,
            learning_rate=3e-4,
            warmup_steps=20,
            precision="bf16",
        )
    else:
        # default: pick the FIRST cfg (canonical / T=5 for txc_base)
        cfg = training_cfg if training_cfg is not None else ARCH_TRAINING_CFGS[arch][0]

    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))

    # Pre-compute the eval_key so the qualitative judge_outputs.jsonl
    # path is set correctly. Mirror the runner's arch_hparams_override
    # merge so train_key matches (commit dfd60850 / runner.py:116).
    arch_spec = load_arch(arch, component=COMPONENT)
    if cfg.arch_hparams_override:
        merged = {**arch_spec.hparams, **cfg.arch_hparams_override}
        arch_spec = arch_spec.model_copy(update={"hparams": merged})
    train_key = compute_train_key(
        arch=arch_spec,
        seed=seed,
        training_cfg=cfg,
        act_cache_key=act_cache_key,
    )
    eval_cfg: dict = {
        "n_features": n_features,
        "_act_cache_key": act_cache_key,
        "_datasource_name": DATASOURCE,
    }
    eval_key = compute_eval_key(
        train_key=train_key,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        eval_cfg=eval_cfg,
    )
    eval_cfg["_eval_key_hint"] = eval_key

    log.info(
        "[c4_base.run] CELL: arch=%s%s seed=%d n_features=%d "
        "n_steps=%d arch_hparams_override=%s ds=%s subject=%s",
        arch, f" ({_cfg_tag(cfg)})" if _cfg_tag(cfg) else "",
        seed, n_features, cfg.n_steps,
        cfg.arch_hparams_override, DATASOURCE, SUBJECT_MODEL_NAME,
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
    sem = m.get("top_N_semantic")
    agree = m.get("judge_agreement")
    njudged = m.get("n_features_judged")
    sem_s = (
        f"{int(sem)}/{int(njudged)}"
        if sem is not None and njudged is not None
        else "-"
    )
    agree_s = f"{agree:.3f}" if agree is not None else "-"
    log.info(
        "[c4_base.run] [%s] arch=%s seed=%d n_features=%d  "
        "SEMANTIC=%s  agreement=%s  eval_key=%s train_key=%s",
        tag, arch, seed, n_features, sem_s, agree_s,
        result.eval_key, result.train_key,
    )
    return result


def main(argv=None):
    p = argparse.ArgumentParser(
        description="C4 BASE replication — 5-arch qualitative latents on "
                    "google/gemma-2-2b L13.",
    )
    p.add_argument(
        "--archs", nargs="+", default=list(DEFAULT_ARCHS),
        choices=list(ARCH_TRAINING_CFGS.keys()),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument(
        "--n-features", type=int, default=DEFAULT_N_FEATURES,
        help="Number of top SAE features to judge per cell.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Use SMOKE training cfg (n_steps=200, B=64) — won't "
             "cache-hit on real C3 BASE checkpoints.",
    )
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    p.add_argument(
        "--cfg-tags", nargs="+", default=None,
        help="Restrict to specific cfg tags per arch "
             "(e.g. 'T10 T20' to run only the txc_base T-sweep cells, "
             "or '' for the canonical default cfg). If omitted, runs "
             "ALL cfgs registered for each requested arch.",
    )
    args = p.parse_args(argv)

    log.info(
        "[c4_base.run] sweep archs=%s seeds=%s n_features=%d ds=%s "
        "subject=%s cfg_tags=%s smoke=%s",
        args.archs, args.seeds, args.n_features, DATASOURCE,
        SUBJECT_MODEL_NAME, args.cfg_tags, args.smoke,
    )

    for arch in args.archs:
        for cfg in ARCH_TRAINING_CFGS[arch]:
            tag = _cfg_tag(cfg)
            if args.cfg_tags is not None and tag not in args.cfg_tags:
                log.info(
                    "[c4_base.run] SKIP arch=%s cfg=%s (not in --cfg-tags %s)",
                    arch, tag or "(default)", args.cfg_tags,
                )
                continue
            for seed in args.seeds:
                run_one_cell(
                    arch=arch,
                    seed=seed,
                    n_features=args.n_features,
                    smoke=args.smoke,
                    training_cfg=cfg,
                    force_train=args.force_train,
                    force_eval=args.force_eval,
                )

    log.info("[c4_base.run] all cells complete")


if __name__ == "__main__":
    sys.exit(main())
