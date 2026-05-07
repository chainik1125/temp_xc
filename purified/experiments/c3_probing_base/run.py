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
    _d_in_from_act_cache,
    my_train_fn,
)
import numpy as np
from temp_bench import runner
from temp_bench.config import (
    compute_act_cache_key, instantiate_arch, load_arch, load_datasource,
)
from temp_bench.data.nlp import list_probe_cache, load_probe_cache
from temp_bench.eval import probing
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


# Per-arch TrainingConfigs. Same as agent_nlp / agent_em_100k at IT —
# only the DATASOURCE changes. Cross-model fairness invariant
# (decisions § 15 + § 16).
#
# txc_base gets MULTIPLE cfgs (T=5 default + T=10 + T=20 per § 17
# T-sweep mission, (decision 2026-05-05)). Other archs stay single-cfg.
# arch_hparams_override flows into compute_train_key (commit dfd60850),
# so each T gets a fresh train_key — no cache collisions.
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


def my_eval_fn(*, model, eval_cfg, component):
    """C3 BASE eval — clones agent_nlp's pattern but applies the
    runner-injected ``_arch_hparams`` (merged after
    ``arch_hparams_override``). agent_nlp's
    ``experiments/c3_probing/run.my_eval_fn`` claims to read
    ``_arch_hparams`` (per its docstring) but in fact calls
    ``load_arch(arch_name)`` which returns the YAML defaults — so the
    ``arch_hparams_override`` path is silently mis-instantiated for
    txc_base T=10 / T=20. This local version fixes that by merging the
    runner-supplied hparams onto the spec before instantiation.

    Surfaced in 'Open questions for the maintainer' — the upstream fix likely
    belongs in ``experiments/c3_probing/run.my_eval_fn``; until then
    my driver carries this local override.
    """
    del model
    arch_name = eval_cfg["_arch_name"]
    arch_hparams = eval_cfg["_arch_hparams"]
    state_dict = eval_cfg["_state_dict"]
    act_cache_key = eval_cfg["_act_cache_key"]
    datasource_name = eval_cfg["_datasource_name"]
    k_feat = int(eval_cfg["k_feat"])
    S = int(eval_cfg.get("S", DEFAULT_S))
    smoke = bool(eval_cfg.get("smoke", False))

    spec = load_arch(arch_name, component=component)
    spec = spec.model_copy(update={"hparams": arch_hparams})
    d_in = _d_in_from_act_cache(act_cache_key)
    m = instantiate_arch(spec, d_in=d_in).cuda().eval()
    m.load_state_dict(state_dict)

    if smoke:
        # Smoke path — synthetic binary labels, mirrors agent_nlp.
        from experiments.c3_probing.run import _smoke_probe_data
        X_train, y_train, X_test, y_test = _smoke_probe_data(act_cache_key, S=S)
        metrics = probing.s_tail_probe(
            m,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            S=S, k_feat=k_feat,
        )
        return metrics, "auc"

    task_names = list_probe_cache(datasource_name)
    if not task_names:
        raise FileNotFoundError(
            f"No probe cache found for datasource {datasource_name!r}. "
            f"Run build_probe_cache(...) first."
        )

    aucs: list[float] = []
    accs: list[float] = []
    per_task_metrics: dict[str, float] = {}
    for tname in task_names:
        task = load_probe_cache(datasource_name, tname)
        r = probing.s_tail_probe(
            m,
            X_train=task["X_train"], y_train=task["y_train"],
            X_test=task["X_test"], y_test=task["y_test"],
            first_real_train=task["first_real_train"],
            first_real_test=task["first_real_test"],
            S=S, k_feat=k_feat,
        )
        per_task_metrics[f"auc__{tname}"] = float(r["auc"])
        per_task_metrics[f"acc__{tname}"] = float(r["acc"])
        aucs.append(float(r["auc"]))
        accs.append(float(r["acc"]))

    aucs_arr = np.asarray(aucs, dtype=np.float64)
    accs_arr = np.asarray(accs, dtype=np.float64)
    metrics = {
        "mean_auc": float(aucs_arr.mean()),
        "std_auc": float(aucs_arr.std(ddof=1)) if len(aucs) > 1 else 0.0,
        "mean_acc": float(accs_arr.mean()),
        "std_acc": float(accs_arr.std(ddof=1)) if len(accs) > 1 else 0.0,
        "n_tasks": float(len(aucs)),
        **per_task_metrics,
    }
    return metrics, "mean_auc"


def _cfg_tag(cfg: TrainingConfig) -> str:
    """Short tag for log lines / cell identity (e.g. ``T10`` for an
    ``arch_hparams_override={'T': 10}``; otherwise ``""``)."""
    if cfg.arch_hparams_override and "T" in cfg.arch_hparams_override:
        return f"T{cfg.arch_hparams_override['T']}"
    if cfg.train_window_size:
        return f"tws={cfg.train_window_size}"
    return ""


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
    if training_cfg is None:
        # default: pick the FIRST cfg for this arch (canonical / T=5 for txc_base)
        training_cfg = ARCH_TRAINING_CFGS[arch][0]
    cfg = training_cfg
    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))
    eval_cfg = {
        "k_feat": k_feat,
        "S": S,
        "smoke": smoke,
        "_act_cache_key": act_cache_key,
        "_datasource_name": DATASOURCE,
    }

    log.info(
        "[c3_base.run] CELL: arch=%s%s seed=%d k_feat=%d S=%d "
        "n_steps=%d batch_size=%d train_window_size=%s "
        "arch_hparams_override=%s ds=%s",
        arch, f" ({_cfg_tag(cfg)})" if _cfg_tag(cfg) else "",
        seed, k_feat, S, cfg.n_steps, cfg.batch_size,
        cfg.train_window_size, cfg.arch_hparams_override, DATASOURCE,
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
    p.add_argument(
        "--cfg-tags", nargs="+", default=None,
        help="Restrict to specific cfg tags per arch "
             "(e.g. 'T10 T20' to run only the txc_base T-sweep cells, "
             "or '' for the canonical default cfg). If omitted, runs "
             "ALL cfgs registered for each requested arch.",
    )
    args = p.parse_args(argv)

    log.info(
        "[c3_base.run] sweep archs=%s seeds=%s k_feats=%s ds=%s "
        "n_steps_override=%s cfg_tags=%s smoke=%s",
        args.archs, args.seeds, args.k_feats, DATASOURCE,
        args.n_steps, args.cfg_tags, args.smoke,
    )

    for arch in args.archs:
        for cfg in ARCH_TRAINING_CFGS[arch]:
            tag = _cfg_tag(cfg)
            if args.cfg_tags is not None and tag not in args.cfg_tags:
                log.info(
                    "[c3_base.run] SKIP arch=%s cfg=%s (not in --cfg-tags %s)",
                    arch, tag or "(default)", args.cfg_tags,
                )
                continue
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
