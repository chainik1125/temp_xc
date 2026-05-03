"""C5 — RLHF steering: per-cell train + V7-tiled-broadcast eval.

Thin orchestration. The trainer lives in :mod:`temp_bench.training.
sae_trainer`; the eval pipeline (V7 hook + Sonnet judge + coh-vs-success
curves) lives in :mod:`temp_bench.case_studies.steering`. This file
just glues:

  for arch in {tsae_paper, txc_base, txc_pro}:
      for seed in {1, 2, 42}:
          runner.run_cell(component="c5", arch=arch, seed=seed,
                          training_cfg=…, eval_cfg=…, train_fn=…, eval_fn=…)

Three notable C5-specifics:

1. **Pre-test V7 on TXC-pro** — the c5.md hypothesis allows falling
   back to PP if V7 produces a degenerate (incoherent) success rate.
   ``--pre-test-only`` runs a 5-concept-cell at one mid-strength on
   ``txc_pro`` and prints whether to switch to PP. The full sweep is
   then run with ``--protocol pp`` if PP is recommended.
2. **Workspace plumbing** — :func:`temp_bench.runner.run_cell` doesn't
   pass ``eval_key`` (and therefore the workspace ``run_dir(eval_key)``)
   into the eval-fn. We compute it from the same inputs the runner
   uses (so the keys match) and thread it via a closure on the
   eval-fn (NOT via ``eval_cfg``, which the runner re-hashes — adding
   ``_workspace`` there would change the runner's eval_key and the
   case study would write to a different ``run_dir`` than the one
   ``metrics.json`` lands in).
3. **Activations come from disk** — the Gemma-2-2b-IT L13 fineweb
   act-cache is on HF (``han1823123123/temp-bench-data``) and ``sync_
   from_hf.sh`` (or the manual ``hf download``) lands it at
   ``results/act_cache/<act_cache_key>/``. We use the canonical
   :func:`temp_bench.data.nlp.batch_iter_from_act_cache` which yields
   full ``(B, seq_len, d_in)`` sequences — the TXC arch's
   ``train_step`` extracts a single random T-window per batch element
   internally (per ``txc_base.py`` docstring).

Provenance: phase7's V7 driver lives at
``origin/han-phase7-unification:experiments/phase7_unification/case_
studies/steering/intervene_paper_clamp_window_tiled_broadcast.py`` —
this file replaces that driver with the framework-discipline version.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("TQDM_DISABLE", "1")

from temp_bench import runner
from temp_bench.cache import run_dir
from temp_bench.case_studies.steering import (
    DEFAULT_COH_THRESHOLDS,
    DEFAULT_STRENGTHS,
    SteeringCaseStudy,
    SteeringConfig,
)
from temp_bench.config import (
    compute_act_cache_key,
    compute_eval_key,
    compute_train_key,
    instantiate_arch,
    load_arch,
    load_datasource,
)
from temp_bench.data.nlp import batch_iter_from_act_cache
from temp_bench.schemas import TrainingConfig
from temp_bench.training.sae_trainer import train_sae

log = logging.getLogger("c5_steering")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


# ── Per-component configuration ───────────────────────────────────────


COMPONENT = "c5"
DATASOURCE = "gemma_2_2b_it_l13_fineweb_24k128"
EVAL_PROTOCOL_VERSION = "1.0.0"

DEFAULT_ARCHS: tuple[str, ...] = ("tsae_paper", "txc_base", "txc_pro")
DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 42)


# ── Train + eval adapters ─────────────────────────────────────────────


def my_train_fn(
    *, arch_name, arch_hparams, seed, training_cfg, act_cache_key, component,
):
    """Build arch via ``instantiate_arch`` + delegate to ``train_sae``.

    Uses the canonical :func:`temp_bench.data.nlp.batch_iter_from_act_cache`
    which yields ``(B, seq_len, d_in)`` full sequences. The TXC archs'
    ``train_step`` does its own random-window extraction; the per-token
    archs flatten ``(B, S, d)`` into ``(B*S, d)`` internally. Same
    iterator works uniformly across families.
    """
    spec = load_arch(arch_name, component=component)
    datasource = load_datasource(DATASOURCE)
    d_in = int(getattr(datasource, "d_in", 2304))
    model = instantiate_arch(spec, d_in=d_in)
    model.cuda()
    log.info("[train] %s seed=%d T=%d", arch_name, seed, model.T)
    batch_iter = batch_iter_from_act_cache(act_cache_key, seed=seed)
    result = train_sae(model, batch_iter, training_cfg)
    return result["state_dict"]


def _make_eval_fn(*, seed: int, workspace: Path):
    """Eval-fn closure: knows the seed + workspace (run_cell doesn't
    thread either through ``eval_cfg``) so :class:`SteeringCaseStudy`
    can persist its jsonl artifacts to the same ``run_dir(eval_key)``
    the runner uses for ``metrics.json``.
    """
    def my_eval_fn(*, model, eval_cfg, component):  # noqa: ARG001
        arch_name: str = eval_cfg["_arch_name"]
        state_dict = eval_cfg["_state_dict"]

        spec = load_arch(arch_name, component=component)
        datasource = load_datasource(DATASOURCE)
        d_in = int(getattr(datasource, "d_in", 2304))
        arch = instantiate_arch(spec, d_in=d_in)
        arch.load_state_dict(state_dict)
        arch.cuda()
        arch.eval()

        cfg = SteeringConfig(
            protocol=eval_cfg.get("protocol", "v7"),
            strengths=tuple(eval_cfg.get("strengths", DEFAULT_STRENGTHS)),
            coh_thresholds=tuple(eval_cfg.get("coh_thresholds", DEFAULT_COH_THRESHOLDS)),
            n_concepts=int(eval_cfg.get("n_concepts", 30)),
        )
        cs = SteeringCaseStudy(workspace, cfg=cfg)
        try:
            cs.setup()
            result = cs.evaluate(arch, seed=seed, arch_name=arch_name)
        finally:
            cs.teardown()
            del arch
            torch.cuda.empty_cache()
        return result.metrics, result.primary_metric

    return my_eval_fn


# ── Workspace helper ──────────────────────────────────────────────────


def _workspace_for(
    *, arch_name: str, seed: int, training_cfg: TrainingConfig, eval_cfg: dict[str, Any],
) -> tuple[Path, str]:
    """Compute the same eval_key the runner will, so we can pre-create
    the workspace and pass it via eval_cfg without any cross-territory
    runner edits. ``eval_cfg`` here must be the dict the runner will
    hash — i.e., **without** the ``_*`` enrichments. Returns
    ``(workspace_path, eval_key)``."""
    arch_spec = load_arch(arch_name, component=COMPONENT)
    datasource = load_datasource(DATASOURCE)
    act_cache_key = compute_act_cache_key(datasource)
    train_key = compute_train_key(
        arch=arch_spec, seed=seed,
        training_cfg=training_cfg, act_cache_key=act_cache_key,
    )
    eval_key = compute_eval_key(
        train_key=train_key,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        eval_cfg=eval_cfg,
    )
    workspace = run_dir(eval_key)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace, eval_key


# ── Main loop ─────────────────────────────────────────────────────────


def run_one_cell(
    *,
    arch_name: str,
    seed: int,
    protocol: str,
    n_concepts: int,
    strengths: tuple[float, ...],
    coh_thresholds: tuple[float, ...],
    n_steps: int | None,
    smoke: bool,
    force_train: bool,
    force_eval: bool,
) -> None:
    eval_cfg: dict[str, Any] = {
        "protocol": protocol,
        "n_concepts": n_concepts,
        "strengths": list(strengths),
        "coh_thresholds": list(coh_thresholds),
    }
    if smoke:
        # Tagged so analysis.py can filter out smoke cells. Same
        # convention as agent_nlp's c3 smoke rows.
        eval_cfg["smoke"] = True
    training_cfg = runner.default_training_cfg(arch_name)
    if n_steps is not None:
        # Override n_steps for smoke testing. Different n_steps → different
        # train_key → fresh checkpoint, no clash with full-sweep cells.
        training_cfg = training_cfg.model_copy(update={"n_steps": n_steps})
    workspace, eval_key = _workspace_for(
        arch_name=arch_name, seed=seed,
        training_cfg=training_cfg, eval_cfg=eval_cfg,
    )
    log.info(
        "[cell] arch=%s seed=%d protocol=%s eval_key=%s",
        arch_name, seed, protocol, eval_key[:16],
    )
    runner.run_cell(
        component=COMPONENT,
        arch_name=arch_name,
        seed=seed,
        datasource_name=DATASOURCE,
        training_cfg=training_cfg,
        eval_cfg=eval_cfg,                    # un-enriched; runner re-hashes this
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=my_train_fn,
        eval_fn=_make_eval_fn(seed=seed, workspace=workspace),
        force_train=force_train,
        force_eval=force_eval,
    )


def main(args: argparse.Namespace) -> None:
    archs = tuple(args.archs) if args.archs else DEFAULT_ARCHS
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    strengths = tuple(args.strengths) if args.strengths else DEFAULT_STRENGTHS
    coh_thresholds = (
        tuple(args.coh_thresholds) if args.coh_thresholds else DEFAULT_COH_THRESHOLDS
    )

    smoke = args.smoke or args.n_steps is not None
    if args.pre_test_only:
        # Fast V7 health-check: 5 concepts × 1 strength on TXC-pro seed=42.
        protocol = "v7"
        run_one_cell(
            arch_name="txc_pro",
            seed=42,
            protocol=protocol,
            n_concepts=5,
            strengths=(strengths[len(strengths) // 2],),
            coh_thresholds=coh_thresholds,
            n_steps=args.n_steps,
            smoke=True,                                       # pre-test ≠ paper cell
            force_train=args.force_train,
            force_eval=args.force_eval,
        )
        return

    for arch in archs:
        for seed in seeds:
            run_one_cell(
                arch_name=arch,
                seed=seed,
                protocol=args.protocol,
                n_concepts=args.n_concepts,
                strengths=strengths,
                coh_thresholds=coh_thresholds,
                n_steps=args.n_steps,
                smoke=smoke,
                force_train=args.force_train,
                force_eval=args.force_eval,
            )


def cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=None,
                    help=f"Default: {' '.join(DEFAULT_ARCHS)}")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help=f"Default: {DEFAULT_SEEDS}")
    ap.add_argument("--protocol", choices=("v7", "pp"), default="v7")
    ap.add_argument("--n-concepts", type=int, default=30)
    ap.add_argument("--strengths", type=float, nargs="*", default=None,
                    help="Latent-space clamp values; default = paper § B.2 grid")
    ap.add_argument("--coh-thresholds", type=float, nargs="*", default=None)
    ap.add_argument("--pre-test-only", action="store_true",
                    help="Health-check V7 on TXC-pro (5 concepts × 1 strength)")
    ap.add_argument("--n-steps", type=int, default=None,
                    help="Override TrainingConfig.n_steps (default 30k); use a "
                         "small value (e.g. 500) for fast smoke tests. "
                         "Implies --smoke.")
    ap.add_argument("--smoke", action="store_true",
                    help="Tag this cell's leaderboard row with smoke=true so "
                         "analysis.py filters it out of paper aggregates.")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
