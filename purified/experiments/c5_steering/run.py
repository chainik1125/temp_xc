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
   into the eval-fn. We compute it ourselves from the same inputs the
   runner uses, then thread it through ``eval_cfg["_workspace"]`` so
   :class:`SteeringCaseStudy` can persist its jsonl artifacts there.
3. **Activations come from disk** — the Gemma-2-2b-IT L13 fineweb
   act-cache is on HF (``han1823123123/temp-bench-data``) and ``sync_
   from_hf.sh`` (or the manual ``hf download``) lands it at
   ``results/act_cache/<act_cache_key>/``. The :func:`build_batch_iter`
   helper memory-maps ``acts.npy`` and yields random ``(B, T, d_in)``
   windows.

Provenance: phase7's V7 driver lives at
``origin/han-phase7-unification:experiments/phase7_unification/case_
studies/steering/intervene_paper_clamp_window_tiled_broadcast.py`` —
this file replaces that driver with the framework-discipline version.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
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
    act_cache_dir,
    compute_act_cache_key,
    compute_eval_key,
    compute_train_key,
    instantiate_arch,
    load_arch,
    load_datasource,
)
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


# ── Activation-cache batch iterator ───────────────────────────────────


def build_batch_iter(
    act_cache_key: str, *, T: int, batch_size: int, seed: int,
):
    """Yield ``(batch_size, T, d_in)`` random windows from ``acts.npy``.

    The full activation tensor is ``(N, S, d_in) fp16`` — for the
    Gemma-2-2b-IT L13 fineweb cache that's ``(24000, 128, 2304)`` ≈
    14 GB on disk. We memory-map it and copy each batch to GPU as
    fp32 (the trainer's autocast handles bf16/fp16).

    For per-token archs (``T=1``), each batch is ``(B, 1, d_in)`` so
    the trainer + arch see a uniform shape.
    """
    cache_dir = act_cache_dir(act_cache_key)
    acts_path = cache_dir / "acts.npy"
    meta = json.loads((cache_dir / "meta.json").read_text())
    N, S, d_in = meta["shape"]
    if T < 1 or T > S:
        raise ValueError(f"window T={T} must be in [1, {S}]")

    # mmap so 14 GB doesn't sit fully in RAM. Random access patterns
    # are cheap on local NVMe / pod ramfs.
    acts = np.load(acts_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    device = torch.device("cuda")

    def _iter(_step: int) -> torch.Tensor:
        # Sample (sequence_idx, start_position) pairs, gather windows.
        seq_idx = rng.integers(0, N, size=batch_size)
        pos_max = S - T + 1
        pos = rng.integers(0, pos_max, size=batch_size)
        out = np.empty((batch_size, T, d_in), dtype=np.float16)
        for bi in range(batch_size):
            si, pi = seq_idx[bi], pos[bi]
            out[bi] = acts[si, pi : pi + T]
        return torch.from_numpy(out.astype(np.float32)).to(device)

    return _iter


# ── Train + eval adapters ─────────────────────────────────────────────


def my_train_fn(
    *, arch_name, arch_hparams, seed, training_cfg, act_cache_key, component,
):
    """Build arch via ``instantiate_arch`` + delegate to ``train_sae``."""
    spec = load_arch(arch_name, component=component)
    datasource = load_datasource(DATASOURCE)
    d_in = int(datasource.d_in) if hasattr(datasource, "d_in") else 2304
    model = instantiate_arch(spec, d_in=d_in)
    model.cuda()

    # Window length: per-token archs (tsae_paper) have T=1, window archs
    # (txc_base T=5, txc_pro T_max=10) have T>1. The arch knows; ask it.
    T = int(getattr(model, "T", arch_hparams.get("T", arch_hparams.get("T_max", 1))))
    batch_iter = build_batch_iter(
        act_cache_key, T=T, batch_size=training_cfg.batch_size, seed=seed,
    )
    log.info("[train] %s seed=%d T=%d", arch_name, seed, T)
    result = train_sae(model, batch_iter, training_cfg)
    return result["state_dict"]


def _make_eval_fn(*, seed: int, eval_protocol_version: str = EVAL_PROTOCOL_VERSION):
    """Eval-fn closure: knows the seed (run-cell doesn't thread it
    through ``eval_cfg``) so :class:`SteeringCaseStudy` can label
    judge_outputs.jsonl entries correctly.
    """
    def my_eval_fn(*, model, eval_cfg, component):  # noqa: ARG001
        arch_name: str = eval_cfg["_arch_name"]
        state_dict = eval_cfg["_state_dict"]
        workspace = Path(eval_cfg["_workspace"])

        spec = load_arch(arch_name, component=component)
        datasource = load_datasource(DATASOURCE)
        d_in = int(datasource.d_in) if hasattr(datasource, "d_in") else 2304
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
    force_train: bool,
    force_eval: bool,
) -> None:
    eval_cfg = {
        "protocol": protocol,
        "n_concepts": n_concepts,
        "strengths": list(strengths),
        "coh_thresholds": list(coh_thresholds),
    }
    training_cfg = runner.default_training_cfg(arch_name)
    workspace, eval_key = _workspace_for(
        arch_name=arch_name, seed=seed,
        training_cfg=training_cfg, eval_cfg=eval_cfg,
    )
    enriched_cfg = {**eval_cfg, "_workspace": str(workspace)}
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
        eval_cfg=enriched_cfg,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=my_train_fn,
        eval_fn=_make_eval_fn(seed=seed),
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
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
