"""Evaluate intermediate training snapshots for the C7 extended-training matrix.

Han 2026-05-04 PM 4×H100 ask: with snapshots saved every N training steps
(see ``run.py --snapshot-every``), this wrapper loads selected milestone
checkpoints and runs the full Δgc + PR-AUC evaluation against each. The
output feeds the paper's "metric vs training step" plot.

Per-step recon-loss curves come from the train log persisted by
``my_train_fn`` (``logs/c7_b<bs>_<arch>_seed<seed>_trainlog.json`` and a
running mirror at ``checkpoints/<train_key>/snapshots/train_log.json``).
Per-step Δgc + PR-AUC curves come from this script.

Usage::

    .venv/bin/python -m experiments.c7_backtracking.evaluate_snapshot \\
        --arch txc_base --seed 42 --batch-size 1024 --n-steps 300000 \\
        --steps 10000 30000 100000 200000 300000

For each requested step, writes:
    checkpoints/<train_key>/snapshots/metrics_step_<step>.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from temp_bench.case_studies.backtracking import (
    DEFAULT_MAGNITUDE_GRID,
    DEFAULT_PR_AUC_S_GRID,
    SonnetBacktrackingJudge,
    build_cohort,
    extract_labeled_sentence_acts,
    load_stage_a,
    run_arch_evaluation,
    split_pos_neg,
)
from temp_bench.config import (
    act_cache_dir,
    checkpoint_dir,
    compute_act_cache_key,
    compute_eval_key,
    compute_train_key,
    instantiate_arch,
    load_arch,
    load_datasource,
    run_dir,
)
from temp_bench.schemas import TrainingConfig

from experiments.c7_backtracking.run import COMPONENT, DATASOURCE, EVAL_PROTOCOL_VERSION

log = logging.getLogger("c7.evaluate_snapshot")


def _instantiate_with_state(arch_name: str, state_dict: dict[str, torch.Tensor]):
    spec = load_arch(arch_name, component=COMPONENT)
    ds = load_datasource(DATASOURCE)
    cache_dir = act_cache_dir(compute_act_cache_key(ds))
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    d_in = int(specs["d_model"])
    model = instantiate_arch(spec, d_in=d_in)
    # Snapshots are saved bf16 if the live arch was cast at training time.
    # The arch's parameters are fp32 by default, so cast each snapshot
    # tensor to match the current parameter dtype before loading.
    cast = {}
    for k, v in state_dict.items():
        target = dict(model.named_parameters()).get(k) or dict(model.named_buffers()).get(k)
        if target is None:
            cast[k] = v
            continue
        cast[k] = v.to(dtype=target.dtype)
    missing, unexpected = model.load_state_dict(cast, strict=False)
    if missing or unexpected:
        log.warning("[c7.eval_snap] state_dict mismatch missing=%s unexpected=%s",
                    missing, unexpected)
    if torch.cuda.is_available():
        model = model.cuda()
    # Re-cast heavy archs to bf16 to match training-time runtime memory.
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > 1e9 and torch.cuda.is_available():
        model = model.bfloat16()
    return model, spec


def evaluate_one_snapshot(
    *,
    arch_name: str,
    seed: int,
    n_steps: int,
    batch_size: int,
    step: int,
    magnitudes: tuple[float, ...] = DEFAULT_MAGNITUDE_GRID,
    cut_fraction: float = 0.25,
    pr_auc_S_grid: tuple[int, ...] = DEFAULT_PR_AUC_S_GRID,
    max_new_tokens: int = 1024,
    gen_batch_size: int = 8,
) -> dict:
    """Load snapshot ``step_<step>.safetensors`` and run the C7 eval.

    Output is persisted to ``checkpoints/<train_key>/snapshots/metrics_step_<step>.json``
    AND returned for in-process callers.
    """
    spec = load_arch(arch_name, component=COMPONENT)
    ds = load_datasource(DATASOURCE)
    act_cache_key = compute_act_cache_key(ds)
    training_cfg = TrainingConfig(n_steps=n_steps, batch_size=batch_size)
    train_key = compute_train_key(
        arch=spec, seed=seed, training_cfg=training_cfg, act_cache_key=act_cache_key,
    )
    snap_dir = checkpoint_dir(train_key) / "snapshots"
    snap_path = snap_dir / f"step_{step}.safetensors"
    if not snap_path.exists():
        raise FileNotFoundError(
            f"Snapshot missing: {snap_path}\n"
            f"  arch={arch_name} seed={seed} bs={batch_size} n_steps={n_steps}\n"
            f"  train_key={train_key}\n"
            f"  Available: {sorted(p.name for p in snap_dir.glob('step_*.safetensors')) if snap_dir.exists() else '<dir missing>'}"
        )
    log.info("[c7.eval_snap] loading %s", snap_path)
    state = load_file(str(snap_path))
    model, _ = _instantiate_with_state(arch_name, state)

    # Workspace per (train_key, step) so judge transcripts are isolated.
    eval_cfg_dict = {
        "magnitudes": list(magnitudes),
        "cut_fraction": cut_fraction,
        "pr_auc_S_grid": list(pr_auc_S_grid),
        "_snapshot_step": step,  # disambiguate from final-state eval
    }
    eval_key = compute_eval_key(
        train_key=train_key,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        eval_cfg=eval_cfg_dict,
    )
    workspace = run_dir(eval_key)
    workspace.mkdir(parents=True, exist_ok=True)

    sa = extract_labeled_sentence_acts()
    pos_neg = split_pos_neg(sa)
    pr_X = sa["X"]
    pr_y = sa["is_bt"].astype(int)
    pr_qids = np.array([k.split("|")[0] for k in sa["keys"]], dtype=object)

    judge = SonnetBacktrackingJudge(workspace=workspace)
    cohort = build_cohort()
    stage_a = load_stage_a()

    result = run_arch_evaluation(
        arch=model,
        seed=seed,
        cohort=cohort,
        stage_a=stage_a,
        workspace=workspace,
        judge=judge,
        magnitudes=magnitudes,
        cut_fraction=cut_fraction,
        arch_name=arch_name,
        feature_mining_acts=pos_neg,
        sentence_acts=pr_X,
        sentence_labels=pr_y,
        sentence_qids=pr_qids,
        pr_auc_S_grid=pr_auc_S_grid,
        max_new_tokens=max_new_tokens,
        gen_batch_size=gen_batch_size,
    )

    out = {
        "arch": arch_name,
        "seed": seed,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "step": step,
        "train_key": train_key,
        "eval_key": eval_key,
        "primary_metric": result.primary_metric,
        "metrics": result.metrics,
    }
    out_path = snap_dir / f"metrics_step_{step}.json"
    out_path.write_text(json.dumps(out))
    log.info("[c7.eval_snap] %s = %.4f → %s", result.primary_metric,
             result.metrics.get(result.primary_metric, float("nan")), out_path)
    return out


def main(*, archs, seed, n_steps, batch_size, steps,
         magnitudes=DEFAULT_MAGNITUDE_GRID, cut_fraction=0.25,
         pr_auc_S_grid=DEFAULT_PR_AUC_S_GRID,
         max_new_tokens=1024, gen_batch_size=8,
         skip_existing=True):
    for arch in archs:
        for step in steps:
            log.info("[c7.eval_snap] arch=%s seed=%d step=%d", arch, seed, step)
            # Skip if metrics file already exists (idempotent).
            spec = load_arch(arch, component=COMPONENT)
            ds = load_datasource(DATASOURCE)
            tk = compute_train_key(
                arch=spec, seed=seed,
                training_cfg=TrainingConfig(n_steps=n_steps, batch_size=batch_size),
                act_cache_key=compute_act_cache_key(ds),
            )
            metrics_path = checkpoint_dir(tk) / "snapshots" / f"metrics_step_{step}.json"
            if skip_existing and metrics_path.exists():
                log.info("[c7.eval_snap] cached → %s", metrics_path)
                continue
            try:
                evaluate_one_snapshot(
                    arch_name=arch, seed=seed, n_steps=n_steps,
                    batch_size=batch_size, step=step,
                    magnitudes=tuple(magnitudes), cut_fraction=cut_fraction,
                    pr_auc_S_grid=tuple(pr_auc_S_grid),
                    max_new_tokens=max_new_tokens, gen_batch_size=gen_batch_size,
                )
            except Exception as exc:
                log.exception("[c7.eval_snap] FAILED arch=%s step=%d: %s",
                              arch, step, exc)
            finally:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return 0


def cli():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", required=True,
                    help="Arch names (space-separated). Resolved via locked_archs.yaml.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, required=True,
                    help="The training-cfg n_steps used to produce the snapshots "
                         "(needed to recompute the train_key).")
    ap.add_argument("--batch-size", type=int, required=True,
                    help="The training-cfg batch_size used to produce the snapshots.")
    ap.add_argument("--steps", type=int, nargs="+", required=True,
                    help="Snapshot steps to evaluate (must exist in "
                         "checkpoints/<train_key>/snapshots/).")
    ap.add_argument("--cut-fraction", type=float, default=0.25)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--gen-batch-size", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluate even if metrics_step_<step>.json exists.")
    args = ap.parse_args()
    raise SystemExit(main(
        archs=args.archs,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        steps=args.steps,
        cut_fraction=args.cut_fraction,
        max_new_tokens=args.max_new_tokens,
        gen_batch_size=args.gen_batch_size,
        skip_existing=not args.force,
    ))


if __name__ == "__main__":
    cli()
