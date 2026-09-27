"""End-to-end smoke test for the C7 pipeline.

Runs ONE cell — topk_sae × seed=42 × magnitudes (−8, 0, +8) — through
the full ``runner.run_cell`` path with reduced training (n_steps=500)
and PR-AUC enabled (sentence_acts loaded from the pre-extracted cache).

Verifies the entire chain:
- ``_build_batch_iter`` reads from the act cache
- ``train_sae`` produces a state_dict
- ``cache.save_checkpoint`` writes a manifest + (on ephemeral) pushes
  to HF
- ``run_arch_evaluation`` loads R1-Distill-Llama, mines a feature,
  generates 61 × 3 = 183 panels under SteeringHook, dispatches Sonnet
  judge with persistence, computes Δgc + PR-AUC.

Result row lands in ``results/leaderboard.jsonl`` keyed by the
short-config eval_key. Sonnet judge calls land in
``results/runs/<eval_key>/judge_outputs.jsonl``.

Usage:
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c7_backtracking.smoke
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from temp_bench import runner
from temp_bench.case_studies.backtracking import (
    SonnetBacktrackingJudge,
    build_cohort,
    extract_labeled_sentence_acts,
    load_stage_a,
    run_arch_evaluation,
    split_pos_neg,
)
from temp_bench.config import (
    instantiate_arch,
    load_arch,
    load_datasource,
    run_dir,
)
from temp_bench.schemas import TrainingConfig

# Reuse run.py adapters.
from experiments.c7_backtracking.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    _build_batch_iter,
    _instantiate_from_state,
    my_train_fn,
)


def _smoke_eval_fn(*, model, eval_cfg, component):
    """Smoke-test eval_fn that pre-loads sentence acts + cohort and forwards
    to run_arch_evaluation with a small magnitude grid."""
    from temp_bench.config import compute_eval_key
    arch_name = eval_cfg["_arch_name"]
    seed = eval_cfg.get("seed", 42)
    state_dict = eval_cfg["_state_dict"]
    arch_module = _instantiate_from_state(arch_name, state_dict, component=component)

    _hash_eval_cfg = {
        k: v for k, v in eval_cfg.items()
        if not k.startswith("_") and k not in (
            "feature_mining_acts", "sentence_acts", "sentence_labels", "sentence_qids",
        )
    }
    eval_key = compute_eval_key(
        train_key=eval_cfg["_train_key"],
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        eval_cfg=_hash_eval_cfg,
    )
    workspace = run_dir(eval_key)
    workspace.mkdir(parents=True, exist_ok=True)
    judge = SonnetBacktrackingJudge(workspace=workspace)

    cohort = build_cohort()
    stage_a = load_stage_a()

    # Load pre-extracted sentence acts → split into D+/D-.
    sa_path = (
        Path("results/c7_backtracking/stage_a/sentence_acts_L10.npz").resolve()
    )
    if not sa_path.exists():
        raise RuntimeError(
            f"Pre-extracted sentence acts missing at {sa_path}. "
            "Run `extract_labeled_sentence_acts()` first (see briefing)."
        )
    sa = np.load(sa_path, allow_pickle=True)
    sentence_acts = {"X": sa["X"], "is_bt": sa["is_bt"], "keys": sa["keys"]}
    pos_neg = split_pos_neg(sentence_acts)

    # PR-AUC inputs: pool sentence acts to (n_sent, T=1, d) for the SAE encode.
    # We use the same windowed acts as mining to keep "what the probe sees"
    # equal to "what the steering feature was mined on".
    pr_X = sentence_acts["X"]
    pr_y = sentence_acts["is_bt"].astype(int)
    # qids extracted from "qid|trace|s_idx" key strings
    pr_qids = np.array([k.split("|")[0] for k in sentence_acts["keys"]], dtype=object)

    result = run_arch_evaluation(
        arch=arch_module,
        seed=seed,
        cohort=cohort,
        stage_a=stage_a,
        workspace=workspace,
        judge=judge,
        magnitudes=(-8.0, 0.0, 8.0),  # smoke grid
        cut_fraction=0.25,
        arch_name=arch_name,
        feature_mining_acts=pos_neg,
        sentence_acts=pr_X,
        sentence_labels=pr_y,
        sentence_qids=pr_qids,
        max_new_tokens=512,            # smaller cap for smoke test
        gen_batch_size=8,
    )
    return result.metrics, result.primary_metric


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("c7.smoke")
    log.info("[smoke] starting")
    cfg = TrainingConfig(
        n_steps=500,        # smoke: 500 steps
        batch_size=64,
        warmup_steps=50,
    )
    res = runner.run_cell(
        component="c7",
        arch_name="topk_sae",
        seed=42,
        datasource_name=DATASOURCE,
        training_cfg=cfg,
        eval_cfg={
            "magnitudes": [-8.0, 0.0, 8.0],
            "cut_fraction": 0.25,
            "_smoke": True,
        },
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=my_train_fn,
        eval_fn=_smoke_eval_fn,
        force_train=False,
        force_eval=False,
    )
    log.info("[smoke] done. eval_key=%s train_key=%s cached=%s",
             res.eval_key, res.train_key, res.cached)
    if res.metrics:
        for k, v in sorted(res.metrics.items()):
            log.info("  %s = %s", k, v)


if __name__ == "__main__":
    main()
