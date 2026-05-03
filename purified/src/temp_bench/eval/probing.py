"""Sparse probing evaluation — C3 (Gemma-IT-L13).

Shared so we don't re-implement the SAEBench-style probe for every
component that wants it. PROTOCOL.md § 11 *Code reuse contract*.

Public API (worker fills in):

- :func:`mean_pool_probe(model, X_train, y_train, X_test, y_test, k_feat) -> dict`
- :func:`s_tail_probe(model, sequences, labels, S, k_feat) -> dict`
- :func:`run_task_suite(model, tasks, k_feats, S, n_jobs=-1) -> list[dict]`

Convention:

- Activations are extracted via the model's ``encode``; we then
  mean-pool over the last ``S`` tokens (Phase 7 S-tail protocol) and
  fit a sparse logistic regression on the top-``k_feat`` mean-difference
  features.
- ``n_jobs`` defaults to ``-1`` (all CPU cores) — see PROTOCOL.md § 13
  multi-GPU; probing is CPU-bound, doesn't compete with training GPUs.
"""

from __future__ import annotations

from typing import Any

from temp_bench.architectures.base import TempBenchArch


def mean_pool_probe(
    model: TempBenchArch,
    *,
    X_train,
    y_train,
    X_test,
    y_test,
    k_feat: int,
    n_jobs: int = -1,
) -> dict[str, float]:
    """Train a sparse linear probe on top-``k_feat`` features after
    mean-pooling encoded activations over the sequence.

    TODO — port from
    ``origin/han-phase7-unification:src/probing/sparse_probing.py``.
    Returns dict with ``"auc"``, ``"f1"``, ``"acc"``, ``"k_feat"``.
    """
    raise NotImplementedError(
        "mean_pool_probe — port from "
        "origin/han-phase7-unification:src/probing/sparse_probing.py"
    )


def s_tail_probe(
    model: TempBenchArch,
    *,
    sequences,
    labels,
    S: int,
    k_feat: int,
    n_jobs: int = -1,
) -> dict[str, float]:
    """Phase 7 S-tail protocol: pool features over the last S tokens
    only, then probe.

    Validity rule: ``S >= T`` (Phase 7 corrected formula). The wrapper
    enforces this and raises on misconfiguration.
    """
    if S < model.T:
        raise ValueError(
            f"S={S} < T={model.T}; Phase 7 S-tail protocol requires S >= T."
        )
    raise NotImplementedError(
        "s_tail_probe — port from Phase 7 sparse-probing pipeline."
    )


def run_task_suite(
    model: TempBenchArch,
    *,
    tasks: list[Any],
    k_feats: tuple[int, ...] = (5, 20),
    S: int = 32,
    n_jobs: int = -1,
) -> list[dict[str, Any]]:
    """Run :func:`s_tail_probe` over every task × k_feat combination.

    Components pass ``tasks`` (e.g. SAEBench standard, or 16-task PAPER
    subset). Returns one dict per (task, k_feat) cell.
    """
    raise NotImplementedError(
        "run_task_suite — port from Phase 7 task-suite runner; "
        "agent_nlp pre-registers task suite per c3.md before launch."
    )
