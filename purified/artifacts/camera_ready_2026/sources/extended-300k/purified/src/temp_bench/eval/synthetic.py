"""Synthetic-data evaluation — C1 (toy Markov) + C2 (toy coupled HMM).

Shared between agent_paper's two toy components. **All synthetic
evaluation goes through this module** — components do not roll their
own NMSE / AUC computations. PROTOCOL.md § 11 *Code reuse contract*.

Public API (worker fills in):

- :func:`reconstruction_nmse(model, batch_iter, n_batches=10) -> float`
- :func:`feature_recovery_auc(model, true_W, k=None) -> dict`
- :func:`global_recovery_gAUC(model, true_W_global) -> float`  (C2)

Each returns standardised metrics that flow into ``run_cell`` →
``leaderboard.jsonl`` → ``analysis.py`` → AUTO-RESULTS in cN.md.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from temp_bench.architectures.base import TempBenchArch

BatchIter = Callable[[int], torch.Tensor]


def reconstruction_nmse(
    model: TempBenchArch,
    batch_iter: BatchIter,
    *,
    n_batches: int = 10,
    batch_size: int = 256,
    device: str | torch.device = "cuda",
) -> float:
    """Normalised MSE: ``E[||x - x_hat||^2] / E[||x||^2]``.

    Lower is better. Component-agnostic; same definition for C1, C2,
    and (if anyone wants) C3 reconstructive comparisons.
    """
    model.to(device).eval()
    sum_se = 0.0
    sum_signal = 0.0
    n_tokens = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x = batch_iter(batch_size).to(device)
            x_hat = model(x)
            sum_se += float((x_hat - x).pow(2).sum())
            sum_signal += float(x.pow(2).sum())
            n_tokens += x.numel() // x.shape[-1]
    return sum_se / max(sum_signal, 1e-12)


def feature_recovery_auc(
    model: TempBenchArch,
    true_W: torch.Tensor,
    *,
    k_active: int | None = None,
) -> dict[str, float]:
    """Per-feature recovery AUC against ground-truth feature vectors.

    Args:
        model: trained arch with ``decoder_directions() -> (d_sae, d_in)``.
        true_W: ground-truth matrix of shape ``(n_features, d_in)``.
        k_active: optional hint for sparsity at evaluation; not all
            archs use it.

    Returns:
        dict with ``"per_feature_auc"`` (mean), ``"min_auc"``, ``"std_auc"``.
        Workers fill in (TODO — port from
        ``origin/han-phase7-unification:src/v2_temporal_schemeC/run_benchmark.py``).
    """
    raise NotImplementedError(
        "feature_recovery_auc — port from "
        "origin/han-phase7-unification:src/v2_temporal_schemeC/run_benchmark.py"
    )


def global_recovery_gAUC(
    model: TempBenchArch,
    true_W_global: torch.Tensor,
) -> float:
    """C2 global feature recovery AUC.

    Different from per-feature AUC because C2's coupled HMM produces
    'global' features (combinations of latent states). Lower bound on
    paper's "global recovery" claim. TODO (port from Phase 3).
    """
    raise NotImplementedError(
        "global_recovery_gAUC — port from Phase 3 coupled-features benchmark."
    )
