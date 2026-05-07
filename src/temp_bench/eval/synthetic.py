"""Synthetic-data evaluation — C1 (toy Markov) + C2 (toy coupled HMM).

Shared between [pipeline]'s two toy components. **All synthetic
evaluation goes through this module** — components do not roll their
own NMSE / AUC computations. PROTOCOL.md § 11 *Code reuse contract*.

Public API:

- :func:`reconstruction_nmse(model, batch_iter, n_batches=10) -> float`
- :func:`feature_recovery(decoder_directions, true_features) -> dict`
- :func:`global_recovery_gAUC(decoder_directions, hidden_features) -> dict`

The two AUC helpers share the same canonical scoring (``feature_recovery``
ports ``origin/wasteland-canonical:src/eval/feature_recovery.py``):

For each true feature we find the best-matching decoder column by
*absolute* cosine similarity, sweep a threshold τ from 0 to 1, and
compute the AUC of (fraction of features recovered) vs τ.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
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
    with torch.no_grad():
        for _ in range(n_batches):
            x = batch_iter(batch_size).to(device)
            x_hat = model(x)
            sum_se += float((x_hat - x).pow(2).sum())
            sum_signal += float(x.pow(2).sum())
    return sum_se / max(sum_signal, 1e-12)


@torch.no_grad()
def feature_recovery(
    decoder_directions: torch.Tensor,
    true_features: torch.Tensor,
    *,
    n_thresholds: int = 50,
) -> dict[str, Any]:
    """Per-feature recovery AUC against ground-truth feature directions.

    Ports ``origin/wasteland-canonical:src/eval/feature_recovery.py``
    (``feature_recovery_score``) verbatim. For each true feature we find
    the SAE decoder column with the highest |cos sim|; we then sweep a
    threshold τ ∈ [0, 1] and integrate the fraction-recovered curve.

    Args:
        decoder_directions: ``(d_sae, d_in)``. Our archs return this
            shape via ``model.decoder_directions()``. The function
            internally transposes to the wasteland's ``(d, h)``
            convention.
        true_features: ``(n_features, d_in)``. Same convention as the
            data generators (``CoupledData.emission_features`` etc.).

    Returns:
        Dict with float metrics:
        - ``auc``: area under recovery curve (0–1; **higher is better**).
        - ``mean_max_cos``: avg best-match |cos sim| per true feature.
        - ``frac_recovered_90``: fraction of features with best |cos| ≥ 0.9.
        - ``frac_recovered_80``: fraction with best |cos| ≥ 0.8.
    """
    # Transpose to wasteland's (d, h) for cos_sims helper.
    decoder = decoder_directions.T                    # (d_in, d_sae)
    truth = true_features.T                           # (d_in, n_features)

    sims = _cos_sims(decoder, truth).abs()            # (d_sae, n_features)
    max_per_true = sims.max(dim=0).values             # (n_features,)

    thresholds = np.linspace(0, 1, n_thresholds)
    curve = np.array([
        (max_per_true.cpu().numpy() >= t).mean() for t in thresholds
    ])
    auc = float(np.trapezoid(curve, thresholds))

    return {
        "auc": auc,
        "mean_max_cos": float(max_per_true.mean()),
        "frac_recovered_90": float((max_per_true >= 0.9).float().mean()),
        "frac_recovered_80": float((max_per_true >= 0.8).float().mean()),
    }


@torch.no_grad()
def global_recovery_gAUC(
    decoder_directions: torch.Tensor,
    hidden_features: torch.Tensor,
    *,
    n_thresholds: int = 50,
) -> dict[str, Any]:
    """C2 global feature recovery — same scoring as ``feature_recovery``,
    just against the K hidden directions instead of the M emission
    directions. Returned dict keys are renamed with a ``g`` prefix:

    - ``gauc``, ``g_mean_max_cos``, ``g_frac_recovered_{90,80}``
    """
    base = feature_recovery(
        decoder_directions, hidden_features,
        n_thresholds=n_thresholds,
    )
    return {
        "gauc": base["auc"],
        "g_mean_max_cos": base["mean_max_cos"],
        "g_frac_recovered_90": base["frac_recovered_90"],
        "g_frac_recovered_80": base["frac_recovered_80"],
    }


def _cos_sims(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Pairwise cos sims between columns of mat1 and mat2.

    Args:
        mat1: ``(d, h1)`` — e.g. decoder columns.
        mat2: ``(d, h2)`` — e.g. true feature directions.
    Returns:
        ``(h1, h2)`` cosine similarity matrix.
    """
    n1 = mat1 / mat1.norm(dim=0, keepdim=True).clamp(min=1e-8)
    n2 = mat2 / mat2.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return n1.T @ n2
