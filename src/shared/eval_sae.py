"""Evaluation utilities for SAEs."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from sae_lens import TopKTrainingSAE

from src.shared.metrics import variance_explained


@dataclass
class EvalResult:
    """Results from evaluating an SAE."""

    true_l0: float | None
    sae_l0: float
    dead_features: int
    shrinkage: float
    mse: float
    ve: float  # variance explained


def eval_sae(
    sae: TopKTrainingSAE,
    generate_hidden_acts_fn: Callable[[int], torch.Tensor],
    n_samples: int = 100_000,
    batch_size: int = 4096,
    true_l0: float | None = None,
) -> EvalResult:
    """Evaluate an SAE on hidden activations.

    Args:
        sae: Trained SAE to evaluate.
        generate_hidden_acts_fn: Callable taking batch_size, returning hidden
            activations of shape (batch_size, d_in).
        n_samples: Total number of evaluation samples.
        batch_size: Evaluation batch size.
        true_l0: Pre-computed true L0 (caller-provided, since computing it
            requires knowledge of the feature activation structure).

    Returns:
        EvalResult with L0, dead features, shrinkage, MSE, and VE.
    """
    device = next(sae.parameters()).device
    sae.eval()

    all_sae_l0 = []
    all_mse = []
    all_true_acts = []
    all_recon_acts = []
    feature_fired = torch.zeros(sae.cfg.d_sae, device=device)

    n_processed = 0
    with torch.no_grad():
        while n_processed < n_samples:
            current_batch = min(batch_size, n_samples - n_processed)
            hidden_acts = generate_hidden_acts_fn(current_batch)

            # SAE forward pass
            sae_out = sae(hidden_acts)
            # Get feature activations for L0 and dead feature counting
            sae_feature_acts = sae.encode(hidden_acts)
            sae_l0 = (sae_feature_acts > 0).float().sum(dim=-1).mean()
            all_sae_l0.append(sae_l0)

            # Track which features fired
            feature_fired += (sae_feature_acts > 0).float().sum(dim=0)

            # MSE
            mse = (hidden_acts - sae_out).pow(2).sum(dim=-1).mean()
            all_mse.append(mse)

            all_true_acts.append(hidden_acts)
            all_recon_acts.append(sae_out)

            n_processed += current_batch

    sae_l0 = torch.stack(all_sae_l0).mean().item()
    mse = torch.stack(all_mse).mean().item()
    dead_features = (feature_fired == 0).sum().item()

    all_true = torch.cat(all_true_acts, dim=0)
    all_recon = torch.cat(all_recon_acts, dim=0)

    # Shrinkage: mean ratio of reconstructed norm to true norm
    true_norms = all_true.norm(dim=-1)
    recon_norms = all_recon.norm(dim=-1)
    nonzero_mask = true_norms > 1e-8
    if nonzero_mask.any():
        shrinkage = (recon_norms[nonzero_mask] / true_norms[nonzero_mask]).mean().item()
    else:
        shrinkage = 0.0

    ve = variance_explained(all_true, all_recon)

    return EvalResult(
        true_l0=true_l0,
        sae_l0=sae_l0,
        dead_features=int(dead_features),
        shrinkage=shrinkage,
        mse=mse,
        ve=ve,
    )
