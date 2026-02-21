"""v0 eval_sae -- backward-compatible wrapper."""

from collections.abc import Callable

import torch
from sae_lens import BatchTopKTrainingSAE

from src.shared.eval_sae import EvalResult, eval_sae as _eval_sae
from src.v0_toy_model.toy_model import ToyModel

__all__ = ["EvalResult", "eval_sae"]


def eval_sae(
    sae: BatchTopKTrainingSAE,
    toy_model: ToyModel,
    generate_batch_fn: Callable[[int], torch.Tensor],
    n_samples: int = 100_000,
    batch_size: int = 4096,
) -> EvalResult:
    """Backward-compatible: computes true_l0 from feature_acts, delegates to shared eval."""
    device = next(sae.parameters()).device
    toy_model = toy_model.to(device)

    # Compute true_l0 from a sample
    with torch.no_grad():
        sample = generate_batch_fn(batch_size)
        true_l0 = (sample > 0).float().sum(dim=-1).mean().item()

    def generate_hidden_acts(bs: int) -> torch.Tensor:
        with torch.no_grad():
            feature_acts = generate_batch_fn(bs)
            return toy_model(feature_acts)

    return _eval_sae(sae, generate_hidden_acts, n_samples, batch_size, true_l0=true_l0)
