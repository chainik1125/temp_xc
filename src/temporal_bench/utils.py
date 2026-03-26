"""Shared utilities: seeding, device selection, orthogonalization."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def orthogonalize(n: int, d: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
    """Generate n orthogonal unit vectors in R^d.

    If n <= d, returns exactly orthogonal vectors via QR decomposition.
    If n > d, raises ValueError (use random directions with target cosine sim instead).

    Returns:
        (n, d) tensor of unit-norm row vectors.
    """
    if n > d:
        raise ValueError(f"Cannot create {n} orthogonal vectors in R^{d}; need n <= d.")
    M = torch.randn(d, n, generator=generator)
    Q, _ = torch.linalg.qr(M)
    return Q[:, :n].T  # (n, d)


def unit_norm_rows(W: torch.Tensor) -> torch.Tensor:
    """Normalize each row of W to unit norm."""
    return F.normalize(W, dim=-1)
