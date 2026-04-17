"""Four aggregation strategies for collapsing per-window TempXC
activations into the per-token `(B, L, d_sae)` shape SAEBench expects.

Strategies:
  - `last`:        position t gets the last-position encoding of the
                   window ending at t  (matches single-token SAE).
  - `mean`:        position t gets the mean over T positions of the
                   window centered at t.
  - `max`:         max over T positions of the window centered at t.
  - `full_window`: no collapse — position t gets the concatenated
                   T × d_sae vector from the window centered at t.
                   Effective feature-dim becomes T × d_sae.

Contract: input is the per-window encoder output `(B, L, T, d_sae)`
produced by windowing an `(B, L, d_in)` LLM-activation sequence and
encoding each T-wide window independently. Output is `(B, L, d_sae)`
for last/mean/max and `(B, L, T × d_sae)` for full_window.

SAE and MLC have no T axis (T=1 implicitly); for those the wrapper
calls `aggregate` with T=1 and all four strategies become identity.

See docs/aniket/experiments/sparse_probing/plan.md § 5 for rationale.
"""

from __future__ import annotations

from typing import Literal

import torch


AggregationName = Literal["last", "mean", "max", "full_window"]


def aggregate(
    windows: torch.Tensor,          # (B, L, T, d_sae)
    strategy: AggregationName,
) -> torch.Tensor:
    """Apply an aggregation strategy to per-window encoder output.

    Args:
        windows: shape (B, L, T, d_sae). Per-token windowed encoder
            output — one T-slice per token position.
        strategy: which aggregation to apply.

    Returns:
        Tensor of shape (B, L, d_sae) for last/mean/max, or
        (B, L, T × d_sae) for full_window.
    """
    assert windows.ndim == 4, f"expected (B, L, T, d_sae), got {tuple(windows.shape)}"
    B, L, T, d_sae = windows.shape

    if strategy == "last":
        return _aggregate_last(windows)
    if strategy == "mean":
        return _aggregate_mean(windows)
    if strategy == "max":
        return _aggregate_max(windows)
    if strategy == "full_window":
        return _aggregate_full(windows)
    raise ValueError(f"unknown aggregation strategy: {strategy}")


def _aggregate_last(windows: torch.Tensor) -> torch.Tensor:
    """(B, L, T, d_sae) → (B, L, d_sae) by taking the last T-position."""
    B, L, T, d_sae = windows.shape
    out = windows[:, :, T - 1, :].contiguous()  # (B, L, d_sae)
    assert out.shape == (B, L, d_sae)
    return out


def _aggregate_mean(windows: torch.Tensor) -> torch.Tensor:
    """(B, L, T, d_sae) → (B, L, d_sae) by averaging over T."""
    B, L, T, d_sae = windows.shape
    out = windows.mean(dim=2).contiguous()
    assert out.shape == (B, L, d_sae)
    return out


def _aggregate_max(windows: torch.Tensor) -> torch.Tensor:
    """(B, L, T, d_sae) → (B, L, d_sae) by max over T."""
    B, L, T, d_sae = windows.shape
    out = windows.max(dim=2).values.contiguous()
    assert out.shape == (B, L, d_sae)
    return out


def _aggregate_full(windows: torch.Tensor) -> torch.Tensor:
    """(B, L, T, d_sae) → (B, L, T × d_sae) by flattening the T axis."""
    B, L, T, d_sae = windows.shape
    out = windows.reshape(B, L, T * d_sae).contiguous()
    assert out.shape == (B, L, T * d_sae)
    return out


def effective_d_sae(base_d_sae: int, T: int, strategy: AggregationName) -> int:
    """Effective feature dimension after aggregation.

    full_window concatenates T × d_sae into a single feature axis so
    the SAEBench wrapper reports a different `cfg.d_sae`; everything
    else keeps `d_sae` unchanged.
    """
    if strategy == "full_window":
        return base_d_sae * T
    return base_d_sae
