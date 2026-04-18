"""Window-aggregation strategies for temporal SAEs inside SAEBench.

SAEBench hands a probing adapter input of shape `(B, L, d_in)` (one
row per sequence position) and expects `(B, L, d_sae)` back. Our
architectures (TXCDR, MLC, Matryoshka-TXCDR) encode **windows**, not
single positions, and some of them collapse the window axis at encode
time. This module defines how an N-position window encoding is
reduced to per-L-position SAE activations.

Four strategies, matching Aniket's pre-registration (plan §5):

- ``last_position``:   for each sequence position t, return the
                       last-position encoding of the T-token window
                       ending at t. Most conservative; matches
                       single-token SAE semantics exactly.
- ``mean_over_t``:     mean the T per-position encodings of the
                       window centred at t. Closest to TXCDR's
                       training objective.
- ``max_over_t``:      max-pool the window. "Did this feature fire
                       anywhere in the window."
- ``full_window``:     don't collapse — return (B, L, T*d_sae).
                       Most expressive; doubles effective d_sae.

For single-token SAEs (T = 1) all four strategies are identical.

Independent Phase 5 implementation; no code reuse from
`src/bench/saebench/aggregation.py`.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch

AggregationName = Literal[
    "last_position", "mean_over_t", "max_over_t", "full_window"
]


def _sliding_windows(
    X: torch.Tensor, T: int
) -> torch.Tensor:
    """Return a tensor of shape (B, L, T, d_in) of stride-1 windows.

    For positions near the start of the sequence (t < T-1), we
    left-pad the window with the earliest real position (clamping).
    This keeps the adapter's output length equal to the input length
    without introducing zero-padded distortion at the boundary.

    Args:
        X: (B, L, d_in)
        T: window size.
    """
    B, L, d = X.shape
    if T == 1:
        return X.unsqueeze(2)

    # For each output position t in [0..L-1], the window is
    # positions [max(0, t-T+1) .. t], left-clamped to 0.
    # Implementation: build an index tensor (L, T) then gather.
    idx = torch.arange(L, device=X.device).unsqueeze(1) - torch.arange(
        T - 1, -1, -1, device=X.device
    ).unsqueeze(0)
    idx.clamp_(min=0)  # shape (L, T)
    # gather over the L axis of X
    windows = X[:, idx, :]  # (B, L, T, d)
    return windows


def aggregate(
    encode_window: Callable[[torch.Tensor], torch.Tensor],
    X: torch.Tensor,
    T: int,
    strategy: AggregationName,
    window_chunk: int = 256,
) -> torch.Tensor:
    """Run a window-level encoder over stride-1 windows of `X` and reduce.

    Args:
        encode_window: callable taking (N, T, d_in) -> (N, T, d_sae)
            (per-position encodings). For shared-latent architectures
            that emit (N, d_sae), the callable must broadcast to
            (N, T, d_sae) internally (e.g. repeat the shared latent
            T times). The aggregation strategy then collapses the T
            axis.
        X: (B, L, d_in)
        T: window size.
        strategy: which aggregation to apply.
        window_chunk: how many windows to encode at once. Trade-off
            between GPU memory and throughput.

    Returns:
        For "last/mean/max": (B, L, d_sae)
        For "full_window":   (B, L, T * d_sae)
    """
    B, L, d_in = X.shape
    # Run the encoder in chunks of windows to keep peak memory bounded.
    windows = _sliding_windows(X, T)  # (B, L, T, d_in)
    windows = windows.reshape(B * L, T, d_in)

    outs = []
    for i in range(0, windows.shape[0], window_chunk):
        chunk = windows[i:i + window_chunk]
        z = encode_window(chunk)  # (N, T, d_sae) (broadcast if shared-latent)
        assert z.dim() == 3 and z.shape[0] == chunk.shape[0] \
            and z.shape[1] == T, (
            f"encode_window must return (N, T, d_sae); got {z.shape}"
        )
        outs.append(z)
    z_all = torch.cat(outs, dim=0)  # (B*L, T, d_sae)
    d_sae = z_all.shape[-1]
    z_all = z_all.reshape(B, L, T, d_sae)

    if strategy == "last_position":
        return z_all[:, :, -1, :]
    if strategy == "mean_over_t":
        return z_all.mean(dim=2)
    if strategy == "max_over_t":
        return z_all.max(dim=2).values
    if strategy == "full_window":
        return z_all.reshape(B, L, T * d_sae)
    raise ValueError(f"Unknown aggregation strategy: {strategy}")
