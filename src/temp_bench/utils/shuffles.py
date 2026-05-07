"""Permutation helpers for ablation controls.

The TXC detection / steering protocols rely on a paired
within-window shuffle ablation: if a TXC's PR-AUC at top-S features
survives token-order permutation **inside each T-window**, the
detection signal is window-density rather than temporal.

The helper here is deliberately small and side-effect-free so that
case-study eval modules (C5, C6, C7) can adopt it without negotiating
shape conventions with each other.

Usage::

    from temp_bench.utils.shuffles import shuffle_within_window

    # x is a batch of T-windows already extracted from the residual
    # stream — (B, T, d_in). Each row's T positions are independently
    # permuted with a deterministic seed.
    x_shuffled = shuffle_within_window(x, T=5, seed=42)
"""

from __future__ import annotations

import torch


def shuffle_within_window(
    x: torch.Tensor,
    T: int,
    seed: int,
    *,
    per_row: bool = True,
) -> torch.Tensor:
    """Permute token order *within* each T-window.

    Args:
        x: ``(B, T, d_in)`` batch of T-windows. ``x.shape[1]`` must
            equal ``T``.
        T: window length (defensive — must match ``x.shape[1]``).
        seed: RNG seed. Same seed → same permutation pattern.
        per_row: if True (default), every row gets its own permutation
            of ``[0..T)``. If False, a single permutation is applied
            to every row (cheaper; matches the wasteland's "global
            shuffle" used in some early ablations). The paper's headline
            ablation uses per-row permutations because they decorrelate
            the "row b sees position 0 at slot 2" pattern across the
            batch — important when downstream pooling is per-row.

    Returns:
        Tensor with the same shape and dtype as ``x``. Original
        ``x`` is not mutated.
    """
    if x.dim() != 3:
        raise ValueError(
            f"shuffle_within_window expects (B, T, d_in); got {tuple(x.shape)}"
        )
    if x.shape[1] != T:
        raise ValueError(
            f"shuffle_within_window: x.shape[1]={x.shape[1]} != T={T}"
        )
    B = x.shape[0]
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    if not per_row:
        perm_cpu = torch.randperm(T, generator=g)
        perm = perm_cpu.to(x.device)
        return x.index_select(dim=1, index=perm).contiguous()
    # Per-row permutations. Sample on CPU for cross-device determinism,
    # then advanced-index. (B, T) gather indices; expand over d_in.
    perms_cpu = torch.stack(
        [torch.randperm(T, generator=g) for _ in range(B)], dim=0
    )
    perms = perms_cpu.to(x.device)
    batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, T)
    return x[batch_idx, perms].contiguous()


def shuffle_within_window_numpy(
    x,                          # numpy array (B, T, d_in)
    T: int,
    seed: int,
    *,
    per_row: bool = True,
):
    """NumPy variant — accepts an ``ndarray`` of shape ``(B, T, d_in)``.

    Mirrors :func:`shuffle_within_window` so callers that have
    already loaded sentence-window activations as ``np.ndarray`` (e.g.
    from a cached ``.npz`` file) don't have to round-trip through
    torch.
    """
    import numpy as np

    if x.ndim != 3:
        raise ValueError(
            f"shuffle_within_window_numpy expects (B, T, d_in); got {x.shape}"
        )
    if x.shape[1] != T:
        raise ValueError(
            f"shuffle_within_window_numpy: x.shape[1]={x.shape[1]} != T={T}"
        )
    rng = np.random.default_rng(int(seed))
    out = np.empty_like(x)
    if not per_row:
        perm = rng.permutation(T)
        return x[:, perm, :]
    for b in range(x.shape[0]):
        out[b] = x[b, rng.permutation(T)]
    return out
