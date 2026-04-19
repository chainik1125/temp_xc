"""Duck-type adapter exposing our SAEs under Venhoff's SAE contract.

Venhoff's annotation / clustering code expects an SAE object with:
  - `.encoder(x)` taking (B, d) → (B, d_sae)
  - `.W_dec`      decoder weights (d_in, d_sae)
  - `.b_dec`      decoder bias (d_in,)
  - `.activation_mean` dataset-global mean (d_in,)

Our `TopKSAE` / `TempXC` / `MLC` classes have `.encode(...)`, `.W_dec`,
`.b_dec` but no `.activation_mean`. This module wraps any of them with
a lightweight shim that:

  1. Exposes `.encoder(x)` — for Path 1 it's pass-through of `.encode`;
     for Path 3 (TempXC) it expects `(B, T, d)` and applies one of the
     four aggregation strategies from `saebench.aggregation`.
  2. Carries the `activation_mean` vector loaded from the sidecar
     pickle produced by `activation_collection.py`.
  3. Exposes `.W_dec` / `.b_dec` via attribute forwarding.

Use `wrap_for_path1(model, mean_pkl_path)` or
`wrap_for_path3(model, mean_pkl_path, T, aggregation)`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.bench.saebench.aggregation import AggregationName, aggregate


def _load_mean(mean_pkl: Path) -> torch.Tensor:
    with mean_pkl.open("rb") as f:
        payload = pickle.load(f)
    mean = np.asarray(payload["activation_mean"], dtype=np.float32)
    assert mean.ndim == 1, f"bad mean shape {mean.shape}"
    return torch.from_numpy(mean)


class _Path1Shim(torch.nn.Module):
    """For SAE: (B, d) → (B, d_sae). Input is already sentence-mean.

    SAE's native `.encode((B, d))` returns `(B, d_sae)`. MLC would need
    `(B, n_layers, d)` input which Path 1's per-sentence-mean collector
    doesn't produce — MLC support is deferred until multi-layer
    collection lands (see `train_small_sae.py::_train_sae`).
    """

    def __init__(self, base: torch.nn.Module, activation_mean: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("activation_mean", activation_mean)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, (
            f"path1 encoder expects (B, d), got {tuple(x.shape)}. "
            "If you're wrapping MLC here, use multi-layer activation "
            "collection and Path 3 instead."
        )
        return self.base.encode(x)

    @property
    def W_dec(self) -> torch.Tensor:
        return self.base.W_dec

    @property
    def b_dec(self) -> torch.Tensor:
        return self.base.b_dec


class _Path3Shim(torch.nn.Module):
    """For TempXC: (B, T, d) → (B, d_sae) via per-position encode + aggregation.

    TempXC's native `.encode(x)` returns the *shared-z* latent
    `(B, d_sae)` — already summed over positions inside the einsum. For
    our Path 3 aggregation ablation we need the **per-position**
    activations `(B, T, d_sae)` so `last`/`mean`/`max`/`full_window`
    have distinct semantics. We replicate `CrosscoderSpec._encode_window`
    inline: per-position pre-activation, shared-z TopK mask computed
    from the summed pre, mask broadcast across T, ReLU.
    """

    def __init__(
        self,
        base: torch.nn.Module,
        activation_mean: torch.Tensor,
        T: int,
        aggregation: AggregationName,
    ):
        super().__init__()
        self.base = base
        self.T = T
        self.aggregation = aggregation
        self.register_buffer("activation_mean", activation_mean)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        """`x` is `(B, T, d_in)` — one window per sentence."""
        import torch.nn.functional as F

        assert x.ndim == 3, f"path3 encoder expects (B, T, d), got {tuple(x.shape)}"
        B, T, d = x.shape
        assert T == self.T, f"T mismatch: got {T}, shim configured for {self.T}"

        # Per-position pre-activation (same einsum as shared-z but without
        # summing over T). Then shared-z TopK mask from the summed pre.
        pre_per_pos = torch.einsum("btd,tds->bts", x, self.base.W_enc)  # (B, T, d_sae)
        pre_sum = pre_per_pos.sum(dim=1) + self.base.b_enc              # (B, d_sae)
        if self.base.k is None:
            z_per_pos = F.relu(pre_per_pos)
        else:
            _, topk_idx = pre_sum.topk(self.base.k, dim=-1)
            mask = torch.zeros_like(pre_sum)
            mask.scatter_(1, topk_idx, 1.0)                             # (B, d_sae)
            z_per_pos = F.relu(pre_per_pos) * mask.unsqueeze(1)         # (B, T, d_sae)

        # aggregate() expects (B, L, T, d_sae); we have L=1 (one window per row).
        windows = z_per_pos.unsqueeze(1)                                # (B, 1, T, d_sae)
        collapsed = aggregate(windows, self.aggregation)                # (B, 1, d_sae) or (B, 1, T*d_sae)
        return collapsed.squeeze(1)

    @property
    def W_dec(self) -> torch.Tensor:
        return self.base.W_dec

    @property
    def b_dec(self) -> torch.Tensor:
        return self.base.b_dec


def wrap_for_path1(base: torch.nn.Module, mean_pkl: Path) -> _Path1Shim:
    return _Path1Shim(base, _load_mean(mean_pkl))


def wrap_for_path3(
    base: torch.nn.Module,
    mean_pkl: Path,
    T: int,
    aggregation: AggregationName,
) -> _Path3Shim:
    return _Path3Shim(base, _load_mean(mean_pkl), T, aggregation)


def argmax_labels(shim: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Argmax over encoder output — matches Venhoff's `annotate_thinking.py`.

    Returns a LongTensor of shape `(B,)` with the single "most active"
    latent per input. For Path 1 inputs `(B, d)` and for Path 3 `(B, T, d)`.
    """
    with torch.no_grad():
        z = shim.encoder(x)
    return z.argmax(dim=-1)
