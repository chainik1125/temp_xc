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
    """For SAE / MLC: (B, d) → (B, d_sae). Input is already sentence-mean."""

    def __init__(self, base: torch.nn.Module, activation_mean: torch.Tensor):
        super().__init__()
        self.base = base
        self.register_buffer("activation_mean", activation_mean)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.encode(x)

    @property
    def W_dec(self) -> torch.Tensor:
        return self.base.W_dec

    @property
    def b_dec(self) -> torch.Tensor:
        return self.base.b_dec


class _Path3Shim(torch.nn.Module):
    """For TempXC: (B, T, d) → (B, d_sae) via per-window encode + aggregation.

    The per-window encoding is TempXC's `.encode(x)` call, which already
    returns `(B, T, d_sae)` — we reshape to `(B, 1, T, d_sae)` (single
    "window" per sentence) and feed through the aggregation helper.
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
        assert x.ndim == 3, f"path3 encoder expects (B, T, d), got {tuple(x.shape)}"
        B, T, d = x.shape
        assert T == self.T, f"T mismatch: got {T}, shim configured for {self.T}"
        z = self.base.encode(x)  # (B, T, d_sae)
        assert z.ndim == 3 and z.shape[0] == B and z.shape[1] == T
        windows = z.unsqueeze(1)  # (B, 1, T, d_sae)
        collapsed = aggregate(windows, self.aggregation)  # (B, 1, d_sae) or (B, 1, T*d_sae)
        out = collapsed.squeeze(1)
        return out

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
