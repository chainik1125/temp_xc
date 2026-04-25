"""Phase 5B candidate D: strided-window TXC.

D1: T_eff=5, stride=2 — encoder sees positions {0, 2, 4, 6, 8} of a
10-token raw span. Same param count as vanilla TXCDR T=5 (5 W_enc
slabs of (d_in, d_sae)).

D2: variable stride at train time, sampled per batch from
`stride_choices` (e.g., {1, 2, 3}). At probe time pick a fixed stride.

D3 lives in `phase5b_h8_strided.py` (extends H8 with strided sampling).

The arch class is identical to vanilla TemporalCrosscoder shape-wise;
the strided sampling lives in the *data generator* (see
`make_strided_window_gen_gpu` in `_train_utils.py`). So we re-use the
existing TemporalCrosscoder class and only carry the (T_eff, stride)
metadata in the trained ckpt.

This file provides convenience wrappers + a probe-time encoder that
routes a 10/15/20-token raw context through the strided-sample
mechanism before applying the trained encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.crosscoder import TemporalCrosscoder
from src.architectures.txc_bare_antidead import TXCBareAntidead
from src.architectures.txc_bare_multidistance_contrastive_antidead import (
    TXCBareMultiDistanceContrastiveAntidead,
)


def stride_subsample(x_full: torch.Tensor, T_eff: int, stride: int,
                     anchor: str = "last") -> torch.Tensor:
    """Take a strided sub-window from a (B, L, d) raw context.

    Args:
        x_full: (B, L, d) — raw token sequence (or sliding window).
        T_eff: effective window length (post-stride).
        stride: gap between sampled positions.
        anchor: "last" places the last sampled token at L-1.
                "first" places the first sampled token at 0.

    Returns:
        (B, T_eff, d) — strided subset.
    """
    B, L, d = x_full.shape
    span = T_eff * stride
    if L < span:
        raise ValueError(f"need L>={span} for T_eff={T_eff}, stride={stride}; got L={L}")

    if anchor == "last":
        # last sampled position is L-1; first is L-1 - (T_eff-1)*stride
        positions = torch.arange(L - 1 - (T_eff - 1) * stride, L, stride,
                                  device=x_full.device, dtype=torch.long)
    elif anchor == "first":
        positions = torch.arange(0, span, stride, device=x_full.device, dtype=torch.long)
    else:
        raise ValueError(f"unknown anchor={anchor}")

    return x_full[:, positions, :]


class StridedTXCBareAntidead(TXCBareAntidead):
    """Thin wrapper: forward(x_raw_span) → strided sample → super().forward.

    Accepts (B, T_eff*stride, d) raw spans OR (B, T_eff, d) pre-sampled.
    """

    def __init__(self, d_in: int, d_sae: int, T_eff: int, k: int,
                 stride: int = 2, **kw):
        super().__init__(d_in, d_sae, T_eff, k, **kw)
        self.stride = int(stride)
        self.T_eff = int(T_eff)
        self.expected_span = self.T_eff * self.stride

    def encode_strided(self, x_raw_span: torch.Tensor) -> torch.Tensor:
        """(B, span, d) → (B, d_sae) latent. Used at probe time."""
        x = stride_subsample(x_raw_span, self.T_eff, self.stride, anchor="last")
        return self.encode(x)

    def forward(self, x: torch.Tensor):
        # If x has the strided length already, pass through.
        if x.ndim == 3 and x.shape[1] == self.T_eff:
            return super().forward(x)
        # If x has the raw span length, sample then forward.
        if x.ndim == 3 and x.shape[1] == self.expected_span:
            x = stride_subsample(x, self.T_eff, self.stride, anchor="last")
            return super().forward(x)
        raise ValueError(
            f"expected last dim=T_eff ({self.T_eff}) or span ({self.expected_span}); "
            f"got shape {tuple(x.shape)}"
        )


class StridedH8(TXCBareMultiDistanceContrastiveAntidead):
    """H8 + strided window (D3). Inherits all H8 logic.

    The data generator pre-samples the strided window, so the model
    accepts the same (B, 1+K, T_eff, d) shape as H8.
    """

    def __init__(self, d_in: int, d_sae: int, T_eff: int, k: int,
                 stride: int = 2, **kw):
        super().__init__(d_in, d_sae, T_eff, k, **kw)
        self.stride = int(stride)
        self.T_eff = int(T_eff)
        self.expected_span = self.T_eff * self.stride
