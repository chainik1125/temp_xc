"""MLC — Multi-Layer Crosscoder (shared latent across L adjacent layers).

Source: ``origin/wasteland-canonical @ [scrubbed-sha]:src/architectures/mlc.py``
adapted to the unified :class:`TempBenchArch` framework.

MLC is the layer-axis analog of a Temporal Crosscoder. The "T" axis on
input/output is reinterpreted as "L" (layer-within-window): a single
shared latent describes the LLM's hidden state at L adjacent residual-
stream layers centred on some anchor layer.

Architecture (L = number of layers in the window):

    W_enc: (L, d, h)  — per-layer encoder projections
    W_dec: (h, L, d)  — per-layer decoder projections
    b_enc: (h,)       — shared encoder bias
    b_dec: (L, d)     — per-layer decoder bias

Encode: ``pre = sum_l einsum("bd,ds->bs", x[:, l, :], W_enc[l]) + b_enc``
        ``z   = TopK(pre)`` → ``(B, h)`` with ``k_pos × L`` non-zeros

Decode: ``x_hat[:, l, :] = z @ W_dec[:, l, :] + b_dec[l, :]`` → ``(B, L, d)``

NOTE — MLC's first-class data format is multi-LAYER ``(B, L, d)``,
not multi-TOKEN. the prior wasteland C7 pipeline registers hooks on
multiple layers simultaneously to build the (B, L=5, d) batches.
**This port treats the framework's T axis as L** for compatibility
with the unified single-layer activation cache pipeline. For paper-
faithful MLC eval at C7, build a multi-layer activation cache (L8/L9/
L10/L11/L12) — TODO under [pipeline] open question.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.architectures.base import ArchConfig, TempBenchArch


class MLC(TempBenchArch):
    """Shared-latent crosscoder across L layers.

    Args (from ``configs/locked_archs.yaml::mlc``):
        d_in:           residual width.
        d_sae:          dictionary size (= ``h``).
        k_pos:          window-level TopK budget (= window L0).
        n_layers:       L — number of layers in the window (default 5).
        center_layer:   the LLM layer index this MLC is centred on.
                        Used by C3 anchor-layer probing; not stored in
                        the model state.
    """

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        k_pos: int,
        n_layers: int = 5,
        center_layer: int = 10,
    ):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name="mlc", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=n_layers,
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self._n_layers = n_layers
        self.k_pos = k_pos
        self.k_win = k_pos * n_layers
        self.center_layer = center_layer

        self.W_enc = nn.Parameter(
            torch.randn(n_layers, d_in, d_sae) * (1.0 / d_in ** 0.5)
        )
        self.W_dec = nn.Parameter(
            torch.randn(d_sae, n_layers, d_in) * (1.0 / d_sae ** 0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(n_layers, d_in))

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        """Unit-norm per dictionary atom over (L, d_in)."""
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``x: (B, L, d_in) → (B, 1, d_sae)`` window-level TopK code.

        Output is ``(B, 1, d_sae)`` with the singleton "T" axis matching
        :class:`TempBenchArch`'s shared-z TXC convention.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, L, d = x.shape
        if L != self._n_layers:
            raise ValueError(
                f"MLC.encode expects L={self._n_layers}; got L={L}."
            )
        pre = torch.einsum("bld,lds->bs", x, self.W_enc) + self.b_enc
        vals, idx = pre.topk(self.k_win, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z.unsqueeze(1)  # (B, 1, d_sae)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``z: (B, 1, d_sae) → (B, L, d_in)``."""
        if z.dim() == 3:
            if z.shape[1] != 1:
                raise ValueError(
                    f"MLC.decode expects (B, 1, d_sae); got T={z.shape[1]}."
                )
            z = z.squeeze(1)
        return torch.einsum("bs,sld->bld", z, self.W_dec) + self.b_dec

    def post_step(self) -> None:
        self._normalize_decoder()

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Args:
            x: (B, seq_len, d_in) from canonical batch_iter.
               When batch_iter is single-layer (current C7 default), we
               sample one random L-window per batch element treating T
               as L. For paper-faithful MLC, swap to multi-layer cache.
        Returns:
            (loss, info) — info has 'mse', 'l0', 'z'.
        """
        if x.dim() != 3 or x.shape[1] < self._n_layers:
            raise ValueError(
                f"MLC.train_step expects (B, seq_len>={self._n_layers}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, seq_len, _ = x.shape
        offsets = torch.randint(
            0, seq_len - self._n_layers + 1, (B,), device=x.device
        )
        idx_t = offsets.unsqueeze(1) + torch.arange(self._n_layers, device=x.device).unsqueeze(0)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, self._n_layers)
        windows = x[batch_idx, idx_t]                                  # (B, L, d_in)
        z = self.encode(windows).squeeze(1)                            # (B, d_sae)
        x_hat = self.decode(z.unsqueeze(1))                            # (B, L, d_in)
        mse = (x_hat - windows).pow(2).sum(dim=-1).mean()
        l0 = (z != 0).float().sum(dim=-1).mean()
        return mse, {"mse": mse.detach(), "l0": l0.detach(), "z": z.detach()}

    def decoder_directions(self) -> torch.Tensor:
        """Anchor-layer decoder directions, ``(d_sae, d_in)``.

        Returns the column for the centre layer (the "canonical" layer
        the MLC was trained around). Per-layer columns accessible via
        ``W_dec[:, layer_idx, :]``.
        """
        center_idx = self._n_layers // 2  # (0..L-1) mid-window index
        return self.W_dec.data[:, center_idx, :].clone()
