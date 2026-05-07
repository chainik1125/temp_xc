"""TopK SAE — single-token sparse autoencoder.

Standard per-token TopK SAE baseline. Adapted to the unified
``TempBenchArch`` interface; treats a ``(B, T, d_in)`` window by
flattening to ``(B*T, d_in)`` for encode + reshaping back on decode.

Ported from
``origin/wasteland-canonical @ [scrubbed-sha]:src/architectures/topk_sae.py``
on 2026-05-03 by [pipeline]. Wasteland file separated nn.Module
(``TopKSAE``) from training spec (``TopKSAESpec``); we keep only the
module here and move training to ``temp_bench/training/``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.architectures.base import ArchConfig, TempBenchArch


class TopKSAE(TempBenchArch):
    """Per-token TopK SAE.

    ``forward(x)`` returns ``x_hat`` only (ABC contract). Use
    ``encode(x)`` and ``decode(z)`` separately for training-loss
    computations that need both ``x_hat`` and ``z``.

    Shapes:
        x:     (B, T, d_in) — windowed input; T may be 1
        z:     (B, T, d_sae)
        x_hat: (B, T, d_in)
    """

    def __init__(self, *, d_in: int, d_sae: int, k_pos: int, T: int = 1):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name="topk_sae", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self.k = k_pos

        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, T, d) or (B, d). Flatten T into batch for per-token encoding.
        squeeze_t = x.dim() == 2
        if squeeze_t:
            x = x.unsqueeze(1)
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)
        x_c = x_flat - self.b_dec
        pre = x_c @ self.W_enc.T + self.b_enc
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z_flat = torch.zeros_like(pre)
        z_flat.scatter_(-1, topk_idx, F.relu(topk_vals))
        z = z_flat.reshape(B, T, self._d_sae)
        if squeeze_t:
            z = z.squeeze(1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_t = z.dim() == 2
        if squeeze_t:
            z = z.unsqueeze(1)
        x_hat = z @ self.W_dec.T + self.b_dec
        if squeeze_t:
            x_hat = x_hat.squeeze(1)
        return x_hat

    def post_step(self) -> None:
        """Re-normalise decoder columns to unit L2 after each opt step."""
        self._normalize_decoder()

    def decoder_directions(self) -> torch.Tensor:
        # (d_sae, d_in) — TempBenchArch convention.
        return self.W_dec.data.T.clone()
