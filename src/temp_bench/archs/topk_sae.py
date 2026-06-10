"""TopK SAE — single-token sparse autoencoder (the canonical baseline).

Adapted for v2 from ``origin/final:src/temp_bench/architectures/topk_sae.py``.

Per-token contract: ``consumes = "token"``. The trainer feeds
``(B, d_in)`` tokens i.i.d. from :class:`ActivationBuffer`. ``train_step``
computes recon + standard L2 loss; no anti-dead AuxK by default
(TopK-SAE is intentionally minimal — anti-dead variants are different
arch entries).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


class TopKSAE(TempBenchArch):
    """Per-token TopK SAE."""

    arch_version: str = "2.0.0"
    consumes: str = "token"

    def __init__(self, *, d_in: int, d_sae: int, k_pos: int, T: int = 1):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name="topk_sae", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T,
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
        """(B, d_in) or (B, T, d_in) → matching z shape."""
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

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """One per-token step. Input: (B, d_in) from ActivationBuffer."""
        if x.dim() != 2:
            raise ValueError(
                f"TopKSAE.train_step expects (B, d_in); got {tuple(x.shape)}."
            )
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        with torch.no_grad():
            l0 = (z != 0).float().sum(dim=-1).mean()
        return {"loss": loss, "mse": loss.detach(), "l0": l0.detach()}

    def post_step(self) -> None:
        self._normalize_decoder()

    def decoder_directions(self) -> torch.Tensor:
        return self.W_dec.data.T.clone()
