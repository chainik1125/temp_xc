"""ConvTXCDR — Part B hypothesis H1: translation-invariant encoder.

Drop-in replacement for TemporalCrosscoder whose encoder is a 1-D conv
with shared weights across T positions (kernel_size=3, same-padding).
Intuition: per-position W_enc[T,d_in,d_sae] bakes position-specific
features; with a conv-encoder + T-pooled output, every T-position
sees the same encoder, so longer T supplies more windows over which
each feature must be consistent.

Changes from TemporalCrosscoder:
- Encoder:
    W_enc : (d_sae, d_in, kernel) conv-weights instead of
    (T, d_in, d_sae) per-position projections.
    pre = sum_t relu(Conv1D(x)[t, :, :]) + b_enc  # (B, d_sae)
- Decoder: unchanged — per-position W_dec[d_sae, T, d_in].
- Sparsity: same TopK (or BatchTopK via subclass) on pooled pre-act.

Latent output shape: (B, d_sae). Same API as TemporalCrosscoder,
so it plugs into the existing probe pipeline unchanged
(last_position uses a single (B, T, d) window; mean_pool slides).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTXCDR(nn.Module):
    """Translation-invariant TXCDR encoder via shared 1-D conv.

    Args:
        d_in: residual-stream dim.
        d_sae: SAE latent dim.
        T: context window length.
        k: per-window TopK budget (same semantics as TemporalCrosscoder).
        kernel_size: conv kernel across T positions. Default 3.
        pool: how to pool across T positions into the (B, d_sae) latent.
              'sum' matches TemporalCrosscoder's semantics; 'mean'
              normalizes for T.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int | None,
        kernel_size: int = 3,
        pool: str = "sum",
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        self.kernel_size = kernel_size
        self.pool = pool

        # Shared encoder across T: Conv1d(d_in, d_sae, kernel=kernel_size)
        # with same-padding so output T-dim = input T-dim.
        pad = kernel_size // 2
        self.enc_conv = nn.Conv1d(
            d_in, d_sae, kernel_size=kernel_size, padding=pad, bias=False,
        )
        # initialize to roughly 1/sqrt(d_in · kernel) scale
        with torch.no_grad():
            self.enc_conv.weight.mul_(1.0 / (d_in * kernel_size) ** 0.5)

        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        # Decoder kept per-position to match TemporalCrosscoder's recon.
        self.W_dec = nn.Parameter(
            torch.randn(d_sae, T, d_in) * (1.0 / d_sae**0.5)
        )
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def _preact(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) -> (B, d_sae) pre-activation."""
        # Conv1d expects (B, C_in=d_in, L=T).
        h = self.enc_conv(x.transpose(1, 2))  # (B, d_sae, T)
        if self.pool == "sum":
            pooled = h.sum(dim=-1)
        elif self.pool == "mean":
            pooled = h.mean(dim=-1)
        else:
            raise ValueError(f"unknown pool={self.pool}")
        return pooled + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self._preact(x)
        if self.k is not None:
            topk_vals, topk_idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, topk_idx, F.relu(topk_vals))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, z
