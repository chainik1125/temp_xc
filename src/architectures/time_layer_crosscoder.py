"""Time×Layer joint crosscoder (brief.md §3: novel architecture menu).

Takes input `(B, T, L, d_in)` (T time positions × L residual-stream
layers). Per-(t, l) encoder produces pre-activations; a single global
TopK over the flattened `(T, L, d_sae)` pre-activation picks the
k most-salient (position, layer, feature) triples. Decoder applies
per-(t, l) to reconstruct the full windowed-multilayer input.

This is brief.md's variant (a): global TopK. Variant (b) — product-
structured TopK per-layer-per-position — is not implemented here.

Parameter budget at T=5, L=5, d_in=2304, d_sae=8192:
    encoder: T·L·d_in·d_sae ≈ 472 M
    decoder: T·L·d_sae·d_in ≈ 472 M
    total   ≈ 944 M fp32 / 1.9 GB fp16 (the A40 fits this next to
    the 18 GB L11–L15 training buffer with room for Adam state).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeLayerCrosscoder(nn.Module):
    """Joint Time×Layer crosscoder with global TopK over (T, L, d_sae).

    Args:
        d_in: per-(t, l) activation dimension (i.e. gemma's d_model, 2304).
        d_sae: per-(t, l) latent width.
        T: number of time positions in the window.
        L: number of residual-stream layers.
        k: total TopK budget across the whole (T*L*d_sae) latent grid.
    """

    def __init__(self, d_in: int, d_sae: int, T: int, L: int, k: int | None):
        super().__init__()
        self.d_in, self.d_sae, self.T, self.L, self.k = d_in, d_sae, T, L, k

        self.W_enc = nn.Parameter(
            torch.randn(T, L, d_in, d_sae) / d_in**0.5
        )
        self.W_dec = nn.Parameter(
            torch.randn(T, L, d_sae, d_in) / d_sae**0.5
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, L, d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, L, d_in) -> z: (B, T, L, d_sae) with joint TopK."""
        # Per-(t, l) projection: pre[b, t, l, s] = x[b, t, l, :] @ W_enc[t, l, :, s].
        pre = torch.einsum("btld,tlds->btls", x, self.W_enc) + self.b_enc
        B, T, L, d_sae = pre.shape
        if self.k is not None:
            pre_flat = pre.reshape(B, T * L * d_sae)
            v, i = pre_flat.topk(self.k, dim=-1)
            z_flat = torch.zeros_like(pre_flat)
            z_flat.scatter_(-1, i, F.relu(v))
            z = z_flat.reshape(B, T, L, d_sae)
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, L, d_sae) -> x_hat: (B, T, L, d_in)."""
        x_hat = torch.einsum("btls,tlsd->btld", z, self.W_dec) + self.b_dec
        return x_hat

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return loss, x_hat, z
