"""
models.py — Three architectures for the temporal crosscoder sweep.

1. TopKSAE              — single-token baseline
2. TemporalCrosscoder   — standard shared-latent crosscoder (ckkissane-style)
3. TemporalCrosscoderPP — per-position latents with full-window encoder context

All three expose the same interface:
    forward(x) → (recon_loss, x_hat, u)
    decoder_directions(pos=0) → (d, h)
    _normalize_decoder()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 1. Standard TopK SAE ───────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    """
    Standard SAE with TopK activation — exactly k latents active per token.
    Input:  x ∈ R^d
    Latent: u ∈ R^h  (k non-zero)
    Output: x_hat ∈ R^d
    """

    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - self.b_dec.unsqueeze(0)
        pre = x_c @ self.W_enc.T + self.b_enc
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        u = torch.zeros_like(pre)
        u.scatter_(-1, topk_idx, F.relu(topk_vals))
        return u

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        return u @ self.W_dec.T + self.b_dec.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        """x: (B, d) → (recon_loss, x_hat, u)"""
        u = self.encode(x)
        x_hat = self.decode(u)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, u

    @property
    def decoder_directions(self) -> torch.Tensor:
        """(d, h) decoder columns."""
        return self.W_dec


# ─── 2. Standard Temporal Crosscoder (shared latent) ────────────────────────────

class TemporalCrosscoder(nn.Module):
    """
    Standard crosscoder with shared latent vector across all positions.
    Matches ckkissane/crosscoder-model-diff-replication.

    Architecture:
        W_enc: (T, d, h) — per-position encoder projections
        W_dec: (h, T, d) — per-position decoder projections
        b_enc: (h,)       — shared encoder bias
        b_dec: (T, d)     — per-position decoder bias

    Encode: z = TopK(einsum("btd,tds->bs", x, W_enc) + b_enc)   → (B, h)
    Decode: x_hat = einsum("bs,std->btd", z, W_dec) + b_dec     → (B, T, d)

    One shared latent of size h with k non-zeros must explain all T positions.
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k

        self.W_enc = nn.Parameter(
            torch.randn(T, d_in, d_sae) * (1.0 / d_in ** 0.5)
        )
        self.W_dec = nn.Parameter(
            torch.randn(d_sae, T, d_in) * (1.0 / d_sae ** 0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        # W_dec: (h, T, d) — normalize each feature across all positions
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) → z: (B, h) with k non-zeros."""
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, topk_idx, F.relu(topk_vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, h) → x_hat: (B, T, d)."""
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        """x: (B, T, d) → (recon_loss, x_hat, z)"""
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, z

    @torch.no_grad()
    def decoder_directions(self, pos: int = 0) -> torch.Tensor:
        """(d, h) decoder columns for a given position.
        W_dec is (h, T, d) → slice [:, pos, :] gives (h, d) → transpose to (d, h)."""
        return self.W_dec[:, pos, :].T