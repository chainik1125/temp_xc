"""
models.py — TopK SAE and Temporal Crosscoder architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TemporalCrosscoder(nn.Module):
    """
    Temporal crosscoder with TopK activation.
    Encoder:  W_enc ∈ R^{h × (T*d_in)}
    Decoders: W_dec ∈ R^{T × d_in × h}  — one decoder slice per position
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k

        self.b_dec = nn.Parameter(torch.zeros(T, d_in))
        self.W_enc = nn.Parameter(torch.empty(d_sae, T * d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(T, d_in, d_sae))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) → u: (B, h) with exactly k non-zero entries."""
        x_centered = x - self.b_dec.unsqueeze(0)
        x_flat = x_centered.reshape(x.shape[0], -1)
        pre = x_flat @ self.W_enc.T + self.b_enc
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        u = torch.zeros_like(pre)
        u.scatter_(-1, topk_idx, F.relu(topk_vals))
        return u

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, h) → x_hat: (B, T, d)"""
        return torch.einsum("tdh,bh->btd", self.W_dec, u) + self.b_dec.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        """x: (B, T, d) → (recon_loss, x_hat, u)"""
        u = self.encode(x)
        x_hat = self.decode(u)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, u

    @torch.no_grad()
    def decoder_directions(self, pos: int = 0) -> torch.Tensor:
        """(d, h) decoder columns for position pos."""
        return self.W_dec[pos]
