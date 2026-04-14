"""
models.py — Three architectures for the temporal crosscoder sweep.

1. TopKSAE              — single-token SAE (building block for StackedSAE)
2. StackedSAE           — SAE(k) applied independently to each of T positions
3. TemporalCrosscoder   — shared-latent crosscoder across T positions

StackedSAE and TemporalCrosscoder both take (B, T, d) input.
Both expose:
    forward(x) → (recon_loss, x_hat, activations)
    decoder_directions → (d, h)
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


# ─── 2. Stacked SAE — independent SAE(k) per position ───────────────────────────

class StackedSAE(nn.Module):
    """
    stacked_SAE(k, T) = T independent SAE(k), one per position.

    Each position has its own TopKSAE with independent weights.
    Each position gets k active latents, so window-level L0 = k * T.

    Input:  x ∈ R^{B × T × d}
    Output: x_hat ∈ R^{B × T × d}
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        self.saes = nn.ModuleList([TopKSAE(d_in, d_sae, k) for _ in range(T)])

    @torch.no_grad()
    def _normalize_decoder(self):
        for sae in self.saes:
            sae._normalize_decoder()

    def forward(self, x: torch.Tensor):
        """x: (B, T, d) → (recon_loss, x_hat, u)

        u: (B, T, h) — per-position activations.
        """
        losses, x_hats, us = [], [], []
        for t, sae in enumerate(self.saes):
            loss_t, x_hat_t, u_t = sae(x[:, t, :])  # (B, d)
            losses.append(loss_t)
            x_hats.append(x_hat_t)
            us.append(u_t)

        x_hat = torch.stack(x_hats, dim=1)  # (B, T, d)
        u = torch.stack(us, dim=1)           # (B, T, h)
        loss = torch.stack(losses).mean()
        return loss, x_hat, u

    def decoder_directions_at(self, pos: int = 0) -> torch.Tensor:
        """(d, h) decoder columns for a specific position's SAE."""
        return self.saes[pos].decoder_directions

    @property
    def decoder_directions(self) -> torch.Tensor:
        """(d, h) decoder columns averaged across all T SAEs."""
        return torch.stack([sae.decoder_directions for sae in self.saes]).mean(dim=0)


# ─── 3. Temporal Crosscoder (shared latent) ──────────────────────────────────────

class TemporalCrosscoder(nn.Module):
    """
    Standard crosscoder with shared latent vector across all positions.

    Architecture:
        W_enc: (T, d, h) — per-position encoder projections
        W_dec: (h, T, d) — per-position decoder projections
        b_enc: (h,)       — shared encoder bias
        b_dec: (T, d)     — per-position decoder bias

    Encode: z = TopK(einsum("btd,tds->bs", x, W_enc) + b_enc)   → (B, h)
    Decode: x_hat = einsum("bs,std->btd", z, W_dec) + b_dec     → (B, T, d)

    The shared latent uses k*T active latents, matching the stacked SAE's
    total window-level L0 = k*T for a fair comparison.
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k * T  # match stacked SAE's total L0

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
    def decoder_directions_at(self, pos: int = 0) -> torch.Tensor:
        """(d, h) decoder columns for a given position.
        W_dec is (h, T, d) → slice [:, pos, :] gives (h, d) → transpose to (d, h)."""
        return self.W_dec[:, pos, :].T

    @property
    def decoder_directions(self) -> torch.Tensor:
        """(d, h) decoder columns averaged across positions."""
        # Average decoder across T positions: (h, T, d) → mean over T → (h, d) → transpose
        return self.W_dec.mean(dim=1).T
