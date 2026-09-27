"""
fast_models.py — Vectorized StackedSAE and TemporalCrosscoder for NLP scale.

Key speedups over parent models.py:
  1. StackedSAE: batched einsum replaces Python for-loop over T positions
  2. torch.compile-friendly (no Python-level control flow in forward)
  3. Designed for fp16 autocast on Turing GPUs (SM 7.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastStackedSAE(nn.Module):
    """
    StackedSAE with fully vectorized forward — no Python loop over T.

    Uses batched matmuls via einsum:
      encode: pre = einsum("btd,thd->bth", x_c, W_enc) + b_enc
      decode: x_hat = einsum("bth,tdh->btd", u, W_dec) + b_dec

    Mathematically identical to T independent TopKSAEs.
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k

        # (T, d) per-position decoder bias
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))
        # (T, h, d) per-position encoder weights
        self.W_enc = nn.Parameter(torch.empty(T, d_sae, d_in))
        # (T, h) per-position encoder bias
        self.b_enc = nn.Parameter(torch.zeros(T, d_sae))
        # (T, d, h) per-position decoder weights
        self.W_dec = nn.Parameter(torch.empty(T, d_in, d_sae))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        # Normalize each position's decoder columns to unit norm
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)  # (T, 1, h)
        self.W_dec.data.div_(norms)

    def forward(self, x: torch.Tensor):
        """x: (B, T, d) → (recon_loss, x_hat, u)  where u: (B, T, h)"""
        # Encode: center, project, topk
        x_c = x - self.b_dec.unsqueeze(0)                         # (B, T, d)
        pre = torch.einsum("btd,thd->bth", x_c, self.W_enc)      # (B, T, h)
        pre = pre + self.b_enc.unsqueeze(0)                        # (B, T, h)

        # TopK per position: flatten (B,T) → apply topk → reshape
        BT = pre.shape[0] * pre.shape[1]
        pre_flat = pre.reshape(BT, self.d_sae)                    # (B*T, h)
        topk_vals, topk_idx = pre_flat.topk(self.k, dim=-1)       # (B*T, k)
        u_flat = torch.zeros_like(pre_flat)
        u_flat.scatter_(-1, topk_idx, F.relu(topk_vals))
        u = u_flat.reshape(pre.shape)                              # (B, T, h)

        # Decode
        x_hat = torch.einsum("bth,tdh->btd", u, self.W_dec)      # (B, T, d)
        x_hat = x_hat + self.b_dec.unsqueeze(0)                   # (B, T, d)

        # Loss: MSE summed over d, averaged over B*T (matches parent)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, u

    @property
    def decoder_directions(self) -> torch.Tensor:
        """(d, h) decoder columns averaged across T positions."""
        return self.W_dec.mean(dim=0)  # (d, h)


class FastTemporalCrosscoder(nn.Module):
    """
    TemporalCrosscoder — identical to parent models.py (already vectorized).

    Duplicated here so both architectures go through the same compile path.
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k * T  # match stacked SAE's total window-level L0

        self.W_enc = nn.Parameter(
            torch.randn(T, d_in, d_sae) * (1.0 / d_in ** 0.5)
        )
        self.W_dec = nn.Parameter(
            torch.randn(d_sae, T, d_in) * (1.0 / d_sae ** 0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def forward(self, x: torch.Tensor):
        """x: (B, T, d) → (recon_loss, x_hat, z)"""
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, topk_idx, F.relu(topk_vals))

        x_hat = torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, z

    @property
    def decoder_directions(self) -> torch.Tensor:
        return self.W_dec.mean(dim=1).T  # (d, h)
