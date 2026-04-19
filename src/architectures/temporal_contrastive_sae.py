"""Temporal Contrastive SAE — Ye et al. 2025 (papers/temporal_sae.md §3.2).

Single-token SAE with Matryoshka-style high-low partition + an InfoNCE
contrastive term over adjacent-token latents.

Architecture is a standard SAE (W_enc, W_dec, biases, TopK gate). Key
additions:
    - The first `h` latent indices are "high-level" features; the
      remaining `d_sae − h` are "low-level" features.
    - Training loss = L_matr + alpha · L_contr where:
        L_matr = ||x − W_dec[:, 0:h] f_H(x) + b_dec||² +
                 ||x − W_dec f(x) + b_dec||²
        L_contr = symmetric InfoNCE over high-level latents of
                  adjacent-token pairs (x_t, x_{t−1}).

This architecture is token-level at inference; the paper motivates the
temporal structure via the training loss, not via the encoder shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalContrastiveSAE(nn.Module):
    """Single-token SAE with Matryoshka H/L partition + contrastive term.

    Encode returns (B, d_sae). Use `.encode_high(x)` to get just the
    high-level slice (B, h); `.encode_low(x)` for the low-level slice.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int | None = None,
        h: int | None = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        # Default partition: half high-level, half low-level
        self.h = h if h is not None else d_sae // 2

        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) / d_in**0.5)
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) / d_sae**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def _topk(self, pre: torch.Tensor) -> torch.Tensor:
        if self.k is None:
            return F.relu(pre)
        v, i = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, i, F.relu(v))
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d) -> z: (B, d_sae)."""
        pre = x @ self.W_enc + self.b_enc
        return self._topk(pre)

    def encode_high(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)[:, : self.h]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def decode_high_only(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from the first h latents only."""
        z_h = z[:, : self.h]
        W_h = self.W_dec[: self.h, :]
        return z_h @ W_h + self.b_dec

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        """Dispatch on input shape.

        - `(B, d)`: matryoshka loss only (single-token fallback).
        - `(B, 2, d)`: paired adjacent tokens — full matryoshka + InfoNCE
          contrastive loss. Returns loss w.r.t. *current* token (index 1).

        In all cases returns `(loss, x_hat_cur, z_cur)` matching the
        project's training-loop convention.
        """
        if x.ndim == 2:
            z = self.encode(x)
            x_hat_full = self.decode(z)
            x_hat_high = self.decode_high_only(z)
            per_tok = x.shape[-1]
            loss_full = (x_hat_full - x).pow(2).sum(dim=-1).mean() / per_tok
            loss_high = (x_hat_high - x).pow(2).sum(dim=-1).mean() / per_tok
            return loss_full + loss_high, x_hat_full, z
        if x.ndim == 3 and x.shape[1] == 2:
            x_prev = x[:, 0]
            x_cur = x[:, 1]
            z_prev = self.encode(x_prev)
            z_cur = self.encode(x_cur)
            per_tok = x.shape[-1]
            l_full_prev = (self.decode(z_prev) - x_prev).pow(2).sum(-1).mean() / per_tok
            l_high_prev = (self.decode_high_only(z_prev) - x_prev).pow(2).sum(-1).mean() / per_tok
            l_full_cur = (self.decode(z_cur) - x_cur).pow(2).sum(-1).mean() / per_tok
            l_high_cur = (self.decode_high_only(z_cur) - x_cur).pow(2).sum(-1).mean() / per_tok
            l_matr = l_full_prev + l_high_prev + l_full_cur + l_high_cur
            z_h_prev = z_prev[:, : self.h]
            z_h_cur = z_cur[:, : self.h]
            l_contr = _info_nce(z_h_cur, z_h_prev)
            total = l_matr + alpha * l_contr
            return total, self.decode(z_cur), z_cur
        raise ValueError(f"Unexpected input shape {tuple(x.shape)}")


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE on L2-normalized z_a and z_b (both (B, h)).

    Matches the loss in Ye et al. 2025 Eq. (5).
    """
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    # B × B similarity matrix; diagonal = positive pair, off-diag = negatives
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


def compute_contrastive_matryoshka_loss(
    model: TemporalContrastiveSAE,
    x_prev: torch.Tensor,
    x_cur: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Full training loss = L_matr(x_prev) + L_matr(x_cur) + α · L_contr.

    Args:
        x_prev: (B, d) activations at position t-1.
        x_cur:  (B, d) activations at position t.
        alpha:  weight of the contrastive term.
    """
    z_prev = model.encode(x_prev)
    z_cur = model.encode(x_cur)
    per_tok = x_prev.shape[-1]

    # Full + high-only reconstructions for each side
    loss_full_prev = (model.decode(z_prev) - x_prev).pow(2).sum(-1).mean() / per_tok
    loss_high_prev = (model.decode_high_only(z_prev) - x_prev).pow(2).sum(-1).mean() / per_tok
    loss_full_cur = (model.decode(z_cur) - x_cur).pow(2).sum(-1).mean() / per_tok
    loss_high_cur = (model.decode_high_only(z_cur) - x_cur).pow(2).sum(-1).mean() / per_tok

    l_matr = loss_full_prev + loss_high_prev + loss_full_cur + loss_high_cur

    # Contrastive on high-level latents only
    z_h_prev = z_prev[:, : model.h]
    z_h_cur = z_cur[:, : model.h]
    l_contr = _info_nce(z_h_cur, z_h_prev)

    total = l_matr + alpha * l_contr
    return total, {
        "l_matr": float(l_matr.detach()),
        "l_contr": float(l_contr.detach()),
        "l_full_cur": float(loss_full_cur.detach()),
    }
