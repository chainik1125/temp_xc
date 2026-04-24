"""FeatureNestedMatryoshkaTXCDR — per-position matryoshka SAE.

Proposed redesign of matryoshka TXCDR that nests **only latent capacity**,
not window size. Unlike PositionMatryoshkaTXCDR (which entangles latent
capacity AND sub-window size), each scale here reconstructs the **full
T-token window** using a different PREFIX of the latent vector.

Design (per scale s, using first prefix_s latents):
    decoder_s : (prefix_s, T, d_in)
    x_hat_s = einsum("bs,std->btd", z[:, :prefix_s], W_dec_s)
    loss_s = MSE(x_hat_s, x)       # compare against FULL x, not centered sub-window

This matches original MatryoshkaSAE semantics:
    early prefix = dominant reconstruction directions (low-capacity SAE)
    later prefix = refinement

Optional multi-scale InfoNCE (α>0): on pair-windows (B, 2, T, d_in), adds
InfoNCE at the first n_contr_scales prefix lengths with γ geometric
decay (matching agentic_txc_02 cycle 02 recipe).

API:
- encode(x): (B, T, d_in) -> z (B, d_sae), same as TemporalCrosscoder.
- decode_scale(z, scale_idx): (B, d_sae) -> (B, T, d_in) using prefix.
- forward(x [, alpha]): returns (loss, x_hat_full, z).

Plugs into the standard _encode_txcdr probe routing unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


class FeatureNestedMatryoshkaTXCDR(nn.Module):
    def __init__(
        self, d_in: int, d_sae: int, T: int, k: int | None,
        n_scales: int | None = None,
        latent_splits: tuple[int, ...] | None = None,
        alpha: float = 0.0,
        n_contr_scales: int = 3,
        gamma: float = 0.5,
    ):
        """
        Args:
            d_in: residual-stream width.
            d_sae: SAE dict dim.
            T: window size.
            k: TopK budget (per window).
            n_scales: number of matryoshka scales. Default T (one per
                position for comparable setup to PositionMatryoshkaTXCDR).
            latent_splits: custom prefix sizes (tuple summing to d_sae).
                Default uniform split.
            alpha: optional multi-scale InfoNCE weight (for pair input).
            n_contr_scales: number of first scales to use for InfoNCE.
            gamma: geometric decay across contrastive scales.
        """
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        if n_scales is None:
            n_scales = int(T)
        self.n_scales = int(n_scales)
        self.alpha = float(alpha)
        self.n_contr_scales = int(min(n_contr_scales, self.n_scales))
        self.gamma = float(gamma)

        if latent_splits is None:
            base = d_sae // self.n_scales
            remainder = d_sae - base * self.n_scales
            splits = [base + (1 if i < remainder else 0) for i in range(self.n_scales)]
        else:
            splits = list(latent_splits)
            assert len(splits) == self.n_scales and sum(splits) == d_sae
        self.latent_splits = tuple(splits)
        self.prefix_sum = tuple(sum(splits[:i + 1]) for i in range(self.n_scales))

        # Shared encoder (same as vanilla TXCDR).
        self.W_enc = nn.Parameter(
            torch.randn(T, d_in, d_sae) * (1.0 / d_in**0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Per-scale decoder, each full-T width.
        # Shape: (prefix_s, T, d_in). Note `T` instead of (s+1).
        self.W_decs = nn.ParameterList()
        self.b_decs = nn.ParameterList()
        for i in range(self.n_scales):
            prefix = self.prefix_sum[i]
            W = torch.randn(prefix, T, d_in) * (1.0 / prefix**0.5)
            self.W_decs.append(nn.Parameter(W))
            self.b_decs.append(nn.Parameter(torch.zeros(T, d_in)))

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        for W in self.W_decs:
            norms = W.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            W.data = W.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
        else:
            z = F.relu(pre)
        return z

    def decode_scale(self, z: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Return (B, T, d_in) reconstruction from first prefix_s latents."""
        prefix = self.prefix_sum[scale_idx]
        W = self.W_decs[scale_idx]       # (prefix, T, d_in)
        b = self.b_decs[scale_idx]       # (T, d_in)
        z_prefix = z[:, :prefix]
        return torch.einsum("bs,std->btd", z_prefix, W) + b

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Default 'full' reconstruction uses the largest scale."""
        return self.decode_scale(z, self.n_scales - 1)

    def _matryoshka_loss(self, x: torch.Tensor, z: torch.Tensor):
        """Sum of per-scale MSEs, each against FULL x (not a sub-window).

        Averaged over scales. Returns (loss, x_hat_full).
        """
        total = 0.0
        last_xhat = None
        for i in range(self.n_scales):
            x_hat_i = self.decode_scale(z, i)  # (B, T, d_in)
            l_i = (x_hat_i - x).pow(2).sum(dim=-1).mean()
            total = total + l_i
            last_xhat = x_hat_i
        total = total / self.n_scales
        return total, last_xhat

    def forward(self, x: torch.Tensor, alpha: float | None = None):
        """(B, T, d) single or (B, 2, T, d) pair.

        Pair path adds multi-scale InfoNCE (α weighted) on first
        n_contr_scales prefix lengths with γ decay.
        """
        eff_alpha = self.alpha if alpha is None else alpha

        if x.ndim == 3:
            z = self.encode(x)
            l_recon, x_hat = self._matryoshka_loss(x, z)
            return l_recon, x_hat, z

        if x.ndim == 4 and x.shape[1] == 2:
            x_prev = x[:, 0]
            x_cur = x[:, 1]

            z_prev = self.encode(x_prev)
            z_cur = self.encode(x_cur)

            l_prev, _ = self._matryoshka_loss(x_prev, z_prev)
            l_cur, x_hat_cur = self._matryoshka_loss(x_cur, z_cur)
            l_recon = l_prev + l_cur

            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            if eff_alpha > 0.0 and self.n_contr_scales > 0:
                for s in range(self.n_contr_scales):
                    prefix = self.prefix_sum[s]
                    w = self.gamma ** s
                    l_contr = l_contr + w * _info_nce(
                        z_cur[:, :prefix], z_prev[:, :prefix]
                    )

            total = l_recon + eff_alpha * l_contr
            return total, x_hat_cur, z_cur

        raise ValueError(
            f"Expected (B, T, d) or (B, 2, T, d); got {tuple(x.shape)}"
        )
