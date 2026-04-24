"""LogMatryoshkaTXCDR — Part B hypothesis H3: log-scale matryoshka.

Standard PositionMatryoshkaTXCDR has T scales (scale-t reconstructs a
t-token sub-window), yielding O(T² · d_sae · d_in) decoder params. That
OOMs on A40 at T ≥ 10 when d_sae=18432.

LogMatryoshkaTXCDR uses a fixed log-spaced scale set (default
{1, 2, 4, 8, 16, 32}) that doesn't grow with T. Scales larger than T
are truncated. Makes matryoshka tractable at T ≥ 30.

Design:
- Scales: e.g. S = (1, 2, 4, 8, 16, 32), truncated to T: max_s = max {s ∈ S : s ≤ T}
- Latent splits partition d_sae into len(S) chunks
- Scale-s decoder: (prefix_sum[s], s, d_in) — decodes from first prefix_sum[s] latents
- Encode: same as TXCDR (single W_enc : (T, d_in, d_sae))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LOG_SCALES = (1, 2, 4, 8, 16, 32)


class LogMatryoshkaTXCDR(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int | None,
        scales: tuple[int, ...] = DEFAULT_LOG_SCALES,
        latent_splits: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k

        # Truncate scales to those <= T
        active_scales = tuple(s for s in scales if s <= T)
        if len(active_scales) == 0:
            raise ValueError(
                f"no scales in {scales} fit within T={T}; min scale must be <= T"
            )
        self.scales = active_scales
        n_scales = len(active_scales)

        if latent_splits is None:
            base = d_sae // n_scales
            remainder = d_sae - base * n_scales
            splits = [base + (1 if i < remainder else 0) for i in range(n_scales)]
        else:
            if len(latent_splits) != n_scales:
                raise ValueError(
                    f"latent_splits len {len(latent_splits)} != n_active_scales {n_scales}"
                )
            splits = list(latent_splits)
            assert sum(splits) == d_sae, (
                f"latent_splits must sum to d_sae={d_sae}, got {sum(splits)}"
            )
        self.latent_splits = tuple(splits)
        self.prefix_sum = tuple(sum(splits[:i + 1]) for i in range(n_scales))

        # Encoder: shared across T like TemporalCrosscoder.
        self.W_enc = nn.Parameter(
            torch.randn(T, d_in, d_sae) * (1.0 / d_in**0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # One decoder per active scale.
        self.W_decs = nn.ParameterList()
        self.b_decs = nn.ParameterList()
        for i, s in enumerate(active_scales):
            prefix = self.prefix_sum[i]
            W = torch.randn(prefix, s, d_in) * (1.0 / prefix**0.5)
            self.W_decs.append(nn.Parameter(W))
            self.b_decs.append(nn.Parameter(torch.zeros(s, d_in)))

    @torch.no_grad()
    def _normalize_decoder(self):
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

    def _window_center(self, x: torch.Tensor, t_size: int) -> torch.Tensor:
        T = x.shape[1]
        start = (T - t_size) // 2
        return x[:, start:start + t_size, :]

    def decode_scale(self, z: torch.Tensor, scale_idx: int) -> torch.Tensor:
        prefix = self.prefix_sum[scale_idx]
        W = self.W_decs[scale_idx]
        b = self.b_decs[scale_idx]
        z_prefix = z[:, :prefix]
        return torch.einsum("bs,std->btd", z_prefix, W) + b

    def forward(self, x: torch.Tensor):
        """Recon loss = mean over scales of MSE vs central sub-window."""
        z = self.encode(x)
        total_loss = 0.0
        last_xhat = None
        for i, s in enumerate(self.scales):
            xhat_s = self.decode_scale(z, i)
            x_center_s = self._window_center(x, s)
            loss_s = (xhat_s - x_center_s).pow(2).sum(dim=-1).mean()
            total_loss = total_loss + loss_s
            last_xhat = xhat_s
        total_loss = total_loss / len(self.scales)
        return total_loss, last_xhat, z
