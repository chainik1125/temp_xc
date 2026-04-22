"""MLC + multi-scale InfoNCE across d_sae prefix lengths.

Agentic cycle 08 candidate (`agentic_mlc_08`). Ports cycle 02's
winning multi-scale contrastive pattern from TXC to MLC family.

Current `MLCContrastive` applies InfoNCE on a single fixed prefix
(`d_sae // 2` latents). Cycle 02 on TXC showed that applying InfoNCE
at multiple scales with γ^s decay gives +0.01 over single-scale
contrastive. We port the idea: apply InfoNCE at three prefix lengths
(d_sae/4, d_sae/2, full d_sae) with γ=0.5 decay.

Note: MLC scales are 1-D (just varying prefix length), not matryoshka
reconstruction scales like in TXC. So this is an analogical port, not
a structural one. Tests whether the multi-scale contrastive pattern
has family-agnostic utility or was tied to matryoshka's nesting.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.mlc_contrastive import MLCContrastive, _info_nce


class MLCContrastiveMultiscale(MLCContrastive):
    """MLC + multi-scale InfoNCE at prefix lengths [d_sae/4, d_sae/2, d_sae].

    Args: same as MLCContrastive, plus:
        prefix_lens: tuple of prefix lengths to apply InfoNCE at.
            Defaults to (d_sae//4, d_sae//2, d_sae).
        gamma: per-scale decay.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        n_layers: int,
        k: int | None,
        prefix_lens: tuple[int, ...] | None = None,
        gamma: float = 0.5,
        h: int | None = None,
    ):
        super().__init__(d_in, d_sae, n_layers, k, h=h)
        if prefix_lens is None:
            prefix_lens = (d_sae // 4, d_sae // 2, d_sae)
        self.prefix_lens = tuple(prefix_lens)
        self.gamma = gamma

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        if x.ndim == 3:
            # single-token fallback — reuse parent
            return super().forward(x, alpha=alpha)

        if x.ndim == 4 and x.shape[1] == 2:
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

            # Multi-scale InfoNCE at each prefix length with γ^s decay
            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            for s, p_len in enumerate(self.prefix_lens):
                z_p_prev = z_prev[:, :p_len]
                z_p_cur = z_cur[:, :p_len]
                l_contr = l_contr + (self.gamma ** s) * _info_nce(z_p_cur, z_p_prev)

            total = l_matr + alpha * l_contr
            return total, self.decode(z_cur), z_cur

        raise ValueError(
            f"Expected (B, L, d) or (B, 2, L, d), got {tuple(x.shape)}"
        )
