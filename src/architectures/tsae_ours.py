"""TSAEOurs — literal port of Ye et al. 2025 Temporal SAE (arxiv 2511.05541).

This module is the Phase 6 reference port. It implements the paper's
§3.2 equations exactly:

    f(x_t)      = TopK( W_enc x_t + b_enc )
    x_hat_H(x)  = W_dec[:, 0:h] f(x)[0:h] + b_dec      (high-level reconstruction)
    x_hat(x)    = W_dec        f(x)       + b_dec      (full reconstruction)

    L_matr(x_t) = ||x_t - x_hat_H(x_t)||^2
                + ||x_t - x_hat(x_t)||^2

    L_contr     = symmetric InfoNCE on the L2-normalised high-level
                  prefix of adjacent-token latents (z_t[:h], z_{t-1}[:h])
                  — paper Eq. (5).

    L           = sum_i L_matr(x_t^i) + alpha · L_contr

The only deviations from the paper are:

- **Activation**: we use plain TopK (k=100 per token) instead of the
  paper's BatchTopK(k=20) to stay consistent with the Phase 5 bench
  sparsity convention.
- **High/low split**: h = d_sae // 2 (50%) per the Phase 6 brief, vs.
  the paper's 20%. We match the bench's other matryoshka archs
  (agentic_txc_02, agentic_mlc_08) which also split at 50%.

Contrast with `TemporalContrastiveSAE` (Aniket's earlier port): that
module applies L_matr to BOTH x_prev and x_cur (4 recon terms per pair),
while this module follows the paper's equation literally and applies
L_matr only to x_cur (2 recon terms per pair). That is the "slightly
different training loop" the Phase 6 brief flags.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSAEOurs(nn.Module):
    """Paper-faithful Ye et al. T-SAE with TopK activation.

    Inference API mirrors `TemporalContrastiveSAE` and `MLCContrastive`
    so downstream probing / encoding code treats all three uniformly.

    Args:
        d_in: residual-stream width (2304 for Gemma-2-2b).
        d_sae: dictionary size (18432 in Phase 5/6).
        k: per-token TopK sparsity (100 in Phase 5 convention).
        h: high-level latent prefix length. Defaults to `d_sae // 2`.
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
        """x: (B, d_in) -> z: (B, d_sae)."""
        pre = x @ self.W_enc + self.b_enc
        return self._topk(pre)

    def encode_high(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)[:, : self.h]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Full reconstruction: z @ W_dec + b_dec."""
        return z @ self.W_dec + self.b_dec

    def decode_high_only(self, z: torch.Tensor) -> torch.Tensor:
        """High-level-only reconstruction from the first h latents.

        Implements paper §3.2's $\\hat{x}_H = W^{dec}_{0:h} f_{0:h}(x) + b^{dec}$.
        """
        z_h = z[:, : self.h]
        W_h = self.W_dec[: self.h, :]
        return z_h @ W_h + self.b_dec

    def _l_matr(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Paper §3.2 matryoshka loss: L_H + L_L on a single token."""
        per_tok = x.shape[-1]
        l_high = (self.decode_high_only(z) - x).pow(2).sum(-1).mean() / per_tok
        l_full = (self.decode(z) - x).pow(2).sum(-1).mean() / per_tok
        return l_high + l_full

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        """Dispatch on input shape.

        - `(B, d_in)`: matryoshka loss only on the single token
          (used as a single-token fallback and for inference-only calls).
        - `(B, 2, d_in)`: adjacent-pair training. `x[:, 0]` is x_prev,
          `x[:, 1]` is x_cur. Follows paper Eq. (5):
          loss = L_matr(x_cur) + alpha · InfoNCE(z_cur[:h], z_prev[:h]).

        Returns `(loss, x_hat_cur, z_cur)` to match the project's
        training-loop convention (`_iterate_train` in train_primary_archs.py).
        """
        if x.ndim == 2:
            z = self.encode(x)
            loss = self._l_matr(x, z)
            return loss, self.decode(z), z
        if x.ndim == 3 and x.shape[1] == 2:
            x_prev = x[:, 0]
            x_cur = x[:, 1]
            z_prev = self.encode(x_prev)
            z_cur = self.encode(x_cur)
            l_matr = self._l_matr(x_cur, z_cur)
            l_contr = _info_nce(z_cur[:, : self.h], z_prev[:, : self.h])
            total = l_matr + alpha * l_contr
            return total, self.decode(z_cur), z_cur
        raise ValueError(
            f"Expected (B, d_in) or (B, 2, d_in), got {tuple(x.shape)}"
        )


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE over cosine similarity — paper Eq. (5)."""
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))
