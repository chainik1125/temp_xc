"""Matryoshka-TXCDR + InfoNCE + orthogonality penalty on scale-1 decoder.

Agentic cycle 01 candidate (`agentic_txc_01`). Extends the current best
`MatryoshkaTXCDRContrastive` (A3 α=1.0) by adding an orthogonality
regularizer on the scale-1 decoder columns.

Rationale: A3 plateaued at α=1.0 (Δ=+0.0259) with no further gain at
α=3.0 (+0.0238), suggesting the contrastive objective has saturated.
Feature diversity may still be sub-optimal within scale-1 latents —
contrastive alone doesn't force column orthogonality, and matryoshka
nesting only loosely encourages it. Adding an explicit Fro-norm
penalty on (W_dec_scale1 @ W_dec_scale1^T - I) should push features
to be more independent, which may improve class separation in
downstream probing.

Scope: scale-1 only. Scale-1 G matrix is (prefix_1, prefix_1) ~= 3700² ~= 54 MB —
manageable. Higher scales have larger prefixes and would blow memory, so
we restrict to scale-1 (which is also where the contrastive head acts).

Inference API matches parent — probing reuses the matryoshka encode path.
"""

from __future__ import annotations

import torch

from src.architectures.matryoshka_txcdr_contrastive import (
    MatryoshkaTXCDRContrastive,
)


class MatryoshkaTXCDRContrastiveOrth(MatryoshkaTXCDRContrastive):
    """A3 + scale-1 orthogonality penalty.

    Args: same as MatryoshkaTXCDRContrastive, plus:
        lam_orth: weight of the orthogonality penalty term.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int | None,
        lam_orth: float = 0.01,
        latent_splits: tuple[int, ...] | None = None,
        contr_prefix: int | None = None,
    ):
        super().__init__(
            d_in, d_sae, T, k,
            latent_splits=latent_splits,
            contr_prefix=contr_prefix,
        )
        self.lam_orth = lam_orth

    def _scale1_orth_penalty(self) -> torch.Tensor:
        """Sum-of-squared off-diagonal of W_scale1 @ W_scale1^T, normalized
        so the penalty is on the same order of magnitude as the InfoNCE
        term (~log(batch_size)) at feature near-orthogonality.

        Scale rationale: Matryoshka recon losses sit at ~16k per step,
        contrastive at α=1 adds ~7 (log(1024)). We want orth to live on
        the same scale as contrastive so λ~O(1) is meaningful, not a
        million-factor mismatch.
        """
        W = self.W_decs[0]                        # (prefix_1, 1, d_in)
        flat = W.reshape(W.shape[0], -1)          # (prefix_1, d_in)
        G = flat @ flat.t()                       # (prefix_1, prefix_1)
        eye = torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
        off_sq_sum = (G - eye).pow(2).sum()
        return off_sq_sum / G.shape[0]            # per-feature normalization

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        total, x_hat, z = super().forward(x, alpha=alpha)
        l_orth = self._scale1_orth_penalty()
        return total + self.lam_orth * l_orth, x_hat, z
