"""TXCDR with basis-expansion decoder — W_dec^(t) = Σ_k α_k(t) · W_base_k.

Phase 5.7 autoresearch candidate A5, brief.md §3.4. Parameterised-
decoder family: per-position decoder matrices are a time-varying
linear combination of `K << T` shared basis matrices.

    W_dec^(t) = Σ_{k=0..K-1} α[t, k] · W_base[k]

where W_base: (K, d_sae, d_in) is K shared dictionaries and α: (T, K)
are the time-dependent combination coefficients. Setting K=1 reduces
to shared-decoder TXCDR; setting K=T recovers vanilla TXCDR.

Default K=3 for T=5 — forces decoder to be smoother than vanilla
(rank-3 time-variation) while still allowing non-trivial per-position
structure.

Decode rewrites without ever forming the full (d_sae, T, d_in) tensor:

    x_hat[b, t, :] = z @ W_dec[t, :] = Σ_k α[t, k] · (z @ W_base[k])

Per-k we compute zW_base_k = z @ W_base[k] ∈ R^{B × d_in} once, then
combine with α coefficients. Per-t cost is O(B × d_in × K); total
decode cost O(B × d_in × (d_sae + T × K)) ≈ vanilla TXCDR.

Param count at d_in=2304, d_sae=18432, T=5, K=3:
    W_enc: 5 × 2304 × 18432 = 212 M
    W_base: 3 × 18432 × 2304 = 127 M  (vs vanilla W_dec 212 M)
    α: 5 × 3 = 15
    total ≈ 340 M (vs vanilla TXCDR ≈ 425 M)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TXCDRBasisExpansion(nn.Module):
    """TXCDR with time-varying decoder as linear combo of K basis matrices.

    Args:
        d_in, d_sae, T, k: same as vanilla TemporalCrosscoder.
        K_basis: number of basis matrices (must be >= 1). Default 3.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int | None,
        K_basis: int = 3,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        self.K_basis = K_basis

        # Encoder identical to vanilla TXCDR.
        self.W_enc = nn.Parameter(
            torch.randn(T, d_in, d_sae) * (1.0 / d_in**0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Basis decoders (K, d_sae, d_in).
        self.W_base = nn.Parameter(
            torch.randn(K_basis, d_sae, d_in) * (1.0 / d_sae**0.5)
        )
        # Time coefficients (T, K). Init with the K=1 basis dominating at
        # all t (identity-ish) + small per-t variation.
        alpha_init = torch.zeros(T, K_basis)
        alpha_init[:, 0] = 1.0
        if K_basis > 1:
            # Small sinusoidal perturbation on other basis channels so they
            # aren't stuck at zero grad.
            t_range = torch.arange(T, dtype=torch.float32)
            for k_idx in range(1, K_basis):
                alpha_init[:, k_idx] = 0.05 * torch.sin(
                    (k_idx + 1) * torch.pi * t_range / max(T - 1, 1)
                )
        self.alpha = nn.Parameter(alpha_init)
        # Per-position decoder bias (as in vanilla TXCDR).
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        """Normalise each basis matrix's per-feature direction.

        Each basis is normalised independently. The per-position
        effective decoder `W_dec[t, j] = Σ_k α[t,k] · W_base[k, j]`
        won't be exactly unit-norm — that's fine; the effective norm
        is absorbed into the α coefficients.
        """
        norms = self.W_base.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_base.data = self.W_base.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) -> z: (B, d_sae) with k non-zeros."""
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_sae) -> x_hat: (B, T, d_in).

        Precomputes zW_base[:, k, :] = z @ W_base[k] once (shape (B, K, d_in)),
        then combines per-t as sum_k α[t, k] * zW_base[:, k, :].
        """
        # zW_base: (B, K, d_in)
        zW_base = torch.einsum("bs,ksd->bkd", z, self.W_base)
        # combine: x_hat[b, t, :] = sum_k α[t, k] * zW_base[b, k, :]
        x_hat = torch.einsum("tk,bkd->btd", self.alpha, zW_base)
        return x_hat + self.b_dec

    def forward(self, x: torch.Tensor):
        """x: (B, T, d_in) -> (recon_loss, x_hat, z)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, z
