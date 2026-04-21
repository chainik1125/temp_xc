"""TXCDR with rotational (Lie-group) decoder parameterization.

Phase 5.7 autoresearch candidate A1, brief.md §3.4. Operationalises
"feature direction rotates across time": the per-position decoder
columns are obtained from a shared base by applying a rotation
exp(t·A) where A is skew-symmetric.

Direct form is `W_dec^(t) = W_base @ exp(t · A)` with A ∈ R^{d_in×d_in}
skew. That A is 5.3M params and exp() costs O(d_in^3). We use a
**rank-K factorisation** instead: A = Q J Q^T with Q ∈ R^{d_in×K}
(orthonormal at init) and J ∈ R^{K×K} skew. Then

    exp(t · A) = I + Q (exp(t · J) − I) Q^T

(matrix exponential restricted to the K-dim rotation subspace; identity
outside it). For K=8 the per-t exp(t · J) is a trivial (8, 8) matrix
exp computed on-CPU/GPU each forward.

Decode rewrites as

    x_hat[b, t, :] = z[b] @ W_dec^(t) = z @ W_base
                                       + (z @ W_base @ Q) @ (exp(t·J) − I) @ Q^T

so the (d_sae, T, d_in) tensor is **never materialised**. Per-t cost is
O(B · K · d_in) — at T=5, B=1024, K=8, d_in=2304 the rotation contribution
is ~95 M FLOPs per slide, trivial vs the (B, d_sae) → (B, d_in) base
matmul.

Encoder is the vanilla TemporalCrosscoder per-position encoder
(unchanged). Inference encode/decode API matches TemporalCrosscoder so
probing reuses the txcdr encode dispatch.

Default K=8 matches the autoresearch plan's "rank ∈ {4, 8, 16}" sweep
range.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TXCDRRotational(nn.Module):
    """TXCDR with W_dec^(t) = W_base @ exp(t · A), A skew rank-K.

    Parameters:
        W_enc: (T, d_in, d_sae) — per-position encoder, identical to
            vanilla TXCDR.
        W_base: (d_sae, d_in) — shared base decoder direction per feature.
        Q: (d_in, K) — orthonormal-at-init rotation subspace basis.
        J_raw: (K, K) — pre-skew parameter; the actual A uses
            J = 0.5 (J_raw − J_raw^T).
        b_enc: (d_sae,)
        b_dec: (T, d_in) — per-position decoder bias (kept like vanilla
            TXCDR, since constant-in-t bias would lose expressivity).
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int | None,
        K_rank: int = 8,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        self.K_rank = K_rank

        self.W_enc = nn.Parameter(
            torch.randn(T, d_in, d_sae) * (1.0 / d_in**0.5)
        )
        self.W_base = nn.Parameter(
            torch.randn(d_sae, d_in) * (1.0 / d_sae**0.5)
        )
        # Orthonormal init for Q via QR of a Gaussian.
        q_init, _ = torch.linalg.qr(torch.randn(d_in, K_rank))
        self.Q = nn.Parameter(q_init)
        # Small skew init so initial rotation is near identity.
        self.J_raw = nn.Parameter(torch.randn(K_rank, K_rank) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        """Normalise W_base rows so the average decoder norm is bounded.

        We normalise on W_base only — the rotation preserves norm (it's a
        rotation around the K-dim subspace), so per-position decoder
        rows stay unit-norm if W_base is unit-norm.
        """
        norms = self.W_base.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_base.data = self.W_base.data / norms

    def _skew(self) -> torch.Tensor:
        """Return the (K, K) skew-symmetric A from J_raw."""
        return 0.5 * (self.J_raw - self.J_raw.t())

    def _rotation_blocks(self) -> torch.Tensor:
        """Return (T, K, K) tensor of (exp(t · J) − I) for t = 0..T-1.

        Subtracting I lets us write x_hat as base + correction (avoids
        a separate t=0 special case).
        """
        J = self._skew()  # (K, K)
        I = torch.eye(self.K_rank, device=J.device, dtype=J.dtype)
        out = []
        for t in range(self.T):
            # exp(t · J) for K=8 is fast and exact via torch.linalg.matrix_exp.
            E = torch.linalg.matrix_exp(t * J)
            out.append(E - I)
        return torch.stack(out, dim=0)  # (T, K, K)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) -> z: (B, d_sae) with k non-zeros (TopK)."""
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            topk_vals, topk_idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, topk_idx, F.relu(topk_vals))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_sae) -> x_hat: (B, T, d_in).

        x_hat[:, t, :] = z @ W_base
                       + (z @ W_base @ Q) @ (exp(t·J) − I) @ Q^T
                       + b_dec[t]
        """
        # base component (constant in t)
        base = z @ self.W_base                # (B, d_in)
        # project z's effective W_base rows into K-dim subspace
        WBQ = self.W_base @ self.Q            # (d_sae, K)
        proj = z @ WBQ                        # (B, K)
        rot = self._rotation_blocks()         # (T, K, K)
        # contrib[:, t, :] = proj @ rot[t] @ Q^T
        contrib = torch.einsum("bk,tkj,dj->btd", proj, rot, self.Q)
        return base.unsqueeze(1) + contrib + self.b_dec  # (B, T, d_in)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, z

    @torch.no_grad()
    def decoder_directions_at(self, pos: int) -> torch.Tensor:
        """(d_in, d_sae) decoder columns for a given position.

        W_dec^(t)[j, :] = W_base[j, :] + (W_base @ Q)[j, :] @ (exp(t·J) − I) @ Q^T
        """
        WBQ = self.W_base @ self.Q  # (d_sae, K)
        rot_t = torch.linalg.matrix_exp(pos * self._skew()) - torch.eye(
            self.K_rank, device=self.W_base.device, dtype=self.W_base.dtype,
        )
        contrib = WBQ @ rot_t @ self.Q.t()  # (d_sae, d_in)
        W_dec_t = self.W_base + contrib    # (d_sae, d_in)
        return W_dec_t.t()                  # (d_in, d_sae)
