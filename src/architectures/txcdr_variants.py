"""TXCDR weight-sharing ladder (sub-phase 5.2) + novel causal variant.

Parameter-sharing variants of the vanilla TemporalCrosscoder:

    - TXCDRSharedDec:  W_enc per-pos, W_dec shared across T positions.
    - TXCDRSharedEnc:  W_enc shared, W_dec per-pos.
    - TXCDRTied:       W_enc[t] = W_dec[:, t, :].T  (tied per-position).
    - TXCDRPos:        shared SAE over T-window with sinusoidal positional
                       embedding added to x_t before encoding.
    - TXCDRCausal:     per-position latents; position t's latent depends
                       only on x[0..t]. Different output shape — returns
                       (B, T, d_sae) from encode; decode uses matching-T.

All share the TopK-over-summed-pre-activation gate except TXCDRCausal,
which applies per-position TopK.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────── weight-sharing ladder


class TXCDRSharedDec(nn.Module):
    """W_enc per-pos (T, d_in, d_sae), W_dec shared (d_sae, d_in)."""

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None):
        super().__init__()
        self.d_in, self.d_sae, self.T, self.k = d_in, d_sae, T, k
        self.W_enc = nn.Parameter(torch.randn(T, d_in, d_sae) / d_in**0.5)
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) / d_sae**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            v, i = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, i, F.relu(v))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Broadcast shared decoder across T positions, add per-pos bias.
        x_shared = z @ self.W_dec  # (B, d_in)
        x_hat = x_shared.unsqueeze(1).expand(-1, self.T, -1) + self.b_dec
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return loss, x_hat, z


class TXCDRSharedEnc(nn.Module):
    """W_enc shared (d_in, d_sae), W_dec per-pos (d_sae, T, d_in)."""

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None):
        super().__init__()
        self.d_in, self.d_sae, self.T, self.k = d_in, d_sae, T, k
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) / d_in**0.5)
        self.W_dec = nn.Parameter(torch.randn(d_sae, T, d_in) / d_sae**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Project each position through shared encoder, sum across T.
        pre = torch.einsum("btd,ds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            v, i = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, i, F.relu(v))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return loss, x_hat, z


class TXCDRTied(nn.Module):
    """Tied weights: W_enc[t] = W_dec[:, t, :].T for each t."""

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None):
        super().__init__()
        self.d_in, self.d_sae, self.T, self.k = d_in, d_sae, T, k
        # Only W_dec parameterized; W_enc derived at forward time.
        self.W_dec = nn.Parameter(torch.randn(d_sae, T, d_in) / d_sae**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def _W_enc(self) -> torch.Tensor:
        # (T, d_in, d_sae) built from W_dec (d_sae, T, d_in).
        return self.W_dec.permute(1, 2, 0).contiguous()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        W_enc = self._W_enc()
        pre = torch.einsum("btd,tds->bs", x, W_enc) + self.b_enc
        if self.k is not None:
            v, i = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, i, F.relu(v))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return loss, x_hat, z


def _sinusoidal_posemb(T: int, d: int) -> torch.Tensor:
    """(T, d) sinusoidal positional encoding, as in attention is all you need."""
    pos = torch.arange(T).unsqueeze(1).float()
    div = torch.exp(
        torch.arange(0, d, 2).float() * -(math.log(10000.0) / d)
    )
    pe = torch.zeros(T, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TXCDRPos(nn.Module):
    """Shared encoder with sinusoidal positional embedding + per-pos decoder.

    Same-param-count extension of Shared SAE that injects positional
    information explicitly. Encoder sums positional-augmented inputs;
    decoder retains per-position projection so reconstruction quality
    is not artificially capped by a shared output direction.
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None):
        super().__init__()
        self.d_in, self.d_sae, self.T, self.k = d_in, d_sae, T, k
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) / d_in**0.5)
        self.W_dec = nn.Parameter(torch.randn(d_sae, T, d_in) / d_sae**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))
        self.register_buffer("pos_emb", _sinusoidal_posemb(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_plus = x + self.pos_emb.unsqueeze(0)
        pre = torch.einsum("btd,ds->bs", x_plus, self.W_enc) + self.b_enc
        if self.k is not None:
            v, i = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, i, F.relu(v))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return loss, x_hat, z


# ──────────────────────────────────── novel: time-sparsity via block TopK


class TXCDRBlockSparseTopK(nn.Module):
    """Time-sparse TXCDR: per-position pre-activations, global TopK over (T, d_sae).

    Vanilla TXCDR sums the per-position pre-activations across T before
    its TopK gate, so every active feature contributes at every position
    of the window by construction. That dilutes temporally-local signals
    — the failure mode Aniket flagged on language-ID and code-detection
    tasks. Here we keep per-position pre-activations and apply a single
    joint TopK across (T, d_sae), so the same feature is allowed to
    fire at *only* the position(s) where it has the strongest signal.

    Total active features per window = `k` (matches k_win semantics).
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None):
        super().__init__()
        self.d_in, self.d_sae, self.T, self.k = d_in, d_sae, T, k
        # Per-position encoder (T, d_in, d_sae); shared encoder bias.
        self.W_enc = nn.Parameter(torch.randn(T, d_in, d_sae) / d_in**0.5)
        self.W_dec = nn.Parameter(torch.randn(d_sae, T, d_in) / d_sae**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) -> z: (B, T, d_sae) with joint-TopK over (T, d_sae)."""
        pre = torch.einsum("btd,tds->bts", x, self.W_enc) + self.b_enc
        B, T, d_sae = pre.shape
        if self.k is not None:
            pre_flat = pre.reshape(B, T * d_sae)
            v, i = pre_flat.topk(self.k, dim=-1)
            z_flat = torch.zeros_like(pre_flat)
            z_flat.scatter_(-1, i, F.relu(v))
            z = z_flat.reshape(B, T, d_sae)
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, d_sae) -> x_hat: (B, T, d)."""
        # Per-position decoder: (B, T, d_sae) x (d_sae, T, d) -> diag over T.
        # Using einsum with matching t index.
        x_hat = torch.einsum("bts,std->btd", z, self.W_dec) + self.b_dec
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return loss, x_hat, z


# ──────────────────────────────────── novel: low-rank residual decoder


class TXCDRLowRankDec(nn.Module):
    """TXCDR with W_dec[t] = W_base + U_t V_t^T (rank-r residual).

    Parameterized decoder from brief.md §"Decoder smoothness". Forces
    per-position decoder columns to be close to a shared base direction,
    with per-position rank-r corrections. Tests the hypothesis that
    vanilla TXCDR's per-position decoder freedom is over-parameterized.

    Encoder is the shared-summed variant (same as vanilla TXCDR): TopK
    over the summed pre-activation. Decoder applies per-position.
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None,
                 rank: int = 8):
        super().__init__()
        self.d_in, self.d_sae, self.T, self.k = d_in, d_sae, T, k
        self.rank = rank
        self.W_enc = nn.Parameter(torch.randn(T, d_in, d_sae) / d_in**0.5)
        # Shared base decoder + per-position low-rank residual.
        self.W_dec_base = nn.Parameter(torch.randn(d_sae, d_in) / d_sae**0.5)
        self.U = nn.Parameter(torch.randn(T, d_sae, rank) / d_sae**0.5)
        self.V = nn.Parameter(torch.randn(T, d_in, rank) / d_in**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    def _W_dec_at(self, t: int) -> torch.Tensor:
        """Compute full (d_sae, d_in) decoder at position t."""
        return self.W_dec_base + self.U[t] @ self.V[t].T

    @torch.no_grad()
    def _normalize_decoder(self):
        # Normalize only W_dec_base; U, V get pulled along proportionally.
        norms = self.W_dec_base.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec_base.data = self.W_dec_base.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            v, i = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, i, F.relu(v))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # x_hat[t] = z @ W_dec[t]  + b_dec[t]
        # Efficient batched: stack (T, d_sae, d_in) from base + low-rank.
        # U: (T, d_sae, r), V: (T, d_in, r) -> (T, d_sae, d_in) correction.
        W_dec_full = self.W_dec_base.unsqueeze(0) + torch.einsum(
            "tsr,tdr->tsd", self.U, self.V,
        )  # (T, d_sae, d_in)
        x_hat = torch.einsum("bs,tsd->btd", z, W_dec_full) + self.b_dec
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return loss, x_hat, z


# ──────────────────────────────────── novel: per-feature rank-K decoder


class TXCDRRankKFeature(nn.Module):
    """TXCDR with W_dec[j] = A_j @ B_j enforced rank-K per feature.

    Motivated by the SVD analysis (analyze_decoder_svd.py): TXCDR-T20's
    per-feature decoder spectrum is flatter than T=5's, indicating the
    model uses more of its available rank than necessary. A hard rank-K
    constraint per feature is cheaper than the low-rank-residual
    parameterization and expresses the "each feature has a simple
    time-profile × direction factorization" hypothesis cleanly.

    For each latent j, the decoder matrix W_dec[j] ∈ R^{T × d_in} is
    parameterized as A_j B_j with A_j ∈ R^{T × K}, B_j ∈ R^{K × d_in}.
    Params: d_sae · K · (T + d_in)  vs  d_sae · T · d_in for vanilla.
    K=4 is ~45% of vanilla TXCDR-T5 params.

    Encoder is per-position (same as vanilla), with TopK over summed
    pre-activation.
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None,
                 K_rank: int = 4):
        super().__init__()
        self.d_in, self.d_sae, self.T, self.k = d_in, d_sae, T, k
        self.K_rank = K_rank
        self.W_enc = nn.Parameter(torch.randn(T, d_in, d_sae) / d_in**0.5)
        # Per-feature rank-K factorization of the (T, d_in) decoder slab.
        self.A = nn.Parameter(torch.randn(d_sae, T, K_rank) / K_rank**0.5)
        self.B = nn.Parameter(torch.randn(d_sae, K_rank, d_in) / d_in**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    def _full_W_dec(self) -> torch.Tensor:
        # Reconstruct (d_sae, T, d_in) tensor on the fly.
        return torch.einsum("jtk,jkd->jtd", self.A, self.B)

    @torch.no_grad()
    def _normalize_decoder(self):
        # Normalize full reconstructed decoder, then rescale A/B equally.
        W = self._full_W_dec()
        norms = W.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        scale = 1.0 / norms.sqrt()  # split between A and B
        self.A.data = self.A.data * scale
        self.B.data = self.B.data * scale

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            v, i = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, i, F.relu(v))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # (B, d_sae) × (d_sae, T, K) × (d_sae, K, d_in)
        # Efficient: (z · A) then × B, per feature. Einsum handles it.
        zA = torch.einsum("bs,stk->btk", z, self.A)  # (B, T, K)
        # zA needs per-feature B — but we already collapsed s via z.
        # The combined op: x_hat[b, t, d] = sum_{s,k} z[b,s] A[s,t,k] B[s,k,d]
        # which factors as (z[b,s] * A[s,t,k]) B[s,k,d] summed over s.
        # Need per-s B, so do it in one shot:
        x_hat = torch.einsum("bs,stk,skd->btd", z, self.A, self.B) + self.b_dec
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return loss, x_hat, z


# ──────────────────────────────────── novel: causal TXCDR


class TXCDRCausal(nn.Module):
    """Causal TXCDR: per-position latents; z_t depends only on x[0..t].

    Encoder applies per-position W_enc[t] but with causal restriction:
    z[t] = TopK(sum_{t'<=t} x[t'] @ W_enc_step[t-t']) where W_enc_step
    is a learned length-T kernel. Equivalent to a causal 1D convolution
    in the position axis.

    Returns (B, T, d_sae) from encode; decoder applies per-position.
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None):
        super().__init__()
        self.d_in, self.d_sae, self.T, self.k = d_in, d_sae, T, k
        # Causal kernel: k-step contribution from position t-k to position t.
        self.W_enc_kernel = nn.Parameter(torch.randn(T, d_in, d_sae) / d_in**0.5)
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) / d_sae**0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) -> z: (B, T, d_sae) with per-position TopK."""
        B, T, d = x.shape
        # z[b, t, s] = sum_{tp=0..t} x[b, tp, :] @ W_enc_kernel[t-tp]
        # Implement with causal conv: pad the position axis, einsum per offset.
        z_parts = []
        for offset in range(T):
            # x shifted right by `offset` positions at the front.
            x_shift = F.pad(x[:, :T - offset, :], (0, 0, offset, 0))
            # Contribute W_enc_kernel[offset]
            z_parts.append(
                torch.einsum("btd,ds->bts", x_shift, self.W_enc_kernel[offset])
            )
        pre = torch.stack(z_parts, dim=0).sum(dim=0) + self.b_enc  # (B, T, s)
        if self.k is not None:
            # Per-position TopK — k per position.
            v, i = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(-1, i, F.relu(v))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, d_sae) -> x_hat: (B, T, d)."""
        return z @ self.W_dec + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return loss, x_hat, z
