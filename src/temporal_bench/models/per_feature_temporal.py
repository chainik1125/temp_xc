"""Proposal 3: Per-feature temporal layer.

Independent SAE encoder followed by a per-feature temporal kernel that
lets each feature talk to ITSELF across time positions. The most
interpretable temporal correction — no cross-feature mixing.

Architecture:
    1. Encode each position independently:
       a_{t,k} = TopK(ReLU(W_enc @ x_t + b_enc), k)
    2. Per-feature temporal correction:
       a_tilde_{t,k} = a_{t,k} + sum_s K_{k,t,s} * a_{s,k}
    3. Decode:
       x_hat_t = W_dec @ a_tilde_t + b_dec
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutput, TemporalAE


class PerFeatureTemporalAE(TemporalAE):
    """SAE with per-feature temporal kernel (Proposal 3).

    Each feature has its own (T, T) temporal kernel that lets it
    propagate information across time positions. No cross-feature mixing.

    The kernel K is initialized to zero, so the model starts as a
    standard independent SAE and gradually learns temporal corrections.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int,
        *,
        causal: bool = False,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        self.causal = causal

        # SAE parameters (shared across positions)
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Per-feature temporal kernel: (d_sae, T, T)
        # Initialized to zero -> starts as independent SAE
        self.K = nn.Parameter(torch.zeros(d_sae, T, T))

        # Causal mask (lower-triangular, not learnable)
        if causal:
            self.register_buffer(
                "causal_mask", torch.tril(torch.ones(T, T))
            )

        # Initialize SAE weights
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def _get_kernel(self) -> torch.Tensor:
        """Return the effective temporal kernel, applying causal mask if needed."""
        if self.causal:
            return self.K * self.causal_mask.unsqueeze(0)
        return self.K

    def encode_independent(self, x: torch.Tensor) -> torch.Tensor:
        """Per-position SAE encoding. x: (B, T, d) -> a: (B, T, m)."""
        B, T, d = x.shape
        x_flat = x.reshape(B * T, d)
        pre = F.relu((x_flat - self.b_dec) @ self.W_enc + self.b_enc)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        a = torch.zeros_like(pre)
        a.scatter_(-1, topk_idx, topk_vals)
        return a.reshape(B, T, self.d_sae)

    def temporal_mix(self, a: torch.Tensor) -> torch.Tensor:
        """Per-feature temporal correction. a: (B, T, m) -> a_tilde: (B, T, m).

        For each feature k:
            a_tilde_{t,k} = a_{t,k} + sum_s K_{k,t,s} * a_{s,k}
        """
        K = self._get_kernel()
        # K: (m, T_out, T_in), a: (B, T_in, m)
        # Want: correction_{b, t, k} = sum_s K_{k, t, s} * a_{b, s, k}
        correction = torch.einsum("kts,bsk->btk", K, a)
        return a + correction

    def decode(self, a: torch.Tensor) -> torch.Tensor:
        """Decode. a: (B, T, m) -> x_hat: (B, T, d)."""
        return a @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> ModelOutput:
        B, T, d = x.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"

        a = self.encode_independent(x)
        a_tilde = self.temporal_mix(a)
        x_hat = self.decode(a_tilde)

        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        metrics = {}
        if self.collect_metrics:
            metrics = {
                "recon_loss": recon_loss.item(),
                "l0": (a_tilde != 0).float().sum(dim=-1).mean().item(),
                "pre_mix_l0": (a != 0).float().sum(dim=-1).mean().item(),
                "kernel_norm": self.K.norm().item(),
            }

        return ModelOutput(
            x_hat=x_hat,
            latents=a_tilde,
            loss=recon_loss,
            metrics=metrics,
        )

    def decoder_directions(self, pos: int | None = None) -> torch.Tensor:
        return self.W_dec.T  # (d, m) — shared across positions

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()
