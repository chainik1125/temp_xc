"""Windowed T-SAE: lift Bhalla 2025's per-token T-SAE recipe to T-token
windowed encoding.

At T=1 this reduces to a standard T-SAE (per-token TopK + adjacency contrastive).
At T>=2 the encoder optionally mixes across positions via a learned (T, T) matrix
M, and the decoder is per-position (TXC-style).

Architecture
------------
    W_enc  : (d_in, d_sae)     — shared per-position encoder
    M      : (T, T)            — optional cross-position mixing  (identity if mix_positions=False)
    W_dec  : (T, d_sae, d_in)  — per-position decoder (TXC convention)
    b_enc  : (d_sae,)          — shared encoder bias
    b_dec  : (T, d_in)         — per-position decoder bias

Forward (B, T, d_in) -> (B, T, d_sae):
    pre_per[b, t', :] = (window[b, t', :] - b_dec[t', :]) @ W_enc + b_enc
    pre[b, t, :]     = sum_{t'} M[t', t] * pre_per[b, t', :]

Sparsity (BatchTopK across the (B*T, d_sae) flat matrix):
    flat_pre = pre.reshape(-1)          # length B*T*d_sae
    budget   = B * T * k
    keep top `budget` entries; zero else; reshape back
    -> avg L0 per token = k

Decode (per-position):
    x_hat[b, t, :] = z[b, t, :] @ W_dec[t] + b_dec[t]

Loss:
    recon  = mse(x_hat, window)                                  (per-token MSE)
    contrast = mean over adjacent (t, t+1) pairs of
               (1 - cos(z[:, t, :high], z[:, t+1, :high]))
        where :high is the first n_temporal_features features
        (Bhalla 2025 20%-80% matryoshka split; defaults to all of d_sae if None)
    total = recon + contrastive_alpha * contrast
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowedTSAE(nn.Module):
    """T-SAE recipe with T-token windowed encoder and per-position decoder."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int,
        *,
        contrastive_alpha: float = 0.1,
        n_temporal_features: Optional[int] = None,  # None => all features in contrastive
        mix_positions: bool = False,
        # antidead bookkeeping (mirrors TSAE/Han convention)
        aux_k: int = 512,
        dead_threshold_tokens: int = 640_000,
        auxk_alpha: float = 1.0 / 32.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        self.contrastive_alpha = float(contrastive_alpha)
        self.n_temporal_features = (
            int(n_temporal_features) if n_temporal_features is not None else d_sae
        )
        self.mix_positions = bool(mix_positions)
        self.aux_k = int(aux_k)
        self.dead_threshold_tokens = int(dead_threshold_tokens)
        self.auxk_alpha = float(auxk_alpha)

        # Encoder: shared (d_in, d_sae)
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Cross-position mixing (T, T). Initialized to identity.
        if mix_positions:
            self.M = nn.Parameter(torch.eye(T))
        else:
            self.register_buffer("M_buf", torch.eye(T), persistent=False)

        # Decoder: per-position (T, d_sae, d_in)
        self.W_dec = nn.Parameter(torch.empty(T, d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        self._init_weights()

        # Antidead tracking
        self.register_buffer(
            "last_active_token_count",
            torch.zeros(d_sae, dtype=torch.long),
        )

    @property
    def mixing_matrix(self) -> torch.Tensor:
        return self.M if self.mix_positions else self.M_buf

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            # Each W_dec[t] starts as W_enc.T (TXC-style coupled init)
            for t in range(self.T):
                self.W_dec[t].copy_(self.W_enc.T)
            self._normalize_decoder_()

    @torch.no_grad()
    def _normalize_decoder_(self):
        # Row-wise unit norm per (t, feature) — i.e. each W_dec[t, j, :] has L2 norm 1
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    # ---- per-token inference API (compatible with finder/Wang/frontier_sweep) ----
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token encode: x (..., d_in) -> z (..., d_sae) post per-token TopK.

        Inference path. BatchTopK is only used during training_loss."""
        pre = (x - self.b_dec.mean(dim=0)) @ self.W_enc + self.b_enc
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Per-token decode using the LAST-position decoder W_dec[-1] —
        matches our finder/steering convention of using W_dec[-1] for TXC."""
        return z @ self.W_dec[-1] + self.b_dec[-1]

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    # ---- training-time forward over (B, T, d_in) windows ----------------
    def _encode_window_pre(self, windows: torch.Tensor) -> torch.Tensor:
        """windows: (B, T, d_in) -> pre-activations (B, T, d_sae) with optional mixing."""
        B, T, d = windows.shape
        assert T == self.T, f"window length {T} != self.T={self.T}"
        # Per-position per-token encoding: subtract position-specific b_dec, then shared W_enc + b_enc
        centered = windows - self.b_dec[None, :, :]                 # (B, T, d_in)
        pre_per = torch.einsum("btd,de->bte", centered, self.W_enc) + self.b_enc  # (B, T, d_sae)
        # Mix across positions: pre[b, t, :] = sum_{t'} M[t', t] * pre_per[b, t', :]
        pre = torch.einsum("bsd,st->btd", pre_per, self.mixing_matrix)
        return pre

    def _batch_topk(self, pre_BTD: torch.Tensor) -> torch.Tensor:
        """BatchTopK across (B*T, d_sae) with budget B*T*k."""
        B, T, D = pre_BTD.shape
        budget = B * T * self.k
        flat = pre_BTD.reshape(-1)
        if budget >= flat.numel():
            return pre_BTD
        topk_vals, topk_idx = flat.topk(budget, sorted=False)
        z_flat = torch.zeros_like(flat)
        z_flat.scatter_(0, topk_idx, topk_vals)
        return z_flat.reshape(B, T, D)

    def training_loss(self, windows: torch.Tensor) -> dict:
        """windows: (B, T, d_in). Returns dict with recon_loss, contrast_loss,
        total_loss, z (B, T, d_sae), x_hat (B, T, d_in)."""
        B, T, _ = windows.shape
        pre = self._encode_window_pre(windows)
        z = self._batch_topk(pre)                                                 # (B, T, d_sae)

        # Per-position decode
        x_hat = torch.einsum("btd,tde->bte", z, self.W_dec) + self.b_dec[None, :, :]
        recon_loss = F.mse_loss(x_hat, windows, reduction="mean")

        if T >= 2 and self.contrastive_alpha > 0:
            high = self.n_temporal_features
            za = z[:, :-1, :high].reshape(-1, high)
            zb = z[:, 1:, :high].reshape(-1, high)
            cos_sim = F.cosine_similarity(za.float(), zb.float(), dim=-1)
            contrast_loss = (1.0 - cos_sim).mean()
        else:
            contrast_loss = torch.tensor(0.0, device=windows.device)

        total = recon_loss + self.contrastive_alpha * contrast_loss
        return {
            "recon_loss": recon_loss,
            "contrast_loss": contrast_loss,
            "total_loss": total,
            "z": z,
            "x_hat": x_hat,
        }

    # ---- antidead helpers (mirror TSAE) ----
    @torch.no_grad()
    def update_dead_counter(self, z: torch.Tensor):
        z_flat = z.reshape(-1, self.d_sae)
        n_tokens = z_flat.shape[0]
        active = (z_flat > 0).any(dim=0)
        self.last_active_token_count += n_tokens
        self.last_active_token_count[active] = 0

    def dead_features(self) -> torch.Tensor:
        return self.last_active_token_count > self.dead_threshold_tokens

    def auxk_loss(self, residual_x: torch.Tensor) -> torch.Tensor:
        """Bricken-style aux reconstruction with top-k dead features.
        residual_x: (..., d_in). Computes auxiliary recon using dead features only."""
        dead_mask = self.dead_features()
        if dead_mask.sum() == 0 or self.aux_k == 0:
            return torch.tensor(0.0, device=residual_x.device)
        residual_flat = residual_x.reshape(-1, self.d_in)
        # Use shared encoder + last-position decoder for aux
        pre = (residual_flat - self.b_dec.mean(dim=0)) @ self.W_enc + self.b_enc
        pre_dead = pre.masked_fill(~dead_mask, float("-inf"))
        actual_aux_k = int(min(self.aux_k, dead_mask.sum().item()))
        topk_vals, topk_idx = pre_dead.topk(actual_aux_k, dim=-1)
        z_aux = torch.zeros_like(pre)
        z_aux.scatter_(-1, topk_idx, topk_vals)
        residual_hat = z_aux @ self.W_dec[-1]
        return F.mse_loss(residual_hat, residual_flat, reduction="mean")
