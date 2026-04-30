"""Temporal Sparse Autoencoder (T-SAE) per Bhalla et al. 2025 (arXiv:2511.05541).

Encoder: per-token TopK SAE (T=1 — same as standard TopKSAE).
Loss   : standard reconstruction MSE + an adjacent-token contrastive term
         that encourages z(token_t) and z(token_{t+1}) to be similar in
         direction. Bhalla et al. argue this disentangles "high-level"
         semantic features (which fire consistently across nearby tokens) from
         "low-level" syntactic features (which fire on single tokens).

Architecture parity with TopKSAE so the resulting model is a drop-in for the
finder + steering pipeline:
  - W_dec shape (d_sae, d_in) — same convention as sae_day.sae.TopKSAE
  - encode() takes (B, d_in) and returns (B, d_sae) post-TopK
  - decode() is linear: z @ W_dec + b_dec

The training forward additionally accepts (B, T, d_in) windows and computes
the contrastive term across consecutive token positions in the window. At
eval / inference time we use the per-token .encode/.decode API only.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSAEAdjacentContrastive(nn.Module):
    """Per-token TopK SAE with adjacent-token contrastive loss."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        *,
        contrastive_alpha: float = 1.0,
        batch_topk: bool = False,
        # antidead bookkeeping (matches the Han / TXC convention so the trainer
        # can hot-swap)
        aux_k: int = 512,
        dead_threshold_tokens: int = 640_000,
        auxk_alpha: float = 1.0 / 32.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.contrastive_alpha = float(contrastive_alpha)
        self.batch_topk = bool(batch_topk)
        self.aux_k = int(aux_k)
        self.dead_threshold_tokens = int(dead_threshold_tokens)
        self.auxk_alpha = float(auxk_alpha)

        # Same shapes as TopKSAE
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self._init_weights()

        # Antidead tracking: per-feature steps-since-last-firing
        self.register_buffer("last_active_token_count",
                             torch.zeros(d_sae, dtype=torch.long))

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder_()

    @torch.no_grad()
    def _normalize_decoder_(self):
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    # ---- per-token API (identical to TopKSAE) -----------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_in) → z: (..., d_sae) post-TopK.

        Always per-token TopK at inference (regardless of self.batch_topk).
        BatchTopK only applies to the training-time path (training_loss)."""
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def _encode_batch_topk(self, flat: torch.Tensor) -> torch.Tensor:
        """flat: (N, d_in) input activations. Keeps the top (N*k) pre-activation
        values across the entire (N, d_sae) post-projection matrix — a single
        global TopK with budget ``N*k`` so average L0 per row is k. Returns
        z of shape (N, d_sae)."""
        N = flat.shape[0]
        budget = N * self.k
        pre_act = (flat - self.b_dec) @ self.W_enc + self.b_enc  # (N, d_sae)
        flat_view = pre_act.reshape(-1)
        if budget >= flat_view.numel():
            return pre_act
        topk_vals, topk_idx = flat_view.topk(budget, sorted=False)
        z_flat = torch.zeros_like(flat_view)
        z_flat.scatter_(0, topk_idx, topk_vals)
        return z_flat.reshape(N, self.d_sae)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-token forward (mirrors TopKSAE.forward signature)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    # ---- training-time forward over a (B, T, d_in) window ------------------
    def training_loss(self, windows: torch.Tensor) -> dict:
        """windows: (B, T, d_in) batch of T consecutive token activations.

        Returns dict with keys:
            recon_loss   — mean MSE over all (B, T) tokens
            contrast_loss — mean negative cosine sim between z_t, z_{t+1}
                             across all adjacent pairs (B, T-1)
            total_loss   — recon_loss + contrastive_alpha * contrast_loss
            z            — (B, T, d_sae) post-TopK firings, for antidead bookkeeping
            x_hat        — (B, T, d_in) reconstruction
        """
        B, T, d = windows.shape
        flat = windows.reshape(B * T, d)
        if self.batch_topk:
            z_flat = self._encode_batch_topk(flat)
        else:
            z_flat = self.encode(flat)
        x_hat_flat = self.decode(z_flat)
        z = z_flat.reshape(B, T, self.d_sae)
        x_hat = x_hat_flat.reshape(B, T, d)

        recon_loss = F.mse_loss(x_hat, windows, reduction="mean")

        if T >= 2 and self.contrastive_alpha > 0:
            z_anchor = z[:, :-1, :].reshape(-1, self.d_sae)
            z_next = z[:, 1:, :].reshape(-1, self.d_sae)
            cos_sim = F.cosine_similarity(z_anchor.float(), z_next.float(), dim=-1)
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

    # ---- antidead helpers (mirror TXCBareAntidead API) ---------------------
    @torch.no_grad()
    def update_dead_counter(self, z: torch.Tensor):
        """Increment dead-counter for features that didn't fire in this batch."""
        z_flat = z.reshape(-1, self.d_sae)
        n_tokens = z_flat.shape[0]
        active = (z_flat > 0).any(dim=0)
        self.last_active_token_count += n_tokens
        self.last_active_token_count[active] = 0

    def dead_features(self) -> torch.Tensor:
        return self.last_active_token_count > self.dead_threshold_tokens

    def auxk_loss(self, residual_x: torch.Tensor) -> torch.Tensor:
        """Auxiliary reconstruction with the top-K dead features (Bricken-style)."""
        dead_mask = self.dead_features()
        if dead_mask.sum() == 0 or self.aux_k == 0:
            return torch.tensor(0.0, device=residual_x.device)
        residual_flat = residual_x.reshape(-1, self.d_in)
        pre = (residual_flat - self.b_dec) @ self.W_enc + self.b_enc
        pre_dead = pre.masked_fill(~dead_mask, float("-inf"))
        actual_aux_k = int(min(self.aux_k, dead_mask.sum().item()))
        topk_vals, topk_idx = pre_dead.topk(actual_aux_k, dim=-1)
        z_aux = torch.zeros_like(pre)
        z_aux.scatter_(-1, topk_idx, topk_vals)
        residual_hat = z_aux @ self.W_dec
        return F.mse_loss(residual_hat, residual_flat, reduction="mean")
