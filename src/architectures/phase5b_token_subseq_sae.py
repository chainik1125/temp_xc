"""Phase 5B candidate C: token-level encoder + sparse subset-sum.

C1: pos_mode="none" — encoder is per-TOKEN (no T axis):
    `pre = x @ W_enc + b_enc`.

C2: pos_mode="sinusoidal" — adds fixed sinusoidal pos emb to INPUT
    before encoding: `pre = (x + sin_pe[t]) @ W_enc + b_enc`.
    Position info is constrained to a rank-d_in subspace of d_sae
    (since `pe[t] @ W_enc` has rank ≤ d_in=2304).

C3: pos_mode="learned" — adds full-rank learnable per-feature,
    per-position bias DIRECTLY in d_sae space:
    `pre = x @ W_enc + b_enc + delta[t]`, where delta is a learnable
    parameter of shape (L_max, d_sae). Each feature j has its own
    learned position-response curve δ[:, j]. Strict generalization of
    C2 (which constrains position info to rank ≤ d_in).

Encoder param scaling is independent of T for all three modes:
    none/sinusoidal:  d_in · d_sae       (= 42M @ d_sae=18432)
    learned:          d_in · d_sae + L_max · d_sae    (+ 2.4M @ L_max=128)

This breaks the per-position W_enc/W_dec convention of vanilla TXC —
intentional. The C-family hypothesis: temporal mixing comes from
the SUM aggregation, not from per-position encoder weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_position_embedding(L: int, d: int,
                                    device: torch.device | None = None) -> torch.Tensor:
    """Standard transformer-style sinusoidal pos embedding (read-only)."""
    pos = torch.arange(L, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d, 2, dtype=torch.float32, device=device)
        * -(math.log(10000.0) / d)
    )
    pe = torch.zeros(L, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TokenSubseqSAE(nn.Module):
    """Token-level encoder + subsequence-summed latent.

    Args:
        d_in: residual stream width.
        d_sae: dictionary size.
        L_max: maximum sequence length (used for pos emb size).
        k: TopK budget on EACH per-token z BEFORE summing. The summed z
           may have up to t_sample * k active features.
        t_sample: number of positions to sum at training time.
        pos_mode: "none" / "sinusoidal" / "learned".
            "none":      no position info (C1).
            "sinusoidal":fixed sinusoidal pos emb added to INPUT (C2).
            "learned":   learnable (L_max, d_sae) bias added to PRE-ACTIVATION (C3).
        pos_scale: scale applied to sinusoidal pos emb only.
        use_pos: legacy bool flag, mapped to pos_mode if pos_mode is "none"
                 (use_pos=True → "sinusoidal"). Kept for backward compat.
    """

    def __init__(self, d_in: int, d_sae: int, L_max: int, k: int,
                 t_sample: int,
                 pos_mode: str | None = None,
                 use_pos: bool = False,
                 pos_scale: float = 1.0,
                 aux_k: int = 512,
                 dead_threshold_tokens: int = 10_000_000,
                 auxk_alpha: float = 1.0 / 32.0):
        super().__init__()
        # Resolve pos_mode (with use_pos legacy fallback)
        if pos_mode is None:
            pos_mode = "sinusoidal" if use_pos else "none"
        if pos_mode not in ("none", "sinusoidal", "learned"):
            raise ValueError(f"unknown pos_mode={pos_mode}")
        self.d_in = d_in
        self.d_sae = d_sae
        self.L_max = int(L_max)
        self.k = int(k)
        self.t_sample = int(t_sample)
        self.pos_mode = pos_mode
        # Keep `use_pos` for downstream code that reads it (e.g. saving meta)
        self.use_pos = (pos_mode == "sinusoidal")
        self.pos_scale = float(pos_scale)
        self.aux_k = aux_k
        self.dead_threshold_tokens = dead_threshold_tokens
        self.auxk_alpha = auxk_alpha

        # Single shared encoder + decoder
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_enc.data)
        nn.init.kaiming_uniform_(self.W_dec.data)
        self._normalize_decoder()
        with torch.no_grad():
            self.W_enc.data = self.W_dec.data.T.clone()

        # Position info storage
        if pos_mode == "sinusoidal":
            pe = _sinusoidal_position_embedding(self.L_max, d_in)
            self.register_buffer("pos_emb", pe * self.pos_scale)
        else:
            self.register_buffer("pos_emb", torch.zeros(0))

        if pos_mode == "learned":
            # Per-(position, feature) learnable bias in d_sae space.
            # Init to zero so early training matches the C1 baseline.
            self.pos_bias = nn.Parameter(torch.zeros(self.L_max, d_sae))
        else:
            # Register a zero-size parameter so state_dict shapes match
            # only when pos_mode is "learned"; otherwise omit entirely.
            self.pos_bias = None

        # Dead-feature tracker (same convention as txc_bare_antidead)
        self.register_buffer(
            "num_tokens_since_fired", torch.zeros(d_sae, dtype=torch.long),
        )
        self.register_buffer("last_auxk_loss", torch.tensor(-1.0))
        self.register_buffer("last_dead_count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder(self):
        if self.W_dec.grad is None:
            return
        normed = self.W_dec.data / (
            self.W_dec.data.norm(dim=-1, keepdim=True) + 1e-6
        )
        parallel = (self.W_dec.grad * normed).sum(dim=-1, keepdim=True)
        self.W_dec.grad.sub_(parallel * normed)

    def _per_token_pre(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """x: (B, t_sample, d_in). positions: (B, t_sample) absolute indices.
        Returns (B, t_sample, d_sae) per-token pre-activation.
        """
        if self.pos_mode == "sinusoidal":
            pe = self.pos_emb[positions]   # (B, t_sample, d_in)
            x_input = x + pe
        else:
            x_input = x
        # Per-token encode: (B, t_sample, d_in) @ (d_in, d_sae) → (B, t_sample, d_sae)
        pre = torch.einsum("btd,ds->bts", x_input, self.W_enc) + self.b_enc
        if self.pos_mode == "learned":
            # Add per-(position, feature) learnable bias in d_sae space.
            pre = pre + self.pos_bias[positions]   # (B, t_sample, d_sae)
        return pre

    def encode_per_token(self, x_full: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """For probe-time:
        x_full: (B, L, d_in) — full sequence or window.
        positions: (B, L) absolute positions (for pos emb). If None, use [0, L).
        Returns (B, L, d_sae) per-token sparse codes.
        """
        B, L, d = x_full.shape
        if positions is None:
            positions = torch.arange(L, device=x_full.device).unsqueeze(0).expand(B, L)
        pre = self._per_token_pre(x_full, positions)
        # Per-token TopK
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, F.relu(vals))
        return z

    def encode_sum_subset(self, x_full: torch.Tensor,
                          subset_indices: torch.Tensor | None = None) -> torch.Tensor:
        """Probe-time aggregation: sum of per-token sparse codes over a subset.

        x_full: (B, L, d_in).
        subset_indices: (B, t_sample) positions to sum. If None, sum all L.
        """
        z_per = self.encode_per_token(x_full)            # (B, L, d_sae)
        if subset_indices is None:
            return z_per.sum(dim=1)
        gi = subset_indices.unsqueeze(-1).expand(-1, -1, self.d_sae)
        z_S = z_per.gather(1, gi)                         # (B, t_sample, d_sae)
        return z_S.sum(dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Default encode interface — sum over WHOLE input window.

        Used by the probe pipeline. If x has shape (B, T, d), sum z's over
        all T positions. If x has shape (B, d), treat as t_sample=1.
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.encode_sum_subset(x, subset_indices=None)

    def decode_one(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_sae) → (B, d_in). Single-token reconstruction."""
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None):
        """x: (B, t_sample, d_in). positions: (B, t_sample) absolute indices.

        Loss = sum_t MSE(x[t], decode(sum_{t' in S} z_t')) — but THAT's the
        "all-positions-share-the-summed-z" interpretation. We use a simpler
        & cleaner one:

          pre_t = encoded x_t with optional pos emb
          z_t   = TopK(pre_t)
          z_sum = sum_t z_t                        # global summary of S
          x_hat_t = decode(z_t)                    # per-token recon

        i.e., per-token recon by per-token z. The SUM is exposed via
        `encode_sum_subset` for probe-time.

        Hypothesis: per-token z's compose well under SUMMATION because
        TopK-sparse + linear decode gives a well-defined sum, and the
        decoder is shared.
        """
        B, t_sample, d = x.shape
        if positions is None:
            positions = torch.randint(0, self.L_max, (B, t_sample), device=x.device)

        pre = self._per_token_pre(x, positions)           # (B, t_sample, d_sae)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, F.relu(vals))                 # (B, t_sample, d_sae)

        # Per-token reconstruction
        x_hat = self.decode_one(z)                        # (B, t_sample, d_in)
        l_recon = (x_hat - x).pow(2).sum(dim=-1).mean()

        # AuxK on aggregated dead-feature signal
        l_auxk = self._auxk(x, x_hat, pre, z)
        total = l_recon + self.auxk_alpha * l_auxk

        # Return summed z as the primary latent (probe API expects this)
        z_sum = z.sum(dim=1)
        return total, x_hat, z_sum

    def _auxk(self, x, x_hat, pre, z):
        """AuxK on dead features over per-token signal."""
        active_mask = (z > 0).any(dim=(0, 1))                    # (d_sae,)
        n_tokens = x.shape[0] * x.shape[1]
        self.num_tokens_since_fired += n_tokens
        self.num_tokens_since_fired[active_mask] = 0
        dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
        n_dead = int(dead_mask.sum().item())
        self.last_dead_count.fill_(n_dead)
        if n_dead == 0:
            self.last_auxk_loss.fill_(0.0)
            return torch.zeros((), device=x.device, dtype=x.dtype)
        k_aux = min(self.aux_k, n_dead)
        auxk_pre = F.relu(pre).masked_fill(~dead_mask.view(1, 1, -1), 0.0)
        vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
        aux_buf = torch.zeros_like(pre)
        aux_buf.scatter_(-1, idx_a, vals_a)
        aux_decode = aux_buf @ self.W_dec
        residual = (x - x_hat.detach())
        l2_a = (residual - aux_decode).pow(2).sum(dim=-1).mean()
        mu = residual.mean(dim=(0, 1), keepdim=True)
        denom = (residual - mu).pow(2).sum(dim=-1).mean()
        l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        self.last_auxk_loss.fill_(float(l_auxk.detach()))
        return l_auxk
