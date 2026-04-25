"""Phase 5B candidate F: subset-encoder TXC with t < T slabs.

Variant of the subseq sampling family where the encoder has only `t`
weight slabs (not T_window). Training samples t positions out of a
T_window-token raw context, sorts them, and assigns sorted-rank-i to
encoder slab i. The model learns position-rank-invariant features
across the full T_window receptive field.

Probe-time strategy options:
  - "fixed_last_t":  take last t raw tokens (= vanilla TXCDR T=t).
                     Wastes the T_window training signal.
  - "stride":        take T_window raw tokens, evenly-stride to t.
                     Equivalent to D1 strided.
  - "random_K_subsets" (default): take T_window raw tokens, sample K
                     random t-subsets, encode each through the t slabs,
                     average the resulting z's. Most closely matches
                     the training distribution.

Anti-dead stack inherited from `TXCBareAntidead` (matched to Track 2 / B2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.txc_bare_antidead import TXCBareAntidead


class SubsetEncoderTXC(TXCBareAntidead):
    """t encoder/decoder slabs, T_window receptive field.

    Shape:
        W_enc ∈ R^{t × d_in × d_sae}
        W_dec ∈ R^{d_sae × t × d_in}

    Training: sample t of T_window positions, sort ascending, encode
    sorted-position-i with slab-i. Recon loss only at sampled positions.

    Probe: depends on probe_strategy (see module docstring).

    Args:
        d_in, d_sae:        same as parent.
        T_window:           raw receptive field at training (e.g. 10, 20, 128).
        t:                  number of encoder/decoder slabs (small; e.g. 5).
        k:                  global TopK on d_sae pre-activation.
        probe_strategy:     "fixed_last_t" / "stride" / "random_K_subsets".
        probe_K:            number of subsets for the "random_K_subsets" probe.
        anti-dead kwargs:   forwarded to parent.
    """

    def __init__(self, d_in: int, d_sae: int, T_window: int, t: int, k: int,
                 probe_strategy: str = "random_K_subsets",
                 probe_K: int = 16,
                 aux_k: int = 512,
                 dead_threshold_tokens: int = 10_000_000,
                 auxk_alpha: float = 1.0 / 32.0):
        # Parent expects T = number of position slabs, so pass t (not T_window).
        super().__init__(d_in, d_sae, t, k,
                          aux_k=aux_k,
                          dead_threshold_tokens=dead_threshold_tokens,
                          auxk_alpha=auxk_alpha)
        if probe_strategy not in ("fixed_last_t", "stride", "random_K_subsets"):
            raise ValueError(f"unknown probe_strategy={probe_strategy}")
        self.T_window = int(T_window)
        self.t = int(t)
        self.probe_strategy = probe_strategy
        self.probe_K = int(probe_K)
        assert 1 <= self.t <= self.T_window

    # ──────────────────────────────────────────────── training

    def forward(self, x: torch.Tensor):
        """x: (B, T_window, d_in). Sample t positions per row, sort,
        assign by rank to t slabs.
        """
        B, T_w, d = x.shape
        assert T_w == self.T_window, f"expected T_window={self.T_window}, got {T_w}"

        # Sample t of T_window positions per row, sorted ascending
        keys = torch.rand(B, T_w, device=x.device)
        _, idx = keys.topk(self.t, dim=-1)               # (B, t), indices
        sample_idx, _ = idx.sort(dim=-1)                  # ascending, (B, t)

        # Gather x at sampled positions: (B, t, d_in)
        gather = sample_idx.unsqueeze(-1).expand(-1, -1, d)
        x_t = x.gather(1, gather)

        # Standard t-position pre-activation
        pre = torch.einsum("btd,tds->bs", x_t, self.W_enc) + self.b_enc
        vals, sel_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, sel_idx, F.relu(vals))             # (B, d_sae)

        # Decode (B, t, d_in)
        x_hat = torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec
        l_recon = (x_hat - x_t).pow(2).sum(dim=-1).mean()

        # AuxK on dead features — parent's logic inlined (it's not factored
        # into a callable method on the base class). The (x_t, x_hat) pair
        # has shape (B, t, d_in), matching the parent's expectations for the
        # dead-feature residual machinery.
        active_mask = (z > 0).any(dim=0)
        n_tokens = x_t.shape[0] * x_t.shape[1]            # B * t
        self.num_tokens_since_fired += n_tokens
        self.num_tokens_since_fired[active_mask] = 0
        dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
        n_dead = int(dead_mask.sum().item())
        self.last_dead_count.fill_(n_dead)
        if n_dead > 0:
            k_aux = min(self.aux_k, n_dead)
            auxk_pre = F.relu(pre).masked_fill(~dead_mask.unsqueeze(0), 0.0)
            vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
            aux_buf = torch.zeros_like(pre)
            aux_buf.scatter_(-1, idx_a, vals_a)
            aux_decode = torch.einsum("bs,std->btd", aux_buf, self.W_dec)
            residual = x_t - x_hat.detach()
            l2_a = (residual - aux_decode).pow(2).sum(dim=-1).mean()
            mu = residual.mean(dim=(0, 1), keepdim=True)
            denom = (residual - mu).pow(2).sum(dim=-1).mean()
            l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
            self.last_auxk_loss.fill_(float(l_auxk.detach()))
        else:
            l_auxk = torch.zeros((), device=x.device, dtype=x.dtype)
            self.last_auxk_loss.fill_(0.0)

        total = l_recon + self.auxk_alpha * l_auxk
        return total, x_hat, z

    # ──────────────────────────────────────────────── probe

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Probe-time encoder. x: (B, T_window, d_in) → (B, d_sae).

        Strategy controlled by self.probe_strategy.
        """
        B, T_w, d = x.shape
        if T_w != self.T_window:
            # Allow shorter inputs (e.g. last_position with smaller anchor):
            # truncate-to-T_window, or pad with first token if shorter.
            if T_w > self.T_window:
                x = x[:, -self.T_window:, :]
                T_w = self.T_window
            else:
                pad = x[:, :1, :].expand(-1, self.T_window - T_w, -1)
                x = torch.cat([pad, x], dim=1)
                T_w = self.T_window

        if self.probe_strategy == "fixed_last_t":
            x_t = x[:, -self.t:, :]                       # (B, t, d_in)
            return self._encode_t_window(x_t)

        if self.probe_strategy == "stride":
            stride = max(1, self.T_window // self.t)
            positions = torch.arange(
                self.T_window - 1 - (self.t - 1) * stride,
                self.T_window, stride, device=x.device,
            )
            if positions.numel() != self.t:
                # Fallback: take last t evenly-spaced positions
                positions = torch.linspace(
                    0, self.T_window - 1, self.t, device=x.device
                ).long()
            x_t = x[:, positions, :]
            return self._encode_t_window(x_t)

        if self.probe_strategy == "random_K_subsets":
            return self._encode_K_random(x)

        raise ValueError(f"unknown probe_strategy={self.probe_strategy}")

    def _encode_t_window(self, x_t: torch.Tensor) -> torch.Tensor:
        """x_t: (B, t, d_in) — already-sampled t positions in slab order."""
        pre = torch.einsum("btd,tds->bs", x_t, self.W_enc) + self.b_enc
        vals, sel = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, sel, F.relu(vals))
        return z

    def _encode_K_random(self, x: torch.Tensor) -> torch.Tensor:
        """Sample probe_K random t-subsets, encode each, average z's."""
        B, T_w, d = x.shape
        z_acc = torch.zeros(B, self.d_sae, device=x.device, dtype=x.dtype)
        for _ in range(self.probe_K):
            keys = torch.rand(B, T_w, device=x.device)
            _, idx = keys.topk(self.t, dim=-1)
            idx, _ = idx.sort(dim=-1)
            gather = idx.unsqueeze(-1).expand(-1, -1, d)
            x_t = x.gather(1, gather)
            z_acc = z_acc + self._encode_t_window(x_t)
        return z_acc / self.probe_K
