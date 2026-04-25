"""Phase 5B candidate A: per-(position, scale) feature-nested matryoshka.

Key design decision (from brief.md §A): mathematically, "per-position
matryoshka where every position uses the same prefix indices" reduces
to vanilla H8 matryoshka. To get something genuinely different, we
either:

  (a) decouple decoder weights at the (position, scale) granularity, OR
  (b) use per-position prefix indices (a permutation of [0, d_sae) per t).

This file implements (a): a separate decoder per (position, scale). The
loss structure mirrors H9's feature-nested matryoshka but with the
scale-decoder tensor split per-position:

  W_decs_pps[s] : (prefix_s, T, d_in)        # H9 — same as PPS where t-axis is shared
  W_decs_pps[s, t] : (prefix_s, d_in)         # PPS — fully decoupled per-position

Practical implementation: H9's W_decs_pps[s] is already shape
(prefix_s, T, d_in), which already gives a separate decoder for each
(s, t) — they're just stored as a contiguous tensor along the T axis.
The mathematical difference between H9 and PPS shows up in the LOSS:
H9 sums per-scale loss over all T positions in one MSE term; PPS
applies T separate loss terms and can apply per-position scale weights.

For Phase 5B's PPS we expose:
- Per-(position, scale) loss weighting via `pos_scale_weights` matrix
  of shape (T, n_scales). Default = 1.0 everywhere (= H9 with explicit
  per-position decomposition).
- Optional per-position prefix permutation via `pos_prefix_perm`
  matrix (T, d_sae) where row t is a permutation; the matryoshka
  prefix at scale s for position t is the first prefix_s indices under
  perm[t].

If pos_prefix_perm is None: PPS is mathematically equivalent to H9.
If pos_prefix_perm is provided: PPS is genuinely novel — per-position
feature ordering.

Adds optional anti-dead stack (via parent class composition) AND
optional multi-distance contrastive (via H8-style InfoNCE on the
shared latent z's H prefix).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.txc_bare_antidead import TXCBareAntidead, geometric_median


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


class PerPosScaleMatryoshkaTXC(nn.Module):
    """Per-(position, scale) feature-nested matryoshka TXC.

    n_scales=2 default → matches H8's (H, Full) scales. n_scales=3 adds a
    middle scale.
    """

    def __init__(
        self, d_in: int, d_sae: int, T: int, k: int,
        n_scales: int = 2,
        latent_splits: tuple[int, ...] | None = None,
        pos_prefix_perm: torch.Tensor | None = None,
        alpha: float = 0.0,
        contr_prefix: int | None = None,
        contr_shifts: tuple[int, ...] = (1,),
        contr_weights: tuple[float, ...] | None = None,
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
        auxk_alpha: float = 1.0 / 32.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = int(T)
        self.k = int(k)
        self.n_scales = int(n_scales)
        self.alpha = float(alpha)
        self.aux_k = aux_k
        self.dead_threshold_tokens = dead_threshold_tokens
        self.auxk_alpha = auxk_alpha
        self.contr_shifts = tuple(contr_shifts)
        if contr_weights is None:
            contr_weights = tuple(1.0 / (1.0 + s) for s in contr_shifts)
        self.contr_weights = tuple(contr_weights)

        # Latent splits
        if latent_splits is None:
            base = d_sae // self.n_scales
            remainder = d_sae - base * self.n_scales
            splits = [base + (1 if i < remainder else 0) for i in range(self.n_scales)]
        else:
            splits = list(latent_splits)
            assert len(splits) == self.n_scales and sum(splits) == d_sae
        self.latent_splits = tuple(splits)
        self.prefix_sum = tuple(sum(splits[:i + 1]) for i in range(self.n_scales))

        if contr_prefix is None:
            contr_prefix = self.prefix_sum[0]
        self.contr_prefix = int(contr_prefix)

        # Encoder: shared per-position W_enc (Track 2 / H8 convention)
        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Per-(scale, position) decoders
        # Storage: W_decs_pps is a ParameterList of length n_scales, each entry
        # shape (prefix_s, T, d_in). Same memory layout as H9 but loss is split
        # per-position (allows per-position weighting).
        self.W_decs_pps = nn.ParameterList()
        self.b_decs_pps = nn.ParameterList()
        for s in range(self.n_scales):
            prefix = self.prefix_sum[s]
            W = torch.empty(prefix, T, d_in)
            self.W_decs_pps.append(nn.Parameter(W))
            self.b_decs_pps.append(nn.Parameter(torch.zeros(T, d_in)))

        # Init
        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc.data[t])
        for s in range(self.n_scales):
            prefix = self.prefix_sum[s]
            nn.init.kaiming_uniform_(
                self.W_decs_pps[s].data.view(prefix, T * d_in),
            )
        self._normalize_decoder()
        with torch.no_grad():
            # Init encoder rows from the FULL-scale decoder
            full_dec = self.W_decs_pps[-1]                 # (d_sae, T, d_in)
            for t in range(T):
                self.W_enc.data[t] = full_dec.data[:, t, :].T

        # Optional per-position prefix permutation
        if pos_prefix_perm is not None:
            assert pos_prefix_perm.shape == (T, d_sae)
            self.register_buffer("pos_prefix_perm",
                                  pos_prefix_perm.long().clone())
        else:
            # Identity perm — every position uses the same default ordering
            self.register_buffer(
                "pos_prefix_perm",
                torch.arange(d_sae).unsqueeze(0).expand(T, d_sae).contiguous(),
            )

        # Anti-dead bookkeeping
        self.register_buffer(
            "num_tokens_since_fired", torch.zeros(d_sae, dtype=torch.long),
        )
        self.register_buffer("last_auxk_loss", torch.tensor(-1.0))
        self.register_buffer("last_dead_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("b_dec_initialized", torch.tensor(False))

    @torch.no_grad()
    def _normalize_decoder(self):
        for W in self.W_decs_pps:
            norms = W.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            W.data = W.data / norms

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder(self):
        for W in self.W_decs_pps:
            if W.grad is None:
                continue
            d_sae_eff = W.shape[0]
            W_flat = W.data.view(d_sae_eff, -1)
            g_flat = W.grad.view(d_sae_eff, -1)
            normed = W_flat / (W_flat.norm(dim=1, keepdim=True) + 1e-6)
            parallel = (g_flat * normed).sum(dim=1, keepdim=True)
            g_flat.sub_(parallel * normed)

    @torch.no_grad()
    def init_b_dec_geometric_median(self, x_sample: torch.Tensor):
        """Init each scale's b_dec from geometric median of x at each position."""
        assert not bool(self.b_dec_initialized), "b_dec already initialized"
        for s in range(self.n_scales):
            for t in range(self.T):
                med = geometric_median(x_sample[:, t, :].float())
                self.b_decs_pps[s].data[t] = med.to(self.b_decs_pps[s].dtype)
        self.b_dec_initialized.fill_(True)

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self._pre_activation(x)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z

    def decode_full(self, z: torch.Tensor) -> torch.Tensor:
        """Default decode = largest scale (matches H9)."""
        return self.decode_scale(z, self.n_scales - 1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_full(z)

    def decode_scale(self, z: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Reconstruct (B, T, d_in) at scale s using prefix indices.

        With identity pos_prefix_perm: equivalent to z[:, :prefix_s] @ W_decs[s].
        With non-identity: position t uses z[:, perm[t][:prefix_s]] @ W_decs[s, :, t, :].
        """
        prefix = self.prefix_sum[scale_idx]
        W = self.W_decs_pps[scale_idx]                  # (prefix, T, d_in)
        b = self.b_decs_pps[scale_idx]                  # (T, d_in)

        if (self.pos_prefix_perm == torch.arange(self.d_sae,
                                                  device=z.device).unsqueeze(0)
                .expand_as(self.pos_prefix_perm)).all():
            # Fast path: identity permutation
            z_pref = z[:, :prefix]
            return torch.einsum("bs,std->btd", z_pref, W) + b
        # Slow path: per-position permutation
        out = torch.empty(z.shape[0], self.T, self.d_in,
                           device=z.device, dtype=z.dtype)
        for t in range(self.T):
            perm_t = self.pos_prefix_perm[t, :prefix]    # (prefix,)
            z_t = z.index_select(1, perm_t)              # (B, prefix)
            out[:, t, :] = z_t @ W[:, t, :]
        return out + b

    def _matryoshka_loss_pps(self, x: torch.Tensor, z: torch.Tensor,
                              pos_scale_weights: torch.Tensor | None = None):
        """Per-(position, scale) MSE.

        pos_scale_weights: (T, n_scales) optional. Default 1.0 everywhere.
        Returns (loss, x_hat_full).
        """
        total = 0.0
        x_hat_full = None
        for s in range(self.n_scales):
            x_hat_s = self.decode_scale(z, s)             # (B, T, d_in)
            for t in range(self.T):
                w = (pos_scale_weights[t, s] if pos_scale_weights is not None
                      else 1.0)
                err_t = (x_hat_s[:, t] - x[:, t]).pow(2).sum(dim=-1).mean()
                total = total + w * err_t
            if s == self.n_scales - 1:
                x_hat_full = x_hat_s
        # Normalize by total (T * n_scales) to keep loss scale ≈ 1-position MSE
        total = total / (self.T * self.n_scales)
        return total, x_hat_full

    def _update_dead_and_auxk(
        self, x: torch.Tensor, x_hat: torch.Tensor,
        pre: torch.Tensor, z: torch.Tensor,
    ) -> torch.Tensor:
        active_mask = (z > 0).any(dim=0)
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
        auxk_pre = F.relu(pre).masked_fill(~dead_mask.unsqueeze(0), 0.0)
        vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
        aux_buf = torch.zeros_like(pre)
        aux_buf.scatter_(-1, idx_a, vals_a)
        # Decode aux at the FULL scale.
        full_W = self.W_decs_pps[-1]
        aux_decode = torch.einsum("bs,std->btd", aux_buf, full_W)
        residual = x - x_hat.detach()
        l2_a = (residual - aux_decode).pow(2).sum(dim=-1).mean()
        mu = residual.mean(dim=(0, 1), keepdim=True)
        denom = (residual - mu).pow(2).sum(dim=-1).mean()
        l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        self.last_auxk_loss.fill_(float(l_auxk.detach()))
        return l_auxk

    def forward(self, x: torch.Tensor, alpha: float | None = None):
        """Accepts (B, T, d) single-window OR (B, 1+K, T, d) multi-distance.

        Multi-distance: anchor + K positives at H8 shifts. Each window
        contributes recon, plus InfoNCE on the H prefix between
        (anchor, positive_k) pairs with self.contr_weights.
        """
        eff_alpha = self.alpha if alpha is None else alpha

        if x.ndim == 3:
            pre = self._pre_activation(x)
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
            l_recon, x_hat = self._matryoshka_loss_pps(x, z)
            l_auxk = self._update_dead_and_auxk(x, x_hat, pre, z)
            total = l_recon + self.auxk_alpha * l_auxk
            return total, x_hat, z

        if x.ndim == 4 and x.shape[1] >= 2:
            K = x.shape[1] - 1
            assert K == len(self.contr_shifts), (
                f"expected {1+len(self.contr_shifts)} window slots, got {x.shape[1]}"
            )
            x_anchor = x[:, 0]
            pre_a = self._pre_activation(x_anchor)
            vals_a, idx_a = pre_a.topk(self.k, dim=-1)
            z_anchor = torch.zeros_like(pre_a)
            z_anchor.scatter_(1, idx_a, F.relu(vals_a))
            l_recon, x_hat = self._matryoshka_loss_pps(x_anchor, z_anchor)

            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            for k_idx, w_s in enumerate(self.contr_weights):
                x_pos = x[:, 1 + k_idx]
                pre_p = self._pre_activation(x_pos)
                vals_p, idx_p = pre_p.topk(self.k, dim=-1)
                z_pos = torch.zeros_like(pre_p)
                z_pos.scatter_(1, idx_p, F.relu(vals_p))
                lr_pos, _ = self._matryoshka_loss_pps(x_pos, z_pos)
                l_recon = l_recon + lr_pos
                if eff_alpha > 0.0:
                    h = self.contr_prefix
                    l_contr = l_contr + w_s * _info_nce(z_anchor[:, :h], z_pos[:, :h])

            l_auxk = self._update_dead_and_auxk(x_anchor, x_hat, pre_a, z_anchor)
            total = l_recon + eff_alpha * l_contr + self.auxk_alpha * l_auxk
            return total, x_hat, z_anchor

        raise ValueError(f"unexpected input shape {tuple(x.shape)}")
