"""Phase 7 Z hill-climb variants: rank-routed slots + shared-weight SubseqH8.

Two variants of SubseqH8 designed to fit the 5090's 32 GB VRAM at large
T_max. The original SubseqH8 in `phase5b_subseq_sampling_txcdr.py` carries
T_max-many encoder/decoder slabs; the encoder forward stores activations
for all T_max positions even though only t_sample positions feed the loss.
That's the OOM bottleneck Z hit at T_max ≥ 14 (see
`docs/han/research_logs/phase7_unification/agent_z_paper/2026-04-29-z-handover.md`).

Both variants here remove that bottleneck.

Variant A — `SubseqRankedH8` (per-sampled-slot weights):
    - W_enc shape (t_sample, d_in, d_sae): one slab per *sample slot*,
      not per absolute position.
    - Train: sample t_sample positions out of T_max, sort ascending,
      slot-k applies to the position with rank k. Encoder forward only
      runs over t_sample positions, so peak activation memory scales
      with t_sample (not T_max).
    - Probe: pick t_sample equally-spaced positions out of T_max and
      route by rank — a deterministic analogue of training-time random
      sampling.
    - All other H8 machinery (matryoshka H/L recon, multi-distance
      InfoNCE, AuxK, decoder-parallel grad removal, b_dec geometric-
      median init) unchanged via inheritance.

Variant B — `SubseqSharedH8` (fully shared encoder/decoder):
    - W_enc shape (d_in, d_sae), W_dec shape (d_sae, d_in): one
      position-agnostic slab.
    - Train: pre-act is W_enc applied to the sum-pooled window (over
      T_max positions). Single (B, d_sae) sparse code per window. The
      decoder predicts a single (B, d) vector applied per-position
      (broadcast).
    - Multi-distance InfoNCE on z_anchor vs z_pos_at_shift_s — the
      contrastive structure is what differentiates this from a plain
      pooled SAE.
    - Smallest possible parameter footprint; trivially fits 5090 even
      at T_max=32.

Both variants accept multi-distance input (B, 1+K, T_max, d) where
K = len(shifts), and a single-window fallback (B, T_max, d).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.txc_bare_antidead import geometric_median
from src.architectures.txc_bare_matryoshka_contrastive_antidead import _info_nce
from src.architectures.txc_bare_multidistance_contrastive_antidead import (
    TXCBareMultiDistanceContrastiveAntidead,
)


def _draw_sample_idx(T_max: int, t_sample: int, B: int,
                     contiguous: bool, device: torch.device) -> torch.Tensor:
    """Return (B, t_sample) sorted indices into [0, T_max)."""
    if contiguous:
        max_off = T_max - t_sample + 1
        offs = torch.randint(0, max_off, (B,), device=device)
        rng = torch.arange(t_sample, device=device)
        return offs.unsqueeze(1) + rng.unsqueeze(0)
    keys = torch.rand(B, T_max, device=device)
    _, idx = keys.topk(t_sample, dim=-1, largest=True, sorted=False)
    idx, _ = idx.sort(dim=-1)
    return idx


def _equally_spaced_offsets(T_max: int, t_sample: int,
                             device: torch.device) -> torch.Tensor:
    """Return (t_sample,) integer positions equally spaced in [0, T_max).

    Probe-time analogue of training's random sample — slot k gets the
    k-th equally-spaced position so the rank-routing is preserved.
    """
    raw = (torch.arange(t_sample, device=device).float() + 0.5) * T_max / t_sample
    return raw.long().clamp(0, T_max - 1)


class SubseqRankedH8(TXCBareMultiDistanceContrastiveAntidead):
    """SubseqH8 with t_sample-many encoder/decoder slabs (rank-routed).

    Inherits from `TXCBareMultiDistanceContrastiveAntidead` with the
    parent's `T` argument set to `t_sample`. That allocates W_enc as
    (t_sample, d_in, d_sae) and W_dec as (d_sae, t_sample, d_in) — the
    parent thinks "T positions" but we re-interpret them as "slots".
    Window length T_max is tracked separately on `self.T_max`.

    `shifts` are interpreted in T_max-window space, NOT slot space, since
    the data generator (`make_multidistance_pair_gen_gpu`) shifts in the
    underlying token grid.
    """

    def __init__(
        self, d_in: int, d_sae: int, T_max: int, t_sample: int, k: int,
        shifts: tuple[int, ...] | None = None,
        weights: tuple[float, ...] | None = None,
        matryoshka_h_size: int | None = None,
        alpha: float = 1.0,
        contiguous: bool = False,
        contr_prefix: int | None = None,
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
        auxk_alpha: float = 1.0 / 32.0,
    ):
        assert 1 <= t_sample <= T_max
        # Auto-shifts in T_max-space if caller didn't supply them. Parent's
        # default uses self.T (== t_sample after super().__init__) which is
        # wrong, so compute here and pass explicitly.
        if shifts is None:
            raw = (1, max(1, T_max // 4), max(1, T_max // 2))
            shifts = tuple(sorted(set(s for s in raw if 1 <= s <= T_max - 1)))
        super().__init__(
            d_in, d_sae, T=t_sample, k=k,
            shifts=shifts, weights=weights,
            matryoshka_h_size=matryoshka_h_size,
            alpha=alpha,
            contr_prefix=contr_prefix,
            aux_k=aux_k,
            dead_threshold_tokens=dead_threshold_tokens,
            auxk_alpha=auxk_alpha,
        )
        for s in self.shifts:
            assert 1 <= s <= T_max - 1, (
                f"shift {s} not in [1, T_max-1] for T_max={T_max}"
            )
        self.T_max = int(T_max)
        self.t_sample = int(t_sample)
        self.contiguous = bool(contiguous)

    def _gather(self, x_window: torch.Tensor,
                sample_idx: torch.Tensor) -> torch.Tensor:
        """x_window: (B, T_max, d). sample_idx: (B, t_sample). -> (B, t_sample, d)."""
        d = x_window.shape[-1]
        gi = sample_idx.unsqueeze(-1).expand(-1, -1, d)
        return x_window.gather(1, gi)

    @torch.no_grad()
    def init_b_dec_geometric_median(self, x_sample: torch.Tensor):
        """Adapter: caller passes (B, T_max, d); parent expects (B, t_sample, d).

        Uses equally-spaced offsets so each slot's b_dec is initialised
        from positions in roughly the same quantile that slot will see
        at probe time (rank-routing).
        """
        if x_sample.ndim == 3 and x_sample.shape[1] == self.T_max:
            offsets = _equally_spaced_offsets(
                self.T_max, self.t_sample, x_sample.device,
            )
            x_S = x_sample[:, offsets]
            super().init_b_dec_geometric_median(x_S)
            return
        if x_sample.ndim == 3 and x_sample.shape[1] == self.t_sample:
            super().init_b_dec_geometric_median(x_sample)
            return
        raise ValueError(
            f"init_b_dec expects (B, T_max={self.T_max}, d) or "
            f"(B, t_sample={self.t_sample}, d), got {tuple(x_sample.shape)}"
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Probe-time: x is (B, T_max, d). Equally-spaced slot routing."""
        if x.ndim == 3 and x.shape[1] == self.T_max:
            offsets = _equally_spaced_offsets(
                self.T_max, self.t_sample, x.device,
            )
            x_S = x[:, offsets]
            pre = self._pre_activation(x_S)
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
            return z
        if x.ndim == 3 and x.shape[1] == self.t_sample:
            # Already at slot dim — straight through.
            pre = self._pre_activation(x)
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
            return z
        raise ValueError(
            f"encode expects (B, T_max={self.T_max}, d) or "
            f"(B, t_sample={self.t_sample}, d), got {tuple(x.shape)}"
        )

    def forward(self, x: torch.Tensor, alpha: float | None = None):
        """Multi-distance forward. x: (B, 1+K, T_max, d) or (B, T_max, d)."""
        eff_alpha = self.alpha if alpha is None else alpha

        if x.ndim == 4 and x.shape[1] >= 2:
            K = x.shape[1] - 1
            assert K == len(self.shifts), (
                f"expected {1+len(self.shifts)} window slots, got {x.shape[1]}"
            )
            B = x.shape[0]
            sample_idx = _draw_sample_idx(
                self.T_max, self.t_sample, B, self.contiguous, x.device,
            )

            x_anchor_S = self._gather(x[:, 0], sample_idx)
            pre_a = self._pre_activation(x_anchor_S)
            vals_a, idx_a = pre_a.topk(self.k, dim=-1)
            z_anchor = torch.zeros_like(pre_a)
            z_anchor.scatter_(1, idx_a, F.relu(vals_a))
            l_recon = self._recon_loss(x_anchor_S, z_anchor)

            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            for k_idx, w_s in enumerate(self.loss_weights):
                x_pos_S = self._gather(x[:, 1 + k_idx], sample_idx)
                pre_p = self._pre_activation(x_pos_S)
                vals_p, idx_p = pre_p.topk(self.k, dim=-1)
                z_pos = torch.zeros_like(pre_p)
                z_pos.scatter_(1, idx_p, F.relu(vals_p))
                l_recon = l_recon + self._recon_loss(x_pos_S, z_pos)
                if eff_alpha > 0.0:
                    h = self.contr_prefix
                    l_contr = l_contr + w_s * _info_nce(
                        z_anchor[:, :h], z_pos[:, :h]
                    )

            x_hat = self.decode(z_anchor)
            l_auxk = self._update_dead_and_auxk(
                x_anchor_S, x_hat, pre_a, z_anchor,
            )
            total = l_recon + eff_alpha * l_contr + self.auxk_alpha * l_auxk
            return total, x_hat, z_anchor

        if x.ndim == 3:
            B = x.shape[0]
            if x.shape[1] == self.T_max:
                sample_idx = _draw_sample_idx(
                    self.T_max, self.t_sample, B, self.contiguous, x.device,
                )
                x_S = self._gather(x, sample_idx)
            elif x.shape[1] == self.t_sample:
                x_S = x
            else:
                raise ValueError(
                    f"single-window forward expects (B, T_max={self.T_max}, d) "
                    f"or (B, t_sample={self.t_sample}, d); got {tuple(x.shape)}"
                )
            pre = self._pre_activation(x_S)
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
            x_hat = self.decode(z)
            l_recon = self._recon_loss(x_S, z)
            l_auxk = self._update_dead_and_auxk(x_S, x_hat, pre, z)
            total = l_recon + self.auxk_alpha * l_auxk
            return total, x_hat, z

        raise ValueError(f"unexpected input shape {tuple(x.shape)}")


class SubseqSharedH8(nn.Module):
    """Shared-weights SubseqH8: one (d_in, d_sae) encoder + (d_sae, d_in) decoder.

    The window pre-activation is W_enc applied to the *pooled* T_max-window
    (sum-pool by default; mean-pool optional). One sparse code per window.
    The decoder predicts a single (B, d) vector that is broadcast across
    all T_max positions for the per-position MSE.

    Multi-distance contrastive shifts the window in token-space, then
    contrasts z_anchor against each shifted z_pos at the H prefix —
    same loss structure as TXCBareMultiDistanceContrastiveAntidead but
    with a position-agnostic encoder.
    """

    def __init__(
        self, d_in: int, d_sae: int, T_max: int, k: int,
        shifts: tuple[int, ...] | None = None,
        weights: tuple[float, ...] | None = None,
        matryoshka_h_size: int | None = None,
        alpha: float = 1.0,
        contr_prefix: int | None = None,
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
        auxk_alpha: float = 1.0 / 32.0,
        sum_pool: bool = True,
    ):
        super().__init__()
        self.d_in = int(d_in)
        self.d_sae = int(d_sae)
        self.T_max = int(T_max)
        self.k = int(k)
        self.aux_k = int(aux_k)
        self.dead_threshold_tokens = int(dead_threshold_tokens)
        self.auxk_alpha = float(auxk_alpha)
        self.sum_pool = bool(sum_pool)

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_dec.data)
        self._normalize_decoder()
        with torch.no_grad():
            self.W_enc.data = self.W_dec.data.T.contiguous()

        if shifts is None:
            raw = (1, max(1, T_max // 4), max(1, T_max // 2))
            shifts = tuple(sorted(set(s for s in raw if 1 <= s <= T_max - 1)))
        for s in shifts:
            assert 1 <= s <= T_max - 1, (
                f"shift {s} not in [1, T_max-1] for T_max={T_max}"
            )
        self.shifts = tuple(shifts)
        if weights is None:
            weights = tuple(1.0 / (1.0 + s) for s in shifts)
        self.loss_weights = tuple(weights)
        assert len(self.loss_weights) == len(self.shifts)

        self.matryoshka_h_size = matryoshka_h_size
        self.alpha = float(alpha)
        if contr_prefix is None:
            contr_prefix = matryoshka_h_size or int(d_sae * 0.2)
        self.contr_prefix = int(contr_prefix)

        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(d_sae, dtype=torch.long),
        )
        self.register_buffer("last_auxk_loss", torch.tensor(-1.0))
        self.register_buffer("last_dead_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("b_dec_initialized", torch.tensor(False))

    @torch.no_grad()
    def _normalize_decoder(self):
        """Unit-norm per decoder atom (rows of W_dec)."""
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder(self):
        if self.W_dec.grad is None:
            return
        normed = self.W_dec.data / (
            self.W_dec.data.norm(dim=1, keepdim=True) + 1e-6
        )
        parallel = (self.W_dec.grad * normed).sum(dim=1, keepdim=True)
        self.W_dec.grad.sub_(parallel * normed)

    @torch.no_grad()
    def init_b_dec_geometric_median(self, x_sample: torch.Tensor):
        """x_sample: (B, T_max, d). b_dec is the median of the pooled window."""
        assert not bool(self.b_dec_initialized), "b_dec already initialized"
        target = self._pool(x_sample.float())
        med = geometric_median(target)
        self.b_dec.data = med.to(self.b_dec.dtype)
        self.b_dec_initialized.fill_(True)

    def _pool(self, x_window: torch.Tensor) -> torch.Tensor:
        """(B, T_max, d) -> (B, d). Sum-pool by default, else mean-pool."""
        return x_window.sum(dim=1) if self.sum_pool else x_window.mean(dim=1)

    def _pre_activation_window(self, x_window: torch.Tensor) -> torch.Tensor:
        return self._pool(x_window) @ self.W_enc + self.b_enc

    def encode(self, x_window: torch.Tensor) -> torch.Tensor:
        pre = self._pre_activation_window(x_window)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, d_sae) -> (B, d). Single position-agnostic output vector."""
        return z @ self.W_dec + self.b_dec

    def _decode_prefix(self, z: torch.Tensor, h_size: int) -> torch.Tensor:
        return z[:, :h_size] @ self.W_dec[:h_size] + self.b_dec

    def _recon_loss(self, x_window: torch.Tensor,
                     z: torch.Tensor) -> torch.Tensor:
        """Per-position MSE: same z reconstructs every position. Matryoshka if set."""
        x_hat = self.decode(z).unsqueeze(1)  # (B, 1, d) — broadcasts across T_max
        l_full = (x_window - x_hat).pow(2).sum(dim=-1).mean()
        if self.matryoshka_h_size is None:
            return l_full
        x_hat_h = self._decode_prefix(z, self.matryoshka_h_size).unsqueeze(1)
        l_h = (x_window - x_hat_h).pow(2).sum(dim=-1).mean()
        return l_h + l_full

    def _update_dead_and_auxk(self, x_window: torch.Tensor,
                                x_hat: torch.Tensor,
                                pre: torch.Tensor,
                                z: torch.Tensor) -> torch.Tensor:
        active_mask = (z > 0).any(dim=0)
        n_tokens = x_window.shape[0] * x_window.shape[1]
        self.num_tokens_since_fired += n_tokens
        self.num_tokens_since_fired[active_mask] = 0
        dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
        n_dead = int(dead_mask.sum().item())
        self.last_dead_count.fill_(n_dead)
        if n_dead == 0:
            self.last_auxk_loss.fill_(0.0)
            return torch.zeros((), device=x_window.device, dtype=x_window.dtype)
        k_aux = min(self.aux_k, n_dead)
        auxk_pre = F.relu(pre).masked_fill(~dead_mask.unsqueeze(0), 0.0)
        vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
        aux_buf = torch.zeros_like(pre)
        aux_buf.scatter_(-1, idx_a, vals_a)
        aux_decode = (aux_buf @ self.W_dec).unsqueeze(1)  # (B, 1, d)
        residual = x_window - x_hat.detach().unsqueeze(1)
        l2_a = (residual - aux_decode).pow(2).sum(dim=-1).mean()
        mu = residual.mean(dim=(0, 1), keepdim=True)
        denom = (residual - mu).pow(2).sum(dim=-1).mean()
        l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        self.last_auxk_loss.fill_(float(l_auxk.detach()))
        return l_auxk

    def forward(self, x: torch.Tensor, alpha: float | None = None):
        """x: (B, 1+K, T_max, d) or (B, T_max, d)."""
        eff_alpha = self.alpha if alpha is None else alpha

        if x.ndim == 4 and x.shape[1] >= 2:
            K = x.shape[1] - 1
            assert K == len(self.shifts), (
                f"expected {1+len(self.shifts)} window slots, got {x.shape[1]}"
            )

            x_anchor = x[:, 0]
            pre_a = self._pre_activation_window(x_anchor)
            vals_a, idx_a = pre_a.topk(self.k, dim=-1)
            z_anchor = torch.zeros_like(pre_a)
            z_anchor.scatter_(1, idx_a, F.relu(vals_a))
            l_recon = self._recon_loss(x_anchor, z_anchor)

            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            for k_idx, w_s in enumerate(self.loss_weights):
                x_pos = x[:, 1 + k_idx]
                pre_p = self._pre_activation_window(x_pos)
                vals_p, idx_p = pre_p.topk(self.k, dim=-1)
                z_pos = torch.zeros_like(pre_p)
                z_pos.scatter_(1, idx_p, F.relu(vals_p))
                l_recon = l_recon + self._recon_loss(x_pos, z_pos)
                if eff_alpha > 0.0:
                    h = self.contr_prefix
                    l_contr = l_contr + w_s * _info_nce(
                        z_anchor[:, :h], z_pos[:, :h]
                    )

            x_hat = self.decode(z_anchor)
            l_auxk = self._update_dead_and_auxk(
                x_anchor, x_hat, pre_a, z_anchor,
            )
            total = l_recon + eff_alpha * l_contr + self.auxk_alpha * l_auxk
            return total, x_hat, z_anchor

        if x.ndim == 3 and x.shape[1] == self.T_max:
            pre = self._pre_activation_window(x)
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
            x_hat = self.decode(z)
            l_recon = self._recon_loss(x, z)
            l_auxk = self._update_dead_and_auxk(x, x_hat, pre, z)
            total = l_recon + self.auxk_alpha * l_auxk
            return total, x_hat, z

        raise ValueError(
            f"unexpected input shape {tuple(x.shape)} for T_max={self.T_max}"
        )
