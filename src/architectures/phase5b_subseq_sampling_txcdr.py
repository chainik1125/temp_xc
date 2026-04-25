"""Phase 5B candidate B: subsequence-sampled TXC.

Train time: encoder W_enc has T_max position slabs. Each step samples
a subset S of size t_sample positions, and pre-activation only sums
over S. The decoder reconstructs ONLY the sampled positions
(reconstructing un-observed positions from z would force the encoder
to be a generative-model — out of scope).

Variants:
- B1 contiguous: random contiguous T-window of size t_sample inside T_max.
- B2 non-contiguous: random subset of size t_sample from [0, T_max).
- B3 same as B2 but at larger T_max.

Probe time: encode the FULL T_max-window (sum over all positions). The
training-time random subsetting is a regulariser intended to make the
latent "subset-redundant"; probe-time evaluation uses the full encoder.

Backbone variants:
- track2 base: bare anti-dead stack (single-window training)
- h8 base: anti-dead + matryoshka + multi-distance contrastive

Per-position W_enc/W_dec is preserved (positions retain meaning).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.txc_bare_antidead import TXCBareAntidead
from src.architectures.txc_bare_multidistance_contrastive_antidead import (
    TXCBareMultiDistanceContrastiveAntidead,
    _info_nce,
)


def _sample_subset_indices(T_max: int, t_sample: int, batch_size: int,
                            contiguous: bool, device: torch.device) -> torch.Tensor:
    """Return (batch_size, t_sample) integer indices into [0, T_max)."""
    if contiguous:
        max_off = T_max - t_sample + 1
        offs = torch.randint(0, max_off, (batch_size,), device=device)
        rng = torch.arange(t_sample, device=device)
        return offs.unsqueeze(1) + rng.unsqueeze(0)               # (B, t_sample)
    # non-contiguous: per-row random permutation, take first t_sample
    keys = torch.rand(batch_size, T_max, device=device)
    _, idx = keys.topk(t_sample, dim=-1, largest=True, sorted=True)
    idx, _ = idx.sort(dim=-1)
    return idx


def _zero_decoder_at_unsampled(decoded_full: torch.Tensor,
                                sample_idx: torch.Tensor) -> torch.Tensor:
    """Zero out decoded values at positions not in `sample_idx`.
    decoded_full: (B, T_max, d). sample_idx: (B, t_sample).
    Returns (B, T_max, d) with non-sampled positions zeroed (so MSE on
    only the sampled positions when summed against zero-padded x).
    """
    B, T_max, d = decoded_full.shape
    mask = torch.zeros(B, T_max, device=decoded_full.device, dtype=torch.bool)
    mask.scatter_(1, sample_idx, True)
    return decoded_full * mask.unsqueeze(-1).float()


class SubseqTXCBareAntidead(TXCBareAntidead):
    """Track 2 + subsequence sampling.

    Encoder/decoder shape unchanged (per-position W_enc/W_dec at T_max).
    Training-time sample of t_sample positions; loss only on those.
    """

    def __init__(self, d_in: int, d_sae: int, T_max: int, k: int,
                 t_sample: int, contiguous: bool = False, **kw):
        super().__init__(d_in, d_sae, T_max, k, **kw)
        self.T_max = int(T_max)
        self.t_sample = int(t_sample)
        self.contiguous = bool(contiguous)
        assert 1 <= t_sample <= T_max

    def _pre_activation_sampled(self, x: torch.Tensor,
                                 sample_idx: torch.Tensor) -> torch.Tensor:
        """Sum encoder contributions over sampled positions only.

        x: (B, T_max, d_in). sample_idx: (B, t_sample) in [0, T_max).
        Returns (B, d_sae).
        """
        B, T_max, d = x.shape
        # Gather per-row positions
        # x_S: (B, t_sample, d_in)
        gather_idx = sample_idx.unsqueeze(-1).expand(-1, -1, d)
        x_S = x.gather(1, gather_idx)
        # W_enc_S: (B, t_sample, d_in, d_sae) — too big. Use a per-row loop or einsum trick.
        # Trick: contributing sum_{t in S_b} x_S[b, t, :] @ W_enc[sample_idx[b, t]]
        # = einsum over a (B, t_sample, d_in, d_sae) gathered weight tensor.
        # That tensor is t_sample * d_in * d_sae * 4 = at t_sample=5, d_in=2304, d_sae=18432 → 6 GB per batch. Too big.
        # Alternative: loop over t_sample in {1..t_sample}, gather W_enc per-row and accumulate.
        # In practice t_sample is small (5) so a small Python loop is fine.
        out = torch.zeros(B, self.d_sae, device=x.device, dtype=x.dtype)
        for j in range(self.t_sample):
            pos_j = sample_idx[:, j]                           # (B,)
            w_j = self.W_enc[pos_j]                            # (B, d_in, d_sae)
            x_j = x_S[:, j, :]                                 # (B, d_in)
            out = out + torch.einsum("bd,bds->bs", x_j, w_j)
        out = out + self.b_enc
        return out

    def encode_full(self, x: torch.Tensor) -> torch.Tensor:
        """Probe-time encoding using ALL T_max positions (no sampling)."""
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Same as encode_full — used by probe pipeline."""
        return self.encode_full(x)

    def _recon_loss_sampled(self, x: torch.Tensor, z: torch.Tensor,
                              sample_idx: torch.Tensor) -> torch.Tensor:
        """MSE only on sampled positions.
        x: (B, T_max, d). z: (B, d_sae). sample_idx: (B, t_sample).
        """
        # Decode full (B, T_max, d) — cheap on T_max≤30
        x_hat = self.decode(z)
        B, T_max, d = x.shape
        gather_idx = sample_idx.unsqueeze(-1).expand(-1, -1, d)
        x_S = x.gather(1, gather_idx)                          # (B, t_sample, d)
        x_hat_S = x_hat.gather(1, gather_idx)
        return (x_S - x_hat_S).pow(2).sum(dim=-1).mean()

    def forward(self, x: torch.Tensor):
        """x: (B, T_max, d). Trains via subsequence sample."""
        B = x.shape[0]
        sample_idx = _sample_subset_indices(
            self.T_max, self.t_sample, B, self.contiguous, x.device,
        )
        # Sampled pre-activation
        pre = self._pre_activation_sampled(x, sample_idx)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        x_hat = self.decode(z)
        l_recon = self._recon_loss_sampled(x, z, sample_idx)

        # AuxK on sampled positions' residual
        # Re-use parent's auxk machinery; supplies sampled-loss residual.
        l_auxk = self._update_dead_and_auxk_sampled(x, x_hat, pre, z, sample_idx)
        total = l_recon + self.auxk_alpha * l_auxk
        return total, x_hat, z

    def _update_dead_and_auxk_sampled(
        self, x: torch.Tensor, x_hat: torch.Tensor,
        pre: torch.Tensor, z: torch.Tensor,
        sample_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Variant of TXCBareAntidead's auxk that uses sampled residual."""
        active_mask = (z > 0).any(dim=0)
        # token count = B * t_sample (not T_max — only sampled positions
        # contribute to z this step)
        n_tokens = x.shape[0] * self.t_sample
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
        aux_decode = torch.einsum("bs,std->btd", aux_buf, self.W_dec)
        # Residual at sampled positions only
        B, T_max, d = x.shape
        gather_idx = sample_idx.unsqueeze(-1).expand(-1, -1, d)
        x_S = x.gather(1, gather_idx)
        x_hat_S = x_hat.detach().gather(1, gather_idx)
        aux_S = aux_decode.gather(1, gather_idx)
        residual = x_S - x_hat_S
        l2_a = (residual - aux_S).pow(2).sum(dim=-1).mean()
        mu = residual.mean(dim=(0, 1), keepdim=True)
        denom = (residual - mu).pow(2).sum(dim=-1).mean()
        l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        self.last_auxk_loss.fill_(float(l_auxk.detach()))
        return l_auxk


class SubseqH8(TXCBareMultiDistanceContrastiveAntidead):
    """H8 + subsequence sampling. Anchor + K positives at the H8 shifts.

    For each sample step:
      - Sample a single subset S of size t_sample from [0, T_max). Apply
        to ALL windows (anchor + each positive shift).
      - Compute z_anchor / z_pos via subset-summed encoder.
      - Recon loss on sampled positions of each window.
      - InfoNCE on z's H prefix as in H8.
    """

    def __init__(self, d_in: int, d_sae: int, T_max: int, k: int,
                 t_sample: int,
                 shifts: tuple[int, ...] | None = None,
                 weights: tuple[float, ...] | None = None,
                 contiguous: bool = False,
                 matryoshka_h_size: int | None = None,
                 alpha: float = 1.0,
                 **kw):
        super().__init__(
            d_in, d_sae, T_max, k,
            shifts=shifts, weights=weights,
            matryoshka_h_size=matryoshka_h_size, alpha=alpha,
            **kw,
        )
        self.T_max = int(T_max)
        self.t_sample = int(t_sample)
        self.contiguous = bool(contiguous)

    def _pre_activation_sampled(self, x: torch.Tensor,
                                 sample_idx: torch.Tensor) -> torch.Tensor:
        """Same as SubseqTXCBareAntidead's; reproduced for clarity."""
        B, T_max, d = x.shape
        out = torch.zeros(B, self.d_sae, device=x.device, dtype=x.dtype)
        for j in range(self.t_sample):
            pos_j = sample_idx[:, j]
            w_j = self.W_enc[pos_j]
            x_j = x.gather(1, sample_idx[:, j:j+1].unsqueeze(-1).expand(-1, 1, d))
            x_j = x_j.squeeze(1)
            out = out + torch.einsum("bd,bds->bs", x_j, w_j)
        out = out + self.b_enc
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Probe-time: full T_max encode (no sampling)."""
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z

    def _recon_sampled(self, x: torch.Tensor, z: torch.Tensor,
                        sample_idx: torch.Tensor) -> torch.Tensor:
        """Matryoshka recon (H + Full) on sampled positions only."""
        x_hat_full = self.decode(z)
        x_hat_h = self._decode_prefix(z, self.matryoshka_h_size) if self.matryoshka_h_size else None
        B, T_max, d = x.shape
        gi = sample_idx.unsqueeze(-1).expand(-1, -1, d)
        x_S = x.gather(1, gi)
        x_hat_full_S = x_hat_full.gather(1, gi)
        l_full = (x_S - x_hat_full_S).pow(2).sum(dim=-1).mean()
        if x_hat_h is None:
            return l_full
        x_hat_h_S = x_hat_h.gather(1, gi)
        l_h = (x_S - x_hat_h_S).pow(2).sum(dim=-1).mean()
        return l_h + l_full

    def forward(self, x: torch.Tensor, alpha: float | None = None):
        """Accepts (B, 1+K, T_max, d) where K = len(shifts).
        Sampled-subset encoder. Recon loss on sampled positions.
        """
        eff_alpha = self.alpha if alpha is None else alpha

        if x.ndim == 4 and x.shape[1] >= 2:
            K = x.shape[1] - 1
            assert K == len(self.shifts), (
                f"expected {1+len(self.shifts)} window slots, got {x.shape[1]}"
            )

            B = x.shape[0]
            # Single shared subset S applied to all windows
            sample_idx = _sample_subset_indices(
                self.T_max, self.t_sample, B, self.contiguous, x.device,
            )

            x_anchor = x[:, 0]
            pre_a = self._pre_activation_sampled(x_anchor, sample_idx)
            vals_a, idx_a = pre_a.topk(self.k, dim=-1)
            z_anchor = torch.zeros_like(pre_a)
            z_anchor.scatter_(1, idx_a, F.relu(vals_a))

            l_recon = self._recon_sampled(x_anchor, z_anchor, sample_idx)

            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            for k_idx, w_s in enumerate(self.loss_weights):
                x_pos = x[:, 1 + k_idx]
                pre_p = self._pre_activation_sampled(x_pos, sample_idx)
                vals_p, idx_p = pre_p.topk(self.k, dim=-1)
                z_pos = torch.zeros_like(pre_p)
                z_pos.scatter_(1, idx_p, F.relu(vals_p))
                l_recon = l_recon + self._recon_sampled(x_pos, z_pos, sample_idx)
                if eff_alpha > 0.0:
                    h = self.contr_prefix
                    l_contr = l_contr + w_s * _info_nce(z_anchor[:, :h], z_pos[:, :h])

            x_hat = self.decode(z_anchor)
            # AuxK using sampled residual on anchor
            l_auxk = self._update_dead_and_auxk_sampled(
                x_anchor, x_hat, pre_a, z_anchor, sample_idx,
            )
            total = l_recon + eff_alpha * l_contr + self.auxk_alpha * l_auxk
            return total, x_hat, z_anchor

        # Single-window fallback
        if x.ndim == 3:
            B = x.shape[0]
            sample_idx = _sample_subset_indices(
                self.T_max, self.t_sample, B, self.contiguous, x.device,
            )
            pre = self._pre_activation_sampled(x, sample_idx)
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
            x_hat = self.decode(z)
            l_recon = self._recon_sampled(x, z, sample_idx)
            l_auxk = self._update_dead_and_auxk_sampled(x, x_hat, pre, z, sample_idx)
            total = l_recon + self.auxk_alpha * l_auxk
            return total, x_hat, z

        raise ValueError(f"unexpected input shape {tuple(x.shape)}")

    def _update_dead_and_auxk_sampled(
        self, x, x_hat, pre, z, sample_idx,
    ):
        """Auxk variant for sampled training (mirrors SubseqTXCBareAntidead)."""
        active_mask = (z > 0).any(dim=0)
        n_tokens = x.shape[0] * self.t_sample
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
        aux_decode = torch.einsum("bs,std->btd", aux_buf, self.W_dec)
        B, T_max, d = x.shape
        gi = sample_idx.unsqueeze(-1).expand(-1, -1, d)
        x_S = x.gather(1, gi)
        x_hat_S = x_hat.detach().gather(1, gi)
        aux_S = aux_decode.gather(1, gi)
        residual = x_S - x_hat_S
        l2_a = (residual - aux_S).pow(2).sum(dim=-1).mean()
        mu = residual.mean(dim=(0, 1), keepdim=True)
        denom = (residual - mu).pow(2).sum(dim=-1).mean()
        l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        self.last_auxk_loss.fill_(float(l_auxk.detach()))
        return l_auxk
