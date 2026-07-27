"""txc_btk_pre_subseq_btkonly — the subseq curriculum ISOLATED on the healthy
BatchTopK backbone (tscale candidate menu #4; CARD_SPLIT § 4 lineage).

Motivation (RESULTS.md C1): txc_pro_r1's L1 screen showed the subseq
curriculum alone carries a steep rising T-curve (ablations exonerated
contrastive and matryoshka for both the slope and the low-T collapse),
while its per-sample-TopK training collapses at low T. This arch grafts
ONLY the curriculum onto `txc_batchtopk_pre_btkonly`, whose training does
not collapse:

- Training: per-row contiguous ``t_sample``-of-``T`` position subset;
  BatchTopK pool = the SAMPLED positions only (B·t_sample tokens at
  k_pos/token — the constant per-token budget convention); shared code =
  sum of sampled survivors; reconstruction + AuxK residual on sampled
  positions only. Encoder/decoder slabs keep full per-position meaning.
- Inference: INHERITED UNCHANGED — full T window, JumpReLU threshold
  path. ``consumes='window'`` end-to-end (no eval_consumes needed).
- ``t_sample=None`` → the CARD § 4 ratio rule ``max(1, T//2)``.
  ``t_sample == T`` (and therefore every T=1 cell) is EXACTLY the parent
  arch — the T=1 anchor coincides with the baseline by construction.

``train_step`` is a line-for-line copy of the parent
(`_TXCBatchTopKBTKBase.train_step`) with deviations tagged ``# subseq:``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from temp_bench.archs import telemetry
from temp_bench.archs.btk_only import TXCBatchTopKPreBTKOnly, _neg_frac


def _sample_contiguous_subset(
    T: int, t_sample: int, batch_size: int, device: torch.device,
) -> torch.Tensor:
    """Per-row contiguous t_sample-window inside [0, T) — (B, t_sample)."""
    max_off = T - t_sample + 1
    offs = torch.randint(0, max_off, (batch_size,), device=device)
    rng = torch.arange(t_sample, device=device)
    return offs.unsqueeze(1) + rng.unsqueeze(0)


class TXCBatchTopKPreSubseqBTKOnly(TXCBatchTopKPreBTKOnly):
    """Pre-squash btk-only TXC + training-time subseq curriculum."""

    arch_version = "1.0.0"
    _registry_name = "txc_btk_pre_subseq_btkonly"

    def __init__(self, *, t_sample: int | None = None, **kw):
        super().__init__(**kw)
        self.config.name = self._registry_name
        # subseq: ratio rule default (CARD § 4); explicit int = ablation.
        self.t_sample = int(t_sample) if t_sample is not None else max(1, self._T // 2)
        assert 1 <= self.t_sample <= self._T

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"{type(self).__name__}.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, T, _d_in = x.shape

        # subseq: t_sample == T degenerates to the parent exactly.
        if self.t_sample == T:
            return super().train_step(x)

        # subseq: contiguous per-row subset; posts computed at full T so
        # every position keeps its own W_enc slab, then the POOL is
        # restricted to the sampled positions (slab-aligned gather).
        sample_idx = _sample_contiguous_subset(T, self.t_sample, B, x.device)
        post_full = self._compute_post(x)                     # (B, T, d_sae)
        gi_s = sample_idx.unsqueeze(-1).expand(-1, -1, self._d_sae)
        post_S = post_full.gather(1, gi_s)                    # (B, t, d_sae)
        gated = self._batchtopk(post_S)                       # pool B·t_sample
        z_shared = gated.sum(dim=1)                           # subseq: _to_shared on sampled axis

        # JumpReLU threshold EMA (min surviving activation) — btk-only flag
        # logic verbatim; statistics come from the sampled pool (same
        # per-token budget; eval-side G1 bands verify empirically).
        step = int(self.global_step.item())
        if step > self.threshold_start_step:
            with torch.no_grad():
                active = gated[gated != 0]
                cur = (
                    active.min().float()
                    if active.numel() > 0
                    else torch.tensor(0.0, device=gated.device)
                )
                if not bool(self.threshold_set.item()):
                    self.threshold.copy_(cur)
                    self.threshold_set.fill_(1)
                else:
                    self.threshold.copy_(
                        self.threshold_beta * self.threshold
                        + (1 - self.threshold_beta) * cur
                    )

        # Reconstruction: shared code decodes all T positions; loss on the
        # SAMPLED positions only (subseq semantics — unobserved positions
        # are not scored; their decoder slabs get no recon gradient).
        x_hat = torch.einsum("bs,std->btd", z_shared, self.W_dec) + self.b_dec
        gi_d = sample_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        x_S = x.gather(1, gi_d)
        x_hat_S = x_hat.gather(1, gi_d)
        l_recon = (x_S - x_hat_S).pow(2).sum(dim=-1).mean()

        # Dead-feature tracking on the shared code.
        with torch.no_grad():
            active_feat = (z_shared != 0).any(dim=0)          # btk-only: != 0
            self.num_tokens_since_fired += B * self.t_sample  # subseq: sampled tokens
            self.num_tokens_since_fired[active_feat] = 0
            dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
            n_dead = int(dead_mask.sum().item())
            if telemetry.due(step):
                nz = gated[gated != 0]
                telemetry.maybe_log(
                    self, step=step, n_dead=n_dead,
                    batch_l0=float(nz.numel()) / B,
                    boundary_min_pre=(float(nz.min().item())
                                      if nz.numel() else 0.0))

        # AuxK on dead features, in shared-code space — revival stays on
        # ReLU'd pre-acts (btk-only conv § 4). subseq: the aux code sums
        # encoder contributions over SAMPLED positions only (zero-masked
        # input) and its residual target is the sampled-position residual.
        if n_dead > 0:
            k_aux = min(self.aux_k, n_dead)
            mask = torch.zeros(B, T, device=x.device, dtype=x.dtype)
            mask.scatter_(1, sample_idx, 1.0)
            x_masked = x * mask.unsqueeze(-1)
            pre_sq = F.relu(self._squashed_preact(x_masked))
            auxk_pre = pre_sq.masked_fill(~dead_mask.unsqueeze(0), 0.0)
            vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
            aux_buf = torch.zeros_like(pre_sq).scatter_(-1, idx_a, vals_a)
            aux_decode = torch.einsum("bs,std->btd", aux_buf, self.W_dec)
            residual = (x_S - x_hat_S).detach()
            aux_S = aux_decode.gather(1, gi_d)
            l2_a = (residual - aux_S).pow(2).sum(dim=-1).mean()
            mu = residual.mean(dim=(0, 1), keepdim=True)
            denom = (residual - mu).pow(2).sum(dim=-1).mean()
            l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        else:
            l_auxk = torch.zeros((), device=x.device, dtype=x.dtype)

        loss = l_recon + self.auxk_alpha * l_auxk

        with torch.no_grad():
            self.global_step += 1
            l0 = (z_shared != 0).float().sum(dim=-1).mean()

        return {
            "loss": loss,
            "mse": l_recon.detach(),
            "l0": l0.detach(),
            "auxk": l_auxk.detach(),
            "dead": torch.tensor(float(n_dead)),
            "threshold": self.threshold.detach().clone(),
            "neg_frac": _neg_frac(gated),
        }
