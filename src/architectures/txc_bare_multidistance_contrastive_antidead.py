"""Part B H8: bare TXC + anti-dead + matryoshka + **MULTI-DISTANCE** InfoNCE.

Addresses: "have we tried extending the contrastive window length beyond 2?"

agentic_txc_02 and Phase 6.2 C2/C3 all contrast windows at shift-1 only
(x_prev = x[t-T:t], x_cur = x[t-T+1:t+1]). This arch contrasts at
multiple shift distances simultaneously:

- shift=1: adjacent-window pairs (local temporal consistency)
- shift=⌊T/4⌋: mid-range shift
- shift=⌊T/2⌋: half-window shift (long-range consistency)

Rationale: at T=5 with shift=1, the anchor and positive share 4/5 tokens
— contrastive signal is "near-redundant windows should share features".
At shift=⌊T/2⌋=2 they share 3/5 tokens; at T=10 shift=5 they share 5/10.
Larger shifts push features to be invariant to mid-range temporal
translation — potentially valuable for T-scaling.

InfoNCE is computed per distance and weighted by 1 / (1+distance)
(inverse-distance decay — handover H4 recipe).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.txc_bare_matryoshka_contrastive_antidead import (
    TXCBareMatryoshkaContrastiveAntidead, _info_nce,
)


def make_multidistance_pair_gen_gpu(buf, T: int, shifts: list[int]):
    """Yield a (B, 1+len(shifts), T, d) tensor: [anchor, pos_shift_s1, pos_shift_s2, ...].

    For each sample, anchor window starts at offset `off`, positives at
    `off + s` for each s in shifts. All windows must fit in buf's seq
    length, so off ∈ [0, L - T - max(shifts)).
    """
    import torch as _torch
    N, L, d = buf.shape
    max_shift = max(shifts)
    assert L >= T + max_shift, f"need L>={T+max_shift}; got L={L}"

    def gen(batch_size: int) -> _torch.Tensor:
        seq = _torch.randint(0, N, (batch_size,), device=buf.device)
        off = _torch.randint(0, L - T - max_shift, (batch_size,), device=buf.device)
        rng = _torch.arange(T, device=buf.device)
        outs = []
        for s in [0] + list(shifts):
            pos = (off + s).unsqueeze(1) + rng.unsqueeze(0)  # (B, T)
            w = buf[seq.unsqueeze(1).expand(-1, T), pos].float()
            outs.append(w)
        return _torch.stack(outs, dim=1)  # (B, 1+K, T, d)

    return gen


class TXCBareMultiDistanceContrastiveAntidead(TXCBareMatryoshkaContrastiveAntidead):
    """Bare TXC + anti-dead + matryoshka + multi-distance InfoNCE.

    Accepts forward input (B, 1+K, T, d) where dim 1 is [anchor, pos_1, pos_2, ...].
    Loss = matryoshka_recon(anchor) + Σ_k (1/(1+shifts[k])) · InfoNCE(anchor, pos_k) + AuxK.

    Args:
        d_in, d_sae, T, k: same as parent.
        shifts: tuple of shift distances. Default ⌊T/4⌋, ⌊T/2⌋ (H4 recipe).
            The shift-1 pair is implicit — anchor itself is at offset 0, so
            shifts should NOT include 1 (caller can add it explicitly).
        weights: per-distance loss weights. Default inverse-distance.
        matryoshka_h_size, alpha, aux_k, ...: same as parent.
    """

    def __init__(
        self, d_in: int, d_sae: int, T: int, k: int,
        shifts: tuple[int, ...] | None = None,
        weights: tuple[float, ...] | None = None,
        matryoshka_h_size: int | None = None,
        alpha: float = 1.0,
        contr_prefix: int | None = None,
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
        auxk_alpha: float = 1.0 / 32.0,
    ):
        super().__init__(
            d_in, d_sae, T, k,
            matryoshka_h_size=matryoshka_h_size,
            alpha=alpha,
            contr_prefix=contr_prefix,
            aux_k=aux_k,
            dead_threshold_tokens=dead_threshold_tokens,
            auxk_alpha=auxk_alpha,
        )
        if shifts is None:
            shifts = tuple(s for s in (1, max(1, T // 4), max(1, T // 2))
                           if s <= T - 1 and s >= 1)
            # dedupe
            shifts = tuple(sorted(set(shifts)))
        self.shifts = tuple(shifts)
        if weights is None:
            # Inverse-distance weighting: w_s = 1 / (1 + s)
            weights = tuple(1.0 / (1.0 + s) for s in shifts)
        self.loss_weights = tuple(weights)
        assert len(self.loss_weights) == len(self.shifts)

    def forward(self, x: torch.Tensor, alpha: float | None = None):
        """Accepts:
        - (B, T, d): single-window recon + AuxK (no contrastive)
        - (B, 2, T, d): parent's (anchor, shift-1) pair path (single-distance)
        - (B, 1+K, T, d): multi-distance [anchor, pos_s1, pos_s2, ...]
        """
        eff_alpha = self.alpha if alpha is None else alpha

        if x.ndim == 3:
            return super().forward(x, alpha=alpha)

        if x.ndim == 4 and x.shape[1] == 2:
            # Fall back to parent (single-distance, shift-1).
            return super().forward(x, alpha=alpha)

        if x.ndim == 4 and x.shape[1] > 2:
            # Multi-distance path. dim 1 = [anchor, pos_s1, pos_s2, ...]
            K = x.shape[1] - 1
            if K != len(self.shifts):
                raise ValueError(
                    f"expected {1+len(self.shifts)} window slots for "
                    f"shifts={self.shifts}, got {x.shape[1]}"
                )

            x_anchor = x[:, 0]
            pre_a = self._pre_activation(x_anchor)
            vals_a, idx_a = pre_a.topk(self.k, dim=-1)
            z_anchor = torch.zeros_like(pre_a)
            z_anchor.scatter_(1, idx_a, F.relu(vals_a))

            l_recon = self._recon_loss(x_anchor, z_anchor)

            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            for k_idx, w_s in enumerate(self.loss_weights):
                x_pos = x[:, 1 + k_idx]
                pre_p = self._pre_activation(x_pos)
                vals_p, idx_p = pre_p.topk(self.k, dim=-1)
                z_pos = torch.zeros_like(pre_p)
                z_pos.scatter_(1, idx_p, F.relu(vals_p))
                # matryoshka recon on each positive too (full loss budget)
                l_recon = l_recon + self._recon_loss(x_pos, z_pos)
                if eff_alpha > 0.0:
                    h = self.contr_prefix
                    l_contr = l_contr + w_s * _info_nce(
                        z_anchor[:, :h], z_pos[:, :h]
                    )

            x_hat = self.decode(z_anchor)
            l_auxk = self._update_dead_and_auxk(x_anchor, x_hat, pre_a, z_anchor)

            total = l_recon + eff_alpha * l_contr + self.auxk_alpha * l_auxk
            return total, x_hat, z_anchor

        raise ValueError(
            f"Expected (B, T, d), (B, 2, T, d) or (B, 1+K, T, d); got {tuple(x.shape)}"
        )
