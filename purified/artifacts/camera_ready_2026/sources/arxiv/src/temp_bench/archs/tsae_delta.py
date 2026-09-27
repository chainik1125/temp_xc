"""T-SAE with the contrastive pair distance exposed — support_synthetic Item 2.

The registered T-SAE (:class:`temp_bench.archs.tsae.TSAEPaper`) hardcodes its
one temporal hyperparameter: the contrastive pair is *consecutive* tokens
(``t_offset = randint(0, T_seq-1)``; pair ``(x[:, t], x[:, t+1])``). The
fairness receipt (``task_hunt/support_synthetic/CARD.md`` § 2) sweeps that
knob, so this variant generalizes exactly those two lines to distance
``pair_delta`` = Δ: ``t_offset = randint(0, T_seq-Δ)``; pair
``(x[:, t], x[:, t+Δ])``. Everything else in ``train_step`` is a verbatim port
of the parent (hard rule 3: the registered arch file is untouched; one class,
many YAML entries — ``tsae_d1 / tsae_d2 / tsae_d4 / tsae_d8``).

At ``pair_delta=1`` the ``randint`` call and all arithmetic are identical to
the parent, so the RNG stream, losses, and parameter trajectories match the
registered T-SAE bit-for-bit — enforced by ``tests/test_support_synthetic.py``.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from temp_bench.archs.tsae import TSAEPaper


class TSAEDelta(TSAEPaper):
    """TSAEPaper with the contrastive pair distance ``pair_delta`` exposed."""

    arch_version: str = "1.0.0"

    def __init__(self, *, pair_delta: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.pair_delta = int(pair_delta)
        if self.pair_delta < 1:
            raise ValueError(f"pair_delta must be >= 1, got {pair_delta}.")

    def _pair_offset(self, T_seq: int) -> int:
        """Sample the anchor offset. At Δ=1 this is the parent's exact
        ``randint(0, T_seq-1)`` call — same RNG consumption, same stream."""
        return torch.randint(0, T_seq - self.pair_delta, (1,)).item()

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Verbatim port of ``TSAEPaper.train_step`` with the pair generalized
        from ``(t, t+1)`` to ``(t, t+Δ)``. Keep op-for-op parity with the
        parent — the Δ=1 bitwise contract test depends on it."""
        d = self.pair_delta
        if x.dim() != 3 or x.shape[1] < d + 1:
            raise ValueError(
                f"TSAEDelta train_step expects (B, seq_len>={d + 1}, d_in) "
                f"for pair_delta={d}; got {tuple(x.shape)}."
            )
        B, T_seq, _ = x.shape
        # Random pair offset; same offset across the batch (parent convention).
        t_offset = self._pair_offset(T_seq)
        x_anchor = x[:, t_offset, :]                     # (B, d_in)
        x_temp = x[:, t_offset + d, :]                   # (B, d_in)

        # Encode both — use_threshold=False during training (BatchTopK).
        f_, post_relu = self._encode_per_token(x_anchor)
        f_temp, _ = self._encode_per_token(x_temp)

        # Threshold EMA update (after warmup).
        step = int(self.global_step.item())
        if step > self.threshold_start_step:
            with torch.no_grad():
                active = f_[f_ > 0]
                cur = (
                    active.min().float()
                    if active.numel() > 0
                    else torch.tensor(0.0, device=f_.device)
                )
                if self.threshold.item() < 0:
                    self.threshold.copy_(cur)
                else:
                    self.threshold.copy_(
                        self.threshold_beta * self.threshold
                        + (1 - self.threshold_beta) * cur
                    )

        # ── Matryoshka cumulative reconstruction ──
        W_chunks = torch.split(self.W_dec, list(self.group_sizes), dim=0)
        f_chunks = torch.split(f_, list(self.group_sizes), dim=1)
        f_temp_chunks = torch.split(f_temp, list(self.group_sizes), dim=1)

        x_recon = self.b_dec.unsqueeze(0).expand_as(x_anchor).clone()

        # Group 0: high-level — gets reconstruction + contrastive.
        W0, f0, f0_temp = W_chunks[0], f_chunks[0], f_temp_chunks[0]
        x_recon = x_recon + f0 @ W0
        l2_0 = ((x_anchor - x_recon).pow(2).sum(dim=-1) * self.group_weights[0]).mean()
        total_l2 = l2_0

        # Temporal contrastive (raw dot InfoNCE between high-level latents).
        logits = f0 @ f0_temp.T                                # (B, B)
        labels = torch.arange(logits.shape[0], device=logits.device)
        temp_loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )

        # Subsequent groups: cumulative matryoshka reconstruction.
        for gi in range(1, self.active_groups):
            x_recon = x_recon + f_chunks[gi] @ W_chunks[gi]
            total_l2 = total_l2 + (
                (x_anchor - x_recon).pow(2).sum(dim=-1).mean()
                * self.group_weights[gi]
            )

        # ── AuxK on dead features ──
        with torch.no_grad():
            did_fire = (f_.sum(dim=0) > 0)
            self.num_tokens_since_fired += B
            self.num_tokens_since_fired[did_fire] = 0
        residual_for_auxk = (x_anchor - x_recon).detach()
        auxk_loss = self._auxiliary_loss(residual_for_auxk, post_relu)

        total = (
            total_l2
            + self.auxk_alpha * auxk_loss
            + self.contrastive_alpha * temp_loss
        )

        with torch.no_grad():
            self.global_step += 1
            l0 = (f_ != 0).float().sum(dim=-1).mean()
            dead = int((self.num_tokens_since_fired >= self.dead_feature_threshold_tokens).sum().item())

        return total, {
            "mse": total_l2.detach(),
            "l0": l0.detach(),
            "auxk": auxk_loss.detach(),
            "temp": temp_loss.detach(),
            "dead": dead,
            "threshold": float(self.threshold.item()),
            "z": f_.detach(),
        }
