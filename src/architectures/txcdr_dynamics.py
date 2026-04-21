"""TXCDR with recurrent sparse latent — features evolve via a learned
dynamical system across a T-window.

Phase 5.7 autoresearch candidate A8, brief.md §4. Rather than
computing latents independently at each position, latents CARRY
STATE across positions via a gated recurrence:

    z_{t+1} = TopK(γ ⊙ z_t + W_enc · x_{t+1} + b_enc)

where γ ∈ (0, 1)^{d_sae} is a per-feature learnable decay (sigmoid
parameterised), W_enc ∈ R^{d_in × d_sae} is shared across time, and
TopK is applied to the gated sum at each step. Initial state:

    z_0 = TopK(W_enc · x_0 + b_enc)

Features that fire at one position persist into later positions
(weighted by γ), so the same feature can be "carried" across the
window. Probing uses the last-position state z_{T-1} which matches
our existing `last_position` aggregation.

Decoder is shared across time: x_hat_t = z_t · W_dec + b_dec. The
time-variation lives entirely in the z_t sequence; decoder is a
plain SAE dictionary. Reconstruction loss sums MSE over all T
positions.

Family:
  - In spirit, closest to the "state-space" / "gated SAE" variants
    discussed in brief.md §4.
  - Orthogonal to A1/A5/A4 decoder/encoder constraints: here the
    constraint is on the latent TRAJECTORY, not on W_dec or W_enc.
  - Aligned with the Tier-1 winning signal (adjacent latents should
    agree): contrastive (A2/A3) does this as a loss; dynamics does
    it as an architectural structure.

Param count at d_in=2304, d_sae=18432, T=5:
    W_enc:  2304 × 18432 ≈ 42 M (vs vanilla TXCDR's 212 M — T× smaller)
    W_dec:  18432 × 2304 ≈ 42 M (vs vanilla's 212 M — T× smaller)
    gate:   18432         ≈ 18 K
    total ≈ 84 M (vs vanilla TXCDR 425 M)

Pre-register: this variant is substantially SMALLER in params than
vanilla TXCDR, which is a potential confound. If it underperforms,
capacity may be the cause rather than the dynamics idea being wrong.
A fair capacity comparison would need a bigger d_sae or T-dependent
encoder; deferred to Part B tuning if A8 signals a direction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TXCDRDynamics(nn.Module):
    """Recurrent sparse latent over a T-token window.

    Args:
        d_in, d_sae, T, k: same as vanilla TemporalCrosscoder, except:
          - k applies per-position (not window-level) since we do a
            separate TopK at each position.
        gate_init: initial value for the raw gate parameter (pre-sigmoid).
            `0.0` → sigmoid = 0.5 (mid-range decay at init).
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int | None,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k

        self.W_enc = nn.Parameter(
            torch.randn(d_in, d_sae) * (1.0 / d_in**0.5)
        )
        self.W_dec = nn.Parameter(
            torch.randn(d_sae, d_in) * (1.0 / d_sae**0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        # Raw gate parameter; sigmoid at forward to stay in (0, 1).
        self.gate_raw = nn.Parameter(torch.full((d_sae,), gate_init))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def _topk(self, pre: torch.Tensor) -> torch.Tensor:
        if self.k is None:
            return F.relu(pre)
        v, i = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, i, F.relu(v))
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) -> z_last: (B, d_sae).

        Returns the last-position latent z_{T-1} (matches our
        `last_position` probing convention). For probing under
        full_window / mean_pool aggregations, use `encode_sequence`
        to get all T latents.
        """
        z_seq = self.encode_sequence(x)
        return z_seq[:, -1, :]

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) -> z_seq: (B, T, d_sae).

        z_0 = TopK(W_enc · x_0 + b_enc)
        z_t = TopK(γ ⊙ z_{t-1} + W_enc · x_t + b_enc)  for t >= 1
        """
        gate = torch.sigmoid(self.gate_raw)            # (d_sae,) in (0, 1)
        B, T, d_in = x.shape
        # Per-position pre-activation (shared encoder): (B, T, d_sae).
        pre_input = torch.einsum("btd,ds->bts", x, self.W_enc) + self.b_enc
        z_prev = self._topk(pre_input[:, 0, :])        # (B, d_sae)
        outs = [z_prev]
        for t in range(1, T):
            pre_t = gate * z_prev + pre_input[:, t, :]
            z_t = self._topk(pre_t)
            outs.append(z_t)
            z_prev = z_t
        return torch.stack(outs, dim=1)                 # (B, T, d_sae)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, d_sae) OR (B, d_sae) -> x_hat: (B, T, d_in) / (B, d_in)."""
        if z.ndim == 3:
            return torch.einsum("bts,sd->btd", z, self.W_dec) + self.b_dec
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        """x: (B, T, d_in) -> (recon_loss, x_hat_full, z_last).

        Loss = sum over T of MSE(x_hat_t, x_t), averaged per batch.
        """
        z_seq = self.encode_sequence(x)
        x_hat = self.decode(z_seq)                      # (B, T, d_in)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, z_seq[:, -1, :]
