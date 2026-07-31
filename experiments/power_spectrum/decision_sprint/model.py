"""A strong learned consumer of frozen shared-SAE trajectories."""

from __future__ import annotations

import torch
from torch import nn

from experiments.power_spectrum.trajectory_bottleneck.model import (
    TrajectoryBottleneck,
)


class FlexibleTrajectoryBottleneck(TrajectoryBottleneck):
    """Trajectory bottleneck with a low-rank position-specific decoder.

    The inherited path decodes output feature ``f`` through frozen shared-SAE
    direction ``f`` with a learned temporal profile.  This extension adds a
    low-rank residual whose output directions may differ at every position.
    It removes the strongest decoder-restriction objection while retaining
    one top-k code per window and substantially fewer temporal parameters than
    a full TXC.
    """

    def __init__(self, *, decoder_rank: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        if decoder_rank < 0:
            raise ValueError("decoder_rank must be nonnegative")
        self.decoder_rank = int(decoder_rank)
        if self.decoder_rank:
            self.decoder_cross_in = nn.Parameter(
                torch.empty(
                    self.d_sae,
                    self.decoder_rank,
                    dtype=self.base_decoder.dtype,
                )
            )
            self.decoder_cross_out = nn.Parameter(
                torch.zeros(
                    self.shape.window,
                    self.decoder_rank,
                    self.shape.d_in,
                    dtype=self.base_decoder.dtype,
                )
            )
            nn.init.normal_(
                self.decoder_cross_in, std=self.decoder_rank**-0.5
            )
        else:
            self.register_parameter("decoder_cross_in", None)
            self.register_parameter("decoder_cross_out", None)

    def decode_sparse(
        self,
        active_values: torch.Tensor,
        active_indices: torch.Tensor,
        *,
        add_bias: bool,
    ) -> torch.Tensor:
        reconstruction = super().decode_sparse(
            active_values, active_indices, add_bias=add_bias
        )
        if self.decoder_rank:
            selected = self.decoder_cross_in[active_indices.long()]
            hidden = (selected * active_values.unsqueeze(-1)).sum(dim=1)
            residual = torch.einsum(
                "br,trd->btd", hidden, self.decoder_cross_out
            )
            reconstruction = reconstruction + residual
        return reconstruction
