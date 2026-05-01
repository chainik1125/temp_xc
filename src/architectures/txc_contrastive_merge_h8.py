"""MYSTERY: Contrastive-merge TXC — z captures CHANGE, not co-occurrence.

Canonical TXC: z[s] = sum_t (x[t] @ W_enc[t,:,s]) (additive — captures
co-occurrence of features across the window).

THIS arch: z[s] = (x[T-1] @ W_enc[T-1,:,s]) - (x[0] @ W_enc[0,:,s])
(contrastive — captures CHANGE from start to end of the window).

For T=2: z = enc(x[1]) - enc(x[0]). Features fire on transitions.

For steering, contrastive features represent "concept becomes active during
the window". Steering writes a "transition into concept" signal — useful
for triggering shifts in style/sentiment mid-generation.

Note: z can be negative (since it's a difference). ReLU after still applies,
so only positive transitions are kept (concept BECOMES active, not LEAVES).

Implementation: subclass TXCBareMultiDistanceContrastiveAntidead, override
_pre_activation. Only T=2 makes intuitive sense; for T>2 we use end-vs-start.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures.txc_bare_multidistance_contrastive_antidead import (
    TXCBareMultiDistanceContrastiveAntidead,
)


class TXCContrastiveMergeH8(TXCBareMultiDistanceContrastiveAntidead):
    """Contrastive-merge encoder; H8 stack everywhere else.

    Encoder forward:
        pre[b, s] = (x[b, T-1, :] @ W_enc[T-1, :, s])
                  - (x[b, 0,   :] @ W_enc[0,   :, s])
                  + b_enc[s]

    Captures end-vs-start CHANGE in feature space.
    """

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        # End: x[:, T-1, :] @ W_enc[T-1, :, :]
        end_pre = torch.einsum("bd,ds->bs", x[:, -1, :], self.W_enc[-1])
        # Start: x[:, 0, :] @ W_enc[0, :, :]
        start_pre = torch.einsum("bd,ds->bs", x[:, 0, :], self.W_enc[0])
        # Difference
        return end_pre - start_pre + self.b_enc
