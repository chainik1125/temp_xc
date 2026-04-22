"""BatchTopK sparsity — pooled-top-k across the (batch, d_sae) pre-activation.

Reference: Bussmann et al. 2024, "BatchTopK Sparse Autoencoders".
At training: flatten the (B, d_sae) pre-activation, keep the top B·k
values across the pool, ReLU them, and zero the rest. This lets each
sample flex its per-sample budget around a mean of k.

At inference: apply a calibrated JumpReLU-style threshold tracked as
an EMA of per-batch cutoff values during training. Since the
per-sample activation-count is no longer exactly k at eval time, the
inference sparsity is approximate, which is the published caveat.

Shared utility for TXCDR / MLC / Matryoshka / MLC-contrastive variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchTopK(nn.Module):
    """Applies BatchTopK to a (B, d_sae) pre-activation tensor.

    Args:
        k: per-sample sparsity budget. Batch-level budget = B * k.
        momentum: EMA momentum for the inference threshold.
    """

    def __init__(self, k: int, momentum: float = 0.99):
        super().__init__()
        self.k = int(k)
        self.momentum = momentum
        # Running threshold used at inference. Initialized to 0 so the
        # first few eval calls — before training has filled it — degrade
        # to a plain ReLU.
        self.register_buffer("threshold", torch.tensor(0.0))

    def forward(self, pre_act: torch.Tensor) -> torch.Tensor:
        """pre_act: (B, d_sae) -> sparse (B, d_sae) ReLU'd."""
        if pre_act.ndim != 2:
            raise ValueError(
                f"BatchTopK expects (B, d_sae); got shape {tuple(pre_act.shape)}"
            )
        B, d = pre_act.shape
        if self.training:
            total_k = B * self.k
            flat = pre_act.reshape(-1)
            if total_k >= flat.numel():
                return F.relu(pre_act)
            top_vals, _ = flat.topk(total_k, sorted=False)
            cutoff = top_vals.min()
            out = torch.where(
                pre_act >= cutoff, F.relu(pre_act), torch.zeros_like(pre_act)
            )
            with torch.no_grad():
                self.threshold.mul_(self.momentum).add_(
                    (1.0 - self.momentum) * cutoff
                )
            return out
        # Eval: threshold-and-ReLU.
        thr = self.threshold
        return torch.where(
            pre_act >= thr, F.relu(pre_act), torch.zeros_like(pre_act)
        )
