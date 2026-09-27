"""Window-code adapters for the stacked (per-position) architectures.

The real-task evaluators (probing, rlhf) expect a window architecture to
emit one code per window, ``(B, T, d_in) -> (B, d_sae)``. The stacked
archs natively emit per-position codes ``(B, T, d_sae)``, which the
paper's eval protocol reduces by max-pooling absolute activations over
positions (App. A eval protocol; identical to the reduction
``evals/em.py::encode_and_pool`` already applies). These subclasses bake
that reduction into ``encode`` so every evaluator sees the same window
code without touching ``temp_bench/core`` or the evaluators.

Training is unchanged: ``train_step`` still operates on per-position
codes with the parent's loss; only the eval-facing ``encode`` pools.
Per-position codes stay reachable via :meth:`encode_per_position` for
feature mining and steering.
"""

from __future__ import annotations

import torch

from temp_bench.archs.btk_only import StackedBatchTopKBTKOnly
from temp_bench.archs.stacked_sae import StackedSAE


def _pool_window(z: torch.Tensor) -> torch.Tensor:
    """(B, T, d_sae) -> (B, d_sae): per-feature value of max |activation|.

    Keeps the sign of the selected activation so signed codes (btk-only)
    survive; for non-negative TopK codes this equals a plain amax.
    """
    idx = z.abs().argmax(dim=1, keepdim=True)          # (B, 1, d_sae)
    return z.gather(1, idx).squeeze(1)


class StackedSAEPooled(StackedSAE):
    """StackedSAE emitting one pooled code per window.

    ``consumes = 'window'``: the trainer feeds ``(B, T, d_in)`` windows,
    which the parent ``train_step`` handles as the seq_len == T case of
    its window sampling.
    """

    arch_version: str = "2.1.0"
    consumes: str = "window"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.config.name = "stacked_sae_pooled"

    def encode_per_position(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, T, d_sae) native per-position codes."""
        return super().encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, d_sae) max-|act| pooled window code."""
        return _pool_window(self.encode_per_position(x))

    def train_step(self, x: torch.Tensor):
        # Same loss as the parent, but routed through the per-position
        # encode: the parent's train_step calls ``self.encode``, which
        # here is pooled and cannot be decoded. A WindowBuffer batch is
        # (B, T, d_in), i.e. the parent's seq_len == T identity-window
        # case, so no window sampling is needed.
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"StackedSAEPooled.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        z = self.encode_per_position(x)
        x_hat = self.decode(z)
        mse = (x_hat - x).pow(2).sum(dim=-1).mean()
        l0 = (z != 0).float().sum(dim=-1).mean()
        return mse, {"mse": mse.detach(), "l0": l0.detach(), "z": z.detach()}


class StackedBTKOnlyPooled(StackedBatchTopKBTKOnly):
    """btk-only stacked twin emitting one pooled code per window
    (ACTMIX-harmonised backbone for the campaign cell tables)."""

    arch_version: str = "1.2.0"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.config.name = "stacked_btkonly_pooled"

    def encode_per_position(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, T, d_sae) native per-position codes."""
        return super().encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, d_sae) max-|act| pooled window code."""
        return _pool_window(self.encode_per_position(x))
