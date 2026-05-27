"""Sequence-level shuffle buffer for archs that do internal window sampling.

Used by ports of v1 architectures (txc_pro, mlc, tsae) that expect
``(B, seq_len, d_in)`` batches and slice their own windows. New v2
architectures should prefer ``ActivationBuffer`` (token) or
``WindowBuffer`` (window) instead.

This is a thin convenience wrapper around the refill source: it just
re-samples ``B`` sequences with replacement on each call. No internal
buffer; the trainer's RAM headroom is the only limit.
"""

from __future__ import annotations

from typing import Callable

import torch

from temp_bench.interfaces.batch_iter import WindowBatchIter

RefillSource = Callable[[int], torch.Tensor]


class SequenceBuffer(WindowBatchIter):
    """Yields ``(B, seq_len, d_in)`` — full sequences sampled with replacement."""

    mode = "sequence"   # type: ignore[assignment]

    def __init__(self, refill: RefillSource, *, seq_len: int, device: str = "cpu",
                 seed: int = 0):
        # WindowBatchIter requires T>=1; we don't use T but pass seq_len for the field.
        super().__init__(T=int(seq_len))
        self.refill_fn = refill
        self.device = device
        # Per-call RNG already inside refill_fn (it's seeded). Nothing else here.

    def __call__(self, batch_size: int) -> torch.Tensor:
        batch = self.refill_fn(batch_size)
        if batch.ndim != 3:
            raise ValueError(
                f"SequenceBuffer refill returned {tuple(batch.shape)}; "
                "expected (B, seq_len, d_in)."
            )
        return batch.to(self.device, dtype=torch.float32)
