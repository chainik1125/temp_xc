"""Window-level shuffle buffer for TXC / T-SAE / MLC / Stacked archs.

These window archs need contiguous T-token chunks rather than i.i.d.
tokens. The buffer holds ``capacity`` whole sequences (each of length
``seq_len``); sampling returns ``batch_size`` random T-windows drawn
uniformly across sequences AND across positions.

Compared to the v1 "sample 1 random T-window per sequence" pattern:
this buffer is bigger (holds many sequences in RAM at once) and samples
windows freely from any position of any held sequence — closer to "i.i.d.
windows" than v1's "1-per-sequence" approximation.

Same refill discipline as :class:`ActivationBuffer`: when buffer
occupancy drops below ``refill_threshold * capacity`` (in sequences),
top up from the refill source.
"""

from __future__ import annotations

from typing import Callable

import torch

from temp_bench.interfaces.batch_iter import WindowBatchIter

RefillSource = Callable[[int], torch.Tensor]


class WindowBuffer(WindowBatchIter):
    """Sequence-level shuffle buffer that yields random T-windows."""

    def __init__(
        self,
        refill: RefillSource,
        *,
        T: int,
        capacity_seqs: int = 16_000,
        refill_threshold: float = 0.5,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        super().__init__(T=T)
        self.refill_fn = refill
        self.capacity_seqs = int(capacity_seqs)
        self.refill_threshold = float(refill_threshold)
        self.device = device
        self._gen = torch.Generator(device="cpu").manual_seed(int(seed))

        # Buffer state: (n_seqs, seq_len, d_in) tensor on cpu.
        self._buf: torch.Tensor | None = None

    @property
    def occupancy_seqs(self) -> int:
        return 0 if self._buf is None else self._buf.shape[0]

    def _refill(self) -> None:
        deficit = self.capacity_seqs - self.occupancy_seqs
        n_seqs = max(1, deficit)
        new = self.refill_fn(n_seqs)                         # (n_seqs, seq_len, d_in)
        if new.ndim != 3:
            raise ValueError(
                f"refill must return (B, seq_len, d_in); got {tuple(new.shape)}."
            )
        if new.shape[1] < self.T:
            raise ValueError(
                f"refill seq_len={new.shape[1]} < T={self.T}; "
                "increase source seq_len or decrease T."
            )
        if self._buf is None:
            self._buf = new[: self.capacity_seqs].clone()
        else:
            combined = torch.cat([self._buf, new], dim=0)
            if combined.shape[0] > self.capacity_seqs:
                idx = torch.randperm(combined.shape[0], generator=self._gen)[
                    : self.capacity_seqs
                ]
                combined = combined[idx].clone()
            self._buf = combined

    def __call__(self, batch_size: int) -> torch.Tensor:
        """Return ``(batch_size, T, d_in)`` — random T-windows."""
        if self._buf is None or self.occupancy_seqs < int(
            self.refill_threshold * self.capacity_seqs
        ):
            self._refill()

        n_seqs, seq_len, _d_in = self._buf.shape
        seq_idx = torch.randint(n_seqs, (batch_size,), generator=self._gen)
        max_pos = seq_len - self.T + 1
        pos_idx = torch.randint(max_pos, (batch_size,), generator=self._gen)

        # Build position grid and 2-axis advanced index.
        offsets = torch.arange(self.T, dtype=torch.long)
        pos_grid = pos_idx[:, None] + offsets[None, :]       # (B, T)
        batch = self._buf[seq_idx[:, None], pos_grid]        # (B, T, d_in)
        return batch.to(self.device, dtype=torch.float32)
