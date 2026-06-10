"""Token-level shuffle buffer (literature-standard SAE training data).

Why a buffer? The standard SAE training recipe (Anthropic monosemanticity
write-up; SAEBench App. B) shuffles a large pool of token activations and
samples i.i.d. tokens per training step. v1 of this framework instead
sampled WHOLE SEQUENCES with replacement, giving 131K correlated tokens
per batch — a real methodological deviation. v2 makes the buffer the
default for per-token archs.

Mechanics:

- The buffer holds up to ``capacity`` tokens (default ~2M).
- Sequences flow in from a refill source (synthetic generator OR
  cached real-LM activations). Each refill chunk: ``(B_refill, seq_len,
  d_in)``; we flatten to ``(B_refill * seq_len, d_in)`` and concatenate
  into the buffer.
- Sampling pops indices uniformly at random; popped slots get refilled
  next refill cycle.
- Refill happens when occupancy drops below ``refill_threshold *
  capacity`` (default 0.5 = half empty).

The buffer subclasses :class:`TokenBatchIter` so the trainer can
``isinstance(it, TokenBatchIter)`` dispatch.
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch

from temp_bench.interfaces.batch_iter import TokenBatchIter

# Type alias: a refill source yields chunks of shape (B, seq_len, d_in).
RefillSource = Callable[[int], torch.Tensor]


class ActivationBuffer(TokenBatchIter):
    """Token-level shuffle buffer.

    Args:
        refill: callable ``refill(n_sequences) -> (n_sequences, seq_len, d_in)``.
            Synthetic data: a generator slice. Real-LM data: a slice of
            the cached ``acts.npy``. Trainer holds one buffer per data
            source.
        capacity: number of tokens to hold at full.
        refill_threshold: float in (0, 1); refill when occupancy
            falls below this fraction.
        seq_len: how many tokens per source-sequence. Used to size
            refill calls so we land on full capacity.
        device: tensors live on this device (default cpu — trainer
            ``.to(cuda)`` per batch).
        seed: int for the internal RNG.
    """

    def __init__(
        self,
        refill: RefillSource,
        *,
        capacity: int = 2_000_000,
        refill_threshold: float = 0.5,
        seq_len: int = 128,
        device: str = "cpu",
        seed: int = 0,
    ) -> None:
        self.refill_fn = refill
        self.capacity = int(capacity)
        self.refill_threshold = float(refill_threshold)
        self.seq_len = int(seq_len)
        self.device = device
        self._gen = torch.Generator(device="cpu").manual_seed(int(seed))

        # Buffer state: a (k, d_in) tensor, where k <= capacity.
        # d_in is set on first refill.
        self._buf: torch.Tensor | None = None

    @property
    def occupancy(self) -> int:
        return 0 if self._buf is None else self._buf.shape[0]

    @property
    def d_in(self) -> int | None:
        return None if self._buf is None else int(self._buf.shape[1])

    def _refill(self) -> None:
        """Top up the buffer with fresh tokens from the source."""
        # How many sequences do we need to fill remaining capacity?
        deficit = self.capacity - self.occupancy
        n_seqs = max(1, deficit // self.seq_len + 1)
        new = self.refill_fn(n_seqs)                         # (n_seqs, T, d_in)
        if new.ndim != 3:
            raise ValueError(
                f"refill must return (B, seq_len, d_in); got {tuple(new.shape)}."
            )
        new = new.reshape(-1, new.shape[-1])                 # (n_seqs * T, d_in)
        if self._buf is None:
            self._buf = new[: self.capacity].clone()
        else:
            self._buf = torch.cat([self._buf, new], dim=0)
            if self._buf.shape[0] > self.capacity:
                # Random subsample down to capacity, preserving shuffle.
                idx = torch.randperm(
                    self._buf.shape[0], generator=self._gen
                )[: self.capacity]
                self._buf = self._buf[idx].clone()

    def __call__(self, batch_size: int) -> torch.Tensor:
        """Return ``(batch_size, d_in)`` float tensor, i.i.d. samples."""
        if self._buf is None or self.occupancy < int(
            self.refill_threshold * self.capacity
        ):
            self._refill()
        idx = torch.randint(
            self.occupancy, (batch_size,), generator=self._gen
        )
        batch = self._buf[idx].to(self.device, dtype=torch.float32)
        # Pop sampled tokens to enforce "without replacement" semantics
        # across a refill cycle? Literature SAE training uses with-
        # replacement for simplicity (and because the buffer is large
        # vs batch). We do the same.
        return batch
