"""Drop-in shim for `temporal_crosscoders.data.CachedDataSource` backed by a
pre-generated colored-source dataset.

The TXC training loop in temporal_crosscoders/train.py only needs an object
that exposes `act_chains` (for occasional shape introspection) and
`sample_windows(batch_size, T) -> (B, T, d)`. We duck-type that surface here
without inheriting from CachedDataSource, so we don't have to touch the
toy-model path: our `x` is already in observation space.
"""

from __future__ import annotations

import torch


class ColoredSourceCache:
    def __init__(self, x: torch.Tensor, device: torch.device):
        """
        Args:
            x: (n_seq, T_chain, d) pre-generated observations.
            device: device to move sampled windows to.
        """
        if x.dim() != 3:
            raise ValueError(f"x must be (n_seq, T_chain, d); got shape {tuple(x.shape)}")
        self.act_chains = x.to(device=device, dtype=torch.float32).contiguous()
        self.num_chains, self.chain_length, self.d = self.act_chains.shape
        self.device = device

    def refresh(self) -> None:
        """No-op: data is fully pre-generated. Keeping the cache deterministic
        is important so the ground-truth basis F stays valid across calls."""

    def sample_windows(self, batch_size: int, T: int) -> torch.Tensor:
        """Sample random sliding windows -> (B, T, d) on self.device."""
        if T > self.chain_length:
            raise ValueError(f"T={T} exceeds chain_length={self.chain_length}")
        max_start = self.chain_length - T
        chain_idx = torch.randint(
            0, self.num_chains, (batch_size,), device=self.device
        )
        start_idx = torch.randint(
            0, max_start + 1, (batch_size,), device=self.device
        )
        offsets = torch.arange(T, device=self.device).unsqueeze(0)  # (1, T)
        pos_idx = start_idx.unsqueeze(1) + offsets                  # (B, T)
        chain_exp = chain_idx.unsqueeze(1).expand(-1, T)            # (B, T)
        return self.act_chains[chain_exp, pos_idx]                  # (B, T, d)
