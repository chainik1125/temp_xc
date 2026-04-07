"""
data.py — Data loading utilities for NLP temporal crosscoder experiments.

Loads cached activations from disk into GPU memory, then serves sliding
windows of shape (B, T, d) via fast torch indexing (no Python loops).
"""

import os
import numpy as np
import torch
from config import CACHE_DIR, DEVICE, LAYER_SPECS


class CachedActivationSource:
    """
    Load cached activations into GPU memory for fast random-access windowing.

    Stored on disk as: (NUM_CHAINS, SEQ_LENGTH, d_act) float32 mmap.
    On init, loads into a contiguous GPU tensor for zero-copy sampling.
    """

    def __init__(
        self,
        layer_key: str,
        cache_dir: str = CACHE_DIR,
        device: torch.device = DEVICE,
    ):
        self.layer_key = layer_key
        self.device = device
        spec = LAYER_SPECS[layer_key]
        self.d_act = spec["d_act"]

        path = os.path.join(cache_dir, f"{layer_key}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cached activations not found at {path}. "
                f"Run cache_activations.py first."
            )

        # Load entire array into GPU for fast sampling
        print(f"  Loading {layer_key} into {device} memory...")
        mmap = np.load(path, mmap_mode="r")
        self.num_chains = mmap.shape[0]
        self.seq_length = mmap.shape[1]
        # Load in chunks to avoid OOM during the numpy→torch copy
        chunk = 500
        parts = []
        for i in range(0, self.num_chains, chunk):
            end = min(i + chunk, self.num_chains)
            parts.append(torch.from_numpy(mmap[i:end].copy()).to(device))
        self.data = torch.cat(parts, dim=0)  # (N, L, d) on GPU
        del parts, mmap
        import gc; gc.collect()
        torch.cuda.empty_cache()
        print(f"  Loaded {layer_key}: {self.data.shape} on {device} "
              f"({self.data.element_size() * self.data.nelement() / 1e9:.2f} GB)")

    def sample_windows(self, batch_size: int, T: int) -> torch.Tensor:
        """Sample random sliding windows → (B, T, d) via vectorized indexing."""
        max_start = self.seq_length - T
        if max_start < 1:
            raise ValueError(
                f"T={T} exceeds seq_length={self.seq_length}. "
                f"Reduce T or increase SEQ_LENGTH in config."
            )

        chain_idx = torch.randint(0, self.num_chains, (batch_size,), device=self.device)
        start_idx = torch.randint(0, max_start + 1, (batch_size,), device=self.device)

        # (B, T) position indices
        offsets = torch.arange(T, device=self.device).unsqueeze(0)  # (1, T)
        pos_idx = start_idx.unsqueeze(1) + offsets                   # (B, T)
        chain_exp = chain_idx.unsqueeze(1).expand(-1, T)             # (B, T)

        return self.data[chain_exp, pos_idx]  # (B, T, d)


class WindowIterator:
    """Infinite iterator yielding (B, T, d) windows from cached activations."""

    def __init__(self, source: CachedActivationSource, batch_size: int, T: int):
        self.source = source
        self.batch_size = batch_size
        self.T = T

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        return self.source.sample_windows(self.batch_size, self.T)
