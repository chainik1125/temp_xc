"""
data.py — Data loading utilities for NLP temporal crosscoder experiments.

Loads cached activations from disk and serves sliding windows of shape (B, T, d).
"""

import os
import numpy as np
import torch
from config import CACHE_DIR, DEVICE, LAYER_SPECS


class CachedActivationSource:
    """
    Load cached activations from a memory-mapped .npy file.

    Stored as: (NUM_CHAINS, SEQ_LENGTH, d_act)
    Serves sliding windows of length T from random chains.
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

        # Memory-map for constant RAM usage
        self.data = np.load(path, mmap_mode="r")  # (N, L, d)
        self.num_chains = self.data.shape[0]
        self.seq_length = self.data.shape[1]
        print(f"  Loaded {layer_key}: {self.data.shape} from {path}")

    def sample_windows(self, batch_size: int, T: int) -> torch.Tensor:
        """Sample random sliding windows → (B, T, d).

        Picks random chains and random starting positions within each chain.
        """
        max_start = self.seq_length - T
        if max_start < 1:
            raise ValueError(
                f"T={T} exceeds seq_length={self.seq_length}. "
                f"Reduce T or increase SEQ_LENGTH in config."
            )

        chain_idx = np.random.randint(0, self.num_chains, size=batch_size)
        start_idx = np.random.randint(0, max_start + 1, size=batch_size)

        # Gather windows from mmap
        windows = np.stack([
            self.data[c, s : s + T, :]
            for c, s in zip(chain_idx, start_idx)
        ])  # (B, T, d)

        return torch.from_numpy(windows.copy()).float().to(self.device)


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
