"""Data pipeline: pre-generates long Markov chains and samples (B, T, d) windows."""

from __future__ import annotations

import torch

from ..config import DataConfig
from .markov import generate_markov_support
from .toy_model import ToyModel


class DataPipeline:
    """Generates temporally correlated activation windows for all models.

    Pre-generates a cache of long sequences and samples random windows from
    them during training. This avoids the per-batch Markov generation overhead.
    """

    def __init__(self, config: DataConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cpu")
        self._gen = torch.Generator().manual_seed(config.seed)
        self.toy_model = ToyModel(
            config.n_features, config.d_model, generator=self._gen
        ).to(self.device)
        self.true_features = self.toy_model.features  # (n_features, d)
        self._cache: dict[float, torch.Tensor] = {}  # rho -> (n_chains, L, d)

    @property
    def n_features(self) -> int:
        return self.config.n_features

    @property
    def d_model(self) -> int:
        return self.config.d_model

    def _get_cache(
        self, rho: float, n_chains: int = 128, chain_len: int = 2048
    ) -> torch.Tensor:
        """Get or create cached long sequences for a given rho.

        Returns:
            x: (n_chains, chain_len, d_model) tensor of pre-generated activations.
        """
        if rho not in self._cache:
            cache_gen = torch.Generator().manual_seed(self.config.seed + hash(str(rho)) % 10000)
            support = generate_markov_support(
                self.config.n_features,
                chain_len,
                self.config.pi,
                rho,
                n_sequences=n_chains,
                generator=cache_gen,
            )
            x = self.toy_model.embed(
                support,
                self.config.magnitude_mean,
                self.config.magnitude_std,
                generator=cache_gen,
            )
            self._cache[rho] = x.to(self.device)
        return self._cache[rho]

    def sample_windows(
        self,
        batch_size: int,
        T: int,
        rho: float,
        pi: float | None = None,
    ) -> torch.Tensor:
        """Sample random (B, T, d) windows from pre-generated chains.

        Args:
            batch_size: Number of windows.
            T: Window length (sequence positions).
            rho: Lag-1 autocorrelation for Markov chains.
            pi: Unused (kept for API compatibility; uses config.pi from cache).

        Returns:
            x: (batch_size, T, d_model) tensor.
        """
        cache = self._get_cache(rho)
        n_chains, chain_len, _ = cache.shape

        # Sample random chain indices and start positions
        chain_idx = torch.randint(n_chains, (batch_size,), generator=self._gen)
        start_idx = torch.randint(chain_len - T, (batch_size,), generator=self._gen)

        # Vectorized window gathering using unfold
        # Reshape cache into (n_chains * chain_len, d) then use advanced indexing
        d = cache.shape[2]
        # Build flat indices: for each sample, indices [start, start+1, ..., start+T-1]
        offsets = torch.arange(T, device=cache.device).unsqueeze(0)  # (1, T)
        flat_pos = start_idx.unsqueeze(1) + offsets  # (B, T)
        windows = cache[chain_idx.unsqueeze(1).expand_as(flat_pos), flat_pos]  # (B, T, d)
        return windows

    def eval_data(
        self,
        n_sequences: int,
        T: int,
        rho: float,
        pi: float | None = None,
        *,
        seed: int = 9999,
    ) -> torch.Tensor:
        """Generate fixed evaluation data with a separate seed.

        Returns:
            x: (n_sequences, T, d_model) tensor.
        """
        if pi is None:
            pi = self.config.pi

        eval_gen = torch.Generator().manual_seed(seed)
        support = generate_markov_support(
            self.config.n_features,
            T,
            pi,
            rho,
            n_sequences=n_sequences,
            generator=eval_gen,
        )
        x = self.toy_model.embed(
            support,
            self.config.magnitude_mean,
            self.config.magnitude_std,
            generator=eval_gen,
        )
        return x.to(self.device)
