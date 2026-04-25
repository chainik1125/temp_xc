"""Data pipeline: pre-generates long Markov chains and samples (B, T, d) windows."""

from __future__ import annotations

import torch

from ..config import DataConfig
from .markov import emit, generate_markov_support, generate_markov_support_hetero
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
    def has_noisy_emissions(self) -> bool:
        return self.config.p_A != 0.0 or self.config.p_B != 1.0

    @property
    def is_hetero_rho(self) -> bool:
        return len(self.config.rho_per_feature) > 0

    def _rho_tensor(self, scalar_rho: float) -> torch.Tensor:
        """Per-feature rho tensor — uses config.rho_per_feature if set, else scalar."""
        if self.is_hetero_rho:
            rhos = torch.tensor(self.config.rho_per_feature, dtype=torch.float32)
            assert rhos.shape[0] == self.config.n_features, (
                f"rho_per_feature has {rhos.shape[0]} entries but "
                f"n_features={self.config.n_features}"
            )
            return rhos
        return torch.full((self.config.n_features,), float(scalar_rho))

    def _generate_hmm(
        self,
        T: int,
        rho: float,
        n_sequences: int,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate (x, s, h) for the given config.

        h: hidden-state Markov chain (n_seq, n_features, T).
        s: emitted support (n_seq, n_features, T) — equal to h when emissions
           are deterministic (p_A=0, p_B=1).
        x: embedded activations (n_seq, T, d_model), built from s since that
           is what the model observes.
        """
        if self.is_hetero_rho:
            rhos = self._rho_tensor(rho)
            h = generate_markov_support_hetero(
                rhos, T, self.config.pi, n_sequences=n_sequences, generator=generator
            )
        else:
            h = generate_markov_support(
                self.config.n_features,
                T,
                self.config.pi,
                rho,
                n_sequences=n_sequences,
                generator=generator,
            )
        if self.has_noisy_emissions:
            s = emit(h, self.config.p_A, self.config.p_B, generator=generator)
        else:
            s = h
        x = self.toy_model.embed(
            s,
            self.config.magnitude_mean,
            self.config.magnitude_std,
            generator=generator,
        )
        return x, s, h

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

        For heterogeneous-rho configs the scalar rho argument is ignored and
        the cache is keyed under a sentinel. Noisy-emission configs sample a
        fresh emission each time the cache is (re)built.

        Returns:
            x: (n_chains, chain_len, d_model) tensor of pre-generated activations.
        """
        key = -1.0 if self.is_hetero_rho else rho
        if key not in self._cache:
            cache_gen = torch.Generator().manual_seed(
                self.config.seed + hash(str(key)) % 10000
            )
            x, _s, _h = self._generate_hmm(
                T=chain_len, rho=rho, n_sequences=n_chains, generator=cache_gen
            )
            self._cache[key] = x.to(self.device)
        return self._cache[key]

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

        # Sample random chain indices and start positions (CPU generator, then move)
        chain_idx = torch.randint(n_chains, (batch_size,), generator=self._gen)
        start_idx = torch.randint(chain_len - T, (batch_size,), generator=self._gen)

        # Move indices to cache device for advanced indexing
        dev = cache.device
        chain_idx = chain_idx.to(dev)
        start_idx = start_idx.to(dev)

        # Build flat indices: for each sample, indices [start, start+1, ..., start+T-1]
        offsets = torch.arange(T, device=dev).unsqueeze(0)  # (1, T)
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
        x, _, _ = self.eval_data_with_support(
            n_sequences, T, rho, pi=pi, seed=seed
        )
        return x

    def eval_data_with_support(
        self,
        n_sequences: int,
        T: int,
        rho: float,
        pi: float | None = None,
        *,
        seed: int = 9999,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same as :meth:`eval_data` but also returns (s, h) support tensors.

        Used by the denoising evaluation for Fig 8/9: local recovery is
        measured against the observed support ``s``, global recovery against
        the hidden state ``h``. Both are (n_sequences, n_features, T).

        Returns:
            x: (n_sequences, T, d_model) tensor.
            s: (n_sequences, n_features, T) observed support.
            h: (n_sequences, n_features, T) hidden state.
        """
        # `pi` arg kept for API compat with eval_data; config.pi is authoritative.
        del pi
        eval_gen = torch.Generator().manual_seed(seed)
        x, s, h = self._generate_hmm(
            T=T, rho=rho, n_sequences=n_sequences, generator=eval_gen
        )
        return x.to(self.device), s.to(self.device), h.to(self.device)
