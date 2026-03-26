"""Data pipeline: generates (B, T, d) windows for training and evaluation."""

from __future__ import annotations

import torch

from ..config import DataConfig
from .markov import generate_markov_support
from .toy_model import ToyModel


class DataPipeline:
    """Generates temporally correlated activation windows for all models.

    All models receive (B, T, d) tensors uniformly.
    """

    def __init__(self, config: DataConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cpu")
        self._gen = torch.Generator().manual_seed(config.seed)
        self.toy_model = ToyModel(
            config.n_features, config.d_model, generator=self._gen
        ).to(self.device)
        self.true_features = self.toy_model.features  # (n_features, d)

    @property
    def n_features(self) -> int:
        return self.config.n_features

    @property
    def d_model(self) -> int:
        return self.config.d_model

    def sample_windows(
        self,
        batch_size: int,
        T: int,
        rho: float,
        pi: float | None = None,
    ) -> torch.Tensor:
        """Generate a batch of (B, T, d) activation windows.

        Args:
            batch_size: Number of windows.
            T: Window length (sequence positions).
            rho: Lag-1 autocorrelation for Markov chains.
            pi: Marginal firing probability (defaults to config.pi).

        Returns:
            x: (batch_size, T, d_model) tensor.
        """
        if pi is None:
            pi = self.config.pi

        support = generate_markov_support(
            self.config.n_features,
            T,
            pi,
            rho,
            n_sequences=batch_size,
            generator=self._gen,
        )
        x = self.toy_model.embed(
            support,
            self.config.magnitude_mean,
            self.config.magnitude_std,
            generator=self._gen,
        )
        return x.to(self.device)

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
