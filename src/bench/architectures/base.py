"""Base architecture interface for the benchmarking framework.

Every architecture implements ArchSpec, which provides a uniform interface
for creation, training, evaluation, and decoder extraction. This is the
core abstraction that makes architecture comparison plug-and-play.

Design adapted from Han's ModelSpec pattern in v2_temporal_schemeC/experiment/.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn


@dataclass
class EvalOutput:
    """Raw evaluation totals from one forward pass.

    Uses sums (not means) so batches of different sizes can be aggregated.
    """

    sum_se: float
    sum_signal: float
    sum_l0: float
    n_tokens: int


@dataclass
class TrainResult:
    """Output from a training run."""

    model: nn.Module
    log: dict[str, list[float]]


class ArchSpec(ABC):
    """Abstract specification for a benchmarkable architecture.

    Each concrete subclass wraps a specific nn.Module and provides:
    - create(): instantiate the model
    - train(): run the training loop
    - eval_forward(): compute losses on a batch
    - decoder_directions(): extract decoder weights for feature recovery
    """

    name: str
    data_format: str  # "flat", "seq", or "window"

    @abstractmethod
    def create(
        self,
        d_in: int,
        d_sae: int,
        k: int | None,
        device: torch.device,
    ) -> nn.Module:
        """Instantiate the model.

        Args:
            d_in: Input dimension (hidden_dim of toy model).
            d_sae: Dictionary size (number of latents).
            k: TopK sparsity. None for L1/ReLU mode.
            device: Torch device.
        """
        ...

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        gen_fn: Callable,
        total_steps: int,
        batch_size: int,
        lr: float,
        device: torch.device,
        log_every: int = 500,
        grad_clip: float = 1.0,
    ) -> dict[str, list[float]]:
        """Train the model and return a log dict.

        Args:
            model: Model to train (from create()).
            gen_fn: Data generator. Signature depends on data_format:
                - "flat": gen_fn(batch_size) -> (B, d)
                - "seq": gen_fn(n_seq) -> (B, T, d)
                - "window": gen_fn(batch_size) -> (B, T, d)
            total_steps: Number of training steps.
            batch_size: Batch size per step.
            lr: Learning rate.
            device: Torch device.
            log_every: Log interval in steps.
            grad_clip: Gradient clipping norm.

        Returns:
            Dict of metric name -> list of values (logged at log_every intervals).
        """
        ...

    @abstractmethod
    def eval_forward(self, model: nn.Module, x: torch.Tensor) -> EvalOutput:
        """Run a forward pass and return evaluation totals.

        Args:
            model: Trained model.
            x: Input batch. Shape depends on data_format:
                - "flat": (B, d)
                - "seq": (B, T, d)
                - "window": (B, T, d)

        Returns:
            EvalOutput with summed metrics.
        """
        ...

    @abstractmethod
    def decoder_directions(
        self, model: nn.Module, pos: int | None = None
    ) -> torch.Tensor:
        """Extract decoder weight matrix for feature recovery evaluation.

        Args:
            model: Trained model.
            pos: Position index for per-position decoders. None for
                position-averaged or position-agnostic decoders.

        Returns:
            Decoder columns of shape (d_in, d_sae).
        """
        ...

    @property
    def n_decoder_positions(self) -> int | None:
        """Number of per-position decoders, or None for single-decoder models."""
        return None


@dataclass
class ModelEntry:
    """A model to include in a sweep experiment.

    Bundles an ArchSpec with a name, the data generator key to use,
    and any training overrides.
    """

    name: str
    spec: ArchSpec
    gen_key: str  # "flat", "seq", or "window_{T}"
    training_overrides: dict[str, Any] = field(default_factory=dict)
