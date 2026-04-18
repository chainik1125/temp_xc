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


def _shuffle_within_sequence(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Permute each sequence's token order with a per-sequence seed.

    Matches `src.bench.data._shuffle_within_sequence_` but operates on
    GPU tensors for the probing path (data.py's version is for CPU
    cached activations). Two archs called with the same seed see
    identically permuted inputs → paired delta metrics are fair.
    """
    assert x.dim() >= 2, f"expected (B, L, ...), got {tuple(x.shape)}"
    B, L = x.shape[0], x.shape[1]
    out = torch.empty_like(x)
    for b in range(B):
        g = torch.Generator(device="cpu").manual_seed(seed + b)
        perm = torch.randperm(L, generator=g).to(x.device)
        out[b] = x[b, perm]
    return out


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

    def encode(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Return per-position feature activations of shape (B, T, d_sae).

        Uniform contract across architectures so `src.shared.temporal_metrics`
        can consume the output without special-casing. The concrete shape
        comes from architecture-specific semantics:

        - Token-independent SAEs (TopKSAE) apply their encoder to each
          position in turn and stack.
        - Per-position SAEs (StackedSAE) call each position-specific
          encoder directly.
        - Shared-latent crosscoder (TXCDRv2) returns per-position
          pre-activation contributions masked by the shared-z TopK support
          — *not* the native `(B, h)` output, which is permutation-invariant
          under within-window shuffling and therefore mathematically
          non-functional as a shuffle-sensitivity metric. See
          `docs/aniket/sprint_coding_dataset_plan.md` § Encode contract.

        Subclasses should override. Default raises so callers fail loudly
        rather than silently returning None.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement encode()"
        )

    @property
    def n_decoder_positions(self) -> int | None:
        """Number of per-position decoders, or None for single-decoder models."""
        return None

    def encode_for_probing(
        self,
        model: nn.Module,
        x: torch.Tensor,
        shuffle_seed: int | None = None,
    ) -> torch.Tensor:
        """Probing-time encode with optional within-sequence shuffle control.

        Contract — SAME for every arch:
            input:  x of shape (B, L, d_in)
            output: features of shape (B, L, d_sae_effective)

        `shuffle_seed` is the reproducibility knob the harness uses to
        get paired ordered / shuffled measurements. When set, each
        sequence in the batch is permuted along its token axis with
        `torch.Generator().manual_seed(shuffle_seed + seq_idx)`, so
        two architectures called with the same `shuffle_seed` see
        identically permuted inputs.

        Subclasses may override if they need more than "shuffle input,
        then call encode()" — the default is correct for every current
        arch (SAE, TempXC, MLC via its own probing path).

        Note: shuffling is deliberately NOT done inside the model's
        `encode()`. The arch is an encoder; whether its input was
        shuffled is a data-pipeline concern. Keeping this separation
        is why SAEBench's per-arch benchmarking stays simple.
        """
        if shuffle_seed is not None:
            x = _shuffle_within_sequence(x, shuffle_seed)
        return self.encode(model, x)


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
