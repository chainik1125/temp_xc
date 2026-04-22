"""BatchTopK variants of the 4 archs compared in Phase 5.7 experiment (ii).

Each subclass overrides `encode()` to replace the per-sample TopK-scatter
pattern with a `BatchTopK` module that pools top-k across (B, d_sae).
Constructor signature is preserved so the existing dispatcher + probe
routing works with only an added meta flag.

Archs exposed:
    TemporalCrosscoderBatchTopK          -> txcdr_t5_batchtopk
    MultiLayerCrosscoderBatchTopK        -> mlc_batchtopk
    MatryoshkaTXCDRContrastiveMultiscaleBatchTopK
                                         -> agentic_txc_02_batchtopk
    MLCContrastiveMultiscaleBatchTopK_BT -> agentic_mlc_08_batchtopk
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures._batchtopk import BatchTopK
from src.architectures.crosscoder import TemporalCrosscoder
from src.architectures.matryoshka_txcdr import PositionMatryoshkaTXCDR
from src.architectures.matryoshka_txcdr_contrastive_multiscale import (
    MatryoshkaTXCDRContrastiveMultiscale,
)
from src.architectures.mlc import MultiLayerCrosscoder
from src.architectures.mlc_contrastive_multiscale import MLCContrastiveMultiscale


class TemporalCrosscoderBatchTopK(TemporalCrosscoder):
    """TXCDR with BatchTopK over (B, d_sae) instead of per-sample TopK.

    k_eff passed to __init__ is the *per-window* average sparsity — identical
    budget to the TopK variant in expectation.
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None):
        super().__init__(d_in, d_sae, T, k)
        if k is not None:
            self.sparsity = BatchTopK(k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is None:
            return F.relu(pre)
        return self.sparsity(pre)


class MultiLayerCrosscoderBatchTopK(MultiLayerCrosscoder):
    """MLC with BatchTopK."""

    def __init__(self, d_in: int, d_sae: int, n_layers: int, k: int | None):
        super().__init__(d_in, d_sae, n_layers, k)
        if k is not None:
            self.sparsity = BatchTopK(k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.einsum("bld,lds->bs", x, self.W_enc) + self.b_enc
        if self.k is None:
            return F.relu(pre)
        return self.sparsity(pre)


class _MatryoshkaBatchTopKMixin:
    """Shared encode override for matryoshka TXCDR variants."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is None:
            return F.relu(pre)
        return self.sparsity(pre)


class MatryoshkaTXCDRContrastiveMultiscaleBatchTopK(
    _MatryoshkaBatchTopKMixin, MatryoshkaTXCDRContrastiveMultiscale
):
    """agentic_txc_02 + BatchTopK sparsity."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k: int | None,
        n_contr_scales: int = 3,
        gamma: float = 0.5,
        latent_splits: tuple[int, ...] | None = None,
    ):
        super().__init__(
            d_in, d_sae, T, k,
            n_contr_scales=n_contr_scales,
            gamma=gamma,
            latent_splits=latent_splits,
        )
        if k is not None:
            self.sparsity = BatchTopK(k)


class MLCContrastiveMultiscaleBatchTopK(MLCContrastiveMultiscale):
    """agentic_mlc_08 + BatchTopK sparsity."""

    def __init__(self, d_in, d_sae, n_layers, k, prefix_lens=None,
                 gamma: float = 0.5):
        super().__init__(
            d_in, d_sae, n_layers, k,
            prefix_lens=prefix_lens, gamma=gamma,
        )
        if k is not None:
            self.sparsity = BatchTopK(k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.einsum("bld,lds->bs", x, self.W_enc) + self.b_enc
        if self.k is None:
            return F.relu(pre)
        return self.sparsity(pre)
