"""Aliased paired-feature data pipeline for the local-vs-predictive benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..data.markov import generate_markov_support
from ..data.toy_model import ToyModel


@dataclass
class AliasedDataConfig:
    """Configuration for the paired-feature aliasing benchmark."""

    n_pairs: int = 10
    d_model: int = 40
    seq_len: int = 64
    pi: float = 0.5
    magnitude_mean: float = 1.0
    magnitude_std: float = 0.15
    seed: int = 42
    eval_n_seq: int = 2000
    cache_n_seq: int = 128
    cache_seq_len: int = 2048
    scaling_n_seq: int = 2048

    @property
    def n_features(self) -> int:
        return 2 * self.n_pairs


@dataclass
class AliasedBatch:
    """A batch of aliased sequences plus hidden-state metadata."""

    x: torch.Tensor
    support: torch.Tensor
    current_feature_idx: torch.Tensor
    next_feature_idx: torch.Tensor
    aliased_state: torch.Tensor
    informative_mask: torch.Tensor

    def sequence_slice(self, start: int, end: int) -> AliasedBatch:
        return AliasedBatch(
            x=self.x[start:end],
            support=self.support[start:end],
            current_feature_idx=self.current_feature_idx[start:end],
            next_feature_idx=self.next_feature_idx[start:end],
            aliased_state=self.aliased_state[start:end],
            informative_mask=self.informative_mask[start:end],
        )

    def time_slice(self, start: int, end: int) -> AliasedBatch:
        return AliasedBatch(
            x=self.x[:, start:end],
            support=self.support[:, :, start:end],
            current_feature_idx=self.current_feature_idx[:, :, start:end],
            next_feature_idx=self.next_feature_idx[:, :, start:end],
            aliased_state=self.aliased_state[:, :, start:end],
            informative_mask=self.informative_mask[:, :, start:end],
        )

    def to(self, device: torch.device) -> AliasedBatch:
        return AliasedBatch(
            x=self.x.to(device),
            support=self.support.to(device),
            current_feature_idx=self.current_feature_idx.to(device),
            next_feature_idx=self.next_feature_idx.to(device),
            aliased_state=self.aliased_state.to(device),
            informative_mask=self.informative_mask.to(device),
        )


class AliasedDataPipeline:
    """Cached paired-feature Markov generator for the aliased benchmark."""

    def __init__(self, config: AliasedDataConfig, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cpu")
        self._gen = torch.Generator().manual_seed(config.seed)
        self.toy_model = ToyModel(
            config.n_features, config.d_model, generator=self._gen
        ).to(self.device)
        self.true_features = self.toy_model.features
        self.scaling_factor = self._compute_scaling_factor()
        self._cache: dict[float, AliasedBatch] = {}

    def _compute_scaling_factor(self) -> float:
        cfg = self.config
        gen = torch.Generator().manual_seed(cfg.seed + 17)
        pair_bits = torch.randint(
            0,
            2,
            (cfg.scaling_n_seq, cfg.n_pairs, 1),
            generator=gen,
        )
        support, *_ = self._metadata_from_bits(pair_bits)
        x = self.toy_model.embed(
            support,
            cfg.magnitude_mean,
            cfg.magnitude_std,
            generator=gen,
        )
        mean_norm = x.reshape(-1, cfg.d_model).norm(dim=-1).mean().item()
        return math.sqrt(cfg.d_model) / max(mean_norm, 1e-8)

    def _metadata_from_bits(
        self, pair_bits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        n_seq, n_pairs, T = pair_bits.shape
        support = torch.zeros(n_seq, cfg.n_features, T)

        pair_offsets = 2 * torch.arange(n_pairs, dtype=torch.long).view(1, n_pairs, 1)
        current_idx = pair_offsets + pair_bits.long()
        support.scatter_(1, current_idx, 1.0)

        next_bits = torch.cat((pair_bits[:, :, 1:], pair_bits[:, :, -1:]), dim=-1)
        next_idx = pair_offsets + next_bits.long()
        aliased_state = 2 * pair_bits.long() + next_bits.long()
        informative_mask = pair_bits != next_bits
        informative_mask[:, :, -1] = False

        return support, current_idx, next_idx, aliased_state, informative_mask

    def _generate_batch(
        self,
        n_seq: int,
        T: int,
        rho: float,
        *,
        generator: torch.Generator,
    ) -> AliasedBatch:
        pair_bits = generate_markov_support(
            self.config.n_pairs,
            T,
            self.config.pi,
            rho,
            n_sequences=n_seq,
            generator=generator,
        )
        support, current_idx, next_idx, aliased_state, informative_mask = (
            self._metadata_from_bits(pair_bits)
        )
        x = self.toy_model.embed(
            support,
            self.config.magnitude_mean,
            self.config.magnitude_std,
            generator=generator,
        )
        x = x * self.scaling_factor
        return AliasedBatch(
            x=x.to(self.device),
            support=support.to(self.device),
            current_feature_idx=current_idx.to(self.device),
            next_feature_idx=next_idx.to(self.device),
            aliased_state=aliased_state.to(self.device),
            informative_mask=informative_mask.to(self.device),
        )

    def _get_cache(self, rho: float) -> AliasedBatch:
        if rho not in self._cache:
            cache_gen = torch.Generator().manual_seed(
                self.config.seed + int(round(rho * 1000)) + 101
            )
            self._cache[rho] = self._generate_batch(
                self.config.cache_n_seq,
                self.config.cache_seq_len,
                rho,
                generator=cache_gen,
            )
        return self._cache[rho]

    def _take_nld(
        self, tensor: torch.Tensor, chain_idx: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        gather_idx = chain_idx.unsqueeze(1).expand_as(positions)
        return tensor[gather_idx, positions]

    def _take_ncl(
        self, tensor: torch.Tensor, chain_idx: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        tensor_ntc = tensor.transpose(1, 2)
        gather_idx = chain_idx.unsqueeze(1).expand_as(positions)
        sliced = tensor_ntc[gather_idx, positions]
        return sliced.transpose(1, 2)

    def sample_batch(self, batch_size: int, T: int, rho: float) -> AliasedBatch:
        cache = self._get_cache(rho)
        max_start = self.config.cache_seq_len - T
        chain_idx = torch.randint(
            cache.x.shape[0], (batch_size,), generator=self._gen
        )
        start_idx = torch.randint(
            max_start + 1, (batch_size,), generator=self._gen
        )
        chain_idx = chain_idx.to(self.device)
        start_idx = start_idx.to(self.device)
        offsets = torch.arange(T, device=self.device).unsqueeze(0)
        positions = start_idx.unsqueeze(1) + offsets

        return AliasedBatch(
            x=self._take_nld(cache.x, chain_idx, positions),
            support=self._take_ncl(cache.support, chain_idx, positions),
            current_feature_idx=self._take_ncl(cache.current_feature_idx, chain_idx, positions),
            next_feature_idx=self._take_ncl(cache.next_feature_idx, chain_idx, positions),
            aliased_state=self._take_ncl(cache.aliased_state, chain_idx, positions),
            informative_mask=self._take_ncl(cache.informative_mask, chain_idx, positions),
        )

    def sample_flat(self, batch_size: int, rho: float) -> torch.Tensor:
        return self.sample_batch(batch_size, 1, rho).x

    def sample_seq(self, batch_size: int, rho: float, *, shuffle: bool = False) -> torch.Tensor:
        batch = self.sample_batch(batch_size, self.config.seq_len, rho)
        if not shuffle:
            return batch.x

        perms = torch.stack(
            [
                torch.randperm(self.config.seq_len, generator=self._gen)
                for _ in range(batch_size)
            ],
            dim=0,
        ).to(self.device)
        return batch.x.gather(
            1, perms.unsqueeze(-1).expand(-1, -1, self.config.d_model)
        )

    def sample_windows(self, batch_size: int, T: int, rho: float) -> torch.Tensor:
        return self.sample_batch(batch_size, T, rho).x

    def eval_batch(self, rho: float, *, seed: int = 9999) -> AliasedBatch:
        eval_gen = torch.Generator().manual_seed(seed + int(round(rho * 1000)))
        return self._generate_batch(
            self.config.eval_n_seq,
            self.config.seq_len,
            rho,
            generator=eval_gen,
        )
