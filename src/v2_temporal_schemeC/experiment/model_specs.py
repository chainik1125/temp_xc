"""Model specifications for the unified experiment runner.

Each ModelSpec wraps a model type (SAE, TFA, TXCDR) with a consistent
interface for creation, training, evaluation, and decoder extraction.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn

from src.v2_temporal_schemeC.relu_sae import (
    ReLUSAE, ReLUSAETrainingConfig, train_relu_sae,
)
from src.v2_temporal_schemeC.tfa.saeTemporal import TemporalSAE
from src.v2_temporal_schemeC.train_tfa import (
    TFATrainingConfig, train_tfa,
)
from src.v2_temporal_schemeC.temporal_crosscoder import (
    TemporalCrosscoder, CrosscoderTrainingConfig, train_crosscoder,
)
from src.v2_temporal_schemeC.stacked_sae import (
    StackedSAE, StackedSAETrainingConfig, train_stacked_sae,
)


@dataclass
class EvalOutput:
    """Summed totals from one eval batch. Use sums (not means) so batches
    of different sizes can be aggregated correctly."""
    sum_se: float
    sum_signal: float
    sum_novel_l0: float
    sum_pred_l0: float
    sum_total_l0: float
    n_tokens: int


@dataclass
class ModelEntry:
    """A model to include in a sweep experiment."""
    name: str
    spec: Any  # ModelSpec instance
    gen_key: str  # "flat", "seq", "seq_shuffled", or "window_{T}"
    training_overrides: dict = field(default_factory=dict)


# ── SAE ──────────────────────────────────────────────────────────────


class SAEModelSpec:
    name: str = "SAE"
    data_format: str = "flat"
    n_decoder_positions: int | None = None

    def create(self, d_in: int, d_sae: int, k: int | None,
               device: torch.device, **kwargs) -> ReLUSAE:
        return ReLUSAE(d_in, d_sae, k=k).to(device)

    def make_train_config(self, total_steps: int, batch_size: int, lr: float,
                          l1_coeff: float = 0.0, log_every: int | None = None,
                          **kwargs) -> ReLUSAETrainingConfig:
        return ReLUSAETrainingConfig(
            total_steps=total_steps,
            batch_size=batch_size,
            lr=lr,
            l1_coeff=l1_coeff,
            log_every=log_every or total_steps,
        )

    def train(self, model: ReLUSAE, gen_fn: Callable,
              config: ReLUSAETrainingConfig,
              device: torch.device) -> tuple[ReLUSAE, dict]:
        return train_relu_sae(model, gen_fn, config, device)

    def eval_forward(self, model: ReLUSAE, x: torch.Tensor) -> EvalOutput:
        x_hat, z = model(x)
        se = (x - x_hat).pow(2).sum().item()
        signal = x.pow(2).sum().item()
        l0 = (z > 0).float().sum(dim=-1).sum().item()
        n = x.shape[0]
        return EvalOutput(
            sum_se=se, sum_signal=signal,
            sum_novel_l0=l0, sum_pred_l0=0.0, sum_total_l0=l0,
            n_tokens=n,
        )

    def decoder_directions(self, model: ReLUSAE,
                           pos: int | None = None) -> torch.Tensor:
        return model.W_dec.data.T  # (d_in, d_sae)


# ── TFA ──────────────────────────────────────────────────────────────


class TFAModelSpec:
    data_format: str = "seq"
    n_decoder_positions: int | None = None

    def __init__(self, use_pos_encoding: bool = False,
                 n_heads: int = 4, n_attn_layers: int = 1,
                 bottleneck_factor: int = 1):
        self.use_pos_encoding = use_pos_encoding
        self.n_heads = n_heads
        self.n_attn_layers = n_attn_layers
        self.bottleneck_factor = bottleneck_factor
        self.name = "TFA-pos" if use_pos_encoding else "TFA"

    def create(self, d_in: int, d_sae: int, k: int | None,
               device: torch.device, **kwargs) -> TemporalSAE:
        sae_diff_type = "relu" if k is None else "topk"
        tfa = TemporalSAE(
            dimin=d_in,
            width=d_sae,
            n_heads=self.n_heads,
            sae_diff_type=sae_diff_type,
            kval_topk=k,
            tied_weights=True,
            n_attn_layers=self.n_attn_layers,
            bottleneck_factor=self.bottleneck_factor,
            use_pos_encoding=self.use_pos_encoding,
        )
        return tfa.to(device)

    def make_train_config(self, total_steps: int, batch_size: int, lr: float,
                          l1_coeff: float = 0.0, log_every: int | None = None,
                          **kwargs) -> TFATrainingConfig:
        return TFATrainingConfig(
            total_steps=total_steps,
            batch_size=batch_size,
            lr=lr,
            l1_coeff=l1_coeff,
            log_every=log_every or total_steps,
        )

    def train(self, model: TemporalSAE, gen_fn: Callable,
              config: TFATrainingConfig,
              device: torch.device) -> tuple[TemporalSAE, dict]:
        return train_tfa(model, gen_fn, config, device)

    def eval_forward(self, model: TemporalSAE,
                     x: torch.Tensor) -> EvalOutput:
        recons, inter = model(x)
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        rf = recons.reshape(-1, D)
        se = (xf - rf).pow(2).sum().item()
        signal = xf.pow(2).sum().item()
        novel_l0 = (inter["novel_codes"] > 0).float().sum(dim=-1).sum().item()
        pred_l0 = (inter["pred_codes"].abs() > 1e-8).float().sum(dim=-1).sum().item()
        n = B * T
        return EvalOutput(
            sum_se=se, sum_signal=signal,
            sum_novel_l0=novel_l0, sum_pred_l0=pred_l0,
            sum_total_l0=novel_l0 + pred_l0,
            n_tokens=n,
        )

    def decoder_directions(self, model: TemporalSAE,
                           pos: int | None = None) -> torch.Tensor:
        return model.D.data.T  # (d_in, d_sae)


# ── TXCDR ────────────────────────────────────────────────────────────


class TXCDRModelSpec:
    data_format: str = "window"

    def __init__(self, T: int):
        self.T = T
        self.name = f"TXCDR T={T}"
        self.n_decoder_positions = T

    def create(self, d_in: int, d_sae: int, k: int | None,
               device: torch.device, **kwargs) -> TemporalCrosscoder:
        return TemporalCrosscoder(d_in, d_sae, self.T, k=k).to(device)

    def make_train_config(self, total_steps: int, batch_size: int, lr: float,
                          l1_coeff: float = 0.0, log_every: int | None = None,
                          **kwargs) -> CrosscoderTrainingConfig:
        return CrosscoderTrainingConfig(
            total_steps=total_steps,
            batch_size=batch_size,
            lr=lr,
            l1_coeff=l1_coeff,
            log_every=log_every or total_steps,
        )

    def train(self, model: TemporalCrosscoder, gen_fn: Callable,
              config: CrosscoderTrainingConfig,
              device: torch.device) -> tuple[TemporalCrosscoder, dict]:
        return train_crosscoder(model, gen_fn, config, device)

    def eval_forward(self, model: TemporalCrosscoder,
                     x: torch.Tensor) -> EvalOutput:
        loss, x_hat, z = model(x)
        se = (x_hat - x).pow(2).sum().item()
        signal = x.pow(2).sum().item()
        l0 = (z > 0).float().sum(dim=-1).sum().item()
        n = x.shape[0]  # number of windows
        return EvalOutput(
            sum_se=se, sum_signal=signal,
            sum_novel_l0=l0, sum_pred_l0=0.0, sum_total_l0=l0,
            n_tokens=n,
        )

    def decoder_directions(self, model: TemporalCrosscoder,
                           pos: int | None = None) -> torch.Tensor:
        if pos is None:
            pos = 0
        return model.decoder_directions(pos)  # (d_in, d_sae)


# ── Stacked SAE ──────────────────────────────────────────────────────


class StackedSAEModelSpec:
    data_format: str = "window"

    def __init__(self, T: int):
        self.T = T
        self.name = f"Stacked SAE T={T}"
        self.n_decoder_positions = T

    def create(self, d_in: int, d_sae: int, k: int | None,
               device: torch.device, **kwargs) -> StackedSAE:
        return StackedSAE(d_in, d_sae, self.T, k=k).to(device)

    def make_train_config(self, total_steps: int, batch_size: int, lr: float,
                          l1_coeff: float = 0.0, log_every: int | None = None,
                          **kwargs) -> StackedSAETrainingConfig:
        return StackedSAETrainingConfig(
            total_steps=total_steps,
            batch_size=batch_size,
            lr=lr,
            l1_coeff=l1_coeff,
            log_every=log_every or total_steps,
        )

    def train(self, model: StackedSAE, gen_fn: Callable,
              config: StackedSAETrainingConfig,
              device: torch.device) -> tuple[StackedSAE, dict]:
        return train_stacked_sae(model, gen_fn, config, device)

    def eval_forward(self, model: StackedSAE,
                     x: torch.Tensor) -> EvalOutput:
        loss, x_hat, z = model(x)
        se = (x_hat - x).pow(2).sum().item()
        signal = x.pow(2).sum().item()
        l0 = (z > 0).float().sum(dim=-1).mean(dim=1).sum().item()  # mean over T, sum over B
        n = x.shape[0]
        return EvalOutput(
            sum_se=se, sum_signal=signal,
            sum_novel_l0=l0, sum_pred_l0=0.0, sum_total_l0=l0,
            n_tokens=n,
        )

    def decoder_directions(self, model: StackedSAE,
                           pos: int | None = None) -> torch.Tensor:
        if pos is None:
            pos = 0
        return model.decoder_directions(pos)  # (d_in, d_sae)
