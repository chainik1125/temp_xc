"""SAEBench BaseSAE adapter for Phase 5 ArchSpec checkpoints.

Wraps a trained checkpoint (TopKSAE / TXCDR / MLC / Stacked / Matryoshka /
TFA) into SAEBench's `BaseSAE` contract. Enables running the SAEBench
sparse-probing eval on our architectures without forking SAEBench.

SAEBench's encode contract is:
    encode(x: (B, L, d_in)) -> (B, L, d_sae)

where `d_in` matches the hooked LLM layer's residual width. Our
architectures differ in how they consume this input:

- `TopKSAESpec` (data_format=="flat"): treats each (B, L) position
  independently; reshape to (B*L, d_in), encode, reshape back.
- `CrosscoderSpec`, `StackedSAESpec` (data_format=="window"): consume a
  T-token window and emit either a shared window latent (TXCDR) or
  T per-position latents (Stacked). The adapter runs stride-1 windows
  and applies a selected aggregation strategy (see aggregation.py).
- `MLCSpec` (data_format=="multi_layer"): consumes (B, L_window, d_in)
  at a FIXED set of layers, not stride-1 within one hooked layer. For
  SAEBench probing we cannot reassemble the multi-layer input from a
  single-layer hook, so MLC is handled through a separate code path
  (MLCProbingAdapter in mlc_adapter.py) that hooks multiple layers.

Dependencies: only imports `sae_bench.custom_saes.base_sae.BaseSAE`
and `sae_bench.custom_saes.custom_sae_config.CustomSAEConfig`. If the
sidecar env isn't installed, the adapter module raises ImportError on
import — callers must install SAEBench in a sidecar uv env per
RUNPOD_INSTRUCTIONS.md before evaluating.

Independent Phase 5 implementation; not ported from Aniket's
`src/bench/saebench/saebench_wrapper.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from src.architectures.base import ArchSpec
from src.architectures.topk_sae import TopKSAE, TopKSAESpec
from src.architectures.crosscoder import TemporalCrosscoder, CrosscoderSpec
from src.architectures.stacked_sae import StackedSAE, StackedSAESpec

from experiments.phase5_downstream_utility.probing.aggregation import (
    AggregationName,
    aggregate,
)


def _require_saebench():
    try:
        from sae_bench.custom_saes.base_sae import BaseSAE  # noqa: F401
        from sae_bench.custom_saes.custom_sae_config import (  # noqa: F401
            CustomSAEConfig,
        )
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "sae_bench is not installed in this env. Install it into a "
            "sidecar uv env (it pins numpy<2, datasets<4) and invoke the "
            "probing runner through that env. See RUNPOD_INSTRUCTIONS.md."
        ) from e


@dataclass
class AdapterConfig:
    """Static metadata the adapter exposes to SAEBench."""

    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    model_name: str
    arch_name: str         # "topk_sae" / "txcdr" / "stacked" / ...
    T: int = 1             # window size
    aggregation: AggregationName = "last_position"
    dtype: str = "float32"
    context_size: int = 128
    prepend_bos: bool = True
    normalize_activations: str = "none"


def build_adapter(
    spec: ArchSpec,
    model: nn.Module,
    cfg: AdapterConfig,
):
    """Create a BaseSAE-compatible adapter for a trained checkpoint.

    Imported lazily so the main env doesn't require sae_bench.
    """
    _require_saebench()
    from sae_bench.custom_saes.base_sae import BaseSAE
    from sae_bench.custom_saes.custom_sae_config import CustomSAEConfig

    # Cache a reference to the relevant class-based wrapper per spec type.
    if isinstance(spec, TopKSAESpec):
        adapter_cls = _TopKAdapter
    elif isinstance(spec, CrosscoderSpec):
        adapter_cls = _TXCDRAdapter
    elif isinstance(spec, StackedSAESpec):
        adapter_cls = _StackedAdapter
    else:
        raise ValueError(
            f"No SAEBench adapter registered for {type(spec).__name__}. "
            "MLC requires a multi-layer hook pipeline; see mlc_adapter.py."
        )

    return adapter_cls(
        BaseSAE=BaseSAE,
        CustomSAEConfig=CustomSAEConfig,
        model=model,
        cfg=cfg,
    )


def _make_base_cfg(CustomSAEConfig, cfg: AdapterConfig, effective_d_sae: int):
    """Populate the BaseSAE config object SAEBench expects."""
    return CustomSAEConfig(
        model_name=cfg.model_name,
        d_in=cfg.d_in,
        d_sae=effective_d_sae,
        hook_layer=cfg.hook_layer,
        hook_name=cfg.hook_name,
        architecture=cfg.arch_name,
        dtype=cfg.dtype,
        context_size=cfg.context_size,
        apply_b_dec_to_input=False,
        finetuning_scaling_factor=False,
        activation_fn_str="topk",
        prepend_bos=cfg.prepend_bos,
        normalize_activations=cfg.normalize_activations,
    )


class _AdapterBase(nn.Module):
    """Shared adapter boilerplate."""

    def __init__(self, BaseSAE, CustomSAEConfig, model: nn.Module,
                 cfg: AdapterConfig, effective_d_sae: int):
        super().__init__()
        self._BaseSAE = BaseSAE
        self.model = model
        self._cfg = cfg
        self.d_in = cfg.d_in
        self.d_sae = effective_d_sae
        self.cfg = _make_base_cfg(CustomSAEConfig, cfg, effective_d_sae)
        # SAEBench reads .dtype + .device directly as attributes.
        self._dtype = getattr(torch, cfg.dtype)
        self._device = next(model.parameters()).device

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    # SAEBench calls forward(x) during sanity checks. It's never used
    # by the sparse_probing eval (which calls encode directly), but we
    # define it for the base contract.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class _TopKAdapter(_AdapterBase):
    """Single-token TopK SAE. All aggregations collapse to last_position."""

    def __init__(self, BaseSAE, CustomSAEConfig, model: TopKSAE,
                 cfg: AdapterConfig):
        super().__init__(BaseSAE, CustomSAEConfig, model, cfg,
                         effective_d_sae=cfg.d_sae)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, L, d = x.shape
        flat = x.reshape(B * L, d)
        z = self.model.encode(flat.to(self._dtype))
        return z.reshape(B, L, -1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B, L, h = z.shape
        x_hat = self.model.decode(z.reshape(B * L, h))
        return x_hat.reshape(B, L, self.d_in)


class _TXCDRAdapter(_AdapterBase):
    """TXCDR — shared window latent; aggregations act on (T, d_sae)."""

    def __init__(self, BaseSAE, CustomSAEConfig, model: TemporalCrosscoder,
                 cfg: AdapterConfig):
        # full_window aggregation inflates the effective d_sae.
        effective = (cfg.T * cfg.d_sae) if cfg.aggregation == "full_window" \
            else cfg.d_sae
        super().__init__(BaseSAE, CustomSAEConfig, model, cfg,
                         effective_d_sae=effective)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        def _enc_window(windows: torch.Tensor) -> torch.Tensor:
            # windows: (N, T, d_in) -> TXCDR encode gives (N, d_sae)
            # broadcast the shared latent to (N, T, d_sae) so aggregation
            # can operate uniformly.
            z = self.model.encode(windows.to(self._dtype))
            return z.unsqueeze(1).expand(-1, windows.shape[1], -1)

        return aggregate(
            _enc_window, x, self._cfg.T, self._cfg.aggregation
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Approximation: TXCDR's decode is window-shaped, not per-position.
        # We don't use decode on the probing path, but implement a
        # last-position reconstruction for the BaseSAE forward contract.
        B, L, h = z.shape
        d_sae = self._cfg.d_sae
        if self._cfg.aggregation == "full_window":
            # z has shape (B, L, T*d_sae); take the last-T slice.
            z_last = z[..., -d_sae:]
        else:
            z_last = z
        flat = z_last.reshape(B * L, d_sae)
        x_hat = self.model.decode(flat)  # (B*L, T, d_in)
        # return only the last-position reconstruction
        x_hat = x_hat[:, -1, :]
        return x_hat.reshape(B, L, self.d_in)


class _StackedAdapter(_AdapterBase):
    """Stacked SAE — T independent per-position SAEs."""

    def __init__(self, BaseSAE, CustomSAEConfig, model: StackedSAE,
                 cfg: AdapterConfig):
        effective = (cfg.T * cfg.d_sae) if cfg.aggregation == "full_window" \
            else cfg.d_sae
        super().__init__(BaseSAE, CustomSAEConfig, model, cfg,
                         effective_d_sae=effective)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        T = self._cfg.T

        def _enc_window(windows: torch.Tensor) -> torch.Tensor:
            # windows: (N, T, d_in) -> stack T per-position SAE encodings
            zs = []
            for t in range(T):
                z_t = self.model.saes[t].encode(windows[:, t, :].to(self._dtype))
                zs.append(z_t)
            return torch.stack(zs, dim=1)  # (N, T, d_sae)

        return aggregate(_enc_window, x, T, self._cfg.aggregation)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Last-position decode only (probing path doesn't use decode).
        B, L, h = z.shape
        d_sae = self._cfg.d_sae
        if self._cfg.aggregation == "full_window":
            z_last = z[..., -d_sae:]
        else:
            z_last = z
        flat = z_last.reshape(B * L, d_sae)
        x_hat = self.model.saes[-1].decode(flat)
        return x_hat.reshape(B, L, self.d_in)
