"""Layer-wise crosscoder (MLC) — 5-layer middle-out around layer 12.

Anthropic-style cross-layer SAE (Lindsey et al. 2024). Encodes
simultaneous residual-stream activations from a window of layers
`{L-2, L-1, L, L+1, L+2}` into a shared latent vector, with TopK
applied on the layer-summed pre-activation.

**Mathematically identical to TemporalCrosscoder.** The only
difference is the semantics of the T axis: MLC's T is a *layer* axis
(simultaneous activations at different depths of the same model), not
a *time* axis (consecutive token positions at the same layer). At the
PyTorch level the encode/decode math is the same einsum; this class
inherits from `TemporalCrosscoder` and relabels so data pipelines and
logging distinguish the two.

**Integration status** (see
`docs/aniket/experiments/sparse_probing/plan.md` § 11):

- *Architecture*: complete here. Trainable via the standard sweep
  runner once the data pipeline below lands.
- *Training data pipeline*: TODO. Needs multi-hook activation caching
  — simultaneous hooks on layers {10, 11, 12, 13, 14} of Gemma-2-2B,
  writing `(n_seq, seq_len, 5, d_model)` tensors instead of the
  single-layer `(n_seq, seq_len, d_model)` caches we currently have.
- *SAEBench sparse-probing eval*: TODO. Needs a custom multi-hook
  variant of `run_eval_single_dataset` (~200 LoC fork) — SAEBench's
  stock pipeline assumes a single hook point.

The `LayerCrosscoder` / `LayerCrosscoderSpec` classes below are the
architecture half. The data / eval halves are tracked in plan.md § 11.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.bench.architectures.base import EvalOutput
from src.bench.architectures.crosscoder import (
    CrosscoderSpec,
    TemporalCrosscoder,
)


class LayerCrosscoder(TemporalCrosscoder):
    """Alias of `TemporalCrosscoder` with a layer-axis semantic label.

    Same `W_enc: (n_layers, d_in, d_sae)`, `W_dec: (d_sae, n_layers, d_in)`,
    `b_enc: (d_sae,)`, `b_dec: (n_layers, d_in)`. Same encode math:
    `z = TopK(einsum("bnd,nds->bs", x, W_enc) + b_enc)`. Same decode.

    Introduced as its own class so `isinstance(model, LayerCrosscoder)`
    can distinguish MLC from TempXC for data-pipeline dispatch. The
    `n_layers` kwarg is an alias for TemporalCrosscoder's `T`.
    """

    def __init__(self, d_in: int, d_sae: int, n_layers: int, k: int | None):
        super().__init__(d_in=d_in, d_sae=d_sae, T=n_layers, k=k)
        # Expose the layer-axis label for downstream code that
        # inspects the model's semantic shape
        self.n_layers = n_layers


class LayerCrosscoderSpec(CrosscoderSpec):
    """ArchSpec for the Layer-wise Crosscoder (MLC).

    Data format is "multi_layer" (not "window") so the sweep's data
    pipeline knows to assemble `(B, n_layers, d_in)` inputs from
    simultaneously-cached multi-layer activations rather than sliding
    T-windows over a single-layer sequence.

    NOTE: sweeps with this spec will fail until the data pipeline
    supports `data_format="multi_layer"`. Architecture is unit-testable
    in isolation via direct forward passes.
    """

    data_format = "multi_layer"

    def __init__(self, n_layers: int = 5):
        # CrosscoderSpec.__init__ sets self.T; we use that as n_layers
        super().__init__(T=n_layers)
        self.n_layers = n_layers
        self.name = f"MLC n_layers={n_layers}"

    def create(self, d_in, d_sae, k, device):
        """Instantiate LayerCrosscoder. Note we use our own class rather
        than falling back to the CrosscoderSpec's TemporalCrosscoder
        so `isinstance(model, LayerCrosscoder)` is queryable.
        """
        return LayerCrosscoder(d_in, d_sae, self.n_layers, k).to(device)

    @property
    def n_decoder_positions(self):
        return self.n_layers

    def decoder_directions(self, model, pos=None):
        if pos is None:
            return model.decoder_dirs_averaged
        return model.decoder_directions_at(pos)

    # encode() inherits from CrosscoderSpec — same long-input
    # windowing + TopK mask logic. The input axis is interpreted as
    # layers rather than time positions, which only matters for how
    # upstream code builds the input tensor.
