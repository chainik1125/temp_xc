"""Multi-Layer Crosscoder — shared latent across L residual-stream layers.

MLC is the layer-axis analog of a Temporal Crosscoder: a single shared latent
describes the LLM's hidden state simultaneously at L adjacent residual-stream
layers centred on some anchor layer.

Architecture (L = number of layers in the window):

    W_enc: (L, d, h)  — per-layer encoder projections
    W_dec: (h, L, d)  — per-layer decoder projections
    b_enc: (h,)       — shared encoder bias
    b_dec: (L, d)     — per-layer decoder bias

Encode: pre = sum_l einsum("bd,ds->bs", x[:, l, :], W_enc[l, :, :]) + b_enc
        z   = TopK(pre)   -> (B, h)   with k non-zeros (shared across layers)

Decode: x_hat[:, l, :] = z @ W_dec[:, l, :] + b_dec[l, :]   -> (B, L, d)

This matches the crosscoders paper's construction (Lindsey et al., 2024)
applied layer-wise rather than time-wise. It is mechanically identical to
`TemporalCrosscoder` — the difference is purely semantic: the "T" axis is
now "L" (layer-within-window) instead of "t" (position-within-window).

Design choice: instead of inheriting from TemporalCrosscoder (which would
hide the semantic distinction), we subclass `ArchSpec` directly and
reimplement. The `data_format = "multi_layer"` flag tells the data pipeline
to emit (B, L, d) batches stacking the L hooked layers.

Reference: conceptually the same math as Anthropic's per-layer crosscoder,
with per-layer encoder/decoder matrices and a shared TopK bottleneck. The
standalone implementation avoids any coupling to the TXCDR module so the
two ArchSpecs can evolve independently (e.g. adding attention-based
encoders to MLC while keeping TXCDR a plain per-position linear encoder).

Independent Phase 5 implementation; does not reuse Aniket's
src/bench/architectures/mlc.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.base import ArchSpec, EvalOutput


class MultiLayerCrosscoder(nn.Module):
    """Shared-latent crosscoder across L adjacent residual-stream layers.

    Args:
        d_in: residual-stream width of each hooked layer (assumed same).
        d_sae: dictionary size.
        n_layers: number of layers in the window (Aniket used 5: L10-14).
        k: window-level TopK. None for ReLU-only (not recommended for NLP).
    """

    def __init__(self, d_in: int, d_sae: int, n_layers: int, k: int | None):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.n_layers = n_layers
        self.k = k

        self.W_enc = nn.Parameter(
            torch.randn(n_layers, d_in, d_sae) * (1.0 / d_in**0.5)
        )
        self.W_dec = nn.Parameter(
            torch.randn(d_sae, n_layers, d_in) * (1.0 / d_sae**0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(n_layers, d_in))

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        # Unit-norm each dictionary atom taken jointly over (layer, d_in).
        # Consistent with TXCDR's convention in src/architectures/crosscoder.py.
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d) -> z: (B, h) with k non-zeros."""
        pre = torch.einsum("bld,lds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, h) -> x_hat: (B, L, d)."""
        return torch.einsum("bs,sld->bld", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        """x: (B, L, d) -> (recon_loss, x_hat, z)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, z

    def decoder_directions_at(self, layer: int) -> torch.Tensor:
        """(d_in, d_sae) decoder columns for a given layer within the window."""
        return self.W_dec[:, layer, :].T

    @property
    def decoder_dirs_averaged(self) -> torch.Tensor:
        """(d_in, d_sae) decoder columns averaged across layers."""
        return self.W_dec.mean(dim=1).T


class MLCSpec(ArchSpec):
    """ArchSpec for the Multi-Layer Crosscoder.

    Args:
        n_layers: number of layers stacked in the input window.
        anchor_layer: which layer index (0..n_layers-1) within the window
            is considered the "canonical" layer for downstream probing.
            Returned by `decoder_directions(pos=None)`. Defaults to the
            center, matching MLC-as-middle-out-window semantics.
    """

    data_format = "multi_layer"

    def __init__(self, n_layers: int, anchor_layer: int | None = None):
        self.n_layers = n_layers
        self.anchor_layer = (
            anchor_layer if anchor_layer is not None else n_layers // 2
        )
        self.name = f"MLC L={n_layers}"

    @property
    def n_decoder_positions(self) -> int:
        return self.n_layers

    def create(self, d_in, d_sae, k, device):
        return MultiLayerCrosscoder(d_in, d_sae, self.n_layers, k).to(device)

    def train(
        self, model, gen_fn, total_steps, batch_size, lr, device,
        log_every: int = 500, grad_clip: float = 1.0,
    ):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        log: dict[str, list[float]] = {"loss": [], "l0": []}
        model.train()

        for step in range(total_steps):
            x = gen_fn(batch_size).to(device)
            loss, _, z = model(x)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            model._normalize_decoder()

            if step % log_every == 0 or step == total_steps - 1:
                with torch.no_grad():
                    l0 = (z > 0).float().sum(dim=-1).mean().item()
                log["loss"].append(loss.item())
                log["l0"].append(l0)

        model.eval()
        return log

    def eval_forward(self, model, x):
        loss, x_hat, z = model(x)
        se = (x_hat - x).pow(2).sum().item()
        signal = x.pow(2).sum().item()
        l0 = (z > 0).float().sum(dim=-1).sum().item()
        return EvalOutput(
            sum_se=se, sum_signal=signal, sum_l0=l0,
            n_tokens=x.shape[0],
        )

    def decoder_directions(self, model, pos: int | None = None):
        if pos is None:
            return model.decoder_directions_at(self.anchor_layer)
        return model.decoder_directions_at(pos)
