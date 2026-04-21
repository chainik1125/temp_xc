"""MLC extended across T adjacent tokens — novel time×layer architecture.

Phase 5.7 autoresearch candidate A4. Combines MLC's layer-axis
sharing with TXCDR's temporal axis:

  - Input shape `(B, T, L, d_in)` — T adjacent tokens × L layers × d_in.
  - **Shared-across-time encoder**: one `(L, d_in, d_sae)` encoder that
    is applied to each of the T token-slabs (SAME weights across t).
    The pre-activations are summed over t, then TopK — so the features
    that "win" are those that fire consistently across ALL T tokens.
  - **Per-(t, l) decoder**: `W_dec: (d_sae, T, L, d_in)`. Decoder is
    free to vary with both t and l.

Distinct from:
  - `MultiLayerCrosscoder`: MLC has T=1 implicit; this is T>1.
  - `TimeLayerCrosscoder`: that model has PER-TOKEN encoder weights
    (W_enc (T, L, d, d_sae)). Here the encoder is shared across t.

Rationale: forces features to be time-invariant on the encoding side
(so they represent "things the sequence is doing at this point,
independent of exact token position") while letting the decoder
re-express per-(t, l). Hypothesis: beats MLC by capturing
sub-window temporal dynamics without blowing up encoder capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLCTemporal(nn.Module):
    """MLC with shared-across-time encoder + per-(t, l) decoder.

    Args:
        d_in: residual-stream width.
        d_sae: dictionary size.
        T: number of adjacent tokens in the window.
        n_layers: number of residual-stream layers in the window.
        k: window-level TopK over the d_sae latents.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        n_layers: int,
        k: int | None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.n_layers = n_layers
        self.k = k

        # Encoder shared across time: (L, d_in, d_sae). Identical shape
        # and initialisation to MLC's encoder.
        self.W_enc = nn.Parameter(
            torch.randn(n_layers, d_in, d_sae) * (1.0 / d_in**0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder per-(t, l): (d_sae, T, L, d_in).
        self.W_dec = nn.Parameter(
            torch.randn(d_sae, T, n_layers, d_in) * (1.0 / d_sae**0.5)
        )
        self.b_dec = nn.Parameter(torch.zeros(T, n_layers, d_in))

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        """Unit-norm each dictionary atom taken jointly over (T, L, d_in).

        torch.Tensor.norm with a >2-element dim tuple dispatches to
        linalg.matrix_norm which only accepts 2 dims. Flatten the trailing
        (T, L, d_in) dims first, then vector_norm over the flattened axis.
        """
        d_sae = self.W_dec.shape[0]
        flat = self.W_dec.reshape(d_sae, -1)                      # (d_sae, T*L*d_in)
        norms = torch.linalg.vector_norm(
            flat, dim=1, keepdim=True,
        ).clamp(min=1e-8)
        flat_normed = flat / norms
        self.W_dec.data = flat_normed.reshape_as(self.W_dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, L, d_in) -> z: (B, d_sae) with k non-zeros.

        Pre-activation sums the MLC-encoded pre per token:
            pre[b, s] = sum_t sum_l x[b, t, l, :] @ W_enc[l, :, s] + b_enc[s]
        """
        # einsum over both t and l — W_enc is (L, d_in, d_sae), so the
        # result is summed over both axes.
        pre = torch.einsum("btld,lds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_sae) -> x_hat: (B, T, L, d_in)."""
        return torch.einsum("bs,stld->btld", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        """x: (B, T, L, d_in) -> (recon_loss, x_hat, z)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, z
