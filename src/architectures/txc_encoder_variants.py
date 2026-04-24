"""TXC encoder-variants — Part B H10 / H11 / H12.

Replaces TXCDR's per-position W_enc:(T,d_in,d_sae) with simpler encoders
to test "per-position weights vs positional prior" hypothesis.

H10: **TXCSharedReluSum** — shared W_enc + positional embedding + ReLU
     per position, then summed across T. Breaks the linearity that would
     let positional bias commute with the sum.
     Formula: pre = Σ_t σ(W_enc (x_t + p_t) + b);  z = TopK(pre).
     Variant H10a: with positional embedding p_t.
     Variant H10b (control): no positional embedding.

H12: **TXCSharedConcatTwoLayer** — shared first-layer W_1 + pos embedding,
     concatenate across positions, shared second layer W_2 projects to
     d_sae.
     Formula: h_t = σ(W_1 (x_t + p_t) + b_1);  z = TopK(σ(W_2 · concat(h_0..h_{T-1}) + b_2)).
     Explicit position preservation through concatenation.

All three keep the per-position decoder (same as TemporalCrosscoder).
Probe routing uses the existing _encode_txcdr dispatch since encode(x)
signature is unchanged: (B, T, d_in) -> (B, d_sae).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TXCEncoderVariantBase(nn.Module):
    """Shared decoder + utilities (per-position W_dec, TopK, encode API)."""

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k

        # Decoder: same as TemporalCrosscoder.
        self.W_dec = nn.Parameter(
            torch.randn(d_sae, T, d_in) * (1.0 / d_sae**0.5)
        )
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def _topk(self, pre: torch.Tensor) -> torch.Tensor:
        if self.k is None:
            return F.relu(pre)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, topk_idx, F.relu(topk_vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, z


class TXCSharedReluSum(_TXCEncoderVariantBase):
    """H10: Shared W_enc + optional pos embedding + ReLU per position,
    then summed across T. `pre = Σ_t σ(W_enc (x_t + p_t) + b)`.

    Args:
        use_pos: if True, add learned positional embedding p_t to each
            x_t before the shared encoder.
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None,
                 use_pos: bool = True):
        super().__init__(d_in, d_sae, T, k)
        self.use_pos = use_pos
        # Shared encoder (no T dimension).
        self.W_enc = nn.Parameter(
            torch.randn(d_in, d_sae) * (1.0 / d_in**0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        if use_pos:
            # Positional embedding per position, in input space.
            self.pos_embed = nn.Parameter(torch.zeros(T, d_in))
        else:
            self.register_parameter("pos_embed", None)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in)
        if self.use_pos:
            x = x + self.pos_embed.unsqueeze(0)  # (1, T, d_in) broadcasts
        # Per-position pre + ReLU, summed over T.
        # (B, T, d_in) @ (d_in, d_sae) = (B, T, d_sae)
        pre_t = torch.einsum("btd,ds->bts", x, self.W_enc) + self.b_enc
        pre = F.relu(pre_t).sum(dim=1)  # (B, d_sae)
        return self._topk(pre)


class TXCSharedConcatTwoLayer(_TXCEncoderVariantBase):
    """H12: Shared W_1 (with pos embed) → concat T outputs → W_2 → TopK.

    `h_t = σ(W_1 (x_t + p_t) + b_1);  z = TopK(σ(W_2 · concat(h_0..h_{T-1}) + b_2))`.

    Args:
        d_hidden: hidden width per position. Default 512 (light variant).
        use_pos: add positional embedding at first layer.
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None,
                 d_hidden: int = 512, use_pos: bool = True):
        super().__init__(d_in, d_sae, T, k)
        self.d_hidden = int(d_hidden)
        self.use_pos = use_pos
        self.W1 = nn.Parameter(
            torch.randn(d_in, d_hidden) * (1.0 / d_in**0.5)
        )
        self.b1 = nn.Parameter(torch.zeros(d_hidden))
        self.W2 = nn.Parameter(
            torch.randn(T * d_hidden, d_sae) * (1.0 / (T * d_hidden)**0.5)
        )
        self.b2 = nn.Parameter(torch.zeros(d_sae))
        if use_pos:
            self.pos_embed = nn.Parameter(torch.zeros(T, d_in))
        else:
            self.register_parameter("pos_embed", None)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in)
        if self.use_pos:
            x = x + self.pos_embed.unsqueeze(0)
        h = F.relu(
            torch.einsum("btd,dh->bth", x, self.W1) + self.b1
        )  # (B, T, d_hidden)
        h_cat = h.reshape(h.shape[0], -1)  # (B, T*d_hidden)
        pre = h_cat @ self.W2 + self.b2  # (B, d_sae)
        return self._topk(pre)
