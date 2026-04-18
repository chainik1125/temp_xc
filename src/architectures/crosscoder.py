"""Temporal Crosscoder — shared-latent crosscoder across T positions.

Encodes a window of T consecutive tokens into a single shared latent
vector z with k*T active features (matching StackedSAE's total L0),
then decodes back to T positions using per-position decoder weights.

Ported from Aniket's original crosscoder models and Han's temporal_crosscoder.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.base import ArchSpec, EvalOutput


class TemporalCrosscoder(nn.Module):
    """Shared-latent temporal crosscoder with TopK sparsity.

    Architecture:
        W_enc: (T, d, h) -- per-position encoder projections
        W_dec: (h, T, d) -- per-position decoder projections
        b_enc: (h,)      -- shared encoder bias
        b_dec: (T, d)    -- per-position decoder bias

    Encode: z = TopK(einsum("btd,tds->bs", x, W_enc) + b_enc) -> (B, h)
    Decode: x_hat = einsum("bs,std->btd", z, W_dec) + b_dec   -> (B, T, d)
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None):
        """k is the number of active latents per window, as-is.

        If you want to match a stacked SAE's total L0 (which uses k per
        position across T positions), the caller should pass k*T here.
        CrosscoderSpec.create handles that scaling; direct callers are
        on their own.
        """
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k

        self.W_enc = nn.Parameter(
            torch.randn(T, d_in, d_sae) * (1.0 / d_in**0.5)
        )
        self.W_dec = nn.Parameter(
            torch.randn(d_sae, T, d_in) * (1.0 / d_sae**0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) -> z: (B, h) with k non-zeros."""
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            topk_vals, topk_idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, topk_idx, F.relu(topk_vals))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, h) -> x_hat: (B, T, d)."""
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        """x: (B, T, d) -> (recon_loss, x_hat, z)"""
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon_loss, x_hat, z

    def decoder_directions_at(self, pos: int) -> torch.Tensor:
        """(d_in, d_sae) decoder columns for a given position."""
        return self.W_dec[:, pos, :].T

    @property
    def decoder_dirs_averaged(self) -> torch.Tensor:
        """(d_in, d_sae) decoder columns averaged across positions."""
        return self.W_dec.mean(dim=1).T


class CrosscoderSpec(ArchSpec):
    """ArchSpec for the Temporal Crosscoder."""

    data_format = "window"

    def __init__(self, T: int):
        self.T = T
        self.name = f"TXCDR T={T}"

    @property
    def n_decoder_positions(self):
        return self.T

    def create(self, d_in, d_sae, k, device):
        # Match stacked SAE's total L0: k per position * T positions.
        k_eff = k * self.T if k is not None else None
        return TemporalCrosscoder(d_in, d_sae, self.T, k_eff).to(device)

    def train(self, model, gen_fn, total_steps, batch_size, lr, device,
              log_every=500, grad_clip=1.0):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        log = {"loss": [], "l0": []}
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
                    window_l0 = (z > 0).float().sum(dim=-1).mean().item()
                log["loss"].append(loss.item())
                log["l0"].append(window_l0)

        model.eval()
        return log

    def eval_forward(self, model, x):
        loss, x_hat, z = model(x)
        se = (x_hat - x).pow(2).sum().item()
        signal = x.pow(2).sum().item()
        l0 = (z > 0).float().sum(dim=-1).sum().item()
        return EvalOutput(sum_se=se, sum_signal=signal, sum_l0=l0,
                          n_tokens=x.shape[0])

    def decoder_directions(self, model, pos=None):
        if pos is None:
            return model.decoder_dirs_averaged
        return model.decoder_directions_at(pos)
