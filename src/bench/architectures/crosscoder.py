"""Temporal Crosscoder — shared-latent crosscoder across T positions.

Encodes a window of T consecutive tokens into a single shared latent
vector z with k*T active features (matching StackedSAE's total L0),
then decodes back to T positions using per-position decoder weights.

Ported from temporal_crosscoders/models.py and Han's temporal_crosscoder.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bench.architectures.base import ArchSpec, EvalOutput


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
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        # Match stacked SAE's total L0: k per position * T positions
        self.k = k * T if k is not None else None

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
        return TemporalCrosscoder(d_in, d_sae, self.T, k).to(device)

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
                print(
                    f"      [{self.name}] step {step:5d}/{total_steps} "
                    f"| loss={loss.item():.4f} | window_l0={window_l0:.2f}",
                    flush=True,
                )

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

    def encode(self, model, x):
        """(B, L, d_in) → (B, L_out, d_sae): per-position contributions
        with shared-z TopK mask applied.

        If L == T: returns shape (B, T, d_sae), one encode on a single
        length-T window.
        If L > T: slides length-T windows over the sequence (stride 1),
        encodes each, and returns the centre-position activation per
        window, shape (B, L-T+1, d_sae). This is the path the sweep's
        temporal metrics take when eval_data has full seq_len=256.

        *Not* the native shared-z `(B, h)` output — that output is a
        sum-over-T before TopK, so it is permutation-invariant under
        within-window shuffling and therefore mathematically
        non-functional as a shuffle-sensitivity signal. The per-position
        formulation preserves the shared-z feature selection (TopK on
        the summed pre-activation, same as the native forward) but
        exposes how much each position contributed to each active
        feature. See `docs/aniket/sprint_coding_dataset_plan.md`
        § Encode contract.
        """
        B, L, d = x.shape
        T = model.T
        if L < T:
            raise ValueError(
                f"crosscoder encode: seq_len {L} < T {T}, cannot form a window"
            )

        if L == T:
            return self._encode_window(model, x)

        # L > T: slide windows, return per-window centre activations.
        # We chunk over the (B*n_windows) flat dim because the per-position
        # pre-activation tensor is (chunk, T, d_sae) — for d_sae ~18k and
        # T=5, full B*n_windows can blow up to 40+ GB. 1024 keeps it under
        # ~400 MB per chunk.
        n_windows = L - T + 1
        windows = x.unfold(1, T, 1).permute(0, 1, 3, 2).contiguous()  # (B, n_windows, T, d)
        flat = windows.reshape(B * n_windows, T, d)
        centre = T // 2
        chunk = 1024
        out = []
        for i in range(0, flat.shape[0], chunk):
            z_chunk = self._encode_window(model, flat[i : i + chunk])  # (chunk, T, d_sae)
            out.append(z_chunk[:, centre, :].cpu())  # take centre, push to CPU
        z_centre = torch.cat(out, dim=0)
        return z_centre.reshape(B, n_windows, -1)

    def _encode_window(self, model, x):
        """Encode a single length-T window batch: (B, T, d_in) → (B, T, d_sae)."""
        B, T, d = x.shape
        pre_per_pos = torch.einsum("btd,tds->bts", x, model.W_enc)  # (B, T, d_sae)
        if model.k is None:
            return F.relu(pre_per_pos)

        pre_sum = pre_per_pos.sum(dim=1) + model.b_enc  # (B, d_sae) — native shared-z pre
        _, topk_idx = pre_sum.topk(model.k, dim=-1)
        mask = torch.zeros_like(pre_sum)
        mask.scatter_(1, topk_idx, 1.0)  # (B, d_sae), 1.0 where feature is in TopK

        z_per_pos = F.relu(pre_per_pos) * mask.unsqueeze(1)  # (B, T, d_sae)

        # Zero-check: IEEE 754 guarantees 0.0 * x == 0.0 for finite x,
        # so masked-out features are exactly zero (not float noise).
        # The check materializes a B×T×d_sae boolean tensor, which for
        # B=1024, T=5, d_sae=18432 is ~94M booleans — fine but not free.
        # Run by default; set CROSSCODER_SKIP_MASK_CHECK=1 to skip when
        # encode() is in a hot path (e.g. large-batch metric backfills).
        import os as _os
        if _os.environ.get("CROSSCODER_SKIP_MASK_CHECK") != "1":
            assert z_per_pos[mask.unsqueeze(1).expand_as(z_per_pos) == 0].abs().max() == 0.0, \
                "crosscoder encode: masked features not exactly zero"

        return z_per_pos
