"""Stacked SAE baseline: T independent TopK SAEs, one per window position.

Each position has its own encoder/decoder pair. Total window sparsity
L0 = k * T (k active latents per position). Parameter count scales
with T, and there is no information sharing across positions at either
training or inference time.

This matches the *per-position TopK* variant described in the midterm
report §2.2: "T independent TopK SAE modules, each selecting k active
latents per position." Both existing implementations that produced the
report's Fig 5 data use the same per-position-TopK design:
    origin/andre:src/v2_crosscoder_comparison/architectures/stacked_sae.py
    origin/aniket-runpod:src/bench/architectures/stacked_sae.py

Variants NOT implemented here (noted for future reference):

* **Concat-then-TopK** — T per-position encoders produce a length-(T*d_sae)
  concatenated latent; a single TopK(k*T) is taken over that concatenation.
  This lets the model redistribute its sparsity budget across positions
  (a position with low activation energy can receive 0 active latents
  while another receives 2k). Same parameter count as this Stacked SAE
  and same matched window L0 = k*T, but strictly more expressive.
  Useful as an intermediate between Stacked SAE and TXCDR — it tests
  whether cross-position *budget* alone helps, independent of TXCDR's
  cross-position *feature pooling*. Not in the report.

* **BatchTopK** — TopK is applied over a flattened batch dimension rather
  than per-token, so the *mean* L0 is k but the per-token L0 fluctuates.
  Implementation on `origin/wip/aliased-benchmark-runpod` in
  `src/temporal_bench/models/batchtopk_sae.py`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutput, TemporalAE


class StackedSAE(TemporalAE):
    """T independent TopK SAEs, one per position in the window.

    Mathematically equivalent to a ``ModuleList`` of T independent ``TopKSAE``
    modules. We store the per-position weights as 3D parameter tensors so the
    forward pass is a single batched einsum (one CUDA kernel launch) rather
    than a Python for-loop over T (T kernel launches). On a 4090 with the
    tiny synthetic-data matrices used here, the looped version is launch-
    overhead-bound and roughly T-times slower than this vectorized form.

    Each position's TopK still fires independently — the top-k op acts on the
    last (latent) axis of the (B, T, m) pre-activation tensor, so position t's
    selection sees only position t's latents. Identical semantics to the
    looped version, identical to the report's Stacked SAE definition (§2.2).
    """

    def __init__(self, d_in: int, d_sae: int, T: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k

        # Per-position weights stacked into 3D tensors.
        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(T, d_sae))
        self.W_dec = nn.Parameter(torch.empty(T, d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        # Initialize each position's encoder/decoder independently, matching
        # what `TopKSAE.__init__` would have produced for each sub-SAE.
        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc[t], a=math.sqrt(5))
            with torch.no_grad():
                self.W_dec.data[t] = self.W_enc.data[t].T
        self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        # Normalize each (m, d) decoder slice's rows to unit norm independently.
        # Matches TopKSAE._normalize_decoder applied per position.
        norms = self.W_dec.data.norm(dim=2, keepdim=True).clamp(min=1e-8)  # (T, m, 1)
        self.W_dec.data.div_(norms)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        B, T, d = x.shape
        assert T == self.T, f"Expected T={self.T}, got T={T}"

        # SAE-tied-bias trick (matches TopKSAE): pre-bias-subtract on input.
        # x: (B, T, d), b_dec: (T, d) -> broadcast to (B, T, d).
        x_centered = x - self.b_dec  # (B, T, d)

        # Per-position encode in one fused op:
        # (B, T, d) x (T, d, m) -> (B, T, m). Position t uses W_enc[t] only.
        pre = torch.einsum("btd,tdm->btm", x_centered, self.W_enc) + self.b_enc
        pre = F.relu(pre)

        # Per-position TopK on the last axis. Position t's top-k sees only
        # its own m latents — disjoint across t.
        _, topk_idx = pre.topk(self.k, dim=-1)  # (B, T, k)
        mask = torch.zeros_like(pre)
        mask.scatter_(-1, topk_idx, 1.0)
        z = pre * mask  # (B, T, m)

        # Per-position decode:
        # (B, T, m) x (T, m, d) -> (B, T, d). Position t uses W_dec[t] only.
        x_hat = torch.einsum("btm,tmd->btd", z, self.W_dec) + self.b_dec

        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        l0 = (z != 0).float().sum(dim=-1).mean().item()

        return ModelOutput(
            x_hat=x_hat,
            latents=z,
            loss=recon_loss,
            metrics={"recon_loss": recon_loss.item(), "l0": l0},
        )

    def decoder_directions(self, pos: int | None = None) -> torch.Tensor:
        """(d, m) decoder columns.

        pos=None averages per-position decoders into a single matrix,
        matching the convention used by TemporalCrosscoder on this branch.
        pos=t returns the t-th position's decoder directly.
        """
        if pos is not None:
            return self.W_dec[pos].T  # (d, m)
        return self.W_dec.mean(dim=0).T  # (d, m)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()

    @property
    def n_positions(self) -> int:
        return self.T
