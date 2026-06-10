"""SAE-arditi — plain TopK SAE in the sae_day / arditi convention.

The § 5.3 (emergent misalignment) baseline. Per-token TopK sparse
autoencoder, trained with a single MSE reconstruction term on
streaming Qwen-14B activations. No AuxK, no Bricken, no contrastive —
the simplest baseline against which the brickenauxk-trained TXC-base
re-test is measured.

Why a separate class from the framework's
:class:`temp_bench.architectures.topk_sae.TopKSAE`: this class matches
the parameter layout used by Dmitry's ``origin/em-nanda`` checkpoints
(``sae_day.TopKSAE``):

    W_enc : (d_in, d_sae)
    b_enc : (d_sae,)
    W_dec : (d_sae, d_in)
    b_dec : (d_in,)

The framework's :class:`TopKSAE` uses the transposed convention
(``W_enc : (d_sae, d_in)``, ``W_dec : (d_in, d_sae)``). Diverging only
on the layout means Dmitry's HF checkpoints
(``han1823123123/temp-bench-models``, originally trained under
``sae_day.TopKSAE``) load directly via
``model.load_state_dict(ckpt["state_dict"])`` with no transpose dance.

Forward path:

    pre = (x - b_dec) @ W_enc + b_enc
    z   = TopK(ReLU(pre), k)        — `use_relu=True` matches dictionary_learning
    x_hat = z @ W_dec + b_dec

Decoder atoms are unit-normalised at init and after every optimizer
step. No AuxK / no anti-dead because the arditi recipe didn't include
them — keeping fidelity to the published baseline matters more than
raw performance for an apples-to-apples compare against TXC-base
+brickenauxk.

Ported from
``origin/em-nanda @ a74d1be4:experiments/separation_scaling/vendor/src/sae_day/sae.py``
(class ``TopKSAE``).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


class SAEArditi(TempBenchArch):
    """Per-token TopK SAE in the sae_day / arditi state-dict convention.

    Shapes:

        x:     (B, T, d_in)  or  (B, d_in)   — windowed input or per-token
        z:     (B, T, d_sae) or  (B, d_sae)
        x_hat: same as ``x``

    For C6 the Wang procedure reads the per-token activation at the
    layer-24 ``resid_post`` hookpoint, so callers usually pass ``T=1``
    (squeezed away on output) or pass a flat ``(B, d_in)`` directly.
    """


    # v2 framework attrs (added during arxiv migration).
    # consumes='token': v1 port: per-token SAE for § 5.3 EM
    arch_version: str = "2.0.0"
    consumes: str = 'token'

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        k_pos: int = 128,
        T: int = 1,
        use_relu: bool = True,
        **_unused,  # forward-compat with locked-yaml additions
    ):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name="sae_arditi", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T,
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self.k = k_pos
        self.use_relu = use_relu

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

    # ── Internals ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    # ── Public TempBenchArch interface ─────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_t = x.dim() == 2
        if squeeze_t:
            x = x.unsqueeze(1)
        B, T, d = x.shape
        flat = x.reshape(B * T, d)
        pre = (flat - self.b_dec) @ self.W_enc + self.b_enc
        if self.use_relu:
            pre = F.relu(pre)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z_flat = torch.zeros_like(pre)
        z_flat.scatter_(-1, topk_idx, topk_vals)
        z = z_flat.reshape(B, T, self._d_sae)
        if squeeze_t:
            z = z.squeeze(1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_t = z.dim() == 2
        if squeeze_t:
            z = z.unsqueeze(1)
        x_hat = z @ self.W_dec + self.b_dec
        if squeeze_t:
            x_hat = x_hat.squeeze(1)
        return x_hat

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Per-arditi: plain MSE recon, no AuxK, no anti-dead. Accept
        # either (B, T, d_in) or (B, seq_len, d_in) — for SAE we treat
        # every (batch, position) as an independent token and don't
        # care which position.
        z = self.encode(x)
        x_hat = self.decode(z)
        mse = (x - x_hat).pow(2).sum(dim=-1).mean()
        z_flat = z.reshape(-1, z.shape[-1])
        l0 = (z_flat != 0).float().sum(dim=-1).mean()
        return mse, {
            "mse": mse.detach(),
            "l0": l0.detach(),
            "z": z.detach(),
        }

    def post_step(self) -> None:
        self._normalize_decoder()

    def decoder_directions(self) -> torch.Tensor:
        """``(d_sae, d_in)`` — per-feature decoder direction. Matches
        the natural ``W_dec`` layout for this arch."""
        return self.W_dec.data.clone()
