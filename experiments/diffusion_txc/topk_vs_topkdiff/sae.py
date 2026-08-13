"""TopK SAE with AuxK, Gao-et-al-faithful, shared by both arms.

x' = x - b_dec;  z = TopK_k(ReLU(W_enc x' + b_enc));  xhat = W_dec z + b_dec.
AuxK: the residual (x - xhat) is additionally reconstructed using the
top-k_aux *dead* latents' preactivations; coefficient 1/32. Decoder columns
renormalized to unit norm after each optimizer step.
"""

from __future__ import annotations

import torch
from torch import nn


class TopKSAE(nn.Module):
    def __init__(self, d: int, H: int, k: int, k_aux: int = 512,
                 aux_coef: float = 1 / 32, dead_window: int = 800):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d, H) / d**0.5)
        self.b_enc = nn.Parameter(torch.zeros(H))
        self.W_dec = nn.Parameter(self.W_enc.data.T.clone())
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.k, self.k_aux, self.aux_coef = k, k_aux, aux_coef
        self.dead_window = dead_window
        self.register_buffer("steps_since_fire", torch.zeros(H))
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encode_pre(x)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        return z.scatter(-1, idx, torch.relu(vals))

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None,
                update_fires: bool = False):
        """Returns (loss, recon, z). `target` defaults to x (recon arm);
        the DSM arm passes the clean target with a corrupted x."""
        if target is None:
            target = x
        pre = self.encode_pre(x)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z = z.scatter(-1, idx, torch.relu(vals))
        recon = z @ self.W_dec + self.b_dec
        loss = (recon - target).pow(2).mean()

        if update_fires:
            with torch.no_grad():
                fired = torch.zeros_like(pre, dtype=torch.bool)
                fired.scatter_(-1, idx, torch.relu(vals) > 0)
                self.steps_since_fire += 1
                self.steps_since_fire[fired.any(dim=0)] = 0

        dead = self.steps_since_fire > self.dead_window
        if dead.any() and self.k_aux > 0:
            resid = (target - recon).detach()
            pre_dead = pre.masked_fill(~dead.unsqueeze(0), float("-inf"))
            k_aux = min(self.k_aux, int(dead.sum()))
            a_vals, a_idx = pre_dead.topk(k_aux, dim=-1)
            z_aux = torch.zeros_like(pre)
            z_aux = z_aux.scatter(-1, a_idx, torch.relu(a_vals))
            aux_recon = z_aux @ self.W_dec
            aux_loss = (aux_recon - resid).pow(2).mean()
            if torch.isfinite(aux_loss):
                loss = loss + self.aux_coef * aux_loss

        return loss, recon, z

    @torch.no_grad()
    def dead_fraction(self) -> float:
        return float((self.steps_since_fire > self.dead_window).float().mean())


def flops_per_token(d: int, H: int) -> float:
    """Training FLOPs/token for the two matmuls, fwd + ~2x bwd."""
    return 12.0 * d * H
