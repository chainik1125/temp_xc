"""TXC-base per-position — same encoder, per-position TopK.

Identical encoder/decoder weight shapes to :class:`TXCBase`
(``W_enc: (T, d_in, d_sae)``) but the TopK budget is applied PER POSITION
instead of jointly across the window. Each of the T positions gets its
own k_pos atoms; the code is ``(B, T, d_sae)`` with k_pos non-zeros per
``(b, t)``.

**Motivation.** ``txc_base`` with ``T = W`` shows window-size degradation
in the linear probe (NTPS ≈ 0.17 at W=16, d_sae=1024) while the encoder's
``FreqFrac`` climbs to ≈ 0.88 — the trained atoms ARE order-sensitive, the
joint TopK just collapses positional structure into one shared sparse
code before the probe sees it. This variant removes the joint pool: each
position is encoded independently with its own basis ``W_enc[t]`` and
its own k_pos budget, so the per-position assignment is preserved in the
code itself. Same parameter count, same training recipe; only the TopK
axis changes.

See ``experiments/freq_bench/freqfrac_diagnostic.py`` for the
representation-vs-readout diagnostic that motivated this arch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


class TXCBasePerPos(TempBenchArch):
    arch_version: str = "1.0.0"
    consumes: str = "window"

    def __init__(self, *, d_in: int, d_sae: int, T: int = 5, k_pos: int = 20,
                 **_ignore):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name="txc_base_perpos", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T,
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        self.k_pos = min(k_pos, d_sae)

        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(T, d_sae))
        self.W_dec = nn.Parameter(torch.empty(T, d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc.data[t])
            nn.init.kaiming_uniform_(self.W_dec.data[t])
        with torch.no_grad():
            self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        """Per-position, per-atom decoder unit norm."""
        norms = self.W_dec.norm(dim=2, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) → (B, T, d_sae) — per-position TopK code."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self._T:
            raise ValueError(
                f"TXCBasePerPos.encode expects (B, T={self._T}, d_in); "
                f"got T={x.shape[1]}."
            )
        pre = torch.einsum("btd,tds->bts", x, self.W_enc) + self.b_enc.unsqueeze(0)
        vals, idx = pre.topk(self.k_pos, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, F.relu(vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bts,tsd->btd", z, self.W_dec) + self.b_dec.unsqueeze(0)

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"TXCBasePerPos.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        z = self.encode(x)
        x_hat = self.decode(z)
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        with torch.no_grad():
            l0 = (z != 0).float().sum(dim=-1).mean()
        return {"loss": loss, "mse": loss.detach(), "l0": l0.detach()}

    def post_step(self) -> None:
        with torch.no_grad():
            self._normalize_decoder()

    def decoder_directions(self) -> torch.Tensor:
        return self.W_dec.data.mean(dim=0).clone()
