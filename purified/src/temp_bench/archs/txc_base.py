"""TXC-base — vanilla TopK temporal crosscoder + tsae_paper anti-dead stack.

Adapted for v2 framework from ``origin/final:purified/src/temp_bench/architectures/txc_base.py``.
Original source: ``origin/han-phase7-unification @ 94119bc0:src/architectures/txc_bare_antidead.py``
("Phase 6.1 Track 2: agentic_txc_10_bare").

Key v2 adaptations:
- Subclasses :class:`temp_bench.interfaces.architecture.TempBenchArch`.
- ``consumes = "window"`` — trainer feeds (B, T, d_in) directly from
  :class:`WindowBuffer`. The arch NO LONGER samples windows internally;
  v1's "1-random-window-per-sequence" path is replaced by the buffer's
  random-window sampling (closer to i.i.d.).
- ``train_step(x)`` returns ``dict[str, Tensor]`` with ``"loss"`` key
  (was ``(loss, info)`` tuple in v1).

Architecture:
- Encoder: ``W_enc (T, d_in, d_sae)``; sums over T into ``(B, d_sae)``.
- Decoder: ``W_dec (d_sae, T, d_in)``; one window-level latent
  reconstructs the full T-window.
- Sparsity: window-level TopK, k_win = k_pos * T.
- Anti-dead stack: AuxK loss on dead features, decoder unit-norm,
  grad-parallel removal via post-accumulate hook.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


class TXCBase(TempBenchArch):
    """Vanilla TopK temporal crosscoder + tsae_paper anti-dead stack."""

    arch_version: str = "2.0.0"     # major bump: v2 contract change (no in-arch sampling)
    consumes: str = "window"

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        T: int = 5,
        k_pos: int = 20,
        auxk_alpha: float = 1.0 / 32.0,
        dead_threshold_tokens: int = 10_000_000,
        aux_k: int | None = None,
        decoder_unit_norm: bool = True,           # noqa: ARG002 — always True in this port
        decoder_grad_orthogonalize: bool = True,  # noqa: ARG002
        bdec_geom_median_init: bool = True,       # noqa: ARG002 — see legacy notes
    ):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name="txc_base", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T,
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        self.k_pos = k_pos
        self.k_win = k_pos * T
        self.auxk_alpha = auxk_alpha
        self.dead_threshold_tokens = dead_threshold_tokens
        self.aux_k = aux_k if aux_k is not None else min(512, d_in // 2)

        # Params
        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, T, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        # Init
        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc.data[t])
        nn.init.kaiming_uniform_(self.W_dec.data.view(d_sae, T * d_in))
        with torch.no_grad():
            self._normalize_decoder()
            for t in range(T):
                self.W_enc.data[t] = self.W_dec.data[:, t, :].T

        # Dead-feature tracker
        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(d_sae, dtype=torch.long),
        )

        # Grad-parallel removal
        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    # ── Decoder norm utilities ──

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    @staticmethod
    def _project_dec_grad(param: torch.Tensor) -> None:
        if param.grad is None:
            return
        d_sae = param.shape[0]
        W_flat = param.data.view(d_sae, -1)
        g_flat = param.grad.data.view(d_sae, -1)
        normed = W_flat / (W_flat.norm(dim=1, keepdim=True) + 1e-6)
        parallel = (g_flat * normed).sum(dim=1, keepdim=True)
        g_flat.sub_(parallel * normed)

    # ── encode / decode ──

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) → (B, 1, d_sae) — one window-level latent.

        For consistency with the legacy (B, d_in) → (B, d_sae) per-token
        contract, accept (B, d_in) too (treats T=1) — used by some evals.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self._T:
            raise ValueError(
                f"TXCBase.encode expects (B, T={self._T}, d_in); got T={x.shape[1]}."
            )
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        vals, idx = pre.topk(self.k_win, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z.unsqueeze(1)               # (B, 1, d_sae)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 1, d_sae) or (B, d_sae) → (B, T, d_in)."""
        if z.dim() == 3:
            if z.shape[1] != 1:
                raise ValueError(
                    f"TXCBase.decode expects (B, 1, d_sae); got T={z.shape[1]}."
                )
            z = z.squeeze(1)
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    # ── train_step ──

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """One step. Input: (B, T, d_in) windows direct from WindowBuffer.

        Returns dict with at least ``"loss"`` (scalar tensor, for backward).
        """
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"TXCBase.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, T, d_in = x.shape
        n_tokens_per_step = B * T

        # Encode
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        vals, idx = pre.topk(self.k_win, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))

        # Reconstruction
        x_hat = torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        # Dead tracker
        with torch.no_grad():
            active = (z > 0).any(dim=0)
            self.num_tokens_since_fired += n_tokens_per_step
            self.num_tokens_since_fired[active] = 0
            dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
            n_dead = int(dead_mask.sum().item())

        # AuxK on dead features
        if n_dead > 0:
            k_aux = min(self.aux_k, n_dead)
            auxk_pre = F.relu(pre).masked_fill(~dead_mask.unsqueeze(0), 0.0)
            vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
            aux_buf = torch.zeros_like(pre)
            aux_buf.scatter_(-1, idx_a, vals_a)
            aux_decode = torch.einsum("bs,std->btd", aux_buf, self.W_dec)
            residual = (x - x_hat).detach()
            l2_a = (residual - aux_decode).pow(2).sum(dim=-1).mean()
            mu = residual.mean(dim=(0, 1), keepdim=True)
            denom = (residual - mu).pow(2).sum(dim=-1).mean()
            l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        else:
            l_auxk = torch.zeros((), device=x.device, dtype=x.dtype)

        loss = l_recon + self.auxk_alpha * l_auxk

        with torch.no_grad():
            l0 = (z != 0).float().sum(dim=-1).mean()

        return {
            "loss": loss,
            "mse": l_recon.detach(),
            "l0": l0.detach(),
            "auxk": l_auxk.detach(),
            "dead": torch.tensor(float(n_dead)),
        }

    # ── post_step (decoder unit-norm projection) ──

    def post_step(self) -> None:
        with torch.no_grad():
            self._normalize_decoder()

    # ── decoder_directions for synthetic eval ──

    def decoder_directions(self) -> torch.Tensor:
        """(d_sae, d_in) — average decoder over T positions."""
        return self.W_dec.data.mean(dim=1).clone()
