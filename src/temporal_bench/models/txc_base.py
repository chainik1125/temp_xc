"""TXC-base — Han's locked vanilla TopK temporal crosscoder + anti-dead stack.

Port of ``purified/src/temp_bench/architectures/txc_base.py`` (origin/final)
into Bill's ``TemporalAE`` interface. Faithful copy of the architectural
details (per-position W_enc/W_dec, window-level TopK with k_win=k_pos*T,
AuxK on dead features, decoder unit-norm, decoder-grad-parallel removal,
tied-encoder init), with two interface differences:

- ``forward(x: (B, T, d)) -> ModelOutput`` instead of separate
  ``train_step`` / ``post_step``. The AuxK term is computed inside
  ``forward`` so Bill's generic trainer can backprop through it.
- ``normalize_decoder()`` is a public method called by Bill's trainer
  after ``optimizer.step()``.

Locked params (configs/locked_archs.yaml::txc_base):
    d_sae = 8 * d_in (paper expansion)
    T = 5
    k_pos = 20  -> k_win = 100
    auxk_alpha = 1/32
    dead_threshold_tokens = 10_000_000
    decoder_unit_norm = True
    decoder_grad_orthogonalize = True
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutput, TemporalAE


class TXCBase(TemporalAE):
    """Vanilla TopK temporal crosscoder + tsae_paper anti-dead stack.

    Args:
        d_in: residual width.
        d_sae: dictionary size (Han default = 8 * d_in).
        T: window length (Han default = 5).
        k_pos: per-token sparsity. Window-level L0 = k_pos * T.
        auxk_alpha: AuxK loss weight (Han default 1/32).
        dead_threshold_tokens: tokens-since-fired before AuxK targets.
        aux_k: budget of dead features per-sample in AuxK loss.
            Defaults to ``min(512, d_in // 2)``.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int = 5,
        k_pos: int = 20,
        auxk_alpha: float = 1.0 / 32.0,
        dead_threshold_tokens: int = 10_000_000,
        aux_k: int | None = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k_pos = k_pos
        self.k_win = k_pos * T
        self.auxk_alpha = auxk_alpha
        self.dead_threshold_tokens = dead_threshold_tokens
        self.aux_k = aux_k if aux_k is not None else min(512, d_in // 2)

        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, T, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc.data[t])
        nn.init.kaiming_uniform_(self.W_dec.data.view(d_sae, T * d_in))
        with torch.no_grad():
            self._normalize_decoder_inplace()
            for t in range(T):
                self.W_enc.data[t] = self.W_dec.data[:, t, :].T

        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(d_sae, dtype=torch.long),
        )

        # Pre-step grad-parallel removal as a tensor hook on W_dec.
        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    @torch.no_grad()
    def _normalize_decoder_inplace(self) -> None:
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

    def forward(self, x: torch.Tensor) -> ModelOutput:
        B, T, d = x.shape
        assert T == self.T, f"TXCBase expects T={self.T}, got T={T}"
        assert d == self.d_in

        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        vals, idx = pre.topk(self.k_win, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))

        x_hat = torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        with torch.no_grad():
            active_mask = (z > 0).any(dim=0)
            n_tokens = B * T
            self.num_tokens_since_fired += n_tokens
            self.num_tokens_since_fired[active_mask] = 0
            dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
            n_dead = int(dead_mask.sum().item())

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

        # Bill's evaluator expects (B, T, d_sae) latents; broadcast the
        # shared window-level latent across positions.
        latents = z.unsqueeze(1).expand(B, T, self.d_sae)
        l0 = (z != 0).float().sum(dim=-1).mean().item()

        return ModelOutput(
            x_hat=x_hat,
            latents=latents,
            loss=loss,
            metrics={
                "recon_loss": l_recon.item(),
                "auxk_loss": float(l_auxk.detach()),
                "dead": n_dead,
                "l0": l0,
            },
        )

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder_inplace()

    def decoder_directions(self, pos: int | None = None) -> torch.Tensor:
        # W_dec: (d_sae, T, d_in) -> directions returned as (d_in, d_sae).
        if pos is not None:
            return self.W_dec.data[:, pos, :].T  # (d_in, d_sae)
        return self.W_dec.data.mean(dim=1).T

    @property
    def n_positions(self) -> int:
        return self.T
