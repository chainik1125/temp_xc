"""Stacked BatchTopK (btk-only) with pooled window codes — Wang-eval port.

Single-file port of the arxiv-branch ``stacked_batchtopk`` →
``btk_only.StackedBatchTopKBTKOnly`` → ``stacked_pooled.StackedBTKOnlyPooled``
stack (branch ``dmitry-stacked-arxiv``), kept **state-dict compatible** with
the checkpoint trained there for the C6 stacked cell
(HF ``dmanningcoe/stacked-sae-rebuttal-2026-07/c6_em/checkpoints/8b8231508a1ce6e3``):
parameters ``W_enc (T, d_in, d_sae)``, ``b_enc (T, d_sae)``,
``W_dec (T, d_sae, d_in)``, ``b_dec (T, d_in)`` and buffers ``threshold``,
``num_tokens_since_fired``, ``global_step``, ``threshold_set``.

Semantics: T independent per-position dictionaries; BatchTopK pools over the
``B·T`` (window, position) pre-activations with budget ``k_pos`` per position;
btk-only (no ReLU on pre-activations, threshold validity via the
``threshold_set`` flag). ``encode`` returns one pooled ``(B, d_sae)`` window
code — per-feature value of max-|activation| over positions, sign kept —
matching the App-A eval reduction; per-position codes stay reachable via
:meth:`encode_per_position` for mining.

Steering note: decoder rows are unit-norm per position (post_step renorm).
No sqrt(T) magnitude rescale applies to this architecture — see
``case_studies.em.decoder_row``, which special-cases the (T, d_sae, d_in)
layout via the ``encode_per_position`` marker attribute.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from temp_bench.architectures.base import ArchConfig, TempBenchArch


class StackedBTKOnlyPooled(TempBenchArch):
    """T independent per-position BatchTopK dicts, pooled window code."""

    arch_version: str = "1.2.0"
    consumes: str = "window"

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        T: int = 5,
        k_pos: int = 25,
        auxk_alpha: float = 1.0 / 32.0,
        dead_threshold_tokens: int = 10_000_000,
        aux_k: int | None = None,
        threshold_start_step: int = 1000,
        threshold_beta: float = 0.999,
        relu_mode: str = "btk-only",
    ):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name="stacked_btkonly_pooled", d_in=d_in, d_sae=d_sae,
            k_pos=k_pos, T=T,
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        self.k_pos = int(k_pos)
        self.auxk_alpha = auxk_alpha
        self.dead_threshold_tokens = dead_threshold_tokens
        self.aux_k = aux_k if aux_k is not None else min(512, d_in // 2)
        self.threshold_start_step = threshold_start_step
        self.threshold_beta = threshold_beta
        self.relu_mode = relu_mode

        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(T, d_sae))
        self.W_dec = nn.Parameter(torch.empty(T, d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))
        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc.data[t])
            nn.init.kaiming_uniform_(self.W_dec.data[t])
        with torch.no_grad():
            self._normalize_decoder()
            for t in range(T):
                self.W_enc.data[t] = self.W_dec.data[t].T

        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.register_buffer(
            "num_tokens_since_fired", torch.zeros(T, d_sae, dtype=torch.long)
        )
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        # btk-only: explicit threshold-validity flag (threshold may be < 0).
        self.register_buffer("threshold_set", torch.tensor(0, dtype=torch.uint8))

        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    # ── decoder-norm utilities ──

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=2, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    @staticmethod
    def _project_dec_grad(param: torch.Tensor) -> None:
        if param.grad is None:
            return
        W = param.data
        g = param.grad.data
        normed = W / (W.norm(dim=2, keepdim=True) + 1e-6)
        parallel = (g * normed).sum(dim=2, keepdim=True)
        g.sub_(parallel * normed)

    # ── encode / decode ──

    def _post(self, x: torch.Tensor) -> torch.Tensor:
        # btk-only: raw per-position pre-activations, no ReLU.
        return torch.einsum("btd,tds->bts", x, self.W_enc) + self.b_enc

    def _batchtopk(self, post: torch.Tensor) -> torch.Tensor:
        n_rows = post.reshape(-1, self._d_sae).shape[0]
        k_total = self.k_pos * n_rows
        flat = post.reshape(-1)
        if k_total >= flat.numel():
            return post
        tk = flat.topk(k_total, sorted=False)
        return (
            torch.zeros_like(flat)
            .scatter_(-1, tk.indices, tk.values)
            .reshape(post.shape)
        )

    def encode_per_position(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) → (B, T, d_sae) native per-position codes."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self._T:
            raise ValueError(
                f"StackedBTKOnlyPooled expects (B, T={self._T}, d_in); "
                f"got T={x.shape[1]}."
            )
        post = self._post(x)
        if (not self.training) and bool(self.threshold_set.item()):
            return post * (post > self.threshold)
        return self._batchtopk(post)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) → (B, d_sae): per-feature max-|act|, sign kept."""
        z = self.encode_per_position(x)
        idx = z.abs().argmax(dim=1, keepdim=True)
        return z.gather(1, idx).squeeze(1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, d_sae) → (B, T, d_in) per-position decode."""
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if z.shape[1] != self._T:
            raise ValueError(
                f"StackedBTKOnlyPooled.decode expects (B, T={self._T}, d_sae); "
                f"got T={z.shape[1]}."
            )
        return torch.einsum("bts,tsd->btd", z, self.W_dec) + self.b_dec

    # ── training (purified tuple convention) ──

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"StackedBTKOnlyPooled.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B = x.shape[0]
        post = self._post(x)
        z = self._batchtopk(post)

        step = int(self.global_step.item())
        if step > self.threshold_start_step:
            with torch.no_grad():
                active = z[z > 0]
                cur = (
                    active.min().float()
                    if active.numel() > 0
                    else torch.tensor(0.0, device=z.device)
                )
                if not bool(self.threshold_set.item()):
                    self.threshold.copy_(cur)
                    self.threshold_set.fill_(1)
                else:
                    self.threshold.copy_(
                        self.threshold_beta * self.threshold
                        + (1 - self.threshold_beta) * cur
                    )

        x_hat = torch.einsum("bts,tsd->btd", z, self.W_dec) + self.b_dec
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        with torch.no_grad():
            did_fire = (z > 0).any(dim=0)
            self.num_tokens_since_fired += B
            self.num_tokens_since_fired[did_fire] = 0
            dead = self.num_tokens_since_fired >= self.dead_threshold_tokens
            n_dead = int(dead.sum().item())

        if n_dead > 0:
            k_aux = min(self.aux_k, n_dead, self._d_sae)
            auxk_pre = post.masked_fill(~dead.unsqueeze(0), 0.0)
            vals, idx = auxk_pre.topk(k_aux, dim=-1, sorted=False)
            aux_buf = torch.zeros_like(post).scatter_(-1, idx, vals)
            aux_decode = torch.einsum("bts,tsd->btd", aux_buf, self.W_dec)
            residual = (x - x_hat).detach()
            l2 = (residual - aux_decode).pow(2).sum(dim=-1).mean()
            mu = residual.mean(dim=(0, 1), keepdim=True)
            denom = (residual - mu).pow(2).sum(dim=-1).mean()
            l_auxk = (l2 / denom.clamp(min=1e-8)).nan_to_num(0.0)
        else:
            l_auxk = torch.zeros((), device=x.device, dtype=x.dtype)

        loss = l_recon + self.auxk_alpha * l_auxk
        with torch.no_grad():
            self.global_step += 1
            l0 = (z != 0).float().sum(dim=-1).mean()
        return loss, {
            "mse": l_recon.detach(), "l0": l0.detach(),
            "auxk": l_auxk.detach(), "dead": torch.tensor(float(n_dead)),
        }

    def post_step(self) -> None:
        with torch.no_grad():
            self._normalize_decoder()

    @property
    def d_sae(self) -> int:
        return self._d_sae

    @property
    def T(self) -> int:
        return self._T

    def decoder_directions(self) -> torch.Tensor:
        """(d_sae, d_in) — per-position decoders averaged over T."""
        return self.W_dec.data.mean(dim=0).clone()
