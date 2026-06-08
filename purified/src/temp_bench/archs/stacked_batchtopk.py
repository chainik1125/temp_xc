"""Stacked BatchTopK SAE — T independent per-position dicts, strong backbone.

Fair-backbone redo (``autoresearch/STATUS.md`` § 4). The per-position
("stacked") baseline on the BatchTopK backbone: ``T`` independent
dictionaries (one per window slot, no weight sharing, independent
decode), trained with BatchTopK → JumpReLU threshold (Bussmann et al.) +
AuxK + decoder unit-norm + grad-parallel removal. Replaces the plain TopK
``stacked_sae`` so the only thing separating it from the crosscoders is
the decode structure (independent per-position vs one shared window code).

BatchTopK pools globally over the ``B·T`` (window, position) activations
with budget ``k_pos`` per position (``k_pos · B · T`` actives) — the same
pool granularity as the per-token archs once throughput is normalised
(``B = base // T`` ⇒ ``B·T = base``). One global JumpReLU threshold is
applied at inference. Dead-feature tracking + AuxK are per-position, since
each slot owns an independent dictionary.

``consumes = "window"``: the trainer feeds ``(B, T, d_in)`` windows from
:class:`WindowBuffer`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


class StackedBatchTopK(TempBenchArch):
    """T independent per-position BatchTopK SAEs (one dict per window slot)."""

    arch_version: str = "1.0.0"
    consumes: str = "window"

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        T: int = 4,
        k_pos: int = 1,
        auxk_alpha: float = 1.0 / 32.0,
        dead_threshold_tokens: int = 10_000_000,
        aux_k: int | None = None,
        threshold_start_step: int = 1000,
        threshold_beta: float = 0.999,
    ):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name="stacked_batchtopk", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T,
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

        # Per-position params (T independent dicts).
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

        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    # ── decoder-norm utilities (per-position) ──

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        # W_dec (T, d_sae, d_in): unit-norm each feature's decoder vector.
        norms = self.W_dec.norm(dim=2, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    @staticmethod
    def _project_dec_grad(param: torch.Tensor) -> None:
        if param.grad is None:
            return
        # param (T, d_sae, d_in): project grad parallel to each decoder vector.
        W = param.data
        g = param.grad.data
        normed = W / (W.norm(dim=2, keepdim=True) + 1e-6)
        parallel = (g * normed).sum(dim=2, keepdim=True)          # (T, d_sae, 1)
        g.sub_(parallel * normed)

    def _batchtopk(self, post: torch.Tensor) -> torch.Tensor:
        """Flat BatchTopK over (B·T·d_sae); budget k_pos per position."""
        n_rows = post.reshape(-1, self._d_sae).shape[0]           # B·T
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

    # ── encode / decode ──

    def _post(self, x: torch.Tensor) -> torch.Tensor:
        """Per-position ReLU pre-activations ``(B, T, d_sae)``."""
        return F.relu(torch.einsum("btd,tds->bts", x, self.W_enc) + self.b_enc)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) → (B, T, d_sae) per-position codes."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self._T:
            raise ValueError(
                f"StackedBatchTopK.encode expects (B, T={self._T}, d_in); "
                f"got T={x.shape[1]}."
            )
        post = self._post(x)
        if (not self.training) and self.threshold.item() >= 0:
            return post * (post > self.threshold)
        return self._batchtopk(post)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, T, d_sae) → (B, T, d_in) per-position decode."""
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if z.shape[1] != self._T:
            raise ValueError(
                f"StackedBatchTopK.decode expects (B, T={self._T}, d_sae); "
                f"got T={z.shape[1]}."
            )
        return torch.einsum("bts,tsd->btd", z, self.W_dec) + self.b_dec

    # ── train_step ──

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"StackedBatchTopK.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, T, _d_in = x.shape

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
                if self.threshold.item() < 0:
                    self.threshold.copy_(cur)
                else:
                    self.threshold.copy_(
                        self.threshold_beta * self.threshold
                        + (1 - self.threshold_beta) * cur
                    )

        x_hat = torch.einsum("bts,tsd->btd", z, self.W_dec) + self.b_dec
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        with torch.no_grad():
            did_fire = (z > 0).any(dim=0)                         # (T, d_sae)
            self.num_tokens_since_fired += B
            self.num_tokens_since_fired[did_fire] = 0
            dead = self.num_tokens_since_fired >= self.dead_threshold_tokens
            n_dead = int(dead.sum().item())

        if n_dead > 0:
            k_aux = min(self.aux_k, n_dead, self._d_sae)
            auxk_pre = post.masked_fill(~dead.unsqueeze(0), 0.0)  # (B, T, d_sae)
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

        return {
            "loss": loss,
            "mse": l_recon.detach(),
            "l0": l0.detach(),
            "auxk": l_auxk.detach(),
            "dead": torch.tensor(float(n_dead)),
            "threshold": self.threshold.detach().clone(),
        }

    def post_step(self) -> None:
        with torch.no_grad():
            self._normalize_decoder()

    def decoder_directions(self) -> torch.Tensor:
        """(d_sae, d_in) — average per-position decoders over T."""
        return self.W_dec.data.mean(dim=0).clone()
