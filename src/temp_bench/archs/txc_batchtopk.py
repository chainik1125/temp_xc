"""BatchTopK temporal crosscoders — pre- and post-squash variants.

Fair-backbone redo (``synthetic/STATUS.md`` § 4). Every arch in the
backtracking comparison now shares the strong **BatchTopK** backbone
(Bussmann et al.: BatchTopK during *training* → a fixed **JumpReLU
threshold** at *inference*) plus AuxK dead-feature revival, decoder
unit-norm, and grad-parallel removal. With the backbone held constant the
*only* variable left is the decode structure — here, the temporal
crosscoder's one shared ``(d_sae)`` window code that reconstructs all ``T``
positions via the per-position decoders ``W_dec[:, t, :]``.

Two variants differ **only in where BatchTopK is applied**:

- ``TXCBatchTopKPre`` (**pre-squash**): BatchTopK on the *per-position*
  pre-activations (pool = ``B·T`` tokens, budget ``k_pos`` PER TOKEN), then
  the surviving per-position activations are **summed** into the shared
  window code. The shared code's support is the union of the per-position
  selections (≤ ``k_pos·T``). Zeroing happens before the squash, yet a
  single shared code still reconstructs every position — a genuine
  crosscoder.

- ``TXCBatchTopKPost`` (**post-squash**): **sum** the per-position
  pre-activations into the squashed code first, then BatchTopK on that
  squashed code (pool = ``B`` windows, budget ``k_pos`` PER WINDOW =
  ``k_win // T``). Each squashed atom is reused at all ``T`` positions, so
  ``k_pos`` shared atoms ≈ ``k_pos·T`` token-activations — parity with the
  per-token archs, and it corrects the legacy ``k_win = k_pos·T``
  over-count.

The resulting density gap (post support ``k_pos`` vs pre support up to
``k_pos·T`` at the same per-token selection budget) *is* the pre-vs-post
effect we measure; NMSE / eAUC / λ are reported outcomes, not equalised.

Both variants reuse the ``txc_base`` parameterisation — ``W_enc
(T,d_in,d_sae)``, ``W_dec (d_sae,T,d_in)`` and the einsum squash.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


class _TXCBatchTopKBase(TempBenchArch):
    """Shared machinery for the BatchTopK temporal crosscoders.

    Subclasses implement two hooks that capture the pre-vs-post squash
    distinction:

    - ``_compute_post(x)`` → the ReLU pre-activations BatchTopK / the
      threshold operate on. Pre-squash: ``(B, T, d_sae)`` per-position;
      post-squash: ``(B, d_sae)`` on the squashed code.
    - ``_to_shared(gated)`` → reduce the gated activations to the shared
      window code ``(B, d_sae)``. Pre-squash sums over ``T``; post-squash
      is identity.
    """

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
            name=self._registry_name, d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T,
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

        # Params (txc_base convention).
        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, T, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        # Init: kaiming decoder, unit-norm, tie encoder = decoder transpose.
        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc.data[t])
        nn.init.kaiming_uniform_(self.W_dec.data.view(d_sae, T * d_in))
        with torch.no_grad():
            self._normalize_decoder()
            for t in range(T):
                self.W_enc.data[t] = self.W_dec.data[:, t, :].T

        # BatchTopK → JumpReLU threshold + dead-feature tracker.
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.register_buffer(
            "num_tokens_since_fired", torch.zeros(d_sae, dtype=torch.long)
        )
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    # ── subclass hooks ──

    _registry_name: str = "txc_batchtopk"

    def _compute_post(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        raise NotImplementedError

    def _to_shared(self, gated: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # ── decoder-norm utilities (from txc_base) ──

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

    # ── BatchTopK selection ──

    def _batchtopk(self, post: torch.Tensor) -> torch.Tensor:
        """Flat BatchTopK over the selection pool.

        ``post`` is ``(B, T, d_sae)`` (pre-squash, pool ``B·T``) or
        ``(B, d_sae)`` (post-squash, pool ``B``). Budget = ``k_pos`` per
        pool row, i.e. ``k_pos · n_rows`` actives globally.
        """
        n_rows = post.reshape(-1, self._d_sae).shape[0]
        k_total = self.k_pos * n_rows
        flat = post.reshape(-1)
        if k_total >= flat.numel():
            return post
        tk = flat.topk(k_total, sorted=False)
        gated = (
            torch.zeros_like(flat)
            .scatter_(-1, tk.indices, tk.values)
            .reshape(post.shape)
        )
        return gated

    def _squashed_preact(self, x: torch.Tensor) -> torch.Tensor:
        """Squashed pre-activation ``(B, d_sae)`` — shared-code space for AuxK."""
        return torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc

    # ── encode / decode ──

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) → (B, 1, d_sae) shared window code.

        BatchTopK while training (or before the threshold is tracked);
        JumpReLU threshold at inference.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self._T:
            raise ValueError(
                f"{type(self).__name__}.encode expects (B, T={self._T}, d_in); "
                f"got T={x.shape[1]}."
            )
        post = self._compute_post(x)
        if (not self.training) and self.threshold.item() >= 0:
            gated = post * (post > self.threshold)
        else:
            gated = self._batchtopk(post)
        return self._to_shared(gated).unsqueeze(1)        # (B, 1, d_sae)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 1, d_sae) or (B, d_sae) → (B, T, d_in)."""
        if z.dim() == 3:
            if z.shape[1] != 1:
                raise ValueError(
                    f"{type(self).__name__}.decode expects (B, 1, d_sae); "
                    f"got T={z.shape[1]}."
                )
            z = z.squeeze(1)
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    # ── train_step ──

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"{type(self).__name__}.train_step expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, T, _d_in = x.shape

        post = self._compute_post(x)              # selection-space ReLU pre-acts
        gated = self._batchtopk(post)             # BatchTopK during training
        z_shared = self._to_shared(gated)         # (B, d_sae)

        # JumpReLU threshold EMA (min surviving activation).
        step = int(self.global_step.item())
        if step > self.threshold_start_step:
            with torch.no_grad():
                active = gated[gated > 0]
                cur = (
                    active.min().float()
                    if active.numel() > 0
                    else torch.tensor(0.0, device=gated.device)
                )
                if self.threshold.item() < 0:
                    self.threshold.copy_(cur)
                else:
                    self.threshold.copy_(
                        self.threshold_beta * self.threshold
                        + (1 - self.threshold_beta) * cur
                    )

        # Reconstruction (shared code decodes all T positions).
        x_hat = torch.einsum("bs,std->btd", z_shared, self.W_dec) + self.b_dec
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        # Dead-feature tracking on the shared code.
        with torch.no_grad():
            active_feat = (z_shared > 0).any(dim=0)
            self.num_tokens_since_fired += B * T
            self.num_tokens_since_fired[active_feat] = 0
            dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
            n_dead = int(dead_mask.sum().item())

        # AuxK on dead features, in shared-code space.
        if n_dead > 0:
            k_aux = min(self.aux_k, n_dead)
            pre_sq = F.relu(self._squashed_preact(x))
            auxk_pre = pre_sq.masked_fill(~dead_mask.unsqueeze(0), 0.0)
            vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
            aux_buf = torch.zeros_like(pre_sq).scatter_(-1, idx_a, vals_a)
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
            self.global_step += 1
            l0 = (z_shared != 0).float().sum(dim=-1).mean()

        return {
            "loss": loss,
            "mse": l_recon.detach(),
            "l0": l0.detach(),
            "auxk": l_auxk.detach(),
            "dead": torch.tensor(float(n_dead)),
            "threshold": self.threshold.detach().clone(),
        }

    # ── hooks / introspection ──

    def post_step(self) -> None:
        with torch.no_grad():
            self._normalize_decoder()

    def decoder_directions(self) -> torch.Tensor:
        """(d_sae, d_in) — average decoder over T positions."""
        return self.W_dec.data.mean(dim=1).clone()


class TXCBatchTopKPre(_TXCBatchTopKBase):
    """Pre-squash: BatchTopK on per-position pre-acts, then sum survivors."""

    _registry_name = "txc_batchtopk_pre"

    def _compute_post(self, x: torch.Tensor) -> torch.Tensor:
        # Per-position pre-activations (keep the T axis).
        return F.relu(torch.einsum("btd,tds->bts", x, self.W_enc) + self.b_enc)

    def _to_shared(self, gated: torch.Tensor) -> torch.Tensor:
        return gated.sum(dim=1)                          # (B, d_sae)


class TXCBatchTopKPost(_TXCBatchTopKBase):
    """Post-squash: sum pre-acts into the squashed code, then BatchTopK."""

    _registry_name = "txc_batchtopk_post"

    def _compute_post(self, x: torch.Tensor) -> torch.Tensor:
        # Squashed pre-activations (sum over T inside the einsum).
        return F.relu(torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc)

    def _to_shared(self, gated: torch.Tensor) -> torch.Tensor:
        return gated                                     # already (B, d_sae)
