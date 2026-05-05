"""TXC-base — vanilla TopK temporal crosscoder with anti-dead stack.

Source: ``origin/han-phase7-unification @ 94119bc0:src/architectures/txc_bare_antidead.py``
("Phase 6.1 Track 2: agentic_txc_10_bare").

Architecture:
- Encoder: per-position weights ``W_enc (T, d_in, d_sae)``; sums over T
  positions into a single window-level pre-activation ``(B, d_sae)``.
- Decoder: ``W_dec (d_sae, T, d_in)``; one window-level latent
  reconstructs the full T-window.
- Sparsity: window-level TopK with ``k_win = k_pos * T``.
- Anti-dead stack (ported from tsae_paper):
  - num_tokens_since_fired buffer + AuxK loss on dead features
    (``auxk_alpha = 1/32``).
  - Per-atom decoder unit-norm constraint (over the (T, d_in) dimensions).
  - Decoder-parallel gradient removal via post-accumulate hook.

Adapted for the unified ``TempBenchArch`` framework (commit 3b70563f):

- ``encode(x)`` returns ``(B, 1, d_sae)`` — one window-level latent per
  input window. Matches the base class's "T may be 1 for shared-z TXCs"
  shape convention.
- ``train_step(x)`` accepts ``(B, seq_len, d_in)`` from the canonical
  batch_iter (full sequences from the activation cache) and randomly
  extracts a single T-window per batch element. Loss is recon + AuxK
  on dead features. NO Bricken (per Decision #7: C6 only by default).
- Decoder unit-norm renormalisation in ``post_step()``; grad-parallel
  removal via ``register_post_accumulate_grad_hook`` on ``W_dec``.

Geometric-median b_dec init from the wasteland is intentionally not
ported here: the canonical SAE trainer doesn't expose a "first-batch"
hook, and zero init has been adequate at d_sae=18432 in Phase 7.
If we revisit, expose it as an explicit `init_from_batch(x)` method.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.architectures.base import ArchConfig, TempBenchArch


class TXCBase(TempBenchArch):
    """Vanilla TopK temporal crosscoder + tsae_paper anti-dead stack.

    Args (from ``configs/locked_archs.yaml::txc_base``):
        d_in:                    residual width.
        d_sae:                   dictionary size.
        T:                       window length (Phase 5 default = 5).
        k_pos:                   per-token sparsity. Window L0 = k_pos * T.
        auxk_alpha:              AuxK loss weight (paper: 1/32).
        dead_threshold_tokens:   tokens-since-fired before AuxK targets.
        bdec_geom_median_init:   accepted (yaml hparam) but not implemented;
                                 the canonical trainer has no first-batch
                                 hook. See module docstring.
        decoder_unit_norm:       accepted; always ``True`` in this port.
        decoder_grad_orthogonalize: accepted; always ``True`` in this port.
    """

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
        multi_window: bool = False,
        bdec_geom_median_init: bool = True,   # noqa: ARG002 — accepted, see docstring
        decoder_unit_norm: bool = True,        # noqa: ARG002
        decoder_grad_orthogonalize: bool = True,  # noqa: ARG002
    ):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name="txc_base", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        self.k_pos = k_pos
        self.k_win = k_pos * T  # window-level TopK budget
        self.auxk_alpha = auxk_alpha
        self.dead_threshold_tokens = dead_threshold_tokens
        self.aux_k = aux_k if aux_k is not None else min(512, d_in // 2)
        # Multi-window sampling toggle (added 2026-05-05; see decisions.md § 14
        # "TXC training-FLOPs parity"). False = original 1-random-window-per-row
        # behavior, used by all in-flight cells. True = stride-T tiling that
        # gives N=seq_len//T windows per row, matching per-token SAE token
        # throughput per step. Toggling False→True via YAML hparam invalidates
        # train_keys (the hparam goes into compute_train_key's hash).
        self._multi_window = multi_window

        # Encoder: per-position (T, d_in, d_sae)
        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        # Decoder: (d_sae, T, d_in); bias per position
        self.W_dec = nn.Parameter(torch.empty(d_sae, T, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        # Init: kaiming on each per-position encoder slice + decoder atoms
        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc.data[t])
        nn.init.kaiming_uniform_(self.W_dec.data.view(d_sae, T * d_in))
        with torch.no_grad():
            self._normalize_decoder()
            # Tie encoder = decoder per-position transpose (paper convention).
            for t in range(T):
                self.W_enc.data[t] = self.W_dec.data[:, t, :].T

        # Dead-feature tracker
        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(d_sae, dtype=torch.long),
        )

        # Pre-step grad-parallel removal as a tensor hook on W_dec.
        # Avoids needing a pre-step trainer extension.
        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    # ── Decoder-norm utilities ──

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        """Unit-norm per decoder atom over (T, d_in)."""
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    @staticmethod
    def _project_dec_grad(param: torch.Tensor) -> None:
        """Project out the W_dec.grad component parallel to each atom."""
        if param.grad is None:
            return
        d_sae = param.shape[0]
        W_flat = param.data.view(d_sae, -1)                          # (d_sae, T*d_in)
        g_flat = param.grad.data.view(d_sae, -1)
        normed = W_flat / (W_flat.norm(dim=1, keepdim=True) + 1e-6)
        parallel = (g_flat * normed).sum(dim=1, keepdim=True)         # (d_sae, 1)
        g_flat.sub_(parallel * normed)

    # ── encode / decode (TempBenchArch contract) ──

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Window-level encoding.

        Input shapes accepted:
            (B, T, d_in)  — single window per batch element.
            (B, d_in)     — treats T=1 (the abstract contract; rarely used
                            for TXC since T is its whole point).

        Output:
            (B, 1, d_sae) — one window-level latent. The ``T`` axis on the
            output is 1 because the encoder collapses across the input T
            positions.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self._T:
            raise ValueError(
                f"TXCBase.encode expects (B, T={self._T}, d_in); got T_input={x.shape[1]}."
            )
        # (B, T, d_in) × (T, d_in, d_sae) → (B, d_sae)
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        vals, idx = pre.topk(self.k_win, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z.unsqueeze(1)  # (B, 1, d_sae)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode (B, 1, d_sae) or (B, d_sae) → (B, T, d_in)."""
        if z.dim() == 3:
            if z.shape[1] != 1:
                raise ValueError(
                    f"TXCBase.decode expects (B, 1, d_sae); got T={z.shape[1]}."
                )
            z = z.squeeze(1)
        # (B, d_sae) × (d_sae, T, d_in) → (B, T, d_in)
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    # ── train_step: extract T-windows from sequences + recon + AuxK ──

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Args:
            x: (B, seq_len, d_in) from the canonical batch_iter.
               If ``self._multi_window`` is False (default — original
               behavior), we sample ONE random T-window per batch row,
               giving (B, T, d_in) effective rows.
               If ``self._multi_window`` is True (opt-in via YAML
               hparam ``multi_window: true``, added 2026-05-05), we tile
               each sequence into ``N = seq_len // T`` non-overlapping
               stride-T windows, giving (B*N, T, d_in) effective rows.
               Per-step token throughput becomes B*N*T ≈ B*seq_len,
               matching per-token SAEs and eliminating the ~25× training
               -FLOPs disadvantage that biases cross-arch comparisons
               against TXC. See decisions.md § 14.

        Returns:
            (loss, info) — info has 'mse', 'l0', 'auxk', 'dead', 'z'.
        """
        if x.dim() != 3 or x.shape[1] < self._T:
            raise ValueError(
                f"TXCBase.train_step expects (B, seq_len>={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, seq_len, d_in = x.shape
        T = self._T
        if self._multi_window:
            # Tile (B, seq_len, d_in) into (B*N, T, d_in) at stride T.
            # Trailing seq_len - N*T tokens are dropped.
            N = seq_len // T
            windows = x[:, : N * T, :].reshape(B, N, T, d_in).reshape(B * N, T, d_in)
            n_tokens_per_step = B * N * T
        else:
            # Original behavior: one random T-window per batch row.
            offsets = torch.randint(
                0, seq_len - T + 1, (B,), device=x.device
            )
            idx_t = offsets.unsqueeze(1) + torch.arange(T, device=x.device).unsqueeze(0)
            batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, T)
            windows = x[batch_idx, idx_t]                              # (B, T, d_in)
            n_tokens_per_step = B * T

        # Encode (raw pre-activation needed for AuxK)
        pre = torch.einsum("btd,tds->bs", windows, self.W_enc) + self.b_enc
        vals, idx = pre.topk(self.k_win, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))

        # Reconstruction
        x_hat = torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec
        l_recon = (windows - x_hat).pow(2).sum(dim=-1).mean()

        # Dead-feature tracker (token count depends on sampling mode)
        with torch.no_grad():
            active_mask = (z > 0).any(dim=0)
            self.num_tokens_since_fired += n_tokens_per_step
            self.num_tokens_since_fired[active_mask] = 0
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
            residual = (windows - x_hat).detach()
            l2_a = (residual - aux_decode).pow(2).sum(dim=-1).mean()
            mu = residual.mean(dim=(0, 1), keepdim=True)
            denom = (residual - mu).pow(2).sum(dim=-1).mean()
            l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        else:
            l_auxk = torch.zeros((), device=x.device, dtype=x.dtype)

        total = l_recon + self.auxk_alpha * l_auxk

        with torch.no_grad():
            l0 = (z != 0).float().sum(dim=-1).mean()  # window-level L0

        return total, {
            "mse": l_recon.detach(),
            "l0": l0.detach(),
            "auxk": l_auxk.detach(),
            "dead": n_dead,
            "z": z.detach(),
        }

    # ── post_step (decoder unit-norm renormalisation) ──

    def post_step(self) -> None:
        with torch.no_grad():
            self._normalize_decoder()

    # ── decoder_directions for C4 (T-averaged) ──

    def decoder_directions(self) -> torch.Tensor:
        """(d_sae, d_in) — average decoder direction across T positions."""
        # W_dec: (d_sae, T, d_in) -> mean over T -> (d_sae, d_in)
        return self.W_dec.data.mean(dim=1).clone()
