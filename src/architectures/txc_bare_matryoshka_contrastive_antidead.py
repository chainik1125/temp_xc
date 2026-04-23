"""Phase 6.2 candidates C1/C2/C3: bare TXC + anti-dead stack + optional
Matryoshka H/L loss and/or temporal InfoNCE contrastive loss.

Reuses `TXCBareAntidead`'s architecture wholesale (encoder + decoder +
AuxK dead-feature machinery + unit-norm decoder constraint + decoder-
parallel gradient removal + geometric-median `b_dec` init). Adds two
optional training signals toggleable via `matryoshka_h_size` and
`alpha`:

* **Matryoshka H/L reconstruction** (enabled when
  `matryoshka_h_size is not None`): in addition to the standard full-
  dictionary reconstruction, add a second MSE term that reconstructs
  `x` from the first `h_size` features only (the "high-level" split).
  This matches tsae_paper's `L_H + L_L` objective adapted to our
  window-shaped decoder.

* **Temporal InfoNCE contrastive** (enabled when `alpha > 0`): when the
  forward pass receives a pair tensor `(B, 2, T, d_in)` — window pairs
  `(x_prev, x_cur)` — compute `z_cur`, `z_prev` and add
  `alpha * InfoNCE(z_cur[:, :contr_prefix], z_prev[:, :contr_prefix])`
  to the total loss. `contr_prefix` defaults to `matryoshka_h_size` (if
  set) else `int(d_sae * 0.2)`.

Three concrete variants drive Phase 6.2:

    C1 (phase62_c1_track2_matryoshka):
        matryoshka_h_size = int(d_sae * 0.2), alpha = 0.0
    C2 (phase62_c2_track2_contrastive):
        matryoshka_h_size = None, alpha = 1.0
    C3 (phase62_c3_track2_matryoshka_contrastive):
        matryoshka_h_size = int(d_sae * 0.2), alpha = 1.0

All three inherit Track 2's TopK sparsity (`k_win = k_pos * T`) and
full anti-dead stack. Forward accepts `(B, T, d)` (single window path:
reconstruction + AuxK, no contrastive) OR `(B, 2, T, d)` (pair path:
matryoshka recon on both windows + optional InfoNCE + AuxK).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.txc_bare_antidead import TXCBareAntidead


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


class TXCBareMatryoshkaContrastiveAntidead(TXCBareAntidead):
    """Bare TXC + anti-dead stack + optional matryoshka + optional InfoNCE.

    Args:
        d_in, d_sae, T, k: same as TXCBareAntidead.
        matryoshka_h_size: if not None, enables H/L matryoshka recon. H
            is the first `h_size` feature indices; L is the rest.
        alpha: InfoNCE weight. 0.0 disables the contrastive term; the
            forward then accepts only single-window input.
        contr_prefix: prefix length on z for the contrastive cosine
            sim. Defaults to `matryoshka_h_size` if set, else
            `int(d_sae * 0.2)`.
        aux_k, dead_threshold_tokens, auxk_alpha: same as parent.
    """

    def __init__(
        self, d_in: int, d_sae: int, T: int, k: int,
        matryoshka_h_size: int | None = None,
        alpha: float = 0.0,
        contr_prefix: int | None = None,
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
        auxk_alpha: float = 1.0 / 32.0,
    ):
        super().__init__(
            d_in, d_sae, T, k,
            aux_k=aux_k,
            dead_threshold_tokens=dead_threshold_tokens,
            auxk_alpha=auxk_alpha,
        )
        self.matryoshka_h_size = matryoshka_h_size
        self.alpha = alpha
        if contr_prefix is None:
            contr_prefix = matryoshka_h_size or int(d_sae * 0.2)
        self.contr_prefix = int(contr_prefix)

    def _decode_prefix(self, z: torch.Tensor, h_size: int) -> torch.Tensor:
        """Reconstruct x from only the first `h_size` feature indices.

        Equivalent to `decode(z * mask)` where `mask` zeros out indices
        >= h_size, but done more cheaply via slicing.
        """
        return (
            torch.einsum("bs,std->btd", z[:, :h_size], self.W_dec[:h_size])
            + self.b_dec
        )

    def _recon_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Matryoshka (H + full) MSE if enabled, else just full-dict MSE."""
        x_hat_full = self.decode(z)
        l_full = (x - x_hat_full).pow(2).sum(dim=-1).mean()
        if self.matryoshka_h_size is None:
            return l_full
        x_hat_h = self._decode_prefix(z, self.matryoshka_h_size)
        l_h = (x - x_hat_h).pow(2).sum(dim=-1).mean()
        return l_h + l_full

    def _compute_auxk(self, x: torch.Tensor, x_hat: torch.Tensor,
                       pre: torch.Tensor) -> torch.Tensor:
        """Same AuxK logic as the parent, factored out for reuse."""
        active_mask = (pre > 0).any(dim=0)  # fallback — corrected below at call site with z
        raise RuntimeError("use _compute_auxk_with_z instead")

    def _update_dead_and_auxk(
        self, x: torch.Tensor, x_hat: torch.Tensor,
        pre: torch.Tensor, z: torch.Tensor,
    ) -> torch.Tensor:
        """Update `num_tokens_since_fired` and compute AuxK loss.
        Factored out of parent.forward for reuse across the two paths.
        """
        active_mask = (z > 0).any(dim=0)
        n_tokens = x.shape[0] * x.shape[1]
        self.num_tokens_since_fired += n_tokens
        self.num_tokens_since_fired[active_mask] = 0
        dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
        n_dead = int(dead_mask.sum().item())
        self.last_dead_count.fill_(n_dead)
        if n_dead == 0:
            self.last_auxk_loss.fill_(0.0)
            return torch.zeros((), device=x.device, dtype=x.dtype)
        k_aux = min(self.aux_k, n_dead)
        auxk_pre = F.relu(pre).masked_fill(~dead_mask.unsqueeze(0), 0.0)
        vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
        aux_buf = torch.zeros_like(pre)
        aux_buf.scatter_(-1, idx_a, vals_a)
        aux_decode = torch.einsum("bs,std->btd", aux_buf, self.W_dec)
        residual = x - x_hat.detach()
        l2_a = (residual - aux_decode).pow(2).sum(dim=-1).mean()
        mu = residual.mean(dim=(0, 1), keepdim=True)
        denom = (residual - mu).pow(2).sum(dim=-1).mean()
        l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        self.last_auxk_loss.fill_(float(l_auxk.detach()))
        return l_auxk

    def forward(self, x: torch.Tensor, alpha: float | None = None):
        """Accepts (B, T, d) single-window or (B, 2, T, d) pair.

        Single-window: matryoshka recon + AuxK. Contrastive term skipped
        (no pair available). This is the path used for non-contrastive
        training or for single-window probe encoding.

        Pair: matryoshka recon on both windows + AuxK on `cur` + optional
        InfoNCE on H-prefix.

        The `alpha` kwarg overrides `self.alpha` for this call (matches
        the dispatcher pattern used by `train_matryoshka_txcdr_*`).
        """
        eff_alpha = self.alpha if alpha is None else alpha

        if x.ndim == 3:
            pre = self._pre_activation(x)
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
            x_hat = self.decode(z)
            l_recon = self._recon_loss(x, z)
            l_auxk = self._update_dead_and_auxk(x, x_hat, pre, z)
            total = l_recon + self.auxk_alpha * l_auxk
            return total, x_hat, z

        if x.ndim == 4 and x.shape[1] == 2:
            x_prev = x[:, 0]
            x_cur = x[:, 1]

            pre_prev = self._pre_activation(x_prev)
            vals_p, idx_p = pre_prev.topk(self.k, dim=-1)
            z_prev = torch.zeros_like(pre_prev)
            z_prev.scatter_(1, idx_p, F.relu(vals_p))

            pre_cur = self._pre_activation(x_cur)
            vals_c, idx_c = pre_cur.topk(self.k, dim=-1)
            z_cur = torch.zeros_like(pre_cur)
            z_cur.scatter_(1, idx_c, F.relu(vals_c))

            l_recon = (
                self._recon_loss(x_prev, z_prev)
                + self._recon_loss(x_cur, z_cur)
            )

            # Contrastive term (H-prefix). If alpha=0 this is a no-op
            # aside from the arithmetic; keep it in the graph so the
            # model params still see a constant-zero contribution with
            # no compute cost.
            l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
            if eff_alpha > 0.0:
                h = self.contr_prefix
                l_contr = _info_nce(z_cur[:, :h], z_prev[:, :h])

            # AuxK on the `cur` path (parent convention).
            x_hat_cur = self.decode(z_cur)
            l_auxk = self._update_dead_and_auxk(x_cur, x_hat_cur, pre_cur, z_cur)

            total = l_recon + eff_alpha * l_contr + self.auxk_alpha * l_auxk
            return total, x_hat_cur, z_cur

        raise ValueError(
            f"Expected (B, T, d) or (B, 2, T, d), got {tuple(x.shape)}"
        )
