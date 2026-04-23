"""Bare window-based TXC + tsae_paper anti-dead stack + BatchTopK sparsity.

Phase 6.1 follow-up #4 candidate (`agentic_txc_12_bare_batchtopk`).
Fills the missing 2×2 cell:

    |                         | TopK          | BatchTopK         |
    |-------------------------|---------------|-------------------|
    | matryoshka + contrastive| (baseline 2/8)| Cycle F 7/8       |
    | bare (no matr. / contr.)| Track 2 6/8   | [this]            |

Recipe = `TXCBareAntidead` + swap the per-sample TopK scatter for a
BatchTopK module that pools top B·k across (B, d_sae). Anti-dead
machinery (AuxK, unit-norm decoder, decoder-parallel gradient removal,
geometric-median b_dec init) is inherited unchanged.

Outcome readings:
    - If this beats Cycle F (≥ 8/8 at seed 42): simpler recipe wins —
      matryoshka + multi-scale contrastive is ornamental for qualitative.
    - If it ties Cycle F (~7/8): matryoshka + contrastive is orthogonal
      to qualitative gain (consistent with Track 2's 6/8 result).
    - If it regresses (~5/8 like Cycle H): BatchTopK + AuxK does NOT
      compose cleanly in general — Cycle H's interaction generalises
      beyond matryoshka.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.architectures._batchtopk import BatchTopK
from src.architectures.txc_bare_antidead import TXCBareAntidead


class TXCBareBatchTopKAntidead(TXCBareAntidead):
    """TXCBareAntidead with BatchTopK sparsity on the primary path.

    The only thing that changes vs. the parent is the sparsity mechanism:
    primary encode uses a BatchTopK module; AuxK stays per-sample
    (paper convention — dead features compete within each sample for
    the residual reconstruction).
    """

    def __init__(
        self, d_in: int, d_sae: int, T: int, k: int,
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
        self.sparsity = BatchTopK(k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self._pre_activation(x)
        return self.sparsity(pre)

    def forward(self, x: torch.Tensor):
        """x: (B, T, d_in) -> (total_loss, x_hat, z). Mirrors parent.forward,
        swapping per-sample TopK for BatchTopK on the primary path.
        """
        pre = self._pre_activation(x)
        z = self.sparsity(pre)  # BatchTopK: pooled top B·k across (B, d_sae)

        x_hat = self.decode(z)
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        # Dead tracker (same as parent; BatchTopK changes per-sample firing
        # counts but `(z > 0).any(dim=0)` is still the right per-latent mask)
        active_mask = (z > 0).any(dim=0)
        n_tokens = x.shape[0] * x.shape[1]
        self.num_tokens_since_fired += n_tokens
        self.num_tokens_since_fired[active_mask] = 0
        dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
        n_dead = int(dead_mask.sum().item())
        self.last_dead_count.fill_(n_dead)

        # AuxK loss — per-sample TopK over dead features (paper convention)
        if n_dead > 0:
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
        else:
            l_auxk = torch.zeros((), device=x.device, dtype=x.dtype)
            self.last_auxk_loss.fill_(0.0)

        total = l_recon + self.auxk_alpha * l_auxk
        return total, x_hat, z
