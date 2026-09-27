"""Bare window-based TXC + tsae_paper anti-dead stack.

Phase 6.1 Track 2 candidate (`agentic_txc_10_bare`). A minimal-
intervention alternative to the briefing's 8-cycle plan: strip
matryoshka AND contrastive loss, keep only the window-based encoder,
and port tsae_paper's full anti-dead machinery. Tests whether
window-based encoding + anti-dead stack alone is sufficient for
qualitative interpretability — independent of matryoshka nesting or
InfoNCE consistency.

Recipe:

    - Encoder: per-position `W_enc[t]` for `t ∈ [0, T)`, shape
      `(T, d_in, d_sae)`. Same as vanilla TXC — sums over T positions
      into a single `(B, d_sae)` pre-activation. (NOT per-token like
      `SharedPerPositionSAE`.)
    - Sparsity: TopK with `k_win = k_pos · T` over the flat d_sae axis,
      per-sample. Keeps Phase 5 probing convention.
    - Decoder: single `W_dec` of shape `(d_sae, T, d_in)` with bias
      `b_dec` of shape `(T, d_in)`. Reconstructs the full T-window.
    - Anti-dead stack (copied from `tsae_paper.py`):
        1. `num_tokens_since_fired` buffer, dead threshold 10M tokens.
        2. AuxK loss: top-k=`aux_k` dead features re-reconstruct the
           residual; `auxk_alpha = 1/32`.
        3. Unit-norm decoder constraint (per-latent over `(T, d_in)`).
        4. Decoder-parallel gradient removal on `W_dec`.
        5. Geometric-median `b_dec` init on the first training batch.

    No matryoshka. No contrastive. No InfoNCE. No BatchTopK
    (deferred — Phase 5.7 experiment (ii) already showed BatchTopK
    regresses TXC sparse probing).

If this wins qualitative at comparable alive fraction to Cycle A
(≥ 5/8 semantic labels), the paper story simplifies: "window-based
encoding + anti-dead machinery is sufficient; matryoshka and
contrastive aren't load-bearing for qualitative."

Encode API matches `PositionMatryoshkaTXCDR` (one (B, d_sae) vector
per window) so downstream probing reuses the matryoshka encoder path
in `run_probing.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def geometric_median(points: torch.Tensor, max_iter: int = 100,
                     tol: float = 1e-5) -> torch.Tensor:
    """Weiszfeld iteration on rows of `points`."""
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess
        weights = 1.0 / (torch.norm(points - guess, dim=1) + 1e-8)
        weights = weights / weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


class TXCBareAntidead(nn.Module):
    """Window-encoder TXC + anti-dead stack; no matryoshka, no contrastive.

    Args:
        d_in: residual-stream width.
        d_sae: dictionary size.
        T: window length.
        k: WINDOW-level TopK budget (usually `k_pos · T`).
        aux_k: budget of dead features per-sample in AuxK loss.
        dead_threshold_tokens: tokens-since-fired to mark a feature dead.
        auxk_alpha: weight on AuxK term.
    """

    def __init__(
        self, d_in: int, d_sae: int, T: int, k: int,
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
        auxk_alpha: float = 1.0 / 32.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k
        self.aux_k = aux_k
        self.dead_threshold_tokens = dead_threshold_tokens
        self.auxk_alpha = auxk_alpha

        # Encoder (per-position weights)
        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        # Decoder (single scale, unit-norm)
        self.W_dec = nn.Parameter(torch.empty(d_sae, T, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        # Kaiming init → unit-norm per-latent atom on decoder → tie encoder
        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc.data[t])
        nn.init.kaiming_uniform_(self.W_dec.data.view(d_sae, T * d_in))
        self._normalize_decoder()
        with torch.no_grad():
            # Encoder init mirrors decoder (paper convention, per-position).
            for t in range(T):
                self.W_enc.data[t] = self.W_dec.data[:, t, :].T

        # Dead-feature tracker
        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(d_sae, dtype=torch.long),
        )
        # Metrics buffers
        self.register_buffer("last_auxk_loss", torch.tensor(-1.0))
        self.register_buffer("last_dead_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("b_dec_initialized", torch.tensor(False))

    @torch.no_grad()
    def _normalize_decoder(self):
        """Unit-norm per decoder atom over (t_size, d_in)."""
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder(self):
        """Project out the component of W_dec.grad parallel to each column.

        Mirrors `tsae_paper.remove_gradient_parallel_to_decoder_directions`
        but for the TXC decoder shape `(d_sae, T, d_in)`. For each atom
        s (viewing it as a flat `(T*d_in,)` vector), subtract the projection
        of `grad[s]` onto it.
        """
        if self.W_dec.grad is None:
            return
        W_flat = self.W_dec.data.view(self.d_sae, -1)              # (d_sae, T*d_in)
        g_flat = self.W_dec.grad.view(self.d_sae, -1)
        normed = W_flat / (W_flat.norm(dim=1, keepdim=True) + 1e-6)  # unit atoms
        parallel = (g_flat * normed).sum(dim=1, keepdim=True)        # (d_sae, 1)
        g_flat.sub_(parallel * normed)
        # g_flat is a view of self.W_dec.grad, so the modification propagates.

    @torch.no_grad()
    def init_b_dec_geometric_median(self, x_sample: torch.Tensor):
        """One-shot init of b_dec from a batch of T-window inputs.

        x_sample: (B, T, d_in). For each position t ∈ [0, T), compute
        the geometric median of x_sample[:, t, :] and assign to
        b_dec[t].
        """
        assert not bool(self.b_dec_initialized), "b_dec already initialized"
        for t in range(self.T):
            med = geometric_median(x_sample[:, t, :].float())
            self.b_dec.data[t] = med.to(self.b_dec.dtype)
        self.b_dec_initialized.fill_(True)

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, d_sae) pre-ReLU, pre-TopK."""
        return torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self._pre_activation(x)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    # Match PositionMatryoshkaTXCDR's interface for downstream probing.
    def decode_scale(self, z: torch.Tensor, scale_idx: int) -> torch.Tensor:
        """Single-scale decoder; `scale_idx` is ignored."""
        del scale_idx
        return self.decode(z)

    @property
    def prefix_sum(self) -> tuple[int, ...]:
        """Single "scale" = full d_sae. Matches matryoshka API shape."""
        return (self.d_sae,) * self.T

    @property
    def decoder_dirs_averaged(self) -> torch.Tensor:
        """(d_in, d_sae) — average decoder direction over the T positions."""
        # W_dec: (d_sae, T, d_in) -> mean over T -> (d_sae, d_in) -> transpose
        return self.W_dec.mean(dim=1).T

    def forward(self, x: torch.Tensor):
        """x: (B, T, d_in) -> (total_loss, x_hat, z).

        AuxK is ALWAYS computed on the training path (no dispatch on
        pair shape — Track 2 doesn't use contrastive pairs).
        """
        # Encode
        pre = self._pre_activation(x)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))

        # Primary reconstruction
        x_hat = self.decode(z)
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        # Update dead tracker
        active_mask = (z > 0).any(dim=0)
        n_tokens = x.shape[0] * x.shape[1]
        self.num_tokens_since_fired += n_tokens
        self.num_tokens_since_fired[active_mask] = 0
        dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
        n_dead = int(dead_mask.sum().item())
        self.last_dead_count.fill_(n_dead)

        # AuxK loss
        if n_dead > 0:
            k_aux = min(self.aux_k, n_dead)
            auxk_pre = F.relu(pre).masked_fill(~dead_mask.unsqueeze(0), 0.0)
            vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
            aux_buf = torch.zeros_like(pre)
            aux_buf.scatter_(-1, idx_a, vals_a)
            # Dead-feature reconstruction through decoder WITHOUT bias (paper)
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
