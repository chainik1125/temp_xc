"""Galaxy 4 — Hierarchical Multi-Scale TXC.

Y's GIGABRAIN architectural proposal (2026-05-01): explicit decomposition
of the latent into window-level features (multi-token concepts) and
per-position features (per-token concepts).

Mechanism:
    Encoder:
      pre_w[b]      = Σ_t x[b, t, :] @ W_enc_w[t]  +  b_enc_w   (window-level)
      pre_pos[b,t]  = x[b, t, :] @ W_enc_pos[t]    +  b_enc_pos[t]
      z_window      = TopK(ReLU(pre_w),  K_window)             (B, d_sae_w)
      z_pos[t]      = TopK(ReLU(pre_pos[t]), K_pos)            (B, d_sae_p)

    Decoder:
      x_hat[t]      = z_window @ W_dec_w[:, t, :]
                    + z_pos[t]  @ W_dec_pos[t, :, :]
                    + b_dec[t]

Matched per-token sparsity: active features per token = K_window + K_pos.
For k_pos=20: K_window=10, K_pos=10 → active=20 per position.

Why this might help vs. TXCBareAntidead:
    The original encoder collapses all multi-position info into a single
    d_sae feature vector before TopK. Per-token vocab features and
    multi-token discourse features compete for the same TopK slots.
    The hierarchical version dedicates separate d_sae groups (and
    separate TopK selectors) to window-level vs. per-position roles,
    letting the model learn explicit multi-scale features.

    Hypothesis: lifts coh ≥ 1.75 win further by unblocking the
    multi-scale tradeoff our experiments hinted at (Y's findings: T=2
    H8 wins per-position; T=2 bare wins right-edge — different
    architectures specialize differently).

Anti-dead stack: tracks dead features per latent group separately;
auxK loss applied to whichever group has dead features.
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


class TXCHierarchicalMultiScale(nn.Module):
    """Hierarchical multi-scale TXC: window-level + per-position latents.

    Args:
        d_in: residual width.
        d_sae_w: dictionary size for window-level features.
        d_sae_p: dictionary size for per-position features.
        T: window length.
        k_window: TopK budget for window-level features (per sample).
        k_pos: TopK budget for per-position features (per sample, per position).
        aux_k: dead-feature aux budget (split equally across groups).
        dead_threshold_tokens: tokens since last fired to mark a feature dead.
        auxk_alpha: weight on AuxK term (default tsae_paper convention).
    """

    def __init__(
        self, d_in: int, d_sae_w: int, d_sae_p: int, T: int,
        k_window: int, k_pos: int,
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
        auxk_alpha: float = 1.0 / 32.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae_w = d_sae_w
        self.d_sae_p = d_sae_p
        self.T = T
        self.k_window = k_window
        self.k_pos = k_pos
        self.aux_k = aux_k
        self.dead_threshold_tokens = dead_threshold_tokens
        self.auxk_alpha = auxk_alpha

        # === Window-level encoder/decoder (same shape as TXCBareAntidead) ===
        self.W_enc_w = nn.Parameter(torch.empty(T, d_in, d_sae_w))
        self.b_enc_w = nn.Parameter(torch.zeros(d_sae_w))
        self.W_dec_w = nn.Parameter(torch.empty(d_sae_w, T, d_in))

        # === Per-position encoder/decoder (per-token features) ===
        # Per-position: each t has its own (d_in, d_sae_p) encoder
        self.W_enc_pos = nn.Parameter(torch.empty(T, d_in, d_sae_p))
        self.b_enc_pos = nn.Parameter(torch.zeros(T, d_sae_p))
        self.W_dec_pos = nn.Parameter(torch.empty(T, d_sae_p, d_in))

        # === Shared decoder bias ===
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        # Init: kaiming, then unit-norm decoder atoms, tie encoder.
        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc_w.data[t])
            nn.init.kaiming_uniform_(self.W_enc_pos.data[t])
        nn.init.kaiming_uniform_(self.W_dec_w.data.view(d_sae_w, T * d_in))
        for t in range(T):
            nn.init.kaiming_uniform_(self.W_dec_pos.data[t])
        self._normalize_decoder_w()
        self._normalize_decoder_pos()

        with torch.no_grad():
            for t in range(T):
                self.W_enc_w.data[t] = self.W_dec_w.data[:, t, :].T
                self.W_enc_pos.data[t] = self.W_dec_pos.data[t].T

        # Dead-feature trackers (per group)
        self.register_buffer("num_tokens_since_fired_w",
                             torch.zeros(d_sae_w, dtype=torch.long))
        self.register_buffer("num_tokens_since_fired_p",
                             torch.zeros(T, d_sae_p, dtype=torch.long))
        # Metrics
        self.register_buffer("last_auxk_loss", torch.tensor(-1.0))
        self.register_buffer("last_dead_count_w", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_dead_count_p", torch.tensor(0, dtype=torch.long))
        self.register_buffer("b_dec_initialized", torch.tensor(False))

    @torch.no_grad()
    def _normalize_decoder_w(self):
        norms = self.W_dec_w.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec_w.data = self.W_dec_w.data / norms

    @torch.no_grad()
    def _normalize_decoder_pos(self):
        # W_dec_pos shape: (T, d_sae_p, d_in). Normalize per (t, j) over d_in.
        norms = self.W_dec_pos.norm(dim=2, keepdim=True).clamp(min=1e-8)
        self.W_dec_pos.data = self.W_dec_pos.data / norms

    @torch.no_grad()
    def _normalize_decoder(self):
        """Wrapper for _flat_train compatibility. Normalizes both decoder groups."""
        self._normalize_decoder_w()
        self._normalize_decoder_pos()

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder(self):
        # Window decoder
        if self.W_dec_w.grad is not None:
            W_flat = self.W_dec_w.data.view(self.d_sae_w, -1)
            g_flat = self.W_dec_w.grad.view(self.d_sae_w, -1)
            normed = W_flat / (W_flat.norm(dim=1, keepdim=True) + 1e-6)
            parallel = (g_flat * normed).sum(dim=1, keepdim=True)
            g_flat.sub_(parallel * normed)
        # Per-pos decoder
        if self.W_dec_pos.grad is not None:
            W_flat = self.W_dec_pos.data  # (T, d_sae_p, d_in)
            g_flat = self.W_dec_pos.grad
            normed = W_flat / (W_flat.norm(dim=2, keepdim=True) + 1e-6)
            parallel = (g_flat * normed).sum(dim=2, keepdim=True)
            g_flat.sub_(parallel * normed)

    @torch.no_grad()
    def init_b_dec_geometric_median(self, x_sample: torch.Tensor):
        assert not bool(self.b_dec_initialized), "b_dec already initialized"
        for t in range(self.T):
            med = geometric_median(x_sample[:, t, :].float())
            self.b_dec.data[t] = med.to(self.b_dec.dtype)
        self.b_dec_initialized.fill_(True)

    def _encode_window(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, d_sae_w) post-TopK z_window."""
        pre = torch.einsum("btd,tds->bs", x, self.W_enc_w) + self.b_enc_w
        vals, idx = pre.topk(self.k_window, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z

    def _encode_pos(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) -> (B, T, d_sae_p) post-TopK z_pos."""
        # Per-position pre-act
        pre = torch.einsum("btd,tds->bts", x, self.W_enc_pos) + self.b_enc_pos
        # Per-position TopK across d_sae_p axis
        vals, idx = pre.topk(self.k_pos, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(2, idx, F.relu(vals))
        return z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Concat window + pos latents into a single (B, d_sae_w + T*d_sae_p) vector.

        For downstream probing pipelines that expect a single z vector.
        Indexing convention: first d_sae_w entries = window features;
        next T * d_sae_p entries = per-position features (in t-major order).
        """
        z_w = self._encode_window(x)
        z_p = self._encode_pos(x)
        z_p_flat = z_p.reshape(z_p.shape[0], -1)
        return torch.cat([z_w, z_p_flat], dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode the concatenated latent z back to (B, T, d_in)."""
        z_w = z[:, : self.d_sae_w]
        z_p = z[:, self.d_sae_w :].reshape(z.shape[0], self.T, self.d_sae_p)
        x_hat_w = torch.einsum("bs,std->btd", z_w, self.W_dec_w)
        x_hat_p = torch.einsum("bts,tsd->btd", z_p, self.W_dec_pos)
        return x_hat_w + x_hat_p + self.b_dec

    def decode_scale(self, z: torch.Tensor, scale_idx: int) -> torch.Tensor:
        del scale_idx
        return self.decode(z)

    @property
    def prefix_sum(self) -> tuple[int, ...]:
        # Single "scale" for matryoshka API: total = d_sae_w + T*d_sae_p
        total = self.d_sae_w + self.T * self.d_sae_p
        return (total,) * self.T

    @property
    def d_sae(self) -> int:
        return self.d_sae_w + self.T * self.d_sae_p

    @property
    def decoder_dirs_averaged(self) -> torch.Tensor:
        """(d_in, d_sae_total) per-feature decoder-direction matrix.

        Layout (matches encode()'s concat order):
          cols 0..d_sae_w-1                       : window features (averaged over T positions)
          cols d_sae_w..d_sae_w + d_sae_p - 1      : per-pos features at t=0
          cols d_sae_w + d_sae_p .. + 2*d_sae_p - 1 : per-pos features at t=1
          ...
        """
        # Window features: average W_dec_w over T positions → (d_sae_w, d_in) → transpose
        dirs_w = self.W_dec_w.mean(dim=1).T  # (d_in, d_sae_w)
        # Per-pos features: W_dec_pos[t] is (d_sae_p, d_in); per-position direction
        # for steering at all positions = the per-position direction itself
        dirs_p = []
        for t in range(self.T):
            dirs_p.append(self.W_dec_pos[t].T)  # (d_in, d_sae_p)
        dirs_p_cat = torch.cat(dirs_p, dim=1)  # (d_in, T * d_sae_p)
        return torch.cat([dirs_w, dirs_p_cat], dim=1)  # (d_in, d_sae_total)

    def forward(self, x: torch.Tensor):
        """x: (B, T, d_in) -> (total_loss, x_hat, z)."""
        # Encode
        z_w = self._encode_window(x)               # (B, d_sae_w)
        z_p = self._encode_pos(x)                  # (B, T, d_sae_p)

        # Decode
        x_hat_w = torch.einsum("bs,std->btd", z_w, self.W_dec_w)
        x_hat_p = torch.einsum("bts,tsd->btd", z_p, self.W_dec_pos)
        x_hat = x_hat_w + x_hat_p + self.b_dec
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        # Update dead trackers
        active_w = (z_w > 0).any(dim=0)
        active_p = (z_p > 0).any(dim=0)  # (T, d_sae_p)
        n_tokens_w = x.shape[0]
        n_tokens_p = x.shape[0]
        self.num_tokens_since_fired_w += n_tokens_w
        self.num_tokens_since_fired_w[active_w] = 0
        self.num_tokens_since_fired_p += n_tokens_p
        # active_p needs to be (T, d_sae_p); align with buffer
        for t in range(self.T):
            self.num_tokens_since_fired_p[t, active_p[t]] = 0
        dead_mask_w = self.num_tokens_since_fired_w >= self.dead_threshold_tokens
        dead_mask_p = self.num_tokens_since_fired_p >= self.dead_threshold_tokens
        n_dead_w = int(dead_mask_w.sum().item())
        n_dead_p = int(dead_mask_p.sum().item())
        self.last_dead_count_w.fill_(n_dead_w)
        self.last_dead_count_p.fill_(n_dead_p)

        # AuxK loss (apply to BOTH groups; share aux_k budget)
        l_auxk = torch.zeros((), device=x.device, dtype=x.dtype)
        n_dead_total = n_dead_w + n_dead_p
        if n_dead_total > 0:
            # Split aux budget proportionally to dead counts
            aux_w = min(self.aux_k * n_dead_w // max(n_dead_total, 1), n_dead_w)
            aux_p = min(self.aux_k - aux_w, n_dead_p)

            residual = x - x_hat.detach()

            if aux_w > 0 and n_dead_w > 0:
                pre_w = torch.einsum("btd,tds->bs", x, self.W_enc_w) + self.b_enc_w
                auxk_pre_w = F.relu(pre_w).masked_fill(~dead_mask_w.unsqueeze(0), 0.0)
                vals_a, idx_a = auxk_pre_w.topk(aux_w, dim=-1, sorted=False)
                aux_buf = torch.zeros_like(pre_w)
                aux_buf.scatter_(-1, idx_a, vals_a)
                aux_decode_w = torch.einsum("bs,std->btd", aux_buf, self.W_dec_w)
                l_auxk = l_auxk + (residual - aux_decode_w).pow(2).sum(dim=-1).mean()

            if aux_p > 0 and n_dead_p > 0:
                pre_p = torch.einsum("btd,tds->bts", x, self.W_enc_pos) + self.b_enc_pos
                auxk_pre_p = F.relu(pre_p).masked_fill(~dead_mask_p.unsqueeze(0), 0.0)
                # TopK per-position for AuxK
                aux_p_per_pos = max(aux_p // self.T, 1)
                vals_a, idx_a = auxk_pre_p.topk(aux_p_per_pos, dim=-1, sorted=False)
                aux_buf = torch.zeros_like(pre_p)
                aux_buf.scatter_(2, idx_a, vals_a)
                aux_decode_p = torch.einsum("bts,tsd->btd", aux_buf, self.W_dec_pos)
                l_auxk = l_auxk + (residual - aux_decode_p).pow(2).sum(dim=-1).mean()

            mu = residual.mean(dim=(0, 1), keepdim=True)
            denom = (residual - mu).pow(2).sum(dim=-1).mean()
            l_auxk = (l_auxk / denom.clamp(min=1e-8)).nan_to_num(0.0)
            self.last_auxk_loss.fill_(float(l_auxk.detach()))
        else:
            self.last_auxk_loss.fill_(0.0)

        # Concat z for downstream
        z = torch.cat([z_w, z_p.reshape(z_p.shape[0], -1)], dim=-1)
        total = l_recon + self.auxk_alpha * l_auxk
        return total, x_hat, z
