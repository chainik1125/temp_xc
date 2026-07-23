"""TXC-pro loss dissection — the post-squash backbone with per-component losses.

One class, four registry entries (``txc_post_plain`` / ``txc_post_mat`` /
``txc_post_ctr`` / ``txc_post_both`` — the ``spectral_txc`` one-class-many-entries
precedent), dissecting the paper's TXC-pro bundle into its loss components on
the ``txc_batchtopk_post`` backbone. Frozen design:
``experiments/explorations/synthetic/loss_dissection/CARD.md``.

- **Backbone**: inherits ``TXCBatchTopKPost`` wholesale (params, init, encode/
  decode, BatchTopK→JumpReLU, AuxK, unit-norm decoder + grad-parallel removal).
  Variants differ ONLY in the training loss; parameter creation is identical,
  so all four share state_dicts at init under one torch seed and their
  untrained rows must coincide.
- **Matryoshka** (``mat_alpha``): paper-faithful H=8 nested prefixes
  ``n_G = floor(G*d_sae/8)``; adds ``mat_alpha * sum_{G<H} l2(x, prefix_G)``
  on the anchor (the G=H term IS the plain recon term). ``_decode_prefix``
  grafted from the Phase-6.2 lineage
  (``txc_bare_matryoshka_contrastive_antidead`` @ 2fa9bdab).
- **Multi-distance contrastive** (``ctr_alpha``): cosine-normalized symmetric
  InfoNCE (verbatim ``_info_nce`` port from the same lineage) between the
  anchor's gated shared code and each shifted positive's, weights
  ``1/(1+Delta)``, shifts ``Delta in {1,2}`` (paper), full code (toy-scale
  TXC-pro sets h_size=d_sae).
- **Data**: ``consumes="sequence"`` — the trainer's SequenceBuffer yields
  ``(B, seq_len, d_in)``; we slice one anchor window per row at
  ``p ~ U{0..seq_len-T-S_MAX}`` (S_MAX=2 for ALL variants, so the anchor
  distribution is family-identical) and positives as deterministic shifts of
  the same offsets (no extra RNG — equal-seed variants consume identical
  training windows until gradients diverge). Positives contribute ONLY the
  InfoNCE terms: no recon / AuxK / threshold / dead-tracking side effects, so
  zero-weight variants reduce exactly to the plain backbone loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from temp_bench.archs.txc_batchtopk import TXCBatchTopKPost


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Cosine-normalized symmetric InfoNCE (verbatim 2fa9bdab port)."""
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


class TXCPostDissect(TXCBatchTopKPost):
    """Post-squash BatchTopK TXC + optional matryoshka / multi-distance InfoNCE."""

    arch_version: str = "1.0.0"
    consumes: str = "sequence"
    _registry_name = "txc_post_dissect"

    S_MAX = 2  # positives' max shift; reserved for ALL variants (CARD § 2)

    def __init__(
        self,
        *,
        mat_alpha: float = 0.0,
        ctr_alpha: float = 0.0,
        mat_groups: int = 8,
        ctr_shifts: tuple[int, ...] = (1, 2),
        **kw,
    ):
        # Parent constructor first and unconditionally: parameter shapes and
        # init RNG draws are identical across variants (loss-only differences).
        super().__init__(**kw)
        self.mat_alpha = float(mat_alpha)
        self.ctr_alpha = float(ctr_alpha)
        self.mat_groups = int(mat_groups)
        self.ctr_shifts = tuple(int(s) for s in ctr_shifts)
        self.ctr_weights = tuple(1.0 / (1.0 + s) for s in self.ctr_shifts)
        if any(s <= 0 or s > self.S_MAX for s in self.ctr_shifts):
            raise ValueError(f"ctr_shifts must lie in [1, {self.S_MAX}]; got {self.ctr_shifts}")

    # ── matryoshka prefix ladder ──

    def _prefix_sizes(self) -> tuple[int, ...]:
        """Nested prefix sizes ``floor(G*d_sae/H)``, deduped, ending at d_sae."""
        H = self.mat_groups
        sizes = {max(1, (G * self._d_sae) // H) for G in range(1, H + 1)}
        sizes.add(self._d_sae)
        return tuple(sorted(sizes))

    def _decode_prefix(self, z: torch.Tensor, n: int) -> torch.Tensor:
        """Reconstruct all T positions from the first ``n`` features only."""
        return torch.einsum("bs,std->btd", z[:, :n], self.W_dec[:n]) + self.b_dec

    # ── sequence slicing ──

    def _slice(self, x: torch.Tensor):
        """(B, seq_len, d_in) → (anchor (B,T,d_in), positives list|None)."""
        if x.dim() != 3:
            raise ValueError(
                f"{type(self).__name__}.train_step expects (B, seq_len, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, L, _d = x.shape
        T = self._T
        hi = L - T - self.S_MAX
        if hi < 0:
            raise ValueError(f"need seq_len >= T + {self.S_MAX} = {T + self.S_MAX}; got {L}.")
        # One randint per step for EVERY variant (identical RNG streams at
        # equal seeds); positives are deterministic shifts of the same offsets.
        p = torch.randint(0, hi + 1, (B,), device=x.device)
        rows = torch.arange(B, device=x.device).unsqueeze(1)
        idx = p.unsqueeze(1) + torch.arange(T, device=x.device).unsqueeze(0)
        anchor = x[rows, idx]
        positives = None
        if self.ctr_alpha > 0:
            positives = [x[rows, idx + s] for s in self.ctr_shifts]
        return anchor, positives

    # ── train_step ──

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        anchor, positives = self._slice(x)
        return self._loss_on(anchor, positives)

    def _loss_on(
        self, x: torch.Tensor, positives: list[torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """Backbone loss on the anchor (op-for-op the parent's train_step) plus
        the guarded matryoshka / contrastive terms. With both alphas at 0 the
        computation is identical to ``TXCBatchTopKPost.train_step`` (contract
        test 1 guards drift)."""
        if x.dim() != 3 or x.shape[1] != self._T:
            raise ValueError(
                f"{type(self).__name__}._loss_on expects (B, T={self._T}, d_in); "
                f"got {tuple(x.shape)}."
            )
        B, T, _d_in = x.shape

        post = self._compute_post(x)              # selection-space ReLU pre-acts
        gated = self._batchtopk(post)             # BatchTopK during training
        z_shared = self._to_shared(gated)         # (B, d_sae)

        # JumpReLU threshold EMA (min surviving activation) — anchor only.
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

        # Dead-feature tracking on the shared code — anchor only.
        with torch.no_grad():
            active_feat = (z_shared > 0).any(dim=0)
            self.num_tokens_since_fired += B * T
            self.num_tokens_since_fired[active_feat] = 0
            dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
            n_dead = int(dead_mask.sum().item())

        # AuxK on dead features, in shared-code space — full-recon residual.
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

        # ── matryoshka: prefix terms G=1..H-1 (G=H IS l_recon) ──
        l_mat = torch.zeros((), device=x.device, dtype=x.dtype)
        if self.mat_alpha > 0:
            for n in self._prefix_sizes()[:-1]:
                x_hat_g = self._decode_prefix(z_shared, n)
                l_mat = l_mat + (x - x_hat_g).pow(2).sum(dim=-1).mean()
            loss = loss + self.mat_alpha * l_mat

        # ── multi-distance contrastive: InfoNCE vs each shifted positive ──
        l_ctr = torch.zeros((), device=x.device, dtype=x.dtype)
        if self.ctr_alpha > 0 and positives is not None:
            for w, pos in zip(self.ctr_weights, positives):
                z_pos = self._to_shared(self._batchtopk(self._compute_post(pos)))
                l_ctr = l_ctr + w * _info_nce(z_shared, z_pos)
            loss = loss + self.ctr_alpha * l_ctr

        with torch.no_grad():
            self.global_step += 1
            l0 = (z_shared != 0).float().sum(dim=-1).mean()

        return {
            "loss": loss,
            "mse": l_recon.detach(),
            "l0": l0.detach(),
            "auxk": l_auxk.detach(),
            "mat": l_mat.detach(),
            "ctr": l_ctr.detach(),
            "dead": torch.tensor(float(n_dead)),
            "threshold": self.threshold.detach().clone(),
        }
