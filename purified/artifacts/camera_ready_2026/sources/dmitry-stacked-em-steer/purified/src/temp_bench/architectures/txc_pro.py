"""TXC-pro — subseq encoder + matryoshka + multi-distance InfoNCE + anti-dead.

Locked headline arch alongside ``txc_base``. Spec from
``configs/locked_archs.yaml::txc_pro`` and ``docs/paper/architecture.md``:

- **Subseq encoder** (Phase 5b SubseqH8):
  - W_enc shape ``(T_max, d_in, d_sae)`` — per-position weights for T_max=10
  - At training: sample t_sample=5 contiguous positions; pre-activation
    sums encoder contributions only over the sampled subset.
  - At inference (probe-time): use the FULL T_max=10 window (no sampling).
- **Matryoshka H+full reconstruction**: in addition to full-dict MSE,
  also reconstruct from the first ``h_size = d_sae // 5`` features
  (the "high-level" prefix). Total recon = L_H + L_full.
- **Multi-distance temporal InfoNCE**: anchor window + positives at
  ``shifts=[1, 2]``. Loss per shift = ``w_s * InfoNCE(z_anchor[:, :h], z_pos[:, :h])``
  where ``w_s = 1 / (1 + s)`` is the inverse-distance weighting.
- **Anti-dead stack** (same as txc_base): per-atom decoder unit-norm,
  decoder-parallel grad removal, AuxK on dead features (auxk_alpha=1/32),
  num_tokens_since_fired tracker.

Sourced (in spirit) from
``origin/han-phase7-unification @ 94119bc0:src/architectures/phase5b_subseq_sampling_txcdr.py::SubseqH8``
which inherits from ``TXCBareMultiDistanceContrastiveAntidead`` →
``TXCBareMatryoshkaContrastiveAntidead`` → ``TXCBareAntidead``. Here we
flatten the inheritance into a single class for the unified
``TempBenchArch`` framework. Docstring header attribution per
PROTOCOL.md § 2.

The arch's ``train_step(x: (B, seq_len, d_in))`` derives multi-window
batches internally so the canonical ``temp_bench.training.train_sae``
batch_iter can stay arch-agnostic.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.architectures.base import ArchConfig, TempBenchArch


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE between two latent batches; eps for stability."""
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


def _sample_contiguous_subset(
    T_max: int, t_sample: int, batch_size: int, device: torch.device,
) -> torch.Tensor:
    """Per-row contiguous t_sample-window inside [0, T_max).

    Returns (B, t_sample) int indices. Wasteland's "contiguous" mode;
    matches `phase5b_subseq_sampling_txcdr::_sample_subset_indices`.
    """
    max_off = T_max - t_sample + 1
    offs = torch.randint(0, max_off, (batch_size,), device=device)
    rng = torch.arange(t_sample, device=device)
    return offs.unsqueeze(1) + rng.unsqueeze(0)


@torch.no_grad()
def _geometric_median(points: torch.Tensor, max_iter: int = 100,
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


class TXCPro(TempBenchArch):
    """Subseq + matryoshka + multi-distance contrastive + anti-dead TXC.

    Args:
        d_in: residual-stream width.
        d_sae: dictionary size.
        T_max: maximum window length (encoder has T_max position slabs).
        t_sample: training-time subset size (must satisfy 1 ≤ t_sample ≤ T_max).
        k_pos: per-token sparsity. Window TopK budget = ``k_pos * t_sample``.
        n_matryoshka: nominal phase identifier (locked yaml uses 8).
            **NOT** functionally used as a count of matryoshka levels —
            we use the wasteland's H+full layout (h_size = d_sae // 5).
        contrastive_shifts: tuple of positive-window shift distances.
            Default (1, 2).
        contrastive_inverse_distance_weight: if True (default), per-shift
            weight = ``1 / (1 + s)``. Else uniform weights.
        contrastive_alpha: scaling on the InfoNCE term. Default 1.0.
        contr_prefix: prefix length for the InfoNCE cosine sim. Default
            ``h_size = d_sae // 5``.
        aux_k: budget of dead features per-sample in AuxK loss. Default 512.
        dead_threshold_tokens: tokens-since-fired to mark a feature dead.
        auxk_alpha: weight on AuxK term. Default 1/32 (paper).
        bdec_geom_median_init: if True, init b_dec via geometric median
            on the first training batch. Default True.
        decoder_unit_norm: ignored (always True). Kept for parity.
        decoder_grad_orthogonalize: ignored (always True). Kept for parity.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T_max: int = 10,
        t_sample: int = 5,
        k_pos: int = 20,
        n_matryoshka: int = 8,                # noqa: ARG002 — phase id, not used
        contrastive_shifts: tuple[int, ...] = (1, 2),
        contrastive_inverse_distance_weight: bool = True,
        contrastive_alpha: float = 1.0,
        contr_prefix: int | None = None,
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
        auxk_alpha: float = 1.0 / 32.0,
        multi_window: bool = False,
        bdec_geom_median_init: bool = True,
        decoder_unit_norm: bool = True,        # noqa: ARG002 — always True in this port
        decoder_grad_orthogonalize: bool = True,  # noqa: ARG002 — always True
        h_size: int | None = None,             # matryoshka prefix; None → d_sae//5
    ):
        super().__init__()
        self.config = ArchConfig(
            name="txc_pro", d_in=d_in, d_sae=d_sae,
            k_pos=k_pos, T=T_max,    # T (probe-time window) = T_max
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self.T_max = T_max
        self.t_sample = t_sample
        self.k_pos = k_pos
        self.k_train = k_pos * t_sample              # window TopK at train time
        self.k_inference = k_pos * T_max             # window TopK at probe time
        self.shifts = tuple(int(s) for s in contrastive_shifts)
        self.contrastive_alpha = float(contrastive_alpha)
        if contrastive_inverse_distance_weight:
            self.loss_weights = tuple(1.0 / (1.0 + s) for s in self.shifts)
        else:
            self.loss_weights = (1.0,) * len(self.shifts)
        # H prefix size (matryoshka) = d_sae // 5 (matches Phase 5b H8 default).
        # C2 overrides via h_size=d_sae to disable matryoshka at toy
        # d_sae=40 where d_sae//5=8 < k_train; otherwise k-sweep crashes.
        self.h_size = int(h_size) if h_size is not None else (d_sae // 5)
        if self.h_size > d_sae:
            raise ValueError(
                f"h_size={self.h_size} exceeds d_sae={d_sae}."
            )
        self.contr_prefix = int(contr_prefix) if contr_prefix is not None else self.h_size
        self.aux_k = aux_k
        self.dead_threshold_tokens = dead_threshold_tokens
        self.auxk_alpha = auxk_alpha
        # Multi-window sampling toggle (added 2026-05-05; see decisions.md § 14
        # "TXC training-FLOPs parity"). False = original 1-anchor-per-row
        # behavior, used by all in-flight cells. True = stride-(T_max+max_shift)
        # tiling that gives N anchor+positives groups per row, matching per-token
        # SAE token throughput per step. Toggling False→True via YAML hparam
        # invalidates train_keys (the hparam goes into compute_train_key's hash).
        self._multi_window = multi_window
        self.bdec_geom_median_init = bdec_geom_median_init

        assert 1 <= t_sample <= T_max
        assert all(s >= 1 for s in self.shifts)

        # Encoder (per-position weights, T_max slabs)
        self.W_enc = nn.Parameter(torch.empty(T_max, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        # Decoder
        self.W_dec = nn.Parameter(torch.empty(d_sae, T_max, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T_max, d_in))

        # Init: kaiming + unit-norm decoder + tied-encoder per position
        for t in range(T_max):
            nn.init.kaiming_uniform_(self.W_enc.data[t])
        nn.init.kaiming_uniform_(self.W_dec.data.view(d_sae, T_max * d_in))
        with torch.no_grad():
            self._normalize_decoder()
            for t in range(T_max):
                # Copy the t-th decoder slice transposed into W_enc[t].
                # Slice copy is a contiguous-buffer operation; the .T view
                # gets implicitly materialised.
                self.W_enc.data[t] = self.W_dec.data[:, t, :].T

        # Dead-feature tracker (one entry per latent)
        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(d_sae, dtype=torch.long),
        )
        # Diagnostic buffers
        self.register_buffer("last_auxk_loss", torch.tensor(-1.0))
        self.register_buffer("last_dead_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_recon_h", torch.tensor(-1.0))
        self.register_buffer("last_recon_full", torch.tensor(-1.0))
        self.register_buffer("last_contr", torch.tensor(-1.0))
        self.register_buffer("b_dec_initialized", torch.tensor(False))

        # Pre-step grad-parallel removal as a tensor hook on W_dec.
        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    # ── Decoder-norm utilities ──

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        """Unit-norm per decoder atom over (T_max, d_in)."""
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = (self.W_dec.data / norms).contiguous()

    @staticmethod
    def _project_dec_grad(param: torch.Tensor) -> None:
        """Project out the W_dec.grad component parallel to each atom."""
        if param.grad is None:
            return
        d_sae = param.shape[0]
        W_flat = param.data.view(d_sae, -1)
        g_flat = param.grad.data.view(d_sae, -1)
        normed = W_flat / (W_flat.norm(dim=1, keepdim=True) + 1e-6)
        parallel = (g_flat * normed).sum(dim=1, keepdim=True)
        g_flat.sub_(parallel * normed)

    # ── Encode / decode (TempBenchArch contract — probe-time uses FULL T_max) ──

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Probe-time encoding using ALL T_max positions (no subsampling).

        Input: (B, T_max, d_in) or (B, d_in) (treated as T=1 collapsed).
        Output: (B, 1, d_sae).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self.T_max:
            raise ValueError(
                f"TXCPro.encode expects (B, T_max={self.T_max}, d_in); "
                f"got T_input={x.shape[1]}."
            )
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        vals, idx = pre.topk(self.k_inference, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z.unsqueeze(1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode (B, 1, d_sae) or (B, d_sae) → (B, T_max, d_in)."""
        if z.dim() == 3:
            if z.shape[1] != 1:
                raise ValueError(
                    f"TXCPro.decode expects (B, 1, d_sae); got T={z.shape[1]}."
                )
            z = z.squeeze(1)
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def _decode_prefix(self, z: torch.Tensor, h_size: int) -> torch.Tensor:
        """Reconstruct from only the first h_size feature indices."""
        return (
            torch.einsum("bs,std->btd", z[:, :h_size], self.W_dec[:h_size])
            + self.b_dec
        )

    # ── Subseq encoder (training-time, with t_sample subset) ──

    def _pre_activation_sampled(
        self, x: torch.Tensor, sample_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-activation summed only over the sampled subset positions.

        x: (B, T_max, d_in). sample_idx: (B, t_sample) in [0, T_max).
        Returns (B, d_sae). Implementation: zero-mask + standard einsum;
        avoids the O(B * d_in * d_sae) per-row gather of W_enc.
        """
        B, T_max, d = x.shape
        mask = torch.zeros(B, T_max, device=x.device, dtype=x.dtype)
        mask.scatter_(1, sample_idx, 1.0)
        x_masked = x * mask.unsqueeze(-1)
        return torch.einsum("btd,tds->bs", x_masked, self.W_enc) + self.b_enc

    def _recon_sampled_matryoshka(
        self, x: torch.Tensor, z: torch.Tensor, sample_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matryoshka (H + full) MSE on sampled positions only.

        Returns (l_recon, l_h).
        """
        x_hat_full = self.decode(z)
        x_hat_h = self._decode_prefix(z, self.h_size)
        B, T_max, d = x.shape
        gi = sample_idx.unsqueeze(-1).expand(-1, -1, d)
        x_S = x.gather(1, gi)
        l_full = (x_S - x_hat_full.gather(1, gi)).pow(2).sum(dim=-1).mean()
        l_h = (x_S - x_hat_h.gather(1, gi)).pow(2).sum(dim=-1).mean()
        return l_h + l_full, l_h

    # ── train_step: derive multi-window batches from a single sequence ──

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Args:
            x: (B, seq_len, d_in) from the canonical batch_iter.
               We sample one offset per batch element and extract
               anchor + positive shift windows of length T_max.
               Requires ``seq_len >= T_max + max(shifts)``.

        Returns:
            (loss, info) — info has 'mse', 'l0', 'auxk', 'dead', 'recon_h',
            'contrastive', 'z'.
        """
        if x.dim() != 3:
            raise ValueError(
                f"TXCPro.train_step expects (B, seq_len, d_in); got {tuple(x.shape)}"
            )
        B, seq_len, _ = x.shape
        max_shift = max(self.shifts) if self.shifts else 0
        min_seq = self.T_max + max_shift
        if seq_len < min_seq:
            raise ValueError(
                f"TXCPro.train_step needs seq_len >= T_max + max_shift = {min_seq}; "
                f"got seq_len={seq_len}."
            )

        # b_dec geometric-median init on the first batch
        if self.bdec_geom_median_init and not bool(self.b_dec_initialized):
            with torch.no_grad():
                # Use the leading T_max positions to seed b_dec[t] per t.
                for t in range(self.T_max):
                    med = _geometric_median(x[:, t, :].float())
                    self.b_dec.data[t] = med.to(self.b_dec.dtype)
                self.b_dec_initialized.fill_(True)

        # Anchor + positive-shift gather. Mode controlled by multi_window:
        #   False (default, original): 1 random anchor per batch row → (B, T_max, d).
        #   True (opt-in 2026-05-05): tile each row at stride=min_seq into
        #     N = seq_len // min_seq non-overlapping anchor+positive groups,
        #     giving (B*N, T_max, d) effective rows. Matches per-token SAE
        #     token throughput per step.
        arange_T = torch.arange(self.T_max, device=x.device)
        if self._multi_window:
            N = seq_len // min_seq
            starts = torch.arange(N, device=x.device) * min_seq            # (N,)
            anchor_pos = starts.unsqueeze(1) + arange_T.unsqueeze(0)        # (N, T_max)
            anchor_pos_b = anchor_pos.unsqueeze(0).expand(B, -1, -1)        # (B, N, T_max)
            batch_idx_bn = (
                torch.arange(B, device=x.device).unsqueeze(1).unsqueeze(2)
                .expand(-1, N, self.T_max)                                  # (B, N, T_max)
            )
            x_anchor = x[batch_idx_bn, anchor_pos_b].reshape(B * N, self.T_max, -1)

            def gather_window(shift: int) -> torch.Tensor:
                pos = (starts.unsqueeze(1) + shift + arange_T.unsqueeze(0))   # (N, T_max)
                pos_b = pos.unsqueeze(0).expand(B, -1, -1)
                return x[batch_idx_bn, pos_b].reshape(B * N, self.T_max, -1)

            x_positives = [gather_window(s) for s in self.shifts]
            effective_B = B * N
        else:
            offsets = torch.randint(
                0, seq_len - min_seq + 1, (B,), device=x.device,
            )
            batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, self.T_max)

            def gather_window(shift: int) -> torch.Tensor:
                idx_t = offsets.unsqueeze(1) + shift + arange_T.unsqueeze(0)
                return x[batch_idx, idx_t]

            x_anchor = gather_window(0)
            x_positives = [gather_window(s) for s in self.shifts]
            effective_B = B

        # Single shared subset S applied to all windows (matches wasteland);
        # under multi_window=True, each effective row gets its own subset.
        sample_idx = _sample_contiguous_subset(
            self.T_max, self.t_sample, effective_B, x.device,
        )

        # Anchor: subset-summed encoder + topk
        pre_a = self._pre_activation_sampled(x_anchor, sample_idx)
        vals_a, idx_a = pre_a.topk(self.k_train, dim=-1)
        z_anchor = torch.zeros_like(pre_a)
        z_anchor.scatter_(1, idx_a, F.relu(vals_a))

        l_recon, l_h = self._recon_sampled_matryoshka(x_anchor, z_anchor, sample_idx)

        # Positives: each contributes its own recon + InfoNCE
        l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
        for k_idx, x_pos in enumerate(x_positives):
            pre_p = self._pre_activation_sampled(x_pos, sample_idx)
            vals_p, idx_p = pre_p.topk(self.k_train, dim=-1)
            z_pos = torch.zeros_like(pre_p)
            z_pos.scatter_(1, idx_p, F.relu(vals_p))
            l_recon_p, _ = self._recon_sampled_matryoshka(x_pos, z_pos, sample_idx)
            l_recon = l_recon + l_recon_p
            if self.contrastive_alpha > 0.0:
                w_s = self.loss_weights[k_idx]
                l_contr = l_contr + w_s * _info_nce(
                    z_anchor[:, :self.contr_prefix],
                    z_pos[:, :self.contr_prefix],
                )

        # AuxK on the anchor path
        x_hat_anchor = self.decode(z_anchor)
        l_auxk = self._update_dead_and_auxk_sampled(
            x_anchor, x_hat_anchor, pre_a, z_anchor, sample_idx,
        )

        total = l_recon + self.contrastive_alpha * l_contr + self.auxk_alpha * l_auxk

        # Diagnostics
        with torch.no_grad():
            self.last_recon_h.fill_(float(l_h.detach()))
            self.last_recon_full.fill_(float((l_recon - l_h).detach() / max(1, len(self.shifts))))
            self.last_contr.fill_(float(l_contr.detach()))

        # Probe-time-shaped z for the trainer log + Bricken
        z_log = z_anchor.unsqueeze(1)         # (B, 1, d_sae)
        l0 = (z_anchor != 0).float().sum(dim=-1).mean()
        info = {
            "mse": l_recon.detach(),
            "l0": l0.detach(),
            "auxk": l_auxk.detach(),
            "dead": float(self.last_dead_count.item()),
            "recon_h": l_h.detach(),
            "contrastive": l_contr.detach(),
            "z": z_log.detach(),
        }
        return total, info

    def _update_dead_and_auxk_sampled(
        self, x: torch.Tensor, x_hat: torch.Tensor,
        pre: torch.Tensor, z: torch.Tensor, sample_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Update num_tokens_since_fired + compute AuxK loss on sampled
        residual (matches wasteland SubseqH8._update_dead_and_auxk_sampled).
        """
        with torch.no_grad():
            active_mask = (z > 0).any(dim=0)
            n_tokens = x.shape[0] * self.t_sample
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

        B, T_max, d = x.shape
        gi = sample_idx.unsqueeze(-1).expand(-1, -1, d)
        x_S = x.gather(1, gi)
        x_hat_S = x_hat.detach().gather(1, gi)
        aux_S = aux_decode.gather(1, gi)
        residual = x_S - x_hat_S
        l2_a = (residual - aux_S).pow(2).sum(dim=-1).mean()
        mu = residual.mean(dim=(0, 1), keepdim=True)
        denom = (residual - mu).pow(2).sum(dim=-1).mean()
        l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        with torch.no_grad():
            self.last_auxk_loss.fill_(float(l_auxk.detach()))
        return l_auxk

    # ── post_step (decoder unit-norm renormalisation) ──

    def post_step(self) -> None:
        with torch.no_grad():
            self._normalize_decoder()

    # ── decoder_directions for C4 (T-averaged) ──

    def decoder_directions(self) -> torch.Tensor:
        """(d_sae, d_in) — average decoder direction across T_max positions."""
        return self.W_dec.data.mean(dim=1).clone()
