"""txc_pro_r1 — revival of the recovered TXC-pro (phase5b_subseq_h8) as NEW ids.

Source of truth: ``docs/recovered/txc_pro_phase5b_subseq_h8.py`` (verbatim
copy of blob 480f3755d, the last committed ``txc_pro.py`` before its
removal in 5dd7337b2 — see ``task_hunt/TXC_PRO_RECOVERY.md``). The
deprecated ``txc_pro`` id and its DEPRECATED_ARCHS render filters are NOT
touched; these are fresh registry ids for the tscale exploration
(``experiments/explorations/tscale/CARD_SPLIT.md`` § 4).

Recipe (unchanged): subseq encoder (contiguous t_sample-of-T_max training
subsets, full window at probe) + matryoshka H+full reconstruction
(h_size = d_sae//5) + multi-distance InfoNCE (shifts (1,2),
inverse-distance weights, H-prefix sims) + anti-dead stack (decoder
unit-norm per atom, decoder-parallel grad projection, AuxK on dead
features, geometric-median b_dec init).

Deviations from the recovered file, each tagged ``# r1:`` inline:

1. ``T`` constructor alias for ``T_max`` (the grid convention
   ``arch_hparams_override={"T": t}``; both given → error).
2. ``t_sample=None`` default derives the CARD § 4 RATIO rule
   ``max(1, T_max // 2)`` — the locked instance (T_max=10 → 5) is this
   rule's fixed point; pass an explicit int for absolute-t ablations.
3. ``eval_consumes = 'window'`` class attr: trains on sequences
   (``consumes='sequence'``, unchanged) but encodes fixed
   (B, T_max, d_in) windows at probe — the probing eval dispatches its
   window path on this attr (CARD § 4 seam; probing.py change is
   byte-identical for archs that do not declare it).
4. ``_sparsify`` seam factored out of encode/train_step so the btk-only
   twin below can drop the ReLU without touching anything else.
5. ``relu_mode`` hparam threaded + asserted (arm hashes into keys), the
   btk_only.py convention.

``TXCProR1`` — faithful composition: per-sample TopK then ReLU on the
selected values (paper family; selected negatives zero out ⇒ realized
l0 ≤ nominal, the composition's own fingerprint). ``TXCProR1BTKOnly`` —
selection over raw pre-acts by signed value, survivors pass signed, no
ReLU in the sparsity path; fired ⇔ z != 0; AuxK unchanged (outside the
sparsity path) per the mac-a btk-only convention.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


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
    """Per-row contiguous t_sample-window inside [0, T_max)."""
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


class TXCProR1(TempBenchArch):
    """Subseq + matryoshka + multi-distance contrastive + anti-dead TXC.

    Faithful recovered composition: TopK on pre-acts, ReLU on the
    selected values. See module docstring for the r1 deviations.
    """

    arch_version: str = "1.0.0"
    consumes: str = "sequence"       # training serving (SequenceBuffer)
    eval_consumes: str = "window"    # r1: probe-time dispatch (CARD § 4 seam)

    REGISTRY_NAME = "txc_pro_r1"
    RELU_MODE = "paper-match"        # r1: composition arm this class implements

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T_max: int | None = None,
        T: int | None = None,                 # r1: grid-convention alias
        t_sample: int | None = None,          # r1: None → max(1, T_max//2)
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
        decoder_unit_norm: bool = True,        # noqa: ARG002 — always True
        decoder_grad_orthogonalize: bool = True,  # noqa: ARG002 — always True
        h_size: int | None = None,             # matryoshka prefix; None → d_sae//5
        relu_mode: str | None = None,          # r1: arm assert (hashes into keys)
        k_anneal_mult: float = 1.0,            # r1-c4: >1 widens k_train early
        k_anneal_steps: int = 0,               # r1-c4: linear wide→nominal over N steps
    ):
        super().__init__()
        # r1: resolve the T alias (grid override convention).
        if T is not None and T_max is not None and int(T) != int(T_max):
            raise ValueError(f"Give T or T_max, not both (T={T}, T_max={T_max}).")
        T_max = int(T if T is not None else (T_max if T_max is not None else 10))
        # r1: ratio rule default (CARD § 4): locked instance 10 → 5.
        t_sample = int(t_sample) if t_sample is not None else max(1, T_max // 2)
        if relu_mode is not None and relu_mode != self.RELU_MODE:
            raise ValueError(
                f"{type(self).__name__} implements relu_mode={self.RELU_MODE!r}; "
                f"got {relu_mode!r}. Use the other registry id for the other arm."
            )

        self.config = ArchConfig(
            name=self.REGISTRY_NAME, d_in=d_in, d_sae=d_sae,
            k_pos=k_pos, T=T_max,    # T (probe-time window) = T_max
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self.T_max = T_max
        self.t_sample = t_sample
        self.k_pos = k_pos
        self.relu_mode = self.RELU_MODE
        # Clip at d_sae for toy benches where k_pos * T can exceed dict size.
        self.k_train = min(k_pos * t_sample, d_sae)
        self.k_inference = min(k_pos * T_max, d_sae)
        # r1-c4: k_train anneal (C4 low-T fix — wide early admission spreads
        # gradient across the dictionary to fight across-row latent
        # concentration; defaults are OFF = pre-C4 behavior, bit-identical).
        # Progress is a plain attr, NOT persisted: scratch L1 screens never
        # resume mid-training, and state_dict/ckpt compat stays untouched.
        self.k_anneal_mult = float(k_anneal_mult)
        self.k_anneal_steps = int(k_anneal_steps)
        if self.k_anneal_mult < 1.0 or self.k_anneal_steps < 0:
            raise ValueError(
                f"k_anneal_mult must be ≥ 1 and k_anneal_steps ≥ 0; got "
                f"{k_anneal_mult}, {k_anneal_steps}."
            )
        self._anneal_step = 0
        self.shifts = tuple(int(s) for s in contrastive_shifts)
        self.contrastive_alpha = float(contrastive_alpha)
        if contrastive_inverse_distance_weight:
            self.loss_weights = tuple(1.0 / (1.0 + s) for s in self.shifts)
        else:
            self.loss_weights = (1.0,) * len(self.shifts)
        # H prefix size (matryoshka) = d_sae // 5 (Phase 5b H8 default).
        self.h_size = int(h_size) if h_size is not None else (d_sae // 5)
        if self.h_size > d_sae:
            raise ValueError(f"h_size={self.h_size} exceeds d_sae={d_sae}.")
        self.contr_prefix = int(contr_prefix) if contr_prefix is not None else self.h_size
        self.aux_k = aux_k
        self.dead_threshold_tokens = dead_threshold_tokens
        self.auxk_alpha = auxk_alpha
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
                self.W_enc.data[t] = self.W_dec.data[:, t, :].T

        self.register_buffer(
            "num_tokens_since_fired", torch.zeros(d_sae, dtype=torch.long),
        )
        self.register_buffer("last_auxk_loss", torch.tensor(-1.0))
        self.register_buffer("last_dead_count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_recon_h", torch.tensor(-1.0))
        self.register_buffer("last_recon_full", torch.tensor(-1.0))
        self.register_buffer("last_contr", torch.tensor(-1.0))
        self.register_buffer("last_neg_frac", torch.tensor(0.0))   # r1: diagnostic
        self.register_buffer("b_dec_initialized", torch.tensor(False))

        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    # ── Sparsify seam (r1: the ONLY composition-dependent code path) ──

    def _k_train_now(self) -> int:
        """r1-c4: annealed training admission — linear from mult·k_train
        down to k_train over k_anneal_steps train_step calls, then constant.
        The serve path (k_inference) is never touched."""
        if self.k_anneal_mult <= 1.0 or self.k_anneal_steps <= 0:
            return self.k_train
        s = self._anneal_step
        if s >= self.k_anneal_steps:
            return self.k_train
        frac = 1.0 - s / self.k_anneal_steps
        k = round(self.k_train * (1.0 + (self.k_anneal_mult - 1.0) * frac))
        return min(max(int(k), self.k_train), self._d_sae)

    def _sparsify(self, pre: torch.Tensor, k: int) -> torch.Tensor:
        """Faithful composition: TopK by value, then ReLU on survivors."""
        vals, idx = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z

    def _fired_mask(self, z: torch.Tensor) -> torch.Tensor:
        """Faithful composition: fired ⇔ z > 0 (post-ReLU convention)."""
        return (z > 0).any(dim=0)

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

    # ── Encode / decode (probe-time uses FULL T_max) ──

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Probe-time encoding using ALL T_max positions (no subsampling).

        Input: (B, T_max, d_in) or (B, d_in) (treated as T=1 collapsed).
        Output: (B, 1, d_sae).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self.T_max:
            raise ValueError(
                f"{type(self).__name__}.encode expects (B, T_max={self.T_max}, "
                f"d_in); got T_input={x.shape[1]}."
            )
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        z = self._sparsify(pre, self.k_inference)
        return z.unsqueeze(1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode (B, 1, d_sae) or (B, d_sae) → (B, T_max, d_in)."""
        if z.dim() == 3:
            if z.shape[1] != 1:
                raise ValueError(
                    f"{type(self).__name__}.decode expects (B, 1, d_sae); "
                    f"got T={z.shape[1]}."
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
        """Pre-activation summed only over the sampled subset positions."""
        B, T_max, d = x.shape
        mask = torch.zeros(B, T_max, device=x.device, dtype=x.dtype)
        mask.scatter_(1, sample_idx, 1.0)
        x_masked = x * mask.unsqueeze(-1)
        return torch.einsum("btd,tds->bs", x_masked, self.W_enc) + self.b_enc

    def _recon_sampled_matryoshka(
        self, x: torch.Tensor, z: torch.Tensor, sample_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matryoshka (H + full) MSE on sampled positions only."""
        x_hat_full = self.decode(z)
        x_hat_h = self._decode_prefix(z, self.h_size)
        B, T_max, d = x.shape
        gi = sample_idx.unsqueeze(-1).expand(-1, -1, d)
        x_S = x.gather(1, gi)
        l_full = (x_S - x_hat_full.gather(1, gi)).pow(2).sum(dim=-1).mean()
        l_h = (x_S - x_hat_h.gather(1, gi)).pow(2).sum(dim=-1).mean()
        return l_h + l_full, l_h

    # ── train_step: derive anchor+positive windows from sequences ──

    def train_step(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if x.dim() != 3:
            raise ValueError(
                f"{type(self).__name__}.train_step expects (B, seq_len, d_in); "
                f"got {tuple(x.shape)}"
            )
        B, seq_len, _ = x.shape
        max_shift = max(self.shifts) if self.shifts else 0
        min_seq = self.T_max + max_shift
        if seq_len < min_seq:
            raise ValueError(
                f"{type(self).__name__}.train_step needs seq_len >= "
                f"T_max + max_shift = {min_seq}; got seq_len={seq_len}."
            )

        if self.bdec_geom_median_init and not bool(self.b_dec_initialized):
            with torch.no_grad():
                for t in range(self.T_max):
                    med = _geometric_median(x[:, t, :].float())
                    self.b_dec.data[t] = med.to(self.b_dec.dtype)
                self.b_dec_initialized.fill_(True)

        arange_T = torch.arange(self.T_max, device=x.device)
        if self._multi_window:
            N = seq_len // min_seq
            starts = torch.arange(N, device=x.device) * min_seq
            anchor_pos = starts.unsqueeze(1) + arange_T.unsqueeze(0)
            anchor_pos_b = anchor_pos.unsqueeze(0).expand(B, -1, -1)
            batch_idx_bn = (
                torch.arange(B, device=x.device).unsqueeze(1).unsqueeze(2)
                .expand(-1, N, self.T_max)
            )
            x_anchor = x[batch_idx_bn, anchor_pos_b].reshape(B * N, self.T_max, -1)

            def gather_window(shift: int) -> torch.Tensor:
                pos = (starts.unsqueeze(1) + shift + arange_T.unsqueeze(0))
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

        sample_idx = _sample_contiguous_subset(
            self.T_max, self.t_sample, effective_B, x.device,
        )

        pre_a = self._pre_activation_sampled(x_anchor, sample_idx)
        k_now = self._k_train_now()            # r1-c4: annealed admission
        self._anneal_step += 1
        z_anchor = self._sparsify(pre_a, k_now)

        l_recon, l_h = self._recon_sampled_matryoshka(x_anchor, z_anchor, sample_idx)

        l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
        for k_idx, x_pos in enumerate(x_positives):
            pre_p = self._pre_activation_sampled(x_pos, sample_idx)
            z_pos = self._sparsify(pre_p, k_now)   # r1-c4: same admission as anchor
            l_recon_p, _ = self._recon_sampled_matryoshka(x_pos, z_pos, sample_idx)
            l_recon = l_recon + l_recon_p
            if self.contrastive_alpha > 0.0:
                w_s = self.loss_weights[k_idx]
                l_contr = l_contr + w_s * _info_nce(
                    z_anchor[:, :self.contr_prefix],
                    z_pos[:, :self.contr_prefix],
                )

        x_hat_anchor = self.decode(z_anchor)
        l_auxk = self._update_dead_and_auxk_sampled(
            x_anchor, x_hat_anchor, pre_a, z_anchor, sample_idx,
        )

        total = l_recon + self.contrastive_alpha * l_contr + self.auxk_alpha * l_auxk

        with torch.no_grad():
            self.last_recon_h.fill_(float(l_h.detach()))
            self.last_recon_full.fill_(
                float((l_recon - l_h).detach() / max(1, len(self.shifts)))
            )
            self.last_contr.fill_(float(l_contr.detach()))
            nz = (z_anchor != 0).sum().float().clamp(min=1.0)   # r1: neg_frac diag
            self.last_neg_frac.fill_(float((z_anchor < 0).sum().float() / nz))

        z_log = z_anchor.unsqueeze(1)
        l0 = (z_anchor != 0).float().sum(dim=-1).mean()
        info = {
            "mse": l_recon.detach(),
            "l0": l0.detach(),
            "auxk": l_auxk.detach(),
            "dead": float(self.last_dead_count.item()),
            "recon_h": l_h.detach(),
            "contrastive": l_contr.detach(),
            "neg_frac": self.last_neg_frac.clone(),   # r1
            "z": z_log.detach(),
        }
        return total, info

    def _update_dead_and_auxk_sampled(
        self, x: torch.Tensor, x_hat: torch.Tensor,
        pre: torch.Tensor, z: torch.Tensor, sample_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Dead tracking + AuxK on sampled residual. AuxK operates on
        ReLU'd pre-acts in BOTH arms (outside the sparsity path — the
        btk-only convention holds it constant)."""
        with torch.no_grad():
            active_mask = self._fired_mask(z)
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

    # ── decoder_directions (T-averaged) ──

    def decoder_directions(self) -> torch.Tensor:
        """(d_sae, d_in) — average decoder direction across T_max positions."""
        return self.W_dec.data.mean(dim=1).clone()


class TXCProR1BTKOnly(TXCProR1):
    """btk-only twin: selection over RAW pre-acts by signed value, survivors
    pass through SIGNED — no ReLU anywhere in the sparsity path (mac-a's
    convention, btk_only.py items 1/3/4: fired ⇔ z != 0; AuxK unchanged).
    Per-sample TopK has no EMA-threshold eval path, so item 2 is moot —
    realized l0 == nominal exactly by construction.
    """

    REGISTRY_NAME = "txc_pro_r1_btkonly"
    RELU_MODE = "btk-only"

    def _sparsify(self, pre: torch.Tensor, k: int) -> torch.Tensor:
        vals, idx = pre.topk(k, dim=-1)          # btk-only: signed-value selection
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, vals)                 # btk-only: no ReLU on survivors
        return z

    def _fired_mask(self, z: torch.Tensor) -> torch.Tensor:
        return (z != 0).any(dim=0)               # btk-only: negative firing is alive
