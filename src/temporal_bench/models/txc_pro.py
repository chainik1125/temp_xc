"""TXC-pro — Han's locked subseq + matryoshka + multi-distance contrastive + anti-dead.

Port of ``purified/src/temp_bench/architectures/txc_pro.py`` (origin/final)
into Bill's ``TemporalAE`` interface.

Architecture (locked, configs/locked_archs.yaml::txc_pro):
    d_sae = 8 * d_in (paper expansion)
    T_max = 10, t_sample = 5
    k_pos = 20  -> k_train = k_pos*t_sample = 100;  k_inference = k_pos*T_max = 200
    n_matryoshka = 8 (phase id; H+full layout via h_size = d_sae // 5)
    contrastive_shifts = (1, 2)
    contrastive_inverse_distance_weight = True  (w_s = 1/(1+s))
    auxk_alpha = 1/32
    dead_threshold_tokens = 10_000_000
    decoder_unit_norm + decoder_grad_orthogonalize
    bdec_geom_median_init = True

Interface dispatch in :meth:`forward`:
    Input shape (B, T_input, d_in):
        T_input == T_max + max_shift (= 12 by default) -> training pass:
            anchor + positives, matryoshka recon, multi-distance InfoNCE, AuxK.
        T_input == T_max (= 10)                       -> eval pass:
            full-window encode + decode at k_inference, plain MSE.
    Anything else raises.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutput, TemporalAE


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


def _sample_contiguous_subset(
    T_max: int, t_sample: int, batch_size: int, device: torch.device,
) -> torch.Tensor:
    max_off = T_max - t_sample + 1
    offs = torch.randint(0, max_off, (batch_size,), device=device)
    rng = torch.arange(t_sample, device=device)
    return offs.unsqueeze(1) + rng.unsqueeze(0)


@torch.no_grad()
def _geometric_median(points: torch.Tensor, max_iter: int = 100,
                      tol: float = 1e-5) -> torch.Tensor:
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess
        weights = 1.0 / (torch.norm(points - guess, dim=1) + 1e-8)
        weights = weights / weights.sum()
        guess = (weights.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


class TXCPro(TemporalAE):
    """Subseq + matryoshka + multi-distance contrastive + anti-dead TXC."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T_max: int = 10,
        t_sample: int = 5,
        k_pos: int = 20,
        contrastive_shifts: tuple[int, ...] = (1, 2),
        contrastive_inverse_distance_weight: bool = True,
        contrastive_alpha: float = 1.0,
        contr_prefix: int | None = None,
        aux_k: int = 512,
        dead_threshold_tokens: int = 10_000_000,
        auxk_alpha: float = 1.0 / 32.0,
        bdec_geom_median_init: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T_max = T_max
        self.t_sample = t_sample
        self.k_pos = k_pos
        self.k_train = k_pos * t_sample
        self.k_inference = k_pos * T_max
        self.shifts = tuple(int(s) for s in contrastive_shifts)
        self.max_shift = max(self.shifts) if self.shifts else 0
        self.train_window = self.T_max + self.max_shift
        self.contrastive_alpha = float(contrastive_alpha)
        if contrastive_inverse_distance_weight:
            self.loss_weights = tuple(1.0 / (1.0 + s) for s in self.shifts)
        else:
            self.loss_weights = (1.0,) * len(self.shifts)
        self.h_size = d_sae // 5
        self.contr_prefix = int(contr_prefix) if contr_prefix is not None else self.h_size
        self.aux_k = aux_k
        self.dead_threshold_tokens = dead_threshold_tokens
        self.auxk_alpha = auxk_alpha
        self.bdec_geom_median_init = bdec_geom_median_init

        assert 1 <= t_sample <= T_max
        assert all(s >= 1 for s in self.shifts)

        self.W_enc = nn.Parameter(torch.empty(T_max, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, T_max, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T_max, d_in))

        for t in range(T_max):
            nn.init.kaiming_uniform_(self.W_enc.data[t])
        nn.init.kaiming_uniform_(self.W_dec.data.view(d_sae, T_max * d_in))
        with torch.no_grad():
            self._normalize_decoder_inplace()
            for t in range(T_max):
                self.W_enc.data[t] = self.W_dec.data[:, t, :].T

        self.register_buffer(
            "num_tokens_since_fired",
            torch.zeros(d_sae, dtype=torch.long),
        )
        self.register_buffer("b_dec_initialized", torch.tensor(False))

        self.W_dec.register_post_accumulate_grad_hook(self._project_dec_grad)

    @torch.no_grad()
    def _normalize_decoder_inplace(self) -> None:
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = (self.W_dec.data / norms).contiguous()

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

    def _decode_full(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def _decode_prefix(self, z: torch.Tensor, h_size: int) -> torch.Tensor:
        return (
            torch.einsum("bs,std->btd", z[:, :h_size], self.W_dec[:h_size])
            + self.b_dec
        )

    def _pre_activation_sampled(
        self, x: torch.Tensor, sample_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, T_max, d = x.shape
        mask = torch.zeros(B, T_max, device=x.device, dtype=x.dtype)
        mask.scatter_(1, sample_idx, 1.0)
        x_masked = x * mask.unsqueeze(-1)
        return torch.einsum("btd,tds->bs", x_masked, self.W_enc) + self.b_enc

    def _topk_window(self, pre: torch.Tensor, k: int) -> torch.Tensor:
        vals, idx = pre.topk(k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z

    def _eval_forward(self, x: torch.Tensor) -> ModelOutput:
        """Probe-time forward: full T_max window, k_inference TopK."""
        B, T, d = x.shape
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        z = self._topk_window(pre, self.k_inference)
        x_hat = self._decode_full(z)
        recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        latents = z.unsqueeze(1).expand(B, T, self.d_sae)
        l0 = (z != 0).float().sum(dim=-1).mean().item()
        return ModelOutput(
            x_hat=x_hat,
            latents=latents,
            loss=recon,
            metrics={"recon_loss": recon.item(), "l0": l0},
        )

    def _train_forward(self, x: torch.Tensor) -> ModelOutput:
        """Training-time forward: anchor + positives, matryoshka + InfoNCE + AuxK."""
        B, T_input, d = x.shape
        x_anchor = x[:, : self.T_max, :]
        positives = [x[:, s : s + self.T_max, :] for s in self.shifts]

        # b_dec geometric-median init on first batch
        if self.bdec_geom_median_init and not bool(self.b_dec_initialized):
            with torch.no_grad():
                for t in range(self.T_max):
                    med = _geometric_median(x_anchor[:, t, :].float())
                    self.b_dec.data[t] = med.to(self.b_dec.dtype)
                self.b_dec_initialized.fill_(True)

        sample_idx = _sample_contiguous_subset(
            self.T_max, self.t_sample, B, x.device,
        )
        gather_idx = sample_idx.unsqueeze(-1).expand(-1, -1, self.d_in)

        # Anchor
        pre_a = self._pre_activation_sampled(x_anchor, sample_idx)
        z_anchor = self._topk_window(pre_a, self.k_train)
        x_hat_full_a = self._decode_full(z_anchor)
        x_hat_h_a = self._decode_prefix(z_anchor, self.h_size)
        x_S_a = x_anchor.gather(1, gather_idx)
        l_full_a = (x_S_a - x_hat_full_a.gather(1, gather_idx)).pow(2).sum(dim=-1).mean()
        l_h_a = (x_S_a - x_hat_h_a.gather(1, gather_idx)).pow(2).sum(dim=-1).mean()
        l_recon = l_full_a + l_h_a

        # Positives + InfoNCE
        l_contr = torch.zeros((), device=x.device, dtype=x.dtype)
        for k_idx, x_pos in enumerate(positives):
            pre_p = self._pre_activation_sampled(x_pos, sample_idx)
            z_pos = self._topk_window(pre_p, self.k_train)
            x_hat_full_p = self._decode_full(z_pos)
            x_hat_h_p = self._decode_prefix(z_pos, self.h_size)
            x_S_p = x_pos.gather(1, gather_idx)
            l_full_p = (x_S_p - x_hat_full_p.gather(1, gather_idx)).pow(2).sum(dim=-1).mean()
            l_h_p = (x_S_p - x_hat_h_p.gather(1, gather_idx)).pow(2).sum(dim=-1).mean()
            l_recon = l_recon + l_full_p + l_h_p
            if self.contrastive_alpha > 0.0:
                w_s = self.loss_weights[k_idx]
                l_contr = l_contr + w_s * _info_nce(
                    z_anchor[:, : self.contr_prefix],
                    z_pos[:, : self.contr_prefix],
                )

        # AuxK on anchor path
        with torch.no_grad():
            active_mask = (z_anchor > 0).any(dim=0)
            n_tokens = B * self.t_sample
            self.num_tokens_since_fired += n_tokens
            self.num_tokens_since_fired[active_mask] = 0
            dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
            n_dead = int(dead_mask.sum().item())

        if n_dead > 0:
            k_aux = min(self.aux_k, n_dead)
            auxk_pre = F.relu(pre_a).masked_fill(~dead_mask.unsqueeze(0), 0.0)
            vals_a, idx_a = auxk_pre.topk(k_aux, dim=-1, sorted=False)
            aux_buf = torch.zeros_like(pre_a)
            aux_buf.scatter_(-1, idx_a, vals_a)
            aux_decode = torch.einsum("bs,std->btd", aux_buf, self.W_dec)
            x_hat_S = x_hat_full_a.detach().gather(1, gather_idx)
            aux_S = aux_decode.gather(1, gather_idx)
            residual = x_S_a - x_hat_S
            l2_a = (residual - aux_S).pow(2).sum(dim=-1).mean()
            mu = residual.mean(dim=(0, 1), keepdim=True)
            denom = (residual - mu).pow(2).sum(dim=-1).mean()
            l_auxk = (l2_a / denom.clamp(min=1e-8)).nan_to_num(0.0)
        else:
            l_auxk = torch.zeros((), device=x.device, dtype=x.dtype)

        loss = l_recon + self.contrastive_alpha * l_contr + self.auxk_alpha * l_auxk

        # Latents reported on the anchor; broadcast to (B, T_max, d_sae)
        latents = z_anchor.unsqueeze(1).expand(B, self.T_max, self.d_sae)
        l0 = (z_anchor != 0).float().sum(dim=-1).mean().item()

        return ModelOutput(
            x_hat=x_hat_full_a,
            latents=latents,
            loss=loss,
            metrics={
                "recon_loss": (l_recon).item(),
                "recon_h": l_h_a.item(),
                "contrastive": float(l_contr.detach()),
                "auxk_loss": float(l_auxk.detach()),
                "dead": n_dead,
                "l0": l0,
            },
        )

    def forward(self, x: torch.Tensor) -> ModelOutput:
        B, T_input, d = x.shape
        assert d == self.d_in
        if T_input == self.T_max:
            return self._eval_forward(x)
        if T_input == self.train_window:
            return self._train_forward(x)
        raise ValueError(
            f"TXCPro.forward got T_input={T_input}; expected "
            f"T_max={self.T_max} (eval) or T_max+max_shift={self.train_window} (train)."
        )

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder_inplace()

    def decoder_directions(self, pos: int | None = None) -> torch.Tensor:
        if pos is not None:
            return self.W_dec.data[:, pos, :].T
        return self.W_dec.data.mean(dim=1).T

    @property
    def n_positions(self) -> int:
        return self.T_max
