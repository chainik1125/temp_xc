"""Agentic TXC-02 — the paper's RLHF TXC arm, TRAINABLE port.

Vendors `MatryoshkaTXCDRContrastiveMultiscale` and its ancestry
(`MatryoshkaTXCDRContrastive`, `PositionMatryoshkaTXCDR`, `_info_nce`)
verbatim from `origin/han-phase7-unification @ 94119bc08`
(`src/architectures/matryoshka_txcdr{,_contrastive,_contrastive_multiscale}.py`),
the pin quoted by COMPOSITION_AUDIT § 6: `agentic_txc_02` =
MatryoshkaTXCDRContrastiveMultiscale(T=5, k_win=500=100·T, scales 3,
γ=0.5), encode = einsum→topk(k_win)→scatter(F.relu) ⇒ TopK→ReLU
per-window. Unlike `paper_v1.py` (eval-only adapters) this port TRAINS,
reproducing the paper's recorded procedure
(`experiments/phase5_downstream_utility/train_primary_archs.py` +
`results/training_logs/agentic_txc_02__seed*.json` at the same pin):

- adjacent T-window pairs, shift = 1 token (`make_pair_window_gen_gpu`);
- multiscale InfoNCE at scales 1..3, γ=0.5, α=1.0 on top of the nested
  matryoshka reconstruction;
- Adam lr 3e-4, batch 1024 pairs, grad_clip 1.0 (trainer-side), decoder
  unit-norm after every step;
- PLATEAU EARLY-STOP mirrored IN-PLUGIN: loss logged every 200 steps;
  stop when (prior5̄ − recent5̄)/|prior5̄| < 0.02 after ≥ 3000 steps
  (upstream anchors converged at 4200/4600/5200 for seeds 42/1/2).
  After convergence `train_step` returns a detached zero-graph loss —
  param grads stay ``None`` so Adam and clipping are true no-ops and
  the weights are frozen at the plateau point, exactly the upstream
  procedure's stopping semantics under a fixed-step outer loop.

Param names match upstream (`W_enc`, `b_enc`, `W_decs`, `b_decs`) so
the archived T=5 anchors (`txcdr-base:agentic_txc_02__seed{1,2,42}.pt`)
load into this class directly (strict=False for the plugin's tracking
buffers). Deviations from vendored compute would be tagged
`# v2-adapter:`; there are none in the math. The only pipeline
difference is batching: upstream sampled (seq, off) pairs from a
preloaded (N, L, d) buffer; here `consumes='sequence'` (tsae precedent)
delivers (B, L, d) sequence batches through the canonical shuffle
buffer and `train_step` samples one offset per sequence — the same
uniform (seq, off) support.

Convention note: this class treats ``k_pos`` as the WINDOW selection
budget k_win (= 100·T for the paper family), matching the sibling
actmix-RLHF cell tables where ``k_pos = 100 * T`` is passed explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


# ── vendored: _info_nce (matryoshka_txcdr_contrastive.py, verbatim) ─────────


def _info_nce(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1, eps=1e-8)
    z_b = F.normalize(z_b, dim=-1, eps=1e-8)
    sim = z_a @ z_b.t()
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels))


class AgenticTXC02(TempBenchArch):
    """Trainable `agentic_txc_02` (position-nested Matryoshka TXCDR +
    multi-scale InfoNCE), vendored compute + TempBench contract."""

    _registry_name = "agentic_txc_02_v1t"
    arch_version: str = "upstream-94119bc08-trainable-1.0.0"
    consumes: str = "sequence"

    def __init__(
        self,
        *,
        d_in: int,
        d_sae: int,
        T: int = 5,
        k_pos: int = 500,
        alpha: float = 1.0,
        n_contr_scales: int | None = None,
        gamma: float = 0.5,
        plateau_threshold: float = 0.02,
        plateau_log_every: int = 200,
        plateau_window: int = 5,
        plateau_min_steps: int = 3000,
    ):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name=self._registry_name, d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T,
        )
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        self.k = int(k_pos)  # window budget k_win (= 100·T paper family)
        self.alpha = float(alpha)
        # v2-adapter: upstream t-sweep convention (train_primary_archs.py
        # ~L2025): n_contr_scales = min(3, T) so T<3 degrades gracefully.
        # Explicit values keep the vendored assert.
        if n_contr_scales is None:
            n_contr_scales = min(3, T)
        assert 1 <= n_contr_scales <= T, (
            f"n_contr_scales={n_contr_scales} must be in [1, T={T}]"
        )
        self.n_contr_scales = int(n_contr_scales)
        self.gamma = float(gamma)

        # vendored: PositionMatryoshkaTXCDR.__init__ (uniform latent split,
        # remainder into earliest prefixes)
        base = d_sae // T
        splits = [base + (1 if i < (d_sae - base * T) else 0) for i in range(T)]
        self.latent_splits = tuple(splits)
        self.prefix_sum = tuple(sum(splits[: i + 1]) for i in range(T))

        self.W_enc = nn.Parameter(
            torch.randn(T, d_in, d_sae) * (1.0 / d_in**0.5)
        )
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        self.W_decs = nn.ParameterList()
        self.b_decs = nn.ParameterList()
        for t_idx in range(T):
            prefix = self.prefix_sum[t_idx]
            t_size = t_idx + 1
            W = torch.randn(prefix, t_size, d_in) * (1.0 / prefix**0.5)
            self.W_decs.append(nn.Parameter(W))
            self.b_decs.append(nn.Parameter(torch.zeros(t_size, d_in)))

        # plateau mirror state (buffers → checkpointed, resume-safe)
        self.plateau_threshold = float(plateau_threshold)
        self.plateau_log_every = int(plateau_log_every)
        self.plateau_window = int(plateau_window)
        self.plateau_min_steps = int(plateau_min_steps)
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("converged_step", torch.tensor(-1, dtype=torch.long))
        self._loss_log: list[float] = []

    # ── vendored: PositionMatryoshkaTXCDR methods (verbatim compute) ───────

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        for W in self.W_decs:
            norms = W.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
            W.data = W.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_in) -> z: (B, d_sae) with k non-zeros (TopK→ReLU)."""
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        if self.k is not None:
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(1, idx, F.relu(vals))
        else:
            z = F.relu(pre)
        return z

    def _window_center(self, x: torch.Tensor, t_size: int) -> torch.Tensor:
        T = x.shape[1]
        start = (T - t_size) // 2
        return x[:, start:start + t_size, :]

    def decode_scale(self, z: torch.Tensor, scale_idx: int) -> torch.Tensor:
        prefix = self.prefix_sum[scale_idx]
        W = self.W_decs[scale_idx]
        b = self.b_decs[scale_idx]
        z_prefix = z[:, :prefix]
        return torch.einsum("bs,std->btd", z_prefix, W) + b

    @property
    def decoder_dirs_averaged(self) -> torch.Tensor:
        dirs = torch.zeros(self.d_in, self._d_sae, device=self.W_decs[0].device)
        counts = torch.zeros(self._d_sae, device=self.W_decs[0].device)
        for t_idx in range(self._T):
            prefix = self.prefix_sum[t_idx]
            W = self.W_decs[t_idx]
            avg = W.mean(dim=1).T
            dirs[:, :prefix] += avg
            counts[:prefix] += 1
        counts = counts.clamp(min=1e-8)
        return dirs / counts

    # ── vendored: MatryoshkaTXCDRContrastive._matryoshka_loss (verbatim) ──

    def _matryoshka_loss(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        losses = []
        for t_idx in range(self._T):
            t_size = t_idx + 1
            x_center = self._window_center(x, t_size)
            x_hat = self.decode_scale(z, t_idx)
            loss_scale = (x_hat - x_center).pow(2).sum(dim=-1).mean()
            losses.append(loss_scale)
        return torch.stack(losses).mean()

    # ── vendored: MatryoshkaTXCDRContrastiveMultiscale pair loss (verbatim
    #    compute; dispatch unrolled — train_step always builds the pair) ───

    def _sample_pairs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, L, d) sequences → shift-1 adjacent T-window pair, one
        uniform offset per sequence (make_pair_window_gen_gpu support)."""
        B, L, d = x.shape
        T = self._T
        assert L >= T + 1, f"need L>={T + 1} for adjacent T-windows; got L={L}"
        off = torch.randint(0, L - T, (B,), device=x.device)
        rng = torch.arange(T, device=x.device)
        pos_prev = off.unsqueeze(1) + rng.unsqueeze(0)
        pos_cur = (off + 1).unsqueeze(1) + rng.unsqueeze(0)
        gather_idx_prev = pos_prev.unsqueeze(-1).expand(-1, -1, d)
        gather_idx_cur = pos_cur.unsqueeze(-1).expand(-1, -1, d)
        x_prev = torch.gather(x, 1, gather_idx_prev).float()
        x_cur = torch.gather(x, 1, gather_idx_cur).float()
        return x_prev, x_cur

    def _pair_loss(self, x_prev: torch.Tensor, x_cur: torch.Tensor):
        z_prev = self.encode(x_prev)
        z_cur = self.encode(x_cur)

        l_matr = (self._matryoshka_loss(x_prev, z_prev)
                  + self._matryoshka_loss(x_cur, z_cur))

        l_contr = torch.zeros((), device=x_cur.device, dtype=x_cur.dtype)
        for s in range(self.n_contr_scales):
            prefix_s = self.prefix_sum[s]
            z_h_prev = z_prev[:, :prefix_s]
            z_h_cur = z_cur[:, :prefix_s]
            l_contr = l_contr + (self.gamma ** s) * _info_nce(z_h_cur, z_h_prev)

        total = l_matr + self.alpha * l_contr
        return total, l_matr, l_contr, z_cur

    # ── TempBench contract ─────────────────────────────────────────────────

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Full-scale reconstruction (scale T-1) → (B, T, d_in)."""
        return self.decode_scale(z, self._T - 1)

    def _plateau(self) -> float | None:
        w = self.plateau_window
        if len(self._loss_log) < 2 * w:
            return None
        recent = sum(self._loss_log[-w:]) / w
        prior = sum(self._loss_log[-2 * w:-w]) / w
        if prior == 0:
            return None
        return (prior - recent) / abs(prior)

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: (B, L, d_in) sequence batch → shift-1 adjacent T-window
        pairs (one offset per sequence, uniform) → paper loss.

        After the plateau rule fires, returns a detached zero-graph loss
        (no model params in the graph) — the trainer's backward/step
        become no-ops and weights freeze at the plateau point."""
        step = int(self.global_step.item())
        if int(self.converged_step.item()) >= 0:
            self.global_step += 1
            return {
                "loss": torch.zeros((), device=x.device, requires_grad=True),
                "l0": torch.zeros(()),
                "converged": torch.ones(()),
            }

        x_prev, x_cur = self._sample_pairs(x)
        total, l_matr, l_contr, z_cur = self._pair_loss(x_prev, x_cur)

        metrics = {
            "loss": total,
            "loss_matryoshka": l_matr.detach(),
            "loss_contrastive": l_contr.detach(),
            "l0": (z_cur > 0).float().sum(dim=-1).mean().detach(),
            "converged": torch.zeros(()),
        }

        # plateau mirror (upstream: log every 200, window 5, thr 0.02,
        # min 3000; stop check only at log steps)
        if step % self.plateau_log_every == 0:
            self._loss_log.append(float(total.detach().item()))
            p = self._plateau()
            if (p is not None and p < self.plateau_threshold
                    and step >= self.plateau_min_steps):
                self.converged_step.fill_(step)
        self.global_step += 1
        return metrics

    def post_step(self) -> None:
        self._normalize_decoder()

    def decoder_directions(self) -> torch.Tensor:
        return self.decoder_dirs_averaged.T  # (d_sae, d_in)
