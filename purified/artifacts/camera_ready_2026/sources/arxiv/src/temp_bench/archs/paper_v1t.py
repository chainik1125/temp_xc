"""Paper-faithful TRAINABLE TXC-base — the {ReLU+TopK} matrix arm.

Commissioned by 4ce0369de / 606e4587d (paper-faithful sprint): make the
paper's exact §5.1 TXC-base composition trainable through the canonical
runner. The paper's shipped cells trained
``origin/han-phase7-unification@94119bc08:src/architectures/
txc_bare_antidead.py:TXCBareAntidead``; the eval-only adapter
(``paper_v1.PaperTXCBaseV1``) vendors its encode/decode. THIS file
vendors the WHOLE upstream class — training stack included — verbatim:

    z = ReLU(TopK_{k_win}(Σ_t x_t · W_enc[t] + b_enc)),  k_win = k_pos·T

- per-SAMPLE TopK over the summed (post-squash) window pre-activation
  (NOT BatchTopK — upstream deferred it on purpose),
- ReLU AFTER selection (selected negatives become exact zeros — the
  paper-era mixing fingerprint; realized l0 ≤ k_win),
- anti-dead stack: 10M-token dead tracker, AuxK(aux_k=512,
  α=1/32) re-reconstructing the residual through the decoder WITHOUT
  bias, unit-norm decoder atoms over (T, d_in), decoder-parallel
  gradient removal, geometric-median b_dec init on the first batch.

``_V1TTXCBareAntidead`` below is line-for-line upstream compute (any
deviation would be tagged ``# v1t-adapter:``; there are none in the
math). The v2 wrapper adds ONLY what the upstream trainer did outside
the class: the dict train_step contract, the first-batch b_dec-init
call site, the grad-projection hook registration
(``register_post_accumulate_grad_hook`` — fires post-accumulate during
backward, the v2 idiom for upstream's post-backward projection), the
``post_step`` renorm call site, and OPT-IN telemetry sampling computed
wrapper-side under ``no_grad`` (vendored math untouched).

State-dict keys are ``inner.*`` — a strict SUPERSET of the eval-only
adapter's (adds last_auxk_loss / last_dead_count / b_dec_initialized
buffers); the archived T5 anchors stay on ``paper_txc_base_v1`` and are
never retrained (alias rule).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from temp_bench.archs import telemetry
from temp_bench.interfaces.architecture import ArchConfig, TempBenchArch


# ── Vendored upstream (94119bc08, verbatim incl. training stack) ──────────


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


class _V1TTXCBareAntidead(nn.Module):
    """94119bc08 TXCBareAntidead — full class, training stack included."""

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

        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, T, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc.data[t])
        nn.init.kaiming_uniform_(self.W_dec.data.view(d_sae, T * d_in))
        self._normalize_decoder()
        with torch.no_grad():
            for t in range(T):
                self.W_enc.data[t] = self.W_dec.data[:, t, :].T

        self.register_buffer(
            "num_tokens_since_fired", torch.zeros(d_sae, dtype=torch.long))
        self.register_buffer("last_auxk_loss", torch.tensor(-1.0))
        self.register_buffer("last_dead_count",
                             torch.tensor(0, dtype=torch.long))
        self.register_buffer("b_dec_initialized", torch.tensor(False))

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=(1, 2), keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder(self):
        if self.W_dec.grad is None:
            return
        W_flat = self.W_dec.data.view(self.d_sae, -1)
        g_flat = self.W_dec.grad.view(self.d_sae, -1)
        normed = W_flat / (W_flat.norm(dim=1, keepdim=True) + 1e-6)
        parallel = (g_flat * normed).sum(dim=1, keepdim=True)
        g_flat.sub_(parallel * normed)

    @torch.no_grad()
    def init_b_dec_geometric_median(self, x_sample: torch.Tensor):
        assert not bool(self.b_dec_initialized), "b_dec already initialized"
        for t in range(self.T):
            med = _geometric_median(x_sample[:, t, :].float())
            self.b_dec.data[t] = med.to(self.b_dec.dtype)
        self.b_dec_initialized.fill_(True)

    def _pre_activation(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self._pre_activation(x)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        pre = self._pre_activation(x)
        vals, idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))

        x_hat = self.decode(z)
        l_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        active_mask = (z > 0).any(dim=0)
        n_tokens = x.shape[0] * x.shape[1]
        self.num_tokens_since_fired += n_tokens
        self.num_tokens_since_fired[active_mask] = 0
        dead_mask = self.num_tokens_since_fired >= self.dead_threshold_tokens
        n_dead = int(dead_mask.sum().item())
        self.last_dead_count.fill_(n_dead)

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


# ── v2 wrapper (trainable plugin) ─────────────────────────────────────────


class PaperTXCBaseV1T(TempBenchArch):
    """Paper §5.1 TXC-base composition, TRAINABLE (paper-faithful arm)."""

    arch_version = "upstream-94119bc08-trainable-1.0.0"
    consumes = "window"

    def __init__(self, d_in: int, d_sae: int = 18432, k_pos: int = 20,
                 T: int = 5, aux_k: int = 512,
                 dead_threshold_tokens: int = 10_000_000,
                 auxk_alpha: float = 1.0 / 32.0):
        nn.Module.__init__(self)
        self.config = ArchConfig(
            name="paper_txc_base_v1t", d_in=d_in, d_sae=d_sae,
            k_pos=k_pos, T=T)
        self.inner = _V1TTXCBareAntidead(
            d_in=d_in, d_sae=d_sae, T=T, k=k_pos * T, aux_k=aux_k,
            dead_threshold_tokens=dead_threshold_tokens,
            auxk_alpha=auxk_alpha)
        # v2 idiom for upstream's post-backward projection (fires on
        # grad accumulation; single backward per step ⇒ identical).
        self.inner.W_dec.register_post_accumulate_grad_hook(
            self._project_dec_grad)
        self._step = 0

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self.inner.T:
            raise ValueError(
                f"paper_txc_base_v1t.encode expects (B, T={self.inner.T}, "
                f"d_in); got T={x.shape[1]}.")
        return self.inner.encode(x)                      # (B, d_sae)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.inner.decode(z)

    def train_step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != self.inner.T:
            raise ValueError(
                f"paper_txc_base_v1t.train_step expects (B, T="
                f"{self.inner.T}, d_in); got {tuple(x.shape)}.")
        # Upstream trainer convention: geometric-median b_dec init on
        # the first training batch.
        if not bool(self.inner.b_dec_initialized):
            self.inner.init_b_dec_geometric_median(x)

        total, x_hat, z = self.inner(x)

        with torch.no_grad():
            l0 = (z != 0).float().sum(dim=-1).mean()
            mse = (x - x_hat).pow(2).sum(dim=-1).mean()
            if telemetry.due(self._step):
                pre = self.inner._pre_activation(x)
                sel_vals = pre.topk(self.inner.k, dim=-1).values
                nz = z[z != 0]
                telemetry.maybe_log(
                    self, step=self._step,
                    n_dead=int(self.inner.last_dead_count.item()),
                    batch_l0=float(nz.numel()) / x.shape[0],
                    boundary_min_pre=float(sel_vals.min().item()))
        self._step += 1

        return {
            "loss": total,
            "mse": mse,
            "l0": l0,
            "auxk": self.inner.last_auxk_loss.detach().clone(),
            "dead": self.inner.last_dead_count.detach().clone().float(),
        }

    def post_step(self) -> None:
        with torch.no_grad():
            self.inner._normalize_decoder()
