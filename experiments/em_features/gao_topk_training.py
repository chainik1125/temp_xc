"""Gao et al. 2024 TopK SAE training recipe, adapted for TemporalCrosscoder
and MultiLayerCrosscoder.

Components (all together; each alone is a partial fix for dead-feature
explosion):

  1. Tied init + geometric-median b_dec: encoder = decoder transpose (already
     the sae_day default) and b_dec = geom-median of a fresh data sample.
  2. Adam(β1=0, β2=0.9999) — empirically large effect on survival
     (Anthropic Feb 2024 update).
  3. LR scaled by 1/sqrt(d_sae / 16384) from a 2e-4 base.
  4. Every step: remove the component of W_dec.grad parallel to W_dec rows,
     so the unit-norm renorm step doesn't fight the optimizer.
  5. Every `normalize_every` steps: unit-norm W_dec rows (joint over slots).
  6. AuxK loss: every step, reconstruct (x − x_hat_live) using the top
     `k_aux` DEAD features' decoder rows. Multiplied by α (default 1/32)
     and normalized by the live MSE so the coefficient is scale-invariant.
  7. Dead-feature bookkeeping: track `num_tokens_since_fired` per feature;
     a feature is "dead" once this exceeds a threshold (default 10M tokens).

Intentionally skipping K-annealing and resampling — AuxK subsumes the need
for both in the Gao recipe.

The public API is a single function ``train_gao_topk(...)`` that takes the
crosscoder, an activation sampler, and returns per-logged-step stats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch


@torch.no_grad()
def geometric_median(X: torch.Tensor, n_iter: int = 30, eps: float = 1e-8) -> torch.Tensor:
    """Weiszfeld's algorithm. X shape (N, d); returns (d,)."""
    y = X.mean(dim=0)
    for _ in range(n_iter):
        diff = X - y
        w = 1.0 / diff.norm(dim=-1).clamp(min=eps)
        y_new = (X * w.unsqueeze(-1)).sum(dim=0) / w.sum()
        if (y_new - y).norm() < eps:
            break
        y = y_new
    return y


@torch.no_grad()
def init_b_dec_geometric_median(crosscoder, sample_fn, n: int = 8192) -> None:
    """Initialize b_dec to the geometric median of per-position activations
    (TXC) or per-layer activations (MLC)."""
    x = sample_fn(n)  # (n, T_or_L, d)
    # Assume crosscoder has attribute .b_dec of shape (T_or_L, d_in)
    T_or_L = x.shape[1]
    gm = torch.stack([geometric_median(x[:, t, :]) for t in range(T_or_L)], dim=0)
    crosscoder.b_dec.data.copy_(gm.to(crosscoder.b_dec.dtype).to(crosscoder.b_dec.device))


def scaled_lr(base: float, d_sae: int, d_sae_ref: int = 16384) -> float:
    """LR = base / sqrt(d_sae / d_sae_ref). Gao 2024."""
    from math import sqrt
    return base / sqrt(max(1.0, d_sae / d_sae_ref))


@torch.no_grad()
def remove_decoder_parallel_grad(crosscoder) -> None:
    """Project out the component of W_dec.grad parallel to each decoder row.
    Assumes rows are approximately unit-norm (which they are between
    normalize_decoder() calls)."""
    W = crosscoder.W_dec  # (T, d_sae, d_in)
    if W.grad is None:
        return
    dot = (W.grad * W.data).sum(dim=-1, keepdim=True)  # (T, d_sae, 1)
    # Row norms squared (in case they drifted).
    norm_sq = W.data.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    W.grad.sub_(dot * W.data / norm_sq)


def _encode_pre_topk(crosscoder, x: torch.Tensor) -> torch.Tensor:
    """Pre-activation (before TopK, after ReLU). For TemporalCrosscoder this
    reproduces the internal computation in .encode(); we need it exposed
    here so AuxK can operate on the dead-feature subset."""
    # TemporalCrosscoder / MultiLayerCrosscoder both use:
    #   pre = einsum("btd,tdm->bm", x, W_enc) + b_enc  (or bld for MLC)
    pre = torch.einsum("btd,tdm->bm", x, crosscoder.W_enc) + crosscoder.b_enc
    if getattr(crosscoder, "use_relu", True):
        pre = torch.relu(pre)
    return pre


def _decode(crosscoder, z: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bm,tmd->btd", z, crosscoder.W_dec) + crosscoder.b_dec


@dataclass
class GaoStats:
    step: int
    loss_main: float
    loss_auxk: float
    loss_total: float
    n_dead: int
    n_features: int
    n_active_this_batch: int
    max_fire_count: int
    elapsed_min: float
    extra: dict = field(default_factory=dict)


def train_gao_topk(
    crosscoder,
    sample_fn: Callable[[int], torch.Tensor],
    *,
    batch_size: int,
    n_steps: int,
    auxk_alpha: float = 1.0 / 32.0,
    k_aux: int | None = None,
    dead_token_threshold: int = 10_000_000,
    lr_base: float = 2e-4,
    normalize_every: int = 100,
    log_every: int = 500,
    device: str = "cuda",
    init_b_dec: bool = True,
    init_n_samples: int = 8192,
    auxk_norm: str = "ema",  # "ema" | "per_step" | "none"
    auxk_ema_decay: float = 0.99,
    adam_betas: tuple = (0.0, 0.9999),
    bricken_resample_every: int = 0,  # 0 = off; else every N steps do Bricken resample
    bricken_resample_min_fires: int = 1,
    ) -> list[GaoStats]:
    """Run the Gao 2024 TopK training loop on an already-built crosscoder.

    sample_fn(n) should return a tensor of shape (n, T_or_L, d_in) on the
    crosscoder's device (or the loop will move it).
    """
    if init_b_dec:
        init_b_dec_geometric_median(crosscoder, sample_fn, n=init_n_samples)

    d_sae = crosscoder.d_sae
    d_in = crosscoder.d_in
    if k_aux is None:
        # Gao default: k_aux = d_model/2 (= d_in/2 here).
        k_aux = max(1, d_in // 2)

    lr = scaled_lr(lr_base, d_sae)
    optim = torch.optim.Adam(crosscoder.parameters(), lr=lr, betas=tuple(adam_betas))

    tokens_since_fired = torch.zeros(d_sae, dtype=torch.long, device=device)
    history: list[GaoStats] = []
    train_t0 = time.time()
    ema_main = None  # running average of loss_main for scale-invariant AuxK norm

    # Lazy import to avoid circular dep if user imports gao_topk_training stand-alone
    if bricken_resample_every > 0:
        from experiments.em_features.dead_feature_resample import DeadFeatureResampler
        resampler = DeadFeatureResampler(
            crosscoder,
            resample_every=bricken_resample_every,
            min_fires=bricken_resample_min_fires,
            n_check=2048,
        )
    else:
        resampler = None

    for step in range(n_steps):
        x = sample_fn(batch_size).to(device).float()

        # -- forward (live) --
        pre = _encode_pre_topk(crosscoder, x)                 # (B, d_sae)
        k = crosscoder.k_total
        topk_vals, topk_idx = pre.topk(k, dim=-1)
        z_live = torch.zeros_like(pre)
        z_live.scatter_(-1, topk_idx, topk_vals)
        x_hat_live = _decode(crosscoder, z_live)              # (B, T, d_in)
        loss_main = (x - x_hat_live).pow(2).sum(dim=-1).mean()

        # -- AuxK: reconstruct the live residual using top-k_aux DEAD features --
        dead_mask = tokens_since_fired >= dead_token_threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead > 0:
            # Use pre-TopK activations restricted to dead features.
            pre_dead = pre.clone()
            pre_dead[:, ~dead_mask] = -float("inf")
            k_use = min(k_aux, n_dead)
            top_dead_vals, top_dead_idx = pre_dead.topk(k_use, dim=-1)
            # Zero out any -inf that survived (if a row had < k_use dead features available).
            top_dead_vals = torch.where(
                torch.isinf(top_dead_vals) | torch.isnan(top_dead_vals),
                torch.zeros_like(top_dead_vals),
                top_dead_vals,
            )
            z_aux = torch.zeros_like(pre)
            z_aux.scatter_(-1, top_dead_idx, top_dead_vals)
            # Decode without b_dec (we're reconstructing a residual, not activations).
            x_aux = torch.einsum("bm,tmd->btd", z_aux, crosscoder.W_dec)
            residual = x - x_hat_live.detach()
            loss_auxk = (residual - x_aux).pow(2).sum(dim=-1).mean()

            # Scale-invariant normalization. "ema" is Gao 2024's actual choice
            # (running mean of main loss), "per_step" matches our buggy v1,
            # "none" drops normalization (use raw auxk loss).
            if auxk_norm == "ema":
                if ema_main is None:
                    ema_main = float(loss_main.detach())
                else:
                    ema_main = auxk_ema_decay * ema_main + (1 - auxk_ema_decay) * float(loss_main.detach())
                denom = ema_main + 1e-8
                loss_auxk_norm = loss_auxk / denom
                loss_total = loss_main + auxk_alpha * loss_auxk_norm
            elif auxk_norm == "per_step":
                loss_auxk_norm = loss_auxk / (loss_main.detach() + 1e-8)
                loss_total = loss_main + auxk_alpha * loss_auxk_norm
            elif auxk_norm == "none":
                loss_total = loss_main + auxk_alpha * loss_auxk
            else:
                raise ValueError(f"unknown auxk_norm: {auxk_norm!r}")
        else:
            loss_auxk = torch.tensor(0.0, device=device)
            loss_total = loss_main
            # Still update EMA for consistency
            if auxk_norm == "ema":
                if ema_main is None:
                    ema_main = float(loss_main.detach())
                else:
                    ema_main = auxk_ema_decay * ema_main + (1 - auxk_ema_decay) * float(loss_main.detach())

        optim.zero_grad(set_to_none=True)
        loss_total.backward()
        remove_decoder_parallel_grad(crosscoder)
        optim.step()

        if resampler is not None:
            resampler.maybe_resample(step + 1, sample_fn)

        # -- bookkeeping: tokens_since_fired per feature --
        with torch.no_grad():
            fired_this_batch = (z_live != 0).any(dim=0)      # (d_sae,)
            tokens_since_fired += batch_size
            tokens_since_fired[fired_this_batch] = 0

        if (step + 1) % normalize_every == 0 and hasattr(crosscoder, "normalize_decoder"):
            crosscoder.normalize_decoder()

        if (step + 1) % log_every == 0:
            with torch.no_grad():
                n_active = int(fired_this_batch.sum().item())
                # quick dead-fraction snapshot via a probe batch
                probe = sample_fn(2048).to(device).float()
                z_probe = crosscoder.encode(probe)
                fire_count = (z_probe != 0).sum(dim=0)
            cumulative_resamples = (
                sum(h.n_resampled for h in resampler.history) if resampler else 0
            )
            history.append(GaoStats(
                step=step + 1,
                loss_main=float(loss_main.detach()),
                loss_auxk=float(loss_auxk.detach()),
                loss_total=float(loss_total.detach()),
                n_dead=int((fire_count == 0).sum().item()),
                n_features=d_sae,
                n_active_this_batch=n_active,
                max_fire_count=int(fire_count.max().item()),
                elapsed_min=(time.time() - train_t0) / 60,
                extra={
                    "tokens_since_fired_dead_count": int((tokens_since_fired >= dead_token_threshold).sum().item()),
                    "n_resampled_so_far": cumulative_resamples,
                    "ema_main": ema_main or 0.0,
                },
            ))
            print(f"[gao] step {step+1:>6}/{n_steps}  "
                  f"main={history[-1].loss_main:.1f}  auxk={history[-1].loss_auxk:.2f}  "
                  f"dead={history[-1].n_dead}/{d_sae} "
                  f"({100*history[-1].n_dead/d_sae:.1f}%)  "
                  f"active_in_batch={n_active}", flush=True)

    return history
