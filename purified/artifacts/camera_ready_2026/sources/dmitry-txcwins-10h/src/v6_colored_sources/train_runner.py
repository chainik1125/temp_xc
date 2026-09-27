"""Training driver for colored-source experiments.

Re-implements the training loop from `temporal_crosscoders/train.py` with
explicit `(d_in, d_sae)` arguments so we can run at the proposal's d=N=128
square geometry without monkey-patching `temporal_crosscoders.config`.

The local-baseline architecture is the regular `TopKSAE` operating on
individual tokens (not the stacked SAE) — this matches the proposal's
"local one-token learner" definition exactly: training data are iid samples
from P(x_t), with no awareness of position or window.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

# Ensure temporal_crosscoders/ is importable so its bare `from models import ...`
# style works.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TXC_DIR = _REPO_ROOT / "temporal_crosscoders"
if str(_TXC_DIR) not in sys.path:
    sys.path.insert(0, str(_TXC_DIR))

from models import TemporalCrosscoder, TopKSAE  # noqa: E402

# Han's vendored architectures (read-only port of the H8 lock-in stack:
# bare TXC + anti-dead + matryoshka + multi-distance InfoNCE).
from temporal_crosscoders.han_arch import (  # noqa: E402
    TXCBareMultiDistanceContrastiveAntidead,
    make_multidistance_pair_gen_gpu,
)
from temporal_crosscoders.han_arch.txc_bare_antidead import TXCBareAntidead  # noqa: E402
from temporal_crosscoders.han_tsae import TemporalSAE  # noqa: E402
# TFA's variant of TemporalSAE (cleaner, with optional sinusoidal pos
# encoding and the Han train_tfa.py training recipe).
from src.bench.architectures._tfa_module import (  # noqa: E402
    TemporalSAE as TFA_TemporalSAE,
)

from .data_adapter import ColoredSourceCache  # noqa: E402
from .metrics import (  # noqa: E402
    chance_adjusted_recovery,
    squared_axis_recovery,
)
from .theory import spectral_oracle  # noqa: E402


def _recovery_auc_torch(W_dec: torch.Tensor, F: torch.Tensor) -> float:
    """AUC of the |cos sim| threshold-sweep survival curve.

    Mirrors src/v5_hmm_sae_baseline/metrics.feature_recovery_score's `auc` field
    but computed in torch so the project runs without a working numpy/torch
    interop layer.

    Args:
        W_dec: (d, n_latents) decoder direction matrix (columns are atoms).
        F: (k, d) ground-truth direction matrix (rows are unit-norm atoms).

    Returns:
        Scalar AUC in [0, 1].
    """
    W_norm = W_dec / W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
    F_norm = F / F.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos = F_norm @ W_norm                            # (k, n_latents)
    max_per_true = cos.abs().max(dim=1).values        # (k,)
    k = F.shape[0]
    thresholds = torch.linspace(0.0, 1.0, k, device=max_per_true.device)
    curve = (max_per_true.unsqueeze(0) >= thresholds.unsqueeze(1)).float().mean(dim=1)
    auc = torch.trapz(curve, thresholds) / (thresholds[-1] - thresholds[0])
    return float(auc.item())


@dataclass
class TrainConfig:
    n_steps: int = 30_000
    batch_size: int = 64
    lr: float = 1e-3
    grad_clip: float = 1.0
    log_interval: int = 1000
    adam_betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class CellResult:
    arch: str            # "sae" | "txc"
    W: int
    k: int
    final_loss: float
    final_l0: float
    recovery_squared: float
    s_adj: float
    recovery_auc: float
    history: list[dict]


def _make_iterator(cache: ColoredSourceCache, batch_size: int, T: int):
    """Pull (B, T, d) windows from our cache. Mirrors CachedWindowIterator
    without the refresh-interval branch (our cache is deterministic)."""
    while True:
        yield cache.sample_windows(batch_size, T)


def _make_token_iterator(cache: ColoredSourceCache, batch_size: int):
    """Pull (B, d) iid token batches by sampling random (chain, position) pairs.

    This is the "local one-token learner" data feed: each batch element is an
    independent sample from P(x_t) with no positional or window structure.
    """
    while True:
        # Sample one position per batch element via a window of length 1.
        x = cache.sample_windows(batch_size, 1)  # (B, 1, d)
        yield x.squeeze(1)  # (B, d)


def _train_one(
    model: nn.Module,
    cache: ColoredSourceCache,
    F: torch.Tensor,
    arch: str,
    W: int,
    k: int,
    H: int,
    train_cfg: TrainConfig,
    device: torch.device,
    *,
    token_iter: bool = False,
) -> CellResult:
    """Train `model` on the cache. If token_iter, sample (B, d) tokens (for
    regular TopKSAE). Otherwise sample (B, W, d) windows (for TXC)."""
    import time
    if token_iter:
        iterator = _make_token_iterator(cache, train_cfg.batch_size)
    else:
        iterator = _make_iterator(cache, train_cfg.batch_size, W)
    opt = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, betas=train_cfg.adam_betas
    )
    history: list[dict] = []
    F_dev = F.to(device=device, dtype=torch.float32)
    N = F.shape[0]
    t_start = time.time()

    for step in range(train_cfg.n_steps):
        x = next(iterator).to(device)
        loss, _, u = model(x)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        if hasattr(model, "_normalize_decoder"):
            model._normalize_decoder()

        last_step = step == train_cfg.n_steps - 1
        if step % train_cfg.log_interval == 0 or last_step:
            with torch.no_grad():
                x_eval = next(iterator).to(device)
                eval_loss, _, u_eval = model(x_eval)
                l0 = (u_eval > 0).float().sum(dim=-1).mean().item()
                F_hat = model.decoder_directions.detach()  # (d, H)
                # Match metric convention: recovered directions as (H, d) rows.
                F_hat_rows = F_hat.T.contiguous()
                rec_sq = squared_axis_recovery(F_dev, F_hat_rows)
                s_adj = chance_adjusted_recovery(rec_sq, N=N, H=H)
                rec_auc = _recovery_auc_torch(F_hat, F_dev)
            history.append({
                "step": step,
                "loss": float(eval_loss.item()),
                "l0": l0,
                "recovery_squared": rec_sq,
                "s_adj": s_adj,
                "recovery_auc": rec_auc,
            })
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / max(elapsed, 1e-6)
            eta_sec = (train_cfg.n_steps - step - 1) / max(steps_per_sec, 1e-6)
            print(
                f"  [{arch} W={W}] step={step:>6d} loss={eval_loss.item():.4f} "
                f"L0={l0:.2f} rec_sq={rec_sq:.3f} S_adj={s_adj:.3f} "
                f"AUC={rec_auc:.3f} | {steps_per_sec:.1f} it/s ETA {eta_sec/60:.1f}m",
                flush=True,
            )

    last = history[-1]
    return CellResult(
        arch=arch,
        W=W,
        k=k,
        final_loss=last["loss"],
        final_l0=last["l0"],
        recovery_squared=last["recovery_squared"],
        s_adj=last["s_adj"],
        recovery_auc=last["recovery_auc"],
        history=history,
    )


def train_sae(
    *,
    cache: ColoredSourceCache,
    F: torch.Tensor,
    k: int,
    H: int,
    d: int,
    device: torch.device,
    train_cfg: TrainConfig,
) -> CellResult:
    """Train a regular TopKSAE on iid tokens. The "local one-token learner"
    baseline — no awareness of position or window. The reported W is 1 by
    convention since each batch element is a single token."""
    sae = TopKSAE(d_in=d, d_sae=H, k=k).to(device)
    return _train_one(
        sae, cache, F, "sae", W=1, k=k, H=H,
        train_cfg=train_cfg, device=device, token_iter=True,
    )


def train_txc(
    *,
    cache: ColoredSourceCache,
    F: torch.Tensor,
    W: int,
    k: int,
    H: int,
    d: int,
    device: torch.device,
    train_cfg: TrainConfig,
) -> CellResult:
    """Train a TemporalCrosscoder at window length W >= 2."""
    if W < 2:
        raise ValueError(f"TXC requires W >= 2, got W={W}")
    txc = TemporalCrosscoder(d_in=d, d_sae=H, T=W, k=k).to(device)
    return _train_one(
        txc, cache, F, "txc", W=W, k=k, H=H,
        train_cfg=train_cfg, device=device, token_iter=False,
    )


def train_txc_global(
    *,
    cache: ColoredSourceCache,
    F: torch.Tensor,
    W: int,
    k_window: int,
    H: int,
    d: int,
    device: torch.device,
    train_cfg: TrainConfig,
    geometric_median_init: bool = True,
) -> CellResult:
    """Train ``TXCBareAntidead`` with a *window-level* TopK budget.

    Distinguished from ``train_txc`` (which uses ``TemporalCrosscoder`` and
    scales ``k`` by ``T``) — this driver passes ``k`` straight through to
    Han's class so ``k_window=1`` really means one active latent for the
    entire window. That's the architecture the polynomial-clock proposal
    prescribes: a global sparsity bottleneck that forces the model to
    compress the whole window into a single temporal-template atom.

    The standard local-recovery / s_adj / chance-adjusted bookkeeping is
    reused from ``_train_one`` but evaluated against the *averaged* decoder
    direction. For temporal-atom recovery on the polynomial-clock task
    you want to evaluate ``model.W_dec`` directly against the bank of
    ``G_β`` templates — see ``run_polynomial_clock.py``.
    """
    import time
    if W < 1:
        raise ValueError(f"W must be >= 1; got {W}")
    if k_window < 1:
        raise ValueError(f"k_window must be >= 1; got {k_window}")

    model = TXCBareAntidead(d_in=d, d_sae=H, T=W, k=k_window).to(device)

    if geometric_median_init:
        with torch.no_grad():
            init_x = cache.sample_windows(min(train_cfg.batch_size * 4, 256), W)
            model.init_b_dec_geometric_median(init_x.float())

    iterator = _make_iterator(cache, train_cfg.batch_size, W)
    opt = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, betas=train_cfg.adam_betas
    )

    history: list[dict] = []
    F_dev = F.to(device=device, dtype=torch.float32)
    N = F.shape[0]
    arch = "txc_global"
    t_start = time.time()

    for step in range(train_cfg.n_steps):
        x = next(iterator).to(device)
        loss, _, z = model(x)
        opt.zero_grad()
        loss.backward()
        if hasattr(model, "remove_gradient_parallel_to_decoder"):
            model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        if hasattr(model, "_normalize_decoder"):
            model._normalize_decoder()

        last_step = step == train_cfg.n_steps - 1
        if step % train_cfg.log_interval == 0 or last_step:
            with torch.no_grad():
                x_eval = next(iterator).to(device)
                eval_loss, _, z_eval = model(x_eval)
                l0 = (z_eval > 0).float().sum(dim=-1).mean().item()
                F_hat = model.decoder_dirs_averaged.detach()
                F_hat_rows = F_hat.T.contiguous()
                rec_sq = squared_axis_recovery(F_dev, F_hat_rows)
                s_adj = chance_adjusted_recovery(rec_sq, N=N, H=H)
            history.append({
                "step": step,
                "loss": float(eval_loss.item()),
                "l0": l0,
                "recovery_squared": rec_sq,
                "s_adj": s_adj,
                "recovery_auc": 0.0,
            })
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / max(elapsed, 1e-6)
            eta_sec = (train_cfg.n_steps - step - 1) / max(steps_per_sec, 1e-6)
            print(
                f"  [{arch} W={W} k_window={k_window}] step={step:>6d} "
                f"loss={eval_loss.item():.4f} L0={l0:.2f} "
                f"rec_sq_local_avg={rec_sq:.3f} S_adj={s_adj:.3f} "
                f"| {steps_per_sec:.1f} it/s ETA {eta_sec/60:.1f}m",
                flush=True,
            )

    last = history[-1]
    return CellResult(
        arch=arch,
        W=W,
        k=k_window,
        final_loss=last["loss"],
        final_l0=last["l0"],
        recovery_squared=last["recovery_squared"],
        s_adj=last["s_adj"],
        recovery_auc=last["recovery_auc"],
        history=history,
    )


def train_tsae(
    *,
    cache: ColoredSourceCache,
    W: int,
    k: int,
    H: int,
    d: int,
    n_heads: int = 8,
    n_attn_layers: int = 1,
    bottleneck_factor: int = 16,
    contrastive_weight: float = 0.1,
    device: torch.device,
    train_cfg: TrainConfig,
    tied_weights: bool = True,
) -> TemporalSAE:
    """Train Bhalla 2025-style TemporalSAE on (B, W, d) windows.

    Recipe:

    - ``sae_diff_type='topk'`` with ``kval_topk = k`` per token (not per
      window). The Bhalla paper uses ``k=20`` by default.
    - Loss = reconstruction MSE + ``contrastive_weight`` * InfoNCE between
      consecutive temporal codes ``(z_t, z_{t+1})``, evaluated per window
      and averaged. Default contrastive weight ``0.1``.
    - Reuses Han's vendored ``TemporalSAE`` (attention-based predicted
      codes + novel codes via TopK on the residual).

    Returns the trained model. Use ``model(x_btd)`` at inference to get
    ``(x_recons, results_dict)`` with ``results_dict['novel_codes']`` and
    ``results_dict['pred_codes']`` of shape ``(B, W, H)`` each — sum
    them for the per-position activation used in probing.
    """
    import time
    if W < 2:
        raise ValueError(f"TSAE attention requires W >= 2; got W={W}")

    # ManualAttention requires (width / bottleneck_factor) % n_heads == 0.
    # Adjust bottleneck_factor so this is always satisfied.
    while H // bottleneck_factor < n_heads or (H // bottleneck_factor) % n_heads != 0:
        if bottleneck_factor > 1:
            bottleneck_factor //= 2
        else:
            n_heads = max(1, n_heads // 2)
            if n_heads == 1:
                break

    model = TemporalSAE(
        dimin=d, width=H, n_heads=n_heads,
        sae_diff_type="topk", kval_topk=k,
        tied_weights=tied_weights,
        n_attn_layers=n_attn_layers,
        bottleneck_factor=bottleneck_factor,
    ).to(device)

    iterator = _make_iterator(cache, train_cfg.batch_size, W)
    opt = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, betas=train_cfg.adam_betas
    )

    arch = "tsae_bhalla"
    t_start = time.time()

    for step in range(train_cfg.n_steps):
        x = next(iterator).to(device).float()                  # (B, W, d)
        x_hat, results = model(x)
        recon = (x_hat - x).pow(2).sum(dim=-1).mean()
        codes = results["novel_codes"] + results["pred_codes"]  # (B, W, H)
        # InfoNCE between consecutive codes: pull z_t and z_{t+1} together.
        contrastive = torch.zeros((), device=device, dtype=x.dtype)
        if contrastive_weight > 0 and W >= 2:
            za = codes[:, :-1].reshape(-1, H)                   # (B*(W-1), H)
            zb = codes[:, 1:].reshape(-1, H)
            za = nn.functional.normalize(za, dim=-1, eps=1e-8)
            zb = nn.functional.normalize(zb, dim=-1, eps=1e-8)
            sim = za @ zb.t()
            labels = torch.arange(za.shape[0], device=device)
            contrastive = 0.5 * (
                nn.functional.cross_entropy(sim, labels)
                + nn.functional.cross_entropy(sim.t(), labels)
            )
        loss = recon + contrastive_weight * contrastive
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()

        last_step = step == train_cfg.n_steps - 1
        if step % train_cfg.log_interval == 0 or last_step:
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / max(elapsed, 1e-6)
            print(
                f"  [{arch} W={W} k={k} a_contr={contrastive_weight}] "
                f"step={step:>5d} recon={recon.item():.4f} "
                f"contr={float(contrastive.item()):.4f} "
                f"| {steps_per_sec:.1f} it/s",
                flush=True,
            )

    return model


def train_tfa(
    *,
    cache: ColoredSourceCache,
    W: int,
    k: int,
    H: int,
    d: int,
    n_heads: int = 4,
    n_attn_layers: int = 1,
    bottleneck_factor: int = 1,
    use_pos_encoding: bool = True,
    device: torch.device,
    train_cfg: TrainConfig,
    weight_decay: float = 1e-4,
    warmup_frac: float = 0.05,
):
    """Train Temporal Feature Analysis (TFA) per the existing
    ``src/bench/architectures/tfa.py`` recipe.

    TFA uses the same ``TemporalSAE`` decomposition as Bhalla 2025
    (predicted codes from causal attention on prior context + novel
    codes from a per-token TopK SAE) but with:

    - **AdamW + cosine LR schedule with warmup** (vs plain Adam in
      `train_tsae`).
    - **Weight-decay split**: 1e-4 on parameters with ndim >= 2,
      0 on biases / 1D.
    - **Optional sinusoidal positional encoding**.
    - **MSE-only reconstruction loss** (no InfoNCE — the contrastive
      term in `train_tsae` was a deliberate addition that does not
      appear in the original TFA recipe).

    Returns the trained model. Use ``model(x_btd)`` at inference to get
    ``(x_recons, results_dict)`` with per-position codes summed via
    ``results['novel_codes'] + results['pred_codes']``.
    """
    import math
    import time
    if W < 2:
        raise ValueError(f"TFA attention requires W >= 2; got W={W}")

    # Adjust bottleneck so width % (bottleneck * n_heads) == 0.
    bf = bottleneck_factor
    while H % (bf * n_heads) != 0:
        if bf > 1:
            bf //= 2
        else:
            n_heads = max(1, n_heads // 2)
            if n_heads == 1 and H % bf != 0:
                break

    sae_diff_type = "topk" if k is not None else "relu"
    model = TFA_TemporalSAE(
        dimin=d, width=H, n_heads=n_heads,
        sae_diff_type=sae_diff_type,
        kval_topk=k,
        tied_weights=True,
        n_attn_layers=n_attn_layers,
        bottleneck_factor=bf,
        use_pos_encoding=use_pos_encoding,
        max_seq_len=max(64, W),
    ).to(device)

    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
    )

    base_lr = train_cfg.lr
    min_lr = base_lr * 0.1
    warmup_steps = max(1, int(train_cfg.n_steps * warmup_frac))
    iterator = _make_iterator(cache, train_cfg.batch_size, W)

    arch = "tfa"
    t_start = time.time()

    model.train()
    for step in range(train_cfg.n_steps):
        if step < warmup_steps:
            current_lr = base_lr * step / max(1, warmup_steps)
        else:
            decay_ratio = (step - warmup_steps) / max(1, train_cfg.n_steps - warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            current_lr = min_lr + coeff * (base_lr - min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        x = next(iterator).to(device).float()
        recons, results = model(x)
        loss = nn.functional.mse_loss(
            recons.reshape(-1, d), x.reshape(-1, d), reduction="sum"
        ) / (x.shape[0] * x.shape[1])

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        last_step = step == train_cfg.n_steps - 1
        if step % train_cfg.log_interval == 0 or last_step:
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / max(elapsed, 1e-6)
            with torch.no_grad():
                novel_l0 = (
                    (results["novel_codes"] > 0).float().sum(dim=-1).mean().item()
                )
            print(
                f"  [{arch} W={W} k={k} pos_enc={use_pos_encoding}] "
                f"step={step:>5d} loss={loss.item():.4f} novel_L0={novel_l0:.2f} "
                f"lr={current_lr:.2e} | {steps_per_sec:.1f} it/s",
                flush=True,
            )

    model.eval()
    return model


def _default_h8_shifts(T: int) -> tuple[int, ...]:
    """Han's default multi-distance shifts: (1, T//4, T//2), deduped, in [1, T-1]."""
    raw = [s for s in (1, max(1, T // 4), max(1, T // 2)) if 1 <= s <= T - 1]
    return tuple(sorted(set(raw)))


def train_txc_h8(
    *,
    cache: ColoredSourceCache,
    F: torch.Tensor,
    W: int,
    k_pos: int,
    H: int,
    d: int,
    device: torch.device,
    train_cfg: TrainConfig,
    alpha: float = 1.0,
    matryoshka_h_size: int | None = None,
    contr_prefix: int | None = None,
    shifts: tuple[int, ...] | None = None,
) -> CellResult:
    """Train Han's H8 lock-in: TXCBareMultiDistanceContrastiveAntidead.

    Architecture: bare TXC + anti-dead + (optional matryoshka H/L) +
    multi-distance InfoNCE on a window prefix. The contrastive term is the
    only thing that directly references the joint distribution of (x_t,
    x_{t+s}) — it's the candidate mechanism for breaking the rotation
    invariance of plain TopK reconstruction on Gaussian sources.

    Args:
        k_pos: per-position TopK budget. Window-level k = k_pos * W.
        alpha: contrastive weight. 0 disables InfoNCE (falls back to bare TXC).
        matryoshka_h_size: enable H/L recon if set; defaults to None (off).
            When None, contr_prefix defaults to int(d_sae * 0.2) per Han.
        contr_prefix: how many latents the contrastive cosine is computed on.
            Pass d_sae to apply the contrastive to the entire dictionary.
        shifts: which lags to contrast. Defaults to (1, T//4, T//2) deduped.
    """
    import time
    if W < 2:
        raise ValueError(f"H8 requires W >= 2, got W={W}")

    if shifts is None:
        shifts = _default_h8_shifts(W)

    k_win = k_pos * W
    model = TXCBareMultiDistanceContrastiveAntidead(
        d_in=d, d_sae=H, T=W, k=k_win,
        shifts=shifts,
        matryoshka_h_size=matryoshka_h_size,
        alpha=alpha,
        contr_prefix=contr_prefix,
    ).to(device)

    # Multi-distance pair generator over our (n_seq, chain_length, d) buffer.
    pair_gen = make_multidistance_pair_gen_gpu(cache.act_chains, W, list(shifts))

    # b_dec geometric-median init on the first batch (paper convention).
    with torch.no_grad():
        first = pair_gen(min(train_cfg.batch_size * 4, 256))[:, 0]  # (B, T, d)
        model.init_b_dec_geometric_median(first.to(device).float())

    opt = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, betas=train_cfg.adam_betas
    )

    history: list[dict] = []
    F_dev = F.to(device=device, dtype=torch.float32)
    N = F.shape[0]
    arch = "txc_h8"
    t_start = time.time()

    for step in range(train_cfg.n_steps):
        x = pair_gen(train_cfg.batch_size).to(device)  # (B, 1+K, T, d)
        loss, _, z = model(x)
        opt.zero_grad()
        loss.backward()
        # Anti-dead recipe: project parallel decoder gradients out, then step.
        if hasattr(model, "remove_gradient_parallel_to_decoder"):
            model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        if hasattr(model, "_normalize_decoder"):
            model._normalize_decoder()

        last_step = step == train_cfg.n_steps - 1
        if step % train_cfg.log_interval == 0 or last_step:
            with torch.no_grad():
                x_eval = pair_gen(train_cfg.batch_size).to(device)
                eval_loss, _, z_eval = model(x_eval)
                l0 = (z_eval > 0).float().sum(dim=-1).mean().item()
                # Decoder dirs: (d, d_sae). Average across the T positions.
                F_hat = model.decoder_dirs_averaged.detach()  # (d, H)
                F_hat_rows = F_hat.T.contiguous()
                rec_sq = squared_axis_recovery(F_dev, F_hat_rows)
                s_adj = chance_adjusted_recovery(rec_sq, N=N, H=H)
            history.append({
                "step": step,
                "loss": float(eval_loss.item()),
                "l0": l0,
                "recovery_squared": rec_sq,
                "s_adj": s_adj,
                "recovery_auc": 0.0,  # AUC not separately computed for H8
            })
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / max(elapsed, 1e-6)
            eta_sec = (train_cfg.n_steps - step - 1) / max(steps_per_sec, 1e-6)
            print(
                f"  [{arch} W={W} shifts={shifts} alpha={alpha}] step={step:>6d} "
                f"loss={eval_loss.item():.4f} L0={l0:.2f} rec_sq={rec_sq:.3f} "
                f"S_adj={s_adj:.3f} | {steps_per_sec:.1f} it/s ETA {eta_sec/60:.1f}m",
                flush=True,
            )

    last = history[-1]
    return CellResult(
        arch=arch,
        W=W,
        k=k_win,
        final_loss=last["loss"],
        final_l0=last["l0"],
        recovery_squared=last["recovery_squared"],
        s_adj=last["s_adj"],
        recovery_auc=last["recovery_auc"],
        history=history,
    )


def oracle_baseline(x: torch.Tensor, F: torch.Tensor, D: int, H: int) -> dict:
    """Spectral oracle ceiling on the same data."""
    F_hat = spectral_oracle(x, lag_D=D, n_components=H)
    rec_sq = squared_axis_recovery(F, F_hat)
    return {
        "recovery_squared": rec_sq,
        "s_adj": chance_adjusted_recovery(rec_sq, N=F.shape[0], H=H),
    }


def random_dictionary_baseline(
    F: torch.Tensor, H: int, n_trials: int, seed: int
) -> dict:
    """Random unit-vector floor."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scores = []
    for _ in range(n_trials):
        R = torch.randn(H, F.shape[1], generator=gen, dtype=F.dtype)
        scores.append(squared_axis_recovery(F, R))
    rec_sq = float(sum(scores) / len(scores))
    return {
        "recovery_squared": rec_sq,
        "s_adj": chance_adjusted_recovery(rec_sq, N=F.shape[0], H=H),
    }
