"""Training driver for colored-source experiments.

Re-implements the training loop from `temporal_crosscoders/train.py` with
explicit `(d_in, d_sae)` arguments so we can run at the proposal's d=N=128
square geometry without monkey-patching `temporal_crosscoders.config`.

Reuses the model classes (StackedSAE, TemporalCrosscoder) directly. Reuses
v5_hmm_sae_baseline.metrics.feature_recovery_score for the AUC continuity
metric, alongside our squared/chance-adjusted recovery from .metrics.
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

from models import StackedSAE, TemporalCrosscoder  # noqa: E402

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
    arch: str            # "stacked_sae" | "txc"
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
) -> CellResult:
    import time
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


def train_pair(
    *,
    cache: ColoredSourceCache,
    F: torch.Tensor,
    W: int,
    k: int,
    H: int,
    d: int,
    device: torch.device,
    train_cfg: TrainConfig,
) -> dict[str, CellResult]:
    """Train a stacked SAE and (if W >= 2) a TXC at this window length.

    Returns:
        dict with keys "stacked_sae" and (if W >= 2) "txc".
    """
    out: dict[str, CellResult] = {}

    sae = StackedSAE(d_in=d, d_sae=H, T=W, k=k).to(device)
    out["stacked_sae"] = _train_one(
        sae, cache, F, "stacked_sae", W, k, H, train_cfg, device
    )

    if W >= 2:
        txc = TemporalCrosscoder(d_in=d, d_sae=H, T=W, k=k).to(device)
        out["txc"] = _train_one(txc, cache, F, "txc", W, k, H, train_cfg, device)

    return out


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
