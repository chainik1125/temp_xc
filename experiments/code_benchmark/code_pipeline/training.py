"""Training helpers for the code_benchmark experiment.

Thin wrappers around the existing SAE / TXC / MLC modules in the
separation_scaling vendor tree. These helpers:

    - package windows correctly for each architecture;
    - share an Adam + periodic-decoder-normalize training loop;
    - log loss curves + sparsity + gradient norm;
    - return a state_dict + loss history for downstream evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

from sae_day.sae import (  # noqa: E402
    TopKSAE,
    TemporalCrosscoder,
    MultiLayerCrosscoder,
)


# ---------------------------------------------------------------------------
# Window builders
# ---------------------------------------------------------------------------


def flatten_for_sae(acts: torch.Tensor) -> torch.Tensor:
    """Input: (N, T, d) → (N*T, d)."""
    return acts.reshape(-1, acts.shape[-1])


def make_txc_windows(acts: torch.Tensor, window_size: int) -> torch.Tensor:
    """Input: (N, T, d) → (N*(T-w+1), w, d).

    Mirrors ``make_causal_windows`` from
    experiments/separation_scaling/vendor/experiments/standard_hmm/
    run_standard_hmm_arch_seed_sweep.py but without the extra permute round-trip.
    """
    unf = acts.unfold(1, window_size, 1)   # (N, T-w+1, d, w)
    unf = unf.permute(0, 1, 3, 2).contiguous()  # (N, T-w+1, w, d)
    return unf.reshape(-1, window_size, acts.shape[-1])


def stack_mlc_layers(acts_per_layer: dict[int, torch.Tensor], layers: list[int]) -> torch.Tensor:
    """Input: {L: (N, T, d)} for each layer → (N*T, L, d)."""
    stacked = torch.stack([acts_per_layer[L] for L in layers], dim=2)  # (N, T, L, d)
    return stacked.reshape(-1, stacked.shape[2], stacked.shape[3])


# ---------------------------------------------------------------------------
# Training loop (unified)
# ---------------------------------------------------------------------------


def _loss_fn(model, x: torch.Tensor) -> torch.Tensor:
    """Sum-over-d MSE, mean-over-batch — matches the vendor objective."""
    x_hat, _ = model(x)
    return (x - x_hat).pow(2).sum(dim=-1).mean()


def train_one_architecture(
    model: torch.nn.Module,
    data: torch.Tensor,
    *,
    n_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_every: int = 100,
    normalize_every: int = 100,
    seed: int = 42,
) -> dict:
    """Train ``model`` on flattened-sample data.

    Returns a dict with:
        loss_history: list[float]             — per-step training loss
        l0_history:   list[float]             — sampled L0 at logged steps
        final_step:   int
        best_loss:    float

    The routine is generic across TopKSAE / TemporalCrosscoder /
    MultiLayerCrosscoder because they all expose ``forward(x) -> (x_hat, z)``
    and a ``normalize_decoder()`` method.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.to(device).train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n_total = data.shape[0]
    loss_history: list[float] = []
    l0_history: list[tuple[int, float]] = []

    best_loss = float("inf")
    for step in range(n_steps):
        idx = torch.randint(n_total, (batch_size,), generator=gen)
        # data may be bf16 on CPU (to keep RAM bounded on containerized hosts).
        # Upcast per-batch on GPU where it's cheap — GPU VRAM is not the
        # bottleneck; container cgroup RAM limit is.
        x = data[idx].to(device, non_blocking=True).float()
        x_hat, z = model(x)
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if (step + 1) % normalize_every == 0 and hasattr(model, "normalize_decoder"):
            model.normalize_decoder()
        loss_history.append(float(loss.detach()))
        best_loss = min(best_loss, loss_history[-1])
        if (step + 1) % log_every == 0:
            with torch.no_grad():
                l0 = (z != 0).float().sum(dim=-1).mean().item()
            l0_history.append((step + 1, l0))
            print(f"[train] step {step+1:>6}/{n_steps}  "
                  f"loss={loss_history[-1]:.6f}  L0={l0:.2f}",
                  flush=True)
    model.eval()
    return {
        "loss_history": loss_history,
        "l0_history": l0_history,
        "final_step": n_steps,
        "best_loss": best_loss,
    }


# ---------------------------------------------------------------------------
# Architecture builders (single source of truth for kwargs → module)
# ---------------------------------------------------------------------------


def build_topk_sae(d_model: int, cfg: dict) -> torch.nn.Module:
    return TopKSAE(
        d_in=d_model,
        d_sae=cfg["d_sae"],
        k=cfg["k"],
        use_relu=cfg.get("use_relu", True),
    )


def build_txc(d_model: int, cfg: dict) -> torch.nn.Module:
    return TemporalCrosscoder(
        d_in=d_model,
        d_sae=cfg["d_sae"],
        T=cfg["T"],
        k_per_pos=cfg.get("k_per_pos"),
        k_total=cfg.get("k_total"),
        use_relu=cfg.get("use_relu", True),
    )


def build_mlc(d_model: int, cfg: dict) -> torch.nn.Module:
    return MultiLayerCrosscoder(
        d_in=d_model,
        d_sae=cfg["d_sae"],
        L=cfg["L"],
        k_per_layer=cfg.get("k_per_layer"),
        k_total=cfg.get("k_total"),
        use_relu=cfg.get("use_relu", True),
    )


FAMILY_BUILDERS = {
    "topk": build_topk_sae,
    "txc": build_txc,
    "mlxc": build_mlc,
}
