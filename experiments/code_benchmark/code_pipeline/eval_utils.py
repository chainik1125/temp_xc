"""Shared helpers for the eval passes.

- ``build_model_from_checkpoint`` reconstitutes any of the three architectures
  from a ``run_training.py`` checkpoint payload.
- ``encode_windows`` returns per-window latents and the (chunk, token) position
  each latent is aligned to (the window's *last* token for TXC; per-token for
  SAE / MLC).
- ``per_token_labels`` joins the labeler to the cached sources.jsonl.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

from sae_day.sae import (  # noqa: E402
    TopKSAE,
    TemporalCrosscoder,
    MultiLayerCrosscoder,
)

from code_pipeline.training import (  # noqa: E402
    flatten_for_sae,
    make_txc_windows,
    stack_mlc_layers,
)
from code_pipeline.python_state_labeler import labels_for_chunk  # noqa: E402


FIELD_NAMES = ["bracket_depth", "indent_spaces", "scope_nesting",
               "scope_kind", "distance_to_header", "has_await"]


# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------


def build_model_from_checkpoint(ckpt: dict, d_model: int) -> tuple[torch.nn.Module, str]:
    family = ckpt["family"]
    kw = ckpt["config"]
    if family == "topk":
        model = TopKSAE(d_in=d_model, d_sae=kw["d_sae"], k=kw["k"],
                        use_relu=kw.get("use_relu", True))
    elif family == "txc":
        model = TemporalCrosscoder(d_in=d_model, d_sae=kw["d_sae"], T=kw["T"],
                                   k_per_pos=kw.get("k_per_pos"),
                                   k_total=kw.get("k_total"),
                                   use_relu=kw.get("use_relu", True))
    elif family == "mlxc":
        model = MultiLayerCrosscoder(d_in=d_model, d_sae=kw["d_sae"], L=kw["L"],
                                     k_per_layer=kw.get("k_per_layer"),
                                     k_total=kw.get("k_total"),
                                     use_relu=kw.get("use_relu", True))
    else:
        raise ValueError(f"Unknown family: {family!r}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, family


# ---------------------------------------------------------------------------
# Per-(chunk, token) latent extraction, aligned to labels
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_sae_per_token(
    model: TopKSAE,
    acts: torch.Tensor,        # (N, T, d)
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SAE latents, 1:1 with tokens.

    Returns ``(latents, x_hat, chunk_idx, token_idx)`` with shape
    ``(N*T, d_sae)``, ``(N*T, d)``, ``(N*T,)``, ``(N*T,)``.
    """
    N, T, d = acts.shape
    model.to(device)
    flat = acts.reshape(-1, d)
    zs = []
    xh = []
    for i in range(0, flat.shape[0], batch_size * T):
        batch = flat[i : i + batch_size * T].to(device).float()
        xb, z = model(batch)
        zs.append(z.cpu())
        xh.append(xb.cpu())
    latents = torch.cat(zs, dim=0).numpy()
    x_hat = torch.cat(xh, dim=0).numpy()
    chunk_idx = np.repeat(np.arange(N), T)
    token_idx = np.tile(np.arange(T), N)
    return latents, x_hat, chunk_idx, token_idx


@torch.no_grad()
def encode_txc_per_window(
    model: TemporalCrosscoder,
    acts: torch.Tensor,        # (N, T, d)
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TXC latents, one per window, assigned to the window's **last** token.

    Returns ``(latents, x_hat_last, chunk_idx, token_idx)`` with
    ``x_hat_last`` being the reconstruction of just the last position of each
    window — this aligns with the same (chunk, token) position the latent is
    labelled by.
    """
    window_size = model.T
    windows = make_txc_windows(acts, window_size)   # (N*(T-w+1), w, d)
    N, T, d = acts.shape
    n_windows_per_chunk = T - window_size + 1
    model.to(device)
    zs = []
    xh_last = []
    for i in range(0, windows.shape[0], batch_size):
        batch = windows[i : i + batch_size].to(device).float()
        xb, z = model(batch)
        zs.append(z.cpu())
        xh_last.append(xb[:, -1, :].cpu())
    latents = torch.cat(zs, dim=0).numpy()
    x_hat_last = torch.cat(xh_last, dim=0).numpy()
    chunk_idx = np.repeat(np.arange(N), n_windows_per_chunk)
    token_idx = np.tile(
        np.arange(window_size - 1, T), N
    )  # the last-position token index
    return latents, x_hat_last, chunk_idx, token_idx


@torch.no_grad()
def encode_mlc_per_token(
    model: MultiLayerCrosscoder,
    acts_by_layer: dict[int, torch.Tensor],   # each (N, T, d)
    layers: list[int],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MLC latents, 1:1 with tokens (one per L-stack at each position).

    Returns ``(latents, x_hat_anchor, chunk_idx, token_idx)`` where
    ``x_hat_anchor`` is the reconstruction at the *anchor layer* position
    within the L-stack (the caller tells us which). For coarse NMSE we use the
    middle layer.
    """
    N, T, d = next(iter(acts_by_layer.values())).shape
    stacked = stack_mlc_layers(acts_by_layer, layers)    # (N*T, L, d)
    model.to(device)
    zs = []
    xh_anchor = []
    anchor_in_stack = len(layers) // 2
    for i in range(0, stacked.shape[0], batch_size):
        batch = stacked[i : i + batch_size].to(device).float()
        xb, z = model(batch)
        zs.append(z.cpu())
        xh_anchor.append(xb[:, anchor_in_stack, :].cpu())
    latents = torch.cat(zs, dim=0).numpy()
    x_hat_anchor = torch.cat(xh_anchor, dim=0).numpy()
    chunk_idx = np.repeat(np.arange(N), T)
    token_idx = np.tile(np.arange(T), N)
    return latents, x_hat_anchor, chunk_idx, token_idx


# ---------------------------------------------------------------------------
# Per-token labels for every chunk in a list of sources.jsonl rows
# ---------------------------------------------------------------------------


def labels_for_sources(sources: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Returns ``{field: (N, T) int64}`` for all chunks in ``sources``."""
    rows: dict[str, list[list[int]]] = {k: [] for k in FIELD_NAMES}
    for row in sources:
        lbl = labels_for_chunk(row["source"],
                               [tuple(o) for o in row["char_offsets"]])
        for k in FIELD_NAMES:
            rows[k].append(lbl[k])
    return {k: np.asarray(v, dtype=np.int64) for k, v in rows.items()}


def gather_labels(
    labels_nt: dict[str, np.ndarray],
    chunk_idx: np.ndarray,
    token_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Index labels at (chunk_idx, token_idx) pairs — returns 1-D arrays."""
    return {k: v[chunk_idx, token_idx] for k, v in labels_nt.items()}


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------


def nmse(x: np.ndarray, x_hat: np.ndarray, eps: float = 1e-12) -> float:
    """Normalized MSE: mean over samples of ``|x - x_hat|² / |x|²``.

    Handles ``x=0`` rows by using ``eps`` in the denominator. Returns a scalar.
    """
    sq_err = ((x - x_hat) ** 2).sum(axis=-1)
    sq_norm = (x ** 2).sum(axis=-1) + eps
    return float((sq_err / sq_norm).mean())


def nmse_per_sample(x: np.ndarray, x_hat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    sq_err = ((x - x_hat) ** 2).sum(axis=-1)
    sq_norm = (x ** 2).sum(axis=-1) + eps
    return sq_err / sq_norm


def l0_mean(latents: np.ndarray) -> float:
    return float((latents != 0).sum(axis=-1).mean())


def explained_variance(x: np.ndarray, x_hat: np.ndarray) -> float:
    """1 - Var(x - x_hat) / Var(x), a scalar."""
    var_err = np.var(x - x_hat, axis=0).sum()
    var_x = np.var(x, axis=0).sum() + 1e-12
    return float(1.0 - var_err / var_x)


# ---------------------------------------------------------------------------
# Bucketing helpers for per-category breakdowns
# ---------------------------------------------------------------------------


def bucketize(values: np.ndarray, thresholds: list[int]) -> np.ndarray:
    """Map each value to bucket index based on [t_0, t_1, ..., t_k, +∞].

    E.g. thresholds = [0, 1, 2, 3] with value 5 → bucket 3 (the "≥3" bucket).
    """
    out = np.zeros_like(values, dtype=np.int64)
    for i, t in enumerate(thresholds):
        out = np.where(values >= t, i, out)
    return out


def bucket_labels_for(
    buckets: np.ndarray,
    thresholds: list[int],
) -> list[str]:
    """Human-readable labels for a thresholds list."""
    labels = []
    for i, t in enumerate(thresholds):
        if i == len(thresholds) - 1:
            labels.append(f"≥{t}")
        else:
            labels.append(f"{t}–{thresholds[i + 1] - 1}")
    return labels
