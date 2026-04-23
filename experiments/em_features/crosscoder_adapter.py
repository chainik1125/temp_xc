"""Adapter that exposes a trained TemporalCrosscoder to the em-features
ActivationSteerer, which expects a flat list of (d_model,) direction vectors.

Two entry points:

    get_txc_feature_directions(txc, feature_ids, position="last")
        -> list[Tensor(d_model,)]
    decompose_diff_on_txc(diff_vec, txc, top_k=200, position="last")
        -> dict mirroring em-features' sae_decomposition output schema.

Directions are unit-normed so that the em-features α grid has a consistent
magnitude interpretation across SAE and TXC runs (SAE decoder rows are
unit-norm by convention; TXC decoder rows are joint-normed across (T, d_in)
per-latent, so raw slices are shorter than unit).
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

from sae_day.sae import TemporalCrosscoder, MultiLayerCrosscoder  # noqa: E402


def _resolve_position(position, T):
    if position == "last":
        return T - 1
    if position == "first":
        return 0
    if isinstance(position, int):
        if not (0 <= position < T):
            raise ValueError(f"position {position} out of [0,{T})")
        return position
    raise ValueError(f"unknown position: {position!r}")


@torch.no_grad()
def get_txc_feature_directions(
    txc: TemporalCrosscoder,
    feature_ids,
    position="last",
) -> list[torch.Tensor]:
    """Return unit-normed write-directions for the given features at the given
    window position. Shape of each returned tensor: (d_in,)."""
    t = _resolve_position(position, txc.T)
    rows = txc.W_dec.data[t, feature_ids, :]  # (k, d_in)
    rows = rows / (rows.norm(dim=-1, keepdim=True) + 1e-8)
    return [rows[i].clone() for i in range(rows.shape[0])]


@torch.no_grad()
def get_mlc_feature_directions(
    mlc: MultiLayerCrosscoder,
    feature_ids,
) -> list[list[torch.Tensor]]:
    """Return per-feature, per-layer unit-normed decoder directions.

    Result shape: ``result[feature_rank_in_list][layer_idx]`` is a ``(d_in,)``
    tensor ready for ``ActivationSteerer(steering_vectors=..., layer_indices=layer_module_idx, ...)``.
    Unit-normed per-slice so α has a consistent magnitude across SAE/TXC/MLC.
    """
    rows = mlc.W_dec.data[:, feature_ids, :]  # (L, k, d_in)
    rows = rows / (rows.norm(dim=-1, keepdim=True) + 1e-8)
    out: list[list[torch.Tensor]] = []
    for fi in range(rows.shape[1]):
        out.append([rows[layer_i, fi].clone() for layer_i in range(rows.shape[0])])
    return out


@torch.no_grad()
def decompose_diff_on_mlc(
    diff_vecs_by_layer: dict[int, torch.Tensor],
    mlc: MultiLayerCrosscoder,
    layers: list[int],
    *,
    top_k: int = 200,
) -> dict:
    """Cosine-similarity decomposition of stacked per-layer diff vectors onto
    MLC decoder rows.

    For each feature i, score = Σ_ℓ cos(W_dec[ℓ, i, :], diff_vecs_by_layer[ℓ]).
    Uses the sum rather than the encoder-latent magnitude so the metric is
    comparable across SAE/TXC/MLC (all three decompose onto decoder rows).
    """
    per_layer_scores = []
    for layer_i, L in enumerate(layers):
        if L not in diff_vecs_by_layer:
            raise KeyError(f"layer {L} missing from diff_vecs_by_layer")
        v = diff_vecs_by_layer[L].float().to(mlc.W_dec.device)
        v_u = v / (v.norm() + 1e-8)
        dec = mlc.W_dec.data[layer_i].float()  # (d_sae, d_in)
        dec_u = dec / (dec.norm(dim=-1, keepdim=True) + 1e-8)
        per_layer_scores.append(dec_u @ v_u)  # (d_sae,)
    sims = torch.stack(per_layer_scores, dim=0).sum(dim=0)  # (d_sae,)

    sorted_sims, sorted_idx = torch.sort(sims, descending=True)
    k = min(top_k, sims.numel() // 2)
    return {
        "similarities": sims,
        "sorted_similarities": sorted_sims,
        "sorted_indices": sorted_idx,
        "top_features": {
            "indices": sorted_idx[:k].cpu().tolist(),
            "similarities": sorted_sims[:k].cpu().tolist(),
            "count": k,
        },
        "bottom_features": {
            "indices": sorted_idx[-k:].flip(0).cpu().tolist(),
            "similarities": sorted_sims[-k:].flip(0).cpu().tolist(),
            "count": k,
        },
    }


@torch.no_grad()
def decompose_diff_on_txc(
    diff_vec: torch.Tensor,
    txc: TemporalCrosscoder,
    *,
    top_k: int = 200,
    position="last",
) -> dict:
    """Cosine-similarity decomposition of a (d_in,) diff vector onto TXC
    decoder rows at `position`.

    Output shape matches the schema used by em-features' sae_decomposition so
    the downstream steering code doesn't need to branch on steerer type.
    """
    if diff_vec.dim() != 1:
        raise ValueError(f"expected 1D diff_vec, got {tuple(diff_vec.shape)}")
    t = _resolve_position(position, txc.T)
    dec = txc.W_dec.data[t].float()  # (d_sae, d_in)
    dec_u = dec / (dec.norm(dim=-1, keepdim=True) + 1e-8)
    v = diff_vec.float().to(dec_u.device)
    v_u = v / (v.norm() + 1e-8)
    sims = dec_u @ v_u  # (d_sae,)

    sorted_sims, sorted_idx = torch.sort(sims, descending=True)
    k = min(top_k, sims.numel() // 2)
    return {
        "similarities": sims,
        "sorted_similarities": sorted_sims,
        "sorted_indices": sorted_idx,
        "top_features": {
            "indices": sorted_idx[:k].cpu().tolist(),
            "similarities": sorted_sims[:k].cpu().tolist(),
            "count": k,
        },
        "bottom_features": {
            "indices": sorted_idx[-k:].flip(0).cpu().tolist(),
            "similarities": sorted_sims[-k:].flip(0).cpu().tolist(),
            "count": k,
        },
    }
