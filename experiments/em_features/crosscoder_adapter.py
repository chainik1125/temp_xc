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

from sae_day.sae import TemporalCrosscoder, MultiLayerCrosscoder, TopKSAE  # noqa: E402


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
def get_custom_sae_feature_directions(
    ckpt_path,
    feature_ids,
    device: str = "cuda",
) -> list[torch.Tensor]:
    """Load a custom TopKSAE trained by run_training_sae_custom.py and return
    unit-normed W_dec rows for the requested feature ids. Each tensor is
    (d_in,)."""
    from pathlib import Path
    ckpt = torch.load(Path(ckpt_path), map_location=device)
    ccfg = ckpt["config"]
    from sae_day.sae import TopKSAE
    sae = TopKSAE(d_in=ccfg["d_in"], d_sae=ccfg["d_sae"], k=ccfg["k"])
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval().to(device)
    dec = sae.W_dec.data.float()  # (d_sae, d_in)
    if dec.shape[1] != ccfg["d_in"]:
        dec = dec.T
    rows = dec[feature_ids]
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
    ranking: str = "last_cos",
) -> dict:
    """Decomposition of a (d_in,) diff vector onto TXC decoder rows.

    Supported ``ranking``:
      - ``last_cos``  (default, legacy): cosine at the ``position`` slot only.
      - ``sum_cos``   sum of |cos| across all T decoder slots — rewards features
                      that write in the misalignment direction at any position.
      - ``max_cos``   max |cos| across slots — like sum_cos but picks the peak.
      - ``mean_abs_cos`` mean |cos| (= sum_cos / T) — same ranking as sum_cos.

    Output shape matches the schema used by em-features' sae_decomposition so
    the downstream steering code doesn't need to branch on steerer type.
    """
    if diff_vec.dim() != 1:
        raise ValueError(f"expected 1D diff_vec, got {tuple(diff_vec.shape)}")

    v = diff_vec.float().to(txc.W_dec.device)
    v_u = v / (v.norm() + 1e-8)

    if ranking == "last_cos":
        t = _resolve_position(position, txc.T)
        dec = txc.W_dec.data[t].float()  # (d_sae, d_in)
        dec_u = dec / (dec.norm(dim=-1, keepdim=True) + 1e-8)
        sims = dec_u @ v_u  # (d_sae,)
    else:
        dec_full = txc.W_dec.data.float()  # (T, d_sae, d_in)
        dec_u_full = dec_full / (dec_full.norm(dim=-1, keepdim=True) + 1e-8)
        per_pos_cos = dec_u_full @ v_u  # (T, d_sae)
        if ranking in ("sum_cos", "mean_abs_cos"):
            sims = per_pos_cos.abs().sum(dim=0)  # (d_sae,)
        elif ranking == "max_cos":
            sims = per_pos_cos.abs().max(dim=0).values  # (d_sae,)
        else:
            raise ValueError(f"unknown ranking {ranking!r}")

    sorted_sims, sorted_idx = torch.sort(sims, descending=True)
    k = min(top_k, sims.numel() // 2)
    return {
        "ranking": ranking,
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
def score_txc_features_by_encoder(
    txc: TemporalCrosscoder,
    misalign_windows: torch.Tensor,     # (N, T, d_in) — diffs or raw acts
    control_windows: torch.Tensor | None = None,  # (M, T, d_in), optional
    *,
    top_k: int = 200,
) -> dict:
    """Rank TXC features by their encoder-latent magnitude on misalignment
    windows minus control windows. Uses the TXC's own encoder — leverages its
    window-level sparse code, which the decoder-cosine rankings ignore.

    If ``control_windows`` is None, score = E[|z_i|] on misalignment only.
    """
    mis = misalign_windows.float().to(txc.W_dec.device)
    z_mis = txc.encode(mis)  # (N, d_sae)
    score = z_mis.abs().mean(dim=0)  # (d_sae,)
    if control_windows is not None:
        ctl = control_windows.float().to(txc.W_dec.device)
        z_ctl = txc.encode(ctl)
        score = score - z_ctl.abs().mean(dim=0)
    sorted_sims, sorted_idx = torch.sort(score, descending=True)
    k = min(top_k, score.numel() // 2)
    return {
        "ranking": "encoder_activation",
        "similarities": score,
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
