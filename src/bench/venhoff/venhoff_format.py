"""Export our trained SAE/TempXC/MLC ckpts in Venhoff's expected format.

Venhoff's `optimize_steering_vectors.py` and `hybrid_token.py` load
SAE checkpoints via `utils/sae.py::load_sae`, which expects:

    checkpoint = torch.load(sae_path)
    checkpoint keys:
        input_dim:    int — d_in (residual stream width)
        num_latents:  int — d_sae
        topk:         int — k (number of active features)
        encoder_weight: Tensor (num_latents, d_in)
        encoder_bias:   Tensor (num_latents,)
        decoder_weight: Tensor (num_latents, d_in)  ← transposed vs ours
        b_dec:         Tensor (d_in,)
        activation_mean: Tensor (d_in,)
        activation_mean_model_id:    str
        activation_mean_layer:       int
        activation_mean_n_examples:  int

Path: `{venhoff_root}/train-saes/results/vars/saes/sae_{model_id}_layer{layer}_clusters{n_clusters}.pt`

For our `TopKSAE` the conversion is mechanical. For TempXC and MLC the
native `.W_enc` is 3-D — we squash across the T / n_layers axis to
produce an "SAE-compatible" encoder because Venhoff's steering scripts
call `sae.encode(x)` with single (B, d_in) vectors (per-sentence-mean
activations), not windowed input. **This is the axis-collapse decision
we accepted for Path 1 pipelines**; it's what makes TempXC's Path 3
shim non-trivial but isn't relevant here.

For Phase 2 (steering-vector training) we only need the SAE to produce
cluster-argmax assignments on per-sentence-mean activations, and for
decoder directions. Both are reproducible from either (a) the TopKSAE
ckpt directly, or (b) a per-position reduction of TempXC/MLC. See
`export_from_tempxc()` for the reduction we apply.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("venhoff.format")


VENHOFF_SAE_SUBPATH = "train-saes/results/vars/saes"


def venhoff_sae_path(
    venhoff_root: Path,
    model_id: str,
    layer: int,
    n_clusters: int,
) -> Path:
    """Canonical path Venhoff's `load_sae()` reads from."""
    return Path(venhoff_root) / VENHOFF_SAE_SUBPATH / f"sae_{model_id}_layer{layer}_clusters{n_clusters}.pt"


def _load_our_ckpt(ckpt_path: Path, device: str = "cpu") -> dict:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    return payload


def _load_our_mean(mean_pkl: Path) -> np.ndarray:
    with mean_pkl.open("rb") as f:
        p = pickle.load(f)
    mean = np.asarray(p["activation_mean"], dtype=np.float32)
    return mean


def export_topk_sae(
    our_ckpt: Path,
    our_mean_pkl: Path,
    out_path: Path,
    model_id: str,
    layer: int,
    n_clusters: int,
    n_examples: int,
) -> Path:
    """Convert a TopKSAE ckpt + Path 1 mean sidecar into Venhoff's schema."""
    payload = _load_our_ckpt(our_ckpt)
    sd = payload["state_dict"]
    mean = _load_our_mean(our_mean_pkl)

    # Our TopKSAE shapes:
    #   W_enc: (d_sae, d_in)       ← encoder weight
    #   b_enc: (d_sae,)
    #   W_dec: (d_in, d_sae)       ← decoder weight, columns are features
    #   b_dec: (d_in,)
    W_enc = sd["W_enc"]
    b_enc = sd["b_enc"]
    W_dec = sd["W_dec"]
    b_dec = sd["b_dec"]
    d_sae, d_in = W_enc.shape
    assert W_dec.shape == (d_in, d_sae), f"bad W_dec shape {W_dec.shape}"

    # Venhoff's `decoder_weight` is (num_latents, d_in) — transpose of ours.
    venhoff_decoder_weight = W_dec.T.contiguous()

    ckpt = {
        "input_dim": int(d_in),
        "num_latents": int(d_sae),
        "topk": int(payload["config"]["k"]),
        "encoder_weight": W_enc.contiguous().float(),
        "encoder_bias": b_enc.contiguous().float(),
        "decoder_weight": venhoff_decoder_weight.float(),
        "b_dec": b_dec.contiguous().float(),
        "activation_mean": torch.from_numpy(mean).float(),
        "activation_mean_model_id": model_id,
        "activation_mean_layer": int(layer),
        "activation_mean_n_examples": int(n_examples),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.rename(out_path)
    log.info("[done] exported sae→venhoff | model=%s | layer=%d | clusters=%d | path=%s",
             model_id, layer, n_clusters, out_path)
    return out_path


def export_from_tempxc(
    our_ckpt: Path,
    our_mean_pkl: Path,
    out_path: Path,
    model_id: str,
    layer: int,
    n_clusters: int,
    n_examples: int,
) -> Path:
    """Reduce TempXC's (T, d_in, d_sae) encoder to a Venhoff-compatible 2-D encoder.

    Venhoff's `sae.encode((B, d_in))` expects a 2-D weight. TempXC's
    `W_enc` is `(T, d_in, d_sae)`. We reduce by **summing over T**
    before transposing — this matches the shared-z forward pass
    semantics (which also sums over T before TopK).

    `activation_mean` is the per-position Path 3 mean; we reduce it
    analogously by averaging over T to get a (d_in,) vector. Path 3's
    mean is saved as (d_model,) already (per-token stats), so no
    conversion needed there.
    """
    payload = _load_our_ckpt(our_ckpt)
    sd = payload["state_dict"]
    mean = _load_our_mean(our_mean_pkl)

    W_enc = sd["W_enc"]        # (T, d_in, d_sae)
    b_enc = sd["b_enc"]        # (d_sae,)
    W_dec = sd["W_dec"]        # (d_sae, T, d_in)
    b_dec = sd["b_dec"]        # (T, d_in)
    T, d_in, d_sae = W_enc.shape
    assert W_dec.shape == (d_sae, T, d_in)

    # Sum over T for encoder: (T, d_in, d_sae) → (d_in, d_sae) → (d_sae, d_in)
    W_enc_2d = W_enc.sum(dim=0).T.contiguous()       # (d_sae, d_in)
    # Mean over T for decoder: (d_sae, T, d_in) → (d_sae, d_in)
    W_dec_2d = W_dec.mean(dim=1).contiguous()        # (d_sae, d_in)
    # Mean over T for bias: (T, d_in) → (d_in,)
    b_dec_2d = b_dec.mean(dim=0).contiguous()        # (d_in,)

    ckpt = {
        "input_dim": int(d_in),
        "num_latents": int(d_sae),
        "topk": int(payload["config"]["k"]),
        "encoder_weight": W_enc_2d.float(),
        "encoder_bias": b_enc.contiguous().float(),
        "decoder_weight": W_dec_2d.float(),
        "b_dec": b_dec_2d.float(),
        "activation_mean": torch.from_numpy(mean).float(),
        "activation_mean_model_id": model_id,
        "activation_mean_layer": int(layer),
        "activation_mean_n_examples": int(n_examples),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.rename(out_path)
    log.info(
        "[done] exported tempxc→venhoff (sum-T reduction) | model=%s | layer=%d | clusters=%d | path=%s",
        model_id, layer, n_clusters, out_path,
    )
    return out_path


def export_from_mlc(
    our_ckpt: Path,
    our_mean_pkl: Path,
    out_path: Path,
    model_id: str,
    layer: int,
    n_clusters: int,
    n_examples: int,
) -> Path:
    """Reduce MLC's (n_layers, d_in, d_sae) encoder by summing over layers.

    Same reduction as TempXC — MLC's architecture is mathematically
    identical, only the axis semantics differ. The activation_mean for
    MLC is (n_layers, d_model); we average over layers for the Venhoff
    1-D format.
    """
    payload = _load_our_ckpt(our_ckpt)
    sd = payload["state_dict"]

    # For MLC the mean pkl payload has shape (n_layers, d_model) not (d_model,)
    with our_mean_pkl.open("rb") as f:
        p = pickle.load(f)
    mean_2d = np.asarray(p["activation_mean"], dtype=np.float32)  # (n_layers, d_model)
    mean_1d = mean_2d.mean(axis=0)                                # (d_model,)

    W_enc = sd["W_enc"]        # (n_layers, d_in, d_sae)
    b_enc = sd["b_enc"]        # (d_sae,)
    W_dec = sd["W_dec"]        # (d_sae, n_layers, d_in)
    b_dec = sd["b_dec"]        # (n_layers, d_in)
    n_layers, d_in, d_sae = W_enc.shape
    assert W_dec.shape == (d_sae, n_layers, d_in)

    W_enc_2d = W_enc.sum(dim=0).T.contiguous()
    W_dec_2d = W_dec.mean(dim=1).contiguous()
    b_dec_2d = b_dec.mean(dim=0).contiguous()

    ckpt = {
        "input_dim": int(d_in),
        "num_latents": int(d_sae),
        "topk": int(payload["config"]["k"]),
        "encoder_weight": W_enc_2d.float(),
        "encoder_bias": b_enc.contiguous().float(),
        "decoder_weight": W_dec_2d.float(),
        "b_dec": b_dec_2d.float(),
        "activation_mean": torch.from_numpy(mean_1d).float(),
        "activation_mean_model_id": model_id,
        "activation_mean_layer": int(layer),
        "activation_mean_n_examples": int(n_examples),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.rename(out_path)
    log.info(
        "[done] exported mlc→venhoff (sum-layers reduction) | model=%s | layer=%d | clusters=%d | path=%s",
        model_id, layer, n_clusters, out_path,
    )
    return out_path


def export(
    arch: str,
    our_ckpt: Path,
    our_mean_pkl: Path,
    out_path: Path,
    model_id: str,
    layer: int,
    n_clusters: int,
    n_examples: int,
) -> Path:
    """Dispatch on arch; return the Venhoff-format ckpt path."""
    if arch == "sae":
        return export_topk_sae(our_ckpt, our_mean_pkl, out_path, model_id, layer, n_clusters, n_examples)
    if arch == "tempxc":
        return export_from_tempxc(our_ckpt, our_mean_pkl, out_path, model_id, layer, n_clusters, n_examples)
    if arch == "mlc":
        return export_from_mlc(our_ckpt, our_mean_pkl, out_path, model_id, layer, n_clusters, n_examples)
    raise ValueError(f"unknown arch {arch!r}")
