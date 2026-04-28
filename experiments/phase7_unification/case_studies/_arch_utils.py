"""Architecture-agnostic helpers for Agent C's case studies.

Includes `load_phase7_model_safe`: a wrapper around
`_load_phase7_model` that defends against meta-dict fields whose value
is explicitly `None`, where the loader does
`float(meta.get("alpha", 1.0))` — that returns `None` (not the default
`1.0`) when `meta["alpha"] is None`, then `float(None)` crashes. Affects
SubseqH8 and TXCBareMultiDistanceContrastiveAntidead branches at present.
Per agent_c_brief.md DO-NOT list, Agent C must not edit
run_probing_phase7.py directly; this wrapper coerces None → canonical
default before delegating, which is functionally equivalent and contained.


Both C.i (HH-RLHF dataset understanding) and C.ii (steering) need to
do two things across heterogeneous arch families:

  1. Extract a `(d_in, d_sae)` per-feature decoder-direction matrix —
     used for AxBench-style additive steering, where the intervention
     adds `strength * unit_norm(d_dec[:, j])` to the L12 residual.

  2. Encode a `(N, S, d_in)` sequence batch into `(N, S, d_sae)`
     per-position activations — used for HH-RLHF analysis where we
     need to aggregate features over the response-token positions
     of each example.

Each Phase 7 src_class has a different encode signature:

  TopKSAE                                  encode((B, d_in)) -> (B, d_sae)
  TemporalMatryoshkaBatchTopKSAE           encode((B, d_in), use_threshold=True) -> (B, d_sae)
  MultiLayerCrosscoder / MLCContrastive*   encode((B, n_layers, d_in)) -> (B, d_sae)
  MatryoshkaTXCDRContrastiveMultiscale     encode((B, T, d_in)) -> (B, d_sae)
  TXCBareAntidead / TemporalCrosscoder /   encode((B, T, d_in)) -> (B, d_sae)
  TXCBareMultiDistanceContrastiveAntidead
  SubseqTXCBareAntidead / SubseqH8         encode((B, T_max, d_in)) -> (B, d_sae)
  TemporalSAE (TFA)                        forward((B, L, d_in)) -> (recons, inter)

For window archs, "encode at position t" means "feed the T-token window
ending at t through encode()". Boundary handling: positions t < T-1 don't
have a full window and are skipped (returning zeros for those positions).

These helpers wrap `_load_phase7_model`'s output uniformly — callers
get a `(N, S, d_sae)` tensor regardless of arch family.
"""
from __future__ import annotations

from typing import Any

import torch


# ──────────────────────────────────────────────── safe-load wrapper ──


_DEFAULTED_NONE_FIELDS = {
    "alpha": 1.0,
    "gamma": 0.5,
    "n_scales": 3,
}


def load_phase7_model_safe(meta: dict, ckpt_path, device):
    """Wrapper around `run_probing_phase7._load_phase7_model` that fills
    in canonical defaults for meta-dict fields whose value is `None`.

    The loader currently does `float(meta.get("alpha", 1.0))` which
    returns `None` (not the default) when `meta["alpha"] is None`,
    causing `float(None)` to crash. This bites archs whose meta is
    serialised with explicit None for unused arch-recipe fields
    (SubseqH8, H8 multi-distance row 32 onward).

    We defensively replace None values with their canonical defaults
    before delegating to the loader.
    """
    from experiments.phase7_unification.run_probing_phase7 import _load_phase7_model
    fixed = dict(meta)
    for key, default in _DEFAULTED_NONE_FIELDS.items():
        if fixed.get(key) is None:
            fixed[key] = default
    return _load_phase7_model(fixed, ckpt_path, device)


# ──────────────────────────────────────────── arch-class taxonomy ──


PER_TOKEN_CLASSES = {"TopKSAE", "TemporalMatryoshkaBatchTopKSAE", "TemporalSAE"}
MLC_CLASSES = {"MultiLayerCrosscoder", "MLCContrastive", "MLCContrastiveMultiscale"}
WINDOW_CLASSES = {
    "TemporalCrosscoder",
    "TXCBareAntidead",
    "MatryoshkaTXCDRContrastiveMultiscale",
    "TXCBareMultiDistanceContrastiveAntidead",
    "SubseqTXCBareAntidead",
    "SubseqH8",
}


def window_T(model, src_class: str, meta: dict) -> int:
    """The window length to slide over a sequence for a given arch.

    Per-token archs return 1 (no sliding). Window archs return T (or T_max
    for subseq-family). MLC archs return 1 (per-token over layer cube).
    """
    if src_class in PER_TOKEN_CLASSES or src_class in MLC_CLASSES:
        return 1
    if meta.get("T") is not None:
        return int(meta["T"])
    if meta.get("T_max") is not None:
        return int(meta["T_max"])
    raise ValueError(f"no T/T_max for src_class={src_class}")


# ─────────────────────────────────────────── decoder directions ──


def decoder_direction_matrix(model, src_class: str) -> torch.Tensor:
    """Return a `(d_in, d_sae)` per-feature decoder-direction matrix.

    Conventions across families:
      - TopKSAE                          : `W_dec` is already `(d_in, d_sae)`.
      - TemporalMatryoshkaBatchTopKSAE   : `W_dec` is `(d_sae, d_in)` (paper
                                            convention, see tsae_paper.py),
                                            so we transpose.
      - MatryoshkaTXCDR* (any)           : has T per-scale decoders; we
                                            return the position-averaged
                                            `decoder_dirs_averaged` field.
      - TemporalCrosscoder + bare/H8     : `W_dec` is `(d_sae, T, d_in)`;
                                            mean over T then transpose.
      - SubseqH8 / SubseqTXC etc.        : same as TemporalCrosscoder
                                            (their `W_dec` shape is the same).
      - TFA (TemporalSAE)                : has a `D` parameter of shape
                                            `(d_sae, d_in)`; transpose.
      - MLC family                       : has per-layer decoders; the
                                            decoder for steering is the
                                            average over layers (analogous
                                            to TXC's average over T).
    """
    if src_class == "TopKSAE":
        return model.W_dec.data                                          # (d_in, d_sae)
    if src_class == "TemporalMatryoshkaBatchTopKSAE":
        return model.W_dec.data.t().contiguous()                         # (d_in, d_sae)
    if src_class.startswith("Matryoshka") or src_class == "MatryoshkaTXCDRContrastiveMultiscale":
        # PositionMatryoshkaTXCDR + descendants expose decoder_dirs_averaged
        return model.decoder_dirs_averaged                               # (d_in, d_sae)
    if src_class in {"TemporalCrosscoder", "TXCBareAntidead",
                     "TXCBareMultiDistanceContrastiveAntidead"}:
        # W_dec is (d_sae, T, d_in); average over T then transpose to (d_in, d_sae)
        return model.W_dec.data.mean(dim=1).t().contiguous()
    if src_class in {"SubseqTXCBareAntidead", "SubseqH8"}:
        # Same shape convention as TemporalCrosscoder (subseq just samples
        # which positions feed gradient; the parameter shape is identical).
        return model.W_dec.data.mean(dim=1).t().contiguous()
    if src_class == "TemporalSAE":
        # TFA uses D (d_sae, d_in); transpose.
        return model.D.data.t().contiguous()
    if src_class in MLC_CLASSES:
        # MLC W_dec is (d_sae, n_layers, d_in); average over layers.
        return model.W_dec.data.mean(dim=1).t().contiguous()
    raise ValueError(f"unknown src_class={src_class!r}")


def unit_decoder_direction(model, src_class: str, feature_idx: int) -> torch.Tensor:
    """Unit-norm decoder direction for a single feature, shape `(d_in,)`.

    AxBench-style steering uses this — multiply by a strength scalar and
    add to the residual stream at every token during decode.
    """
    D = decoder_direction_matrix(model, src_class)                       # (d_in, d_sae)
    d = D[:, feature_idx]                                                # (d_in,)
    return d / d.norm().clamp(min=1e-8)


# ────────────────────────────────────────────────── encode helpers ──


def _slide_windows(seq: torch.Tensor, T: int) -> torch.Tensor:
    """`(N, S, d_in)` -> `(N, S - T + 1, T, d_in)` sliding T-window stride 1."""
    return seq.unfold(dimension=1, size=T, step=1).movedim(-1, 2)


@torch.no_grad()
def encode_per_position(
    model,
    src_class: str,
    seq: torch.Tensor,
    *,
    T: int | None = None,
    use_threshold: bool = True,
    chunk: int = 32,
) -> torch.Tensor:
    """Encode a sequence batch into `(N, S, d_sae)` per-position
    aggregated activations. The arch-uniform contract:

      - Per-token archs (SAE, T-SAE, TFA): input shape `(N, S, d_in)`;
        every position 0..S-1 is encoded.
      - MLC archs: input shape `(N, S, n_layers, d_in)` — every position
        encoded via the MLC layer-cube `encode((B, n_layers, d_in))`.
      - Window archs: input shape `(N, S, d_in)`; each window of T
        positions is encoded; the resulting `(d_sae,)` is attributed to
        the LAST position of the window (window covering tokens
        t-T+1..t lands at index t). Positions 0..T-2 receive zero.

    `chunk` controls the inner GPU batch over (N*S) per-token forwards
    or (N*K) per-window forwards. 32 fits in <2 GB even for d_sae=18432.
    """
    device = next(model.parameters()).device
    seq = seq.to(device)
    if src_class in MLC_CLASSES:
        # MLC takes (B, n_layers, d_in) input.
        N, S, n_lay, d_in = seq.shape
        d_sae = _d_sae_of(model, src_class)
        out = torch.zeros((N, S, d_sae), dtype=torch.float32, device=device)
        flat = seq.reshape(N * S, n_lay, d_in)
        for i in range(0, flat.shape[0], chunk * S):
            j = min(i + chunk * S, flat.shape[0])
            sub = flat[i:j]
            z = model.encode(sub)
            out.view(N * S, d_sae)[i:j] = z.float()
        return out
    N, S, d_in = seq.shape
    d_sae = _d_sae_of(model, src_class)
    out = torch.zeros((N, S, d_sae), dtype=torch.float32, device=device)

    if src_class in PER_TOKEN_CLASSES:
        # Flatten to (N*S, d_in), encode in chunks.
        flat = seq.reshape(N * S, d_in)
        for i in range(0, flat.shape[0], chunk * S):
            j = min(i + chunk * S, flat.shape[0])
            sub = flat[i:j]
            if src_class == "TemporalMatryoshkaBatchTopKSAE":
                z = model.encode(sub, use_threshold=use_threshold)
                if isinstance(z, tuple):
                    z = z[0]
            elif src_class == "TemporalSAE":
                # TFA expects (B, L, d_in) but we want per-token. Feed
                # each token as a length-1 sequence and pull the novel-codes
                # slice from its inter dict.
                recons, inter = model(sub.unsqueeze(1))
                z = inter["novel_codes"].squeeze(1)
            else:  # TopKSAE
                z = model.encode(sub)
            out.view(N * S, d_sae)[i:j] = z.float()
        return out

    # Window archs.
    T = T if T is not None else 1
    if T < 1:
        raise ValueError(f"T must be >=1, got {T}")
    if S < T:
        return out  # nothing to encode
    windows = _slide_windows(seq, T)                                    # (N, K, T, d_in)
    K = windows.shape[1]                                                # = S - T + 1
    flat_w = windows.reshape(N * K, T, d_in)
    out_per_window = torch.zeros(
        (N * K, d_sae), dtype=torch.float32, device=device,
    )
    for i in range(0, flat_w.shape[0], chunk):
        j = min(i + chunk, flat_w.shape[0])
        sub = flat_w[i:j]
        z = model.encode(sub)
        out_per_window[i:j] = z.float()
    out_per_window = out_per_window.view(N, K, d_sae)
    # Right-edge attribution: window [t-T+1 .. t] -> position t (=window index + T-1)
    out[:, T - 1: T - 1 + K, :] = out_per_window
    return out


def _d_sae_of(model, src_class: str) -> int:
    """Read the dictionary size from a loaded model, regardless of arch."""
    if hasattr(model, "d_sae"):
        return int(model.d_sae)
    if hasattr(model, "dict_size"):
        return int(model.dict_size)
    if hasattr(model, "width"):
        return int(model.width)        # TFA naming
    raise AttributeError(f"can't determine d_sae for {src_class}")
