"""Ambiguous-pair HMM: the proposal's "weaker" Section-4 alternative.

The colored-source theorem (the strong version we already implemented in
`colored_sources.py`) bounds local *direction* recovery — but its Gaussian
sources give the standard SAE/TXC family no foothold (rotation invariance).
This module implements the proposal's HMM-compatible alternative (proposal
lines 1003–1043), which bounds local *which-pair-fired* identification
instead. The setup is a perfect fit for SAE/TXC: sparse, non-negative,
and the ground-truth directions are honest dictionary atoms.

Construction
------------

R ambiguous classes, each with two unit directions

    f_{y,+} = a e_0 + b e_y,  f_{y,-} = a e_0 - b e_y,  with a^2 + b^2 = 1.

By construction f_{y,+} + f_{y,-} = 2 a e_0 for every y. The HMM emits
3-token segments

    cue(y_t)  ->  ambiguous_middle(y_t)  ->  readout(y_t)

with y_t iid Uniform({1, ..., R}). Emissions:

    cue:     x = c_y + sigma * eps          (c_y is a y-specific cue direction)
    middle:  x = 2 a e_0 + sigma * eps      (DETERMINISTIC in y — the ambiguity)
    readout: x = r_y + sigma * eps          (r_y is a y-specific readout direction)

Local impossibility
-------------------

Conditional on x_t at a middle position, P(y | x_t) = P(y) = 1/R, so any
local one-position learner has Bayes-optimal pair-classification accuracy
exactly 1/R. A temporal learner that sees the cue or readout positions can
recover y from the surrounding activations. The proposal frames this as
the easier (and weaker) cousin of the colored-source theorem: it bounds
*which* of two pre-defined feature pairs fired, not *what direction* the
features point in.

This module:

- Generates the data stream.
- Provides labels at every position so a probe can be trained on middle
  positions only.
- Stores the ground-truth basis so a future direction-recovery metric can
  be added on top if needed.

The trained-architecture comparison happens in `run_pair_experiment.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AmbiguousPairConfig:
    """Parameters for the ambiguous-pair HMM dataset.

    Attributes:
        R: Number of ambiguous classes (must satisfy R + 1 <= d).
        d: Ambient observation dimension.
        a: Coefficient on the shared e_0 direction; a^2 + b^2 = 1.
            Default 1/sqrt(2) gives equal weight to the shared and
            class-specific components.
        sigma: Observation noise standard deviation.
        n_seq: Number of independent sequences.
        n_segments_per_seq: Number of [cue, middle, readout] segments per
            sequence; chain length = 3 * n_segments_per_seq.
        seed: RNG seed.
    """

    R: int = 8
    d: int = 64
    a: float = 0.7071067811865476  # 1 / sqrt(2)
    sigma: float = 0.1
    n_seq: int = 256
    n_segments_per_seq: int = 256
    seed: int = 0

    @property
    def b(self) -> float:
        return math.sqrt(max(0.0, 1.0 - self.a * self.a))

    @property
    def T_chain(self) -> int:
        return 3 * self.n_segments_per_seq


def _haar_orthonormal(n: int, d: int, *, generator: torch.Generator) -> torch.Tensor:
    """Sample n orthonormal rows in R^d via QR of an iid Gaussian (n <= d)."""
    if n > d:
        raise ValueError(f"n={n} > d={d}")
    A = torch.randn(d, d, generator=generator, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :n].T.contiguous()  # (n, d)


def generate_ambiguous_pair_dataset(cfg: AmbiguousPairConfig) -> dict:
    """Generate the cue / middle / readout HMM stream.

    Returns a dict with:

        x: (n_seq, 3*n_segments_per_seq, d) observations.
        y: (n_seq, T_chain) integer class label, valid at every position
           — at cue/middle/readout positions of a given segment they all
           share the same y.
        position_kind: (n_seq, T_chain) integer in {0, 1, 2} for
           {cue, middle, readout}.
        e0: (d,) the shared direction in the ambiguous pair.
        cue_dirs: (R, d) the cue directions c_y.
        readout_dirs: (R, d) the readout directions r_y.
        pair_dirs: (R, 2, d) — pair_dirs[y, 0] = f_{y,+}, pair_dirs[y, 1] = f_{y,-}.
        config: the input cfg.
    """
    if cfg.R + 1 > cfg.d:
        raise ValueError(
            f"need d >= R + 1 to fit one e_0 + R class directions; got d={cfg.d}, R={cfg.R}"
        )

    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    # 1 + R + R orthonormal anchor directions: e_0, e_y (used for the pair
    # construction AND as cue), and r_y (readout). We use 1 + 2R rows to
    # keep cue and readout independent.
    anchor_count = 1 + 2 * cfg.R
    if anchor_count > cfg.d:
        # Fall back to less-orthogonal cue/readout: reuse e_y for both.
        anchor = _haar_orthonormal(1 + cfg.R, cfg.d, generator=gen)
        e0 = anchor[0]
        e_class = anchor[1:]            # (R, d)
        cue_dirs = e_class
        readout_dirs = e_class
    else:
        anchor = _haar_orthonormal(anchor_count, cfg.d, generator=gen)
        e0 = anchor[0]
        e_class = anchor[1 : 1 + cfg.R]
        readout_dirs = anchor[1 + cfg.R :]
        cue_dirs = e_class

    # Pair directions f_{y, +/-} = a e_0 +/- b e_y.
    a = cfg.a
    b = cfg.b
    pair_plus = a * e0.unsqueeze(0) + b * e_class               # (R, d)
    pair_minus = a * e0.unsqueeze(0) - b * e_class              # (R, d)
    pair_dirs = torch.stack([pair_plus, pair_minus], dim=1)     # (R, 2, d)

    # Sample y_t iid Uniform([0, R)) per segment per sequence.
    n_seq = cfg.n_seq
    n_seg = cfg.n_segments_per_seq
    y_seg = torch.randint(0, cfg.R, (n_seq, n_seg), generator=gen)

    # Expand y to per-token labels (each segment contributes 3 tokens).
    y_tokens = y_seg.repeat_interleave(3, dim=1)                # (n_seq, T_chain)
    position_kind = (
        torch.tensor([0, 1, 2], dtype=torch.long)
        .repeat(n_seg)
        .unsqueeze(0)
        .expand(n_seq, -1)
        .contiguous()
    )

    T = cfg.T_chain
    d = cfg.d

    # Build the clean signal: per-token activation in observation space.
    # cue:     c_y
    # middle:  2 a e_0
    # readout: r_y
    cue_tokens = cue_dirs[y_tokens]                              # (n_seq, T, d)
    readout_tokens = readout_dirs[y_tokens]
    middle_tok = (2.0 * a) * e0
    middle_tokens = middle_tok.expand(n_seq, T, d)

    is_cue = (position_kind == 0).unsqueeze(-1)
    is_middle = (position_kind == 1).unsqueeze(-1)
    is_readout = (position_kind == 2).unsqueeze(-1)

    x_clean = (
        torch.where(is_cue, cue_tokens, torch.zeros_like(cue_tokens))
        + torch.where(is_middle, middle_tokens, torch.zeros_like(middle_tokens))
        + torch.where(is_readout, readout_tokens, torch.zeros_like(readout_tokens))
    )

    eps = torch.randn(n_seq, T, d, generator=gen, dtype=torch.float64)
    x = x_clean + cfg.sigma * eps

    return {
        "x": x,
        "y": y_tokens,
        "position_kind": position_kind,
        "e0": e0,
        "cue_dirs": cue_dirs,
        "readout_dirs": readout_dirs,
        "pair_dirs": pair_dirs,
        "config": cfg,
    }


def isotropy_diagnostics(data: dict) -> dict:
    """Empirical check of the ambiguity claim: middle activations should be
    invariant in y, and the conditional class distribution given x_middle
    should be uniform.

    Returns a dict with:
        middle_mean: (d,) sample mean of middle-position activations.
                     Expected ~ 2 a e_0.
        middle_per_class_mean: (R, d) per-class sample means at middle.
                               All rows expected ~ middle_mean (no
                               y-information leaks into middle).
        max_class_drift: max ||per_class_mean - global_mean||_2 across y.
                         Should be on the order of sigma / sqrt(N_per_class).
    """
    x = data["x"]
    y = data["y"]
    pos = data["position_kind"]
    cfg: AmbiguousPairConfig = data["config"]

    is_middle = pos == 1
    middle_x = x[is_middle]                                      # (N_mid, d)
    middle_y = y[is_middle]                                      # (N_mid,)

    middle_mean = middle_x.mean(dim=0)
    per_class_mean = torch.stack([
        middle_x[middle_y == c].mean(dim=0) if (middle_y == c).any() else torch.zeros(cfg.d, dtype=middle_x.dtype)
        for c in range(cfg.R)
    ])
    drift = (per_class_mean - middle_mean.unsqueeze(0)).norm(dim=1)
    return {
        "middle_mean": middle_mean,
        "middle_per_class_mean": per_class_mean,
        "max_class_drift": float(drift.max().item()),
        "n_middle_samples": int(is_middle.sum().item()),
        "n_classes": cfg.R,
    }
