"""AC-only signed-motion metrics — FrequencyBench § 5 add-on to § 4.

This module is NOT a routed evaluator (the experiment→evaluator map lives
in ``temp_bench/core/runner.py``, which the framework's hard rules forbid
editing). Instead :class:`SyntheticRecovery` calls
:func:`signed_motion_metrics` whenever the materialised synthetic data
carries sign labels (the ``toy_signed_motion_*`` datasource). For every
other synthetic bench this code never runs and the § 4 metrics are
byte-identical, so the evaluator's ``protocol_version`` does not move.

Two metrics:

- ``s_temp`` (headline): normalised sign-recovery score
  ``2 · (probe_acc − 0.5)`` — 0 = chance, 1 = oracle. The probe is a
  LINEAR logistic regression on the arch's codes for a fixed window. The
  linearity is load-bearing: the sign is the interaction term
  ``Q_{t+1} − Q_t = S·v``, which a per-token encoder only exposes as an
  additive ``Σ_t h_t(Q_t)`` score across positions — and additive scores
  cannot separate +v from −v orbits (equal per-phase totals). A window
  encoder that learns a zero-mean filter exposes the step directly, so
  the sign becomes linearly decodable. Hence the prediction: SAE-family
  archs ≈ chance, window crosscoders > chance.

- ``atom_dc_fraction``: for archs whose decoder is a genuine
  ``(d_sae, T, d_in)`` window tensor (TXC-base), the mean fraction of
  per-atom energy in the time-constant (DC) component,
  ``T·||mean_t a_t||² / Σ_t ||a_t||²`` ∈ [0, 1]. ≪ 1 ⇒ the atom is a
  zero-mean (AC) filter — the mechanism that lets a window encoder read
  the step. ``None`` for token/per-position archs (no shared window
  decoder, so there is no aligned ``a_t`` across positions).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from temp_bench.interfaces.architecture import TempBenchArch


@torch.no_grad()
def _tile_examples(
    model: TempBenchArch, windows_L: torch.Tensor, T: int, win_signs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Tile length-L windows and treat each tile-code as a separate example.

    ``windows_L: (W, L, d_in)`` → ``(W·(L/T), per_tile_code_dim)`` codes with a
    matching ``(W·(L/T),)`` sign label (each tile inherits its window's sign).

    Crucially the feature dimension is the *single-tile* code size (``d_sae``
    for a window crosscoder), NOT the concatenation over tiles. With only
    ``2M`` distinct windows in the data, a probe that concatenated all tiles
    would have ≥ #distinct-windows features and could *memorize* them (and
    "generalize" because train/eval share the same window set). Probing one
    native window-code at a time keeps features `< 2M` in the scarce regime,
    so any separation is genuine. For a per-token arch (T=1) a tile is one
    token — whose code carries zero sign information by the DPI — the cleanest
    possible negative control.
    """
    device = next(model.parameters()).device
    W, L, d_in = windows_L.shape
    n_tiles = L // T
    tiles = windows_L.to(device, dtype=torch.float32).reshape(W * n_tiles, T, d_in)
    z = model.encode(tiles).reshape(W * n_tiles, -1)
    y = np.repeat(win_signs, n_tiles)
    return z.detach().float().cpu().numpy(), y


def _train_sign_probe(
    model: TempBenchArch,
    x: torch.Tensor,
    sign_labels: torch.Tensor,
    *,
    L: int,
    n_windows: int = 1024,
    seed: int = 0,
) -> dict[str, float]:
    """Linear logistic-regression sign probe (C=1.0) on single-tile codes.

    Sequences are split into disjoint train / eval pools; length-L windows are
    sampled from each pool and tiled into the arch's native T (§ 4 of the
    guidance). Each tile-code is one probe example, so the score measures
    whether the sign is linearly decodable from a single native window-code —
    leak-free (split by sequence) and memorization-free (features = one tile's
    code, `< 2M` in the scarce regime). Returns ``{probe_acc, s_temp}``.
    """
    from sklearn.linear_model import LogisticRegression

    from temp_bench.evals.synthetic_recovery import _check_tileable, _sample_windows

    T = _check_tileable(model, L)
    model.eval()
    n = x.shape[0]
    split = n // 2
    y = sign_labels.detach().cpu().numpy().astype(np.int64)

    win_tr, idx_tr = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_ev, idx_ev = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    z_tr, y_tr = _tile_examples(model, win_tr, T, y[:split][idx_tr])
    z_ev, y_ev = _tile_examples(model, win_ev, T, y[split:][idx_ev])

    # Guard the degenerate single-class split.
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_ev)) < 2:
        return {"probe_acc": 0.5, "s_temp": 0.0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # silence LR convergence chatter
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(z_tr, y_tr)
        acc = float(clf.score(z_ev, y_ev))

    return {"probe_acc": acc, "s_temp": float(2.0 * (acc - 0.5))}


def _atom_dc_fraction(model: TempBenchArch) -> float | None:
    """Mean DC-energy fraction over decoder atoms, or ``None``.

    Only defined when the decoder is a shared window tensor
    ``W_dec ∈ R^{d_sae × T × d_in}`` with ``T > 1`` (TXC-base). Per-token /
    per-position archs have no aligned ``a_t`` across positions, so the
    metric is undefined for them.
    """
    W = getattr(model, "W_dec", None)
    if not isinstance(W, torch.Tensor) or W.dim() != 3:
        return None
    d_sae, T, _ = W.shape
    if T <= 1:
        return None
    a = W.detach().float().cpu()                            # (d_sae, T, d_in)
    dc = a.mean(dim=1)                                      # (d_sae, d_in)
    dc_energy = T * dc.pow(2).sum(dim=1)                    # (d_sae,)
    total_energy = a.pow(2).sum(dim=(1, 2)).clamp(min=1e-12)
    frac = (dc_energy / total_energy)                      # (d_sae,)
    return float(frac.mean())


def signed_motion_metrics(
    model: TempBenchArch, data, *, eval_window_L: int
) -> dict[str, float]:
    """Return ``{s_temp, sign_probe_acc[, atom_dc_fraction]}`` for AC data.

    ``data`` is a :class:`temp_bench.data.synthetic.SyntheticData` whose
    ``extra['sign_labels']`` holds the hidden ±1 sign per sequence;
    ``eval_window_L`` is the common (tiled) evaluation-window length.
    """
    sign_labels = data.extra["sign_labels"]
    probe = _train_sign_probe(model, data.x, sign_labels, L=eval_window_L)
    out: dict[str, float] = {
        "s_temp": probe["s_temp"],
        "sign_probe_acc": probe["probe_acc"],
    }
    dc = _atom_dc_fraction(model)
    if dc is not None:
        out["atom_dc_fraction"] = dc
    return out
