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

# Token archs (T=1) get the full window of context so the chance result is
# NOT an artefact of only seeing one token. Window archs use their native T.
COMMON_PROBE_WINDOW = 5


def _probe_window(model: TempBenchArch) -> int:
    """How many leading tokens the sign probe reads for this arch.

    Window/sequence archs whose encoder consumes exactly ``T`` tokens are
    fed their native window (TXC-base requires this). Token archs (T=1) are
    handed the common multi-token window so the linear reader gets the same
    temporal extent everyone else does.
    """
    arch_T = int(getattr(model.config, "T", 1) or 1)
    return arch_T if arch_T > 1 else COMMON_PROBE_WINDOW


@torch.no_grad()
def _window_codes(model: TempBenchArch, x_window: torch.Tensor) -> np.ndarray:
    """Encode a ``(B, T_probe, d_in)`` window → flat ``(B, n_feats)`` codes.

    TXC-base (``consumes='window'``) returns one window-level latent
    ``(B, 1, d_sae)``; per-token / per-position archs return ``(B, T, d_sae)``.
    Either way we flatten everything past the batch dim — the probe reads
    whatever the encoder makes available over the window.
    """
    device = next(model.parameters()).device
    x_window = x_window.to(device, dtype=torch.float32)
    z = model.encode(x_window)
    z = z.reshape(z.shape[0], -1)
    return z.detach().float().cpu().numpy()


def _train_sign_probe(
    model: TempBenchArch,
    x: torch.Tensor,
    sign_labels: torch.Tensor,
) -> dict[str, float]:
    """Linear logistic-regression sign probe (C=1.0).

    Splits sequences in half (train / held-out eval) so the reported
    accuracy is on sequences with phases the probe never saw. Returns
    ``{probe_acc, s_temp}``.
    """
    from sklearn.linear_model import LogisticRegression

    T_probe = _probe_window(model)
    if x.shape[1] < T_probe:
        raise ValueError(
            f"signed-motion probe needs seq_len >= {T_probe}; got {x.shape[1]}."
        )

    model.eval()
    n = x.shape[0]
    split = n // 2
    window = x[:, :T_probe, :]
    z = _window_codes(model, window)                       # (n, n_feats)
    y = sign_labels.detach().cpu().numpy().astype(np.int64)

    z_tr, z_ev = z[:split], z[split:]
    y_tr, y_ev = y[:split], y[split:]

    # Guard the degenerate single-class split (shouldn't happen with a
    # balanced ±1 draw, but keep the sweep robust).
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


def signed_motion_metrics(model: TempBenchArch, data) -> dict[str, float]:
    """Return ``{s_temp, sign_probe_acc[, atom_dc_fraction]}`` for AC data.

    ``data`` is a :class:`temp_bench.data.synthetic.SyntheticData` whose
    ``extra['sign_labels']`` holds the hidden ±1 sign per sequence.
    """
    sign_labels = data.extra["sign_labels"]
    probe = _train_sign_probe(model, data.x, sign_labels)
    out: dict[str, float] = {
        "s_temp": probe["s_temp"],
        "sign_probe_acc": probe["probe_acc"],
    }
    dc = _atom_dc_fraction(model)
    if dc is not None:
        out["atom_dc_fraction"] = dc
    return out
