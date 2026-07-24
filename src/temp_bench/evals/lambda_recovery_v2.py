"""λ-recovery v2 — probe-capacity knobs for the λ readout (contingency build).

This is the CONTINGENCY implementation for the λ-readout methods decision
(`briefings/probe-adequacy.md`): runpod-d and runpod-e report — independently,
**under review, not yet adopted** — that the v1 probe (`lambda_recovery.py`,
unregularized OLS on ``p = d_sae`` features with ``n = n_windows·(L/T)`` rows)
sits at ``n ≈ p`` for dense codes at large ``T`` (their 2026-07-24 LOG
entries). Whether the canonical readout changes is mac-local's decision; this
module makes that decision *executable*. Frozen convention + adoption
mechanics: `experiments/explorations/task_hunt/lambda_intensity/PROBE_V2_SPEC.md`.

**v1 is frozen.** `lambda_recovery.py` is never edited and keeps producing
bit-identical numbers; this module *imports* v1's window sampler and tile
readout, so the readout convention (per-tile code, λ at the tile's leading
edge, per-token archs at single positions, shuffled-train-target chance
floor, fixed sampling seed 0) is identical **by construction**. Only the
probe-capacity knobs change, each explicit in ``eval_cfg``:

- ``lambda_probe_v2`` (bool)  — the opt-in dispatch flag read by
  :class:`~temp_bench.evals.synthetic_recovery.SyntheticRecovery`. Absent →
  this module never runs and every existing row is byte-identical.
- ``lambda_v2_probe``      — ``"ridge"`` (default) | ``"ols"``. Ridge selects
  its penalty by inner validation INSIDE the train half only
  (`sklearn.linear_model.RidgeCV`, efficient leave-one-out; deterministic,
  never touches the eval half). ``"ols"`` is the exact α → 0 limit: the same
  `LinearRegression` fit as v1 (contract test (a) reproduces v1 with it).
- ``lambda_v2_alphas``     — the frozen α grid, default
  ``np.logspace(-2, 4, 13)`` (the grid of runpod-d's committed diagnostic;
  runpod-e's ``logspace(-1, 4, 12)`` is an interior subset).
- ``lambda_v2_n_windows``  — probe windows per half, default **8192**: at the
  largest panel tile (``T = 16``, ``L = 32``) that yields
  ``n_rows = 8192·(32/16) = 16384 = 8·p`` at the panel anchor
  ``p = d_sae = 2048`` — the briefing's ``n_rows ≥ 8·p`` adequacy line, and
  the setting both diagnostics used.
- ``lambda_v2_split``      — ``"trace"`` (default) | ``"half"``. ``"half"``
  is v1's raw ``n // 2`` sequence split. ``"trace"`` advances the split to
  the next TRACE boundary at-or-after ``n // 2``, where the sequence → trace
  map is ``data.extra["trace_ids"]`` and a datasource that declares none
  means every sequence is its own trace — so on synthetic benches ``"trace"``
  degenerates to exactly v1's split, and on the Ward panels it stops the one
  boundary trace from straddling the halves
  (`lambda_intensity/results/split_forensics.json`: committed panel numbers
  leak ZERO eval draws at nw = 1024; the half-split leaks 2/8192 at the v2
  default nw, the snap leaks none).

Emits only ``*_v2``-suffixed keys (plus fit-size receipts), so v1 columns are
never shadowed — a v2 row carries its paired v1 readout on the same windows.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from temp_bench.interfaces.architecture import TempBenchArch

# Frozen defaults — mirrored in PROBE_V2_SPEC.md; panel re-runs set every
# knob explicitly in eval_cfg so the values are pinned into eval_key.
DEFAULT_PROBE = "ridge"
DEFAULT_ALPHAS = tuple(float(a) for a in np.logspace(-2, 4, 13))
DEFAULT_N_WINDOWS = 8192
DEFAULT_SPLIT = "trace"

_ZERO = {"lambda_recovery_v2": 0.0, "lambda_r2_v2": 0.0,
         "lambda_chance_v2": 0.0, "lambda_alpha_v2": 0.0,
         "lambda_v2_n_train_rows": 0.0, "lambda_v2_n_eval_rows": 0.0}


def _split_index(n: int, mode: str, trace_ids) -> int:
    """The sequence index where the train half ends.

    ``"half"``  → v1's raw ``n // 2``.
    ``"trace"`` → the next trace boundary at-or-after ``n // 2``. With no
    ``trace_ids`` every sequence is its own trace (every index is a
    boundary), so this equals ``n // 2`` — v1-identical for synthetic
    benches by construction.
    """
    if mode == "half" or trace_ids is None:
        return n // 2
    t = np.asarray(trace_ids).reshape(-1)
    if len(t) != n:
        raise ValueError(
            f"trace_ids has {len(t)} entries for {n} sequences")
    if not np.all(np.diff(t) >= 0):
        raise ValueError(
            "trace_ids is not non-decreasing — the boundary-snap trace "
            "split assumes a trace-contiguous stream (true of the Ward "
            "caches; see split_forensics.json). Refusing to guess.")
    s = n // 2
    while s < n and t[s] == t[s - 1]:
        s += 1
    return s


def _fit_probe(z_tr, t_tr, z_ev, probe: str, alphas):
    """Fit train-half probe, predict eval half. Returns (pred, reg)."""
    from sklearn.linear_model import LinearRegression, RidgeCV

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if probe == "ols":
            reg = LinearRegression().fit(z_tr, t_tr)
        elif probe == "ridge":
            reg = RidgeCV(alphas=np.asarray(alphas, dtype=float)
                          ).fit(z_tr, t_tr)
        else:
            raise ValueError(f"unknown lambda_v2_probe {probe!r}")
        pred = reg.predict(z_ev)
    return pred, reg


def _train_lambda_probe_v2(
    model: TempBenchArch,
    x: torch.Tensor,
    lam: torch.Tensor,
    *,
    L: int,
    n_windows: int,
    probe: str,
    alphas,
    split_mode: str,
    trace_ids,
    seed: int = 0,
) -> dict[str, float]:
    """The v1 probe pipeline with capacity knobs; readout code is v1's own."""
    from temp_bench.evals.lambda_recovery import _tile_lambda_examples
    from temp_bench.evals.synthetic_recovery import (
        _check_tileable,
        _sample_windows,
    )

    T = _check_tileable(model, L)
    model.eval()
    n = x.shape[0]
    split = _split_index(n, split_mode, trace_ids)
    if split == 0 or split >= n:
        # A trace spanning the midpoint to the end would empty a pool
        # (impossible on the Ward panels — 300 traces of ≤ 15 windows).
        return dict(_ZERO)
    lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)

    # Identical to v1: x and λ windows share the seed → position-aligned;
    # train pool seed, eval pool seed + 1.
    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_l_tr, _ = _sample_windows(lam3[:split], L=L, n_windows=n_windows, seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    win_l_ev, _ = _sample_windows(lam3[split:], L=L, n_windows=n_windows, seed=seed + 1)

    z_tr, t_tr = _tile_lambda_examples(model, win_x_tr, win_l_tr, T)
    z_ev, t_ev = _tile_lambda_examples(model, win_x_ev, win_l_ev, T)

    # v1's NaN-target guard, verbatim semantics.
    tr_m, ev_m = np.isfinite(t_tr), np.isfinite(t_ev)
    if not tr_m.all():
        z_tr, t_tr = z_tr[tr_m], t_tr[tr_m]
    if not ev_m.all():
        z_ev, t_ev = z_ev[ev_m], t_ev[ev_m]
    if len(t_tr) < 2 or len(t_ev) < 2:
        return dict(_ZERO)
    if np.std(t_tr) < 1e-9 or np.std(t_ev) < 1e-9:
        return dict(_ZERO)

    pred, reg = _fit_probe(z_tr, t_tr, z_ev, probe, alphas)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = float(reg.score(z_ev, t_ev))
    corr = float(np.corrcoef(pred, t_ev)[0, 1]) if np.std(pred) > 1e-12 else 0.0

    # Chance floor: the FULL probe procedure (α re-selected for ridge) on
    # shuffled train targets — v1's permutation seed.
    rngp = np.random.default_rng(seed + 7)
    perm = rngp.permutation(len(t_tr))
    pred0, _ = _fit_probe(z_tr, t_tr[perm], z_ev, probe, alphas)
    chance = float(np.corrcoef(pred0, t_ev)[0, 1]) if np.std(pred0) > 1e-12 else 0.0

    return {
        "lambda_recovery_v2": corr,
        "lambda_r2_v2": r2,
        "lambda_chance_v2": chance,
        # Never NaN (NaN → JSON null breaks the cached-row read).
        "lambda_alpha_v2": float(getattr(reg, "alpha_", 0.0)),
        "lambda_v2_n_train_rows": float(len(t_tr)),
        "lambda_v2_n_eval_rows": float(len(t_ev)),
    }


def lambda_recovery_v2_metrics(
    model: TempBenchArch, data, *, eval_window_L: int, eval_cfg: dict
) -> dict[str, float]:
    """Return the ``*_v2`` λ-recovery metrics per the eval_cfg knobs.

    Reads ``data.extra['lambda_labels']`` exactly like v1 and
    ``data.extra['trace_ids']`` (optional) for the trace split.
    """
    lam = data.extra["lambda_labels"]
    if not torch.is_tensor(lam):
        lam = torch.as_tensor(lam)
    return _train_lambda_probe_v2(
        model, data.x, lam.float(),
        L=eval_window_L,
        n_windows=int(eval_cfg.get("lambda_v2_n_windows", DEFAULT_N_WINDOWS)),
        probe=str(eval_cfg.get("lambda_v2_probe", DEFAULT_PROBE)),
        alphas=eval_cfg.get("lambda_v2_alphas", DEFAULT_ALPHAS),
        split_mode=str(eval_cfg.get("lambda_v2_split", DEFAULT_SPLIT)),
        trace_ids=data.extra.get("trace_ids"),
    )
