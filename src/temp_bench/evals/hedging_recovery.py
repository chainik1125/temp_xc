"""Hedging-drift confidence-state recovery — stage-6 add-on #2.

Like ``lambda_recovery`` / ``changepoint_recovery``, this is NOT a routed
evaluator (the experiment→evaluator map in ``temp_bench/core/runner.py`` is
off-limits). :class:`SyntheticRecovery` calls :func:`hedging_metrics` whenever
the materialised data carries ``extra['conf_labels']`` (the
``toy_hedging_drift*`` datasource). For every other bench the block is a
no-op, so the § 4 metrics are byte-identical and the evaluator's
``protocol_version`` stays 1.3.0.

Headline metric ``conf_recovery``: held-out **R²** of a ridge probe (the
spec's § 4 metric) predicting the continuous confidence state ``c_i`` at the
tile's leading edge from the tile's code — chance = 0 (predicting the pooled
mean), oracle = 1 (the generating ``c_i`` itself). The oracle is NOT reachable
under the emission's per-token folded-normal magnitude (multiplicative noise);
the per-token and per-``T`` window ceilings are quantified in the committed
gating stats, and temporal denoising of that noise is precisely the DC axis
under test. Probe design is the standard per-tile leading-edge convention:
feature dim = one tile's ``d_sae`` code (memorization-free), sequences split
train/eval (leak-free), linear-family probe only.

Also reports ``conf_corr`` (held-out Pearson) and the empirical chance floor
``conf_chance`` (the same probe fit on shuffled train targets, scored as R²).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from temp_bench.evals.lambda_recovery import _tile_lambda_examples
from temp_bench.interfaces.architecture import TempBenchArch


def hedging_metrics(
    model: TempBenchArch, data, *, eval_window_L: int,
    n_windows: int = 1024, seed: int = 0,
) -> dict[str, float]:
    """Return ``{conf_recovery, conf_corr, conf_chance}`` for hedging data.

    ``data.extra['conf_labels']`` is the ``(n_seqs, seq_len)`` confidence
    stream; ``eval_window_L`` is the common tiled evaluation-window length.
    """
    from sklearn.linear_model import Ridge

    from temp_bench.evals.synthetic_recovery import _check_tileable, _sample_windows

    T = _check_tileable(model, eval_window_L)
    model.eval()
    x = data.x
    c = torch.as_tensor(data.extra["conf_labels"]).float()
    c3 = c.reshape(c.shape[0], c.shape[1], 1)               # (N, seq_len, 1)
    n = x.shape[0]
    split = n // 2

    # x and c windows share the seed → identical (seq, offset) → aligned.
    L = eval_window_L
    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_c_tr, _ = _sample_windows(c3[:split], L=L, n_windows=n_windows, seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    win_c_ev, _ = _sample_windows(c3[split:], L=L, n_windows=n_windows, seed=seed + 1)

    z_tr, t_tr = _tile_lambda_examples(model, win_x_tr, win_c_tr, T)
    z_ev, t_ev = _tile_lambda_examples(model, win_x_ev, win_c_ev, T)

    if np.std(t_tr) < 1e-9 or np.std(t_ev) < 1e-9:
        return {"conf_recovery": 0.0, "conf_corr": 0.0, "conf_chance": 0.0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = Ridge(alpha=1.0).fit(z_tr, t_tr)
        pred = reg.predict(z_ev)
        r2 = float(reg.score(z_ev, t_ev))
        corr = (float(np.corrcoef(pred, t_ev)[0, 1])
                if np.std(pred) > 1e-12 else 0.0)
        # Empirical chance floor: same probe on shuffled train targets.
        perm = np.random.default_rng(seed + 7).permutation(len(t_tr))
        reg0 = Ridge(alpha=1.0).fit(z_tr, t_tr[perm])
        floor = float(reg0.score(z_ev, t_ev))

    return {"conf_recovery": r2, "conf_corr": corr, "conf_chance": floor}
