"""Self-exciting intensity (λ) recovery — autoresearch #1 backtracking add-on.

Like ``signed_motion_recovery``, this is NOT a routed evaluator (the
experiment→evaluator map in ``temp_bench/core/runner.py`` is off-limits).
:class:`SyntheticRecovery` calls :func:`lambda_recovery_metrics` whenever the
materialised data carries ``extra['lambda_labels']`` (the
``toy_backtracking_selfexcite*`` datasource). For every other bench the block is
a no-op, so the § 4 metrics are byte-identical and the evaluator's
``protocol_version`` does not move.

Headline metric ``lambda_recovery``: the held-out Pearson correlation between a
**linear** probe's prediction and the true conditional intensity ``λ`` — chance
= 0, oracle = 1. The probe is fit per *tile*, predicting ``λ`` at the tile's
**leading edge** (last position), whose ``K``-step event history lives inside
the tile (for ``T ≥ K+1``). The linearity + per-tile design are load-bearing:

- **Per-token (T=1)** tiles are single tokens, whose code is a function of
  ``b_i`` alone; by the data-processing inequality the best linear (indeed best
  *any*) readout of ``λ_i`` is ``corr = √(Var λ/Var b) ≈ 0.41`` — a *provable*
  floor, independent of ``d_sae`` or probe. This is the negative control.
- **Window (T≥2)** tiles expose the recent event history, so a window code that
  linearly carries it lets the probe recover ``λ`` (ceiling ≈ 0.91 at ``T=2``,
  ≈ 0.99 at ``T≥4`` for ``K=2``). The per-token→window gap is the headline.
- **Feature dimension = one tile's code (``d_sae``)**, never the concatenation
  over tiles — so with ``d_sae`` in the scarce regime the probe cannot memorise
  (the signed-motion lesson). λ being *linear in the history* (logit-linear)
  means a window's recovery reflects genuine linear exposure, not a lookup.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from temp_bench.interfaces.architecture import TempBenchArch


@torch.no_grad()
def _tile_lambda_examples(
    model: TempBenchArch, win_x: torch.Tensor, win_l: torch.Tensor, T: int
) -> tuple[np.ndarray, np.ndarray]:
    """Tile length-L windows; each tile-code → λ at the tile's LAST position.

    ``win_x: (W, L, d_in)`` → ``(W·(L/T), tile_code_dim)`` codes; ``win_l:
    (W, L, 1)`` → ``(W·(L/T),)`` targets, taking ``λ`` at each tile's leading
    edge (position ``T-1`` within the tile), whose event history is inside the
    tile. The feature dim is the *single-tile* code (``d_sae`` for a window
    crosscoder), not the concatenation over tiles.
    """
    device = next(model.parameters()).device
    W, L, d_in = win_x.shape
    n_tiles = L // T
    tiles = win_x.to(device, dtype=torch.float32).reshape(W * n_tiles, T, d_in)
    z = model.encode(tiles).reshape(W * n_tiles, -1).detach().float().cpu().numpy()
    lam = win_l.reshape(W, n_tiles, T)[:, :, T - 1].reshape(-1)
    return z, lam.detach().float().cpu().numpy()


def _train_lambda_probe(
    model: TempBenchArch,
    x: torch.Tensor,
    lam: torch.Tensor,
    *,
    L: int,
    n_windows: int = 1024,
    seed: int = 0,
) -> dict[str, float]:
    """Linear-regression λ probe on single-tile codes (leak-free, mem-free).

    Sequences split into disjoint train/eval pools; length-L windows sampled
    from each and tiled. ``x`` and ``λ`` windows are sampled with the *same*
    seed so they are position-aligned. Returns ``{lambda_recovery, lambda_r2,
    lambda_chance}``.
    """
    from sklearn.linear_model import LinearRegression

    from temp_bench.evals.synthetic_recovery import _check_tileable, _sample_windows

    T = _check_tileable(model, L)
    model.eval()
    n = x.shape[0]
    split = n // 2
    lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)            # (N, seq_len, 1)

    # x and λ windows share the seed → identical (seq, offset) → position-aligned.
    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_l_tr, _ = _sample_windows(lam3[:split], L=L, n_windows=n_windows, seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    win_l_ev, _ = _sample_windows(lam3[split:], L=L, n_windows=n_windows, seed=seed + 1)

    z_tr, t_tr = _tile_lambda_examples(model, win_x_tr, win_l_tr, T)
    z_ev, t_ev = _tile_lambda_examples(model, win_x_ev, win_l_ev, T)

    if np.std(t_tr) < 1e-9 or np.std(t_ev) < 1e-9:
        return {"lambda_recovery": 0.0, "lambda_r2": 0.0, "lambda_chance": 0.0}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = LinearRegression().fit(z_tr, t_tr)
        pred = reg.predict(z_ev)
        r2 = float(reg.score(z_ev, t_ev))
    corr = float(np.corrcoef(pred, t_ev)[0, 1]) if np.std(pred) > 1e-12 else 0.0

    # Empirical chance floor: same probe on shuffled train targets.
    rngp = np.random.default_rng(seed + 7)
    perm = rngp.permutation(len(t_tr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg0 = LinearRegression().fit(z_tr, t_tr[perm])
        pred0 = reg0.predict(z_ev)
    chance = float(np.corrcoef(pred0, t_ev)[0, 1]) if np.std(pred0) > 1e-12 else 0.0

    return {"lambda_recovery": corr, "lambda_r2": r2, "lambda_chance": chance}


def lambda_recovery_metrics(
    model: TempBenchArch, data, *, eval_window_L: int
) -> dict[str, float]:
    """Return ``{lambda_recovery, lambda_r2, lambda_chance}`` for self-exciting data.

    ``data.extra['lambda_labels']`` is the ``(n_seqs, seq_len)`` hidden intensity;
    ``eval_window_L`` is the common tiled evaluation-window length.
    """
    lam = data.extra["lambda_labels"]
    if not torch.is_tensor(lam):
        lam = torch.as_tensor(lam)
    return _train_lambda_probe(model, data.x, lam.float(), L=eval_window_L)
