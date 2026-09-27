"""Assumption→consequence discourse-state recovery — stage-6 add-on #1.

Like ``lambda_recovery`` / ``changepoint_recovery``, this is NOT a routed
evaluator (the experiment→evaluator map in ``temp_bench/core/runner.py`` is
off-limits). :class:`SyntheticRecovery` calls :func:`assumption_metrics`
whenever the materialised data carries ``extra['state_labels']`` (the
``toy_assumption_consequence*`` datasource). For every other bench the block is
a no-op, so the § 4 metrics are byte-identical and the evaluator's
``protocol_version`` stays 1.3.0.

Two latent probes, per *tile* at the tile's leading edge (feature dim = one
tile's ``d_sae`` code, never concatenated — memorization-free), split by
sequence (leak-free), all LINEAR (what the code makes *linearly* available):

- ``state_recovery`` (**DC**): multinomial-logistic probe → the discourse
  state ``s_i ∈ {N, A, C}``, balanced accuracy normalized to
  [chance = 1/3, oracle = 1] (the state direction is in the span, dominant).
- ``nextstate_recovery`` (**AC-directed**, the primary axis): multinomial-
  logistic probe → the NEXT state ``s_{i+1}`` from the tile's code — a code
  that supports next-state prediction above the marginal must carry the
  directed grammar (spec § 3). Balanced accuracy normalized to
  [chance = 1/3, oracle = the Bayes-balanced rule of the generating Markov
  conditional, scored on the same eval targets]. NOTE (documented in gating
  BEFORE any grid): the mirror is order-1, so ``s_i`` is a sufficient
  statistic — the per-token INFO ceiling equals the oracle; the architectural
  question is which *trained code* linearly exposes it at the leading edge.
  Tiles whose leading edge is the final sequence position carry the invalid
  sentinel ``-1`` and are masked.

Each probe also reports an empirical chance floor (the same probe fit on
shuffled train targets), per the conventions § 5.
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.evals.changepoint_recovery import (
    _balacc,
    _logistic_probe,
    _tile_label_examples,
)
from temp_bench.interfaces.architecture import TempBenchArch


def bayes_balanced_rule(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Per-current-state next-state prediction maximizing balanced accuracy.

    The balanced-optimal rule weights each class inversely by its frequency:
    predict ``argmax_j P(j | s_i) / pi_j``.
    """
    return np.argmax(np.asarray(P, float) / np.asarray(pi, float)[None, :], axis=1)


def assumption_metrics(
    model: TempBenchArch, data, *, eval_window_L: int,
    n_windows: int = 1024, seed: int = 0,
) -> dict[str, float]:
    """Return the state / next-state recovery metrics for the g7 Markov data.

    ``data.extra`` carries ``state_labels`` and ``next_state_labels`` (both
    ``(n_seqs, seq_len)``; next-state's final column is the ``-1`` sentinel)
    plus the generating ``P`` / ``pi``; ``eval_window_L`` is the common tiled
    evaluation-window length.
    """
    from temp_bench.evals.synthetic_recovery import _check_tileable, _sample_windows

    T = _check_tileable(model, eval_window_L)
    model.eval()
    x = data.x
    labels = torch.stack(
        [torch.as_tensor(data.extra[k]).float()
         for k in ("state_labels", "next_state_labels")],
        dim=-1,
    )                                                       # (N, seq_len, 2)
    n = x.shape[0]
    split = n // 2

    # x and label windows share the seed → identical (seq, offset) → aligned.
    L = eval_window_L
    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_y_tr, _ = _sample_windows(labels[:split], L=L, n_windows=n_windows, seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    win_y_ev, _ = _sample_windows(labels[split:], L=L, n_windows=n_windows, seed=seed + 1)

    z_tr, y_tr = _tile_label_examples(model, win_x_tr, win_y_tr, T)
    z_ev, y_ev = _tile_label_examples(model, win_x_ev, win_y_ev, T)
    st_tr, nx_tr = y_tr[:, 0].astype(np.int64), y_tr[:, 1].astype(np.int64)
    st_ev, nx_ev = y_ev[:, 0].astype(np.int64), y_ev[:, 1].astype(np.int64)

    out: dict[str, float] = {}

    # DC: current discourse state, normalized to [chance = 1/3, oracle = 1].
    n_states = int(data.extra.get("n_states", 3))
    chance = 1.0 / n_states
    st_bal, st_floor = _logistic_probe(z_tr, st_tr, z_ev, st_ev, seed=seed)
    out["state_recovery"] = (st_bal - chance) / (1.0 - chance)
    out["state_balacc"] = st_bal
    out["state_chance"] = st_floor

    # AC-directed: next state, masking the -1 sentinel (leading edge at the
    # final sequence position has no successor inside the sequence).
    m_tr, m_ev = nx_tr >= 0, nx_ev >= 0
    if m_tr.sum() < n_states * 2 or m_ev.sum() < n_states * 2:
        out.update({"nextstate_recovery": 0.0, "nextstate_balacc": chance,
                    "nextstate_chance": chance, "nextstate_oracle_balacc": chance})
        return out
    nx_bal, nx_floor = _logistic_probe(z_tr[m_tr], nx_tr[m_tr],
                                       z_ev[m_ev], nx_ev[m_ev], seed=seed)
    # Empirical, sample-matched oracle: the Bayes-balanced rule of the
    # generating one-step conditional, applied to the TRUE current state on
    # the same eval tiles.
    rule = bayes_balanced_rule(np.asarray(data.extra["P"], float),
                               np.asarray(data.extra["pi"], float))
    oracle_bal = _balacc(nx_ev[m_ev], rule[st_ev[m_ev]])
    out["nextstate_recovery"] = ((nx_bal - chance)
                                 / max(oracle_bal - chance, 1e-9))
    out["nextstate_balacc"] = nx_bal
    out["nextstate_chance"] = nx_floor
    out["nextstate_oracle_balacc"] = oracle_bal
    return out
