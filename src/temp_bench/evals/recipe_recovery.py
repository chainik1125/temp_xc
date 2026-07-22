"""Recipe-instruction phase/equality recovery — stage-6 add-on #3.

Like ``lambda_recovery`` / ``changepoint_recovery`` / ``assumption_recovery``,
this is NOT a routed evaluator (the experiment→evaluator map in
``temp_bench/core/runner.py`` is off-limits). :class:`SyntheticRecovery` calls
:func:`recipe_metrics` whenever the materialised data carries
``extra['equality_labels']`` (the ``toy_recipe_instruction*`` datasource). For
every other bench the block is a no-op, so the § 4 metrics are byte-identical
and the evaluator's ``protocol_version`` stays 1.3.0.

Two latent probes, per *tile* at the tile's leading edge (feature dim = one
tile's ``d_sae`` code, never concatenated — memorization-free), split by
sequence (leak-free), all LINEAR (what the code makes *linearly* available):

- ``phase_recovery`` (**DC control**): multinomial-logistic probe → the phase
  class ``c_t ∈ {0..4}``, balanced accuracy normalized to [chance = 1/5,
  oracle = 1]. Balanced accuracy is the balancing anchor (spec § 3 caveat:
  classes 2–4 sit at 6–7% marginal) — the same treatment the assumption bench
  gave its 6.6%-marginal A state. The phase-signature direction is in the
  span and dominant, so per-token archs should sit near oracle — this is the
  control, not the claim.
- ``equality_residual_recovery`` (**PRIMARY — the re-scoped regime-3
  residual axis**, stage-6 #3b, 2026-07-23 review decision): the same
  logistic probe's balanced accuracy re-normalized to
  [pair-additive ceiling = 0.771, exact = 1.0] — the share of ``e_t``
  reachable ONLY by position-mixing. The floor constant is the
  § 8-measured ceiling for ANY additive readout on this substrate
  (``recipe_instruction_phase_runs/results/recipe_gating_stats.json``,
  ``analytic.e_balacc_pair_additive``, committed in ``b463c4a0``) — never
  re-derived here. NOT clipped: negative values are informative (the code's
  linear ``e_t`` readout fell below even the additive ceiling).
- ``equality_recovery`` (diagnostic): the original chance-normalized form,
  balanced accuracy over [chance = 0.5, oracle = 1] — kept as a diagnostic
  key after the § 8 STOP showed the substrate's class-conditional
  continuation rates leak ``e_t`` into raw-linear readouts (per-token 0.614
  / window 0.720 — the gating record quantifies the leak), so distance
  from 0.5 conflates additive access with the regime-3 residual.
  ``equality_base_rate`` (the pooled match rate, the spec's raw-accuracy
  chance quantity) is reported alongside for reference.

Each probe also reports an empirical chance floor (the same probe fit on
shuffled train targets), per the conventions § 5.
"""

from __future__ import annotations

import numpy as np
import torch

from temp_bench.evals.changepoint_recovery import (
    _logistic_probe,
    _tile_label_examples,
)
from temp_bench.interfaces.architecture import TempBenchArch

# The § 8-measured pair-additive access ceiling for e_t on this substrate
# (recipe_instruction_phase_runs/results/recipe_gating_stats.json,
# analytic.e_balacc_pair_additive, commit b463c4a0). The re-scoped primary
# axis (stage-6 #3b, mac-local review 2026-07-23) scores the residual above
# this floor — the share only position-mixing can reach.
EQ_ADDITIVE_CEILING = 0.771


def recipe_metrics(
    model: TempBenchArch, data, *, eval_window_L: int,
    n_windows: int = 1024, seed: int = 0,
) -> dict[str, float]:
    """Return the phase / equality recovery metrics for hier_categorical data.

    ``data.extra`` carries ``phase_class_labels`` (categorical) and
    ``equality_labels`` (binary, ``e_0 = 0``), each ``(n_seqs, seq_len)``;
    ``eval_window_L`` is the common tiled evaluation-window length.
    """
    from temp_bench.evals.synthetic_recovery import _check_tileable, _sample_windows

    T = _check_tileable(model, eval_window_L)
    model.eval()
    x = data.x
    labels = torch.stack(
        [torch.as_tensor(data.extra[k]).float()
         for k in ("phase_class_labels", "equality_labels")],
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
    ph_tr, eq_tr = y_tr[:, 0].astype(np.int64), y_tr[:, 1].astype(np.int64)
    ph_ev, eq_ev = y_ev[:, 0].astype(np.int64), y_ev[:, 1].astype(np.int64)

    out: dict[str, float] = {}

    # DC control: phase class, normalized to [chance = 1/n_phases, 1].
    n_phases = int(data.extra.get("n_phases", 5))
    chance = 1.0 / n_phases
    ph_bal, ph_floor = _logistic_probe(z_tr, ph_tr, z_ev, ph_ev, seed=seed)
    out["phase_recovery"] = (ph_bal - chance) / (1.0 - chance)
    out["phase_balacc"] = ph_bal
    out["phase_chance"] = ph_floor

    # Equality-adjacency flag: the re-scoped PRIMARY is the regime-3
    # residual over [additive ceiling, exact]; the chance-normalized form
    # stays as a diagnostic (see module docstring for the § 8 provenance of
    # the 0.771 constant — measured, frozen, never re-derived here).
    eq_bal, eq_floor = _logistic_probe(z_tr, eq_tr, z_ev, eq_ev, seed=seed)
    out["equality_residual_recovery"] = ((eq_bal - EQ_ADDITIVE_CEILING)
                                         / (1.0 - EQ_ADDITIVE_CEILING))
    out["equality_recovery"] = (eq_bal - 0.5) / 0.5
    out["equality_balacc"] = eq_bal
    out["equality_chance"] = eq_floor
    out["equality_base_rate"] = float(eq_ev.mean())
    return out
