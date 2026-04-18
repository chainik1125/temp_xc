"""Plateau detection for training loops.

Usage inside an ArchSpec.train loop:

    from src.bench.plateau import should_stop_plateau

    for step in range(total_steps):
        # ... training step ...
        if step % log_every == 0:
            log["loss"].append(loss.item())
            if should_stop_plateau(log["loss"], step=step,
                                   threshold_pct=0.01, min_steps=5000):
                break

Default threshold_pct=0.01 → stop when loss has dropped <1% over the
last ~2×log_every-step window (tracks the mean-of-last-N-vs-mean-of-
previous-N heuristic used in the sprint analysis). min_steps prevents
early exits from transient plateaus at the start of training.
"""

from __future__ import annotations


def should_stop_plateau(
    loss_history: list[float],
    step: int,
    threshold_pct: float | None,
    min_steps: int = 5000,
    window_size: int = 4,
) -> bool:
    """Return True if training has plateaued.

    `loss_history` is the list of logged loss values; we compare the
    mean of the last `window_size` entries against the mean of the
    preceding `window_size` entries. If the fractional drop is below
    threshold_pct (e.g. 0.01 = 1%), we call it plateaued.

    Returns False if:
      - threshold_pct is None (plateau check disabled)
      - step < min_steps (floor to prevent early exits)
      - fewer than 2*window_size log entries collected
    """
    if threshold_pct is None:
        return False
    if step < min_steps:
        return False
    if len(loss_history) < 2 * window_size:
        return False
    recent = loss_history[-window_size:]
    prior = loss_history[-2 * window_size : -window_size]
    recent_mean = sum(recent) / len(recent)
    prior_mean = sum(prior) / len(prior)
    if prior_mean <= 0:
        return False
    frac_drop = (prior_mean - recent_mean) / prior_mean
    return frac_drop < threshold_pct
