"""Depth-sweep screen runner (depth_sweep_card.md § deviations 3).

screen.py's frozen protocol with (a) LAYERS = the sweep's hs list,
(b) OUT = results/lambda_depth_sweep.json so the frozen
lambda_screen.json is never touched, and ALL cells (hs13 included)
recompute on the locally rebuilt cache — single-generation profile.
The D-K1 gate compares this file's base/hs13 per-token AUC against
the frozen store before any depth claim.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.lambda_intensity.depth_sweep_screen
"""

from __future__ import annotations

from experiments.explorations.task_hunt.lambda_intensity import screen

screen.LAYERS = [7, 10, 13, 16, 19]  # hs = resid_post L{6,9,12,15,18}
screen.OUT = screen.HERE / "results" / "lambda_depth_sweep.json"

if __name__ == "__main__":
    screen.main()
