"""Depth-sweep cache build (depth_sweep_card.md § deviations 1–2).

`cache_depth.main` with the sweep capture list L{6,9,12,15,18} —
odd blocks L9/L15 approved (LOG 121807fb0). Everything else
(batch, seq_len, dtype, stream, output layout) byte-identical.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.lambda_intensity.depth_sweep_build <base|distill>
"""

from __future__ import annotations

import sys

from experiments.explorations.conversion_depth import cache_depth

SWEEP_LAYERS = [6, 9, 12, 15, 18]

cache_depth.LAYERS = SWEEP_LAYERS
cache_depth.HS_CAPTURE = [0] + [k + 1 for k in SWEEP_LAYERS]

if __name__ == "__main__":
    cache_depth.main(sys.argv[1])
