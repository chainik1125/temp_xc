"""Position-only floor for the hedging-LEVEL Stage-2 panel.

Pre-registered in `card_stage2.md` § 10.2 (adopted from the runpod-b
draft's ambient-ramp guard); OFF-leaderboard. Fits LinearRegression from
leading-edge position features alone — [p, p²] on the 128-token cache
grid — to the frozen slope8 target, under the evaluator's EXACT sampling
(same split at 2022, same seeds 0/1, same finite-target mask), per panel
T. If this floor is not low, the ambient position ramp explains part of
every arch's recovery and the panel reading must discount it.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.confidence.position_floor
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels" / "confidence.npz"
EVAL_L = 32
N_WINDOWS = 1024
TS = (1, 2, 4, 8, 16)


def _rows(grid, seed, T):
    """Replicate `synthetic_recovery._sample_windows` + the leading-edge
    tiling of `lambda_recovery._tile_lambda_examples` for one pool."""
    rng = np.random.default_rng(seed)
    n_total, seq_len = grid.shape
    seq_idx = rng.integers(0, n_total, size=N_WINDOWS)
    offsets = rng.integers(0, seq_len - EVAL_L + 1, size=N_WINDOWS)
    lead = offsets[:, None] + (np.arange(EVAL_L // T) + 1) * T - 1  # (W, n_tiles)
    t = grid[seq_idx[:, None], lead].ravel()
    p = lead.ravel().astype(np.float64)
    m = np.isfinite(t)
    X = np.stack([p[m], p[m] ** 2], axis=1)
    return X, t[m]


def main() -> None:
    z = np.load(LABELS)
    lam = np.where(np.isfinite(z["slope8"]) & z["valid"], z["slope8"],
                   np.nan).astype(np.float64)
    split = lam.shape[0] // 2
    out = {"meta": {"features": "[p, p^2] leading-edge position only",
                    "card": "card_stage2.md §10.2", "off_leaderboard": True}}
    for T in TS:
        Xtr, ttr = _rows(lam[:split], 0, T)
        Xev, tev = _rows(lam[split:], 1, T)
        reg = LinearRegression().fit(Xtr, ttr)
        pred = reg.predict(Xev)
        r = float(np.corrcoef(pred, tev)[0, 1]) if np.std(pred) > 1e-12 else 0.0
        out[f"T{T}"] = {"pos_floor_r": r, "n_train": int(len(ttr)),
                        "n_eval": int(len(tev))}
        print(f"T={T:2d}  pos_floor_r={r:+.4f}  n={len(ttr)}/{len(tev)}")
    dst = HERE / "results" / "stage2_position_floor.json"
    dst.write_text(json.dumps(out, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
