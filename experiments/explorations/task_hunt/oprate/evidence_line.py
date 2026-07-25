"""Evidence-line REGRESSION analog for the oprate Stage-2 panel (label-side).

`CARD_STAGE2.md` § 3: screen-side visible-evidence AUCs do not
transplant to Stage 2, so the binding comparator is computed here, at
panel time, under the panel eval's own conventions — same
`_sample_windows` calls (seeds 0 / 1), same first-half/second-half
sequence split, same L = 32 tiling, same leading-edge target, same
non-finite drop rule as `temp_bench.evals.lambda_recovery`. Features =
the in-tile count of case-class tokens (`op == 2`, the bundle's class
map) — the "counting visible event sentences" reading, given the same
probe class (OLS) the v1 eval uses. A window cell that does not beat
this r at matched T earns no latent-state language.

Also emits the card § 2.7 duty: how many sampled tiles drop to the
non-finite leading-edge guard, per T (train and eval pools).

Label-side only — no model, no leaderboard row. Output:
`results/evidence_line_case.json`.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.oprate.evidence_line [target]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from temp_bench.evals.synthetic_recovery import _sample_windows

HERE = Path(__file__).resolve().parent
NPZ = HERE.parent / "labels" / "oprate.npz"
CLASS_ID = {"case": 2, "ver": 3}
L = 32
TS = (1, 2, 4, 8, 16)
N_WINDOWS = 1024          # the v1 eval default — the convention under test


def analog_for_T(count_grid, rate_grid, T: int) -> dict:
    """OLS r from in-tile event count → leading-edge target, v1 conventions."""
    from sklearn.linear_model import LinearRegression

    n = count_grid.shape[0]
    split = n // 2
    cnt3 = torch.from_numpy(count_grid.astype(np.float32))[..., None]
    lam3 = torch.from_numpy(rate_grid.astype(np.float32))[..., None]

    rows = {}
    for pool, xs, ls, seed in (("train", cnt3[:split], lam3[:split], 0),
                               ("eval", cnt3[split:], lam3[split:], 1)):
        win_c, _ = _sample_windows(xs, L=L, n_windows=N_WINDOWS, seed=seed)
        win_l, _ = _sample_windows(ls, L=L, n_windows=N_WINDOWS, seed=seed)
        W = win_c.shape[0]
        n_tiles = L // T
        # in-tile count = sum of the indicator over the tile
        c = win_c.reshape(W * n_tiles, T).sum(1).numpy()
        t = win_l.reshape(W, n_tiles, T)[:, :, T - 1].reshape(-1).numpy()
        m = np.isfinite(t)
        rows[pool] = (c[m][:, None], t[m], int((~m).sum()), int(len(t)))

    (X_tr, y_tr, drop_tr, n_tr), (X_ev, y_ev, drop_ev, n_ev) = (
        rows["train"], rows["eval"])
    reg = LinearRegression().fit(X_tr, y_tr)
    pred = reg.predict(X_ev)
    r = float(np.corrcoef(pred, y_ev)[0, 1]) if np.std(pred) > 1e-12 else 0.0
    return {"T": T, "r": r,
            "n_train_rows": int(len(y_tr)), "n_eval_rows": int(len(y_ev)),
            "dropped_train": drop_tr, "dropped_eval": drop_ev,
            "sampled_train": n_tr, "sampled_eval": n_ev,
            "drop_frac_train": drop_tr / n_tr, "drop_frac_eval": drop_ev / n_ev}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "case"
    z = np.load(NPZ)
    count_grid = (z["op"] == CLASS_ID[target]).astype(np.float32)
    rate_grid = z[f"rate_{target}"]
    out = {"target": target, "class_id": CLASS_ID[target],
           "n_windows": N_WINDOWS, "L": L,
           "feature": f"in-tile count of op=={CLASS_ID[target]} tokens",
           "probe": "OLS (LinearRegression), v1 split + seeds",
           "per_T": [analog_for_T(count_grid, rate_grid, T) for T in TS]}
    for row in out["per_T"]:
        print(f"T={row['T']:>2}  r={row['r']:+.4f}  "
              f"rows tr/ev {row['n_train_rows']}/{row['n_eval_rows']}  "
              f"dropped {row['dropped_train']}/{row['dropped_eval']} "
              f"({row['drop_frac_train']:.1%}/{row['drop_frac_eval']:.1%})")
    dst = HERE / "results" / f"evidence_line_{target}.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
