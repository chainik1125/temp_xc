"""Evidence line for the diafaces tt panel (PANEL_CARD.md § 4) —
label-side, read-only, no activations.

Per panel T ∈ {2,4,8,16,32}: Pearson r between the tt VISIBLE floor
feature (kernel-WLS slope over previous turns COMPLETE inside the
trailing T-token window — the screen's floor, `screen._tt_visible_
feats` feature 0) and the ttrend label, over all finite-label,
non-boundary tokens at in-doc position ≥ T (window fits). This is
"what boundary-counting affords" drawn under every recovery curve;
computable before the panel lands because it never touches
activations.

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.panel_evidence_line
Writes results/panel_evidence_line.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.diafaces.screen import (
    _TurnTable,
    _tt_visible_feats,
)

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
PANEL_TS = (2, 4, 8, 16, 32)
PANELS = {"tt": ("gpt2", "ttrend"), "dq": ("llama31", "dqgap")}


def run(face: str):
    tag, vname = PANELS[face]
    zd = np.load(LABELS / f"dialevel_dailydialog_{tag}.npz")
    zf = np.load(LABELS / f"diafaces_dailydialog_{tag}.npz")
    val, boundary, off = zf[vname], zd["is_boundary"], zd["doc_off"]
    n_tok = val.shape[0]
    doc_of = np.searchsorted(off, np.arange(n_tok), side="right") - 1
    pos_of = np.arange(n_tok) - off[doc_of]
    tbl = _TurnTable(zd) if face == "tt" else None
    iq = zf["is_qtok"].astype(np.float32) if face == "dq" else None

    out = {"face": face, "model": tag,
           "population": f"finite {vname}, non-boundary, pos >= T",
           "per_T": {}}
    for T in PANEL_TS:
        m = np.isfinite(val) & (boundary == 0) & (pos_of >= T)
        idx = np.flatnonzero(m)
        y = val[idx]
        if face == "tt":
            feats = _tt_visible_feats(tbl, idx, T).float().numpy()
            score, active = feats[:, 0], feats[:, 1] >= 2
        else:
            win = idx[:, None] - np.arange(T - 1, -1, -1)[None, :]
            score = iq[win].sum(1)
            active = score > 0
        n_act = int(active.sum())
        r = float(np.corrcoef(score, y)[0, 1]) if score.std() > 0 else 0.0
        r_sub = (float(np.corrcoef(score[active], y[active])[0, 1])
                 if n_act > 2 and score[active].std() > 0 else float("nan"))
        out["per_T"][T] = {
            "n": int(len(idx)), "n_floor_active": n_act,
            "floor_active_frac": round(n_act / max(len(idx), 1), 4),
            "pearson_r": round(r, 4),
            "pearson_r_floor_active_only": round(r_sub, 4)}
        print(f"[{face}] T{T:>3}: n={len(idx):6d} active={n_act:6d} "
              f"({out['per_T'][T]['floor_active_frac']:.2%}) "
              f"r={r:+.4f} r_active={r_sub:+.4f}")
    dst = HERE / "results" / f"panel_evidence_line_{face}.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"-> {dst}")


def main():
    import sys
    for face in (sys.argv[1:] or ["tt", "dq"]):
        run(face)


if __name__ == "__main__":
    main()
