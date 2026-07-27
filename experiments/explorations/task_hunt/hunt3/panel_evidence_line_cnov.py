"""Evidence line for the PROPOSED cnov panel (PANEL_CARD_DRAFT_CNOV.md § 3
S4) — label-side, read-only, no activations. mac-b per the 02:20 no-idle
allocation ("the panel's KILL clause input, your instrument").

Convention = diafaces/panel_evidence_line.py verbatim: per T, Pearson r
between the SCREEN's committed visible-floor feature (first-in-WINDOW
kernel rate, `floor_rate_T*` in labels/hunt3_dailydialog_gpt2.npz — the
in-window analog of the out-of-window cnov definition) and the cnov label,
over all finite-label, non-boundary tokens at in-doc position >= T. This is
"what in-window novelty bookkeeping affords" — drawn under every recovery
curve BEFORE any panel cell lands. T ladder = every T with a committed
floor array ({4,8,16,32,64}); the draft card's arms are {8,16,32} with
claiming {16,32}. (No T2 floor array exists and the draft has no T2 arm.)

Run: .venv/bin/python -m experiments.explorations.task_hunt.hunt3.panel_evidence_line_cnov
Writes results/panel_evidence_line_cnov.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "labels"
FLOOR_TS = (4, 8, 16, 32, 64)


def main():
    zd = np.load(LABELS / "dialevel_dailydialog_gpt2.npz")
    zh = np.load(LABELS / "hunt3_dailydialog_gpt2.npz")
    val, boundary, off = zh["cnov"], zd["is_boundary"], zd["doc_off"]
    n_tok = val.shape[0]
    assert boundary.shape[0] == n_tok, "stream/label length mismatch"
    doc_of = np.searchsorted(off, np.arange(n_tok), side="right") - 1
    pos_of = np.arange(n_tok) - off[doc_of]

    out = {"face": "cnov", "model": "gpt2",
           "floor_feature": "first-in-window kernel rate (screen floor, "
                            "labels floor_rate_T*)",
           "population": "finite cnov, non-boundary, pos >= T",
           "per_T": {}}
    for T in FLOOR_TS:
        score_all = zh[f"floor_rate_T{T}"]
        m = (np.isfinite(val) & np.isfinite(score_all) & (boundary == 0)
             & (pos_of >= T))
        idx = np.flatnonzero(m)
        y, score = val[idx], score_all[idx]
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
        print(f"[cnov] T{T:>3}: n={len(idx):6d} active={n_act:6d} "
              f"({out['per_T'][T]['floor_active_frac']:.2%}) "
              f"r={r:+.4f} r_active={r_sub:+.4f}")
    dst = HERE / "results" / "panel_evidence_line_cnov.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
