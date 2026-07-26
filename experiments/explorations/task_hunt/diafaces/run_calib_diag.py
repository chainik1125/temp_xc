"""diafaces/run_calib_diag.py — the OPTIONAL thin-pool diagnostic cell
(mac-local ruling af2247d43 § 2; taken at mac-a's discretion, ≤ $2).

ONE cell, NON-CLAIMING: `txc_batchtopk_post_btkonly` at the 8·T
secondary configuration k_pos = 256 @ T = 32 (d_sae 2048), seed 3,
trained — the deep-selection regime where the relu-mix secondary arm
realized 178.16/256 = 0.696 (leaderboard row `d859b36e8cfcbfb1`,
freeze 50af78f12). This is the one regime on this substrate where the
compositions MUST diverge (positive pool < k during training), i.e.
the implementation's positive receipt + writeup color for the
calibration's identity result.

Pre-registered (mini-note in LOG before launch): the btk-only cell
DIVERGES from its relu-mix twin — train-time selection fills toward
nominal (eval l0 well above 178), recovery ≠ 0.2471 in some direction
(either direction fine; non-claiming), unlike the 20 calib cells whose
twins were exact. Same constants as CALIB_CARD § 2 otherwise (buffer
524288, n_steps 8000, eval_L 32, V2 eval_extra verbatim).

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.run_calib_diag [workers]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as _np

from explorations.synthetic import design, grid
from experiments.explorations.task_hunt.lambda_intensity.run_stage2 import (
    D_SAE,
    EVAL_L,
    N_STEPS,
    _describe,
)

DS = "dial_real_ttrend_gpt2_l7"
BUFFER_TOKENS = 524_288
HERE = Path(__file__).resolve().parent

V2 = {"lambda_probe_v2": True, "lambda_v2_probe": "ridge",
      "lambda_v2_alphas": list(_np.logspace(-2, 4, 13)),
      "lambda_v2_n_windows": 8192, "lambda_v2_split": "trace"}


def _diag_cells():
    cells = design.uniform_cells(
        DS, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE], k_pos_sweep=(256,),
        archs=(("txc_batchtopk_post_btkonly", "post"),), window_ts=(32,),
        L=EVAL_L, seeds=(3,), untrained=False, log=print)
    for c in cells:
        c["buffer_tokens"] = BUFFER_TOKENS
        c["eval_extra"] = dict(V2)
    assert len(cells) == 1, f"enumeration drifted: {len(cells)} != 1"
    c = cells[0]
    assert (c["arch"], c["T"], c["k_pos"], c["seed"], c["kind"]) == \
        ("txc_batchtopk_post_btkonly", 32, 256, 3, "trained"), c
    assert c["eval_extra"]["lambda_probe_v2"] is True
    return cells


PANEL_FILE = HERE / "results" / f"calib_diag_{DS}.json"


def _merge_into_panel(new_results):
    existing = (json.loads(PANEL_FILE.read_text())
                if PANEL_FILE.exists() else [])
    keyf = lambda c: (c["arch"], c["T"], c["d_sae"], c["k_pos"], c["seed"],
                      c["n_steps"], c.get("kind"))
    by_key = {keyf(r): r for r in existing}
    for r in new_results:
        if r.get("ok"):
            by_key[keyf(r)] = r
    tmp = PANEL_FILE.with_name(PANEL_FILE.name + ".tmp")
    tmp.write_text(json.dumps(list(by_key.values()), indent=2))
    tmp.replace(PANEL_FILE)
    print(f"[merge] diag panel now {len(by_key)} cells", flush=True)


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out = HERE / "results" / f"calib_diag_run_{DS}.json"
    results = grid.run_pool(_diag_cells(), out, max_workers=workers,
                            describe=_describe, tag=f"diafaces-calib-diag/{DS}")
    _merge_into_panel(results)


if __name__ == "__main__":
    main()
