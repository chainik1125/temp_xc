"""diafaces/run_panel.py — frozen clone of the λ̂ Stage-2 cell
enumeration with container-partitioning filters (the seedtopup
`--only-seed` device, generalized). DS frozen: dial_real_dqgap_llama31_8b_l14 (PANEL_CARD.md, amended gate).

Cells are EXACTLY `lambda_intensity/run_stage2._cells(DS)` (5 archs,
T ∈ {2,4,8,16}, seeds {1,2,42}, trained + untrained, buffer 524288 —
byte-identical panel config; the fill argument moves to the card:
dialevel stream 0.81–0.88 M tokens ≥ buffer, complete fill, no wrap).

Partitioning (cannot enlarge or reorder the frozen set):
  --block tsae            only trained tsae cells (one per container,
                          high-CPU — the scheduling lesson + Han's
                          day-2 GPU amendment)
  --block main            everything else (H100 per the amendment)
  --only-seed N           restrict either block to one seed

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.run_panel \
       [workers] [--block tsae|main] [--only-seed N]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from explorations.synthetic import design, grid
from experiments.explorations.task_hunt.lambda_intensity.run_stage2 import (
    D_SAE,
    EVAL_L,
    K_POS,
    N_STEPS,
    PANEL,
    _describe,
)

# Both frozen panels (tt: 7ba2e10fd card; dq: cfa341c34 card); selected
# by --panel. The v2-DEFECT AMENDMENT (LOG 2026-07-26 mac-a): the first
# enumeration cloned the λ̂ runner, which PREDATES PROBE_V2_SPEC and
# carries no eval_extra — every first-run row landed v1-only, breaching
# both cards' paired-columns term. Fixed here by attaching the oprate
# § 2 V2 block verbatim to every cell; v2 keys hash into eval_key, so
# re-run rows are new rows, never cache collisions.
PANEL_DS = {"tt": "dial_real_ttrend_gpt2_l7",
            "dq": "dial_real_dqgap_llama31_8b_l14"}
DS = PANEL_DS["dq"]              # default; --panel overrides in main()
# λ̂ ladder + T32 per mac-local's panel-ladder requirement (44594b696):
# a gate fired on T32 order-carriage must be panelled AT T32.
WINDOW_TS = (2, 4, 8, 16, 32)
BUFFER_TOKENS = 524_288   # UNCHANGED; disclosures per card (tt fills, dq 1.12×)
HERE = Path(__file__).resolve().parent

# PROBE_V2_SPEC.md § 2, verbatim from oprate/run_stage2.py — paired v2
# columns on every row.
import numpy as _np
V2 = {"lambda_probe_v2": True, "lambda_v2_probe": "ridge",
      "lambda_v2_alphas": list(_np.logspace(-2, 4, 13)),
      "lambda_v2_n_windows": 8192, "lambda_v2_split": "trace"}


def _lambda_cells(ds: str):
    cells = design.uniform_cells(
        ds, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE], k_pos_sweep=K_POS,
        archs=PANEL, window_ts=WINDOW_TS, L=EVAL_L, untrained_kpos=K_POS[0],
        log=print)
    for c in cells:
        c["buffer_tokens"] = BUFFER_TOKENS
        c["eval_extra"] = V2
    return cells


def _key(c):
    return (c["arch"], c["T"], c["d_sae"], c["k_pos"], c["seed"],
            c["n_steps"], c.get("kind"))


def _cells(block: str | None, only_seed: int | None,
           only_cells: set | None = None):
    out = []
    for c in _lambda_cells(DS):
        is_tsae_tr = c["arch"] == "tsae" and c.get("kind") == "trained"
        if block == "tsae" and not is_tsae_tr:
            continue
        if block == "main" and is_tsae_tr:
            continue
        if only_seed is not None and c["seed"] != only_seed:
            continue
        # --only-cells "arch:T:seed:kind,..." — OOM re-pass selector
        # (selection only, like --panel/--block: cannot enlarge or
        # reorder the frozen set).
        if only_cells is not None and \
                (c["arch"], c["T"], c["seed"], c["kind"]) not in only_cells:
            continue
        out.append(c)
    return out


def _merge_into_panel(new_results):
    existing = (json.loads(PANEL_FILE.read_text())
                if PANEL_FILE.exists() else [])
    by_key = {_key(r): r for r in existing}
    added = 0
    for r in new_results:
        if not r.get("ok"):
            continue
        if _key(r) not in by_key:
            added += 1
        by_key[_key(r)] = r
    merged = list(by_key.values())
    tmp = PANEL_FILE.with_name(PANEL_FILE.name + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2))
    tmp.replace(PANEL_FILE)
    print(f"[merge] panel now {len(merged)} cells (+{added} new)", flush=True)


def main():
    global DS, PANEL_FILE
    argv = list(sys.argv[1:])
    block = only_seed = None
    if "--panel" in argv:
        i = argv.index("--panel")
        DS = PANEL_DS[argv[i + 1]]
        del argv[i:i + 2]
    PANEL_FILE = HERE / "results" / f"stage2_{DS}.json"
    if "--block" in argv:
        i = argv.index("--block")
        block = argv[i + 1]
        assert block in ("tsae", "main")
        del argv[i:i + 2]
    if "--only-seed" in argv:
        i = argv.index("--only-seed")
        only_seed = int(argv[i + 1])
        del argv[i:i + 2]
    only_cells = None
    if "--only-cells" in argv:
        i = argv.index("--only-cells")
        only_cells = set()
        for spec in argv[i + 1].split(","):
            arch, T, seed, kind = spec.split(":")
            only_cells.add((arch, int(T), int(seed), kind))
        del argv[i:i + 2]
    workers = int(argv[0]) if argv else 3
    suffix = (f"_{block}" if block else "") + \
        (f"_s{only_seed}" if only_seed is not None else "") + \
        ("_repass" if only_cells is not None else "")
    out = HERE / "results" / f"panel_{DS}{suffix}.json"
    results = grid.run_pool(_cells(block, only_seed, only_cells), out,
                            max_workers=workers, describe=_describe,
                            tag=f"diafaces-panel/{DS}{suffix}")
    _merge_into_panel(results)


if __name__ == "__main__":
    main()
