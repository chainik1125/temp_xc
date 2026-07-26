"""diafaces/run_panel.py — frozen clone of the λ̂ Stage-2 cell
enumeration with container-partitioning filters (the seedtopup
`--only-seed` device, generalized). DS frozen: dial_real_ttrend_gpt2_l7 (PANEL_CARD.md).

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

DS = "dial_real_ttrend_gpt2_l7"
# λ̂ ladder + T32 per mac-local's panel-ladder requirement (44594b696):
# a gate fired on T32 order-carriage must be panelled AT T32.
WINDOW_TS = (2, 4, 8, 16, 32)
BUFFER_TOKENS = 524_288   # UNCHANGED; dialevel stream 0.81–0.88M ≥ buffer
HERE = Path(__file__).resolve().parent
PANEL_FILE = HERE / "results" / f"stage2_{DS}.json"


def _lambda_cells(ds: str):
    cells = design.uniform_cells(
        ds, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE], k_pos_sweep=K_POS,
        archs=PANEL, window_ts=WINDOW_TS, L=EVAL_L, untrained_kpos=K_POS[0],
        log=print)
    for c in cells:
        c["buffer_tokens"] = BUFFER_TOKENS
    return cells


def _key(c):
    return (c["arch"], c["T"], c["d_sae"], c["k_pos"], c["seed"],
            c["n_steps"], c.get("kind"))


def _cells(block: str | None, only_seed: int | None):
    out = []
    for c in _lambda_cells(DS):
        is_tsae_tr = c["arch"] == "tsae" and c.get("kind") == "trained"
        if block == "tsae" and not is_tsae_tr:
            continue
        if block == "main" and is_tsae_tr:
            continue
        if only_seed is not None and c["seed"] != only_seed:
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
    argv = list(sys.argv[1:])
    block = only_seed = None
    if "--block" in argv:
        i = argv.index("--block")
        block = argv[i + 1]
        assert block in ("tsae", "main")
        del argv[i:i + 2]
    if "--only-seed" in argv:
        i = argv.index("--only-seed")
        only_seed = int(argv[i + 1])
        del argv[i:i + 2]
    workers = int(argv[0]) if argv else 3
    suffix = (f"_{block}" if block else "") + \
        (f"_s{only_seed}" if only_seed is not None else "")
    out = HERE / "results" / f"panel_{DS}{suffix}.json"
    results = grid.run_pool(_cells(block, only_seed), out,
                            max_workers=workers, describe=_describe,
                            tag=f"diafaces-panel/{DS}{suffix}")
    _merge_into_panel(results)


if __name__ == "__main__":
    main()
