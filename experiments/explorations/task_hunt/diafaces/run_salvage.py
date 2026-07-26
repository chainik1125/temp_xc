"""diafaces/run_salvage.py — salvage W1 frozen executor (SALVAGE_CARD.md).

Fresh-seed {3,4,5} confirmation panel for the ttrend TXC-post arm:
72 cells = 30 PRIMARY post (k_pos 8, panel-identical) + 30 SECONDARY
post (k_pos 8·T, budget-parity, non-claiming) + 12 per-token baseline
cells (batchtopk_sae + tsae @ T1). Enumeration is asserted (count,
arm split, paired-v2 columns on every cell — the day-2 defect lesson
as a hard pre-run assert).

Partitioning (selection-only; cannot enlarge or reorder the frozen set):
  --block tsae                     trained tsae cells only (high-CPU)
  --block main                     everything else (H100)
  --only-seed N                    restrict either block to one seed
  --only-cells arch:T:seed:kind:k  OOM re-pass selector (k_pos included —
                                   the two post arms share (arch,T,seed,kind))

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.run_salvage \
       [workers] [--block tsae|main] [--only-seed N] [--only-cells ...]
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
    K_POS,
    N_STEPS,
    _describe,
)

DS = "dial_real_ttrend_gpt2_l7"
SEEDS = (3, 4, 5)                 # FRESH — the entire point of the card
WINDOW_TS = (2, 4, 8, 16, 32)
BUFFER_TOKENS = 524_288           # UNCHANGED; tt stream 526,208 = complete fill
ARCHS = (
    ("batchtopk_sae", "token"),
    ("tsae", "token"),
    ("txc_batchtopk_post", "post"),
)
HERE = Path(__file__).resolve().parent

# PROBE_V2_SPEC.md § 2, verbatim (paired v2 columns on every row).
V2 = {"lambda_probe_v2": True, "lambda_v2_probe": "ridge",
      "lambda_v2_alphas": list(_np.logspace(-2, 4, 13)),
      "lambda_v2_n_windows": 8192, "lambda_v2_split": "trace"}


def _salvage_cells():
    cells = design.uniform_cells(
        DS, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE], k_pos_sweep=K_POS,
        archs=ARCHS, window_ts=WINDOW_TS, L=EVAL_L, seeds=SEEDS,
        untrained_kpos=K_POS[0], log=print)
    # SECONDARY arm (SALVAGE_CARD § 2): mirror every post cell at
    # k_pos = 8·T (postmatched code-rate convention), trained AND
    # untrained. Non-claiming regardless of outcome.
    secondary = []
    for c in cells:
        if c["arch"] != "txc_batchtopk_post":
            continue
        s = dict(c)
        s["k_pos"] = K_POS[0] * s["T"]
        secondary.append(s)
    cells = cells + secondary
    for c in cells:
        c["buffer_tokens"] = BUFFER_TOKENS
        c["eval_extra"] = dict(V2)

    # Card § 3 asserts — hard-fail BEFORE any cell runs.
    assert len(cells) == 72, f"enumeration drifted: {len(cells)} != 72"
    n_prim = sum(1 for c in cells if c["arch"] == "txc_batchtopk_post"
                 and c["k_pos"] == K_POS[0])
    n_sec = sum(1 for c in cells if c["arch"] == "txc_batchtopk_post"
                and c["k_pos"] == K_POS[0] * c["T"] and c["T"] > 1)
    n_base = sum(1 for c in cells if c["arch"] in ("batchtopk_sae", "tsae"))
    assert (n_prim, n_sec, n_base) == (30, 30, 12), (n_prim, n_sec, n_base)
    assert all(set(c["eval_extra"]) == set(V2)
               and c["eval_extra"]["lambda_probe_v2"] is True
               for c in cells), "v2 columns missing — the defect assert"
    assert all(c["seed"] in SEEDS for c in cells)
    return cells


def _key(c):
    return (c["arch"], c["T"], c["d_sae"], c["k_pos"], c["seed"],
            c["n_steps"], c.get("kind"))


def _cells(block, only_seed, only_cells=None):
    out = []
    for c in _salvage_cells():
        is_tsae_tr = c["arch"] == "tsae" and c.get("kind") == "trained"
        if block == "tsae" and not is_tsae_tr:
            continue
        if block == "main" and is_tsae_tr:
            continue
        if only_seed is not None and c["seed"] != only_seed:
            continue
        if only_cells is not None and \
                (c["arch"], c["T"], c["seed"], c["kind"], c["k_pos"]) \
                not in only_cells:
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
    print(f"[merge] salvage panel now {len(merged)} cells (+{added} new)",
          flush=True)


PANEL_FILE = HERE / "results" / f"salvage_stage2_{DS}.json"


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
    only_cells = None
    if "--only-cells" in argv:
        i = argv.index("--only-cells")
        only_cells = set()
        for spec in argv[i + 1].split(","):
            arch, T, seed, kind, k = spec.split(":")
            only_cells.add((arch, int(T), int(seed), kind, int(k)))
        del argv[i:i + 2]
    workers = int(argv[0]) if argv else 3
    suffix = (f"_{block}" if block else "") + \
        (f"_s{only_seed}" if only_seed is not None else "") + \
        ("_repass" if only_cells is not None else "")
    out = HERE / "results" / f"salvage_{DS}{suffix}.json"
    results = grid.run_pool(_cells(block, only_seed, only_cells), out,
                            max_workers=workers, describe=_describe,
                            tag=f"diafaces-salvage/{DS}{suffix}")
    _merge_into_panel(results)


if __name__ == "__main__":
    main()
