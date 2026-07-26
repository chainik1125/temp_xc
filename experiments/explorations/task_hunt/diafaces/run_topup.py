"""diafaces/run_topup.py — n=6 top-up frozen executor (TOPUP_CARD.md;
mac-local ruling ad76b0f15 item 3).

24 cells, seeds {6,7,8}, PRIMARY arm only (k_pos 8) at claiming Ts:
post T ∈ {16,32} × {tr,un} + sae/tsae T1 × {tr,un}. Same asserts as
run_salvage (count, split, per-cell paired-v2 eval_extra).

Partitioning selectors identical to run_salvage (selection-only).

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.run_topup \
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
SEEDS = (6, 7, 8)
WINDOW_TS = (16, 32)              # claiming Ts only (ruling item 3)
BUFFER_TOKENS = 524_288
ARCHS = (
    ("batchtopk_sae", "token"),
    ("tsae", "token"),
    ("txc_batchtopk_post", "post"),
)
HERE = Path(__file__).resolve().parent

# PROBE_V2_SPEC.md § 2, verbatim.
V2 = {"lambda_probe_v2": True, "lambda_v2_probe": "ridge",
      "lambda_v2_alphas": list(_np.logspace(-2, 4, 13)),
      "lambda_v2_n_windows": 8192, "lambda_v2_split": "trace"}


def _topup_cells():
    cells = design.uniform_cells(
        DS, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE], k_pos_sweep=K_POS,
        archs=ARCHS, window_ts=WINDOW_TS, L=EVAL_L, seeds=SEEDS,
        untrained_kpos=K_POS[0], log=print)
    for c in cells:
        c["buffer_tokens"] = BUFFER_TOKENS
        c["eval_extra"] = dict(V2)
    assert len(cells) == 24, f"enumeration drifted: {len(cells)} != 24"
    n_post = sum(1 for c in cells if c["arch"] == "txc_batchtopk_post")
    n_base = sum(1 for c in cells if c["arch"] in ("batchtopk_sae", "tsae"))
    assert (n_post, n_base) == (12, 12), (n_post, n_base)
    assert all(c["k_pos"] == K_POS[0] for c in cells), "primary arm only"
    assert all(c["seed"] in SEEDS for c in cells)
    assert all(set(c["eval_extra"]) == set(V2)
               and c["eval_extra"]["lambda_probe_v2"] is True
               for c in cells), "v2 columns missing — the defect assert"
    return cells


def _key(c):
    return (c["arch"], c["T"], c["d_sae"], c["k_pos"], c["seed"],
            c["n_steps"], c.get("kind"))


def _cells(block, only_seed, only_cells=None):
    out = []
    for c in _topup_cells():
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


PANEL_FILE = HERE / "results" / f"topup_stage2_{DS}.json"


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
    print(f"[merge] topup panel now {len(merged)} cells (+{added} new)",
          flush=True)


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
    out = HERE / "results" / f"topup_{DS}{suffix}.json"
    results = grid.run_pool(_cells(block, only_seed, only_cells), out,
                            max_workers=workers, describe=_describe,
                            tag=f"diafaces-topup/{DS}{suffix}")
    _merge_into_panel(results)


if __name__ == "__main__":
    main()
