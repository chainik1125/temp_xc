"""diafaces/run_calib.py — ACTMIX Stage 2 frozen executor (CALIB_CARD.md).

btk-only arm ONLY (20 cells): {batchtopk_sae_btkonly, tsae_btkonly} @ T1
+ txc_batchtopk_post_btkonly @ T ∈ {4,16,32}, seeds {3,4}, trained +
untrained, k_pos = 8. The relu-mix arm is REUSED from the canonical
leaderboard by the 20 eval_keys cited in CALIB_CARD § 3 — never re-run.

Partitioning (selection-only; cannot enlarge or reorder the frozen set):
  --block tsae                     trained tsae_btkonly cells (high-CPU)
  --block main                     everything else (H100)
  --only-seed N                    restrict either block to one seed
  --only-cells arch:T:seed:kind:k  OOM re-pass selector

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.run_calib \
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
SEEDS = (3, 4)                    # CALIB_CARD § 2 — matches the reused rows
WINDOW_TS = (4, 16, 32)
BUFFER_TOKENS = 524_288           # UNCHANGED; tt stream 526,208 = complete fill
ARCHS = (
    ("batchtopk_sae_btkonly", "token"),
    ("tsae_btkonly", "token"),
    ("txc_batchtopk_post_btkonly", "post"),
)
HERE = Path(__file__).resolve().parent

# PROBE_V2_SPEC.md § 2, verbatim (paired v2 columns on every row).
V2 = {"lambda_probe_v2": True, "lambda_v2_probe": "ridge",
      "lambda_v2_alphas": list(_np.logspace(-2, 4, 13)),
      "lambda_v2_n_windows": 8192, "lambda_v2_split": "trace"}


def _calib_cells():
    cells = design.uniform_cells(
        DS, F=D_SAE, n_steps=N_STEPS, d_saes=[D_SAE], k_pos_sweep=K_POS,
        archs=ARCHS, window_ts=WINDOW_TS, L=EVAL_L, seeds=SEEDS,
        untrained_kpos=K_POS[0], log=print)
    for c in cells:
        c["buffer_tokens"] = BUFFER_TOKENS
        c["eval_extra"] = dict(V2)

    # CALIB_CARD § 4 asserts — hard-fail BEFORE any cell runs.
    assert len(cells) == 20, f"enumeration drifted: {len(cells)} != 20"
    n_tr = sum(1 for c in cells if c["kind"] == "trained")
    n_un = sum(1 for c in cells if c["kind"] == "untrained")
    assert (n_tr, n_un) == (10, 10), (n_tr, n_un)
    assert all(c["k_pos"] == K_POS[0] for c in cells), "k drifted from 8"
    assert all(c["seed"] in SEEDS for c in cells)
    assert all(c["arch"].endswith("_btkonly") for c in cells), \
        "relu-mix cells are REUSED, never re-run (CALIB_CARD § 3)"
    assert all(set(c["eval_extra"]) == set(V2)
               and c["eval_extra"]["lambda_probe_v2"] is True
               for c in cells), "v2 columns missing — the defect assert"
    return cells


def _key(c):
    return (c["arch"], c["T"], c["d_sae"], c["k_pos"], c["seed"],
            c["n_steps"], c.get("kind"))


def _cells(block, only_seed, only_cells=None):
    out = []
    for c in _calib_cells():
        is_tsae_tr = c["arch"] == "tsae_btkonly" and c.get("kind") == "trained"
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


PANEL_FILE = HERE / "results" / f"calib_stage2_{DS}.json"


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
    print(f"[merge] calib panel now {len(merged)} cells (+{added} new)",
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
    out = HERE / "results" / f"calib_{DS}{suffix}.json"
    results = grid.run_pool(_cells(block, only_seed, only_cells), out,
                            max_workers=workers, describe=_describe,
                            tag=f"diafaces-calib/{DS}{suffix}")
    _merge_into_panel(results)


if __name__ == "__main__":
    main()
