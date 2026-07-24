"""Probe-truth campaign — the leaderboard grid (CARD_PROBE_TRUTH.md § 2.2–2.3).

Every cell runs through the ONE canonical pathway
(`temp_bench.core.runner.run_experiment` via `explorations.synthetic.grid`)
with the frozen v2 flags in ``eval_extra``, so each row carries BOTH readouts
on the same windows: the unchanged v1 columns (`lambda_recovery`, OLS,
n_windows = 1024) and the `*_v2` columns (RidgeCV, n_windows = 8192). The v2
flags hash into ``eval_key``, so every row written here is new by
construction — 0 duplicate keys, 0 existing rows rewritten.

Two stages:

``--stage existing``  the 22 mirror checkpoints that survive on disk (843
                      leaderboard rows, 843 train_keys, 22 checkpoints; the
                      rest pruned with no HF restore path). Training is a
                      cache hit; only the eval runs. Card § 2.2.

``--stage train``     the ladder, card § 2.3 — lines C (capacity, the core),
                      P (matched post at k = 8·T), M (mirror-canonical
                      control), S (Stacked p > n), plus untrained controls at
                      every line point. Submitted in the card's priority order
                      C ≻ P ≻ M ≻ S, T = 16 first inside each line, untrained
                      last, so a short night still ships the informative end.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.support_synthetic.run_probe_truth \
        --stage train --workers 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from explorations.synthetic import grid

DS = "toy_backtracking_selfexcite_d64"
N_STEPS = 30_000
L = 32
SEEDS = (1, 2, 42)
HERE = Path(__file__).resolve().parent

# The frozen v2 convention (PROBE_V2_SPEC.md § 1). Every knob is explicit so
# its value is pinned into eval_key rather than inherited from a code default.
V2 = {
    "lambda_probe_v2": True,
    "lambda_v2_probe": "ridge",
    "lambda_v2_alphas": [float(a) for a in np.logspace(-2, 4, 13)],
    "lambda_v2_n_windows": 8192,
    "lambda_v2_split": "trace",          # synthetic: degenerates to v1's n//2
}

# line -> (arch, k_pos spec, d_sae list, T list). k_pos "8T" = 8·T (the
# code-rate convention for post, card_stage2_postmatched.md § 2).
LINES = (
    ("C", "txc_batchtopk_pre", 8, [256, 1024, 2048], [16, 8, 4, 2]),
    ("P", "txc_batchtopk_post", "8T", [2048], [16, 8, 4, 2]),
    ("M", "txc_batchtopk_pre", 1, [20], [16, 8, 4, 2]),
    ("S", "stacked_batchtopk", 8, [512], [16, 4]),
)


def _k(spec, T: int) -> int:
    return 8 * T if spec == "8T" else int(spec)


def _cell(arch, T, d, k, seed, n_steps, kind):
    return {"ds": DS, "arch": arch, "T": T, "d_sae": d, "k_pos": k,
            "seed": seed, "n_steps": n_steps, "kind": kind,
            "eval_window_L": L, "eval_extra": dict(V2)}


def train_cells() -> list[dict]:
    """The card § 2.3 ladder, in priority order (trained first, untrained last)."""
    trained, untrained = [], []
    for line, arch, kspec, d_saes, ts in LINES:
        for T in ts:
            k = _k(kspec, T)
            for d in d_saes:
                if d < (k * T if arch != "txc_batchtopk_post" else k):
                    print(f"[skip] {line} {arch}/T{T}/d{d}/k{k}: dict-infeasible")
                    continue
                for seed in SEEDS:
                    trained.append(_cell(arch, T, d, k, seed, N_STEPS, f"{line}-trained"))
                    untrained.append(_cell(arch, T, d, k, seed, 0, f"{line}-untrained"))
    return trained + untrained


def existing_cells() -> list[dict]:
    """Every mirror leaderboard row whose checkpoint still exists on disk."""
    from temp_bench.core.config import checkpoint_dir
    seen, cells = set(), []
    lb = Path("results/leaderboard.jsonl")
    for line in lb.read_text().splitlines():
        r = json.loads(line)
        if r.get("datasource") != DS:
            continue
        if not (checkpoint_dir(r["train_key"]) / "model.safetensors").exists():
            continue
        o = r["training_cfg"]["arch_hparams_override"]
        key = (r["arch"], o["T"], o["d_sae"], o["k_pos"], r["seed"],
               r["training_cfg"]["n_steps"])
        if key in seen:
            continue
        seen.add(key)
        cells.append(_cell(r["arch"], o["T"], o["d_sae"], o["k_pos"], r["seed"],
                           r["training_cfg"]["n_steps"], "existing"))
    return cells


def _describe(res):
    m = res["metrics"]
    def g(k):
        return m.get(k, float("nan"))
    return (f"v1={g('lambda_recovery'):+.3f} v2={g('lambda_recovery_v2'):+.3f} "
            f"a={g('lambda_alpha_v2'):.3g} l0w={g('l0_per_window'):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "existing"], required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    cells = train_cells() if a.stage == "train" else existing_cells()
    out = HERE / "results" / f"probe_truth_grid_{a.stage}.json"
    grid.run_pool(cells, out, max_workers=a.workers, describe=_describe,
                  tag=f"probe-truth/{a.stage}")


if __name__ == "__main__":
    main()
