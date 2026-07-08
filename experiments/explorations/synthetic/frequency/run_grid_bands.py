"""Addendum grid — the matched-budget band-granularity comparison (P3).

The main grid (run_grid.py) compares the arch FAMILY at equal k_pos, where each
arch *allocates* its per-token budget differently (post fires k_pos per window,
pre/spectral fire k_pos·T). That is the standard fairness convention, but it
means spectral (multiband, k_win=k_pos·T atoms) is NOT density-matched to the
monolithic TXC-post (k_pos atoms) — so a spectral>post gap conflates band
structure with active-atom count.

This addendum isolates the **band-partition effect at matched budget**: the same
spectral_txc arch/backbone/total budget (k_win=k_pos·T), partitioned into
1 band (full = vanilla DCT crosscoder), 2 bands (dcac = DC/AC), or 4 bands
(multiband — already in the main grid). All three fire k_win total atoms/window,
differing ONLY in how those atoms are split across DCT bands. multiband vs dcac
vs full = pure band-structure effect (P3; amendment A6).

Grid: {spectral_txc_full, spectral_txc_dcac} × d_sae {32,64,101,256} ×
T {2,4,8,16} × seeds {1,2,42} on the circle (headline), trained + untrained at
the anchor. (multiband is already in the main grid → not repeated.) Cache-safe
append to the canonical leaderboard.

    .venv/bin/python -m experiments.explorations.synthetic.frequency.run_grid_bands [max_workers]
"""

from __future__ import annotations

import os

os.environ.setdefault("TEMP_BENCH_ALLOW_DIRTY", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("AGENT_NAME", "autoresearch")

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CIRCLE = "toy_cyclic_circle_M101_d128"
L = 32
N_STEPS = 6000
D_SAES = [32, 64, 101, 256]
ANCHOR = 101
SEEDS = [1, 2, 42]
T_WINDOW = [2, 4, 8, 16]
BAND_ARCHS = ["spectral_txc_full", "spectral_txc_dcac"]

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "frequency_bands_results.json"


def _batch_size(T: int) -> int:
    return 1024 if T == 1 else 1024 // T


def _cells():
    cells = []
    for seed in SEEDS:
        for arch in BAND_ARCHS:
            for T in T_WINDOW:
                for d_sae in D_SAES:
                    cells.append({"arch": arch, "T": T, "d_sae": d_sae, "k_pos": 1,
                                  "seed": seed, "n_steps": N_STEPS, "kind": "trained"})
                cells.append({"arch": arch, "T": T, "d_sae": ANCHOR, "k_pos": 1,
                              "seed": seed, "n_steps": 0, "kind": "untrained"})
    return cells


def run_one(cell):
    from temp_bench.core.runner import run_experiment
    from temp_bench.core.schemas import TrainingConfig
    try:
        override = {"k_pos": cell["k_pos"], "d_sae": cell["d_sae"], "T": cell["T"]}
        tcfg = TrainingConfig(n_steps=cell["n_steps"], batch_size=_batch_size(cell["T"]),
                              buffer_tokens=2_000_000, arch_hparams_override=override)
        ecfg = {"smoke": False, "k_pos": cell["k_pos"], "eval_window_L": L}
        r = run_experiment(experiment="synthetic", arch_name=cell["arch"],
                           seed=cell["seed"], datasource_name=CIRCLE,
                           training_cfg=tcfg, eval_cfg=ecfg,
                           agent="autoresearch", allow_dirty=True)
        return {**cell, "metrics": {k: float(v) for k, v in r.row.metrics.items()},
                "train_cached": r.train_cached, "eval_cached": r.eval_cached, "ok": True}
    except Exception as e:
        import traceback
        return {**cell, "ok": False, "error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-1500:]}


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cells = _cells()
    print(f"[bands] {len(cells)} cells, max_workers={max_workers}", flush=True)
    t0 = time.time()
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_one, c): c for c in cells}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            done += 1
            el = time.time() - t0
            tag = f"{res['arch']}/T{res['T']}/d{res['d_sae']}/s{res['seed']}/{res['kind']}"
            if res.get("ok"):
                vr = res["metrics"].get("velocity_recovery", float("nan"))
                print(f"[{done}/{len(cells)} {el:6.0f}s] {tag:<44} vel={vr:+.3f} "
                      f"(cache t={res['train_cached']} e={res['eval_cached']})", flush=True)
            else:
                print(f"[{done}/{len(cells)} {el:6.0f}s] {tag:<44} FAILED {res['error']}", flush=True)
            OUT.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"[bands] DONE {n_ok}/{len(cells)} ok in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
