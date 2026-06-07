"""Run the backtracking (self-exciting) benchmark grid in parallel.

Grid (docs/autoresearch/backtracking_bench_spec.md § 5):
  archs/T : (topk_sae,1) (tsae,1) (txc_base,2/4/8) (stacked_sae,2/4/8)   [8]
  d_sae   : 8, 16, 20, 40   (anchored on F=20; scarce {8,16,20} + over-complete 40)
  seeds   : 1, 2, 42
  k_pos=1, eval_window_L=32, n_steps=30000                  -> 96 trained cells
Plus the UNTRAINED-encoder control (n_steps=0) at d_sae=20 for all 8 (arch,T),
3 seeds -> 24 cells. Total 120.

Each cell goes through the canonical runner (flock-safe leaderboard append, so
parallel workers are safe). CUDA is initialised only inside workers (the parent
never imports torch), so fork-based pools are safe. Results also dumped to
docs/autoresearch/backtracking_grid_results.json as they complete.

    .venv/bin/python -m experiments.autoresearch.run_backtracking_grid [max_workers]
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

DS = "toy_backtracking_selfexcite_d64"
L = 32
K_POS = 1
N_STEPS = 30_000
D_SAES = [8, 16, 20, 40]
SEEDS = [1, 2, 42]
ARCH_T = [("topk_sae", 1), ("tsae", 1),
          ("txc_base", 2), ("txc_base", 4), ("txc_base", 8),
          ("stacked_sae", 2), ("stacked_sae", 4), ("stacked_sae", 8)]
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "autoresearch" / "backtracking_grid_results.json"


def _cells():
    cells = []
    for seed in SEEDS:
        for arch, T in ARCH_T:
            for d_sae in D_SAES:
                cells.append({"arch": arch, "T": T, "d_sae": d_sae,
                              "seed": seed, "n_steps": N_STEPS, "kind": "trained"})
            # untrained control at the F anchor (d_sae=20) only
            cells.append({"arch": arch, "T": T, "d_sae": 20,
                          "seed": seed, "n_steps": 0, "kind": "untrained"})
    return cells


def run_one(cell):
    from temp_bench.core.runner import run_experiment
    from temp_bench.core.schemas import TrainingConfig
    try:
        override = {"k_pos": K_POS, "d_sae": cell["d_sae"], "T": cell["T"]}
        tcfg = TrainingConfig(n_steps=cell["n_steps"], batch_size=1024,
                              buffer_tokens=2_000_000, arch_hparams_override=override)
        ecfg = {"smoke": False, "k_pos": K_POS, "eval_window_L": L}
        r = run_experiment(experiment="synthetic", arch_name=cell["arch"],
                           seed=cell["seed"], datasource_name=DS,
                           training_cfg=tcfg, eval_cfg=ecfg,
                           agent="autoresearch", allow_dirty=True)
        return {**cell, "metrics": {k: float(v) for k, v in r.row.metrics.items()},
                "train_cached": r.train_cached, "eval_cached": r.eval_cached, "ok": True}
    except Exception as e:  # keep the grid going; record the failure
        import traceback
        return {**cell, "ok": False, "error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-1500:]}


def main():
    max_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    cells = _cells()
    print(f"[grid] {len(cells)} cells, max_workers={max_workers}", flush=True)
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
                lr = res["metrics"].get("lambda_recovery", float("nan"))
                ea = res["metrics"].get("eauc", float("nan"))
                print(f"[{done}/{len(cells)} {el:6.0f}s] {tag:<42} "
                      f"λ={lr:.3f} eauc={ea:.3f} (cache t={res['train_cached']} e={res['eval_cached']})",
                      flush=True)
            else:
                print(f"[{done}/{len(cells)} {el:6.0f}s] {tag:<42} FAILED {res['error']}", flush=True)
            OUT.write_text(json.dumps(results, indent=2))  # incremental dump
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"[grid] DONE {n_ok}/{len(cells)} ok in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
