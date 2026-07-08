"""Shared grid driver: enumerate cells, run them through the canonical runner.

Each bench's ``run_grid.py`` declares its cell list (arch × ``d_sae`` × ``T`` ×
seeds × datasources + the untrained / ``k_pos`` controls) and calls
:func:`run_pool`. The per-cell work goes through the ONE canonical pathway,
:func:`temp_bench.core.runner.run_experiment` — never a bespoke leaderboard
append. A cell is a plain dict so the pool can pickle it:

    {ds, arch, T, d_sae, k_pos, seed, n_steps, kind, [eval_window_L, buffer_tokens]}

``kind`` is a free-form label used only in progress output. ``run_cell`` reads
``eval_window_L`` (default 32) and ``buffer_tokens`` (default 2e6) off the cell.
"""

from __future__ import annotations

import os

# Set before any worker imports torch (dirty-tree escape hatch + thread caps so
# parallel workers don't oversubscribe cores; identify the caller in rows).
os.environ.setdefault("TEMP_BENCH_ALLOW_DIRTY", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("AGENT_NAME", "autoresearch")

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def batch_size(T: int, base: int = 1024) -> int:
    """Throughput-normalised batch: window archs (T>1) get ``base // T`` so every
    cell reconstructs ~``base`` token-positions/step (equal ``B·T`` BatchTopK pool)."""
    return base if T == 1 else base // T


def run_cell(cell: dict) -> dict:
    """Run one grid cell through the canonical runner; never raises.

    Returns ``{**cell, metrics, train_cached, eval_cached, ok}`` on success, or
    ``{**cell, ok: False, error, tb}`` on failure (so a parallel grid keeps going
    and records the failure).
    """
    from temp_bench.core.runner import run_experiment
    from temp_bench.core.schemas import TrainingConfig
    try:
        k_pos = cell["k_pos"]
        override = {"k_pos": k_pos, "d_sae": cell["d_sae"], "T": cell["T"]}
        tcfg = TrainingConfig(
            n_steps=cell["n_steps"], batch_size=batch_size(cell["T"]),
            buffer_tokens=cell.get("buffer_tokens", 2_000_000),
            arch_hparams_override=override,
        )
        ecfg = {"smoke": False, "k_pos": k_pos,
                "eval_window_L": cell.get("eval_window_L", 32)}
        r = run_experiment(
            experiment="synthetic", arch_name=cell["arch"], seed=cell["seed"],
            datasource_name=cell["ds"], training_cfg=tcfg, eval_cfg=ecfg,
            agent="autoresearch", allow_dirty=True,
        )
        return {**cell, "metrics": {k: float(v) for k, v in r.row.metrics.items()},
                "train_cached": r.train_cached, "eval_cached": r.eval_cached, "ok": True}
    except Exception as e:  # keep the grid going; record the failure
        import traceback
        return {**cell, "ok": False, "error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-1500:]}


def run_pool(cells: list[dict], out_path: Path, *, max_workers: int = 6,
             describe=None, tag: str = "grid") -> list[dict]:
    """Run ``cells`` in a process pool, dumping results to ``out_path`` as they land.

    ``describe(res) -> str`` formats the per-cell metric summary for the progress
    line (default: none). Returns the list of result dicts.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] {len(cells)} cells, max_workers={max_workers}", flush=True)
    t0 = time.time()
    results, done = [], 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_cell, c): c for c in cells}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            done += 1
            el = time.time() - t0
            label = (f"{res.get('ds','')}/{res['arch']}/T{res['T']}/d{res['d_sae']}"
                     f"/k{res.get('k_pos', 1)}/s{res['seed']}/{res.get('kind', '')}")
            if res.get("ok"):
                extra = describe(res) if describe else ""
                cache = f"(cache t={res['train_cached']} e={res['eval_cached']})"
                print(f"[{done}/{len(cells)} {el:6.0f}s] {label:<52} {extra} {cache}", flush=True)
            else:
                print(f"[{done}/{len(cells)} {el:6.0f}s] {label:<52} FAILED {res['error']}", flush=True)
            out_path.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"[{tag}] DONE {n_ok}/{len(cells)} ok in {time.time()-t0:.0f}s -> {out_path}", flush=True)
    return results
