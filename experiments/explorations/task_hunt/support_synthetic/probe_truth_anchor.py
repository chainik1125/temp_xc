"""Stage 3 — the TRUTH ANCHOR for trained mirror codes (card § 3, § 4).

**Off the leaderboard** (`probe_capacity.py` precedent): nothing here is
appended to `results/leaderboard.jsonl` and no eval protocol moves. It
re-fits the λ probe on the SAME checkpoints the grid rows were scored on,
at row counts the frozen conventions do not use — which is exactly why it
must not become one.

Trained codes have no analytic truth, so the card fixes an estimator
*before* any number exists:

> refit on ``n_rows ≥ 32·p`` (floor 16384, cap 65536) drawn by the same
> committed sampler from the same split, with **both** OLS and RidgeCV;
> the anchor is their mean. Licensed only if (a) realized ``n_rows ≥ 16·p``,
> (b) ``|anchor_ols − anchor_ridge| ≤ 0.02``, (c) the v1 replication check
> reproduces the row's committed ``lambda_recovery`` to ≤ 1e-6.

(b) is the reason the anchor is credible rather than assumed: as ``n/p``
grows the two probe families **must** converge on the population value, so
a residual gap is proof that ``n/p`` is not yet large enough — and there no
truth is claimed. (c) is `probe_capacity.py`'s licence kept verbatim: this
module re-implements nothing (it imports the committed sampler, tiler and
split), so an exact v1 reproduction proves it reads the same code on the
same rows.

Also emits the **2×2** — ``{n_windows 1024, 8192} × {ols, ridge}`` — so a
gap between v1 and v2 can be attributed to the row count or to the penalty
separately. ``nw 1024 + ols`` *is* v1 (contract-tested), which is what makes
it the replication check.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.support_synthetic.probe_truth_anchor \
        [grid_stage] [workers]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TQDM_DISABLE", "1")

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DS = "toy_backtracking_selfexcite_d64"
L = 32
N_WINDOWS_2X2 = (1024, 8192)
PROBES = ("ols", "ridge")


def _anchor_n_windows(p: int, T: int) -> int:
    """Card § 3: target n_rows ≥ 32·p, floor 16384, cap 65536."""
    return int(np.ceil(min(max(32 * p, 16384), 65536) / (L // T)))


def _load_model(cell: dict):
    """Re-instantiate the checkpoint `grid.run_cell` wrote for this cell."""
    from temp_bench.core.config import (
        checkpoint_dir,
        compute_data_key,
        compute_train_key,
        import_by_path,
        load_arch,
        load_datasource,
    )
    from temp_bench.core.schemas import TrainingConfig
    from temp_bench.core.trainer import _infer_d_in
    from explorations.synthetic.grid import batch_size

    arch_spec = load_arch(cell["arch"], section="synthetic")
    data_spec = load_datasource(cell["ds"])
    override = {"k_pos": cell["k_pos"], "d_sae": cell["d_sae"], "T": cell["T"]}
    arch_spec = arch_spec.model_copy(
        update={"hparams": {**arch_spec.hparams, **override}})
    tcfg = TrainingConfig(
        n_steps=cell["n_steps"], batch_size=batch_size(cell["T"]),
        buffer_tokens=cell.get("buffer_tokens", 2_000_000),
        arch_hparams_override=override)
    train_key = compute_train_key(
        arch=arch_spec, seed=cell["seed"], training_cfg=tcfg,
        data_key=compute_data_key(data_spec), section="synthetic")
    path = checkpoint_dir(train_key) / "model.safetensors"
    if not path.exists():
        return None, train_key, data_spec
    from safetensors.torch import load_file
    cls = import_by_path(arch_spec.class_path)
    model = cls(d_in=_infer_d_in(data_spec), **arch_spec.hparams)
    model.load_state_dict(load_file(str(path)))
    model.eval()
    return model, train_key, data_spec


def _rows(model, x, lam, *, n_windows: int, T: int, seed: int = 0):
    """The committed sampler / tiler on v1's split and seeds."""
    from temp_bench.evals.lambda_recovery import _tile_lambda_examples
    from temp_bench.evals.synthetic_recovery import _sample_windows

    n = x.shape[0]
    split = n // 2
    lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)
    wx_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    wl_tr, _ = _sample_windows(lam3[:split], L=L, n_windows=n_windows, seed=seed)
    wx_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    wl_ev, _ = _sample_windows(lam3[split:], L=L, n_windows=n_windows, seed=seed + 1)
    z_tr, t_tr = _tile_lambda_examples(model, wx_tr, wl_tr, T)
    z_ev, t_ev = _tile_lambda_examples(model, wx_ev, wl_ev, T)
    return z_tr, t_tr, z_ev, t_ev


def _fit(z_tr, t_tr, z_ev, t_ev, probe: str) -> dict:
    from sklearn.linear_model import LinearRegression, RidgeCV
    from temp_bench.evals.lambda_recovery_v2 import DEFAULT_ALPHAS
    reg = (LinearRegression() if probe == "ols"
           else RidgeCV(alphas=np.asarray(DEFAULT_ALPHAS, dtype=float)))
    reg.fit(z_tr, t_tr)
    pred = reg.predict(z_ev)
    r = float(np.corrcoef(pred, t_ev)[0, 1]) if np.std(pred) > 1e-12 else 0.0
    return {"r": r, "r2": float(reg.score(z_ev, t_ev)),
            "alpha": float(getattr(reg, "alpha_", 0.0)),
            "nnz_per_row": float((z_tr != 0).sum(axis=1).mean())}


def anchor_cell(cell: dict) -> dict:
    """One grid cell → the 2×2 + the truth anchor + the replication licence."""
    try:
        from temp_bench.core.config import load_datasource
        from temp_bench.data.synthetic import materialise

        model, train_key, _ = _load_model(cell)
        base = {k: cell[k] for k in
                ("ds", "arch", "T", "d_sae", "k_pos", "seed", "n_steps", "kind")}
        if model is None:
            return {**base, "ok": False, "error": "no checkpoint",
                    "train_key": train_key}
        data = materialise(load_datasource(cell["ds"]), seed=cell["seed"])
        x = data.x
        lam = torch.as_tensor(data.extra["lambda_labels"]).float()
        T = cell["T"]
        out = {**base, "train_key": train_key, "ok": True, "grid": []}

        p = None
        for nw in N_WINDOWS_2X2:
            z_tr, t_tr, z_ev, t_ev = _rows(model, x, lam, n_windows=nw, T=T)
            p = int(z_tr.shape[1])
            row = {"n_windows": nw, "n_rows": int(z_tr.shape[0]), "p": p,
                   "p_over_n": p / max(z_tr.shape[0], 1)}
            for probe in PROBES:
                f = _fit(z_tr, t_tr, z_ev, t_ev, probe)
                row[probe] = f["r"]
                row[f"{probe}_alpha"] = f["alpha"]
                row["nnz_per_row"] = f["nnz_per_row"]
            out["grid"].append(row)
            del z_tr, z_ev
        out["p"] = p
        # Licence (c): nw 1024 + ols IS v1 — must reproduce the row's metric.
        v1_row = cell.get("metrics", {}).get("lambda_recovery")
        local_v1 = [q for q in out["grid"] if q["n_windows"] == 1024][0]["ols"]
        out["v1_row"] = None if v1_row is None else float(v1_row)
        out["v1_replication_delta"] = (
            None if v1_row is None else abs(float(v1_row) - local_v1))

        nw_a = _anchor_n_windows(p, T)
        z_tr, t_tr, z_ev, t_ev = _rows(model, x, lam, n_windows=nw_a, T=T)
        a = {"n_windows": nw_a, "n_rows": int(z_tr.shape[0]),
             "n_over_p": float(z_tr.shape[0]) / max(p, 1)}
        for probe in PROBES:
            a[probe] = _fit(z_tr, t_tr, z_ev, t_ev, probe)["r"]
        a["ols_ridge_gap"] = abs(a["ols"] - a["ridge"])
        a["anchor"] = 0.5 * (a["ols"] + a["ridge"])
        rep = out["v1_replication_delta"]
        a["licensed"] = bool(a["n_over_p"] >= 16.0
                             and a["ols_ridge_gap"] <= 0.02
                             and rep is not None and rep <= 1e-6)
        out["anchor"] = a
        return out
    except Exception as e:                       # keep the pool going
        import traceback
        return {**{k: cell.get(k) for k in
                   ("ds", "arch", "T", "d_sae", "k_pos", "seed", "n_steps", "kind")},
                "ok": False, "error": f"{type(e).__name__}: {e}",
                "tb": traceback.format_exc()[-1200:]}


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "train"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    src = RES / f"probe_truth_grid_{stage}.json"
    cells = [c for c in json.loads(src.read_text()) if c.get("ok")]
    # Cheapest (smallest p) first so a truncated run still covers the ladder.
    cells.sort(key=lambda c: (c["d_sae"] * (c["T"] if c["arch"].startswith("stacked")
                                            else 1), c["T"]))
    out_path = RES / f"probe_truth_anchor_{stage}.json"
    print(f"[anchor/{stage}] {len(cells)} cells, workers={workers}", flush=True)
    t0, res = time.time(), []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(anchor_cell, c): c for c in cells}
        for fut in as_completed(futs):
            r = fut.result()
            res.append(r)
            if r.get("ok"):
                a = r["anchor"]
                g = [q for q in r["grid"] if q["n_windows"] == 1024][0]
                print(f"[{len(res)}/{len(cells)} {time.time()-t0:6.0f}s] "
                      f"{r['arch']}/T{r['T']}/d{r['d_sae']}/k{r['k_pos']}/s{r['seed']}"
                      f"/{r['kind']:<13} p/n={g['p_over_n']:.3f} "
                      f"v1={g['ols']:+.3f} anchor={a['anchor']:+.3f} "
                      f"gap={a['ols_ridge_gap']:.3f} lic={a['licensed']} "
                      f"rep={r['v1_replication_delta']:.1e}", flush=True)
            else:
                print(f"[{len(res)}/{len(cells)} {time.time()-t0:6.0f}s] "
                      f"{r.get('arch')}/T{r.get('T')}/d{r.get('d_sae')} "
                      f"FAILED {r.get('error')}", flush=True)
            tmp = out_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, indent=1))
            os.replace(tmp, out_path)
    print(f"[anchor/{stage}] DONE {sum(1 for r in res if r.get('ok'))}/{len(cells)} "
          f"in {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
